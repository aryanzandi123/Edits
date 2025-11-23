#!/usr/bin/env python3
"""
Database Sync Layer

Extracts pipeline JSON output and syncs to PostgreSQL.

Design Philosophy:
- Non-invasive: Pipeline code unchanged
- Idempotent: Safe to run multiple times
- Transactional: All-or-nothing updates
- Backward Compatible: Maintains file cache

Strategy:
- Store FULL interactor JSON in interactions.data (JSONB)
- Preserves all fields: evidence[], functions[], pmids[], etc.
- No data loss from pipeline output
"""

from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import sys

# Fix Windows console encoding for Greek letters and special characters
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from models import Protein, Interaction, db


def deduplicate_functions(functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicates functions by name (case-insensitive), preferring more complete entries.

    Args:
        functions: List of function dicts

    Returns:
        Deduplicated list of functions
    """
    if not functions:
        return []

    # Group by function name (case-insensitive)
    seen = {}

    for func in functions:
        func_name = (func.get("function", "") or "").strip().lower()
        if not func_name:
            continue

        if func_name in seen:
            # Prefer entries with more fields populated
            existing = seen[func_name]
            existing_fields = sum(1 for v in existing.values() if v not in [None, "", []])
            current_fields = sum(1 for v in func.values() if v not in [None, "", []])

            # Prefer validated entries (with direct_arrow) or more complete entries
            is_validated = func.get("_arrow_validated") or func.get("direct_arrow")
            existing_validated = existing.get("_arrow_validated") or existing.get("direct_arrow")

            if is_validated and not existing_validated:
                seen[func_name] = func  # Prefer validated
            elif current_fields > existing_fields:
                seen[func_name] = func  # Prefer more complete
            # else: keep existing
        else:
            seen[func_name] = func

    return list(seen.values())


class DatabaseSyncLayer:
    """Syncs pipeline output to PostgreSQL."""

    @staticmethod
    def _validate_and_fix_chain(interactor_data: Dict[str, Any], protein_symbol: str) -> Dict[str, Any]:
        """
        Validate and fix false chain assignments before database write.

        Detects and corrects false chains created by old schema_validator Strategy 3
        that blindly assigned first direct interactor as upstream for all proteins.

        Args:
            interactor_data: Interactor dict with potential false chain
            protein_symbol: Query protein symbol for logging

        Returns:
            Fixed interactor_data (modified in-place, but returned for chaining)
        """
        # Detect false chain from old Strategy 3
        if interactor_data.get('_chain_inferred_strategy') == 'first_direct_interactor':
            partner_symbol = interactor_data.get('primary', 'UNKNOWN')
            false_upstream = interactor_data.get('upstream_interactor')

            print(f"[DB_SYNC] ⚠️  Clearing false chain: {protein_symbol} → {partner_symbol} (false upstream: {false_upstream})")

            # Clear false chain data
            interactor_data['upstream_interactor'] = None
            interactor_data['mediator_chain'] = []

            # Add correction markers
            interactor_data['_chain_inference_corrected'] = True
            interactor_data['_correction_timestamp'] = datetime.now().isoformat()
            interactor_data['_false_upstream_removed'] = false_upstream

            # Add missing chain markers for transparency
            interactor_data['_chain_missing'] = True
            interactor_data['_inference_failed'] = 'no_biological_hints'

            # Remove the problematic strategy marker
            del interactor_data['_chain_inferred_strategy']

        return interactor_data

    def sync_query_results(
        self,
        protein_symbol: str,
        snapshot_json: Dict[str, Any],
        ctx_json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """
        Sync pipeline results to database.

        Args:
            protein_symbol: Query protein (e.g., "ATXN3")
            snapshot_json: Output from pipeline (format: {snapshot_json: {...}})
            ctx_json: Rich metadata (optional, from metadata file)

        Returns:
            Stats: {
                "proteins_created": int,
                "interactions_created": int,
                "interactions_updated": int
            }

        Raises:
            ValueError: If snapshot_json format is invalid
            Exception: If database transaction fails (rolled back automatically)
        """
        # Validate input
        if not protein_symbol:
            raise ValueError("protein_symbol cannot be empty")

        if not isinstance(snapshot_json, dict):
            raise ValueError("snapshot_json must be a dict")

        # Handle both formats: {"snapshot_json": {...}} and direct {...}
        snapshot_data = snapshot_json.get("snapshot_json", snapshot_json)

        if not isinstance(snapshot_data, dict):
            raise ValueError("snapshot_data must be a dict")

        stats = {
            "proteins_created": 0,
            "interactions_created": 0,
            "interactions_updated": 0
        }

        try:
            # Transaction wrapper (all-or-nothing)
            with db.session.begin_nested():
                # Step 1: Get or create main protein
                main_protein = self._get_or_create_protein(protein_symbol)
                if main_protein.query_count == 0:
                    stats["proteins_created"] += 1

                # Step 2: Extract interactors from snapshot
                interactors = snapshot_data.get("interactors", [])

                if not isinstance(interactors, list):
                    print(f"[WARN]WARNING: interactors is not a list, got {type(interactors)}", file=sys.stderr)
                    interactors = []

                # Step 3: Process each interactor
                for interactor_data in interactors:
                    if not isinstance(interactor_data, dict):
                        print(f"[WARN]WARNING: Skipping invalid interactor data: {type(interactor_data)}", file=sys.stderr)
                        continue

                    partner_symbol = interactor_data.get("primary")
                    if not partner_symbol:
                        print(f"[WARN]WARNING: Skipping interactor with no 'primary' field", file=sys.stderr)
                        continue

                    # Validate and fix false chains BEFORE database write
                    interactor_data = self._validate_and_fix_chain(interactor_data, protein_symbol)

                    # Get or create partner protein
                    partner_protein = self._get_or_create_protein(partner_symbol)
                    if partner_protein.query_count == 0:
                        stats["proteins_created"] += 1

                    # Save interaction (stores ENTIRE interactor_data in JSONB)
                    created = self._save_interaction(
                        protein_a=main_protein,
                        protein_b=partner_protein,
                        data=interactor_data,
                        discovered_in=protein_symbol
                    )

                    if created:
                        stats["interactions_created"] += 1
                    else:
                        stats["interactions_updated"] += 1

                    # Update partner protein's total_interactions count (bidirectional)
                    partner_protein.total_interactions = db.session.query(Interaction).filter(
                        (Interaction.protein_a_id == partner_protein.id) |
                        (Interaction.protein_b_id == partner_protein.id)
                    ).count()

                    # Process chain relationships for indirect interactions
                    if interactor_data.get("interaction_type") == "indirect":
                        chain_stats = self.sync_chain_relationships(
                            query_protein=protein_symbol,
                            interactor_data=interactor_data
                        )
                        stats["interactions_created"] += chain_stats["chain_links_created"]
                        stats["interactions_updated"] += chain_stats["chain_links_updated"]

                # Step 4: Update main protein metadata
                main_protein.last_queried = datetime.utcnow()
                main_protein.query_count += 1

                # CRITICAL: Count ALL interactions (bidirectional due to canonical ordering)
                # This includes reverse links where protein was discovered as someone else's interactor
                main_protein.total_interactions = db.session.query(Interaction).filter(
                    (Interaction.protein_a_id == main_protein.id) |
                    (Interaction.protein_b_id == main_protein.id)
                ).count()

            # Commit transaction
            db.session.commit()

        except Exception as e:
            # Rollback on any error
            db.session.rollback()
            print(f"[ERROR]Database sync failed: {e}", file=sys.stderr)
            raise

        return stats

    def _get_or_create_protein(self, symbol: str) -> Protein:
        """
        Get existing protein or create new one.

        Args:
            symbol: Protein symbol (e.g., "ATXN3")

        Returns:
            Protein instance

        Side effects:
            - Creates new protein in database if not exists
            - Flushes to get ID (does not commit)
        """
        if not symbol:
            raise ValueError("symbol cannot be empty")

        # Query existing
        protein = Protein.query.filter_by(symbol=symbol).first()

        if not protein:
            # Create new
            protein = Protein(
                symbol=symbol,
                first_queried=datetime.utcnow(),
                last_queried=datetime.utcnow(),
                query_count=0,
                total_interactions=0,
                extra_data={}
            )
            db.session.add(protein)
            db.session.flush()  # Get ID without committing

        return protein

    def _lookup_arrow_for_pair(self, from_protein_symbol: str, to_protein_symbol: str) -> str:
        """
        Look up the arrow type for an interaction between two proteins.

        Used to build chain_with_arrows for indirect interactions.

        Args:
            from_protein_symbol: Source protein symbol
            to_protein_symbol: Target protein symbol

        Returns:
            Arrow type ('activates', 'inhibits', 'binds', 'complex')
            Returns 'binds' as default if interaction not found
        """
        # Get protein objects
        from_protein = Protein.query.filter_by(symbol=from_protein_symbol).first()
        to_protein = Protein.query.filter_by(symbol=to_protein_symbol).first()

        if not from_protein or not to_protein:
            return 'binds'  # Default if proteins don't exist

        # Query interaction with canonical ordering
        protein_a_id = min(from_protein.id, to_protein.id)
        protein_b_id = max(from_protein.id, to_protein.id)

        interaction = Interaction.query.filter_by(
            protein_a_id=protein_a_id,
            protein_b_id=protein_b_id
        ).first()

        if not interaction:
            return 'binds'  # Default if interaction doesn't exist

        # Get arrow type
        # Priority: arrows JSONB field > arrow field (backward compat)
        if interaction.arrows:
            # Determine direction based on canonical ordering
            if from_protein.id < to_protein.id:
                # Natural order: from=a, to=b
                # Check a_to_b direction
                if 'a_to_b' in interaction.arrows and interaction.arrows['a_to_b']:
                    return interaction.arrows['a_to_b'][0]  # Primary arrow
                elif 'bidirectional' in interaction.arrows and interaction.arrows['bidirectional']:
                    return interaction.arrows['bidirectional'][0]
            else:
                # Reversed order: from=b, to=a
                # Check b_to_a direction
                if 'b_to_a' in interaction.arrows and interaction.arrows['b_to_a']:
                    return interaction.arrows['b_to_a'][0]  # Primary arrow
                elif 'bidirectional' in interaction.arrows and interaction.arrows['bidirectional']:
                    return interaction.arrows['bidirectional'][0]

        # Fallback to legacy arrow field
        return interaction.arrow or 'binds'

    def _save_interaction(
        self,
        protein_a: Protein,
        protein_b: Protein,
        data: Dict[str, Any],
        discovered_in: str
    ) -> bool:
        """
        Save or update interaction with CANONICAL ORDERING.

        Strategy:
        - Enforces protein_a_id < protein_b_id to prevent duplicates
        - Stores FULL interactor data in JSONB
        - Preserves original direction in data JSONB
        - Transforms direction when storing in reversed order

        Args:
            protein_a: Main protein (queried protein)
            protein_b: Partner protein (interactor)
            data: Complete interactor dict from pipeline
            discovered_in: Which protein query found this interaction

        Returns:
            True if created new interaction, False if updated existing

        Side effects:
            - Creates or updates interaction in database
            - Flushes to database (does not commit)
        """
        if not protein_a or not protein_b:
            raise ValueError("protein_a and protein_b cannot be None")

        if protein_a.id == protein_b.id:
            raise ValueError("Cannot create self-interaction")

        if not isinstance(data, dict):
            raise ValueError("data must be a dict")

        # CANONICAL ORDERING: Always store with lower ID as protein_a
        # This prevents (A,B) and (B,A) from both existing in database
        original_direction = data.get("direction")

        # NEW: Convert query-relative direction to protein-absolute direction
        # protein_a (arg) = query protein (main)
        # protein_b (arg) = partner protein (primary)
        # We store directions as absolute: "a_to_b", "b_to_a", "bidirectional"
        if protein_a.id < protein_b.id:
            # Natural canonical order: query protein has lower ID
            canonical_a = protein_a
            canonical_b = protein_b
            # Convert query-relative → absolute:
            # main_to_primary (query → partner) = a_to_b (protein_a → protein_b)
            # primary_to_main (partner → query) = b_to_a (protein_b → protein_a)
            if original_direction == "main_to_primary":
                stored_direction = "a_to_b"
            elif original_direction == "primary_to_main":
                stored_direction = "b_to_a"
            else:
                # bidirectional stays bidirectional, handle None/other
                stored_direction = original_direction or "bidirectional"
        else:
            # Reversed canonical order: partner protein becomes protein_a in storage
            canonical_a = protein_b  # partner (was arg protein_b, now stored protein_a)
            canonical_b = protein_a  # query (was arg protein_a, now stored protein_b)
            # After swap: canonical_a (partner) < canonical_b (query)
            # Convert query-relative → absolute:
            # main_to_primary (query → partner) = b_to_a (after swap: protein_b → protein_a)
            # primary_to_main (partner → query) = a_to_b (after swap: protein_a → protein_b)
            if original_direction == "main_to_primary":
                stored_direction = "b_to_a"
            elif original_direction == "primary_to_main":
                stored_direction = "a_to_b"
            else:
                # bidirectional stays bidirectional
                stored_direction = original_direction or "bidirectional"

        # Store original direction in data for retrieval
        data_copy = data.copy()
        data_copy["_original_direction"] = original_direction
        data_copy["_query_context"] = discovered_in

        # Check if interaction exists (using canonical IDs)
        interaction = Interaction.query.filter_by(
            protein_a_id=canonical_a.id,
            protein_b_id=canonical_b.id
        ).first()

        # Extract denormalized fields (for fast queries)
        confidence = data.get("confidence")
        arrow = data.get("arrow")
        interaction_type = data.get("interaction_type", "direct")  # Default to 'direct' for backward compatibility
        upstream_interactor = data.get("upstream_interactor")  # Can be None for direct interactions

        # NOTE: upstream_interactor CAN be one of the proteins in the interaction pair
        # This is VALID when querying the regulating protein
        # Example: Query VCP → finds IκBα with upstream=VCP
        # Meaning: VCP regulates IκBα (VCP→IκBα directed edge)
        # This is NOT a self-interaction (VCP→VCP would be invalid)
        # The self-link will be filtered out in sync_chain_relationships (line 446)

        # Extract chain metadata for indirect interactions
        mediator_chain = data.get("mediator_chain")  # e.g., ["VCP", "LAMP2"] for multi-hop paths
        depth = data.get("depth", 1)  # Default to 1 (direct) if not specified
        chain_context = data.get("chain_context")  # Full chain context from all perspectives

        # NEW (Issue #4): Extract arrows field (multiple arrow types per direction)
        arrows_raw = data.get("arrows", {})
        if not arrows_raw and arrow:
            # Backward compatibility: convert single arrow to arrows dict
            arrows_raw = {"main_to_primary": [arrow]}

        # CRITICAL: Flip arrow directions for canonical ordering (same logic as direction field)
        if protein_a.id < protein_b.id:
            # Natural canonical order: no flip needed
            arrows = arrows_raw
        else:
            # Reversed canonical order: flip arrow directions
            # main_to_primary becomes b_to_a (after swap)
            # primary_to_main becomes a_to_b (after swap)
            arrows = {}
            if "main_to_primary" in arrows_raw:
                arrows["b_to_a"] = arrows_raw["main_to_primary"]
            if "primary_to_main" in arrows_raw:
                arrows["a_to_b"] = arrows_raw["primary_to_main"]
            if "bidirectional" in arrows_raw:
                arrows["bidirectional"] = arrows_raw["bidirectional"]

        # Extract function_context from tagged functions
        functions = data.get("functions", [])
        function_contexts = set()
        for fn in functions:
            if isinstance(fn, dict) and "_context" in fn:
                fn_context_type = fn["_context"].get("type", "direct")
                function_contexts.add(fn_context_type)

        # Determine overall function_context for this interaction
        if not function_contexts:
            function_context = "direct"  # Default
        elif len(function_contexts) == 1:
            function_context = list(function_contexts)[0]  # Single type
        else:
            function_context = "mixed"  # Multiple types

        if interaction:
            # UPDATE existing interaction (merge data intelligently)
            # CRITICAL FIX: Merge functions and evidence instead of choosing one or the other
            existing_evidence = interaction.data.get("evidence", [])
            new_evidence = data_copy.get("evidence", [])
            existing_functions = interaction.data.get("functions", [])
            new_functions = data_copy.get("functions", [])

            # Merge evidence arrays (deduplicate by PMID)
            merged_evidence = existing_evidence.copy()
            existing_pmids = {ev.get("pmid") for ev in existing_evidence if ev.get("pmid")}
            for ev in new_evidence:
                if ev.get("pmid") not in existing_pmids:
                    merged_evidence.append(ev)

            # Merge function arrays (deduplicate by function name)
            merged_functions = existing_functions.copy()
            existing_fn_names = {fn.get("function") for fn in existing_functions if fn.get("function")}
            for fn in new_functions:
                fn_name = fn.get("function")
                if fn_name and fn_name not in existing_fn_names:
                    merged_functions.append(fn)

            # Deduplicate merged functions (case-insensitive, prefer validated)
            merged_functions = deduplicate_functions(merged_functions)

            # Determine which data is richer (use as base)
            if len(new_evidence) > len(existing_evidence):
                # New data is richer, use it as base and merge in old functions
                interaction.data = data_copy
                interaction.data["functions"] = merged_functions
                interaction.data["evidence"] = merged_evidence
                interaction.direction = stored_direction
            else:
                # Existing data is richer or equal, merge in new functions/evidence
                interaction.data["functions"] = merged_functions
                interaction.data["evidence"] = merged_evidence
                interaction.data["_last_seen_in"] = discovered_in
                interaction.data["_last_updated"] = datetime.utcnow().isoformat()
                # Update direction if new query provides better direction info
                if stored_direction and stored_direction != "bidirectional":
                    interaction.direction = stored_direction

            interaction.confidence = confidence
            interaction.arrow = arrow
            interaction.arrows = arrows  # NEW (Issue #4): Multiple arrow types

            # CRITICAL: Never downgrade direct→indirect
            # This prevents chain processing from corrupting direct interactions
            if interaction.interaction_type == "direct" and interaction_type == "indirect":
                # Existing direct interaction takes precedence
                # Don't overwrite with chain-derived metadata from mediator role
                print(f"[DB SYNC] Preserving direct interaction: {canonical_a.symbol}↔{canonical_b.symbol} (refusing downgrade to indirect)", file=sys.stderr)
                # Keep existing direct metadata
            else:
                # Safe to update (creating new, upgrading indirect, or updating direct with direct)
                interaction.interaction_type = interaction_type
                interaction.upstream_interactor = upstream_interactor
                interaction.mediator_chain = mediator_chain
                interaction.depth = depth
                interaction.chain_context = chain_context
                interaction.chain_with_arrows = data.get("chain_with_arrows")  # NEW (Issue #2)
                interaction.function_context = function_context

            interaction.updated_at = datetime.utcnow()
            created = False
        else:
            # CREATE new interaction
            interaction = Interaction(
                protein_a_id=canonical_a.id,
                protein_b_id=canonical_b.id,
                data=data_copy,
                confidence=confidence,
                direction=stored_direction,
                arrow=arrow,
                arrows=arrows,  # NEW (Issue #4): Multiple arrow types
                interaction_type=interaction_type,
                upstream_interactor=upstream_interactor,
                mediator_chain=mediator_chain,
                depth=depth,
                chain_context=chain_context,
                chain_with_arrows=data.get("chain_with_arrows"),  # NEW (Issue #2): Typed arrows for chain segments
                function_context=function_context,
                discovered_in_query=discovered_in,
                discovery_method='pipeline'
            )
            db.session.add(interaction)
            created = True

        db.session.flush()  # Persist to DB (does not commit)
        return created

    def sync_chain_relationships(
        self,
        query_protein: str,
        interactor_data: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Store chain relationships from ALL protein perspectives.

        For a chain like ATXN3 → VCP → LAMP2:
        - ATXN3-LAMP2: indirect (depth=2, mediator_chain=["VCP"])
        - VCP-LAMP2: direct (depth=1, mediator_chain=[])

        This ensures bidirectional queries work:
        - Query ATXN3 → sees LAMP2 as indirect via VCP
        - Query VCP → sees LAMP2 as direct
        - Query LAMP2 → sees VCP as direct, ATXN3 as indirect

        Args:
            query_protein: Main query protein symbol
            interactor_data: Interactor dict with chain metadata

        Returns:
            Stats: {
                "chain_links_created": int,
                "chain_links_updated": int
            }
        """
        stats = {
            "chain_links_created": 0,
            "chain_links_updated": 0
        }

        # Extract chain information
        target_protein = interactor_data.get("primary")
        interaction_type = interactor_data.get("interaction_type", "direct")
        upstream_interactor = interactor_data.get("upstream_interactor")
        mediator_chain = interactor_data.get("mediator_chain", [])

        if not target_protein:
            return stats

        # Case 1: Direct interaction (no chain)
        if interaction_type == "direct" and not mediator_chain:
            # Already handled by sync_query_results
            return stats

        # Case 2: Indirect interaction with chain
        # Example: ATXN3 → VCP → LAMP2
        # mediator_chain = ["VCP"]
        # We need to store:
        # 1. ATXN3-LAMP2 (indirect, depth=2)
        # 2. VCP-LAMP2 (direct, depth=1)

        if not mediator_chain or not isinstance(mediator_chain, list):
            # If no proper chain but marked as indirect, use upstream_interactor
            if interaction_type == "indirect" and upstream_interactor:
                mediator_chain = [upstream_interactor]
            else:
                return stats

        # Build full chain: [query, mediator1, mediator2, ..., target]
        full_chain = [query_protein] + mediator_chain + [target_protein]

        # Process each link in the chain
        for i in range(len(full_chain) - 1):
            source_symbol = full_chain[i]
            target_symbol = full_chain[i + 1]

            # Get or create proteins
            source_protein = self._get_or_create_protein(source_symbol)
            target_protein_obj = self._get_or_create_protein(target_symbol)

            # CRITICAL FIX: Skip self-links (source == target)
            # This happens when upstream_interactor == query_protein
            # Example: VCP query with IκBα having upstream=VCP
            # Creates chain ["VCP", "VCP", "IκBα"] which has VCP→VCP self-link
            # This is VALID metadata (VCP regulates IκBα) but should not create VCP→VCP interaction
            if source_symbol == target_symbol:
                print(f"[DB SYNC] Skipping self-link in chain: {source_symbol}→{source_symbol} (valid upstream=query case)", file=sys.stderr)
                continue  # Skip to next link in chain

            # CRITICAL: Skip first link (query→mediator) if it's already a direct interaction
            # This prevents corrupting direct interactions with chain-derived indirect metadata
            if i == 0 and len(full_chain) > 2:
                # Check if this link already exists as direct interaction
                canonical_a_id = min(source_protein.id, target_protein_obj.id)
                canonical_b_id = max(source_protein.id, target_protein_obj.id)

                existing = Interaction.query.filter_by(
                    protein_a_id=canonical_a_id,
                    protein_b_id=canonical_b_id
                ).first()

                if existing and existing.interaction_type == "direct":
                    # Already saved correctly as direct - skip chain re-save
                    print(f"[DB SYNC] Skipping chain re-save of direct interaction: {source_symbol}↔{target_symbol}", file=sys.stderr)
                    continue  # Skip to next link in chain

            # Calculate depth: direct links (i → i+1) have depth=1
            link_depth = 1

            # Build chain context for this specific link
            link_data = interactor_data.copy()
            link_data["primary"] = target_symbol

            # Set correct interaction_type for this link
            if i == 0 and len(full_chain) > 2:
                # First link of multi-hop chain: could be direct or indirect
                # Keep original type
                pass
            elif i > 0:
                # Intermediate and final links in chain: always direct
                link_data["interaction_type"] = "direct"
                link_data["upstream_interactor"] = None
                link_depth = 1

            # Store mediator chain for indirect relationships
            if i == 0 and len(full_chain) > 2:
                # Query → Target (indirect): store full mediator chain
                link_data["mediator_chain"] = mediator_chain
                link_data["depth"] = len(mediator_chain) + 1
            else:
                # Direct link: no mediators
                link_data["mediator_chain"] = []
                link_data["depth"] = 1

            # Store chain context (for reconstruction)
            link_data["chain_context"] = {
                "full_chain": full_chain,
                "link_position": i,
                "query_protein": query_protein
            }

            # NEW (Issue #2): Build chain_with_arrows - includes arrow types for each segment
            chain_with_arrows = []
            for j in range(len(full_chain) - 1):
                from_protein = full_chain[j]
                to_protein = full_chain[j + 1]

                # Look up arrow type for this pair from database
                arrow_type = self._lookup_arrow_for_pair(from_protein, to_protein)

                chain_with_arrows.append({
                    "from": from_protein,
                    "to": to_protein,
                    "arrow": arrow_type
                })

            link_data["chain_with_arrows"] = chain_with_arrows

            # Save interaction
            created = self._save_interaction(
                protein_a=source_protein,
                protein_b=target_protein_obj,
                data=link_data,
                discovered_in=query_protein
            )

            if created:
                stats["chain_links_created"] += 1
            else:
                stats["chain_links_updated"] += 1

        return stats
