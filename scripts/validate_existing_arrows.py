#!/usr/bin/env python3
"""
Retroactive Arrow Validation Script
====================================

Validates and corrects arrows, directions, and effects for all existing
interactions in the PostgreSQL database.

NOTE: As of the latest update, arrow validation and direct link extraction
      are now AUTOMATIC in the query pipeline (runner.py). This script is for
      retroactive validation of existing database records only.

      Automatic (new queries):  runner.py → utils/arrow_validator_integrated.py
      Manual (existing data):   This script

Features:
- Processes all interactions from the database
- Uses Gemini 2.5 Pro with thinking mode for validation
- Processes 3-4 interactions in parallel
- Logs all corrections for review
- Supports dry-run mode (no database writes)
- Progress tracking and error handling
- Can filter by specific proteins or process all
- Phase 2: Extracts direct mediator links from indirect chains (3-tier strategy)

Usage:
    # Dry run (no database writes, just log corrections)
    python scripts/validate_existing_arrows.py --dry-run

    # Apply corrections to database
    python scripts/validate_existing_arrows.py

    # Validate specific protein only
    python scripts/validate_existing_arrows.py --protein ATXN3

    # Validate with verbose logging
    python scripts/validate_existing_arrows.py --verbose

    # Process only first N interactions (for testing)
    python scripts/validate_existing_arrows.py --limit 10 --dry-run
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from models import db, Protein, Interaction
from app import app  # Import Flask app for database context

# Import arrow validator
try:
    from utils.arrow_effect_validator import validate_single_interaction
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False
    print("[ERROR] Arrow validator not available. Install dependencies first.")
    sys.exit(1)


# Constants
MAX_WORKERS = 12
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def deduplicate_functions(functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicates functions by name, preferring validated/more complete entries.

    CRITICAL: Validated entries (_arrow_validated=True) ALWAYS trump non-validated entries.
    This ensures arrow validation results replace all previous data.

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
            existing = seen[func_name]

            # Check validation flags (arrow_validated is set by arrow_effect_validator.py)
            # We look at the PARENT interaction's _arrow_validated flag, not individual functions
            # But we can also check for arrow_context which is only added by validator
            is_validated = func.get("arrow_context") is not None or func.get("direct_arrow") is not None
            existing_validated = existing.get("arrow_context") is not None or existing.get("direct_arrow") is not None

            # RULE 1: Validated entries ALWAYS replace non-validated (trumping rule)
            if is_validated and not existing_validated:
                seen[func_name] = func  # REPLACE with validated entry
            elif existing_validated and not is_validated:
                # Keep existing validated entry, discard non-validated
                pass
            else:
                # Both validated or both non-validated: prefer more complete
                existing_fields = sum(1 for v in existing.values() if v not in [None, "", []])
                current_fields = sum(1 for v in func.values() if v not in [None, "", []])

                if current_fields > existing_fields:
                    seen[func_name] = func  # Prefer more complete
                # else: keep existing
        else:
            seen[func_name] = func

    return list(seen.values())


def check_existing_direct_interaction(
    mediator: str,
    target: str
) -> Optional[Interaction]:
    """
    Query database for existing direct interaction between two proteins.

    Args:
        mediator: First protein symbol
        target: Second protein symbol

    Returns:
        Interaction ORM object if found with interaction_type='direct', else None
    """
    mediator_protein = Protein.query.filter_by(symbol=mediator).first()
    target_protein = Protein.query.filter_by(symbol=target).first()

    if not mediator_protein or not target_protein:
        return None

    # Check canonical ordering (protein_a_id < protein_b_id)
    if mediator_protein.id < target_protein.id:
        existing = db.session.query(Interaction).filter(
            Interaction.protein_a_id == mediator_protein.id,
            Interaction.protein_b_id == target_protein.id
        ).first()
    else:
        existing = db.session.query(Interaction).filter(
            Interaction.protein_a_id == target_protein.id,
            Interaction.protein_b_id == mediator_protein.id
        ).first()

    # Only return if it's a direct interaction (not indirect/shared)
    if existing:
        interaction_type = existing.data.get('interaction_type', 'direct')
        if interaction_type == 'direct':
            return existing

    return None


def extract_direct_link_evidence(
    indirect_data: Dict[str, Any],
    mediator: str,
    target: str
) -> Optional[Dict[str, Any]]:
    """
    Extracts evidence for direct mediator→target link from indirect interaction data.

    Args:
        indirect_data: Indirect interaction data (full chain)
        mediator: Mediator protein symbol (e.g., "VCP")
        target: Target protein symbol (e.g., "UFD1")

    Returns:
        Dict with evidence subset relevant to mediator→target, or None if not found
    """
    # Search evidence for papers mentioning both mediator and target
    evidence = indirect_data.get("evidence", [])
    relevant_evidence = []

    for ev in evidence:
        # Check if this paper discusses the mediator→target relationship
        title = (ev.get("paper_title", "") or "").lower()
        quote = (ev.get("relevant_quote", "") or "").lower()

        mediator_lower = mediator.lower()
        target_lower = target.lower()

        # Simple heuristic: both proteins mentioned in title or quote
        if (mediator_lower in title and target_lower in title) or \
           (mediator_lower in quote and target_lower in quote):
            relevant_evidence.append(ev)

    if relevant_evidence:
        return {
            "evidence": relevant_evidence,
            "has_evidence": True
        }

    return None


def validate_pair_specific_evidence(
    evidence: List[Dict[str, Any]],
    protein_a: str,
    protein_b: str
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Filter evidence to only pair-specific mentions (both proteins mentioned).

    Args:
        evidence: List of evidence dicts
        protein_a: First protein symbol
        protein_b: Second protein symbol

    Returns:
        (has_pair_evidence, filtered_evidence)
    """
    if not evidence:
        return (False, [])

    protein_a_lower = protein_a.lower()
    protein_b_lower = protein_b.lower()

    pair_specific = []

    for ev in evidence:
        title = (ev.get('paper_title') or '').lower()
        quote = (ev.get('relevant_quote') or '').lower()

        # Both proteins must be mentioned in title OR quote
        a_mentioned = protein_a_lower in title or protein_a_lower in quote
        b_mentioned = protein_b_lower in title or protein_b_lower in quote

        if a_mentioned and b_mentioned:
            pair_specific.append(ev)

    return (len(pair_specific) > 0, pair_specific)


def query_direct_interaction_pair(
    protein_a: str,
    protein_b: str,
    api_key: str,
    verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Minimal pipeline query for ONLY the direct A↔B interaction.

    Uses max_depth=1 (no chains), minimal rounds for efficiency.

    Args:
        protein_a: Query protein symbol
        protein_b: Target protein symbol (to check if found)
        api_key: Google API key for LLM
        verbose: Enable verbose logging

    Returns:
        Interactor dict for protein_b if found, None otherwise
    """
    try:
        # Import pipeline runner
        from runner import run_pipeline
        from pipeline.config_dynamic import create_dynamic_config

        if verbose:
            print(f"    [TIER 2] Querying pipeline: {protein_a} → {protein_b} (direct only)")

        # Create minimal config (1 round, depth 1, no chains)
        config = create_dynamic_config(
            num_interactor_rounds=1,  # Minimal: just immediate interactors
            num_function_rounds=1,    # Minimal: basic function mapping
            protein_name=protein_a
        )

        # Run pipeline with minimal depth
        result = run_pipeline(
            user_query=protein_a,
            num_interactor_rounds=1,
            num_function_rounds=1,
            max_depth=1,  # NO CHAINS: direct interactions only
            verbose=False  # Suppress pipeline logs
        )

        if not result or 'snapshot_json' not in result:
            if verbose:
                print(f"    [TIER 2] No results from pipeline")
            return None

        # Check if protein_b is in interactors
        interactors = result['snapshot_json'].get('interactors', [])
        for interactor in interactors:
            if interactor.get('primary', '').upper() == protein_b.upper():
                # Found direct interaction!
                if verbose:
                    num_funcs = len(interactor.get('functions', []))
                    print(f"    [TIER 2] Found direct interaction ({num_funcs} functions)")
                return interactor

        if verbose:
            print(f"    [TIER 2] {protein_b} not found in {protein_a} interactors")
        return None

    except Exception as e:
        if verbose:
            print(f"    [TIER 2] Pipeline query failed: {str(e)}")
        return None


def build_direct_mediator_link(
    mediator: str,
    target: str,
    indirect_data: Dict[str, Any],
    main_protein: str
) -> Optional[Dict[str, Any]]:
    """
    Builds a direct mediator→target link from indirect chain data.

    Args:
        mediator: Mediator protein symbol
        target: Target protein symbol
        indirect_data: Full indirect interaction data
        main_protein: Original query protein (for tracking)

    Returns:
        Dict ready for validation, or None if insufficient evidence
    """
    # Extract evidence for direct link
    evidence_result = extract_direct_link_evidence(indirect_data, mediator, target)

    if not evidence_result or not evidence_result.get("has_evidence"):
        # No evidence for direct link
        return None

    # Build interactor dict for the direct link
    direct_link = {
        "primary": target,
        "direction": "bidirectional",  # Default; will be corrected by validator
        "arrow": "binds",  # Default; will be corrected by validator
        "interaction_type": "direct",  # This is a DIRECT link
        "functions": [],  # Will be populated by validator
        "evidence": evidence_result["evidence"],
        "pmids": [ev.get("pmid") for ev in evidence_result["evidence"] if ev.get("pmid")],
        "_inferred_from_chain": True,  # Flag to indicate this was extracted
        "_original_chain": f"{main_protein}→{mediator}→{target}",
        "upstream_interactor": None,  # No upstream for direct links
        "mediator_chain": None,
        "depth": 1
    }

    return direct_link


def process_indirect_interaction(
    interaction_data: Dict[str, Any],
    db_interactions_map: Dict[int, Any],
    verbose: bool = False,
    api_key: str = None
) -> Optional[Dict[str, Any]]:
    """
    Processes an indirect interaction to extract direct mediator link using 3-tier strategy.

    For indirect chain like ATXN3→VCP→UFD1:
    - Tier 1: Check if VCP→UFD1 already exists in database as direct
    - Tier 2: Query pipeline for VCP to find UFD1 direct interaction
    - Tier 3: Extract from chain evidence (fallback)

    Args:
        interaction_data: Indirect interaction data dict
        db_interactions_map: Map of interaction_id → ORM object
        verbose: Enable detailed logging
        api_key: Google API key for Tier 2 pipeline queries

    Returns:
        Dict with direct link data for validation, or None
    """
    data = interaction_data["data"]
    interaction_type = data.get("interaction_type", "direct")

    # Only process indirect interactions
    if interaction_type != "indirect":
        return None

    # Extract chain information
    upstream_interactor = interaction_data.get("upstream_interactor")
    mediator_chain = interaction_data.get("mediator_chain", [])
    main_protein = interaction_data["main_protein"]
    target_protein = interaction_data["partner_protein"]

    # Determine mediator (last protein before target)
    mediator = upstream_interactor or (mediator_chain[-1] if mediator_chain else None)

    if not mediator:
        if verbose:
            print(f"    [TIER 3] No mediator found for {main_protein}→{target_protein}")
        return None

    # ========================================
    # TIER 1: Check existing database for direct interaction
    # ========================================
    existing = check_existing_direct_interaction(mediator, target_protein)
    if existing:
        if verbose:
            print(f"    [TIER 1] Found existing direct interaction in database (ID: {existing.id})")

        # Use existing data
        return {
            "id": f"existing_{existing.id}",
            "data": existing.data,
            "main_protein": mediator,
            "partner_protein": target_protein,
            "upstream_interactor": None,
            "mediator_chain": [],
            "depth": 1,
            "interaction_type": "direct",
            "_is_inferred": False,
            "_evidence_tier": 1,
            "_existing_db_id": existing.id
        }

    # ========================================
    # TIER 2: Query pipeline for direct pair
    # ========================================
    if api_key:
        pipeline_result = query_direct_interaction_pair(
            mediator,
            target_protein,
            api_key,
            verbose=verbose
        )

        if pipeline_result:
            # Validate evidence is pair-specific
            evidence = pipeline_result.get('evidence', [])
            has_evidence, filtered_evidence = validate_pair_specific_evidence(
                evidence,
                mediator,
                target_protein
            )

            if has_evidence:
                if verbose:
                    print(f"    [TIER 2] Pipeline found direct interaction ({len(filtered_evidence)} evidence papers)")

                # Use pipeline result
                pipeline_result['evidence'] = filtered_evidence
                pipeline_result['function_context'] = 'direct'
                pipeline_result['_inferred_from_chain'] = True

                return {
                    "id": f"pipeline_{mediator}_{target_protein}",
                    "data": pipeline_result,
                    "main_protein": mediator,
                    "partner_protein": target_protein,
                    "upstream_interactor": None,
                    "mediator_chain": [],
                    "depth": 1,
                    "interaction_type": "direct",
                    "_is_inferred": True,
                    "_evidence_tier": 2
                }

    # ========================================
    # TIER 3: Extract from chain evidence (fallback)
    # ========================================
    if verbose:
        print(f"    [TIER 3] Extracting from chain evidence (fallback)")

    direct_link = build_direct_mediator_link(
        mediator,
        target_protein,
        data,
        main_protein
    )

    if not direct_link:
        if verbose:
            print(f"    [TIER 3] No evidence found for {mediator}→{target_protein} direct link")
        return None

    # Return as pseudo-interaction for validation
    return {
        "id": f"inferred_{mediator}_{target_protein}",
        "data": direct_link,
        "main_protein": mediator,
        "partner_protein": target_protein,
        "upstream_interactor": None,
        "mediator_chain": [],
        "depth": 1,
        "interaction_type": "direct",
        "_is_inferred": True,
        "_evidence_tier": 3
    }


def validate_interaction_record(
    interaction_data: Dict[str, Any],
    api_key: str,
    verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Validates a single database interaction record.

    Args:
        interaction_data: Dict with interaction data (not ORM object)
        api_key: Google AI API key
        verbose: Enable detailed logging

    Returns:
        Dict with corrections or None if no changes needed
    """
    try:
        # Extract pre-fetched data
        interaction_id = interaction_data["id"]
        data = interaction_data["data"]
        main_protein = interaction_data["main_protein"]
        partner_protein = interaction_data["partner_protein"]

        # Build interactor object for validation
        interactor = {
            "primary": partner_protein,
            "direction": data.get("direction", "unknown"),
            "arrow": data.get("arrow", "unknown"),
            "interaction_type": data.get("interaction_type", "direct"),
            "functions": data.get("functions", []),
            "evidence": data.get("evidence", []),
            "_original_direction": data.get("_original_direction"),
            # Chain fields for indirect interactions (from table columns)
            "upstream_interactor": interaction_data.get("upstream_interactor"),
            "mediator_chain": interaction_data.get("mediator_chain", []),
            "depth": interaction_data.get("depth", 1),
        }

        # Validate
        corrected = validate_single_interaction(
            interactor,
            main_protein,
            api_key,
            verbose=verbose
        )

        # Check if corrections were applied
        validation_metadata = corrected.get("_validation_metadata", {})
        corrections_count = validation_metadata.get("corrections_applied", 0)

        if corrections_count > 0:
            return {
                "interaction_id": interaction_id,
                "main_protein": main_protein,
                "partner_protein": partner_protein,
                "original_data": interactor,
                "corrected_data": corrected,
                "corrections_count": corrections_count
            }

        return None

    except Exception as e:
        print(f"[ERROR] Failed to validate interaction {interaction_id}: {e}")
        return None


def invalidate_cache_files(main_protein: str, partner_protein: str, verbose: bool = False) -> bool:
    """
    Invalidates (deletes) cache files for proteins after database update.

    This ensures the visualization will fetch fresh data from the database
    instead of reading stale cached files.

    Args:
        main_protein: Main protein symbol (e.g., "ATXN3")
        partner_protein: Partner protein symbol (e.g., "VCP")
        verbose: Enable detailed logging

    Returns:
        True if successful
    """
    try:
        cache_dir = Path("cache")
        pruned_dir = cache_dir / "pruned"

        deleted_count = 0

        # Delete main protein cache
        main_cache = cache_dir / f"{main_protein}.json"
        if main_cache.exists():
            main_cache.unlink()
            deleted_count += 1
            if verbose:
                print(f"    [CACHE] Deleted {main_cache}")

        # Delete partner protein cache (in case it was queried)
        partner_cache = cache_dir / f"{partner_protein}.json"
        if partner_cache.exists():
            partner_cache.unlink()
            deleted_count += 1
            if verbose:
                print(f"    [CACHE] Deleted {partner_cache}")

        # Delete pruned caches involving these proteins
        if pruned_dir.exists():
            # Pattern: <PARENT>_for_<PROTEIN>.json
            for pruned_file in pruned_dir.glob("*.json"):
                filename = pruned_file.stem
                # Check if either protein is mentioned in filename
                if main_protein in filename or partner_protein in filename:
                    pruned_file.unlink()
                    deleted_count += 1
                    if verbose:
                        print(f"    [CACHE] Deleted {pruned_file}")

        if deleted_count > 0:
            print(f"    [CACHE] Invalidated {deleted_count} cache file(s)")

        return True

    except Exception as e:
        print(f"[WARNING] Failed to invalidate cache files: {e}")
        return False


def apply_corrections_to_db(
    interaction: Interaction,
    corrected_data: Dict[str, Any],
    dry_run: bool = True
) -> bool:
    """
    Applies corrections to database record.

    Args:
        interaction: Interaction ORM object
        corrected_data: Corrected interactor data
        dry_run: If True, don't actually write to database

    Returns:
        True if successful
    """
    try:
        # Merge corrections into existing data
        data = interaction.data.copy()

        # Update interaction-level fields
        data["direction"] = corrected_data.get("direction", data.get("direction"))
        data["arrow"] = corrected_data.get("arrow", data.get("arrow"))
        data["interaction_type"] = corrected_data.get("interaction_type", data.get("interaction_type"))

        # Update functions and deduplicate
        corrected_functions = corrected_data.get("functions", [])
        if corrected_functions:
            data["functions"] = corrected_functions

        # Deduplicate functions (prefer validated entries)
        data["functions"] = deduplicate_functions(data.get("functions", []))

        # Add function_context differentiation for dual-track system
        # Indirect interactions get "net" context (full chain effects)
        # Direct interactions get "direct" context (pair-specific effects)
        interaction_type = data.get("interaction_type", "direct")
        if interaction_type == "indirect":
            data["function_context"] = "net"  # NET effects through full chain
        elif not data.get("function_context"):
            # Default to "direct" for direct interactions if not already set
            data["function_context"] = "direct"

        # Update validation metadata
        data["_validation_metadata"] = corrected_data.get("_validation_metadata", {})
        data["_validation_metadata"]["validated_at"] = datetime.utcnow().isoformat()

        if not dry_run:
            # Write to database
            interaction.data = data
            interaction.updated_at = datetime.utcnow()

            # Also update denormalized fields for consistency
            interaction.direction = data.get("direction")
            interaction.arrow = data.get("arrow")

            db.session.commit()

        return True

    except Exception as e:
        print(f"[ERROR] Failed to apply corrections to interaction {interaction.id}: {e}")
        if not dry_run:
            db.session.rollback()
        return False


def log_corrections(corrections: List[Dict[str, Any]], log_file: Path):
    """
    Logs all corrections to a JSON file for review.

    Args:
        corrections: List of correction dicts
        log_file: Path to log file
    """
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(corrections, f, indent=2, ensure_ascii=False)
        print(f"[LOG] Corrections saved to: {log_file}")
    except Exception as e:
        print(f"[ERROR] Failed to write log file: {e}")


def diagnose_missing_arrows(interaction_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Diagnoses functions with missing or empty arrow fields.
    Now also tracks indirect interactions and chain data.

    Args:
        interaction_data_list: List of interaction data dicts

    Returns:
        Diagnostic report dict with indirect interaction tracking
    """
    total_functions = 0
    missing_arrows = 0
    interactions_with_issues = []

    # Track indirect interactions
    indirect_count = 0
    indirect_missing_chain = 0

    for interaction_data in interaction_data_list:
        data = interaction_data["data"]
        functions = data.get("functions", [])
        total_functions += len(functions)

        # Check if indirect interaction
        interaction_type = interaction_data.get("interaction_type", "direct")
        upstream = interaction_data.get("upstream_interactor")
        mediator_chain = interaction_data.get("mediator_chain", [])

        if interaction_type == "indirect":
            indirect_count += 1
            if not upstream and not mediator_chain:
                indirect_missing_chain += 1

        issue_functions = []
        for func in functions:
            if not func.get("arrow") or func.get("arrow") == "":
                missing_arrows += 1
                issue_functions.append({
                    "function": func.get("function", "Unknown"),
                    "current_arrow": func.get("arrow", ""),
                    "interaction_arrow": data.get("arrow", "unknown")
                })

        if issue_functions:
            interactions_with_issues.append({
                "main_protein": interaction_data["main_protein"],
                "partner_protein": interaction_data["partner_protein"],
                "interaction_arrow": data.get("arrow", "unknown"),
                "interaction_type": interaction_type,  # ADD
                "mediator_chain": mediator_chain,      # ADD
                "upstream_interactor": upstream,        # ADD
                "functions_with_missing_arrows": issue_functions
            })

    return {
        "total_interactions": len(interaction_data_list),
        "total_functions": total_functions,
        "missing_arrows": missing_arrows,
        "indirect_interactions": indirect_count,              # NEW
        "indirect_missing_chain_data": indirect_missing_chain, # NEW
        "interactions_with_issues": interactions_with_issues
    }


def main():
    parser = argparse.ArgumentParser(description="Validate arrows for existing database interactions")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database, just log corrections")
    parser.add_argument("--diagnose", action="store_true", help="Only report missing function arrows (no validation/fixes)")
    parser.add_argument("--fix-function-arrows", action="store_true", help="Focus on fixing missing function arrows")
    parser.add_argument("--protein", type=str, help="Only validate interactions for specific protein")
    parser.add_argument("--limit", type=int, help="Limit number of interactions to process (for testing)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Load environment
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY not set in environment")
        sys.exit(1)

    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"arrow_validation_{timestamp}.json"

    mode = "DRY RUN" if args.dry_run else "LIVE"
    print(f"\n{'='*60}")
    print(f"RETROACTIVE ARROW VALIDATION ({mode})")
    print(f"{'='*60}")

    # Query interactions
    with app.app_context():
        query = Interaction.query

        # Filter by protein if specified
        if args.protein:
            protein = Protein.query.filter_by(symbol=args.protein).first()
            if not protein:
                print(f"[ERROR] Protein '{args.protein}' not found in database")
                sys.exit(1)
            query = query.filter(
                (Interaction.protein_a_id == protein.id) |
                (Interaction.protein_b_id == protein.id)
            )
            print(f"[INFO] Filtering to protein: {args.protein}")

        # Apply limit if specified
        if args.limit:
            query = query.limit(args.limit)
            print(f"[INFO] Limiting to {args.limit} interactions")

        interactions = query.all()

        if not interactions:
            print("[INFO] No interactions found to validate")
            sys.exit(0)

        print(f"[INFO] Found {len(interactions)} interactions to validate")
        print(f"[INFO] Extracting data from database...\n")

        # Extract all data from database BEFORE parallel processing
        # This avoids application context issues in worker threads
        interaction_data_list = []
        interaction_map = {}  # Map interaction_id -> ORM object for later updates

        for interaction in interactions:
            # Get protein symbols
            protein_a = db.session.get(Protein, interaction.protein_a_id)
            protein_b = db.session.get(Protein, interaction.protein_b_id)

            if not protein_a or not protein_b:
                print(f"[WARNING] Skipping interaction {interaction.id} (missing protein records)")
                continue

            # Determine main protein (use discovered_in_query or default to protein_a)
            main_protein = interaction.discovered_in_query or protein_a.symbol
            partner_protein = protein_b.symbol if main_protein == protein_a.symbol else protein_a.symbol

            # Create plain dict for worker (no ORM objects)
            interaction_data = {
                "id": interaction.id,
                "data": interaction.data,
                "main_protein": main_protein,
                "partner_protein": partner_protein,
                # Chain fields from table columns (for indirect interactions)
                "upstream_interactor": interaction.upstream_interactor,
                "mediator_chain": interaction.mediator_chain or [],
                "depth": interaction.depth or 1,
                "interaction_type": interaction.interaction_type or "direct"
            }

            interaction_data_list.append(interaction_data)
            interaction_map[interaction.id] = interaction

        print(f"[INFO] Processing {len(interaction_data_list)} interactions with {MAX_WORKERS} workers\n")

        # DIAGNOSTIC MODE: Report missing arrows and exit
        if args.diagnose:
            print(f"{'='*60}")
            print(f"DIAGNOSTIC MODE: Analyzing function arrows...")
            print(f"{'='*60}\n")

            report = diagnose_missing_arrows(interaction_data_list)

            print(f"Total Interactions: {report['total_interactions']}")
            print(f"Total Functions: {report['total_functions']}")
            print(f"Missing Function Arrows: {report['missing_arrows']}")
            print(f"Indirect Interactions: {report['indirect_interactions']}")  # NEW
            print(f"Indirect Missing Chain Data: {report['indirect_missing_chain_data']}")  # NEW
            print(f"Interactions with Issues: {len(report['interactions_with_issues'])}\n")

            if report['interactions_with_issues']:
                print(f"{'='*60}")
                print("FUNCTIONS WITH MISSING ARROWS:")
                print(f"{'='*60}\n")

                for issue in report['interactions_with_issues'][:10]:  # Show first 10
                    main = issue['main_protein']
                    partner = issue['partner_protein']
                    interaction_type = issue.get('interaction_type', 'direct')

                    # Show chain context for indirect interactions
                    if interaction_type == "indirect":
                        mediator_chain = issue.get('mediator_chain', [])
                        if mediator_chain:
                            chain_str = " → ".join([main] + mediator_chain + [partner])
                            print(f"{chain_str} (indirect)")
                        else:
                            upstream = issue.get('upstream_interactor', '?')
                            print(f"{main} → {partner} (via {upstream})")
                    else:
                        print(f"{main} ↔ {partner} (interaction: {issue['interaction_arrow']})")

                    for func in issue['functions_with_missing_arrows']:
                        print(f"  • {func['function']}")
                        print(f"    Current arrow: '{func['current_arrow']}' (empty/missing)")
                        print(f"    Would fallback to: '{func['interaction_arrow']}'\n")

                if len(report['interactions_with_issues']) > 10:
                    print(f"... and {len(report['interactions_with_issues']) - 10} more interactions with issues\n")

            # Save diagnostic report
            diag_file = LOG_DIR / f"diagnostic_{timestamp}.json"
            with open(diag_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            print(f"{'='*60}")
            print(f"Diagnostic report saved to: {diag_file}")
            print(f"{'='*60}\n")

            return  # Exit without validation

        # Validate interactions in parallel
        all_corrections = []
        validated_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks with plain dicts (not ORM objects)
            future_to_id = {
                executor.submit(
                    validate_interaction_record,
                    interaction_data,
                    api_key,
                    args.verbose
                ): interaction_data["id"]
                for interaction_data in interaction_data_list
            }

            # Collect results as they complete
            for future in as_completed(future_to_id):
                interaction_id = future_to_id[future]
                validated_count += 1

                try:
                    correction = future.result()

                    if correction:
                        # Corrections were made
                        main = correction["main_protein"]
                        partner = correction["partner_protein"]
                        count = correction["corrections_count"]

                        # Add chain context to correction record
                        original_data = correction["original_data"]
                        interaction_type = original_data.get("interaction_type", "direct")
                        mediator_chain = original_data.get("mediator_chain", [])
                        upstream = original_data.get("upstream_interactor")

                        correction["chain_context"] = {
                            "interaction_type": interaction_type,
                            "mediator_chain": mediator_chain,
                            "upstream_interactor": upstream,
                            "depth": original_data.get("depth", 1)
                        }

                        all_corrections.append(correction)

                        # Display with chain context for indirect interactions
                        if interaction_type == "indirect":
                            if mediator_chain:
                                chain_str = " → ".join([main] + mediator_chain + [partner])
                                print(f"  [{validated_count}/{len(interaction_data_list)}] ✓ {chain_str}: {count} correction(s)")
                            elif upstream:
                                print(f"  [{validated_count}/{len(interaction_data_list)}] ✓ {main} → {partner} (via {upstream}): {count} correction(s)")
                            else:
                                print(f"  [{validated_count}/{len(interaction_data_list)}] ✓ {main} → {partner} (indirect): {count} correction(s)")
                        else:
                            print(f"  [{validated_count}/{len(interaction_data_list)}] ✓ {main} ↔ {partner}: {count} correction(s)")

                        # Apply to database if not dry-run
                        if not args.dry_run:
                            # Get ORM object from map
                            interaction = interaction_map.get(interaction_id)
                            if interaction:
                                success = apply_corrections_to_db(
                                    interaction,
                                    correction["corrected_data"],
                                    dry_run=False
                                )
                                if success:
                                    # Invalidate cache files to force fresh database read
                                    invalidate_cache_files(main, partner, verbose=args.verbose)
                                else:
                                    error_count += 1
                    else:
                        # No corrections needed
                        if args.verbose:
                            print(f"  [{validated_count}/{len(interaction_data_list)}] → No corrections needed")

                except Exception as exc:
                    error_count += 1
                    print(f"  [{validated_count}/{len(interaction_data_list)}] ✗ Error: {exc}")

        # ========================================
        # PHASE 2: Extract and validate direct mediator links from indirect interactions
        # ========================================
        print(f"\n{'='*60}")
        print(f"PHASE 2: EXTRACTING DIRECT MEDIATOR LINKS")
        print(f"{'='*60}\n")

        direct_links_extracted = []
        direct_links_validated = 0
        direct_links_created = 0
        tier1_count = 0  # Database evidence
        tier2_count = 0  # Pipeline queries
        tier3_count = 0  # Chain extraction

        for interaction_data in interaction_data_list:
            # Only process indirect interactions
            if interaction_data.get("interaction_type") != "indirect":
                continue

            # Extract direct mediator link (with 3-tier strategy)
            direct_link_data = process_indirect_interaction(
                interaction_data,
                interaction_map,
                verbose=args.verbose,
                api_key=api_key  # Pass API key for Tier 2 pipeline queries
            )

            if direct_link_data:
                direct_links_extracted.append(direct_link_data)

                # Track tier usage
                tier = direct_link_data.get("_evidence_tier", 3)
                if tier == 1:
                    tier1_count += 1
                elif tier == 2:
                    tier2_count += 1
                else:
                    tier3_count += 1

        print(f"[INFO] Found {len(direct_links_extracted)} indirect interactions with extractable direct links")
        print(f"[INFO] Evidence sources: Tier 1 (Database)={tier1_count}, Tier 2 (Pipeline)={tier2_count}, Tier 3 (Extraction)={tier3_count}\n")

        if direct_links_extracted:
            # Validate direct links sequentially (not parallel)
            for idx, direct_link_data in enumerate(direct_links_extracted):
                mediator = direct_link_data["main_protein"]
                target = direct_link_data["partner_protein"]
                tier = direct_link_data.get("_evidence_tier", 3)

                # Tier indicators for logging
                tier_labels = {1: "TIER 1:DB", 2: "TIER 2:PIPELINE", 3: "TIER 3:EXTRACT"}
                tier_label = tier_labels.get(tier, "UNKNOWN")

                try:
                    print(f"  [{idx+1}/{len(direct_links_extracted)}] [{tier_label}] Validating: {mediator} → {target}")

                    # TIER 1: Skip validation (already validated, from existing DB)
                    if tier == 1:
                        existing_id = direct_link_data.get("_existing_db_id")
                        print(f"    → Using existing database record (ID: {existing_id}), no validation needed")
                        direct_links_validated += 1
                        continue  # Skip to next

                    # TIER 2 & 3: Validate the direct link
                    correction = validate_interaction_record(
                        direct_link_data,
                        api_key,
                        verbose=args.verbose
                    )

                    direct_links_validated += 1

                    # Check if this link already exists in database
                    # Query for existing interaction between mediator and target
                    mediator_protein = Protein.query.filter_by(symbol=mediator).first()
                    target_protein = Protein.query.filter_by(symbol=target).first()

                    if mediator_protein and target_protein:
                        # Check for existing interaction (canonical ordering)
                        if mediator_protein.id < target_protein.id:
                            existing = db.session.query(Interaction).filter(
                                Interaction.protein_a_id == mediator_protein.id,
                                Interaction.protein_b_id == target_protein.id
                            ).first()
                        else:
                            existing = db.session.query(Interaction).filter(
                                Interaction.protein_a_id == target_protein.id,
                                Interaction.protein_b_id == mediator_protein.id
                            ).first()

                        if existing:
                            print(f"    → Direct link already exists in database (ID: {existing.id})")
                            # Apply corrections if validator made any
                            if correction and not args.dry_run:
                                success = apply_corrections_to_db(
                                    existing,
                                    correction["corrected_data"],
                                    dry_run=False
                                )
                                if success:
                                    print(f"    → Applied corrections to existing direct link")
                        else:
                            # Create new direct link in database
                            print(f"    → Creating new direct link in database")
                            if not args.dry_run:
                                # Use corrected data if available, otherwise use original
                                link_data = correction["corrected_data"] if correction else direct_link_data["data"]

                                # Mark as direct interaction with direct function context
                                link_data["function_context"] = "direct"
                                link_data["interaction_type"] = "direct"

                                # CRITICAL FIX: Extract direct_arrow from arrow_context for direct mediator links
                                # The link_data contains net arrow (e.g., "inhibits" for ATXN3→RHEB→MTOR)
                                # but we need the direct arrow (e.g., "activates" for RHEB→MTOR)
                                direct_arrow = link_data.get("arrow", "binds")  # Default fallback
                                for func in link_data.get("functions", []):
                                    if func.get("arrow_context") and func["arrow_context"].get("direct_arrow"):
                                        direct_arrow = func["arrow_context"]["direct_arrow"]
                                        print(f"    → Extracted direct arrow: {direct_arrow} (from arrow_context)")
                                        break

                                # Create new interaction record
                                # Determine canonical ordering
                                if mediator_protein.id < target_protein.id:
                                    new_interaction = Interaction(
                                        protein_a_id=mediator_protein.id,
                                        protein_b_id=target_protein.id,
                                        confidence=link_data.get("confidence", 0.5),
                                        direction=link_data.get("direction", "bidirectional"),
                                        arrow=direct_arrow,  # USE EXTRACTED DIRECT ARROW
                                        data=link_data,
                                        discovered_in_query=direct_link_data["data"].get("_original_chain", "").split("→")[0],
                                        discovery_method="indirect_chain_extraction",
                                        interaction_type="direct",
                                        created_at=datetime.utcnow(),
                                        updated_at=datetime.utcnow()
                                    )
                                else:
                                    # Flip direction for canonical ordering
                                    flipped_direction = link_data.get("direction", "bidirectional")
                                    if flipped_direction == "main_to_primary":
                                        flipped_direction = "primary_to_main"
                                    elif flipped_direction == "primary_to_main":
                                        flipped_direction = "main_to_primary"

                                    new_interaction = Interaction(
                                        protein_a_id=target_protein.id,
                                        protein_b_id=mediator_protein.id,
                                        confidence=link_data.get("confidence", 0.5),
                                        direction=flipped_direction,
                                        arrow=direct_arrow,  # USE EXTRACTED DIRECT ARROW
                                        data=link_data,
                                        discovered_in_query=direct_link_data["data"].get("_original_chain", "").split("→")[0],
                                        discovery_method="indirect_chain_extraction",
                                        interaction_type="direct",
                                        created_at=datetime.utcnow(),
                                        updated_at=datetime.utcnow()
                                    )

                                db.session.add(new_interaction)
                                db.session.commit()
                                direct_links_created += 1
                                print(f"    → Created direct link successfully (ID: {new_interaction.id})")

                except Exception as exc:
                    error_count += 1
                    print(f"  [{idx+1}/{len(direct_links_extracted)}] ✗ Error validating {mediator}→{target}: {exc}")
                    if not args.dry_run:
                        db.session.rollback()

        print(f"\n[PHASE 2 COMPLETE]")
        print(f"  Direct links extracted: {len(direct_links_extracted)}")
        print(f"  Evidence sources:")
        print(f"    - Tier 1 (Database): {tier1_count}")
        print(f"    - Tier 2 (Pipeline): {tier2_count}")
        print(f"    - Tier 3 (Extraction): {tier3_count}")
        print(f"  Direct links validated: {direct_links_validated}")
        print(f"  New direct links created: {direct_links_created}\n")

        # Log corrections
        if all_corrections:
            log_corrections(all_corrections, log_file)

        # Summary
        print(f"\n{'='*60}")
        print(f"VALIDATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total interactions processed: {validated_count}")
        print(f"Corrections applied: {len(all_corrections)}")
        print(f"Errors encountered: {error_count}")

        if args.dry_run:
            print(f"\n[DRY RUN] No changes written to database")
            print(f"[DRY RUN] Review corrections in: {log_file}")
            print(f"[DRY RUN] Run without --dry-run to apply corrections")
        else:
            print(f"\n[LIVE] Corrections written to database")
            print(f"[LIVE] Corrections logged to: {log_file}")

        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
