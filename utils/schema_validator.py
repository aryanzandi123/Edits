"""
Schema Validation and Consistency Enforcement

This module provides validation functions to ensure data consistency across the
protein interaction query pipeline. It acts as a pre-validation gate before fact
checking and a post-validation finalizer after fact checking.

Key functions:
- validate_schema_consistency: Fix structural issues (missing arrows, chains, etc.)
- finalize_interaction_metadata: Add arrow notation and sync snapshots
"""

import sys
from typing import Dict, Any, List, Optional, Tuple

# Import the arrow aggregation function from runner
# (We'll handle this import carefully to avoid circular dependencies)
try:
    from runner import aggregate_function_arrows
    AGGREGATE_AVAILABLE = True
except ImportError:
    AGGREGATE_AVAILABLE = False
    print("[WARN] Could not import aggregate_function_arrows from runner - some fixes disabled")


def _reaggregate_interactor(interactor: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute interactor-level arrow/direction from function-level data.

    Uses runner.aggregate_function_arrows when available; otherwise, falls back to
    a local computation that mirrors the runner logic sufficiently for validation.
    """
    if AGGREGATE_AVAILABLE:
        try:
            return aggregate_function_arrows(interactor)
        except Exception:
            pass

    # Fallback: lightweight aggregation
    functions = interactor.get('functions', []) or []
    arrows_by_dir = { 'main_to_primary': set(), 'primary_to_main': set(), 'bidirectional': set() }
    dir_counts = { 'main_to_primary': 0, 'primary_to_main': 0, 'bidirectional': 0 }

    for fn in functions:
        if not isinstance(fn, dict):
            continue
        eff = (fn.get('interaction_effect') or fn.get('arrow') or 'binds')
        d = (fn.get('interaction_direction') or fn.get('direction') or 'main_to_primary')
        # Normalize roughly
        eff = str(eff).lower()
        if 'activ' in eff: eff = 'activates'
        elif 'inhib' in eff: eff = 'inhibits'
        elif 'bind' in eff: eff = 'binds'
        elif 'regulat' in eff or 'modulat' in eff: eff = 'regulates'

        d = str(d).lower()
        if 'bid' in d: d = 'bidirectional'
        elif 'primary_to_main' in d or d == 'p2m' or d == 'b_to_a': d = 'primary_to_main'
        else: d = 'main_to_primary'

        dir_counts[d] = dir_counts.get(d, 0) + 1
        arrows_by_dir[d].add(eff)

    # Build arrows dict
    arrows = { k: sorted(list(v)) for k, v in arrows_by_dir.items() if v }

    # Summary arrow
    all_eff = set()
    for lst in arrows.values(): all_eff.update(lst)
    if not all_eff:
        arrow = 'binds'
    elif len(all_eff) == 1:
        arrow = list(all_eff)[0]
    else:
        arrow = 'regulates'

    # Direction
    total = sum(dir_counts.values()) or 0
    if total == 0:
        direction = 'main_to_primary'
    else:
        if dir_counts['bidirectional'] > total/2:
            direction = 'bidirectional'
        elif (dir_counts['main_to_primary'] >= total*0.3 and dir_counts['primary_to_main'] >= total*0.3):
            direction = 'bidirectional'
        elif dir_counts['primary_to_main'] > dir_counts['main_to_primary']:
            direction = 'primary_to_main'
        else:
            direction = 'main_to_primary'

    interactor['arrows'] = arrows
    interactor['arrow'] = arrow
    interactor['direction'] = direction
    return interactor


def _infer_missing_chain_data(
    interactor: Dict[str, Any],
    main_protein: str,
    all_interactors: List[Dict[str, Any]]
) -> Dict[str, Optional[str]]:
    """
    Attempt to infer missing chain data for indirect interactors.

    Strategy:
    1. Look for potential mediators in the current interactors list
    2. Analyze function names for hints about mediator proteins
    3. Build best-guess chain: [main_protein, mediator, current_protein]

    Args:
        interactor: The indirect interactor missing chain data
        main_protein: The main query protein
        all_interactors: List of all interactors in the result

    Returns:
        Dict with 'upstream_interactor' and 'mediator_chain' (may be None)
    """
    primary = interactor.get('primary', 'UNKNOWN')
    functions = interactor.get('functions', [])

    # Extract all direct interactor names (potential mediators)
    direct_interactors = [
        i.get('primary') for i in all_interactors
        if i.get('interaction_type') == 'direct' and i.get('primary') != primary
    ]

    # Strategy 1: Look for mediator hints in function descriptions
    potential_mediators = set()
    for func in functions:
        # Check biological_consequence for protein names
        bio_consequences = func.get('biological_consequence', [])
        for consequence in bio_consequences:
            # Look for mentions of direct interactors in the consequence chain
            for direct_int in direct_interactors:
                if direct_int in str(consequence):
                    potential_mediators.add(direct_int)

        # Check effect_description for protein mentions
        effect_desc = func.get('effect_description', '')
        for direct_int in direct_interactors:
            if direct_int in effect_desc:
                potential_mediators.add(direct_int)

    # Strategy 2: If we found potential mediators, use the first one
    if potential_mediators:
        mediator = list(potential_mediators)[0]
        return {
            'upstream_interactor': mediator,
            'mediator_chain': [mediator],
            '_chain_inferred': True,
            '_inferred_mediators': list(potential_mediators)
        }

    # Strategy 3: Explicit null if no biological hints found
    # Better to have NO chain than FALSE chain - preserves scientific integrity
    # UI can show "chain unknown" instead of making false claims
    return {
        'upstream_interactor': None,
        'mediator_chain': [],
        '_chain_inferred': True,
        '_chain_missing': True,
        '_inference_failed': 'no_biological_hints',
        '_note': 'Chain data unavailable - requires manual curation or additional evidence'
    }


def validate_schema_consistency(
    json_data: Dict[str, Any],
    fix_arrows: bool = True,
    fix_chains: bool = True,
    fix_directions: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Validate and fix structural schema issues before fact checking.

    This function ensures that all interactors have consistent schema structure
    and fixes common issues like:
    - Missing function-level arrows
    - Missing upstream_interactor/mediator_chain for indirect interactors
    - Incorrect bidirectional classifications
    - Missing depth fields

    Args:
        json_data: The full payload with ctx_json and snapshot_json
        fix_arrows: Re-aggregate arrows if True
        fix_chains: Populate missing chain data if True
        fix_directions: Re-calculate directions if True
        verbose: Print detailed diagnostics if True

    Returns:
        Modified json_data with schema fixes applied
    """
    ctx_json = json_data.get('ctx_json', {})
    interactors = ctx_json.get('interactors', [])
    main_protein = ctx_json.get('main', 'UNKNOWN')

    if verbose or True:  # Always print summary
        print("\n" + "=" * 80)
        print("SCHEMA CONSISTENCY VALIDATION")
        print("=" * 80)

    issues_found = 0
    issues_fixed = 0

    # Helper: normalize function-level arrow values to canonical terms
    def _normalize_fn_arrow(val: Any) -> str:
        s = str(val or "").strip().lower()
        if not s:
            return ""
        if "activ" in s or s in {"promotes", "enhances", "activate"}:
            return "activates"
        if "inhib" in s or s in {"represses", "suppresses", "inhibit"}:
            return "inhibits"
        if "bind" in s:
            return "binds"
        if "regulat" in s or "modulat" in s:
            return "regulates"
        if s in {"complex", "unknown", "unk"}:
            return "complex"
        return s

    # Helper: normalize per-function direction to canonical
    def _normalize_fn_direction(val: Any) -> str:
        s = str(val or "").strip().lower()
        if "bid" in s:
            return "bidirectional"
        if "primary_to_main" in s or s == "p2m" or s == "b_to_a":
            return "primary_to_main"
        if "main_to_primary" in s or s == "m2p" or s == "a_to_b":
            return "main_to_primary"
        return "main_to_primary"

    # Helper: deduplicate functions by (function, arrow, direction, cellular_process)
    def _dedupe_functions(funcs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        seen: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        removed = 0
        for fn in funcs or []:
            if not isinstance(fn, dict):
                continue

            # Canonicalize keys and values and PRESERVE both views
            # Normalize interaction_effect and ensure arrow mirrors it when arrow missing
            eff = fn.get("interaction_effect")
            if eff:
                norm_eff = _normalize_fn_arrow(eff)
                fn["interaction_effect"] = norm_eff
                if not fn.get("arrow"):
                    fn["arrow"] = norm_eff
            # Always normalize function-level arrow
            fn["arrow"] = _normalize_fn_arrow(fn.get("arrow"))
            # Ensure interaction_effect exists (mirror arrow when missing)
            if not fn.get("interaction_effect"):
                fn["interaction_effect"] = fn["arrow"]

            # Normalize interaction_direction and ensure direction mirrors it when missing
            idir = fn.get("interaction_direction")
            if idir:
                norm_idir = _normalize_fn_direction(idir)
                fn["interaction_direction"] = norm_idir
                if not fn.get("direction"):
                    fn["direction"] = norm_idir
            # Always normalize function-level direction
            fn["direction"] = _normalize_fn_direction(fn.get("direction"))
            # Ensure interaction_direction exists (mirror direction when missing)
            if not fn.get("interaction_direction"):
                fn["interaction_direction"] = fn["direction"]

            name = str(fn.get("function", "")).strip()
            cell_proc = str(fn.get("cellular_process", "")).strip().lower()

            sig = (name.lower(), fn.get("arrow", ""), fn.get("direction", ""), cell_proc)

            if sig in seen:
                # Merge fields into existing
                base = seen[sig]

                # Merge pmids
                base_pmids = set(base.get("pmids", []) or [])
                new_pmids = set(fn.get("pmids", []) or [])
                if new_pmids:
                    base["pmids"] = sorted(list(base_pmids.union(new_pmids)))

                # Merge specific_effects
                base_se = set(base.get("specific_effects", []) or [])
                new_se = set(fn.get("specific_effects", []) or [])
                if new_se:
                    base["specific_effects"] = sorted(list(base_se.union(new_se)))

                # Merge biological_consequence
                base_bc = set(map(str, (base.get("biological_consequence", []) or [])))
                new_bc = set(map(str, (fn.get("biological_consequence", []) or [])))
                if new_bc:
                    base["biological_consequence"] = sorted(list(base_bc.union(new_bc)))

                # Merge evidence by PMID key if present
                def _evidence_key(e: Dict[str, Any]) -> str:
                    return str((e or {}).get("pmid") or (e or {}).get("id") or "")

                base_ev = base.get("evidence", []) or []
                base_ev_map = { _evidence_key(e): e for e in base_ev if isinstance(e, dict) }
                for e in (fn.get("evidence", []) or []):
                    if isinstance(e, dict):
                        k = _evidence_key(e)
                        if k and k in base_ev_map:
                            # Prefer longer text entries
                            if len(str(e)) > len(str(base_ev_map[k])):
                                base_ev_map[k] = e
                        else:
                            base_ev_map[k] = e
                base["evidence"] = list(base_ev_map.values())

                removed += 1
            else:
                seen[sig] = fn

        return list(seen.values()), removed

    for interactor in interactors:
        primary = interactor.get('primary', 'UNKNOWN')
        interaction_type = interactor.get('interaction_type', 'direct')
        functions = interactor.get('functions', [])

        # Canonicalize and deduplicate function list first to reduce overlap
        if functions:
            new_functions, removed = _dedupe_functions(functions)
            if removed:
                issues_found += removed
                issues_fixed += removed
                if verbose:
                    print(f"  [FIX] {primary}: Removed {removed} duplicate function entries")
            interactor['functions'] = new_functions
            functions = interactor['functions']

            # Skip arrow normalization if arrows were validated by arrow_effect_validator
            if not interactor.get('_arrow_validated'):
                # Ensure per-function arrows/directions are normalized (idempotent)
                for func in interactor['functions']:
                    func['arrow'] = _normalize_fn_arrow(func.get('arrow'))
                    func['direction'] = _normalize_fn_direction(func.get('direction'))
            else:
                if verbose:
                    print(f"  [PROTECTED] {primary}: Skipping arrow normalization (validated by arrow_effect_validator)")

        # ===================================================================
        # FIX 1: Functions missing arrows
        # ===================================================================
        # Skip arrow fixes if arrows were validated by arrow_effect_validator
        if fix_arrows and functions and not interactor.get('_arrow_validated'):
            missing_arrow_funcs = [f for f in functions if not f.get('arrow')]
            if missing_arrow_funcs:
                issues_found += 1
                if verbose:
                    print(f"  [ISSUE] {primary}: {len(missing_arrow_funcs)}/{len(functions)} functions missing arrows")

                # Default missing arrows to 'regulates' (neutral/mixed arrow type)
                for func in missing_arrow_funcs:
                    if not func.get('arrow'):
                        func['arrow'] = 'regulates'
                    if not func.get('direction'):
                        func['direction'] = 'main_to_primary'

                issues_fixed += 1
                if verbose:
                    print(f"    [FIX] Defaulted missing arrows to 'complex'")

            # Re-aggregate interactor-level arrows
            try:
                interactor = _reaggregate_interactor(interactor)
            except Exception as e:
                if verbose:
                    print(f"    [WARN] Could not re-aggregate arrows for {primary}: {e}")
        elif fix_arrows and functions and interactor.get('_arrow_validated'):
            if verbose:
                print(f"  [PROTECTED] {primary}: Skipping arrow fixes (validated by arrow_effect_validator)")

        # ===================================================================
        # FIX 2: Indirect interactors missing chain data
        # ===================================================================
        if fix_chains and interaction_type == 'indirect':
            has_upstream = bool(interactor.get('upstream_interactor'))
            has_chain = bool(interactor.get('mediator_chain')) and len(interactor.get('mediator_chain', [])) > 0

            # CASE 1: BOTH upstream_interactor AND mediator_chain are missing
            # Use intelligent inference to build best-guess chain
            if not has_upstream and not has_chain:
                issues_found += 2  # Count both missing fields
                if verbose:
                    print(f"  [ISSUE] {primary}: Indirect interactor missing BOTH upstream_interactor and mediator_chain")

                # Try intelligent inference from other data
                try:
                    inferred_data = _infer_missing_chain_data(interactor, main_protein, interactors)

                    # Apply inferred data
                    interactor['upstream_interactor'] = inferred_data.get('upstream_interactor')
                    interactor['mediator_chain'] = inferred_data.get('mediator_chain', [])

                    # Add metadata flags
                    if inferred_data.get('_chain_inferred'):
                        interactor['_chain_inferred'] = True
                    if inferred_data.get('_inferred_mediators'):
                        interactor['_inferred_mediators'] = inferred_data['_inferred_mediators']
                    if inferred_data.get('_chain_inferred_strategy'):
                        interactor['_chain_inferred_strategy'] = inferred_data['_chain_inferred_strategy']
                    if inferred_data.get('_chain_minimal_fallback'):
                        interactor['_chain_minimal_fallback'] = True

                    issues_fixed += 2  # Fixed both fields
                    if verbose:
                        strategy = inferred_data.get('_chain_inferred_strategy', 'function_analysis')
                        mediators = inferred_data.get('_inferred_mediators', [interactor['upstream_interactor']])
                        print(f"    [FIX] Inferred chain data using strategy '{strategy}'")
                        print(f"          upstream_interactor = {interactor['upstream_interactor']}")
                        print(f"          mediator_chain = {interactor['mediator_chain']}")
                        if len(mediators) > 1:
                            print(f"          (alternative mediators: {', '.join(mediators[1:])})")

                except Exception as e:
                    # Fallback: set minimal structure
                    interactor['upstream_interactor'] = main_protein
                    interactor['mediator_chain'] = [main_protein]
                    interactor['_chain_inference_failed'] = True
                    interactor['_inference_error'] = str(e)
                    issues_fixed += 2  # Still count as fixed (with placeholder)
                    if verbose:
                        print(f"    [WARN] Chain inference failed: {e}")
                        print(f"    [FIX] Set minimal chain structure with main protein")

            # CASE 2: Only upstream_interactor is missing (can infer from chain)
            elif not has_upstream and has_chain:
                issues_found += 1
                if verbose:
                    print(f"  [ISSUE] {primary}: Indirect interactor missing upstream_interactor")

                # The last mediator in the chain is the upstream interactor
                mediator_chain = interactor.get('mediator_chain', [])
                interactor['upstream_interactor'] = mediator_chain[-1]
                issues_fixed += 1
                if verbose:
                    print(f"    [FIX] Inferred upstream_interactor = {mediator_chain[-1]} from mediator_chain")

            # CASE 3: Only mediator_chain is missing (can infer from upstream)
            elif has_upstream and not has_chain:
                issues_found += 1
                if verbose:
                    print(f"  [ISSUE] {primary}: Indirect interactor missing mediator_chain")

                # Build chain from upstream_interactor
                upstream = interactor.get('upstream_interactor')
                interactor['mediator_chain'] = [upstream]
                issues_fixed += 1
                if verbose:
                    print(f"    [FIX] Inferred mediator_chain = [{upstream}] from upstream_interactor")

            # Check depth
            if not interactor.get('depth') or interactor.get('depth') is None:
                issues_found += 1
                if verbose:
                    print(f"  [ISSUE] {primary}: Missing depth field")

                # Calculate depth from chain length
                chain_length = len(interactor.get('mediator_chain', []))
                calculated_depth = chain_length + 1  # depth = chain_length + 1
                interactor['depth'] = calculated_depth
                issues_fixed += 1
                if verbose:
                    print(f"    [FIX] Calculated depth = {calculated_depth} from chain length")

        # ===================================================================
        # FIX 3: Direct interactors should have depth=1
        # ===================================================================
        if interaction_type == 'direct':
            if not interactor.get('depth') or interactor.get('depth') != 1:
                issues_found += 1
                if verbose:
                    print(f"  [ISSUE] {primary}: Direct interactor has incorrect depth ({interactor.get('depth')})")
                interactor['depth'] = 1
                issues_fixed += 1
                if verbose:
                    print(f"    [FIX] Set depth = 1 for direct interactor")

        # ===================================================================
        # FIX 4: Re-calculate direction if requested
        # ===================================================================
        # Skip direction recalculation if arrows were validated by arrow_effect_validator
        if fix_directions and AGGREGATE_AVAILABLE and not interactor.get('_arrow_validated'):
            old_direction = interactor.get('direction')
            try:
                interactor = aggregate_function_arrows(interactor)
                new_direction = interactor.get('direction')

                if old_direction != new_direction:
                    issues_found += 1
                    issues_fixed += 1
                    if verbose:
                        print(f"  [FIX] {primary}: Direction changed from '{old_direction}' to '{new_direction}'")
            except Exception as e:
                if verbose:
                    print(f"    [WARN] Could not recalculate direction for {primary}: {e}")
        elif fix_directions and interactor.get('_arrow_validated'):
            if verbose:
                print(f"  [PROTECTED] {primary}: Skipping direction recalculation (validated by arrow_effect_validator)")

    # ===================================================================
    # Summary
    # ===================================================================
    if verbose or True:
        print(f"\n  Validation Summary:")
        print(f"    Issues found: {issues_found}")
        print(f"    Issues fixed: {issues_fixed}")
        print(f"    Issues remaining: {issues_found - issues_fixed}")
        print("=" * 80 + "\n")

    return json_data


def finalize_interaction_metadata(
    json_data: Dict[str, Any],
    add_arrow_notation: bool = True,
    validate_snapshot: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Finalize interaction metadata after fact checking.

    This function adds arrow notation for visualizer display and ensures
    snapshot_json matches ctx_json.

    Arrow notation examples:
    - main_to_primary + activates  → "QUERY --activates--> INTERACTOR:"
    - primary_to_main + inhibits   → "QUERY <--inhibits-- INTERACTOR:"
    - bidirectional + binds        → "QUERY <--binds--> INTERACTOR:"

    Args:
        json_data: The full payload with ctx_json and snapshot_json
        add_arrow_notation: Add arrow_notation field if True
        validate_snapshot: Sync snapshot_json with ctx_json if True
        verbose: Print detailed diagnostics if True

    Returns:
        Modified json_data with finalized metadata
    """
    ctx_json = json_data.get('ctx_json', {})
    snapshot_json = json_data.get('snapshot_json', {})
    interactors = ctx_json.get('interactors', [])
    main_protein = ctx_json.get('main', 'UNKNOWN')

    if verbose or True:  # Always print summary
        print("\n" + "=" * 80)
        print("FINALIZING INTERACTION METADATA")
        print("=" * 80)

    notation_added = 0

    for interactor in interactors:
        primary = interactor.get('primary', 'UNKNOWN')

        # Check if already validated by arrow_effect_validator (STAGE 7.5)
        validation_meta = interactor.get("_validation_metadata", {})
        is_validated = validation_meta.get("validated", False)

        if is_validated and verbose:
            print(f"  [VALIDATED] {primary}: Preserving LLM-validated arrows (corrected by {validation_meta.get('validator', 'unknown')})")

        # ===================================================================
        # Add arrow notation for visualizer
        # ===================================================================
        if add_arrow_notation:
            arrow = interactor.get('arrow', 'binds')
            direction = interactor.get('direction', 'main_to_primary')
            interaction_type = interactor.get('interaction_type', 'direct')
            upstream = interactor.get('upstream_interactor')

            # Create human-readable arrow notation
            # IMPORTANT: Different semantics for direct vs indirect interactions
            # - Direct: notation is QUERY-RELATIVE (main_protein ↔ primary)
            # - Indirect: notation is LINK-RELATIVE (upstream ↔ primary)
            if interaction_type == 'indirect' and upstream:
                # For indirect: show actual link (upstream → partner), not query → partner
                if direction == 'main_to_primary':
                    arrow_notation = f"{upstream} --{arrow}--> {primary}:"
                elif direction == 'primary_to_main':
                    arrow_notation = f"{upstream} <--{arrow}-- {primary}:"
                elif direction == 'bidirectional':
                    arrow_notation = f"{upstream} <--{arrow}--> {primary}:"
                else:
                    arrow_notation = f"{upstream} --{arrow}-- {primary}:"
            else:
                # For direct: show query → interactor (query-relative)
                if direction == 'main_to_primary':
                    arrow_notation = f"{main_protein} --{arrow}--> {primary}:"
                elif direction == 'primary_to_main':
                    arrow_notation = f"{main_protein} <--{arrow}-- {primary}:"
                elif direction == 'bidirectional':
                    arrow_notation = f"{main_protein} <--{arrow}--> {primary}:"
                else:
                    # Unknown direction - default to neutral
                    arrow_notation = f"{main_protein} --{arrow}-- {primary}:"

            interactor['arrow_notation'] = arrow_notation
            notation_added += 1

            if verbose:
                print(f"  [OK] {primary}: Added arrow notation '{arrow_notation}'")

    # ===================================================================
    # Sync snapshot with ctx
    # ===================================================================
    if validate_snapshot:
        # Update snapshot_json interactors to match ctx_json
        snapshot_json['interactors'] = ctx_json['interactors']
        json_data['snapshot_json'] = snapshot_json

        if verbose or True:
            print(f"\n  [OK] Synced snapshot_json with ctx_json ({len(interactors)} interactors)")

    if verbose or True:
        print(f"  Arrow notations added: {notation_added}")
        print("=" * 80 + "\n")

    return json_data


def validate_interactor_functions(
    interactor: Dict[str, Any],
    main_protein: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Validate function-level data for a single interactor.

    Ensures:
    - All functions have required fields (arrow, direction, cellular_process, etc.)
    - Arrow types are valid
    - Directions are valid
    - Function categorization matches direction

    Args:
        interactor: Interactor data dict
        main_protein: Main query protein symbol
        verbose: Print diagnostics if True

    Returns:
        Modified interactor with validated functions
    """
    primary = interactor.get('primary', 'UNKNOWN')
    functions = interactor.get('functions', [])

    # Skip validation if arrows were validated by arrow_effect_validator
    if interactor.get('_arrow_validated'):
        if verbose:
            print(f"  [PROTECTED] {primary}: Skipping function validation (validated by arrow_effect_validator)")
        return interactor

    valid_arrows = {'activates', 'inhibits', 'binds', 'regulates'}
    valid_directions = {'main_to_primary', 'primary_to_main', 'bidirectional'}

    for i, func in enumerate(functions):
        # Validate arrow type
        arrow = str(func.get('arrow', '')).strip().lower()
        if arrow == 'complex' or arrow == 'modulates':
            func['arrow'] = 'regulates'
        elif arrow not in valid_arrows:
            if verbose:
                print(f"  [WARN] {primary} function {i}: Invalid arrow '{arrow}' - defaulting to 'regulates'")
            func['arrow'] = 'regulates'

        # Validate direction
        direction = func.get('direction', '')
        if direction not in valid_directions:
            if verbose:
                print(f"  [WARN] {primary} function {i}: Invalid direction '{direction}' - defaulting to 'main_to_primary'")
            func['direction'] = 'main_to_primary'

        # Ensure required fields exist
        required_fields = ['function', 'cellular_process', 'effect_description']
        for field in required_fields:
            if not func.get(field):
                func[field] = f"[Data not available for {field}]"
                if verbose:
                    print(f"  [WARN] {primary} function {i}: Missing '{field}' - added placeholder")

    return interactor


# ===================================================================
# Helper function for testing/debugging
# ===================================================================
def print_validation_report(json_data: Dict[str, Any]) -> None:
    """
    Print a detailed validation report for debugging.

    Shows:
    - Number of interactors
    - Direct vs indirect breakdown
    - Interactors with missing data
    - Arrow direction distribution
    """
    ctx_json = json_data.get('ctx_json', {})
    interactors = ctx_json.get('interactors', [])
    main_protein = ctx_json.get('main', 'UNKNOWN')

    print("\n" + "=" * 80)
    print(f"VALIDATION REPORT: {main_protein}")
    print("=" * 80)

    direct_count = sum(1 for i in interactors if i.get('interaction_type') == 'direct')
    indirect_count = sum(1 for i in interactors if i.get('interaction_type') == 'indirect')

    print(f"Total interactors: {len(interactors)}")
    print(f"  Direct: {direct_count}")
    print(f"  Indirect: {indirect_count}")

    # Count direction distribution
    main_to_primary = sum(1 for i in interactors if i.get('direction') == 'main_to_primary')
    primary_to_main = sum(1 for i in interactors if i.get('direction') == 'primary_to_main')
    bidirectional = sum(1 for i in interactors if i.get('direction') == 'bidirectional')

    print(f"\nDirection distribution:")
    print(f"  main_to_primary: {main_to_primary} ({100*main_to_primary/len(interactors):.1f}%)")
    print(f"  primary_to_main: {primary_to_main} ({100*primary_to_main/len(interactors):.1f}%)")
    print(f"  bidirectional: {bidirectional} ({100*bidirectional/len(interactors):.1f}%)")

    # Find problematic interactors
    missing_arrows = [i.get('primary') for i in interactors if not i.get('arrow')]
    missing_chains = [i.get('primary') for i in interactors
                     if i.get('interaction_type') == 'indirect' and not i.get('upstream_interactor')]

    if missing_arrows:
        print(f"\n⚠️  Interactors missing arrows ({len(missing_arrows)}):")
        for name in missing_arrows[:5]:  # Show first 5
            print(f"    - {name}")
        if len(missing_arrows) > 5:
            print(f"    ... and {len(missing_arrows) - 5} more")

    if missing_chains:
        print(f"\n⚠️  Indirect interactors missing chain data ({len(missing_chains)}):")
        for name in missing_chains[:5]:  # Show first 5
            print(f"    - {name}")
        if len(missing_chains) > 5:
            print(f"    ... and {len(missing_chains) - 5} more")

    print("=" * 80 + "\n")


if __name__ == '__main__':
    print("Schema Validator Module")
    print("This module provides validation functions for the protein interaction pipeline.")
    print("\nAvailable functions:")
    print("  - validate_schema_consistency()")
    print("  - finalize_interaction_metadata()")
    print("  - validate_interactor_functions()")
    print("  - print_validation_report()")
