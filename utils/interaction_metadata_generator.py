#!/usr/bin/env python3
"""
Interaction Metadata Generator
Synthesizes comprehensive interaction-level metadata from function-level evidence.

This module runs AFTER evidence validation to:
1. Determine interaction arrow/intent based on ALL function-level arrows
2. Generate MECHANISM field from all cellular_process fields
3. Generate EFFECT field from all effect_description fields
4. Generate SUMMARY field (1-2 sentence biological overview)
5. Compile ALL evidence from ALL functions (deduplicate PMIDs)
6. Remove confidence fields from output
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class MetadataGeneratorError(RuntimeError):
    """Raised when metadata generation fails."""
    pass


def determine_interaction_arrow(functions: List[Dict[str, Any]]) -> str:
    """
    Determine interaction-level arrow based on ALL function-level arrows.

    Logic:
    - If ALL functions activate: return "activates"
    - If ALL functions inhibit: return "inhibits"
    - If MIXED (some activate, some inhibit): return "regulates" or "modulates"
    - If only binding/no clear direction: return "binds"

    Args:
        functions: List of function dictionaries

    Returns:
        str: The determined arrow type ("activates", "inhibits", "regulates", or "binds")
    """
    if not functions:
        return "binds"

    arrows = [f.get("arrow", "").lower() for f in functions if f.get("arrow")]

    if not arrows:
        return "binds"

    # Count arrow types
    activates_count = sum(1 for a in arrows if a in ["activates", "activate", "promotes", "enhances"])
    inhibits_count = sum(1 for a in arrows if a in ["inhibits", "inhibit", "suppresses", "represses"])
    binds_count = sum(1 for a in arrows if a == "binds")

    # Decision logic
    if activates_count > 0 and inhibits_count == 0:
        return "activates"
    elif inhibits_count > 0 and activates_count == 0:
        return "inhibits"
    elif activates_count > 0 and inhibits_count > 0:
        # Mixed effects - use "regulates" or "modulates"
        return "regulates"
    elif binds_count > 0:
        return "binds"
    else:
        return "binds"  # Default fallback


def determine_interaction_intent(functions: List[Dict[str, Any]], current_intent: str) -> str:
    """
    Determine or refine interaction-level intent based on functions.

    Args:
        functions: List of function dictionaries
        current_intent: Current intent from interaction level

    Returns:
        str: Refined intent description
    """
    if not functions:
        return current_intent or "binding"

    # If current intent is already specific, keep it
    if current_intent and current_intent not in ["binding", "interaction", "unknown"]:
        return current_intent

    # Extract mechanisms from cellular_process fields
    mechanisms = []
    for func in functions:
        cellular_process = func.get("cellular_process", "")
        if cellular_process:
            # Extract key mechanism terms
            lower = cellular_process.lower()
            if "phosphorylat" in lower:
                mechanisms.append("phosphorylation")
            elif "ubiquitin" in lower:
                mechanisms.append("ubiquitination")
            elif "deubiquitin" in lower:
                mechanisms.append("deubiquitination")
            elif "acetylat" in lower:
                mechanisms.append("acetylation")
            elif "deacetylat" in lower:
                mechanisms.append("deacetylation")
            elif "methylat" in lower:
                mechanisms.append("methylation")
            elif "sumoylat" in lower:
                mechanisms.append("sumoylation")

    if mechanisms:
        # Return most common mechanism or first one
        return mechanisms[0]

    return current_intent or "regulation"


def generate_mechanism_field(functions: List[Dict[str, Any]]) -> str:
    """
    Synthesize MECHANISM field from ALL function cellular_process fields.

    Describes HOW the interaction occurs at the molecular level:
    - Binding domains
    - Post-translational modifications
    - Conformational changes
    - Complex formation

    Args:
        functions: List of function dictionaries

    Returns:
        str: Synthesized mechanism description
    """
    if not functions:
        return "Molecular mechanism not fully characterized"

    # Collect all cellular_process descriptions
    mechanisms = []
    for func in functions:
        cellular_process = func.get("cellular_process", "")
        if cellular_process and len(cellular_process.strip()) > 10:
            mechanisms.append(cellular_process.strip())

    if not mechanisms:
        return "Molecular mechanism not fully characterized"

    # If only one mechanism, return it
    if len(mechanisms) == 1:
        return mechanisms[0]

    # If multiple mechanisms, create a synthesis
    # Strategy: Combine key molecular details from all mechanisms
    combined = " ".join(mechanisms)

    # If combined is too long, create a summary
    if len(combined) > 500:
        # Extract key mechanism types
        mech_types = []
        for m in mechanisms:
            # Extract first sentence or first 100 chars
            first_part = m.split('.')[0][:100]
            if first_part not in mech_types:
                mech_types.append(first_part)

        return "; ".join(mech_types[:3]) + ("..." if len(mech_types) > 3 else "")

    return combined


def generate_effect_field(functions: List[Dict[str, Any]]) -> str:
    """
    Synthesize EFFECT field from ALL function effect_description fields.

    Describes WHAT happens as a result of the interaction:
    - Directional outcomes (activation, inhibition, modulation)
    - Biological processes affected
    - Specific molecular effects

    Args:
        functions: List of function dictionaries

    Returns:
        str: Synthesized effect description
    """
    if not functions:
        return "Functional effects not fully characterized"

    # Collect all effect_description fields
    effects = []
    for func in functions:
        effect_desc = func.get("effect_description", "")
        if effect_desc and len(effect_desc.strip()) > 5:
            effects.append(effect_desc.strip())

    if not effects:
        # Fallback: use function names + arrows
        effect_parts = []
        for func in functions:
            func_name = func.get("function", "")
            arrow = func.get("arrow", "")
            if func_name and arrow:
                if arrow.lower() in ["activates", "activate"]:
                    effect_parts.append(f"Enhances {func_name.lower()}")
                elif arrow.lower() in ["inhibits", "inhibit"]:
                    effect_parts.append(f"Reduces {func_name.lower()}")

        if effect_parts:
            return "; ".join(effect_parts[:3]) + ("..." if len(effect_parts) > 3 else "")

        return "Functional effects not fully characterized"

    # If only one effect, return it
    if len(effects) == 1:
        return effects[0]

    # If multiple effects, combine them intelligently
    # Strategy: Join with semicolons, limit to ~300 chars
    combined = "; ".join(effects)

    if len(combined) > 300:
        # Truncate intelligently
        return "; ".join(effects[:2]) + ("; ..." if len(effects) > 2 else "")

    return combined


def generate_summary_field(
    main_protein: str,
    interactor: str,
    functions: List[Dict[str, Any]],
    arrow: str
) -> str:
    """
    Generate SUMMARY field: 1-2 sentence overview of interaction and biological significance.

    Must synthesize information from ALL functions to create a coherent statement
    about the overall biological role of this interaction.

    Args:
        main_protein: Main protein symbol
        interactor: Interactor protein symbol
        functions: List of all function dictionaries for this interaction
        arrow: Determined interaction arrow

    Returns:
        str: 1-2 sentence summary
    """
    if not functions:
        return f"{main_protein} and {interactor} interact, though the functional significance remains to be fully elucidated."

    # Extract key biological themes from functions
    function_names = [f.get("function", "") for f in functions if f.get("function")]

    # Determine action verb based on arrow
    if arrow == "activates":
        action = "activates"
    elif arrow == "inhibits":
        action = "inhibits"
    elif arrow == "regulates":
        action = "regulates"
    else:
        action = "interacts with"

    # Create summary based on number of functions
    if len(function_names) == 1:
        summary = f"{main_protein} {action} {interactor} to modulate {function_names[0].lower()}"
    elif len(function_names) == 2:
        summary = f"{main_protein} {action} {interactor} to regulate {function_names[0].lower()} and {function_names[1].lower()}"
    else:
        # Multiple functions - create broader summary
        summary = f"{main_protein} {action} {interactor} to control multiple cellular processes including {function_names[0].lower()}, {function_names[1].lower()}"
        if len(function_names) > 2:
            summary += f", and {function_names[2].lower()}"

    # Add biological context from biological_consequence if available
    consequences = []
    for func in functions[:2]:  # Check first 2 functions
        bio_cons = func.get("biological_consequence", [])
        if isinstance(bio_cons, list) and bio_cons:
            consequences.extend(bio_cons[:1])  # Take first consequence from each

    if consequences:
        # Extract final outcome from first consequence
        first_cons = consequences[0]
        # Try to extract the final part after last arrow
        if "→" in first_cons:
            parts = first_cons.split("→")
            final_outcome = parts[-1].strip()
            summary += f", ultimately affecting {final_outcome.lower()}"

    # Ensure it ends with a period
    if not summary.endswith("."):
        summary += "."

    return summary


def compile_evidence(functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compile ALL evidence from ALL function boxes, deduplicating by PMID.

    Args:
        functions: List of function dictionaries

    Returns:
        List[Dict]: Compiled evidence array with duplicates removed
    """
    if not functions:
        return []

    # Collect all evidence entries
    all_evidence = []
    seen_pmids: Set[str] = set()

    for func in functions:
        func_evidence = func.get("evidence", [])
        if not isinstance(func_evidence, list):
            continue

        for evidence_entry in func_evidence:
            if not isinstance(evidence_entry, dict):
                continue

            pmid = evidence_entry.get("pmid", "")

            # If PMID exists and we've seen it, skip
            if pmid and pmid in seen_pmids:
                continue

            # Add to result
            all_evidence.append(deepcopy(evidence_entry))

            # Mark PMID as seen
            if pmid:
                seen_pmids.add(pmid)

    return all_evidence


def remove_confidence_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove all confidence fields from the payload (both interaction and function level).

    Args:
        data: The full payload dictionary

    Returns:
        Dict: Cleaned payload without confidence fields
    """
    cleaned = deepcopy(data)

    # Remove from ctx_json interactors
    if "ctx_json" in cleaned and "interactors" in cleaned["ctx_json"]:
        for interactor in cleaned["ctx_json"]["interactors"]:
            # Remove interaction-level confidence
            if "confidence" in interactor:
                del interactor["confidence"]

            # Remove function-level confidence
            if "functions" in interactor:
                for func in interactor["functions"]:
                    if "confidence" in func:
                        del func["confidence"]

    # Remove from snapshot_json interactors
    if "snapshot_json" in cleaned and "interactors" in cleaned["snapshot_json"]:
        for interactor in cleaned["snapshot_json"]["interactors"]:
            # Remove interaction-level confidence
            if "confidence" in interactor:
                del interactor["confidence"]

            # Remove function-level confidence
            if "functions" in interactor:
                for func in interactor["functions"]:
                    if "confidence" in func:
                        del func["confidence"]

    return cleaned


def generate_interaction_metadata(
    payload: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Main function: Generate comprehensive interaction-level metadata from function-level data.

    For each interactor:
    1. Determine interaction arrow from ALL function arrows
    2. Refine interaction intent
    3. Generate MECHANISM field from all cellular_process fields
    4. Generate EFFECT field from all effect_description fields
    5. Generate SUMMARY field (1-2 sentence overview)
    6. Compile ALL evidence from ALL functions
    7. Remove confidence fields

    Args:
        payload: The full pipeline payload with ctx_json and snapshot_json
        verbose: Enable detailed logging

    Returns:
        Dict: Updated payload with synthesized interaction metadata
    """
    if verbose:
        print(f"\n{'='*80}")
        print("GENERATING INTERACTION-LEVEL METADATA")
        print(f"{'='*80}")

    # Work on deep copy to avoid mutation
    result = deepcopy(payload)

    # Get main protein
    main_protein = result.get("ctx_json", {}).get("main", "UNKNOWN")

    # Process ctx_json interactors
    ctx_interactors = result.get("ctx_json", {}).get("interactors", [])

    if verbose:
        print(f"Processing {len(ctx_interactors)} interactors for {main_protein}...")

    for idx, interactor in enumerate(ctx_interactors):
        interactor_name = interactor.get("primary", "UNKNOWN")
        functions = interactor.get("functions", [])

        if verbose:
            print(f"\n[{idx+1}/{len(ctx_interactors)}] {main_protein} ↔ {interactor_name}")
            print(f"  Functions: {len(functions)}")

        # Check if already validated by arrow_effect_validator (STAGE 7.5)
        # If validated, skip arrow recalculation to preserve LLM corrections
        validation_meta = interactor.get("_validation_metadata", {})
        is_validated = validation_meta.get("validated", False)

        if is_validated:
            if verbose:
                print(f"  [SKIP] Arrow already validated by {validation_meta.get('validator', 'unknown')}")
            determined_arrow = interactor.get("arrow", "regulates")  # Use existing validated arrow
        else:
            # 1. Determine interaction arrow
            determined_arrow = determine_interaction_arrow(functions)
            old_arrow = interactor.get("arrow", "")
            interactor["arrow"] = determined_arrow

            if verbose and old_arrow != determined_arrow:
                print(f"  Arrow: {old_arrow} → {determined_arrow}")

        # 2. Refine interaction intent
        current_intent = interactor.get("intent", "")
        refined_intent = determine_interaction_intent(functions, current_intent)
        interactor["intent"] = refined_intent

        # 3. Generate MECHANISM field
        mechanism = generate_mechanism_field(functions)
        interactor["mechanism"] = mechanism

        if verbose:
            print(f"  Mechanism: {mechanism[:80]}...")

        # 4. Generate EFFECT field
        effect = generate_effect_field(functions)
        interactor["effect"] = effect

        if verbose:
            print(f"  Effect: {effect[:80]}...")

        # 5. Generate SUMMARY field
        summary = generate_summary_field(main_protein, interactor_name, functions, determined_arrow)
        interactor["summary"] = summary

        if verbose:
            print(f"  Summary: {summary[:80]}...")

        # 6. Compile evidence from all functions
        compiled_evidence = compile_evidence(functions)

        # Update interaction-level evidence (merge with existing if present)
        existing_evidence = interactor.get("evidence", [])
        if not isinstance(existing_evidence, list):
            existing_evidence = []

        # Merge compiled evidence with existing (deduplicate by PMID)
        seen_pmids = {e.get("pmid") for e in existing_evidence if e.get("pmid")}
        for evidence_entry in compiled_evidence:
            pmid = evidence_entry.get("pmid")
            if not pmid or pmid not in seen_pmids:
                existing_evidence.append(evidence_entry)
                if pmid:
                    seen_pmids.add(pmid)

        interactor["evidence"] = existing_evidence

        if verbose:
            print(f"  Evidence: {len(existing_evidence)} total citations")

    # Also update snapshot_json if present
    if "snapshot_json" in result:
        snapshot_interactors = result["snapshot_json"].get("interactors", [])

        # Match snapshot interactors with ctx interactors by primary name
        ctx_map = {i.get("primary"): i for i in ctx_interactors}

        for snap_int in snapshot_interactors:
            primary = snap_int.get("primary")
            if primary in ctx_map:
                ctx_int = ctx_map[primary]
                # Copy synthesized fields
                snap_int["arrow"] = ctx_int.get("arrow")
                snap_int["intent"] = ctx_int.get("intent")
                snap_int["mechanism"] = ctx_int.get("mechanism")
                snap_int["effect"] = ctx_int.get("effect")
                snap_int["summary"] = ctx_int.get("summary")
                snap_int["evidence"] = ctx_int.get("evidence", [])

    # 7. Remove all confidence fields
    result = remove_confidence_fields(result)

    if verbose:
        print(f"\n{'='*80}")
        print("[OK]INTERACTION METADATA GENERATION COMPLETE")
        print(f"{'='*80}")

    return result


def main():
    """CLI entry point for testing/debugging."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate interaction-level metadata from function-level data"
    )
    parser.add_argument(
        "input_json",
        type=str,
        help="Path to validated JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path (default: <input>_with_metadata.json)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Load input
    input_path = Path(args.input_json)
    if not input_path.exists():
        sys.exit(f"❌ Input file not found: {input_path}")

    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    # Generate metadata
    result = generate_interaction_metadata(payload, verbose=args.verbose)

    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_with_metadata{input_path.suffix}"

    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("[OK]Done!")


if __name__ == "__main__":
    main()
