#!/usr/bin/env python3
"""
Function Name Cleaner
Removes "regulation" and other generic terms from function names.
"""
import re
from typing import Dict, Any, List


def clean_function_name(function_name: str) -> str:
    """
    Clean a function name by removing redundant words like "regulation".

    The arrow field already indicates HOW something is regulated (activates/inhibits),
    so "regulation" in the function name is redundant.

    Examples:
        "Apoptosis Regulation" → "Apoptosis"
        "Mitophagy Regulation" → "Mitophagy"
        "Regulation of ATXN3 Stability" → "ATXN3 Stability"
        "ATXN3 Aggregation Regulation" → "ATXN3 Aggregation"
        "Cell Cycle Regulation" → "Cell Cycle Progression"
    """
    if not function_name:
        return function_name

    original = function_name
    cleaned = function_name

    # Pattern 1: "X Regulation" → "X"
    cleaned = re.sub(r'\s+Regulation$', '', cleaned, flags=re.IGNORECASE)

    # Pattern 2: "Regulation of X" → "X"
    cleaned = re.sub(r'^Regulation\s+of\s+', '', cleaned, flags=re.IGNORECASE)

    # Pattern 3: "X Regulation of Y" → "X Y" (preserve both parts)
    # Example: "Transcriptional Regulation of p53" → "Transcriptional p53"
    cleaned = re.sub(r'\s+Regulation\s+of\s+', ' ', cleaned, flags=re.IGNORECASE)

    # Special case: "Cell Cycle Regulation" → "Cell Cycle Progression"
    if re.match(r'^Cell\s+Cycle$', cleaned, flags=re.IGNORECASE):
        cleaned = "Cell Cycle Progression"

    # Pattern 4: Remove outcome-based suffixes
    # "Apoptosis Suppression" → "Apoptosis"
    # "Autophagy Induction" → "Autophagy"
    # WHY: Creates double-negatives ("inhibits suppression" = confusing!)
    outcome_patterns = [
        (r'\s+Suppression$', ''),
        (r'\s+Promotion$', ''),
        (r'\s+Enhancement$', ''),
        (r'\s+Inhibition$', ''),
        (r'\s+Induction$', ''),
        (r'\s+Activation$', ''),
        (r'\s+Stimulation$', ''),
    ]

    for pattern, replacement in outcome_patterns:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # Clean up any double spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Log if change was made
    if cleaned != original:
        try:
            print(f"  Cleaned function name: '{original}' -> '{cleaned}'")
        except UnicodeEncodeError:
            # Handle Windows console encoding issues
            print(f"  Cleaned function name: [contains special chars] -> [cleaned]")

    return cleaned


def clean_payload_function_names(payload: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Clean all function names in a pipeline payload (ctx_json).

    Args:
        payload: Pipeline output with ctx_json containing interactors
        verbose: Print cleaning operations

    Returns:
        Modified payload with cleaned function names
    """
    if verbose:
        print("\n" + "="*80)
        print("CLEANING FUNCTION NAMES")
        print("="*80)

    ctx_json = payload.get("ctx_json", {})
    interactors = ctx_json.get("interactors", [])

    total_cleaned = 0

    for interactor in interactors:
        primary = interactor.get("primary", "UNKNOWN")
        functions = interactor.get("functions", [])

        if not functions:
            continue

        if verbose:
            print(f"\nProcessing {primary}...")

        for func in functions:
            original_name = func.get("function", "")
            if not original_name:
                continue

            cleaned_name = clean_function_name(original_name)

            if cleaned_name != original_name:
                func["function"] = cleaned_name
                total_cleaned += 1

    if verbose:
        print(f"\n{'='*80}")
        print(f"SUMMARY: Cleaned {total_cleaned} function name(s)")
        print(f"{'='*80}\n")

    return payload


def clean_snapshot_function_names(snapshot: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Clean all function names in a snapshot_json.

    Args:
        snapshot: Snapshot JSON with main and interactors
        verbose: Print cleaning operations

    Returns:
        Modified snapshot with cleaned function names
    """
    if verbose:
        print("\n" + "="*80)
        print("CLEANING SNAPSHOT FUNCTION NAMES")
        print("="*80)

    interactors = snapshot.get("interactors", [])
    total_cleaned = 0

    for interactor in interactors:
        primary = interactor.get("primary", "UNKNOWN")
        functions = interactor.get("functions", [])

        if not functions:
            continue

        if verbose:
            print(f"\nProcessing {primary}...")

        for func in functions:
            original_name = func.get("function", "")
            if not original_name:
                continue

            cleaned_name = clean_function_name(original_name)

            if cleaned_name != original_name:
                func["function"] = cleaned_name
                total_cleaned += 1

    if verbose:
        print(f"\n{'='*80}")
        print(f"SUMMARY: Cleaned {total_cleaned} function name(s)")
        print(f"{'='*80}\n")

    return snapshot


if __name__ == "__main__":
    # Test cases
    test_names = [
        # Regulation patterns (original)
        "Apoptosis Regulation",
        "Mitophagy Regulation",
        "Regulation of ATXN3 Stability",
        "ATXN3 Aggregation Regulation",
        "Cell Cycle Regulation",
        "Transcriptional Regulation of p53",

        # Outcome-based patterns (new)
        "Apoptosis Suppression",  # The problematic VCP-Akt case!
        "Cell Survival Promotion",
        "Autophagy Induction",
        "Growth Inhibition",
        "Transcription Activation",
        "Protein Degradation Enhancement",
        "mTOR Inhibition",

        # Should not change
        "DNA Repair",
        "Autophagy",
        "Apoptosis",
        "Cell Growth",
    ]

    print("Testing function name cleaning:")
    print("="*80)
    for name in test_names:
        cleaned = clean_function_name(name)
        status = "[CLEANED]" if cleaned != name else "[NO CHANGE]"
        print(f"{status} '{name}' -> '{cleaned}'")
