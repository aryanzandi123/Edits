"""
AI-Powered Function Deduplication Script

Uses Gemini 2.5 Flash to intelligently detect and remove duplicate functions
for the same interaction, even when function names differ slightly.

If one function is more correct than another, keeps the better one.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from copy import deepcopy

try:
    import google.genai as genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Error: google-genai not installed. Run: pip install google-genai")
    sys.exit(1)

from dotenv import load_dotenv

load_dotenv()


def call_gemini_flash(prompt: str, api_key: str) -> str:
    """Call Gemini 2.5 Flash for lightweight deduplication checks."""
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,  # Low temperature for consistency
            max_output_tokens=8000,
        )
    )

    return response.text.strip()


def compare_functions(func1: Dict[str, Any], func2: Dict[str, Any],
                      interaction: str, api_key: str) -> Tuple[bool, int]:
    """
    Compare two functions using AI to determine if they're duplicates.

    Returns:
        (is_duplicate, better_index):
            - is_duplicate: True if functions describe the same thing
            - better_index: 1 if func1 is better, 2 if func2 is better, 0 if equal
    """
    func1_name = func1.get("function", "Unknown")
    func2_name = func2.get("function", "Unknown")

    func1_process = func1.get("cellular_process", "")
    func2_process = func2.get("cellular_process", "")

    func1_effect = func1.get("effect_description", "")
    func2_effect = func2.get("effect_description", "")

    func1_pmids = func1.get("pmids", [])
    func2_pmids = func2.get("pmids", [])

    # CRITICAL: Include direction field for proper deduplication
    func1_direction = func1.get("direction", "unknown")
    func2_direction = func2.get("direction", "unknown")

    func1_arrow = func1.get("arrow", "unknown")
    func2_arrow = func2.get("arrow", "unknown")

    prompt = f"""You are a molecular biology expert tasked with identifying duplicate functions.

INTERACTION: {interaction}

FUNCTION 1:
- Name: {func1_name}
- Direction: {func1_direction}
- Arrow: {func1_arrow}
- Cellular Process: {func1_process}
- Effect: {func1_effect}
- PMIDs: {', '.join(map(str, func1_pmids))}

FUNCTION 2:
- Name: {func2_name}
- Direction: {func2_direction}
- Arrow: {func2_arrow}
- Cellular Process: {func2_process}
- Effect: {func2_effect}
- PMIDs: {', '.join(map(str, func2_pmids))}

TASK:
1. Determine if these two functions describe the SAME biological function (even if worded differently)
2. If they are duplicates, determine which one is MORE CORRECT/COMPLETE

CRITICAL RULES:
- Functions with DIFFERENT directions (main_to_primary vs primary_to_main) are NOT duplicates
  Example: "IRE1A → Sel1L: Activates Sel1L expression" vs "Sel1L → IRE1A: Degrades IRE1A" are DIFFERENT
- Functions with SAME direction describing same process ARE duplicates
  Example: "IRE1A Protein Degradation" with direction "primary_to_main" appearing twice IS a duplicate

IMPORTANT:
- Functions are duplicates if they describe the same biological process/outcome AND have the same direction
- Minor wording differences don't make them different functions
- "DNA Repair" and "DNA Damage Repair" are duplicates
- "Autophagy" and "ER-phagy" are NOT duplicates (ER-phagy is specific)
- Different interaction directions = ALWAYS NOT duplicates

OUTPUT FORMAT (respond with ONLY this format):
DUPLICATE: [YES or NO]
BETTER: [1 or 2 or EQUAL]
REASON: [brief explanation]

Example output:
DUPLICATE: YES
BETTER: 2
REASON: Function 2 has more specific mechanistic details and correct PMIDs.
"""

    try:
        response = call_gemini_flash(prompt, api_key)

        # Parse response
        lines = response.strip().split('\n')
        is_duplicate = False
        better_index = 0

        for line in lines:
            line = line.strip()
            if line.startswith("DUPLICATE:"):
                is_duplicate = "YES" in line.upper()
            elif line.startswith("BETTER:"):
                if "1" in line:
                    better_index = 1
                elif "2" in line:
                    better_index = 2
                else:
                    better_index = 0

        return is_duplicate, better_index

    except Exception as e:
        print(f"  ⚠ Error comparing functions: {e}", file=sys.stderr)
        # On error, assume not duplicate to be safe
        return False, 0


def deduplicate_interactor_functions(interactor: Dict[str, Any],
                                     interaction_name: str,
                                     api_key: str,
                                     verbose: bool = True) -> Dict[str, Any]:
    """
    Remove duplicate functions from a single interactor using AI comparison.

    Returns:
        Modified interactor with duplicates removed
    """
    functions = interactor.get("functions", [])

    if len(functions) <= 1:
        return interactor

    if verbose:
        print(f"\n  Checking {len(functions)} functions for {interactor.get('primary', 'Unknown')}...")

    # Track which functions to keep
    keep_functions = []
    skip_indices = set()

    for i in range(len(functions)):
        if i in skip_indices:
            continue

        current_func = functions[i]
        is_duplicate_of_later = False

        # Compare with all later functions
        for j in range(i + 1, len(functions)):
            if j in skip_indices:
                continue

            later_func = functions[j]

            if verbose:
                print(f"    Comparing '{current_func.get('function')}' vs '{later_func.get('function')}'...", end=" ")

            is_dup, better = compare_functions(current_func, later_func, interaction_name, api_key)

            if is_dup:
                if verbose:
                    print(f"DUPLICATE (keeping function {better if better else 'both'})")

                if better == 1:
                    # Current is better, skip the later one
                    skip_indices.add(j)
                elif better == 2:
                    # Later is better, skip current and stop checking
                    skip_indices.add(i)
                    is_duplicate_of_later = True
                    break
                else:
                    # Equal quality, keep first one (current), skip later
                    skip_indices.add(j)
            else:
                if verbose:
                    print("Different functions")

        if not is_duplicate_of_later:
            keep_functions.append(current_func)

    # Update interactor with deduplicated functions
    result = deepcopy(interactor)
    result["functions"] = keep_functions

    removed_count = len(functions) - len(keep_functions)
    if removed_count > 0 and verbose:
        print(f"  [OK]Removed {removed_count} duplicate function(s)")

    return result


def deduplicate_json_file(json_path: str, api_key: str,
                          output_path: str = None,
                          verbose: bool = True) -> None:
    """
    Process a JSON file and remove duplicate functions for each interaction.
    """
    json_path = Path(json_path)

    if not json_path.exists():
        print(f"Error: File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"\n{'='*80}")
        print(f"AI-Powered Function Deduplication")
        print(f"{'='*80}")
        print(f"Processing: {json_path.name}")

    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get interactors from ctx_json or snapshot_json (legacy format fallback)
    ctx_json = data.get("ctx_json", {})
    snapshot_json = data.get("snapshot_json", {})

    # Try ctx_json first (new format), then snapshot_json (legacy format)
    main_protein = ctx_json.get("main") or snapshot_json.get("main", "Unknown")
    interactors = ctx_json.get("interactors") or snapshot_json.get("interactors", [])

    if not interactors:
        print("No interactors found in ctx_json or snapshot_json", file=sys.stderr)
        return

    if verbose:
        print(f"Main protein: {main_protein}")
        print(f"Found {len(interactors)} interactors\n")

    # Process each interactor
    modified_interactors = []
    total_removed = 0

    for interactor in interactors:
        primary = interactor.get("primary", "Unknown")
        interaction_name = f"{main_protein} ↔ {primary}"

        original_func_count = len(interactor.get("functions", []))

        deduplicated = deduplicate_interactor_functions(
            interactor,
            interaction_name,
            api_key,
            verbose=verbose
        )

        new_func_count = len(deduplicated.get("functions", []))
        removed = original_func_count - new_func_count
        total_removed += removed

        modified_interactors.append(deduplicated)

    # Update data - write to whichever format we read from
    if ctx_json.get("interactors") is not None:
        ctx_json["interactors"] = modified_interactors
        data["ctx_json"] = ctx_json

    # Also update snapshot_json if present (or if that's the only format)
    if "snapshot_json" in data and "interactors" in data["snapshot_json"]:
        # For legacy format, snapshot_json has the full interactor data
        if not ctx_json.get("interactors"):
            data["snapshot_json"]["interactors"] = modified_interactors
        else:
            # For new format, just update function arrays
            snapshot_lookup = {i.get("primary"): i for i in data["snapshot_json"]["interactors"]}
            for mod_int in modified_interactors:
                primary = mod_int.get("primary")
                if primary in snapshot_lookup:
                    snapshot_lookup[primary]["functions"] = mod_int.get("functions", [])

    # Save output
    if output_path is None:
        output_path = json_path

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n{'='*80}")
        print(f"[OK]Deduplication complete!")
        print(f"  Total duplicate functions removed: {total_removed}")
        print(f"  Saved to: {output_path}")
        print(f"{'='*80}\n")


def deduplicate_payload(payload: Dict[str, Any], api_key: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Deduplicate functions in a payload dict (for use in pipeline integration).

    Args:
        payload: JSON payload dict with ctx_json
        api_key: Google API key for Gemini
        verbose: Print progress messages

    Returns:
        Modified payload with duplicates removed
    """
    # Get interactors from ctx_json
    ctx_json = payload.get("ctx_json", {})
    main_protein = ctx_json.get("main", "Unknown")
    interactors = ctx_json.get("interactors", [])

    if not interactors:
        if verbose:
            print("No interactors found - skipping deduplication", file=sys.stderr)
        return payload

    if verbose:
        print(f"\n{'='*80}")
        print(f"AI-Powered Function Deduplication")
        print(f"{'='*80}")
        print(f"Main protein: {main_protein}")
        print(f"Found {len(interactors)} interactors\n")

    # Process each interactor
    modified_interactors = []
    total_removed = 0

    for interactor in interactors:
        primary = interactor.get("primary", "Unknown")
        interaction_name = f"{main_protein} ↔ {primary}"

        original_func_count = len(interactor.get("functions", []))

        deduplicated = deduplicate_interactor_functions(
            interactor,
            interaction_name,
            api_key,
            verbose=verbose
        )

        new_func_count = len(deduplicated.get("functions", []))
        removed = original_func_count - new_func_count
        total_removed += removed

        modified_interactors.append(deduplicated)

    # Update payload
    ctx_json["interactors"] = modified_interactors

    # Also update snapshot_json if present
    if "snapshot_json" in payload and "interactors" in payload["snapshot_json"]:
        snapshot_lookup = {i.get("primary"): i for i in payload["snapshot_json"]["interactors"]}
        for mod_int in modified_interactors:
            primary = mod_int.get("primary")
            if primary in snapshot_lookup:
                snapshot_lookup[primary]["functions"] = mod_int.get("functions", [])

    if verbose:
        print(f"\n[OK]Deduplication complete!")
        print(f"  Total duplicate functions removed: {total_removed}\n")

    return payload


def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage: python deduplicate_functions.py <json_file> [output_file]")
        print("\nExample:")
        print("  python deduplicate_functions.py cache/ATXN3.json")
        print("  python deduplicate_functions.py cache/ATXN3.json cache/ATXN3_dedup.json")
        sys.exit(1)

    json_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment", file=sys.stderr)
        sys.exit(1)

    deduplicate_json_file(json_file, api_key, output_file, verbose=True)


if __name__ == "__main__":
    main()
