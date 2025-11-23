#!/usr/bin/env python3
"""
Complete Pipeline Script
Runs runner_enhanced.py → evidence_validator.py → claim_fact_checker.py → visualizer.py
Outputs: [PROTEIN].json and [PROTEIN].html
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any


def run_command(cmd: list, description: str) -> float:
    """Run a command and handle errors. Returns elapsed time in seconds."""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}\n")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n✓ {description} completed successfully ({elapsed:.1f}s)")
    return elapsed


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load JSON from file."""
    return json.loads(file_path.read_text(encoding='utf-8'))


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save JSON to file."""
    file_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding='utf-8'
    )


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of file content."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_hash_tracking(tracking_file: Path) -> Dict[str, str]:
    """Load hash tracking data from file."""
    if not tracking_file.exists():
        return {}
    try:
        with open(tracking_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_hash_tracking(tracking_file: Path, tracking_data: Dict[str, str]) -> None:
    """Save hash tracking data to file."""
    try:
        with open(tracking_file, 'w', encoding='utf-8') as f:
            json.dump(tracking_data, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Warning: Could not save hash tracking: {e}", file=sys.stderr)


def should_run_pmid_update(json_file: Path, tracking_file: Path) -> bool:
    """
    Check if update_cache_pmids.py should run for this JSON file.
    Returns True if file is new or content has changed since last run.
    """
    if not json_file.exists():
        return False

    # Calculate current hash
    current_hash = calculate_file_hash(json_file)

    # Load tracking data
    tracking_data = load_hash_tracking(tracking_file)

    # Get stored hash for this file
    file_key = json_file.name
    stored_hash = tracking_data.get(file_key)

    # If no stored hash or hash differs, we need to run the update
    if stored_hash is None or stored_hash != current_hash:
        return True

    return False


def mark_pmid_update_complete(json_file: Path, tracking_file: Path) -> None:
    """Mark that update_cache_pmids.py has been run for this JSON file."""
    if not json_file.exists():
        return

    # Calculate hash
    current_hash = calculate_file_hash(json_file)

    # Load and update tracking data
    tracking_data = load_hash_tracking(tracking_file)
    tracking_data[json_file.name] = current_hash

    # Save tracking data
    save_hash_tracking(tracking_file, tracking_data)


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline: runner.py → evidence_validator.py → claim_fact_checker.py → visualizer"
    )
    parser.add_argument(
        "protein",
        nargs="?",  # Make protein optional
        help="Protein name to analyze"
    )
    parser.add_argument(
        "--skip-runner",
        action="store_true",
        help="Skip runner.py and use existing JSON"
    )
    parser.add_argument(
        "--skip-validator",
        action="store_true",
        help="Skip evidence_validator.py"
    )
    parser.add_argument(
        "--skip-factchecker",
        action="store_true",
        help="Skip claim_fact_checker.py"
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization generation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for all steps"
    )
    parser.add_argument(
        "--interactor-rounds",
        type=int,
        help="Number of interactor discovery rounds (default: 3, min: 3, max: 10)"
    )
    parser.add_argument(
        "--function-rounds",
        type=int,
        help="Number of function mapping rounds (default: 3, min: 3, max: 10)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for number of discovery rounds interactively"
    )

    args = parser.parse_args()

    # Interactive mode: prompt for protein name if not provided
    protein = args.protein
    if not protein:
        print(f"\n{'='*80}")
        print("PROTEIN INTERACTION PIPELINE")
        print(f"{'='*80}\n")
        try:
            protein = input("Enter protein name (HGNC symbol): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nCancelled.")
            sys.exit(0)

        if not protein:
            print("Error: Protein name is required")
            sys.exit(1)

    verbose_flag = ["--verbose"] if args.verbose else []

    # Determine number of rounds (interactive or from args)
    num_interactor_rounds = 3  # Default
    num_function_rounds = 3    # Default

    # Always prompt for rounds if no command-line args provided
    if not args.interactor_rounds and not args.function_rounds and not args.skip_runner:
        print(f"\n{'='*80}")
        print("PIPELINE CONFIGURATION")
        print(f"{'='*80}")
        print("\nDefault configuration:")
        print("  - Interactor discovery rounds: 3 (1a, 1b, 1c)")
        print("  - Function mapping rounds: 3 (2a, 2a2, 2a3)")
        print("\nYou can customize the number of rounds for more comprehensive results.")
        print("More rounds = more interactors and functions discovered (but longer runtime)")
        print("\nPress Enter to use defaults, or specify custom values:")
        print(f"{'='*80}\n")

        try:
            interactor_input = input("Number of interactor discovery rounds (3-10, default 3): ").strip()
            if interactor_input:
                num_interactor_rounds = int(interactor_input)
                num_interactor_rounds = max(3, min(10, num_interactor_rounds))

            function_input = input("Number of function mapping rounds (3-10, default 3): ").strip()
            if function_input:
                num_function_rounds = int(function_input)
                num_function_rounds = max(3, min(10, num_function_rounds))
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\nUsing defaults (3 rounds each)")

    # Override with command-line args if provided
    if args.interactor_rounds:
        num_interactor_rounds = max(3, min(10, args.interactor_rounds))
    if args.function_rounds:
        num_function_rounds = max(3, min(10, args.function_rounds))

    # Define file paths
    base_path = Path.cwd()
    runner_output = base_path / f"{protein}_pipeline.json"
    validated_output = base_path / f"{protein}_validated.json"
    factchecked_output = base_path / f"{protein}_factchecked.json"
    final_json = base_path / f"{protein}.json"
    final_html = base_path / f"{protein}.html"

    print(f"\n{'='*80}")
    print(f"COMPLETE PIPELINE FOR: {protein}")
    print(f"{'='*80}")
    print(f"Output files:")
    print(f"  - Final JSON: {final_json}")
    print(f"  - Final HTML: {final_html}")
    print(f"{'='*80}\n")

    # Track timing for each step
    step_times = {}
    overall_start_time = time.time()

    # Step 1: Run runner.py
    current_json = runner_output
    if not args.skip_runner:
        # Show configuration summary
        print(f"{'='*80}")
        print("RUNNING PIPELINE WITH:")
        print(f"{'='*80}")
        print(f"  Protein: {protein}")
        print(f"  Interactor discovery rounds: {num_interactor_rounds}")
        print(f"  Function mapping rounds: {num_function_rounds}")
        print(f"{'='*80}\n")

        cmd = [
            sys.executable,
            "runner.py",
            protein,
            "--output", str(runner_output),
            "--interactor-rounds", str(num_interactor_rounds),
            "--function-rounds", str(num_function_rounds)
        ] + verbose_flag

        step_times['runner'] = run_command(cmd, "1. Running pipeline (runner.py)")
    else:
        if not runner_output.exists():
            print(f"❌ ERROR: {runner_output} not found and --skip-runner was specified")
            sys.exit(1)
        print(f"⏭️  Skipping runner.py, using existing: {runner_output}")

    # Step 2: Run evidence_validator.py
    if not args.skip_validator:
        cmd = [
            sys.executable,
            "evidence_validator.py",
            str(current_json),
            "--output", str(validated_output)
        ] + verbose_flag

        step_times['validator'] = run_command(cmd, "2. Validating evidence (evidence_validator.py)")
        current_json = validated_output
    else:
        print(f"⏭️  Skipping evidence_validator.py")

    # Step 3: Run claim_fact_checker.py
    if not args.skip_factchecker:
        cmd = [
            sys.executable,
            "claim_fact_checker.py",
            str(current_json),
            "--output", str(factchecked_output)
        ] + verbose_flag

        step_times['factchecker'] = run_command(cmd, "3. Fact-checking claims (claim_fact_checker.py)")
        current_json = factchecked_output
    else:
        print(f"⏭️  Skipping claim_fact_checker.py")

    # Step 4: Copy final JSON to [PROTEIN].json
    print(f"\n{'='*80}")
    print(f"STEP: Creating final JSON output")
    print(f"{'='*80}")

    data = load_json(current_json)
    save_json(data, final_json)
    print(f"✓ Saved final JSON: {final_json}")

    # Step 5: Refresh PMIDs using PubMed (ONLY if JSON changed)
    tracking_file = Path(".pmid_cache_state.json")

    if should_run_pmid_update(final_json, tracking_file):
        print(f"\n{'='*80}")
        print(f"JSON content changed - running PMID refresh")
        print(f"{'='*80}")
        cmd = [
            sys.executable,
            "update_cache_pmids.py",
            str(final_json),
        ]
        step_times['pmid_refresh'] = run_command(cmd, "4. Refreshing PMIDs (update_cache_pmids.py)")

        # Mark as complete after successful run
        mark_pmid_update_complete(final_json, tracking_file)
    else:
        print(f"\n{'='*80}")
        print(f"STEP: Skipping PMID refresh (JSON unchanged)")
        print(f"{'='*80}")
        print(f"✓ JSON hash matches previous run - no PMID update needed")
        step_times['pmid_refresh'] = 0.0

    data = load_json(final_json)

    # Step 6: Run visualizer.py
    if not args.skip_viz:
        cmd = [
            sys.executable,
            "visualizer.py",
            str(final_json),
            str(final_html)
        ]

        step_times['visualizer'] = run_command(cmd, "5. Generating visualization (visualizer.py)")
    else:
        print(f"⏭️  Skipping visualization")

    # Calculate total time
    overall_elapsed = time.time() - overall_start_time
    overall_elapsed_min = overall_elapsed / 60

    # Final summary
    print(f"\n{'='*80}")
    print("✓ COMPLETE PIPELINE FINISHED")
    print(f"{'='*80}")

    # Print timing breakdown
    print(f"\nTIMING SUMMARY:")
    print(f"  Total time: {overall_elapsed_min:.1f} minutes ({overall_elapsed:.0f}s)")
    print(f"\n  Step breakdown:")
    if 'runner' in step_times:
        print(f"    1. Runner (pipeline):          {step_times['runner']:.1f}s ({step_times['runner']/60:.1f} min)")
    if 'validator' in step_times:
        print(f"    2. Evidence validator:         {step_times['validator']:.1f}s ({step_times['validator']/60:.1f} min)")
    if 'factchecker' in step_times:
        print(f"    3. Fact checker:               {step_times['factchecker']:.1f}s ({step_times['factchecker']/60:.1f} min)")
    if 'pmid_refresh' in step_times:
        print(f"    4. PubMed refresh:             {step_times['pmid_refresh']:.1f}s ({step_times['pmid_refresh']/60:.1f} min)")
    if 'visualizer' in step_times:
        print(f"    5. Visualizer:                 {step_times['visualizer']:.1f}s ({step_times['visualizer']/60:.1f} min)")

    print(f"\n{'='*80}")
    print("FINAL OUTPUTS")
    print(f"{'='*80}")
    print(f"  📄 JSON: {final_json}")
    if not args.skip_viz:
        print(f"  🌐 HTML: {final_html}")

    print(f"\nIntermediate files:")
    if runner_output.exists():
        print(f"  - {runner_output}")
    if validated_output.exists():
        print(f"  - {validated_output}")
    if factchecked_output.exists():
        print(f"  - {factchecked_output}")

    # Print summary statistics if available
    if 'ctx_json' in data:
        ctx = data['ctx_json']
        interactors = ctx.get('interactors', [])
        total_functions = sum(len(i.get('functions', [])) for i in interactors)
        total_pmids = sum(len(i.get('pmids', [])) for i in interactors)

        print(f"\n{'='*80}")
        print("DATA SUMMARY")
        print(f"{'='*80}")
        print(f"  Interactors found: {len(interactors)}")
        print(f"  Functions mapped: {total_functions}")
        print(f"  Citations collected: {total_pmids}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
