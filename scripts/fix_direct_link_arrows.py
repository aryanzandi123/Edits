#!/usr/bin/env python3
"""
Database migration script to fix arrow fields for direct mediator links.

This script identifies and corrects interactions where the interaction-level arrow field
doesn't match the direct_arrow specified in arrow_context within the functions.

Specifically addresses cases like RHEB→MTOR where:
- interaction.arrow = "inhibits" (WRONG - from net effect)
- arrow_context.direct_arrow = "activates" (CORRECT - actual direct interaction)
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, db
from models import Protein, Interaction


def find_mismatched_arrows(dry_run=False):
    """
    Find all direct mediator links where interaction.arrow doesn't match arrow_context.direct_arrow

    Args:
        dry_run: If True, only report issues without making changes

    Returns:
        List of (interaction, correct_arrow) tuples
    """
    mismatches = []

    with app.app_context():
        # Query all direct interactions
        direct_interactions = db.session.query(Interaction).filter(
            Interaction.function_context == 'direct'
        ).all()

        print(f"\n{'='*70}")
        print(f"SCANNING FOR ARROW MISMATCHES IN DIRECT MEDIATOR LINKS")
        print(f"{'='*70}")
        print(f"Found {len(direct_interactions)} interactions with function_context='direct'\n")

        for interaction in direct_interactions:
            # Get protein symbols for logging
            protein_a = db.session.get(Protein, interaction.protein_a_id)
            protein_b = db.session.get(Protein, interaction.protein_b_id)

            if not protein_a or not protein_b:
                continue

            symbol_a = protein_a.symbol
            symbol_b = protein_b.symbol

            # Check if functions have arrow_context with direct_arrow
            data = interaction.data or {}
            functions = data.get("functions", [])

            direct_arrow_found = None
            for func in functions:
                arrow_context = func.get("arrow_context", {})
                if arrow_context and arrow_context.get("direct_arrow"):
                    direct_arrow_found = arrow_context["direct_arrow"]
                    break

            if direct_arrow_found:
                # Compare with interaction-level arrow
                current_arrow = interaction.arrow

                if current_arrow != direct_arrow_found:
                    print(f"[MISMATCH] {symbol_a} ↔ {symbol_b} (ID: {interaction.id})")
                    print(f"  Current arrow:  {current_arrow}")
                    print(f"  Correct arrow:  {direct_arrow_found}")
                    print(f"  Function:       {functions[0].get('function', 'N/A')}")
                    print(f"  Discovery:      {interaction.discovered_in_query}")

                    mismatches.append((interaction, direct_arrow_found))
                    print()

        print(f"{'='*70}")
        print(f"SCAN COMPLETE: Found {len(mismatches)} mismatches")
        print(f"{'='*70}\n")

    return mismatches


def fix_mismatched_arrows(mismatches, dry_run=True):
    """
    Apply corrections to interactions with mismatched arrows

    Args:
        mismatches: List of (interaction, correct_arrow) tuples
        dry_run: If True, only show what would be changed

    Returns:
        Number of interactions updated
    """
    if not mismatches:
        print("No mismatches to fix.\n")
        return 0

    updated_count = 0

    with app.app_context():
        print(f"\n{'='*70}")
        print(f"{'DRY RUN' if dry_run else 'APPLYING FIXES'}: Correcting arrow fields")
        print(f"{'='*70}\n")

        for interaction, correct_arrow in mismatches:
            protein_a = db.session.get(Protein, interaction.protein_a_id)
            protein_b = db.session.get(Protein, interaction.protein_b_id)

            if not protein_a or not protein_b:
                continue

            symbol_a = protein_a.symbol
            symbol_b = protein_b.symbol

            if dry_run:
                print(f"[WOULD UPDATE] {symbol_a} ↔ {symbol_b} (ID: {interaction.id})")
                print(f"  {interaction.arrow} → {correct_arrow}")
            else:
                print(f"[UPDATING] {symbol_a} ↔ {symbol_b} (ID: {interaction.id})")
                print(f"  {interaction.arrow} → {correct_arrow}")

                try:
                    # Update the arrow field
                    interaction.arrow = correct_arrow
                    interaction.updated_at = datetime.utcnow()

                    db.session.commit()
                    updated_count += 1
                    print(f"  ✓ Updated successfully")

                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    db.session.rollback()

            print()

        print(f"{'='*70}")
        if dry_run:
            print(f"DRY RUN COMPLETE: Would update {len(mismatches)} interactions")
            print(f"Run with --apply to make actual changes")
        else:
            print(f"FIXES APPLIED: Updated {updated_count}/{len(mismatches)} interactions")
        print(f"{'='*70}\n")

    return updated_count


def generate_report(mismatches, output_file=None):
    """
    Generate a JSON report of all mismatches found

    Args:
        mismatches: List of (interaction, correct_arrow) tuples
        output_file: Optional path to write JSON report
    """
    report = {
        "scan_timestamp": datetime.utcnow().isoformat(),
        "total_mismatches": len(mismatches),
        "mismatches": []
    }

    with app.app_context():
        for interaction, correct_arrow in mismatches:
            protein_a = db.session.get(Protein, interaction.protein_a_id)
            protein_b = db.session.get(Protein, interaction.protein_b_id)

            if not protein_a or not protein_b:
                continue

            report["mismatches"].append({
                "interaction_id": interaction.id,
                "protein_a": protein_a.symbol,
                "protein_b": protein_b.symbol,
                "current_arrow": interaction.arrow,
                "correct_arrow": correct_arrow,
                "discovered_in_query": interaction.discovered_in_query,
                "discovery_method": interaction.discovery_method,
                "function_context": interaction.function_context
            })

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report written to: {output_file}\n")

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix arrow fields for direct mediator links in the database"
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Apply fixes to database (default is dry-run)'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Path to save JSON report of mismatches'
    )

    args = parser.parse_args()
    dry_run = not args.apply

    # Find mismatches
    mismatches = find_mismatched_arrows(dry_run=dry_run)

    # Generate report if requested
    if args.report:
        generate_report(mismatches, args.report)

    # Fix mismatches
    if mismatches:
        fix_mismatched_arrows(mismatches, dry_run=dry_run)

    # Exit with appropriate code
    sys.exit(0 if not mismatches or args.apply else 1)


if __name__ == "__main__":
    main()
