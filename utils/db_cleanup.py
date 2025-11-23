"""
Database JSON Cleanup Module
=============================

Post-validation cleanup for interaction JSON in PostgreSQL database.

Runs AFTER arrow validation to:
1. Remove redundant fields (interaction_effect, interaction_direction, etc.)
2. Deduplicate evidence entries by PMID
3. Remove DELETED functions (marked invalid by fact-checker)
4. Remove derived fields (arrow_notation, pmids arrays)
5. Optionally archive validation metadata

Target: ~57% size reduction while preserving all scientific data.
"""

import json
import sys
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class DatabaseJSONCleaner:
    """
    Comprehensive JSON cleanup for database interactions.

    Usage:
        cleaner = DatabaseJSONCleaner(dry_run=True)
        cleaner.clean_interaction_data(data)
    """

    def __init__(self, dry_run: bool = True, archive_validation: bool = False):
        """
        Initialize cleaner.

        Args:
            dry_run: If True, don't modify database (just report)
            archive_validation: If True, remove validation metadata fields
        """
        self.dry_run = dry_run
        self.archive_validation = archive_validation

        # Statistics tracking
        self.stats = {
            'interactions_processed': 0,
            'functions_removed': 0,          # DELETED functions
            'evidence_deduped': 0,           # Duplicate PMID entries
            'fields_removed': {
                'interaction_evidence': 0,   # Redundant interaction.evidence[]
                'arrow_notation': 0,         # Derived display strings
                'interaction_effect': 0,     # Duplicate of arrow
                'interaction_direction': 0,  # Duplicate of direction
                'pmids_array': 0,            # Redundant with evidence[].pmid
                'validation_meta': 0,        # Validation artifacts
            },
            'bytes_saved': 0,
        }

        # Samples for reporting
        self.samples = []

    def clean_interaction_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a single interaction's JSONB data.

        Args:
            data: Interaction data dict (from interaction.data JSONB column)

        Returns:
            Cleaned data dict
        """
        # Track original size
        original_size = len(json.dumps(data))

        # Deep copy to avoid mutations
        cleaned = deepcopy(data)

        # Operation 1: Remove interaction-level evidence (redundant with function evidence)
        if 'evidence' in cleaned:
            del cleaned['evidence']
            self.stats['fields_removed']['interaction_evidence'] += 1

        # Operation 2: Remove arrow_notation (derived field)
        if 'arrow_notation' in cleaned:
            del cleaned['arrow_notation']
            self.stats['fields_removed']['arrow_notation'] += 1

        # Operation 3: Clean functions
        functions = cleaned.get('functions', [])
        cleaned_functions = []

        for fn in functions:
            # Skip DELETED functions (marked invalid by fact-checker)
            if fn.get('validity') == 'DELETED':
                self.stats['functions_removed'] += 1
                continue

            # Clean individual function
            cleaned_fn = self._clean_function(fn)
            cleaned_functions.append(cleaned_fn)

        cleaned['functions'] = cleaned_functions

        # Track size reduction
        new_size = len(json.dumps(cleaned))
        bytes_saved = original_size - new_size
        self.stats['bytes_saved'] += bytes_saved
        self.stats['interactions_processed'] += 1

        # Save sample for reporting
        if len(self.samples) < 10:
            self.samples.append({
                'protein_pair': f"{data.get('primary', 'UNKNOWN')}",
                'before_size': original_size,
                'after_size': new_size,
                'reduction_percent': round((bytes_saved / original_size) * 100) if original_size > 0 else 0
            })

        return cleaned

    def _clean_function(self, fn: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a single function object.

        Removes:
        - interaction_effect (duplicate of arrow)
        - interaction_direction (duplicate of direction)
        - pmids array (redundant with evidence[].pmid)
        - validation metadata (if archive_validation=True)

        Deduplicates:
        - evidence entries by PMID

        Args:
            fn: Function dict

        Returns:
            Cleaned function dict
        """
        # Remove interaction_effect if it equals arrow (100% redundant)
        if 'interaction_effect' in fn:
            if fn.get('interaction_effect') == fn.get('arrow'):
                del fn['interaction_effect']
                self.stats['fields_removed']['interaction_effect'] += 1

        # Remove interaction_direction if it equals direction (100% redundant)
        if 'interaction_direction' in fn:
            if fn.get('interaction_direction') == fn.get('direction'):
                del fn['interaction_direction']
                self.stats['fields_removed']['interaction_direction'] += 1

        # Remove pmids array (redundant with evidence[].pmid)
        if 'pmids' in fn:
            del fn['pmids']
            self.stats['fields_removed']['pmids_array'] += 1

        # Deduplicate evidence by PMID
        if 'evidence' in fn:
            original_count = len(fn['evidence'])
            fn['evidence'] = self._deduplicate_evidence(fn['evidence'])
            deduped_count = original_count - len(fn['evidence'])
            self.stats['evidence_deduped'] += deduped_count

        # Archive validation metadata (optional)
        if self.archive_validation:
            validation_fields = ['validity', 'validation_note', 'evidence_source']
            for field in validation_fields:
                if field in fn:
                    # Only remove if value indicates successful validation
                    if fn[field] in ['TRUE', 'fact_checker_verified', 'validated']:
                        del fn[field]
                        self.stats['fields_removed']['validation_meta'] += 1

        return fn

    def _deduplicate_evidence(self, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate evidence entries by PMID, keeping richest metadata.

        Strategy:
        - Group by PMID
        - For duplicates, prefer:
          * Longer assay descriptions (more specific)
          * Species with cell line details
        - Merge unique relevant quotes with " | " separator

        Args:
            evidence: List of evidence dicts

        Returns:
            Deduplicated list
        """
        if not evidence:
            return []

        seen_pmids = {}
        no_pmid = []  # Evidence without PMID (keep all)

        for ev in evidence:
            pmid = ev.get('pmid')

            # Keep evidence without PMID
            if not pmid:
                no_pmid.append(ev)
                continue

            if pmid not in seen_pmids:
                # First occurrence of this PMID
                seen_pmids[pmid] = ev
            else:
                # Duplicate PMID - merge metadata
                existing = seen_pmids[pmid]

                # Prefer longer assay description (more specific)
                existing_assay_len = len(existing.get('assay', ''))
                new_assay_len = len(ev.get('assay', ''))
                if new_assay_len > existing_assay_len:
                    existing['assay'] = ev['assay']

                # Prefer species with cell line detail
                existing_species = existing.get('species', '').lower()
                new_species = ev.get('species', '').lower()
                if 'cells' in new_species or 'cell line' in new_species:
                    existing['species'] = ev['species']

                # Merge unique relevant quotes
                existing_quote = existing.get('relevant_quote', '').strip()
                new_quote = ev.get('relevant_quote', '').strip()
                if new_quote and new_quote != existing_quote:
                    if existing_quote:
                        existing['relevant_quote'] = f"{existing_quote} | {new_quote}"
                    else:
                        existing['relevant_quote'] = new_quote

                # Prefer DOI if existing doesn't have one
                if not existing.get('doi') and ev.get('doi'):
                    existing['doi'] = ev['doi']

                # Prefer full author list (more authors = more complete)
                if ev.get('authors') and len(ev.get('authors', '')) > len(existing.get('authors', '')):
                    existing['authors'] = ev['authors']

        # Combine deduplicated PMIDs with no-PMID evidence
        return list(seen_pmids.values()) + no_pmid

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cleanup statistics.

        Returns:
            Statistics dict
        """
        total_fields_removed = sum(self.stats['fields_removed'].values())

        return {
            'interactions_processed': self.stats['interactions_processed'],
            'functions_removed': self.stats['functions_removed'],
            'evidence_deduped': self.stats['evidence_deduped'],
            'total_fields_removed': total_fields_removed,
            'fields_removed_breakdown': self.stats['fields_removed'],
            'bytes_saved': self.stats['bytes_saved'],
            'kb_saved': round(self.stats['bytes_saved'] / 1024, 1),
            'mb_saved': round(self.stats['bytes_saved'] / (1024 * 1024), 2),
        }

    def print_stats(self):
        """Print cleanup statistics to console."""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("CLEANUP SUMMARY")
        print("="*60)
        print(f"Interactions processed: {stats['interactions_processed']}")
        print(f"Functions removed (DELETED): {stats['functions_removed']}")
        print(f"Evidence entries deduped: {stats['evidence_deduped']}")
        print(f"Total fields removed: {stats['total_fields_removed']}")
        print("\nFields removed by type:")
        for field_type, count in stats['fields_removed_breakdown'].items():
            if count > 0:
                print(f"  - {field_type}: {count}")
        print(f"\nBytes saved: {stats['bytes_saved']:,} ({stats['kb_saved']} KB / {stats['mb_saved']} MB)")

        if stats['interactions_processed'] > 0:
            avg_reduction = (stats['bytes_saved'] / stats['interactions_processed']) / 1024
            print(f"Average reduction per interaction: {avg_reduction:.1f} KB")

        print("="*60)

    def save_report(self, filename: str):
        """
        Save detailed cleanup report to JSON file.

        Args:
            filename: Output file path
        """
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'mode': 'dry_run' if self.dry_run else 'live',
            'archive_validation': self.archive_validation,
            'summary': self.get_stats(),
            'samples': self.samples
        }

        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Detailed report saved to: {filename}")


# Validation helper
def validate_cleanup(before_data: Dict[str, Any], after_data: Dict[str, Any]) -> bool:
    """
    Validate that no critical data was lost during cleanup.

    Checks:
    - All non-DELETED functions preserved
    - All PMIDs preserved
    - Arrow data preserved
    - Direction preserved

    Args:
        before_data: Original data
        after_data: Cleaned data

    Returns:
        True if validation passed

    Raises:
        AssertionError if validation fails
    """
    # Check: All non-DELETED functions preserved
    before_fns = before_data.get('functions', [])
    after_fns = after_data.get('functions', [])
    before_valid = [f for f in before_fns if f.get('validity') != 'DELETED']

    assert len(after_fns) == len(before_valid), \
        f"Valid functions lost! Before: {len(before_valid)}, After: {len(after_fns)}"

    # Check: All PMIDs preserved
    before_pmids = set()
    for fn in before_fns:
        for ev in fn.get('evidence', []):
            if ev.get('pmid'):
                before_pmids.add(ev['pmid'])

    after_pmids = set()
    for fn in after_fns:
        for ev in fn.get('evidence', []):
            if ev.get('pmid'):
                after_pmids.add(ev['pmid'])

    assert before_pmids == after_pmids, \
        f"PMIDs lost! Before: {len(before_pmids)}, After: {len(after_pmids)}"

    # Check: Arrow data preserved
    assert before_data.get('arrow') == after_data.get('arrow'), \
        "Interaction arrow changed!"

    # Check: Direction preserved
    assert before_data.get('direction') == after_data.get('direction'), \
        "Interaction direction changed!"

    return True


if __name__ == "__main__":
    """
    Command-line interface for database JSON cleanup.

    Usage:
        # Dry run (preview changes without writing to database)
        python utils/db_cleanup.py --dry-run

        # Apply cleanup to all interactions
        python utils/db_cleanup.py

        # Cleanup specific protein only
        python utils/db_cleanup.py --protein ATXN3

        # Archive validation metadata (remove validation fields)
        python utils/db_cleanup.py --archive-validation

        # Limit number of interactions (for testing)
        python utils/db_cleanup.py --limit 10 --dry-run
    """
    import argparse
    from datetime import datetime
    from pathlib import Path

    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models import db, Protein, Interaction
    from app import app  # Import Flask app for database context

    # Parse arguments
    parser = argparse.ArgumentParser(description="Clean up database JSON (remove redundant fields, deduplicate evidence)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing to database")
    parser.add_argument("--archive-validation", action="store_true", help="Remove validation metadata fields")
    parser.add_argument("--protein", type=str, help="Only cleanup interactions for specific protein")
    parser.add_argument("--limit", type=int, help="Limit number of interactions to process (for testing)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Create cleaner instance
    cleaner = DatabaseJSONCleaner(dry_run=args.dry_run, archive_validation=args.archive_validation)

    mode = "DRY RUN" if args.dry_run else "LIVE"
    print(f"\n{'='*60}")
    print(f"DATABASE JSON CLEANUP ({mode})")
    print(f"{'='*60}")
    if args.archive_validation:
        print(f"⚠️  Archive mode: Will remove validation metadata")
    print()

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
            print("[INFO] No interactions found to clean")
            sys.exit(0)

        print(f"[INFO] Found {len(interactions)} interactions to process\n")

        # Process each interaction
        processed_count = 0
        error_count = 0

        for interaction in interactions:
            try:
                # Get protein symbols for display
                protein_a = Protein.query.get(interaction.protein_a_id)
                protein_b = Protein.query.get(interaction.protein_b_id)

                if not protein_a or not protein_b:
                    print(f"[WARNING] Skipping interaction {interaction.id} (missing protein records)")
                    continue

                # Clean the interaction data
                cleaned_data = cleaner.clean_interaction_data(interaction.data)

                # Write to database if not dry-run
                if not args.dry_run:
                    interaction.data = cleaned_data
                    interaction.updated_at = datetime.utcnow()
                    db.session.commit()

                processed_count += 1

                if args.verbose:
                    print(f"  [{processed_count}/{len(interactions)}] ✓ {protein_a.symbol} ↔ {protein_b.symbol}")

            except Exception as e:
                error_count += 1
                print(f"  [{processed_count + 1}/{len(interactions)}] ✗ Error: {e}")
                if not args.dry_run:
                    db.session.rollback()

        # Print statistics
        cleaner.print_stats()

        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("Logs")
        log_dir.mkdir(exist_ok=True)
        report_file = log_dir / f"cleanup_report_{timestamp}.json"
        cleaner.save_report(str(report_file))

        # Summary
        print(f"\n{'='*60}")
        print(f"CLEANUP COMPLETE")
        print(f"{'='*60}")
        print(f"Interactions processed: {processed_count}")
        print(f"Errors encountered: {error_count}")

        if args.dry_run:
            print(f"\n[DRY RUN] No changes written to database")
            print(f"[DRY RUN] Review report: {report_file}")
            print(f"[DRY RUN] Run without --dry-run to apply changes")
        else:
            print(f"\n[LIVE] Changes written to database")
            print(f"[LIVE] Report saved: {report_file}")

        print(f"{'='*60}\n")
