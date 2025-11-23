"""
Migration script to fix false chain assignments caused by Strategy 3 bug.

This script identifies and corrects interactors that were falsely assigned
upstream_interactor values using the 'first_direct_interactor' strategy,
which arbitrarily assigned the first direct interactor (often Calreticulin)
as the upstream mediator for all proteins lacking chain data.

Usage:
    # File cache mode (legacy):
    python utils/fix_false_chains.py --dry-run  # Preview changes
    python utils/fix_false_chains.py            # Apply changes
    python utils/fix_false_chains.py --backup   # Create backup before changes

    # Database mode (recommended):
    python utils/fix_false_chains.py --database --dry-run  # Preview database changes
    python utils/fix_false_chains.py --database            # Fix database

Note: With database sync integration (db_sync.py), false chains are now
      automatically fixed during query. This script is for cleaning existing
      data that was created before the fix.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import argparse


def find_affected_files(cache_dir: str = 'cache') -> List[Path]:
    """
    Find all JSON files with false chain assignments.

    Returns:
        List of Path objects for affected files
    """
    affected_files = []
    cache_path = Path(cache_dir)

    for json_file in cache_path.rglob('*.json'):
        if json_file.name.endswith('_metadata.json'):
            continue  # Skip metadata files

        try:
            data = json.loads(json_file.read_text(encoding='utf-8'))

            # Check ctx_json for false chains
            ctx_json = data.get('ctx_json', data)  # Handle both formats
            interactors = ctx_json.get('interactors', [])

            has_false_chains = any(
                i.get('_chain_inferred_strategy') == 'first_direct_interactor'
                for i in interactors
            )

            if has_false_chains:
                affected_files.append(json_file)

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"⚠️  Error reading {json_file}: {e}")
            continue

    return affected_files


def analyze_file(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a single file and return statistics about false chains.

    Returns:
        Dict with analysis results
    """
    data = json.loads(file_path.read_text(encoding='utf-8'))
    ctx_json = data.get('ctx_json', data)
    main_protein = ctx_json.get('main', 'UNKNOWN')
    interactors = ctx_json.get('interactors', [])

    false_chains = [
        i for i in interactors
        if i.get('_chain_inferred_strategy') == 'first_direct_interactor'
    ]

    # Group by false upstream assignment
    false_upstreams = {}
    for interactor in false_chains:
        upstream = interactor.get('upstream_interactor', 'UNKNOWN')
        if upstream not in false_upstreams:
            false_upstreams[upstream] = []
        false_upstreams[upstream].append(interactor.get('primary', 'UNKNOWN'))

    return {
        'file': file_path,
        'main_protein': main_protein,
        'total_interactors': len(interactors),
        'false_chain_count': len(false_chains),
        'false_upstreams': false_upstreams
    }


def fix_file(file_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    """
    Fix false chain assignments in a single file.

    Returns:
        Dict with fix statistics
    """
    data = json.loads(file_path.read_text(encoding='utf-8'))
    ctx_json = data.get('ctx_json', data)
    interactors = ctx_json.get('interactors', [])

    fixed_count = 0
    fixed_proteins = []

    for interactor in interactors:
        if interactor.get('_chain_inferred_strategy') == 'first_direct_interactor':
            protein_name = interactor.get('primary', 'UNKNOWN')
            false_upstream = interactor.get('upstream_interactor')

            # Clear false chain data
            interactor['upstream_interactor'] = None
            interactor['mediator_chain'] = []
            interactor['_chain_inference_corrected'] = True
            interactor['_correction_timestamp'] = datetime.now().isoformat()
            interactor['_false_upstream_removed'] = false_upstream

            # Add note about missing chain
            interactor['_chain_missing'] = True
            interactor['_inference_failed'] = 'no_biological_hints'

            fixed_count += 1
            fixed_proteins.append(protein_name)

    if not dry_run and fixed_count > 0:
        # Write updated data back to file
        file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

    return {
        'file': file_path.name,
        'fixed_count': fixed_count,
        'fixed_proteins': fixed_proteins
    }


def create_backup(cache_dir: str = 'cache') -> Path:
    """
    Create a backup of the cache directory.

    Returns:
        Path to backup directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = Path(f'{cache_dir}_backup_{timestamp}')

    print(f"📦 Creating backup: {backup_dir}")
    shutil.copytree(cache_dir, backup_dir)
    print(f"✅ Backup created successfully")

    return backup_dir


def fix_database_chains(dry_run: bool = False) -> Dict[str, Any]:
    """
    Fix false chain assignments in PostgreSQL database.

    Args:
        dry_run: If True, only preview changes without applying

    Returns:
        Dict with fix statistics
    """
    import os
    import sys

    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from models import Protein, Interaction, db
    from app import app

    print("\n🔍 Scanning database for false chain assignments...")

    stats = {
        'proteins_scanned': 0,
        'proteins_affected': 0,
        'interactions_fixed': 0,
        'false_upstreams': {}
    }

    with app.app_context():
        # Query all interactions from database
        all_interactions = Interaction.query.all()
        stats['proteins_scanned'] = Protein.query.count()

        print(f"   Found {len(all_interactions)} interactions to check")

        for interaction in all_interactions:
            # Get interaction data (JSONB column)
            data = interaction.data

            if not isinstance(data, dict):
                continue

            # Check for false chain marker
            if data.get('_chain_inferred_strategy') == 'first_direct_interactor':
                protein_a = Protein.query.get(interaction.protein_a_id)
                protein_b = Protein.query.get(interaction.protein_b_id)

                partner_symbol = data.get('primary', 'UNKNOWN')
                false_upstream = data.get('upstream_interactor')

                print(f"   ❌ Found false chain: {protein_a.symbol if protein_a else '?'} → {partner_symbol} (false upstream: {false_upstream})")

                # Track statistics
                stats['interactions_fixed'] += 1
                if false_upstream not in stats['false_upstreams']:
                    stats['false_upstreams'][false_upstream] = []
                stats['false_upstreams'][false_upstream].append(partner_symbol)

                if not dry_run:
                    # Fix the chain data
                    data['upstream_interactor'] = None
                    data['mediator_chain'] = []
                    data['_chain_inference_corrected'] = True
                    data['_correction_timestamp'] = datetime.now().isoformat()
                    data['_false_upstream_removed'] = false_upstream
                    data['_chain_missing'] = True
                    data['_inference_failed'] = 'no_biological_hints'

                    # Remove problematic marker
                    if '_chain_inferred_strategy' in data:
                        del data['_chain_inferred_strategy']

                    # Update database (flag as modified for SQLAlchemy)
                    from sqlalchemy.orm.attributes import flag_modified
                    flag_modified(interaction, 'data')

        if not dry_run and stats['interactions_fixed'] > 0:
            db.session.commit()
            print(f"   ✅ Committed {stats['interactions_fixed']} fixes to database")

    # Count affected proteins
    stats['proteins_affected'] = len(set(
        protein
        for proteins in stats['false_upstreams'].values()
        for protein in proteins
    ))

    return stats


def main():
    parser = argparse.ArgumentParser(description='Fix false chain assignments in cache files or database')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying them')
    parser.add_argument('--backup', action='store_true', help='Create backup before applying changes')
    parser.add_argument('--cache-dir', default='cache', help='Path to cache directory (default: cache)')
    parser.add_argument('--database', action='store_true', help='Fix database instead of file cache')
    args = parser.parse_args()

    # DATABASE MODE
    if args.database:
        print("🗄️  DATABASE MODE: Scanning PostgreSQL for false chain assignments")
        print("=" * 80)

        stats = fix_database_chains(dry_run=args.dry_run)

        print("\n" + "=" * 80)
        print("📊 DATABASE SUMMARY:")
        print(f"   Proteins scanned: {stats['proteins_scanned']}")
        print(f"   Proteins affected: {stats['proteins_affected']}")
        print(f"   Interactions fixed: {stats['interactions_fixed']}")

        if stats['false_upstreams']:
            print(f"   Most common false upstreams:")
            for upstream, proteins in sorted(stats['false_upstreams'].items(), key=lambda x: len(x[1]), reverse=True)[:5]:
                print(f"      - {upstream}: {len(proteins)} false assignments")

        if args.dry_run:
            print("\n🔍 DRY RUN - No changes applied to database")
            print("   Run without --dry-run to apply fixes")
        else:
            print(f"\n✨ Database migration complete!")
            print(f"   Total interactions corrected: {stats['interactions_fixed']}")

        return

    # FILE CACHE MODE (original logic)
    cache_dir = args.cache_dir

    print("🔍 Scanning cache directory for false chain assignments...")
    affected_files = find_affected_files(cache_dir)

    if not affected_files:
        print("✅ No false chain assignments found!")
        return

    print(f"\n📊 Found {len(affected_files)} affected files")
    print("=" * 80)

    # Analyze each file
    total_false_chains = 0
    false_upstream_summary = {}

    for file_path in affected_files:
        analysis = analyze_file(file_path)
        total_false_chains += analysis['false_chain_count']

        print(f"\n📄 {analysis['main_protein']} ({file_path.name})")
        print(f"   Total interactors: {analysis['total_interactors']}")
        print(f"   False chains: {analysis['false_chain_count']}")

        for false_upstream, proteins in analysis['false_upstreams'].items():
            print(f"   ❌ False upstream '{false_upstream}' → {len(proteins)} proteins:")
            for protein in proteins[:5]:  # Show first 5
                print(f"      - {protein}")
            if len(proteins) > 5:
                print(f"      ... and {len(proteins) - 5} more")

            # Track for summary
            if false_upstream not in false_upstream_summary:
                false_upstream_summary[false_upstream] = 0
            false_upstream_summary[false_upstream] += len(proteins)

    print("\n" + "=" * 80)
    print(f"📊 SUMMARY:")
    print(f"   Total files affected: {len(affected_files)}")
    print(f"   Total false chains: {total_false_chains}")
    print(f"   Most common false upstreams:")
    for upstream, count in sorted(false_upstream_summary.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      - {upstream}: {count} false assignments")

    if args.dry_run:
        print("\n🔍 DRY RUN - No changes applied")
        print("   Run without --dry-run to apply fixes")
        return

    # Confirm before proceeding
    print("\n⚠️  This will modify the cache files listed above.")
    if not args.backup:
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted")
            return

    # Create backup if requested
    if args.backup:
        create_backup(cache_dir)

    # Apply fixes
    print("\n🔧 Applying fixes...")
    total_fixed = 0

    for file_path in affected_files:
        result = fix_file(file_path, dry_run=False)
        total_fixed += result['fixed_count']
        print(f"   ✅ {result['file']}: Fixed {result['fixed_count']} false chains")

    print(f"\n✨ Migration complete!")
    print(f"   Total false chains corrected: {total_fixed}")
    print(f"   Files updated: {len(affected_files)}")

    if args.backup:
        print(f"\n💾 Backup available at: {cache_dir}_backup_*")


if __name__ == '__main__':
    main()
