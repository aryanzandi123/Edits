#!/usr/bin/env python3
"""
Protein Interaction Database Layer

Manages a protein-centric database where interactions are stored symmetrically
and can be queried across all previous queries. This enables knowledge building
and eliminates redundant searches.

Directory Structure:
    cache/
        proteins/
            ATXN3/
                metadata.json
                interactions/
                    VCP.json
                    HDAC6.json
            VCP/
                metadata.json
                interactions/
                    ATXN3.json  (symmetric copy)
                    UFD1L.json
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import shutil


# Cache directory configuration
CACHE_DIR = Path("cache")
PROTEINS_DIR = CACHE_DIR / "proteins"
OLD_CACHE_DIR = CACHE_DIR  # For backward compatibility


def _ensure_protein_dir(protein: str) -> Path:
    """Ensure protein directory and interactions subdirectory exist."""
    protein_dir = PROTEINS_DIR / protein
    interactions_dir = protein_dir / "interactions"
    protein_dir.mkdir(parents=True, exist_ok=True)
    interactions_dir.mkdir(parents=True, exist_ok=True)
    return protein_dir


def _interaction_file_path(protein_a: str, protein_b: str) -> Path:
    """Get path to interaction file."""
    return PROTEINS_DIR / protein_a / "interactions" / f"{protein_b}.json"


def _metadata_file_path(protein: str) -> Path:
    """Get path to protein metadata file."""
    return PROTEINS_DIR / protein / "metadata.json"


def _load_json_safe(file_path: Path) -> Optional[Dict[str, Any]]:
    """Safely load JSON file, return None if not found or invalid."""
    try:
        if not file_path.exists():
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return None


def _save_json_safe(file_path: Path, data: Dict[str, Any]) -> bool:
    """Safely save JSON file, return True on success."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except IOError as e:
        print(f"Error: Failed to save {file_path}: {e}")
        return False


def get_all_interactions(protein: str) -> List[Dict[str, Any]]:
    """
    Get ALL known interactions for a protein from the database.

    This scans:
    1. proteins/{protein}/interactions/*.json (direct interactions)
    2. proteins/*/interactions/{protein}.json (reverse interactions)

    Returns a list of interaction dictionaries with 'primary' field.

    Args:
        protein: Protein symbol (e.g., "ATXN3")

    Returns:
        List of interaction dicts in snapshot_json interactor format
    """
    interactions = []
    seen_partners: Set[str] = set()

    # Method 1: Direct interactions (proteins/{protein}/interactions/*.json)
    protein_dir = PROTEINS_DIR / protein
    interactions_dir = protein_dir / "interactions"

    if interactions_dir.exists():
        for interaction_file in interactions_dir.glob("*.json"):
            interaction_data = _load_json_safe(interaction_file)
            if interaction_data:
                partner = interaction_file.stem  # Filename without .json
                if partner not in seen_partners:
                    interactions.append(interaction_data)
                    seen_partners.add(partner)

    # Method 2: Reverse interactions (proteins/*/interactions/{protein}.json)
    # This finds interactions discovered when other proteins were queried
    if PROTEINS_DIR.exists():
        for other_protein_dir in PROTEINS_DIR.iterdir():
            if not other_protein_dir.is_dir():
                continue

            other_protein = other_protein_dir.name
            if other_protein == protein:
                continue  # Skip self

            reverse_interaction_file = other_protein_dir / "interactions" / f"{protein}.json"
            if reverse_interaction_file.exists() and other_protein not in seen_partners:
                interaction_data = _load_json_safe(reverse_interaction_file)
                if interaction_data:
                    # Flip the perspective for consistency
                    flipped_data = _flip_interaction_perspective(interaction_data, protein)
                    interactions.append(flipped_data)
                    seen_partners.add(other_protein)

    return interactions


def _flip_interaction_perspective(
    interaction_data: Dict[str, Any],
    new_main: str
) -> Dict[str, Any]:
    """
    Flip interaction from protein_a→protein_b to protein_b→protein_a perspective.

    This reverses direction and swaps protein_a/protein_b while keeping
    the core interaction data intact.
    """
    flipped = interaction_data.copy()

    # Swap protein_a and protein_b
    protein_a = interaction_data.get("protein_a")
    protein_b = interaction_data.get("protein_b")

    if protein_a and protein_b:
        flipped["protein_a"] = protein_b
        flipped["protein_b"] = protein_a

    # Update 'primary' field to reflect the new partner
    if protein_a == new_main:
        flipped["primary"] = protein_b
    else:
        flipped["primary"] = protein_a

    # Reverse direction
    direction = interaction_data.get("direction", "")
    if direction == "main_to_primary":
        flipped["direction"] = "primary_to_main"
    elif direction == "primary_to_main":
        flipped["direction"] = "main_to_primary"
    # bidirectional stays bidirectional

    return flipped


def save_interaction(
    protein_a: str,
    protein_b: str,
    interaction_data: Dict[str, Any]
) -> bool:
    """
    Save an interaction symmetrically in the database.

    Writes:
    - proteins/{protein_a}/interactions/{protein_b}.json
    - proteins/{protein_b}/interactions/{protein_a}.json (symmetric copy)

    Args:
        protein_a: First protein (usually the queried protein)
        protein_b: Second protein (interactor)
        interaction_data: Full interaction dict from snapshot_json

    Returns:
        True if both saves succeeded
    """
    # Ensure both protein directories exist
    _ensure_protein_dir(protein_a)
    _ensure_protein_dir(protein_b)

    # Enrich interaction data with database metadata
    enriched_data = interaction_data.copy()
    enriched_data["protein_a"] = protein_a
    enriched_data["protein_b"] = protein_b

    # Ensure 'primary' field is set correctly
    if "primary" not in enriched_data:
        enriched_data["primary"] = protein_b

    # Add discovery metadata if not present
    now = datetime.utcnow().isoformat() + "Z"
    if "discovered_in_query" not in enriched_data:
        enriched_data["discovered_in_query"] = protein_a
    if "first_discovered" not in enriched_data:
        enriched_data["first_discovered"] = now
    enriched_data["last_updated"] = now

    # Save from protein_a's perspective
    file_a = _interaction_file_path(protein_a, protein_b)
    success_a = _save_json_safe(file_a, enriched_data)

    # Create symmetric copy from protein_b's perspective
    symmetric_data = _flip_interaction_perspective(enriched_data, protein_b)
    file_b = _interaction_file_path(protein_b, protein_a)
    success_b = _save_json_safe(file_b, symmetric_data)

    return success_a and success_b


def build_query_snapshot(protein: str) -> Dict[str, Any]:
    """
    Build a snapshot_json compatible view for a protein.

    This generates the format expected by visualizer and API endpoints:
    {
        "snapshot_json": {
            "main": "PROTEIN",
            "interactors": [...]
        }
    }

    Args:
        protein: Protein symbol to build snapshot for

    Returns:
        Dict with snapshot_json structure
    """
    interactions = get_all_interactions(protein)

    # Build interactors list in snapshot format
    interactors = []
    for interaction in interactions:
        # Remove database-specific fields
        interactor_data = {
            k: v for k, v in interaction.items()
            if k not in ["protein_a", "protein_b", "discovered_in_query",
                        "first_discovered", "last_updated"]
        }
        interactors.append(interactor_data)

    return {
        "snapshot_json": {
            "main": protein,
            "interactors": interactors
        }
    }


def update_protein_metadata(protein: str, query_completed: bool = True) -> bool:
    """
    Update protein metadata after a query.

    Args:
        protein: Protein symbol
        query_completed: If True, increment query count

    Returns:
        True on success
    """
    metadata_path = _metadata_file_path(protein)
    metadata = _load_json_safe(metadata_path) or {}

    now = datetime.utcnow().isoformat() + "Z"

    # Initialize or update metadata
    if not metadata:
        metadata = {
            "protein": protein,
            "first_queried": now,
            "last_queried": now,
            "query_count": 1,
            "total_interactions": 0,
            "interaction_partners": []
        }
    else:
        if query_completed:
            metadata["query_count"] = metadata.get("query_count", 0) + 1
        metadata["last_queried"] = now

    # Update interaction count and partners
    interactions = get_all_interactions(protein)
    metadata["total_interactions"] = len(interactions)
    metadata["interaction_partners"] = [
        i.get("primary") for i in interactions if i.get("primary")
    ]

    return _save_json_safe(metadata_path, metadata)


def get_protein_metadata(protein: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a protein."""
    return _load_json_safe(_metadata_file_path(protein))


def list_all_proteins() -> List[str]:
    """List all proteins in the database."""
    if not PROTEINS_DIR.exists():
        return []

    return [
        p.name for p in PROTEINS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ]


def delete_protein(protein: str) -> bool:
    """
    Delete a protein and all its interactions from the database.

    WARNING: This also removes symmetric interactions from partner proteins.

    Args:
        protein: Protein to delete

    Returns:
        True on success
    """
    protein_dir = PROTEINS_DIR / protein

    if not protein_dir.exists():
        return False

    # First, remove symmetric interactions from partner proteins
    interactions_dir = protein_dir / "interactions"
    if interactions_dir.exists():
        for interaction_file in interactions_dir.glob("*.json"):
            partner = interaction_file.stem
            symmetric_file = _interaction_file_path(partner, protein)
            if symmetric_file.exists():
                symmetric_file.unlink()

    # Remove the protein directory
    try:
        shutil.rmtree(protein_dir)
        return True
    except Exception as e:
        print(f"Error deleting protein {protein}: {e}")
        return False


def database_exists() -> bool:
    """Check if the new database structure exists."""
    return PROTEINS_DIR.exists() and any(PROTEINS_DIR.iterdir())


def get_database_stats() -> Dict[str, Any]:
    """Get statistics about the database."""
    proteins = list_all_proteins()
    total_interactions = 0

    for protein in proteins:
        interactions_dir = PROTEINS_DIR / protein / "interactions"
        if interactions_dir.exists():
            total_interactions += len(list(interactions_dir.glob("*.json")))

    return {
        "total_proteins": len(proteins),
        "total_interaction_files": total_interactions,
        "unique_interactions": total_interactions // 2,  # Each is stored twice
        "proteins": proteins
    }


# ============================================================================
# OLD CACHE COMPATIBILITY LAYER
# ============================================================================

def load_from_old_cache(protein: str) -> Optional[Dict[str, Any]]:
    """
    Load data from old cache format for backward compatibility.

    Old format: cache/{PROTEIN}.json with snapshot_json inside.

    Returns:
        Dict with snapshot_json and ctx_json (if metadata file exists)
    """
    old_cache_file = OLD_CACHE_DIR / f"{protein}.json"
    old_metadata_file = OLD_CACHE_DIR / f"{protein}_metadata.json"

    if not old_cache_file.exists():
        return None

    snapshot_data = _load_json_safe(old_cache_file)
    if not snapshot_data:
        return None

    result = {
        "snapshot_json": snapshot_data.get("snapshot_json", {})
    }

    # Try to load metadata (ctx_json)
    if old_metadata_file.exists():
        metadata = _load_json_safe(old_metadata_file)
        if metadata:
            result["ctx_json"] = metadata.get("ctx_json", {})

    return result


def save_to_old_cache(protein: str, snapshot_data: Dict[str, Any]) -> bool:
    """
    Save to old cache format for backward compatibility during transition.

    Args:
        protein: Protein symbol
        snapshot_data: Dict with 'snapshot_json' key

    Returns:
        True on success
    """
    old_cache_file = OLD_CACHE_DIR / f"{protein}.json"
    return _save_json_safe(old_cache_file, snapshot_data)


if __name__ == "__main__":
    # Simple test/demo
    print("Protein Interaction Database")
    print("=" * 80)

    if database_exists():
        stats = get_database_stats()
        print(f"Database found!")
        print(f"  Proteins: {stats['total_proteins']}")
        print(f"  Unique interactions: {stats['unique_interactions']}")
        print(f"  Interaction files: {stats['total_interaction_files']}")
        print(f"\nProteins in database:")
        for protein in stats['proteins']:
            meta = get_protein_metadata(protein)
            if meta:
                print(f"  - {protein}: {meta.get('total_interactions', 0)} interactions")
    else:
        print("No database found. Run migrate_cache.py to create it.")
