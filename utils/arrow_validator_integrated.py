#!/usr/bin/env python3
"""
Integrated arrow validator for automatic pipeline use.

This module provides direct mediator link extraction that works on in-memory JSON
without requiring database access. Used by both:
- runner.py (automatic pipeline)
- scripts/validate_existing_arrows.py (manual validation)
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def extract_direct_mediator_links_from_json(
    payload: Dict[str, Any],
    api_key: str = None,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Extract direct mediator links from indirect interactions in pipeline JSON.

    For chains like ATXN3→RHEB→MTOR, extracts RHEB→MTOR as a direct link.
    Uses 3-tier strategy:
    - Tier 1: Skip (requires database - manual script only)
    - Tier 2: Query pipeline for direct pair
    - Tier 3: Extract from chain evidence

    Args:
        payload: Pipeline JSON result with snapshot_json and ctx_json
        api_key: Google API key for Tier 2 pipeline queries
        verbose: Enable detailed logging

    Returns:
        List of direct link interaction dicts ready to merge into payload
    """
    snapshot = payload.get('snapshot_json', {})
    interactors = snapshot.get('interactors', [])

    if not interactors:
        if verbose:
            print("[DIRECT LINK EXTRACTION] No interactors to process")
        return []

    direct_links = []
    processed_pairs = set()  # Track to avoid duplicates

    for interactor_data in interactors:
        # Only process indirect interactions
        interaction_type = interactor_data.get('interaction_type', 'direct')
        if interaction_type != 'indirect':
            continue

        # Extract chain information
        primary = interactor_data.get('primary')
        upstream_interactor = interactor_data.get('upstream_interactor')
        mediator_chain = interactor_data.get('mediator_chain', [])

        # Determine mediator (last protein before target)
        mediator = upstream_interactor or (mediator_chain[-1] if mediator_chain else None)

        if not mediator or not primary:
            continue

        # Create normalized pair key to avoid duplicates
        pair_key = tuple(sorted([mediator, primary]))
        if pair_key in processed_pairs:
            continue

        processed_pairs.add(pair_key)

        if verbose:
            print(f"[DIRECT LINK] Processing: {mediator} → {primary}")

        # ========================================
        # TIER 2: Query pipeline for direct pair
        # ========================================
        if api_key:
            direct_link = query_direct_pair_simple(
                mediator,
                primary,
                api_key,
                verbose=verbose
            )

            if direct_link:
                if verbose:
                    print(f"  ✓ [TIER 2] Found direct interaction via pipeline")
                direct_links.append(direct_link)
                continue

        # ========================================
        # TIER 3: Extract from chain evidence
        # ========================================
        if verbose:
            print(f"  → [TIER 3] Extracting from chain evidence")

        direct_link = extract_from_chain_evidence(
            mediator,
            primary,
            interactor_data
        )

        if direct_link:
            if verbose:
                print(f"  ✓ [TIER 3] Extracted from chain evidence")
            direct_links.append(direct_link)
        else:
            if verbose:
                print(f"  ✗ [TIER 3] No evidence found")

    if verbose:
        print(f"\n[DIRECT LINK EXTRACTION] Extracted {len(direct_links)} direct mediator links")

    return direct_links


def query_direct_pair_simple(
    protein_a: str,
    protein_b: str,
    api_key: str,
    verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Query the pipeline for a direct protein-protein interaction.

    Simplified version that returns a ready-to-merge interaction dict.

    Args:
        protein_a: First protein symbol
        protein_b: Second protein symbol
        api_key: Google API key
        verbose: Enable logging

    Returns:
        Interaction dict or None
    """
    try:
        # Import runner to use its pipeline
        from runner import run_pipeline_for_protein

        if verbose:
            print(f"    [TIER 2] Querying pipeline for {protein_a}...")

        # Query pipeline with minimal config (1 round only, no functions)
        result = run_pipeline_for_protein(
            protein_symbol=protein_a,
            max_interactor_rounds=1,
            max_function_rounds=0,
            api_key=api_key,
            verbose=False
        )

        if not result or 'snapshot_json' not in result:
            return None

        # Find interaction with protein_b
        interactors = result['snapshot_json'].get('interactors', [])

        for interactor in interactors:
            if interactor.get('primary') == protein_b:
                # Found the direct interaction!
                # Mark it as extracted and direct context
                interactor['function_context'] = 'direct'
                interactor['_inferred_from_chain'] = True
                interactor['_evidence_tier'] = 2

                if verbose:
                    functions_count = len(interactor.get('functions', []))
                    print(f"    [TIER 2] ✓ Found with {functions_count} function(s)")

                return interactor

        if verbose:
            print(f"    [TIER 2] ✗ {protein_b} not found in {protein_a} interactors")

        return None

    except Exception as e:
        if verbose:
            print(f"    [TIER 2] ✗ Pipeline error: {e}")
        return None


def extract_from_chain_evidence(
    mediator: str,
    target: str,
    chain_interaction: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Extract direct mediator link from chain interaction evidence.

    This is the Tier 3 fallback when pipeline query fails.

    Args:
        mediator: Mediator protein symbol
        target: Target protein symbol
        chain_interaction: Full chain interaction data

    Returns:
        Direct link interaction dict or None
    """
    functions = chain_interaction.get('functions', [])

    if not functions:
        return None

    # Try to extract evidence that mentions both proteins
    extracted_functions = []

    for func in functions:
        # Check if function evidence mentions both mediator and target
        evidence = func.get('evidence', [])
        relevant_evidence = []

        for paper in evidence:
            quote = paper.get('relevant_quote', '').lower()
            title = paper.get('paper_title', '').lower()

            # Check if both proteins are mentioned
            if (mediator.lower() in quote or mediator.lower() in title) and \
               (target.lower() in quote or target.lower() in title):
                relevant_evidence.append(paper)

        if relevant_evidence:
            # Create function entry for direct link
            extracted_func = {
                'function': func.get('function'),
                'arrow': func.get('arrow', 'regulates'),
                'cellular_process': func.get('cellular_process'),
                'effect_description': func.get('effect_description'),
                'evidence': relevant_evidence,
                'pmids': [e.get('pmid') for e in relevant_evidence if e.get('pmid')],
                'confidence': func.get('confidence', 0.5)
            }
            extracted_functions.append(extracted_func)

    if not extracted_functions:
        return None

    # Build direct link interaction
    direct_link = {
        'primary': target,
        'direction': chain_interaction.get('direction', 'bidirectional'),
        'arrow': extracted_functions[0].get('arrow', 'regulates'),
        'confidence': chain_interaction.get('confidence', 0.5),
        'intent': chain_interaction.get('intent', 'binding'),
        'functions': extracted_functions,
        'evidence': extracted_functions[0].get('evidence', []),
        'pmids': extracted_functions[0].get('pmids', []),
        'function_context': 'direct',
        '_inferred_from_chain': True,
        '_evidence_tier': 3,
        '_original_chain': f"{chain_interaction.get('main', 'Unknown')}→{mediator}→{target}"
    }

    return direct_link


def merge_direct_links_into_payload(
    payload: Dict[str, Any],
    direct_links: List[Dict[str, Any]],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Merge extracted direct links into the payload.

    Args:
        payload: Original pipeline payload
        direct_links: Extracted direct link interactions
        verbose: Enable logging

    Returns:
        Updated payload with merged links
    """
    if not direct_links:
        return payload

    snapshot = payload.get('snapshot_json', {})
    interactors = snapshot.get('interactors', [])

    # Track existing interactors to avoid duplicates
    existing_primaries = {i.get('primary') for i in interactors}

    added_count = 0
    for link in direct_links:
        primary = link.get('primary')

        # Check if this interactor already exists
        if primary in existing_primaries:
            if verbose:
                print(f"  [MERGE] Skipping {primary} (already exists)")
            continue

        # Add to interactors
        interactors.append(link)
        existing_primaries.add(primary)
        added_count += 1

        if verbose:
            print(f"  [MERGE] ✓ Added {primary} as direct mediator link")

    if verbose:
        print(f"[MERGE] Added {added_count}/{len(direct_links)} new direct links")

    # Update payload
    snapshot['interactors'] = interactors
    payload['snapshot_json'] = snapshot

    # Also update ctx_json if present
    if 'ctx_json' in payload:
        payload['ctx_json']['interactors'] = interactors

    return payload


# ============================================================================
# Helper functions for compatibility with validation script
# ============================================================================

def validate_arrows_for_payload(
    payload: Dict[str, Any],
    api_key: str = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Full validation pipeline: arrows + direct link extraction.

    This is the main entry point called by runner.py.

    Args:
        payload: Pipeline JSON result
        api_key: Google API key
        verbose: Enable logging

    Returns:
        Validated and enhanced payload
    """
    if verbose:
        print("\n" + "="*60)
        print("INTEGRATED ARROW VALIDATION + DIRECT LINK EXTRACTION")
        print("="*60)

    # Step 1: Basic arrow validation (if arrow validator available)
    try:
        from utils.arrow_effect_validator import validate_arrows_and_effects

        if verbose:
            print("[STAGE 1] Running arrow/effect validation...")

        payload = validate_arrows_and_effects(
            payload=payload,
            api_key=api_key,
            verbose=verbose
        )

        if verbose:
            print("[STAGE 1] ✓ Arrow validation complete")

    except ImportError:
        if verbose:
            print("[STAGE 1] ⚠ Arrow validator not available, skipping")

    # Step 2: Extract direct mediator links
    if verbose:
        print("\n[STAGE 2] Extracting direct mediator links...")

    direct_links = extract_direct_mediator_links_from_json(
        payload=payload,
        api_key=api_key,
        verbose=verbose
    )

    # Step 3: Merge into payload
    if direct_links:
        if verbose:
            print(f"\n[STAGE 3] Merging {len(direct_links)} direct links...")

        payload = merge_direct_links_into_payload(
            payload=payload,
            direct_links=direct_links,
            verbose=verbose
        )

    if verbose:
        print("\n" + "="*60)
        print("VALIDATION COMPLETE")
        print("="*60 + "\n")

    return payload
