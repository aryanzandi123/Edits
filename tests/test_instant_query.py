#!/usr/bin/env python3
"""
Test instant query feature

Demonstrates that querying a protein already in the database returns instantly.
"""

import requests
import time
import json

# Test with local Flask server
BASE_URL = "http://localhost:5000"

def test_instant_query():
    """Test that querying ATXN3 (already in database) returns instantly."""

    print("\n" + "="*80)
    print("TEST: Instant Query for Protein Already in Database")
    print("="*80)

    # Query ATXN3 which is already in the database (from migration)
    protein = "ATXN3"

    print(f"\n1. Querying {protein} (already in database)...")

    start_time = time.time()

    response = requests.post(
        f"{BASE_URL}/api/query",
        json={"protein": protein}
    )

    elapsed = time.time() - start_time

    print(f"   Response time: {elapsed:.3f} seconds")

    if response.status_code == 200:
        data = response.json()
        status = data.get("status")
        source = data.get("source")
        count = data.get("count")

        print(f"   Status: {status}")
        print(f"   Source: {source}")
        print(f"   Interactions: {count}")

        if status == "complete" and source == "database" and elapsed < 1.0:
            print(f"\n   [SUCCESS] Instant query worked! < 1 second")
            print(f"   Database returned {count} interactions instantly")
        elif status == "complete" and source == "cache":
            print(f"\n   [SUCCESS] Fallback to cache worked!")
        elif status == "processing":
            print(f"\n   [FAIL] Still running pipeline (should be instant!)")
            return False
        else:
            print(f"\n   [WARNING] Unexpected response: {data}")
    else:
        print(f"   [ERROR] Request failed: {response.status_code}")
        print(f"   {response.text}")
        return False

    # Now fetch the visualization to confirm
    print(f"\n2. Fetching visualization...")

    viz_response = requests.get(f"{BASE_URL}/api/results/{protein}")

    if viz_response.status_code == 200:
        results = viz_response.json()
        interactors = results.get("snapshot_json", {}).get("interactors", [])
        print(f"   Found {len(interactors)} interactors in visualization")
        print(f"   [OK] Visualization data ready")
    else:
        print(f"   [ERROR] Failed to fetch results: {viz_response.status_code}")

    print("\n" + "="*80)
    return True


def test_new_protein_query():
    """Test that querying a new protein (not in database) runs pipeline."""

    print("\n" + "="*80)
    print("TEST: New Protein Query (Not in Database)")
    print("="*80)

    # Use a fake protein that definitely isn't in the database
    protein = "TESTPROTEIN123"

    print(f"\n1. Querying {protein} (NOT in database)...")

    start_time = time.time()

    response = requests.post(
        f"{BASE_URL}/api/query",
        json={"protein": protein}
    )

    elapsed = time.time() - start_time

    print(f"   Response time: {elapsed:.3f} seconds")

    if response.status_code == 200:
        data = response.json()
        status = data.get("status")

        print(f"   Status: {status}")

        if status == "processing":
            print(f"\n   [SUCCESS] Pipeline started for new protein!")
            print(f"   (You can cancel this job in the UI)")
        else:
            print(f"\n   [WARNING] Expected 'processing', got '{status}'")
    else:
        print(f"   [ERROR] Request failed: {response.status_code}")
        print(f"   {response.text}")

    print("\n" + "="*80)


def test_comparison():
    """Compare instant query vs pipeline query."""

    print("\n" + "="*80)
    print("COMPARISON: Instant vs Pipeline Query")
    print("="*80)

    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│ Protein in Database (ATXN3)        │ < 1 second       │")
    print("│ Protein NOT in Database (NEW)      │ 5-10 minutes     │")
    print("└─────────────────────────────────────────────────────────┘")

    print("\nWith the new system:")
    print("  - Re-querying known proteins: INSTANT")
    print("  - First-time queries: Full pipeline")
    print("  - Re-query button: Finds NEW interactions and adds to existing")

    print("\n" + "="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("INSTANT QUERY TEST SUITE")
    print("="*80)
    print("\nMake sure Flask server is running:")
    print("  python app.py")
    print("\n" + "="*80)

    try:
        # Test 1: Instant query for ATXN3
        test_instant_query()

        # Test 2: New protein query
        # test_new_protein_query()  # Commented out - starts a real pipeline job

        # Comparison
        test_comparison()

    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Could not connect to Flask server!")
        print("Start the server with: python app.py")
        print("\n" + "="*80)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        print("\n" + "="*80)
