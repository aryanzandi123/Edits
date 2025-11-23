#!/usr/bin/env python3
"""
Test that the visualization endpoint works with database-only data
"""

import requests
import os
from pathlib import Path

BASE_URL = "http://localhost:5000"

def test_visualize_from_database():
    """Test that /api/visualize works when only database data exists."""

    print("\n" + "="*80)
    print("TEST: Visualize from Database (No Old Cache)")
    print("="*80)

    protein = "ATXN3"

    # 1. Remove old cache file if it exists (to simulate database-only scenario)
    old_cache = Path(f"cache/{protein}.json")
    if old_cache.exists():
        print(f"\n1. Removing old cache file to test database-only scenario...")
        old_cache.unlink()
        print(f"   Removed: {old_cache}")
    else:
        print(f"\n1. Old cache file doesn't exist (good for testing!)")

    # 2. Verify database has data
    import protein_database as pdb
    interactions = pdb.get_all_interactions(protein)
    print(f"\n2. Database check:")
    print(f"   Interactions in database: {len(interactions)}")

    if not interactions:
        print(f"   [ERROR] Database has no data for {protein}!")
        print(f"   Run migration first: python migrate_cache.py --yes")
        return False

    # 3. Try to visualize (should create cache from database)
    print(f"\n3. Requesting visualization...")

    try:
        response = requests.get(f"{BASE_URL}/api/visualize/{protein}")

        if response.status_code == 200:
            print(f"   Status: {response.status_code} OK")
            print(f"   [SUCCESS] Visualization loaded!")

            # Check if cache file was created
            if old_cache.exists():
                size = old_cache.stat().st_size
                print(f"\n4. Cache file created from database:")
                print(f"   File: {old_cache}")
                print(f"   Size: {size:,} bytes")
                print(f"   [OK] Database-to-cache conversion works!")
            else:
                print(f"\n4. [WARNING] Cache file not created")

            return True

        elif response.status_code == 404:
            print(f"   Status: {response.status_code} NOT FOUND")
            print(f"   Response: {response.text}")
            print(f"\n   [FAIL] Still getting 'Result not found' error!")
            print(f"   The fix didn't work.")
            return False

        else:
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\n   [ERROR] Could not connect to Flask server!")
        print(f"   Start the server with: python app.py")
        return False

    except Exception as e:
        print(f"\n   [ERROR] Test failed: {e}")
        return False


def test_full_instant_query_flow():
    """Test complete flow: query → instant → visualize."""

    print("\n" + "="*80)
    print("TEST: Complete Instant Query Flow")
    print("="*80)

    protein = "VCP"

    # Remove old cache to force database-only scenario
    old_cache = Path(f"cache/{protein}.json")
    if old_cache.exists():
        old_cache.unlink()
        print(f"Removed old cache: {old_cache}")

    try:
        # Step 1: Query
        print(f"\n1. Querying {protein}...")
        query_response = requests.post(
            f"{BASE_URL}/api/query",
            json={"protein": protein}
        )

        query_data = query_response.json()
        status = query_data.get("status")
        source = query_data.get("source")

        print(f"   Status: {status}")
        print(f"   Source: {source}")

        if status != "complete":
            print(f"   [FAIL] Expected 'complete', got '{status}'")
            return False

        # Step 2: Visualize
        print(f"\n2. Loading visualization...")
        viz_response = requests.get(f"{BASE_URL}/api/visualize/{protein}")

        if viz_response.status_code == 200:
            print(f"   Status: 200 OK")
            print(f"   [SUCCESS] Complete flow works!")

            # Check cache created
            if old_cache.exists():
                print(f"\n3. Cache file created: {old_cache}")
                print(f"   [OK] End-to-end instant query works!")
                return True

        else:
            print(f"   Status: {viz_response.status_code}")
            print(f"   [FAIL] Visualization failed!")
            return False

    except Exception as e:
        print(f"   [ERROR] {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VISUALIZATION FIX TEST SUITE")
    print("="*80)
    print("\nMake sure Flask server is running:")
    print("  python app.py")
    print("\n" + "="*80)

    try:
        # Test 1: Visualize from database only
        success1 = test_visualize_from_database()

        # Test 2: Complete flow
        success2 = test_full_instant_query_flow()

        print("\n" + "="*80)
        if success1 and success2:
            print("ALL TESTS PASSED!")
            print("="*80)
            print("\nThe fix works! You can now:")
            print("  1. Query proteins in database → Instant load")
            print("  2. Visualize immediately (no 'Result not found')")
            print("  3. Cache created automatically from database")
        else:
            print("SOME TESTS FAILED")
            print("="*80)
            print("\nCheck the errors above.")

        print("\n" + "="*80 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Test suite failed: {e}")
        print("\n" + "="*80 + "\n")
