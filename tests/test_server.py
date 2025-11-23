#!/usr/bin/env python3
"""Quick test script to verify server is responding correctly"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_query_endpoint():
    """Test the /api/query endpoint with ATXN3"""
    print("Testing /api/query endpoint with ATXN3...")

    try:
        response = requests.post(
            f"{BASE_URL}/api/query",
            json={
                "protein": "ATXN3",
                "interactor_rounds": 3,
                "function_rounds": 3,
                "skip_validation": False
            },
            timeout=5
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        data = response.json()
        if data.get("status") == "processing":
            print("[OK] Server correctly started a new job for ATXN3")
        elif data.get("status") == "complete":
            print("[ERROR] Server says ATXN3 is cached (but it shouldn't be)")
        else:
            print(f"[WARN] Unexpected status: {data.get('status')}")

    except requests.exceptions.ConnectionError:
        print("[ERROR] Server is not running on localhost:5000")
    except Exception as e:
        print(f"[ERROR] Error: {e}")

def test_cache_check():
    """Check what's actually in the cache directory"""
    import os
    cache_dir = "cache"

    print(f"\nChecking cache directory: {cache_dir}")
    if os.path.exists(cache_dir):
        files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        print(f"Cached proteins: {files}")

        if "ATXN3.json" in files:
            print("[X] ATXN3.json exists in cache")
        else:
            print("[OK] ATXN3.json does NOT exist in cache")
    else:
        print("[ERROR] Cache directory doesn't exist")

if __name__ == "__main__":
    test_cache_check()
    print("\n" + "="*60 + "\n")
    test_query_endpoint()
