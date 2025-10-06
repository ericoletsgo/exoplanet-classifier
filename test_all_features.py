#!/usr/bin/env python3
"""
Comprehensive API Testing Script
Tests all endpoints and features of the Exoplanet Classifier API
"""

import requests
import json
import sys

def test_endpoint(name, method, url, data=None, expected_status=200):
    """Test a single endpoint"""
    print(f"{name}:")
    try:
        if method.upper() == 'GET':
            response = requests.get(url)
        elif method.upper() == 'POST':
            response = requests.post(url, json=data)
        else:
            print(f"   Error: Unsupported method {method}")
            return False
            
        print(f"   Status: {response.status_code}")
        
        if response.status_code == expected_status:
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
                except:
                    print(f"   Response: {response.text[:200]}...")
            print(f"   [OK] PASSED")
            return True
        else:
            print(f"   Error: {response.text}")
            print(f"   [FAIL] FAILED")
            return False
            
    except Exception as e:
        print(f"   Error: {e}")
        print(f"   [FAIL] FAILED")
        return False

def main():
    base_url = 'http://localhost:8000'
    print('=== COMPREHENSIVE API TESTING ===')
    print()
    
    tests = [
        # Basic endpoints
        ("Health Check", "GET", f"{base_url}/"),
        ("Features", "GET", f"{base_url}/features"),
        ("Metrics", "GET", f"{base_url}/metrics"),
        ("Algorithms", "GET", f"{base_url}/algorithms"),
        ("Models List", "GET", f"{base_url}/models"),
        
        # Data endpoints
        ("Random Example", "GET", f"{base_url}/random-example/koi"),
        ("Datasets", "GET", f"{base_url}/datasets/koi?page=1&page_size=5"),
        ("Feature Correlations", "GET", f"{base_url}/feature-correlations"),
        
        # Prediction endpoints
        ("Prediction", "POST", f"{base_url}/predict", {
            'features': {
                'koi_period': 10.5,
                'koi_depth': 1000,
                'koi_duration': 2.5,
                'koi_steff': 5000,
                'koi_srad': 1.0
            }
        }),
        ("Predict Raw", "POST", f"{base_url}/predict-raw", {
            'koi_period': 10.5,
            'koi_depth': 1000,
            'koi_duration': 2.5
        }),
        
        # Model evaluation (if models exist)
        ("Model Evaluation", "GET", f"{base_url}/models/20251004_193916/evaluate"),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, method, url, *data in tests:
        test_data = data[0] if data else None
        if test_endpoint(name, method, url, test_data):
            passed += 1
        print()
    
    print(f"=== RESULTS: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("[SUCCESS] ALL TESTS PASSED! API is fully functional.")
        return 0
    else:
        print("[ERROR] Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
