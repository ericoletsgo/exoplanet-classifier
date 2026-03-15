#!/usr/bin/env python3
"""
Comprehensive test suite for Exoplanet Classifier API
Tests all endpoints and functionality
"""

import requests
import json
import time
import sys

# Test configuration
BASE_URL = "http://localhost:8000"  # Change to your deployed URL for production testing
API_URL = f"{BASE_URL}/api"

def test_endpoint(endpoint, method="GET", data=None, expected_status=200):
    """Test a single endpoint"""
    url = f"{API_URL}{endpoint}"
    print(f"Testing {method} {endpoint}...")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == expected_status:
            print(f"✅ {endpoint} - Status: {response.status_code}")
            return True, response.json() if response.content else None
        else:
            print(f"❌ {endpoint} - Status: {response.status_code}, Error: {response.text}")
            return False, None
    except Exception as e:
        print(f"❌ {endpoint} - Exception: {str(e)}")
        return False, None

def test_all_features():
    """Test all API endpoints"""
    print("🚀 Testing Exoplanet Classifier API")
    print("=" * 50)
    
    tests = [
        # Basic endpoints
        ("/", "GET"),
        ("/health", "GET"),
        ("/features", "GET"),
        
        # Prediction endpoints
        ("/predict", "POST", {
            "features": {
                "koi_dikco_msky": 0.1,
                "koi_dicco_msky": 0.2,
                "koi_max_mult_ev": 0.3,
                "koi_model_snr": 0.4,
                "koi_dikco_mra": 0.5,
                "koi_fwm_srao": 0.6,
                "koi_fwm_sdeco": 0.7,
                "koi_fwm_sra_err": 0.8,
                "koi_fwm_sdec_err": 0.9,
                "koi_fwm_srao_err": 1.0,
                "koi_period": 10.5,
                "koi_depth": 1000.0,
                "koi_duration": 2.5,
                "koi_prad": 1.2,
                "koi_impact": 0.3,
                "koi_steff": 5500.0,
                "koi_srad": 1.0,
                "koi_slogg": 4.4,
                "koi_kepmag": 12.0
            }
        }),
        
        # Metrics and analysis
        ("/metrics", "GET"),
        ("/feature-correlations", "GET"),
        ("/models", "GET"),
        
        # Dataset endpoints
        ("/datasets/koi", "GET"),
        ("/datasets/koi/columns", "GET"),
        ("/random-example/koi", "GET"),
        
        # Batch prediction
        ("/batch-predict", "POST", {
            "records": [
                {
                    "koi_dikco_msky": 0.1,
                    "koi_dicco_msky": 0.2,
                    "koi_max_mult_ev": 0.3,
                    "koi_model_snr": 0.4,
                    "koi_dikco_mra": 0.5,
                    "koi_fwm_srao": 0.6,
                    "koi_fwm_sdeco": 0.7,
                    "koi_fwm_sra_err": 0.8,
                    "koi_fwm_sdec_err": 0.9,
                    "koi_fwm_srao_err": 1.0,
                    "koi_period": 10.5,
                    "koi_depth": 1000.0,
                    "koi_duration": 2.5,
                    "koi_prad": 1.2,
                    "koi_impact": 0.3,
                    "koi_steff": 5500.0,
                    "koi_srad": 1.0,
                    "koi_slogg": 4.4,
                    "koi_kepmag": 12.0
                }
            ]
        })
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if len(test) == 2:
            endpoint, method = test
            data = None
        else:
            endpoint, method, data = test
        
        success, response = test_endpoint(endpoint, method, data)
        if success:
            passed += 1
        
        time.sleep(0.5)  # Small delay between requests
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working perfectly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return False

def test_performance():
    """Test performance of key endpoints"""
    print("\n⚡ Performance Testing")
    print("=" * 30)
    
    performance_tests = [
        ("/features", "GET"),
        ("/metrics", "GET"),
        ("/feature-correlations", "GET"),
        ("/models", "GET")
    ]
    
    for endpoint, method in performance_tests:
        start_time = time.time()
        success, _ = test_endpoint(endpoint, method)
        end_time = time.time()
        
        if success:
            duration = end_time - start_time
            print(f"⏱️ {endpoint}: {duration:.2f}s")
            
            if duration > 10:
                print(f"⚠️ {endpoint} is slow ({duration:.2f}s)")
            elif duration > 5:
                print(f"⚡ {endpoint} is moderate ({duration:.2f}s)")
            else:
                print(f"🚀 {endpoint} is fast ({duration:.2f}s)")

if __name__ == "__main__":
    print("Exoplanet Classifier API Test Suite")
    print("Make sure your API is running on", BASE_URL)
    print()
    
    # Test all functionality
    all_passed = test_all_features()
    
    # Test performance
    test_performance()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! Your API is ready for production.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)
