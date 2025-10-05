#!/usr/bin/env python3
"""
Comprehensive API Testing Script with Full Feature Support
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

def create_full_feature_test():
    """Create a test with all 96 features that the model expects"""
    # All 96 features from the model
    all_features = [
        'koi_period', 'koi_period_err1', 'koi_period_err2', 'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2', 
        'koi_time0', 'koi_time0_err1', 'koi_time0_err2', 'koi_eccen', 'koi_impact', 'koi_impact_err1', 
        'koi_impact_err2', 'koi_duration', 'koi_duration_err1', 'koi_duration_err2', 'koi_depth', 'koi_depth_err1', 
        'koi_depth_err2', 'koi_ror', 'koi_ror_err1', 'koi_ror_err2', 'koi_srho', 'koi_srho_err1', 'koi_srho_err2', 
        'koi_prad', 'koi_prad_err1', 'koi_prad_err2', 'koi_sma', 'koi_incl', 'koi_teq', 'koi_insol', 
        'koi_insol_err1', 'koi_insol_err2', 'koi_dor', 'koi_dor_err1', 'koi_dor_err2', 'koi_ldm_coeff4', 
        'koi_ldm_coeff3', 'koi_ldm_coeff2', 'koi_ldm_coeff1', 'koi_max_sngle_ev', 'koi_max_mult_ev', 
        'koi_model_snr', 'koi_tce_plnt_num', 'koi_bin_oedp_sig', 'koi_steff', 'koi_steff_err1', 'koi_steff_err2', 
        'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2', 'koi_smet', 'koi_smet_err1', 'koi_smet_err2', 
        'koi_srad', 'koi_srad_err1', 'koi_srad_err2', 'koi_smass', 'koi_smass_err1', 'koi_smass_err2', 
        'ra', 'dec', 'koi_kepmag', 'koi_gmag', 'koi_rmag', 'koi_imag', 'koi_zmag', 'koi_jmag', 'koi_hmag', 
        'koi_kmag', 'koi_fwm_stat_sig', 'koi_fwm_sra', 'koi_fwm_sra_err', 'koi_fwm_sdec', 'koi_fwm_sdec_err', 
        'koi_fwm_srao', 'koi_fwm_srao_err', 'koi_fwm_sdeco', 'koi_fwm_sdeco_err', 'koi_fwm_prao', 
        'koi_fwm_prao_err', 'koi_fwm_pdeco', 'koi_fwm_pdeco_err', 'koi_dicco_mra', 'koi_dicco_mra_err', 
        'koi_dicco_mdec', 'koi_dicco_mdec_err', 'koi_dicco_msky', 'koi_dicco_msky_err', 'koi_dikco_mra', 
        'koi_dikco_mra_err', 'koi_dikco_mdec', 'koi_dikco_mdec_err', 'koi_dikco_msky', 'koi_dikco_msky_err'
    ]
    
    # Create test data with realistic values
    test_features = {}
    for feature in all_features:
        if 'err' in feature:
            test_features[feature] = 0.1  # Small error values
        elif 'koi_period' in feature:
            test_features[feature] = 10.5
        elif 'koi_depth' in feature:
            test_features[feature] = 1000.0
        elif 'koi_duration' in feature:
            test_features[feature] = 2.5
        elif 'koi_steff' in feature:
            test_features[feature] = 5000.0
        elif 'koi_srad' in feature:
            test_features[feature] = 1.0
        elif 'ra' in feature or 'dec' in feature:
            test_features[feature] = 180.0  # Coordinate values
        elif 'koi_kepmag' in feature or 'koi_gmag' in feature or 'koi_rmag' in feature:
            test_features[feature] = 12.0  # Magnitude values
        else:
            test_features[feature] = 1.0  # Default value
    
    return test_features

def main():
    base_url = 'http://localhost:8000'
    print('=== COMPREHENSIVE API TESTING WITH FULL FEATURES ===')
    print()
    
    # Create test data with all 96 features
    full_test_features = create_full_feature_test()
    print(f"Created test data with {len(full_test_features)} features")
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
        
        # Prediction endpoints with full features
        ("Prediction (Full Features)", "POST", f"{base_url}/predict", {
            'features': full_test_features
        }),
        ("Predict Raw (Full Features)", "POST", f"{base_url}/predict-raw", full_test_features),
        
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
