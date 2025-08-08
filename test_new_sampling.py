#!/usr/bin/env python3
"""
Test script for new sampling functionality
"""

import numpy as np
from pennylane.labs.trotter_error.product_formulas.error import (
    SamplingConfig, 
    perturbation_error_with_config,
    _initialize_convergence_info,
    _get_z_score
)

def test_basic_functionality():
    """Test basic functionality of new features"""
    print("=== Testing Basic Functionality ===")
    
    # Test 1: SamplingConfig creation
    print("Test 1: SamplingConfig creation")
    try:
        config1 = SamplingConfig()
        print(f"✓ Basic config: {config1.sampling_method}")
        
        config2 = SamplingConfig(use_adaptive_stopping=True, confidence_level=0.95)
        print(f"✓ Adaptive config: {config2.use_adaptive_stopping}")
        
        config3 = SamplingConfig(use_convergence_stopping=True, convergence_tolerance=1e-6)
        print(f"✓ Convergence config: {config3.use_convergence_stopping}")
        
        # Test mutual exclusion
        try:
            config_bad = SamplingConfig(use_adaptive_stopping=True, use_convergence_stopping=True)
            print("✗ Should have failed")
        except ValueError as e:
            print(f"✓ Mutual exclusion check: {str(e)[:50]}...")
            
    except Exception as e:
        print(f"✗ SamplingConfig test failed: {e}")
        return False
    
    # Test 2: Helper functions
    print("\nTest 2: Helper functions")
    try:
        # Test z-score
        z = _get_z_score(0.95)
        print(f"✓ Z-score for 95%: {z}")
        
        # Test convergence info initialization
        fake_commutators = [('A',), ('B',), ('C',)]
        config = SamplingConfig(return_convergence_info=True)
        conv_info = _initialize_convergence_info(fake_commutators, config, order=3, timestep=0.1)
        print(f"✓ Convergence info: {conv_info['global']['total_commutators']} commutators")
        
    except Exception as e:
        print(f"✗ Helper functions test failed: {e}")
        return False
    
    print("✓ All basic functionality tests passed!")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n🎉 All tests passed! Ready for next step.")
    else:
        print("\n❌ Some tests failed. Need to fix issues.")
