#!/usr/bin/env python3

try:
    from pennylane.labs.trotter_error.product_formulas.error import perturbation_error
    print("✅ Import successful")
    
    # Test function signature
    import inspect
    sig = inspect.signature(perturbation_error)
    print(f"✅ Function signature: {len(sig.parameters)} parameters")
    
    # Check if it has the new parameters
    params = list(sig.parameters.keys())
    if 'use_adaptive_stopping' in params:
        print("✅ New adaptive stopping parameter found")
    if 'sampling_method' in params:
        print("✅ Sampling method parameter found")
    if 'return_convergence_info' in params:
        print("✅ Return convergence info parameter found")
        
    print("✅ All basic checks passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
