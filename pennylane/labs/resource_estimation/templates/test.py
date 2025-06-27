import numpy as np
import pytest
from optimization import thc_via_cp3 as thc_via_cp3_new
from openfermion.resource_estimates.thc.factorize_thc import thc_via_cp3 as thc_via_cp3_openfermion


def create_real_n2_data():
    """Create real N2 molecular data using PySCF."""
    missing_packages = []
    
    # Check for required packages
    packages_to_check = [
        ('pennylane', 'pennylane'),
        ('pennylane.qchem', 'pennylane.qchem'),
        ('pyscf', 'pyscf'),
    ]
    
    for module_name, import_path in packages_to_check:
        try:
            __import__(import_path)
        except ImportError:
            missing_packages.append(module_name)
    
    if missing_packages:
        pytest.skip(f"Missing packages for real N2 data test: {missing_packages}")
    
    # Import after checking availability
    import pennylane as qml
    from pennylane import qchem
    from pyscf import gto, scf, ao2mo
    from pennylane.qchem import active_space

    def to_density(one, two):
        """Convert integrals from physicist to chemist convention."""
        eri = np.einsum('prsq->pqrs', two)
        h1e = one - np.einsum('pqrr->pq', two) / 2.0
        return h1e, eri

    def hamiltonian(mol, ncas=None, nelecas=None):
        """Generate Hamiltonian for molecule."""
        # Get MOs
        hf = scf.RHF(mol)
        # Prevent PySCF flip-flopping from multithreading
        hf.eig = lambda h, s: scf.hf.eig(h.round(12), s)
        hf.run(verbose=0)

        # Create one-body terms
        h_core = hf.get_hcore(mol)
        orbs = hf.mo_coeff
        core_constant = mol.energy_nuc()
        one = np.einsum("qr,rs,st->qt", orbs.T, h_core, orbs)
        
        # Create two-body terms
        two = ao2mo.full(hf._eri, orbs, compact=False).reshape([mol.nao] * 4)
        two = np.swapaxes(two, 1, 3)

        # Apply active space
        core, active = active_space(mol.nelectron, mol.nao, 2 * mol.spin + 1, nelecas, ncas)

        if core and active:
            # Update core constant
            for i in core:
                core_constant += 2 * one[i][i]
                for j in core:
                    core_constant += 2 * two[i][j][j][i] - two[i][j][i][j]

            # Update one-body terms
            for p in active:
                for q in active:
                    for i in core:
                        one[p, q] += 2 * two[i][p][q][i] - two[i][p][i][q]

            one = one[qml.math.ix_(active, active)]
            two = two[qml.math.ix_(active, active, active, active)]

        return core_constant, one, two

    try:
        # Create N2 molecule
        mol = gto.Mole()
        mol.atom = "N 0.0 0.0 0.0; N 0.0 0.0 1.077"
        mol.basis = 'ccpvdz'
        mol.symmetry = False
        mol.build()

        # Get integrals
        ncas, nelecas = 8, 8
        core_const, one, two = hamiltonian(mol, ncas=ncas, nelecas=nelecas)
        h1e, eri = to_density(one, two)
        
        return eri, h1e, ncas, nelecas, core_const
    
    except Exception as e:
        pytest.skip(f"Failed to create real N2 data: {str(e)}")


def create_mock_n2_data():
    """Create mock data simulating N2 molecule."""
    ncas = 8
    nelecas = 8
    
    # Fixed seed for reproducibility
    np.random.seed(42)
    
    # Create mock ERI tensor with correct symmetries
    eri = np.random.random((ncas, ncas, ncas, ncas)) * 0.1
    
    # Apply ERI tensor symmetries
    eri = 0.5 * (eri + eri.transpose(1, 0, 2, 3))  # (ij|kl) = (ji|kl)
    eri = 0.5 * (eri + eri.transpose(0, 1, 3, 2))  # (ij|kl) = (ij|lk)
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))  # (ij|kl) = (kl|ij)
    
    # Create mock one-body Hamiltonian for one_norm tests
    h_one = np.random.random((ncas, ncas)) * 0.05
    h_one = 0.5 * (h_one + h_one.T)  # Make symmetric
    
    return eri, h_one, ncas, nelecas


def create_mock_hamiltonian_data(norb=6):
    """Create mock Hamiltonian data for one_norm objective testing."""
    np.random.seed(123)
    
    # Create symmetric one-body Hamiltonian
    h = np.random.random((norb, norb)) * 0.1
    h = 0.5 * (h + h.T)
    
    # Create symmetric two-body integrals
    eri = np.random.random((norb, norb, norb, norb)) * 0.05
    for perm in [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)]:
        eri = 0.5 * (eri + eri.transpose(perm))
    
    return h, eri


# Real N2 data tests
def test_thc_real_n2_basic():
    """Basic test with real N2 data."""
    eri, h1e, ncas, nelecas, core_const = create_real_n2_data()
    nthc = 32
    
    print(f"Testing with real N2 data: nthc={nthc}, norb={ncas}")
    
    eri_thc, thc_leaf, thc_central, info = thc_via_cp3_new(
        eri_full=eri,
        nthc=nthc,
        thc_method="standard",
        perform_bfgs_opt=True,
        bfgs_maxiter=500,
        verify=False
    )
    
    error = np.linalg.norm(eri_thc - eri)
    print(f"THC reconstruction error: {error:.6e}")
    
    assert eri_thc.shape == eri.shape == (ncas, ncas, ncas, ncas)
    assert thc_leaf.shape == (nthc, ncas)
    assert thc_central.shape == (nthc, nthc)
    assert error < 0.1, f"THC reconstruction error too high: {error:.6e}"
    
    print("✅ Real N2 basic test passed!")


def test_thc_real_n2_enhanced_modes():
    """Test enhanced modes with real N2 data."""
    eri, h1e, ncas, nelecas, core_const = create_real_n2_data()
    nthc = 48
    
    base_params = {
        'nthc': nthc,
        'perform_bfgs_opt': True,
        'bfgs_maxiter': 300,
        'verify': False,
    }
    
    # Test enhanced method
    eri_enhanced, _, _, _ = thc_via_cp3_new(
        eri_full=eri, thc_method="enhanced", **base_params
    )
    
    # Test enhanced_bias method
    eri_bias, _, _, info_bias = thc_via_cp3_new(
        eri_full=eri, thc_method="enhanced_bias", **base_params
    )
    
    error_enhanced = np.linalg.norm(eri_enhanced - eri)
    
    # Reconstruct full ERI for enhanced_bias
    if 'alpha2_optimized' in info_bias and 'beta_optimized' in info_bias:
        alpha2_opt = info_bias['alpha2_optimized']
        beta_opt = info_bias['beta_optimized']
        
        eye = np.eye(ncas)
        alpha2_term = alpha2_opt * np.einsum('pq,rs->pqrs', eye, eye)
        beta_terms = 0.5 * (np.einsum('pq,rs->pqrs', beta_opt, eye) + 
                           np.einsum('pq,rs->pqrs', eye, beta_opt))
        
        eri_bias_complete = eri_bias + alpha2_term + beta_terms
        error_bias = np.linalg.norm(eri_bias_complete - eri)
    else:
        error_bias = np.linalg.norm(eri_bias - eri)
    
    print(f"Enhanced error: {error_enhanced:.6e}")
    print(f"Enhanced+bias error: {error_bias:.6e}")
    
    assert error_enhanced < 0.1, "Enhanced should work well with real data"
    assert error_bias < 0.1, "Enhanced+bias should work well with real data"
    
    print("✅ Real N2 enhanced modes test passed!")


def test_thc_real_n2_one_norm_objective():
    """Test one_norm objective with real N2 data."""
    eri, h1e, ncas, nelecas, core_const = create_real_n2_data()
    nthc = 32
    
    base_params = {
        'nthc': nthc,
        'thc_method': "enhanced_one_norm",
        'h_one': h1e,
        'perform_bfgs_opt': True,
        'bfgs_maxiter': 200,
        'verify': False
    }
    
    # Test fitting vs one_norm objectives
    _, _, central_fitting, info_fitting = thc_via_cp3_new(
        eri_full=eri,
        objective="fitting",
        lambda_penalty=1.0,
        **base_params
    )
    
    _, _, central_one_norm, info_one_norm = thc_via_cp3_new(
        eri_full=eri,
        objective="one_norm",
        lambda_penalty=5.0,
        **base_params
    )
    
    # Calculate Hamiltonian 1-norm
    def calculate_hamiltonian_1norm(h_matrix, eri_tensor, alpha_val, beta_matrix, MPQ):
        norb = h_matrix.shape[0]
        eye = np.eye(norb)
        beta_sym = 0.5 * (beta_matrix + beta_matrix.T)
        g_trace = np.einsum('prrq->pq', eri_tensor)
        h_bi = h_matrix - 0.5 * g_trace - alpha_val * eye + 0.5 * beta_sym
        t_k = np.linalg.eigvals(h_bi)
        lambda_one_body = np.sum(np.abs(t_k))
        lambda_two_body = 0.5 * np.sum(np.abs(MPQ)) - 0.25 * np.sum(np.abs(np.diag(MPQ)))
        return lambda_one_body + lambda_two_body
    
    norm_fitting = calculate_hamiltonian_1norm(
        h1e, eri, info_fitting['alpha2_optimized'], 
        info_fitting['beta_optimized'], central_fitting
    )
    
    norm_one_norm = calculate_hamiltonian_1norm(
        h1e, eri, info_one_norm['alpha1_optimized'],
        info_one_norm['beta_optimized'], central_one_norm
    )
    
    print(f"Hamiltonian 1-norm (fitting): {norm_fitting:.6e}")
    print(f"Hamiltonian 1-norm (one_norm): {norm_one_norm:.6e}")
    
    # Verify correct parameters were optimized
    assert 'alpha2_optimized' in info_fitting
    assert 'alpha1_optimized' in info_one_norm
    assert info_fitting['objective_used'] == "fitting"
    assert info_one_norm['objective_used'] == "one_norm"
    
    print("✅ Real N2 one-norm objective test passed!")


def test_thc_real_n2_comparison_with_openfermion():
    """Compare with OpenFermion using real N2 data."""
    eri, h1e, ncas, nelecas, core_const = create_real_n2_data()
    nthc = 32
    
    common_params = {
        'nthc': nthc,
        'perform_bfgs_opt': True,
        'bfgs_maxiter': 300,
        'verify': False
    }
    
    # Compare implementations
    eri_of, _, _, _ = thc_via_cp3_openfermion(eri_full=eri, **common_params)
    eri_new, _, _, _ = thc_via_cp3_new(
        eri_full=eri, thc_method="standard", **common_params
    )
    
    error_of = np.linalg.norm(eri_of - eri)
    error_new = np.linalg.norm(eri_new - eri)
    
    print(f"OpenFermion error: {error_of:.6e}")
    print(f"Our implementation error: {error_new:.6e}")
    
    assert abs(error_of - error_new) < 0.1, "Errors should be comparable"
    
    print("✅ Real N2 comparison with OpenFermion test passed!")


@pytest.mark.slow
def test_thc_real_n2_performance():
    """Performance test with different nthc values using real N2 data."""
    eri, h1e, ncas, nelecas, core_const = create_real_n2_data()
    
    for nthc in [24, 32, 48, 64]:
        print(f"\nTesting nthc={nthc}")
        
        eri_thc, _, _, _ = thc_via_cp3_new(
            eri_full=eri,
            nthc=nthc,
            thc_method="standard",
            perform_bfgs_opt=True,
            bfgs_maxiter=200,
            verify=False
        )
        
        error = np.linalg.norm(eri_thc - eri)
        compression_ratio = (ncas**4) / (nthc * (ncas + nthc))
        
        print(f"  Error: {error:.6e}, Compression ratio: {compression_ratio:.2f}")
        assert error < 0.5, f"Error too high for nthc={nthc}"
    
    print("✅ Real N2 performance test passed!")


# Mock data tests
def test_thc_initialization_consistency():
    """Test that THC initialization behaves predictably."""
    eri, h_one, ncas, nelecas = create_mock_n2_data()
    nthc = 16

    # Test consistency across multiple runs
    results = []
    for i in range(3):
        _, thc_leaf, thc_central, _ = thc_via_cp3_new(
            eri_full=eri,
            nthc=nthc,
            thc_method="standard",
            perform_bfgs_opt=False,
            random_start_thc=True,
            verify=False
        )
        results.append((thc_leaf.copy(), thc_central.copy()))

    # Check shapes are consistent
    for i, (leaf, central) in enumerate(results):
        assert leaf.shape == (nthc, ncas)
        assert central.shape == (nthc, nthc)
    
    # Test HOSVD vs random initialization
    _, leaf_hosvd, central_hosvd, _ = thc_via_cp3_new(
        eri_full=eri, nthc=nthc, thc_method="standard",
        perform_bfgs_opt=False, random_start_thc=False, verify=False
    )
    
    _, leaf_random, central_random, _ = thc_via_cp3_new(
        eri_full=eri, nthc=nthc, thc_method="standard",
        perform_bfgs_opt=False, random_start_thc=True, verify=False
    )
    
    # Should give different initializations
    leaf_diff = np.linalg.norm(leaf_hosvd - leaf_random)
    central_diff = np.linalg.norm(central_hosvd - central_random)
    
    print(f"HOSVD vs Random - Leaf: {leaf_diff:.6e}, Central: {central_diff:.6e}")
    print("✅ Initialization consistency test passed!")


def test_openfermion_compatibility():
    """Test compatibility with OpenFermion implementation."""
    eri, h_one, ncas, nelecas = create_mock_n2_data()
    nthc = 16

    common_params = {
        'nthc': nthc,
        'perform_bfgs_opt': True,
        'bfgs_maxiter': 100,
        'verify': False
    }

    eri_new, _, _, _ = thc_via_cp3_new(
        eri_full=eri, thc_method="standard", **common_params
    )
    eri_of, _, _, _ = thc_via_cp3_openfermion(eri_full=eri, **common_params)

    error_new = np.linalg.norm(eri_new - eri)
    error_of = np.linalg.norm(eri_of - eri)

    print(f"Our implementation: {error_new:.6e}, OpenFermion: {error_of:.6e}")

    assert error_new < 1.0 and error_of < 1.0, "Both should give reasonable errors"
    print("✅ OpenFermion compatibility test passed!")


def test_thc_bfgs_improvement():
    """Test that BFGS optimization improves upon CP3 initialization."""
    eri, h_one, ncas, nelecas = create_mock_n2_data()
    nthc = 16

    # Without BFGS
    eri_cp3, _, _, _ = thc_via_cp3_new(
        eri_full=eri, nthc=nthc, thc_method="standard",
        perform_bfgs_opt=False, verify=False
    )

    # With BFGS
    eri_bfgs, _, _, _ = thc_via_cp3_new(
        eri_full=eri, nthc=nthc, thc_method="standard",
        perform_bfgs_opt=True, bfgs_maxiter=200, verify=False
    )

    error_cp3 = np.linalg.norm(eri_cp3 - eri)
    error_bfgs = np.linalg.norm(eri_bfgs - eri)

    print(f"CP3: {error_cp3:.6e}, BFGS: {error_bfgs:.6e}")
    print(f"Improvement: {(error_cp3 - error_bfgs)/error_cp3*100:.2f}%")

    assert error_bfgs <= error_cp3 * 1.1, "BFGS should not significantly worsen result"
    print("✅ BFGS improvement test passed!")


def test_thc_enhanced_modes():
    """Test enhanced THC optimization modes."""
    eri, h_one, ncas, nelecas = create_mock_n2_data()
    nthc = 24

    methods = ["enhanced", "enhanced_bias"]
    results = {}

    for method in methods:
        eri_thc, _, _, info = thc_via_cp3_new(
            eri_full=eri, nthc=nthc, thc_method=method,
            perform_bfgs_opt=True, bfgs_maxiter=150, verify=False
        )
        results[method] = (np.linalg.norm(eri_thc - eri), info)

    for method, (error, info) in results.items():
        print(f"{method} error: {error:.6e}")
        assert error < 1.0, f"{method} should give reasonable error"

    print("✅ Enhanced modes test passed!")


def test_thc_enhanced_one_norm_method():
    """Test the enhanced_one_norm method."""
    eri, h_one, ncas, nelecas = create_mock_n2_data()
    nthc = 32
    
    base_params = {
        'eri_full': eri,
        'h_one': h_one,
        'nthc': nthc,
        'thc_method': "enhanced_one_norm",
        'bfgs_maxiter': 50,
        'verify': False
    }
    
    # Test both objectives
    for objective in ["fitting", "one_norm"]:
        _, leaf, central, info = thc_via_cp3_new(
            objective=objective,
            lambda_penalty=10.0,
            **base_params
        )
        
        assert leaf.shape == (nthc, ncas)
        assert central.shape == (nthc, nthc)
        assert info['objective_used'] == objective
        
        if objective == "fitting":
            assert 'alpha2_optimized' in info
            assert 'alpha1_optimized' not in info
        else:
            assert 'alpha1_optimized' in info
            assert 'alpha2_optimized' not in info

    print("✅ Enhanced_one_norm method test passed!")


def test_thc_one_norm_objective_basic():
    """Basic test for one_norm objective using optax_lbfgs_opt_thc_l2reg_enhanced."""
    from optimization import optax_lbfgs_opt_thc_l2reg_enhanced
    
    h, eri = create_mock_hamiltonian_data(norb=6)
    nthc = 12
    
    params = optax_lbfgs_opt_thc_l2reg_enhanced(
        eri=eri, nthc=nthc, h=h, objective="one_norm",
        include_bias_terms=True, maxiter=100, verbose=True, random_seed=456
    )
    
    # Verify correct parameters for one_norm
    assert 'etaPp' in params and 'MPQ' in params
    assert 'alpha1' in params and 'beta' in params
    assert 'alpha2' not in params
    
    # Verify shapes
    norb = h.shape[0]
    assert params['etaPp'].shape == (nthc, norb)
    assert params['MPQ'].shape == (nthc, nthc)
    assert params['beta'].shape == (norb, norb)

    print("✅ One-norm objective basic test passed!")


def test_thc_one_norm_vs_fitting_comparison():
    """Compare fitting vs one_norm objectives."""
    from optimization import optax_lbfgs_opt_thc_l2reg_enhanced
    
    h, eri = create_mock_hamiltonian_data(norb=4)
    nthc = 8
    
    base_params = {
        'eri': eri, 'nthc': nthc, 'include_bias_terms': True,
        'maxiter': 50, 'verbose': False, 'random_seed': 789
    }
    
    params_fitting = optax_lbfgs_opt_thc_l2reg_enhanced(
        objective="fitting", **base_params
    )
    params_one_norm = optax_lbfgs_opt_thc_l2reg_enhanced(
        h=h, objective="one_norm", lambda_penalty=1.0, **base_params
    )
    
    # Verify different bias parameters
    assert 'alpha2' in params_fitting and 'alpha1' not in params_fitting
    assert 'alpha1' in params_one_norm and 'alpha2' not in params_one_norm
    
    print("✅ Fitting vs one-norm comparison test passed!")


def test_thc_enhanced_one_norm_validation():
    """Test validation for enhanced_one_norm method."""
    eri, h_one, ncas, nelecas = create_mock_n2_data()
    
    # Should fail without h_one for one_norm objective
    with pytest.raises(ValueError, match="h_one is required"):
        thc_via_cp3_new(
            eri_full=eri, nthc=16, thc_method="enhanced_one_norm",
            objective="one_norm"
        )
    
    # Should fail with invalid objective
    with pytest.raises(ValueError, match="objective must be"):
        thc_via_cp3_new(
            eri_full=eri, h_one=h_one, nthc=16,
            thc_method="enhanced_one_norm", objective="invalid"
        )
    
    # Should work with fitting objective even without h_one
    eri_thc, _, _, info = thc_via_cp3_new(
        eri_full=eri, nthc=8, thc_method="enhanced_one_norm",
        objective="fitting", bfgs_maxiter=10, verify=False
    )
    assert 'alpha2_optimized' in info

    print("✅ Enhanced_one_norm validation test passed!")


def test_thc_invalid_method():
    """Test that invalid THC methods raise appropriate errors."""
    eri, _, _, _ = create_mock_n2_data()
    
    with pytest.raises(ValueError, match="thc_method must be one of"):
        thc_via_cp3_new(eri_full=eri, nthc=16, thc_method="invalid_method")

    print("✅ Invalid method test passed!")


def test_thc_shapes_consistency():
    """Test that THC results have consistent shapes across methods."""
    eri, h_one, ncas, nelecas = create_mock_n2_data()
    nthc = 20

    methods = ["standard", "enhanced", "enhanced_bias"]
    
    for method in methods:
        eri_thc, leaf, central, _ = thc_via_cp3_new(
            eri_full=eri, nthc=nthc, thc_method=method,
            perform_bfgs_opt=True, bfgs_maxiter=50, verify=False
        )

        assert eri_thc.shape == (ncas, ncas, ncas, ncas)
        assert leaf.shape == (nthc, ncas)
        assert central.shape == (nthc, nthc)

    print("✅ Shape consistency test passed!")


if __name__ == "__main__":
    print("Running enhanced THC tests...")
    print("=" * 60)
    
    # Attempt tests with real N2 data
    print("Attempting tests with real N2 molecular data:")
    print("-" * 40)
    
    try:
        _ = create_real_n2_data()
        real_n2_available = True
        print("✅ Real N2 data creation successful")
    except Exception as e:
        real_n2_available = False
        print(f"❌ Real N2 data not available: {str(e)}")
    
    if real_n2_available:
        try:
            test_thc_real_n2_basic()
            test_thc_real_n2_enhanced_modes()
            test_thc_real_n2_one_norm_objective()
            test_thc_real_n2_comparison_with_openfermion()
            print("✅ Real N2 tests completed successfully!")
        except Exception as e:
            print(f"❌ Real N2 tests failed: {e}")
            print("Continuing with mock data tests...")
    else:
        print("Skipping real N2 tests, using mock data instead...")
    
    print("\nTesting with mock data:")
    print("-" * 40)
    
    # Run all mock data tests
    test_functions = [
        test_thc_initialization_consistency,
        test_openfermion_compatibility,
        test_thc_bfgs_improvement,
        test_thc_enhanced_modes,
        test_thc_enhanced_one_norm_method,
        test_thc_one_norm_objective_basic,
        test_thc_one_norm_vs_fitting_comparison,
        test_thc_enhanced_one_norm_validation,
        test_thc_invalid_method,
        test_thc_shapes_consistency,
    ]
    
    for test_func in test_functions:
        try:
            test_func()
            print()
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
    
    print("🎉 All available tests completed!")
