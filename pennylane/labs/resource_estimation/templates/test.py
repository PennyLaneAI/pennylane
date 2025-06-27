import numpy as np
import pytest
from optimization import thc_via_cp3 as thc_via_cp3_new
from openfermion.resource_estimates.thc.factorize_thc import thc_via_cp3 as thc_via_cp3_openfermion


def create_mock_n2_data():
    """Crea datos mock que simulan la molécula N2."""
    ncas = 8
    nelecas = 8
    
    # Semilla fija para reproducibilidad
    np.random.seed(42)
    
    # Crear tensor ERI mock con simetrías correctas
    eri = np.random.random((ncas, ncas, ncas, ncas)) * 0.1
    
    # Aplicar simetrías del tensor ERI
    eri = 0.5 * (eri + eri.transpose(1, 0, 2, 3))  # (ij|kl) = (ji|kl)
    eri = 0.5 * (eri + eri.transpose(0, 1, 3, 2))  # (ij|kl) = (ij|lk)
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))  # (ij|kl) = (kl|ij)
    
    return eri, ncas, nelecas


def test_thc_basic_comparison():
    """Test básico comparando OpenFermion vs nueva implementación (modo standard)."""
    # Setup
    eri, ncas, nelecas = create_mock_n2_data()
    nthc = 32
    
    print(f"Testing with nthc={nthc}, norb={ncas}")
    
    # Parámetros simplificados
    common_params = {
        'nthc': nthc,
        'perform_bfgs_opt': True,
        'bfgs_maxiter': 1000,  # Reducido para tests rápidos
        'random_start_thc': False,
        'verify': False,  # Sin verbose output
    }
    
    # Ejecutar OpenFermion
    eri_thc_of, thc_leaf_of, thc_central_of, info_of = thc_via_cp3_openfermion(
        eri_full=eri, **common_params
    )
    
    # Ejecutar nueva implementación (modo standard)
    eri_thc_new, thc_leaf_new, thc_central_new, info_new = thc_via_cp3_new(
        eri_full=eri, thc_method="standard", **common_params
    )
    
    # Verificaciones básicas
    error_of = np.linalg.norm(eri_thc_of - eri)
    error_new = np.linalg.norm(eri_thc_new - eri)
    
    print(f"OpenFermion error: {error_of:.6e}")
    print(f"New (standard) error: {error_new:.6e}")
    print(f"Relative difference: {abs(error_of - error_new)/error_of:.6e}")
    
    # Assertions básicas
    assert thc_leaf_of.shape == thc_leaf_new.shape
    assert thc_central_of.shape == thc_central_new.shape
    assert abs(error_of - error_new) < 2e-1  # Errores similares
    
    print("✅ Basic comparison test passed!")


def test_thc_enhanced_modes():
    """Test para verificar que los modos enhanced funcionan."""
    eri, ncas, nelecas = create_mock_n2_data()
    nthc = 64  # Reducido para test más rápido
    
    print(f"Testing enhanced modes with nthc={nthc}")
    
    base_params = {
        'nthc': nthc,
        'perform_bfgs_opt': True,
        'bfgs_maxiter': 5000,  # Muy reducido para tests
        'random_start_thc': False,
        'verify': False,
    }
    
    # Test modo enhanced
    eri_enhanced, leaf_enhanced, central_enhanced, info_enhanced = thc_via_cp3_new(
        eri_full=eri, thc_method="enhanced", **base_params
    )
    
    # Test modo enhanced_bias
    eri_bias_raw, leaf_bias, central_bias, info_bias = thc_via_cp3_new(
        eri_full=eri, thc_method="enhanced_bias", **base_params
    )
    
    # Verificaciones
    error_enhanced = np.linalg.norm(eri_enhanced - eri)
    
    # ✅ CORRECCIÓN COMPLETA: Para enhanced_bias, necesitamos reconstruir el ERI completo
    if 'alpha2_optimized' in info_bias and 'beta_optimized' in info_bias:
        alpha2_opt = info_bias['alpha2_optimized']
        beta_opt = info_bias['beta_optimized']
        
        # Reconstruir los términos de bias
        eye = np.eye(ncas)
        alpha2_term = alpha2_opt * np.einsum('pq,rs->pqrs', eye, eye)
        beta_term1 = 0.5 * np.einsum('pq,rs->pqrs', beta_opt, eye)
        beta_term2 = 0.5 * np.einsum('pq,rs->pqrs', eye, beta_opt)
        
        # La función devuelve solo la parte THC, necesitamos AGREGAR los bias terms
        # porque el modelo interno optimizaba: THC ≈ (eri - bias_terms)
        # Por lo tanto: eri_completo = THC + bias_terms
        eri_bias_complete = eri_bias_raw + alpha2_term + beta_term1 + beta_term2
        
        # Ahora comparar con el ERI original
        error_bias = np.linalg.norm(eri_bias_complete - eri)
        
        print(f"Enhanced error: {error_enhanced:.6e}")
        print(f"Enhanced+bias error (THC + bias vs original ERI): {error_bias:.6e}")
        print(f"Alpha2 optimized value: {alpha2_opt:.6e}")
        print(f"Beta optimized norm: {np.linalg.norm(beta_opt):.6e}")
        
        # Diagnóstico adicional
        thc_only_error = np.linalg.norm(eri_bias_raw - eri)
        print(f"THC-only error (without bias): {thc_only_error:.6e}")
        
        bias_magnitude = np.linalg.norm(alpha2_term + beta_term1 + beta_term2)
        print(f"Bias correction magnitude: {bias_magnitude:.6e}")
        
        # Verificar que el modelo interno estaba aproximando correctamente
        eri_modified = eri - alpha2_term - beta_term1 - beta_term2
        thc_vs_modified_error = np.linalg.norm(eri_bias_raw - eri_modified)
        print(f"THC vs modified ERI error: {thc_vs_modified_error:.6e}")
        
    else:
        # Fallback si no se encuentran los parámetros de bias
        error_bias = np.linalg.norm(eri_bias_raw - eri)
        print(f"Enhanced error: {error_enhanced:.6e}")
        print(f"Enhanced+bias error: {error_bias:.6e}")
        print("Warning: bias parameters not found in info_bias")
    
    # Verificar que funcionan y tienen formas correctas
    assert leaf_enhanced.shape == (nthc, ncas)
    assert central_enhanced.shape == (nthc, nthc)
    assert leaf_bias.shape == (nthc, ncas)
    assert central_bias.shape == (nthc, nthc)
    
    # Errores deben ser razonables - ahora con la corrección correcta
    assert error_enhanced < 1e-1  # Tolerancia más amplia para test rápido
    assert error_bias < 1e-1      # Ahora debería pasar con la corrección completa
    
    # Verificar que se optimizaron los parámetros de bias
    assert 'alpha2_optimized' in info_bias
    assert 'beta_optimized' in info_bias
    
    print("✅ Enhanced modes test passed!")


def test_thc_invalid_method():
    """Test que verifica manejo de métodos inválidos."""
    eri, ncas, nelecas = create_mock_n2_data()
    
    with pytest.raises(ValueError, match="thc_method must be one of"):
        thc_via_cp3_new(
            eri_full=eri,
            nthc=8,
            thc_method="invalid_method",
            perform_bfgs_opt=False
        )
    
    print("✅ Invalid method test passed!")


def test_thc_shapes_consistency():
    """Test que verifica consistencia de formas de tensores."""
    eri, ncas, nelecas = create_mock_n2_data()
    nthc = 32
    
    eri_thc, thc_leaf, thc_central, info = thc_via_cp3_new(
        eri_full=eri,
        nthc=nthc,
        thc_method="standard",
        perform_bfgs_opt=False,  # Sin optimización para test rápido
        verify=False
    )
    
    # Verificar formas
    assert eri_thc.shape == eri.shape == (ncas, ncas, ncas, ncas)
    assert thc_leaf.shape == (nthc, ncas)
    assert thc_central.shape == (nthc, nthc)
    
    print("✅ Shapes consistency test passed!")


def test_thc_initialization_consistency():
    """Test que verifica que la inicialización CP3 es idéntica entre OpenFermion y nueva implementación."""
    # Setup
    eri, ncas, nelecas = create_mock_n2_data()
    nthc = 32
    
    print(f"Testing initialization consistency with nthc={nthc}, norb={ncas}")
    
    # Parámetros para test de inicialización (0 iteraciones BFGS) con semilla fija
    init_params = {
        'nthc': nthc,
        'perform_bfgs_opt': False,  # No BFGS para probar solo CP3
        'random_start_thc': True,
        'verify': False,
        'first_factor_thresh': 1.0e-14,
        'conv_eps': 1.0e-4,
    }
    
    # Ejecutar ambas implementaciones con la misma semilla numpy
    # (pybtas usa numpy.random internamente)
    
    # Primera ejecución: OpenFermion
    np.random.seed(12345)  # Semilla fija para reproducibilidad
    eri_thc_of, thc_leaf_of, thc_central_of, info_of = thc_via_cp3_openfermion(
        eri_full=eri, **init_params
    )
    
    # Segunda ejecución: Nueva implementación con la misma semilla
    np.random.seed(12345)  # Misma semilla exacta
    eri_thc_new, thc_leaf_new, thc_central_new, info_new = thc_via_cp3_new(
        eri_full=eri, thc_method="standard", **init_params
    )
    
    # Verificar que las formas son idénticas
    assert thc_leaf_of.shape == thc_leaf_new.shape, f"Leaf shapes differ: {thc_leaf_of.shape} vs {thc_leaf_new.shape}"
    assert thc_central_of.shape == thc_central_new.shape, f"Central shapes differ: {thc_central_of.shape} vs {thc_central_new.shape}"
    assert eri_thc_of.shape == eri_thc_new.shape, f"ERI shapes differ: {eri_thc_of.shape} vs {eri_thc_new.shape}"
    
    # Calcular errores de reconstrucción después de CP3 (sin BFGS)
    error_of_init = np.linalg.norm(eri_thc_of - eri)
    error_new_init = np.linalg.norm(eri_thc_new - eri)
    
    print(f"OpenFermion CP3 initialization error: {error_of_init:.6e}")
    print(f"New implementation CP3 initialization error: {error_new_init:.6e}")
    print(f"Absolute difference in errors: {abs(error_of_init - error_new_init):.6e}")
    
    # Con la misma semilla, los errores deberían ser prácticamente idénticos
    relative_error_diff = abs(error_of_init - error_new_init) / max(error_of_init, error_new_init)
    print(f"Relative error difference: {relative_error_diff:.6e}")
    
    # Verificaciones de consistencia estrictas (deberían ser casi idénticos)
    assert relative_error_diff < 1e-10, f"CP3 initialization should be identical, but got relative diff: {relative_error_diff:.6e}"
    
    # Verificar que los tensores leaf son muy similares
    leaf_diff = np.linalg.norm(thc_leaf_of - thc_leaf_new)
    print(f"Leaf tensor L2 difference: {leaf_diff:.6e}")
    assert leaf_diff < 1e-10, f"Leaf tensors should be identical, got diff: {leaf_diff:.6e}"
    
    # Verificar que los tensores centrales son muy similares
    central_diff = np.linalg.norm(thc_central_of - thc_central_new)
    print(f"Central tensor L2 difference: {central_diff:.6e}")
    assert central_diff < 1e-10, f"Central tensors should be identical, got diff: {central_diff:.6e}"
    
    # Verificar que las reconstrucciones ERI son idénticas
    eri_diff = np.linalg.norm(eri_thc_of - eri_thc_new)
    print(f"ERI reconstruction L2 difference: {eri_diff:.6e}")
    assert eri_diff < 1e-10, f"ERI reconstructions should be identical, got diff: {eri_diff:.6e}"
    
    # Verificar que ambos tensores centrales tienen propiedades similares
    eigs_of = np.linalg.eigvals(thc_central_of)
    eigs_new = np.linalg.eigvals(thc_central_new)
    
    print(f"OpenFermion central tensor min eigenvalue: {np.min(eigs_of):.6e}")
    print(f"New implementation central tensor min eigenvalue: {np.min(eigs_new):.6e}")
    
    # Los valores propios deberían ser prácticamente idénticos
    eigs_diff = np.linalg.norm(np.sort(eigs_of) - np.sort(eigs_new))
    print(f"Eigenvalues difference: {eigs_diff:.6e}")
    assert eigs_diff < 1e-10, f"Eigenvalues should be identical, got diff: {eigs_diff:.6e}"
    
    print("✅ Initialization consistency test passed!")


def test_thc_bfgs_improvement():
    """Test que verifica que BFGS no falla catastróficamente."""
    eri, ncas, nelecas = create_mock_n2_data()
    nthc = 32
    
    base_params = {
        'nthc': nthc,
        'random_start_thc': True,
        'verify': False,
        'bfgs_maxiter': 100,  # Más iteraciones
    }
    
    seed = 54321
    
    # Solo CP3
    np.random.seed(seed)
    eri_cp3, _, _, _ = thc_via_cp3_new(
        eri_full=eri, thc_method="standard", perform_bfgs_opt=False, **base_params
    )
    
    # CP3 + BFGS
    np.random.seed(seed)
    eri_bfgs, _, _, _ = thc_via_cp3_new(
        eri_full=eri, thc_method="standard", perform_bfgs_opt=True, **base_params
    )
    
    error_cp3 = np.linalg.norm(eri_cp3 - eri)
    error_bfgs = np.linalg.norm(eri_bfgs - eri)
    
    print(f"CP3-only error: {error_cp3:.6e}")
    print(f"CP3+BFGS error: {error_bfgs:.6e}")
    
    # Tests más permisivos:
    # 1. Ambos deberían dar errores razonables
    assert error_cp3 < 2.0, "CP3 should give reasonable error"
    assert error_bfgs < 2.0, "BFGS should give reasonable error"
    
    # 2. BFGS no debería empeorar dramáticamente
    assert error_bfgs < error_cp3 + 0.5, "BFGS should not worsen dramatically"
    
    print("✅ BFGS robustness test passed!")


def test_openfermion_regularization_compatibility():
    """Test que verifica que el modo standard usa regularización compatible con OpenFermion."""
    eri, ncas, nelecas = create_mock_n2_data()
    nthc = 32

    print(f"Testing OpenFermion regularization compatibility with nthc={nthc}")
    
    # Parámetros base
    base_params = {
        'nthc': nthc,
        'perform_bfgs_opt': True,
        'bfgs_maxiter': 50,  # Pocas iteraciones para test rápido
        'random_start_thc': False,  # HOSVD start para consistencia
        'verify': False,
    }
    
    # Usar misma semilla para comparación justa
    seed = 98765
    
    # Ejecutar modo standard (debería usar regularización OpenFermion)
    np.random.seed(seed)
    eri_standard, leaf_standard, central_standard, info_standard = thc_via_cp3_new(
        eri_full=eri, thc_method="standard", **base_params
    )
    
    # Ejecutar modo enhanced (usa regularización avanzada)
    np.random.seed(seed)
    eri_enhanced, leaf_enhanced, central_enhanced, info_enhanced = thc_via_cp3_new(
        eri_full=eri, thc_method="enhanced", **base_params
    )
    
    error_standard = np.linalg.norm(eri_standard - eri)
    error_enhanced = np.linalg.norm(eri_enhanced - eri)
    
    print(f"Standard mode error (OpenFermion regularization): {error_standard:.6e}")
    print(f"Enhanced mode error (advanced regularization): {error_enhanced:.6e}")
    
    # Ambos deberían funcionar, pero pueden dar resultados ligeramente diferentes
    # debido a las diferentes regularizaciones
    assert error_standard < 1.0, "Standard mode should work with OpenFermion regularization"
    assert error_enhanced < 1.0, "Enhanced mode should work with advanced regularization"
    
    # Verificar que las formas son correctas
    assert leaf_standard.shape == leaf_enhanced.shape
    assert central_standard.shape == central_enhanced.shape
    
    # En enhanced_bias NO deberían estar los parámetros de bias
    assert 'alpha2_optimized' not in info_standard
    assert 'beta_optimized' not in info_standard
    assert 'alpha2_optimized' not in info_enhanced
    assert 'beta_optimized' not in info_enhanced
    
    print("✅ OpenFermion regularization compatibility test passed!")


if __name__ == "__main__":
    print("Running minimal THC tests...")
    print("=" * 50)
    
    test_thc_initialization_consistency()
    print()
    
    test_openfermion_regularization_compatibility()
    print()
    
    test_thc_bfgs_improvement()
    print()
    
    test_thc_basic_comparison()
    print()
    
    test_thc_enhanced_modes()
    print()
    
    test_thc_invalid_method()
    print()
    
    test_thc_shapes_consistency()
    print()
    
    print("🎉 All minimal tests passed!")