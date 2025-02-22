import numpy as np
from jax import numpy as jnp
import jax
import itertools as it

from pennylane.qchem_jax.matrices import repulsion_tensor
import pennylane as qml

import pytest


@pytest.mark.parametrize(
        ("symbols", "geometry", "alpha", "e_ref"),
        [
            (
                ["H", "H"],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                np.array(
                    [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]]
                ),
                # electron repulsion tensor obtained from pyscf with mol.intor('int2e')
                np.array(
                    [
                        [
                            [[0.77460594, 0.56886157], [0.56886157, 0.65017755]],
                            [[0.56886157, 0.45590169], [0.45590169, 0.56886157]],
                        ],
                        [
                            [[0.56886157, 0.45590169], [0.45590169, 0.56886157]],
                            [[0.65017755, 0.56886157], [0.56886157, 0.77460594]],
                        ],
                    ]
                ),
            )
        ],
    )
def test_repulsion_tensor(symbols, geometry, alpha, e_ref):
    r"""Test that repulsion_tensor returns the correct matrix."""
    mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    e = repulsion_tensor(mol.basis_set)
    assert np.allclose(e, e_ref)