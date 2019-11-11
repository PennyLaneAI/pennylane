# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the :mod:`pennylane.templates.parameters` module.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
import numpy as np
import pennylane as qml

@pytest.fixture(scope="module",
                params=[2, 3])
def n_wires(request):
    """Number of qubits or modes."""
    return request.param

@pytest.fixture(scope="module",
                params=[2, 3])
def repeat(request):
    """Number of layers."""
    return request.param

@pytest.fixture(scope="module",
                params=[2, 10])
def n_rots(request):
    """Number of rotations in random layer."""
    return request.param


@pytest.fixture(scope="module")
def n_if(n_wires):
    """Number of beamsplitters in an interferometer."""
    return n_wires * (n_wires - 1) // 2


a = (repeat, n_wires)
b = (repeat, n_if)

############
# Functions
# RPT - have a 'repeat' keyword argument

RPT_NRML = [qml.init.random_layers_normal,
            qml.init.strong_ent_layers_normal,
            qml.init.cvqnn_layers_theta_normal,
            qml.init.cvqnn_layers_phi_normal,
            qml.init.cvqnn_layers_varphi_normal,
            qml.init.cvqnn_layers_r_normal,
            qml.init.cvqnn_layers_phi_r_normal,
            qml.init.cvqnn_layers_a_normal,
            qml.init.cvqnn_layers_phi_a_normal,
            qml.init.cvqnn_layers_kappa_normal,
            ]
RPT_UNIF = [qml.init.random_layers_uniform,
            qml.init.strong_ent_layers_uniform,
            qml.init.cvqnn_layers_theta_uniform,
            qml.init.cvqnn_layers_phi_uniform,
            qml.init.cvqnn_layers_varphi_uniform,
            qml.init.cvqnn_layers_r_uniform,
            qml.init.cvqnn_layers_phi_r_uniform,
            qml.init.cvqnn_layers_a_uniform,
            qml.init.cvqnn_layers_phi_a_uniform,
            qml.init.cvqnn_layers_kappa_uniform,
            ]
RPT_ALL = [qml.init.cvqnn_layers_all]
NRML = [qml.init.interferometer_theta_normal,
        qml.init.interferometer_phi_normal,
        qml.init.interferometer_varphi_normal]
UNIF = [qml.init.interferometer_theta_uniform,
        qml.init.interferometer_phi_uniform,
        qml.init.interferometer_varphi_uniform]
ALL = [qml.init.interferometer_all]

###########
# Shapes

RPT_SHAPES = [(repeat, n_rots), (repeat, n_wires, 3), b, b, a, a, a, a, a, a]
SHAPES = [b, b, a]
RPT_ALL_SHAPES = [[b, b, a, a, a, b, b, a, a, a, a]]
ALL_SHAPES = [[b, b, a]]

###############
# Combinations

RPT_UNIF_SHP = list(zip(RPT_UNIF, RPT_SHAPES))
RPT_NRML_SHP = list(zip(RPT_NRML, RPT_SHAPES))
UNIF_SHP = list(zip(RPT_UNIF, SHAPES))
NRML_SHP = list(zip(NRML, SHAPES))
RPT_ALL_SHP = list(zip(RPT_ALL, RPT_ALL_SHAPES))
ALL_SHP = list(zip(ALL, ALL_SHAPES))
RPT_SHP = RPT_UNIF_SHP + RPT_NRML_SHP
SHP = UNIF_SHP + NRML_SHP
RPT_COMBINED = RPT_UNIF + RPT_NRML + RPT_ALL
COMBINED = UNIF + NRML + ALL


class TestInitRepeated:
    """Tests the initialization functions from the ``init`` module for
    functions that have a ``repeat`` keyword argument."""

    @pytest.mark.parametrize("init, shp", RPT_SHP)
    def test_rpt_shape(self, init, shp, n_wires, repeat):
        """Confirm that initialization functions with keyword ``repeat``
         return an array with the correct shape."""
        p = init(n_wires=n_wires, n_layers=repeat, seed=0)
        assert p.shape == shp

    @pytest.mark.parametrize("init, shp", RPT_ALL_SHP)
    def test_all_rpt_shape(self, init, shp, n_wires, repeat):
        """Confirm that ``all`` initialization functions
        which have a ``repeat`` argument
         return an array with the correct shape."""
        p = init(n_wires=n_wires, n_layers=repeat, seed=0)
        shapes = [p_.shape for p_ in p]
        assert shapes == shp

    @pytest.mark.parametrize("init", RPT_COMBINED)
    def test_rpt_same_output_for_same_seed(self, init, n_wires, repeat, seed, tol):
        """Confirm that initialization function returns a deterministic output
        for a fixed seed."""
        p1 = init(n_layers=repeat, n_wires=n_wires, seed=seed)
        p2 = init(n_layers=repeat, n_wires=n_wires, seed=seed)
        assert np.allclose(p1, p2, atol=tol)

    @pytest.mark.parametrize("init", RPT_COMBINED)
    def test_rpt_diff_output_for_diff_seed(self, init, n_wires, repeat, seed, tol):
        """Confirm that initialization function returns a different output for
        different seeds."""
        p1 = init(n_layers=repeat, n_wires=n_wires, seed=seed)
        p2 = init(n_layers=repeat, n_wires=n_wires, seed=seed+1)
        assert not np.allclose(p1, p2, atol=tol)

    @pytest.mark.parametrize("init", RPT_UNIF)
    def test_rpt_interval_uniform(self, init, seed, n_wires, tol):
        """Test that uniformly sampled initialization functions lie
        in correct interval."""
        p = init(n_layers=repeat, n_wires=n_wires, low=1, high=1, seed=seed)
        p_mean = np.mean(p)
        assert np.isclose(p_mean, 1, atol=tol)

    @pytest.mark.parametrize("init", RPT_NRML)
    def test_rpt_interval_normal(self, init, seed, n_wires, tol):
        """Test that normal samples of initialization functions lie
        in correct interval."""
        p = init(n_layers=repeat, n_wires=n_wires, mean=1, std=0, seed=seed)
        p_mean = np.mean(p)
        assert np.isclose(p_mean, 1, atol=tol)


class TestInitNotRepeated:
    """Tests the initialization functions from the ``init`` module for
    functions that have no ``repeat`` keyword argument."""

    @pytest.mark.parametrize("init, shp", SHP)
    def test_shape(self, init, shp, n_wires):
        """Confirm that initialization functions without ``repeat``
         returns an array with the correct shape."""
        p = init(n_wires=n_wires, seed=0)
        assert p.shape == shp

    @pytest.mark.parametrize("init, shp", ALL_SHP)
    def test_all_shape(self, init, shp, n_wires):
        """Confirm that ``all`` initialization functions
         return an array with the correct shape."""
        p = init(n_wires=n_wires, seed=0)
        shapes = [p_.shape for p_ in p]
        assert shapes == shp

    @pytest.mark.parametrize("init", COMBINED)
    def test_same_output_for_same_seed(self, init, n_wires, seed, tol):
        """Confirm that initialization function returns a deterministic output
        for a fixed seed."""
        p1 = init(n_wires=n_wires, seed=seed)
        p2 = init(n_wires=n_wires, seed=seed)
        assert np.allclose(p1, p2, atol=tol)

    @pytest.mark.parametrize("init", COMBINED)
    def test_diff_output_for_diff_seed(self, init, n_wires, seed, tol):
        """Confirm that initialization function returns a different output for
        different seeds."""
        p1 = init(n_wires=n_wires, seed=seed)
        p2 = init(n_wires=n_wires, seed=seed + 1)
        assert not np.allclose(p1, p2, atol=tol)

    @pytest.mark.parametrize("init", UNIF)
    def test_interval_uniform(self, init, seed, n_wires, tol):
        """Test that uniformly sampled initialization functions lie
        in correct interval."""
        p = init(n_wires=n_wires, low=1, high=1, seed=seed)
        p_mean = np.mean(p)
        assert np.isclose(p_mean, 1, atol=tol)

    @pytest.mark.parametrize("init", NRML)
    def test_interval_normal(self, init, seed, n_wires, tol):
        """Test that normal samples of initialization functions lie
        in correct interval."""
        p = init(n_wires=n_wires, mean=1, std=0, seed=seed)
        p_mean = np.mean(p)
        assert np.isclose(p_mean, 1, atol=tol)
