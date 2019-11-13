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

# Fix inputs to create fixture parameters
n_wires = 3
repeat = 2
n_rots = 10
n_if = n_wires * (n_wires - 1) / 2
wire_block = (repeat, n_wires)
intf_block = (repeat, n_if)

#######################################
# Signatures

base = {'n_wires': n_wires}
rpt = {'n_layers': repeat, 'n_wires': n_wires}
rpt_nrml = {'n_layers': repeat, 'n_wires': n_wires, 'mean': 0, 'std': 1}
rpt_uni = {'n_layers': repeat, 'n_wires': n_wires, 'low': 0, 'high': 1}
base_nrml = {'n_wires': n_wires, 'mean': 0, 'std': 1}
base_uni = {'n_wires': n_wires, 'low': 0, 'high': 1}
rnd_rpt_nrml = {'n_layers': repeat, 'n_wires': n_wires, 'n_rots': n_rots, 'mean': 0, 'std': 1}
rnd_rpt_uni = {'n_layers': repeat, 'n_wires': n_wires, 'n_rots': n_rots, 'low': 0, 'high': 1}

#######################################
# Functions and their signatures

# Functions returning a single parameter array
INIT = [(qml.init.random_layers_normal, rnd_rpt_nrml),
        (qml.init.strong_ent_layers_normal, rpt_nrml),
        (qml.init.cvqnn_layers_theta_normal, rpt_nrml),
        (qml.init.cvqnn_layers_phi_normal, rpt_nrml),
        (qml.init.cvqnn_layers_varphi_normal, rpt_nrml),
        (qml.init.cvqnn_layers_r_normal, rpt_nrml),
        (qml.init.cvqnn_layers_phi_r_normal, rpt_nrml),
        (qml.init.cvqnn_layers_a_normal, rpt_nrml),
        (qml.init.cvqnn_layers_phi_a_normal, rpt_nrml),
        (qml.init.cvqnn_layers_kappa_normal, rpt_nrml),
        (qml.init.interferometer_theta_normal, base_nrml),
        (qml.init.interferometer_phi_normal, base_nrml),
        (qml.init.interferometer_varphi_normal, base_nrml),
        (qml.init.random_layers_uniform, rnd_rpt_uni),
        (qml.init.strong_ent_layers_uniform, rpt_uni),
        (qml.init.cvqnn_layers_theta_uniform, rpt_uni),
        (qml.init.cvqnn_layers_phi_uniform, rpt_uni),
        (qml.init.cvqnn_layers_varphi_uniform, rpt_uni),
        (qml.init.cvqnn_layers_r_uniform, rpt_uni),
        (qml.init.cvqnn_layers_phi_r_uniform, rpt_uni),
        (qml.init.cvqnn_layers_a_uniform, rpt_uni),
        (qml.init.cvqnn_layers_phi_a_uniform, rpt_uni),
        (qml.init.cvqnn_layers_kappa_uniform, rpt_uni),
        (qml.init.interferometer_theta_uniform, base_uni),
        (qml.init.interferometer_phi_uniform, base_uni),
        (qml.init.interferometer_varphi_uniform, base_uni)
        ]
# Functions returning a list of parameter arrays
INITALL = [(qml.init.cvqnn_layers_all, rpt),
           (qml.init.interferometer_all, base)]

###########
# Shapes

SHAPE = [(repeat, n_rots), (repeat, n_wires, 3), intf_block, intf_block, wire_block, wire_block,
         wire_block, wire_block, wire_block, wire_block, (n_if,), (n_if,), (n_wires,),
         (repeat, n_rots), (repeat, n_wires, 3), intf_block, intf_block, wire_block, wire_block,
         wire_block, wire_block, wire_block, wire_block, (n_if,), (n_if,), (n_wires,),
         ]
ALL_SHAPE = [[intf_block, intf_block, wire_block, wire_block, wire_block, intf_block, intf_block, wire_block, wire_block, wire_block, wire_block],
             [(n_if,), (n_if,), (n_wires,)]]

###############
# Combinations

INIT_SHAPES = [i + (s,) for i, s in zip(INIT, SHAPE)]
INIT_ALL_SHAPES = [i + (s,) for i, s in zip(INITALL, ALL_SHAPE)]

##################


class TestInitRepeated:
    """Tests the initialization functions from the ``init`` module."""

    @pytest.mark.parametrize("init, sgntr, shp", INIT_SHAPES)
    def test_shape(self, init, sgntr, shp, seed):
        """Confirm that initialization functions
         return an array with the correct shape."""
        s = {**sgntr, 'seed': seed}
        p = init(**s)
        assert p.shape == shp

    @pytest.mark.parametrize("init, sgntr, shp", INIT_ALL_SHAPES)
    def test_all_shape(self, init, sgntr, shp, seed):
        """Confirm that ``all`` initialization functions
         return an array with the correct shape."""
        s = {**sgntr, 'seed': seed}
        p = init(**s)
        shapes = [p_.shape for p_ in p]
        assert shapes == shp

    @pytest.mark.parametrize("init, sgntr", INIT)
    def test_same_output_for_same_seed(self, init, sgntr, seed, tol):
        """Confirm that initialization functions return a deterministic output
        for a fixed seed."""
        s = {**sgntr, 'seed': seed}
        p1 = init(**s)
        p2 = init(**s)
        assert np.allclose(p1, p2, atol=tol)

    @pytest.mark.parametrize("init, sgntr", INIT)
    def test_diff_output_for_diff_seed(self, init, sgntr, seed, tol):
        """Confirm that initialization function returns a different output for
        different seeds."""
        s = {**sgntr, 'seed': seed}
        p1 = init(**s)
        s = {**s, 'seed': seed + 1}
        p2 = init(**s)
        assert not np.allclose(p1, p2, atol=tol)

    @pytest.mark.parametrize("init, sgntr", INIT)
    def test_interval(self, init, sgntr, seed, tol):
        """Test that sampled parameters lie in correct interval."""
        s = {**sgntr, 'seed': seed}

        # Case A: Uniformly distributed parameters
        if 'low' in s.keys() and 'high' in s.keys():
            s['low'] = 1
            s['high'] = 1
            p = init(**s)
            p_mean = np.mean(p)
            assert np.isclose(p_mean, 1, atol=tol)

        # Case B: Normally distributed parameters
        if 'mean' in s.keys() and 'std' in s.keys():
            s['mean'] = 1
            s['std'] = 0

        p = init(**s)
        p_mean = np.mean(p)
        assert np.isclose(p_mean, 1, atol=tol)

