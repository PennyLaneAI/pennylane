# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
n_rotations = 10
n_if = n_wires * (n_wires - 1) / 2

#######################################
# Common keyword arguments

base = {'n_wires': n_wires}
rpt = {'n_layers': repeat, 'n_wires': n_wires}
rpt_nrml = {'n_layers': repeat, 'n_wires': n_wires, 'mean': 0, 'std': 1}
rpt_uni = {'n_layers': repeat, 'n_wires': n_wires, 'low': 0, 'high': 1}
base_nrml = {'n_wires': n_wires, 'mean': 0, 'std': 1}
base_uni = {'n_wires': n_wires, 'low': 0, 'high': 1}
rnd_rpt_nrml1 = {'n_layers': repeat, 'n_wires': n_wires, 'n_rots': n_rotations, 'mean': 0, 'std': 1}
rnd_rpt_uni1 = {'n_layers': repeat, 'n_wires': n_wires, 'n_rots': n_rotations, 'low': 0, 'high': 1}
rnd_rpt_nrml2 = {'n_layers': repeat, 'n_wires': n_wires, 'n_rots': None, 'mean': 0, 'std': 1}
rnd_rpt_uni2 = {'n_layers': repeat, 'n_wires': n_wires, 'n_rots': None, 'low': 0, 'high': 1}

#######################################
# Functions and their signatures

# Functions returning a single parameter array
INIT_KWARGS_SHAPES = [(qml.init.random_layers_normal, rnd_rpt_nrml1, (repeat, n_rotations)),
                      (qml.init.random_layers_normal, rnd_rpt_nrml2, (repeat, n_wires)),
                      (qml.init.strong_ent_layers_normal, rpt_nrml, (repeat, n_wires, 3)),
                      (qml.init.cvqnn_layers_theta_normal, rpt_nrml, (repeat, n_if)),
                      (qml.init.cvqnn_layers_phi_normal, rpt_nrml, (repeat, n_if)),
                      (qml.init.cvqnn_layers_varphi_normal, rpt_nrml, (repeat, n_wires),),
                      (qml.init.cvqnn_layers_r_normal, rpt_nrml, (repeat, n_wires),),
                      (qml.init.cvqnn_layers_phi_r_normal, rpt_nrml, (repeat, n_wires),),
                      (qml.init.cvqnn_layers_a_normal, rpt_nrml, (repeat, n_wires),),
                      (qml.init.cvqnn_layers_phi_a_normal, rpt_nrml, (repeat, n_wires),),
                      (qml.init.cvqnn_layers_kappa_normal, rpt_nrml, (repeat, n_wires),),
                      (qml.init.interferometer_theta_normal, base_nrml, (n_if,)),
                      (qml.init.interferometer_phi_normal, base_nrml, (n_if,)),
                      (qml.init.interferometer_varphi_normal, base_nrml, (n_wires,)),
                      (qml.init.random_layers_uniform, rnd_rpt_uni1, (repeat, n_rotations)),
                      (qml.init.random_layers_uniform, rnd_rpt_uni2, (repeat, n_wires)),
                      (qml.init.strong_ent_layers_uniform, rpt_uni, (repeat, n_wires, 3)),
                      (qml.init.cvqnn_layers_theta_uniform, rpt_uni, (repeat, n_if)),
                      (qml.init.cvqnn_layers_phi_uniform, rpt_uni, (repeat, n_if)),
                      (qml.init.cvqnn_layers_varphi_uniform, rpt_uni, (repeat, n_wires)),
                      (qml.init.cvqnn_layers_r_uniform, rpt_uni, (repeat, n_wires)),
                      (qml.init.cvqnn_layers_phi_r_uniform, rpt_uni, (repeat, n_wires)),
                      (qml.init.cvqnn_layers_a_uniform, rpt_uni, (repeat, n_wires)),
                      (qml.init.cvqnn_layers_phi_a_uniform, rpt_uni, (repeat, n_wires)),
                      (qml.init.cvqnn_layers_kappa_uniform, rpt_uni, (repeat, n_wires)),
                      (qml.init.interferometer_theta_uniform, base_uni, (n_if,)),
                      (qml.init.interferometer_phi_uniform, base_uni, (n_if,)),
                      (qml.init.interferometer_varphi_uniform, base_uni, (n_wires,)),
                      (qml.init.qaoa_embedding_normal, rpt_nrml, (repeat, 2*n_wires)),
                      (qml.init.qaoa_embedding_uniform, rpt_uni, (repeat, 2*n_wires)),
                      (qml.init.qaoa_embedding_uniform, {'n_layers': 2, 'n_wires': 1, 'low': 0, 'high': 1}, (2, 1)),
                      (qml.init.qaoa_embedding_uniform, {'n_layers': 2, 'n_wires': 2, 'low': 0, 'high': 1}, (2, 3)),
                      (qml.init.qaoa_embedding_normal, {'n_layers': 2, 'n_wires': 1, 'mean': 0, 'std': 1}, (2, 1)),
                      (qml.init.qaoa_embedding_normal, {'n_layers': 2, 'n_wires': 2, 'mean': 0, 'std': 1}, (2, 3)),
                      ]
# Functions returning a list of parameter arrays
INITALL_KWARGS_SHAPES = [(qml.init.cvqnn_layers_all, rpt,
                          [(repeat, n_if), (repeat, n_if), (repeat, n_wires), (repeat, n_wires), (repeat, n_wires),
                           (repeat, n_if), (repeat, n_if), (repeat, n_wires), (repeat, n_wires), (repeat, n_wires),
                           (repeat, n_wires)]),
                         (qml.init.interferometer_all, base, [(n_if,), (n_if,), (n_wires,)])]

# Without shapes
INIT_KWARGS = [i[0:2] for i in INIT_KWARGS_SHAPES]
INITALL_KWARGS = [i[0:2] for i in INITALL_KWARGS_SHAPES]

#################


class TestInit:
    """Tests the initialization functions from the ``init`` module."""

    @pytest.mark.parametrize("init, sgntr, shp", INIT_KWARGS_SHAPES)
    def test_shape(self, init, sgntr, shp, seed):
        """Confirm that initialization functions
         return an array with the correct shape."""
        s = {**sgntr, 'seed': seed}
        p = init(**s)
        assert p.shape == shp

    @pytest.mark.parametrize("init, sgntr, shp", INITALL_KWARGS_SHAPES)
    def test_all_shape(self, init, sgntr, shp, seed):
        """Confirm that ``all`` initialization functions
         return an array with the correct shape."""
        s = {**sgntr, 'seed': seed}
        p = init(**s)
        shapes = [p_.shape for p_ in p]
        assert shapes == shp

    @pytest.mark.parametrize("init, sgntr", INIT_KWARGS)
    def test_same_output_for_same_seed(self, init, sgntr, seed, tol):
        """Confirm that initialization functions return a deterministic output
        for a fixed seed."""
        s = {**sgntr, 'seed': seed}
        p1 = init(**s)
        p2 = init(**s)
        assert np.allclose(p1, p2, atol=tol)

    @pytest.mark.parametrize("init, sgntr", INIT_KWARGS)
    def test_diff_output_for_diff_seed(self, init, sgntr, seed, tol):
        """Confirm that initialization function returns a different output for
        different seeds."""
        s = {**sgntr, 'seed': seed}
        p1 = init(**s)
        s = {**s, 'seed': seed + 1}
        p2 = init(**s)
        assert not np.allclose(p1, p2, atol=tol)

    @pytest.mark.parametrize("init, sgntr", INIT_KWARGS)
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
