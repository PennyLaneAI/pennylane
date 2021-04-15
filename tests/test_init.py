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
Unit tests for the :mod:`pennylane.init` module.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
import numpy as np
import pennylane as qml


#######################################
# Functions and their signatures

# Functions returning a single parameter array
# function name, kwargs and target shape
INIT_KWARGS_SHAPES = [
    (
        qml.init.random_layers_normal,
        {"n_layers": 2, "n_wires": 3, "n_rots": 10, "mean": 0, "std": 1},
        (2, 10),
    ),
    (
        qml.init.random_layers_normal,
        {"n_layers": 2, "n_wires": 1, "n_rots": 10, "mean": 0, "std": 1},
        (2, 10),
    ),
    (
        qml.init.random_layers_normal,
        {"n_layers": 2, "n_wires": 3, "n_rots": None, "mean": 0, "std": 1},
        (2, 3),
    ),
    (
        qml.init.strong_ent_layers_normal,
        {"n_layers": 2, "n_wires": 3, "mean": 0, "std": 1},
        (2, 3, 3),
    ),
    (
        qml.init.strong_ent_layers_normal,
        {"n_layers": 2, "n_wires": 1, "mean": 0, "std": 1},
        (2, 1, 3),
    ),
    (
        qml.init.cvqnn_layers_theta_normal,
        {"n_layers": 2, "n_wires": 3, "mean": 0, "std": 1},
        (2, 3),
    ),
    (
        qml.init.cvqnn_layers_theta_normal,
        {"n_layers": 2, "n_wires": 1, "mean": 0, "std": 1},
        (2, 0),
    ),
    (qml.init.cvqnn_layers_phi_normal, {"n_layers": 2, "n_wires": 3, "mean": 0, "std": 1}, (2, 3)),
    (qml.init.cvqnn_layers_phi_normal, {"n_layers": 2, "n_wires": 1, "mean": 0, "std": 1}, (2, 0)),
    (
        qml.init.cvqnn_layers_varphi_normal,
        {"n_layers": 2, "n_wires": 3, "mean": 0, "std": 1},
        (2, 3),
    ),
    (
        qml.init.cvqnn_layers_varphi_normal,
        {"n_layers": 2, "n_wires": 1, "mean": 0, "std": 1},
        (2, 1),
    ),
    (
        qml.init.cvqnn_layers_r_normal,
        {"n_layers": 2, "n_wires": 3, "mean": 0, "std": 1},
        (2, 3),
    ),
    (
        qml.init.cvqnn_layers_r_normal,
        {"n_layers": 2, "n_wires": 1, "mean": 0, "std": 1},
        (2, 1),
    ),
    (
        qml.init.cvqnn_layers_phi_r_normal,
        {"n_layers": 2, "n_wires": 3, "mean": 0, "std": 1},
        (2, 3),
    ),
    (
        qml.init.cvqnn_layers_phi_r_normal,
        {"n_layers": 2, "n_wires": 1, "mean": 0, "std": 1},
        (2, 1),
    ),
    (
        qml.init.cvqnn_layers_a_normal,
        {"n_layers": 2, "n_wires": 3, "mean": 0, "std": 1},
        (2, 3),
    ),
    (
        qml.init.cvqnn_layers_a_normal,
        {"n_layers": 2, "n_wires": 1, "mean": 0, "std": 1},
        (2, 1),
    ),
    (
        qml.init.cvqnn_layers_phi_a_normal,
        {"n_layers": 2, "n_wires": 3, "mean": 0, "std": 1},
        (2, 3),
    ),
    (
        qml.init.cvqnn_layers_phi_a_normal,
        {"n_layers": 2, "n_wires": 1, "mean": 0, "std": 1},
        (2, 1),
    ),
    (
        qml.init.cvqnn_layers_kappa_normal,
        {"n_layers": 2, "n_wires": 3, "mean": 0, "std": 1},
        (2, 3),
    ),
    (
        qml.init.cvqnn_layers_kappa_normal,
        {"n_layers": 2, "n_wires": 1, "mean": 0, "std": 1},
        (2, 1),
    ),
    (qml.init.interferometer_theta_normal, {"n_wires": 3, "mean": 0, "std": 1}, (3,)),
    (qml.init.interferometer_theta_normal, {"n_wires": 1, "mean": 0, "std": 1}, (0,)),
    (qml.init.interferometer_phi_normal, {"n_wires": 3, "mean": 0, "std": 1}, (3,)),
    (qml.init.interferometer_phi_normal, {"n_wires": 1, "mean": 0, "std": 1}, (0,)),
    (qml.init.interferometer_varphi_normal, {"n_wires": 3, "mean": 0, "std": 1}, (3,)),
    (qml.init.interferometer_varphi_normal, {"n_wires": 1, "mean": 0, "std": 1}, (1,)),
    (
        qml.init.random_layers_uniform,
        {"n_layers": 2, "n_wires": 3, "n_rots": 10, "low": 0, "high": 1},
        (2, 10),
    ),
    (
        qml.init.random_layers_uniform,
        {"n_layers": 2, "n_wires": 3, "n_rots": None, "low": 0, "high": 1},
        (2, 3),
    ),
    (
        qml.init.random_layers_uniform,
        {"n_layers": 2, "n_wires": 1, "n_rots": None, "low": 0, "high": 1},
        (2, 1),
    ),
    (
        qml.init.random_layers_uniform,
        {"n_layers": 2, "n_wires": 1, "n_rots": 10, "low": 0, "high": 1},
        (2, 10),
    ),
    (
        qml.init.strong_ent_layers_uniform,
        {"n_layers": 2, "n_wires": 3, "low": 0, "high": 1},
        (2, 3, 3),
    ),
    (
        qml.init.strong_ent_layers_uniform,
        {"n_layers": 2, "n_wires": 1, "low": 0, "high": 1},
        (2, 1, 3),
    ),
    (
        qml.init.cvqnn_layers_theta_uniform,
        {"n_layers": 2, "n_wires": 3, "low": 0, "high": 1},
        (2, 3),
    ),
    (
        qml.init.cvqnn_layers_theta_uniform,
        {"n_layers": 2, "n_wires": 1, "low": 0, "high": 1},
        (2, 0),
    ),
    (qml.init.cvqnn_layers_phi_uniform, {"n_layers": 2, "n_wires": 3, "low": 0, "high": 1}, (2, 3)),
    (qml.init.cvqnn_layers_phi_uniform, {"n_layers": 2, "n_wires": 1, "low": 0, "high": 1}, (2, 0)),
    (
        qml.init.cvqnn_layers_varphi_uniform,
        {"n_layers": 2, "n_wires": 3, "low": 0, "high": 1},
        (2, 3),
    ),
    (
        qml.init.cvqnn_layers_varphi_uniform,
        {"n_layers": 2, "n_wires": 1, "low": 0, "high": 1},
        (2, 1),
    ),
    (qml.init.cvqnn_layers_r_uniform, {"n_layers": 2, "n_wires": 3, "low": 0, "high": 1}, (2, 3)),
    (qml.init.cvqnn_layers_r_uniform, {"n_layers": 2, "n_wires": 1, "low": 0, "high": 1}, (2, 1)),
    (
        qml.init.cvqnn_layers_phi_r_uniform,
        {"n_layers": 2, "n_wires": 3, "low": 0, "high": 1},
        (2, 3),
    ),
    (
        qml.init.cvqnn_layers_phi_r_uniform,
        {"n_layers": 2, "n_wires": 1, "low": 0, "high": 1},
        (2, 1),
    ),
    (qml.init.cvqnn_layers_a_uniform, {"n_layers": 2, "n_wires": 3, "low": 0, "high": 1}, (2, 3)),
    (qml.init.cvqnn_layers_a_uniform, {"n_layers": 2, "n_wires": 1, "low": 0, "high": 1}, (2, 1)),
    (
        qml.init.cvqnn_layers_phi_a_uniform,
        {"n_layers": 2, "n_wires": 3, "low": 0, "high": 1},
        (2, 3),
    ),
    (
        qml.init.cvqnn_layers_phi_a_uniform,
        {"n_layers": 2, "n_wires": 1, "low": 0, "high": 1},
        (2, 1),
    ),
    (
        qml.init.cvqnn_layers_kappa_uniform,
        {"n_layers": 2, "n_wires": 3, "low": 0, "high": 1},
        (2, 3),
    ),
    (
        qml.init.cvqnn_layers_kappa_uniform,
        {"n_layers": 2, "n_wires": 1, "low": 0, "high": 1},
        (2, 1),
    ),
    (qml.init.interferometer_theta_uniform, {"n_wires": 3, "low": 0, "high": 1}, (3,)),
    (qml.init.interferometer_theta_uniform, {"n_wires": 1, "low": 0, "high": 1}, (0,)),
    (qml.init.interferometer_phi_uniform, {"n_wires": 3, "low": 0, "high": 1}, (3,)),
    (qml.init.interferometer_phi_uniform, {"n_wires": 1, "low": 0, "high": 1}, (0,)),
    (qml.init.interferometer_varphi_uniform, {"n_wires": 3, "low": 0, "high": 1}, (3,)),
    (qml.init.interferometer_varphi_uniform, {"n_wires": 1, "low": 0, "high": 1}, (1,)),
    (
        qml.init.qaoa_embedding_normal,
        {"n_layers": 2, "n_wires": 3, "mean": 0, "std": 1},
        (2, 2 * 3),
    ),
    (
        qml.init.qaoa_embedding_uniform,
        {"n_layers": 2, "n_wires": 3, "low": 0, "high": 1},
        (2, 2 * 3),
    ),
    (qml.init.qaoa_embedding_uniform, {"n_layers": 2, "n_wires": 1, "low": 0, "high": 1}, (2, 1)),
    (qml.init.qaoa_embedding_uniform, {"n_layers": 2, "n_wires": 2, "low": 0, "high": 1}, (2, 3)),
    (qml.init.qaoa_embedding_normal, {"n_layers": 2, "n_wires": 1, "mean": 0, "std": 1}, (2, 1)),
    (qml.init.qaoa_embedding_normal, {"n_layers": 2, "n_wires": 2, "mean": 0, "std": 1}, (2, 3)),
    (
        qml.init.simplified_two_design_initial_layer_uniform,
        {"n_wires": 1, "low": 0, "high": 1},
        (1,),
    ),
    (
        qml.init.simplified_two_design_initial_layer_uniform,
        {"n_wires": 3, "low": 0, "high": 1},
        (3,),
    ),
    (
        qml.init.simplified_two_design_initial_layer_normal,
        {"n_wires": 1, "mean": 0, "std": 1},
        (1,),
    ),
    (
        qml.init.simplified_two_design_initial_layer_normal,
        {"n_wires": 3, "mean": 0, "std": 1},
        (3,),
    ),
    (
        qml.init.simplified_two_design_weights_uniform,
        {"n_layers": 2, "n_wires": 1, "low": 0, "high": 1},
        (0,),
    ),
    (
        qml.init.simplified_two_design_weights_uniform,
        {"n_layers": 2, "n_wires": 2, "low": 0, "high": 1},
        (2, 1, 2),
    ),
    (
        qml.init.simplified_two_design_weights_uniform,
        {"n_layers": 2, "n_wires": 4, "low": 0, "high": 1},
        (2, 3, 2),
    ),
    (
        qml.init.simplified_two_design_weights_normal,
        {"n_layers": 2, "n_wires": 1, "mean": 0, "std": 1},
        (0,),
    ),
    (
        qml.init.simplified_two_design_weights_normal,
        {"n_layers": 2, "n_wires": 2, "mean": 0, "std": 1},
        (2, 1, 2),
    ),
    (
        qml.init.simplified_two_design_weights_normal,
        {"n_layers": 2, "n_wires": 4, "mean": 0, "std": 1},
        (2, 3, 2),
    ),
    (
        qml.init.basic_entangler_layers_normal,
        {"n_layers": 2, "n_wires": 1, "mean": 0, "std": 1},
        (2, 1),
    ),
    (
        qml.init.basic_entangler_layers_normal,
        {"n_layers": 2, "n_wires": 2, "mean": 0, "std": 1},
        (2, 2),
    ),
    (
        qml.init.basic_entangler_layers_uniform,
        {"n_layers": 2, "n_wires": 1, "low": 0, "high": 1},
        (2, 1),
    ),
    (
        qml.init.basic_entangler_layers_uniform,
        {"n_layers": 2, "n_wires": 2, "low": 0, "high": 1},
        (2, 2),
    ),
]
# Functions returning a list of parameter arrays
INITALL_KWARGS_SHAPES = [
    (qml.init.cvqnn_layers_all, {"n_layers": 2, "n_wires": 3}, [(2, 3)] * 11),
    (qml.init.interferometer_all, {"n_wires": 3}, [(3,), (3,), (3,)]),
]

# Without target shapes
INIT_KWARGS = [i[0:2] for i in INIT_KWARGS_SHAPES]

#################


class TestInit:
    """Tests the initialization functions from the ``init`` module."""

    @pytest.mark.parametrize("init, sgntr, shp", INIT_KWARGS_SHAPES)
    def test_shape(self, init, sgntr, shp, seed):
        """Confirm that initialization functions
        return an array with the correct shape."""
        s = {**sgntr, "seed": seed}
        p = init(**s)
        assert p.shape == shp

    @pytest.mark.parametrize("init, sgntr, shp", INITALL_KWARGS_SHAPES)
    def test_all_shape(self, init, sgntr, shp, seed):
        """Confirm that ``all`` initialization functions
        return an array with the correct shape."""

        s = {**sgntr, "seed": seed}
        p = init(**s)
        shapes = [p_.shape for p_ in p]
        assert shapes == shp

    @pytest.mark.parametrize("init, sgntr", INIT_KWARGS)
    def test_same_output_for_same_seed(self, init, sgntr, seed, tol):
        """Confirm that initialization functions return a deterministic output
        for a fixed seed."""

        # exclude case of empty parameter list
        if len(init(**sgntr).flatten()) == 0:
            pytest.skip("test is skipped for empty parameter array")

        s = {**sgntr, "seed": seed}
        p1 = init(**s)
        p2 = init(**s)
        assert np.allclose(p1, p2, atol=tol)

    @pytest.mark.parametrize("init, sgntr", INIT_KWARGS)
    def test_diff_output_for_diff_seed(self, init, sgntr, seed, tol):
        """Confirm that initialization function returns a different output for
        different seeds."""

        # exclude case of empty parameter list
        if len(init(**sgntr).flatten()) == 0:
            pytest.skip("test is skipped for empty parameter array")

        s = {**sgntr, "seed": seed}
        p1 = init(**s)
        s = {**s, "seed": seed + 1}
        p2 = init(**s)

        if p1.shape != (0,):
            assert not np.allclose(p1, p2, atol=tol)

    @pytest.mark.parametrize("init, sgntr", INIT_KWARGS)
    def test_interval(self, init, sgntr, seed, tol):
        """Test that sampled parameters lie in correct interval."""

        # exclude case of empty parameter list
        if len(init(**sgntr).flatten()) == 0:
            pytest.skip("test is skipped for empty parameter array")

        s = {**sgntr, "seed": seed}

        # Case A: Uniformly distributed parameters
        if "low" in s.keys() and "high" in s.keys():
            s["low"] = 1
            s["high"] = 1
            p = init(**s)
            p_mean = np.mean(p)
            assert np.isclose(p_mean, 1, atol=tol)

        # Case B: Normally distributed parameters
        if "mean" in s.keys() and "std" in s.keys():
            s["mean"] = 1
            s["std"] = 0

        p = init(**s)
        p_mean = np.mean(p)
        assert np.isclose(p_mean, 1, atol=tol)

    @pytest.mark.parametrize("init, sgntr", INIT_KWARGS)
    def test_zero_wires(self, init, sgntr):
        """Test that edge case of zero wires returns empty parameter array."""

        if "n_wires" in sgntr:
            sgntr["n_wires"] = 0

        p = init(**sgntr)

        assert p.flatten().shape == (0,)

    def test_particle_conserving_u2_init(self, tol):
        """Test the functions 'particle_conserving_u2_uniform' and
        'particle_conserving_u2_normal'."""

        n_layers = 2
        n_wires = 4

        # check the shape
        exp_shape = (n_layers, 2 * n_wires - 1)
        params = qml.init.particle_conserving_u2_uniform(n_layers, n_wires)
        assert params.shape == exp_shape

        params = qml.init.particle_conserving_u2_normal(n_layers, n_wires)
        assert params.shape == exp_shape

        # check deterministic output for a fixed seed
        seed = 1975
        p1 = qml.init.particle_conserving_u2_uniform(n_layers, n_wires, seed=seed)
        p2 = qml.init.particle_conserving_u2_uniform(n_layers, n_wires, seed=seed)
        assert np.allclose(p1, p2, atol=tol)

        p1 = qml.init.particle_conserving_u2_normal(n_layers, n_wires, seed=seed)
        p2 = qml.init.particle_conserving_u2_normal(n_layers, n_wires, seed=seed)
        assert np.allclose(p1, p2, atol=tol)

        # check that the output is different for different seeds
        p1 = qml.init.particle_conserving_u2_uniform(n_layers, n_wires, seed=seed)
        p2 = qml.init.particle_conserving_u2_uniform(n_layers, n_wires, seed=seed + 1)
        assert not np.allclose(p1, p2, atol=tol)

        p1 = qml.init.particle_conserving_u2_normal(n_layers, n_wires, seed=seed)
        p2 = qml.init.particle_conserving_u2_normal(n_layers, n_wires, seed=seed + 1)
        assert not np.allclose(p1, p2, atol=tol)

    def test_particle_conserving_u2_init_exceptions(self):
        """Test exceptions the functions 'particle_conserving_u2_uniform' and
        'particle_conserving_u2_normal'."""

        n_layers = 4
        n_wires = 1

        msg_match = "The number of qubits must be greater than one"

        with pytest.raises(ValueError, match=msg_match):
            qml.init.particle_conserving_u2_uniform(n_layers, n_wires)

        with pytest.raises(ValueError, match=msg_match):
            qml.init.particle_conserving_u2_normal(n_layers, n_wires)

    def test_particle_conserving_u1_init(self, tol):
        """Test the functions 'particle_conserving_u1_uniform' and
        'particle_conserving_u1_normal'."""

        n_layers = 2
        n_wires = 4

        # check the shape
        exp_shape = (n_layers, n_wires - 1, 2)
        params = qml.init.particle_conserving_u1_uniform(n_layers, n_wires)
        assert params.shape == exp_shape

        params = qml.init.particle_conserving_u1_normal(n_layers, n_wires)
        assert params.shape == exp_shape

        # check deterministic output for a fixed seed
        seed = 1975
        p1 = qml.init.particle_conserving_u1_uniform(n_layers, n_wires, seed=seed)
        p2 = qml.init.particle_conserving_u1_uniform(n_layers, n_wires, seed=seed)
        assert np.allclose(p1, p2, atol=tol)

        p1 = qml.init.particle_conserving_u1_normal(n_layers, n_wires, seed=seed)
        p2 = qml.init.particle_conserving_u1_normal(n_layers, n_wires, seed=seed)
        assert np.allclose(p1, p2, atol=tol)

        # check that the output is different for different seeds
        p1 = qml.init.particle_conserving_u1_uniform(n_layers, n_wires, seed=seed)
        p2 = qml.init.particle_conserving_u1_uniform(n_layers, n_wires, seed=seed + 1)
        assert not np.allclose(p1, p2, atol=tol)

        p1 = qml.init.particle_conserving_u1_normal(n_layers, n_wires, seed=seed)
        p2 = qml.init.particle_conserving_u1_normal(n_layers, n_wires, seed=seed + 1)
        assert not np.allclose(p1, p2, atol=tol)

    def test_particle_conserving_u1_init_exceptions(self):
        """Test exceptions the functions 'particle_conserving_u1_uniform' and
        'particle_conserving_u1_normal'."""

        n_layers = 4
        n_wires = 1

        msg_match = "The number of qubits must be greater than one"

        with pytest.raises(ValueError, match=msg_match):
            qml.init.particle_conserving_u1_uniform(n_layers, n_wires)

        with pytest.raises(ValueError, match=msg_match):
            qml.init.particle_conserving_u1_normal(n_layers, n_wires)
