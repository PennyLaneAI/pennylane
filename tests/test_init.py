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
<<<<<<< HEAD
from pennylane.init import (cvqnn_layers_all,
                            interferometer_all,
=======
from pennylane.init import (cvqnn_layers_uniform,
                            cvqnn_layers_normal,
                            interferometer_uniform,
                            interferometer_normal,
>>>>>>> make_layer_private
                            random_layers_uniform,
                            random_layers_normal,
                            strong_ent_layers_uniform,
                            strong_ent_layers_normal)


@pytest.fixture(scope="module",
                params=[2, 3])
def n_subsystems(request):
    """Number of qubits or modes."""
    return request.param


@pytest.fixture(scope="module",
                params=[None, 2, 10])
def n_rots(request):
    """Number of rotations in random layer."""
    return request.param


class TestInitCVQNN:
    """Tests the init module for a cv quantum neural network."""

    def test_cvqnnlayers_all_shape(self, n_subsystems, n_layers):
        """Confirm that cvqnn_layers_all()
         returns an array with the correct shape."""
        a = (n_layers, n_subsystems)
        b = (n_layers, n_subsystems * (n_subsystems - 1) // 2)
        p = cvqnn_layers_all(n_wires=n_subsystems, n_layers=n_layers, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a, a, a, b, b, a, a, a, a]

    def test_cvqnnlayers_all_same_output_for_same_seed(self, seed, tol):
        """Confirm that cvqnn_layers_all() returns a deterministic output for a fixed seed."""
        n_wires = 3
        n_layers = 2
<<<<<<< HEAD
        p1 = cvqnn_layers_all(n_layers=n_layers, n_wires=n_wires, seed=seed)
        p2 = cvqnn_layers_all(n_layers=n_layers, n_wires=n_wires, seed=seed)
        assert np.allclose(p1, p2, atol=tol, rtol=0.)

    def test_cvqnnlayers_all_diff_output_for_diff_seed(self, seed, tol):
        """Confirm that cvqnn_layers_all() returns a different output for
        different seeds."""
        n_wires = 3
        n_layers = 2
        p1 = cvqnn_layers_all(n_layers=n_layers, n_wires=n_wires, seed=seed)
        p2 = cvqnn_layers_all(n_layers=n_layers, n_wires=n_wires, seed=seed+1)
        assert not np.allclose(p1, p2, atol=tol, rtol=0.)


class TestInitInterferometer:
    """Tests the init module for an interferometer."""

    def test_interferometer_all_shape(self, n_subsystems):
        """Confirm that interferometer_all()
         returns an array with the correct shape."""
=======
        n_if = n_wires * (n_wires - 1) // 2
        p = cvqnn_layers_uniform(n_layers=n_layers, n_wires=n_wires, low=low, high=high,
                                mean_active=mean_a, std_active=std_a, seed=seed)
        np.random.seed(seed)
        theta_1 = np.random.uniform(low=low, high=high, size=(n_layers, n_if))
        phi_1 = np.random.uniform(low=low, high=high, size=(n_layers, n_if))
        varphi_1 = np.random.uniform(low=low, high=high, size=(n_layers, n_wires))
        r = np.random.normal(loc=mean_a, scale=std_a, size=(n_layers, n_wires))
        phi_r = np.random.uniform(low=low, high=high, size=(n_layers, n_wires))
        theta_2 = np.random.uniform(low=low, high=high, size=(n_layers, n_if))
        phi_2 = np.random.uniform(low=low, high=high, size=(n_layers, n_if))
        varphi_2 = np.random.uniform(low=low, high=high, size=(n_layers, n_wires))
        a = np.random.normal(loc=mean_a, scale=std_a, size=(n_layers, n_wires))
        phi_a = np.random.uniform(low=low, high=high, size=(n_layers, n_wires))
        k = np.random.normal(loc=mean_a, scale=std_a, size=(n_layers, n_wires))
        p_target = [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]
        assert np.allclose(p, p_target, atol=tol, rtol=0.)

    def test_cvqnnlayers_normal_dimensions(self, n_subsystems, n_layers):
        """Confirm that pennylane.init.cvqnn_layers_normal()
         returns an array with the right dimensions."""
        a = (n_layers, n_subsystems)
        b = (n_layers, n_subsystems * (n_subsystems - 1) // 2)
        p = cvqnn_layers_normal(n_wires=n_subsystems, n_layers=n_layers, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a, a, a, b, b, a, a, a, a]

    def test_cvqnnlayers_normal_edgecase(self, seed, tol):
        """Test sampling edge case of pennylane.init.cvqnn_layers_normal()."""
        n_layers = 3
        p = cvqnn_layers_normal(n_layers=n_layers, n_wires=10, mean=1, std=0, mean_active=1, std_active=0, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_cvqnnlayers_normal_range(self, seed, tol):
        """Confirm that pennylane.init.cvqnn_layers_normal() invokes the correct np.random sampling function."""
        mean = -0.5
        std = 1
        mean_a = 0.5
        std_a = 2
        n_wires = 3
        n_layers = 3
        n_if = n_wires * (n_wires - 1) // 2
        p = cvqnn_layers_normal(n_layers=n_layers, n_wires=n_wires, mean=mean, std=std, mean_active=mean_a, std_active=std_a, seed=seed)
        np.random.seed(seed)
        theta_1 = np.random.normal(loc=mean, scale=std, size=(n_layers, n_if))
        phi_1 = np.random.normal(loc=mean, scale=std, size=(n_layers, n_if))
        varphi_1 = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
        r = np.random.normal(loc=mean_a, scale=std_a, size=(n_layers, n_wires))
        phi_r = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
        theta_2 = np.random.normal(loc=mean, scale=std, size=(n_layers, n_if))
        phi_2 = np.random.normal(loc=mean, scale=std, size=(n_layers, n_if))
        varphi_2 = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
        a = np.random.normal(loc=mean_a, scale=std_a, size=(n_layers, n_wires))
        phi_a = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
        k = np.random.normal(loc=mean_a, scale=std_a, size=(n_layers, n_wires))
        p_target = [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]
        assert np.allclose(p, p_target, atol=tol, rtol=0.)


class TestParsInterferometer:
    """Tests the pennylane.init module for an interferometer."""

    def test_interferometer_uniform_dimensions(self, n_subsystems):
        """Confirm that pennylane.init.interferometer_uniform()
         returns an array with the right dimensions."""
>>>>>>> make_layer_private
        a = (n_subsystems, )
        b = (n_subsystems * (n_subsystems - 1) // 2, )
        p = interferometer_all(n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a]

    def test_interferometer_all_same_output_for_same_seed(self, seed, tol):
        """Confirm that interferometer_all() returns deterministic output for
         a fixed seed."""
        n_wires = 3
        p1 = interferometer_all(n_wires=n_wires, seed=seed)
        p2 = interferometer_all(n_wires=n_wires, seed=seed)
        assert np.allclose(p1, p2, atol=tol, rtol=0.)

    def test_interferometer_all_diff_output_for_diff_seed(self, seed, tol):
        """Confirm that interferometer_all() returns different output for
         different seeds."""
        n_wires = 3
        p1 = interferometer_all(n_wires=n_wires, seed=seed)
        p2 = interferometer_all(n_wires=n_wires, seed=seed+1)
        assert not np.allclose(p1, p2, atol=tol, rtol=0.)


class TestParsStronglyEntangling:
    """Tests the init module for a strongly entangling circuit."""

    def test_stronglyentangling_uniform_shape(self, n_subsystems, n_layers):
        """Confirm that the strong_ent_layers_uniform()
         returns an array with the correct shape."""
        a = (n_layers, n_subsystems, 3)
        p = strong_ent_layers_uniform(n_layers=n_layers, n_wires=n_subsystems, seed=0)
        assert p.shape == a

    def test_stronglyentangling_uniform_interval(self, seed, tol):
        """Test samples of strong_ent_layers_uniform() lie in correct interval."""
        n_layers = 3
        p = strong_ent_layers_uniform(n_layers=n_layers, n_wires=10, low=1, high=1, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_stronglyentangling_uniform_same_output_for_same_seed(self, seed, tol):
        """Confirm that strong_ent_layers_uniform() returns deterministic output for
        a fixed seed."""
        n_wires = 3
        n_layers = 3
<<<<<<< HEAD
        p1 = strong_ent_layers_uniform(n_layers=n_layers, n_wires=n_wires, seed=seed)
        p2 = strong_ent_layers_uniform(n_layers=n_layers, n_wires=n_wires, seed=seed)
        assert np.allclose(p1, p2, atol=tol, rtol=0.)

    def test_stronglyentangling_uniform_diff_output_for_diff_seed(self, seed, tol):
        """Confirm that strong_ent_layers_uniform() returns different output for
        two different seeds."""
        n_wires = 3
        n_layers = 3
        p1 = strong_ent_layers_uniform(n_layers=n_layers, n_wires=n_wires, seed=seed)
        p2 = strong_ent_layers_uniform(n_layers=n_layers, n_wires=n_wires, seed=seed+1)
        assert not np.allclose(p1, p2, atol=tol, rtol=0.)

    def test_stronglyentangling_normal_shape(self, n_subsystems, n_layers):
        """Confirm that strong_ent_layers_normal()
         returns an array with the correct shape."""
=======
        p = strong_ent_layers_uniform(n_layers=n_layers, n_wires=n_wires, low=low, high=high, seed=seed)
        np.random.seed(seed)
        p_target = np.random.uniform(low=low, high=high, size=(n_layers, n_wires, 3))
        assert np.allclose(p[0], p_target, atol=tol, rtol=0.)

    def test_stronglyentanglinglayers_normal_dimensions(self, n_subsystems, n_layers):
        """Confirm that the pennylane.init.strong_ent_layers_normal()
         returns an array with the right dimensions."""
>>>>>>> make_layer_private
        a = (n_layers, n_subsystems, 3)
        p = strong_ent_layers_normal(n_layers=n_layers, n_wires=n_subsystems, seed=0)
        assert p.shape == a

    def test_stronglyentangling_normal_interval(self, seed, tol):
        """Test samples of strong_ent_layers_normal() lie in correct interval."""
        n_layers = 3
        p = strong_ent_layers_normal(n_layers=n_layers, n_wires=10, mean=1, std=0, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_stronglyentangling_normal_same_output_for_same_seed(self, seed, tol):
        """Confirm that strong_ent_layers_normal() returns deterministic output for
        a fixed seed."""
        n_wires = 3
        n_layers = 3
<<<<<<< HEAD
        p1 = strong_ent_layers_normal(n_layers=n_layers, n_wires=n_wires,  seed=seed)
        p2 = strong_ent_layers_normal(n_layers=n_layers, n_wires=n_wires, seed=seed)
        assert np.allclose(p1, p2, atol=tol, rtol=0.)

    def test_stronglyentangling_normal_diff_output_for_diff_seed(self, seed, tol):
        """Confirm that strong_ent_layers_normal() returns different output for
        different seeds."""
        n_wires = 3
        n_layers = 3
        p1 = strong_ent_layers_normal(n_layers=n_layers, n_wires=n_wires,  seed=seed)
        p2 = strong_ent_layers_normal(n_layers=n_layers, n_wires=n_wires, seed=seed+1)
        assert not np.allclose(p1, p2, atol=tol, rtol=0.)
=======
        p = strong_ent_layers_normal(n_layers=n_layers, n_wires=n_wires, mean=mean, std=std, seed=seed)
        np.random.seed(seed)
        p_target = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires, 3))
        assert np.allclose(p[0], p_target, atol=tol, rtol=0.)
>>>>>>> make_layer_private


class TestParsRandom:
    """Tests the init module for a random circuit."""

    def test_randomlayers_uniform_shape(self, n_subsystems, n_layers, n_rots):
        """Confirm that the random_layers_uniform()
         returns an array with the correct shape."""
        if n_rots is None:
            n_rots = n_subsystems
        a = (n_layers, n_rots)
        p = random_layers_uniform(n_layers=n_layers, n_wires=n_subsystems, n_rots=n_rots, seed=0)
        assert p.shape == a

    def test_randomlayers_uniform_interval(self, seed, tol):
        """Test samples of random_layers_uniform() lie in correct interval."""
        n_layers = 3
        p = random_layers_uniform(n_layers=n_layers, n_wires=10, low=1, high=1, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_randomlayers_uniform_same_output_for_same_seed(self, seed, tol):
        """Confirm that random_layers_uniform() returns deterministic output for
        fixed seed."""
        n_wires = 3
        n_rots = 5
        n_layers = 3
<<<<<<< HEAD
        p1 = random_layers_uniform(n_layers=n_layers, n_wires=n_wires, n_rots=n_rots, seed=seed)
        p2 = random_layers_uniform(n_layers=n_layers, n_wires=n_wires, n_rots=n_rots, seed=seed)
        assert np.allclose(p1, p2, atol=tol, rtol=0.)

    def test_randomlayers_normal_shape(self, n_subsystems, n_layers, n_rots):
        """Confirm that the random_layers_normal()
         returns an array with the correct shape."""
=======
        p = random_layers_uniform(n_layers=n_layers, n_wires=n_wires, n_rots=n_rots, low=low, high=high, seed=seed)
        np.random.seed(seed)
        p_target = np.random.uniform(low=low, high=high, size=(n_layers, n_rots))
        assert np.allclose(p[0], p_target, atol=tol, rtol=0.)

    def test_randomlayers_normal_dimensions(self, n_subsystems, n_layers, n_rots):
        """Confirm that the pennylane.init.random_layers_normal()
         returns an array with the right dimensions."""
>>>>>>> make_layer_private
        if n_rots is None:
            n_rots = n_subsystems
        a = (n_layers, n_rots)
        p = random_layers_normal(n_layers=n_layers, n_wires=n_subsystems, n_rots=n_rots, seed=0)
        assert p.shape == a

    def test_randomlayers_normal_interval(self, seed, tol):
        """Test samples of random_layers_normal() lie in correct interval."""
        n_layers = 3
        p = random_layers_normal(n_layers=n_layers, n_wires=10, mean=1, std=0, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_randomlayers_normal_same_output_for_same_seed(self, seed, tol):
        """Confirm that random_layers_normal() returns a deterministic output for a
         fixed seed."""
        n_wires = 3
        n_rots = 5
        n_layers = 3
<<<<<<< HEAD
        p1 = random_layers_normal(n_layers=n_layers, n_wires=n_wires, n_rots=n_rots, seed=seed)
        p2 = random_layers_normal(n_layers=n_layers, n_wires=n_wires, n_rots=n_rots, seed=seed)
        assert np.allclose(p1, p2, atol=tol, rtol=0.)

    def test_randomlayers_normal_diff_output_for_diff_seed(self, seed, tol):
        """Confirm that random_layers_normal() returns different outputs for different
        seeds."""
        n_wires = 3
        n_rots = 5
        n_layers = 3
        p1 = random_layers_normal(n_layers=n_layers, n_wires=n_wires, n_rots=n_rots, seed=seed)
        p2 = random_layers_normal(n_layers=n_layers, n_wires=n_wires, n_rots=n_rots, seed=seed+1)
        assert not np.allclose(p1, p2, atol=tol, rtol=0.)
=======
        p = random_layers_normal(n_layers=n_layers, n_wires=n_wires, n_rots=n_rots, mean=mean, std=std, seed=seed)
        np.random.seed(seed)
        p_target = np.random.normal(loc=mean, scale=std, size=(n_layers, n_rots))
        assert np.allclose(p[0], p_target, atol=tol, rtol=0.)

>>>>>>> make_layer_private
