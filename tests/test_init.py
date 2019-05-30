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
from pennylane.init import (cvqnn_layer_uniform,
                            cvqnn_layers_uniform,
                            cvqnn_layer_normal,
                            cvqnn_layers_normal,
                            interferometer_uniform,
                            interferometer_normal,
                            random_layer_uniform,
                            random_layers_uniform,
                            random_layer_normal,
                            random_layers_normal,
                            strong_ent_layer_uniform,
                            strong_ent_layers_uniform,
                            strong_ent_layer_normal,
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


class TestParsCVQNN:
    """Tests the pennylane.init module for a cv-quantum neural network."""

    def test_cvqnnlayers_uniform_dimensions(self, n_subsystems, n_layers):
        """Confirm that pennylane.init.cvqnn_layers_uniform()
         returns an array with the right dimensions."""
        a = (n_layers, n_subsystems)
        b = (n_layers, n_subsystems * (n_subsystems - 1) // 2)
        p = cvqnn_layers_uniform(n_wires=n_subsystems, n_layers=n_layers, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a, a, a, b, b, a, a, a, a]

    def test_cvqnnlayers_uniform_interval(self, seed):
        """Confirm that no uniform sample in pennylane.init.cvqnn_layers_uniform() lies outside of interval."""
        low = -2
        high = 1
        p = cvqnn_layers_uniform(n_layers=2, n_wires=10, low=low, high=high, seed=seed)
        p_uni = [p[i] for i in [0, 1, 2, 4, 5, 6, 7, 9]]
        assert all([(p_ <= high).all() and (p_ >= low).all() for p_ in p_uni])

    def test_cvqnnlayers_uniform_edgecase(self, seed, tol):
        """Test sampling edge case of pennylane.init.cvqnn_layers_uniform()."""
        p = cvqnn_layers_uniform(n_layers=2, n_wires=10, low=1, high=1, mean_active=1, std_active=0, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_cvqnnlayers_uniform_seed(self, seed, tol):
        """Confirm that pennylane.init.cvqnn_layers_uniform() invokes the correct np.random sampling function
        for a given seed."""
        low = -2
        high = 1
        mean_a = 0.5
        std_a = 2
        n_wires = 3
        n_layers = 2
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

    def test_cvqnnlayer_uniform_dimensions(self, n_subsystems):
        """Confirm that pennylane.init.cvqnn_layer_uniform()
         returns an array with the right dimensions."""
        a = (n_subsystems, )
        b = (n_subsystems * (n_subsystems - 1) // 2, )
        p = cvqnn_layer_uniform(n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a, a, a, b, b, a, a, a, a]

    def test_cvqnnlayer_uniform_interval(self, seed):
        """Confirm that no uniform sample in pennylane.init.cvqnn_layer_uniform() lies outside of interval."""
        low = -2
        high = 1
        p = cvqnn_layer_uniform(n_wires=10, low=low, high=high, seed=seed)
        p_uni = [p[i] for i in [0, 1, 2, 4, 5, 6, 7, 9]]
        assert all([(p_ <= high).all() and (p_ >= low).all() for p_ in p_uni])

    def test_cvqnnlayer_uniform_edgecase(self, seed, tol):
        """Test sampling edge case of pennylane.init.cvqnn_layer_uniform()."""
        p = cvqnn_layer_uniform(n_wires=10, low=1, high=1, mean_active=1, std_active=0, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_cvqnnlayer_uniform_seed(self, seed, tol):
        """Confirm that pennylane.init.cvqnn_layer_uniform() invokes the correct np.random sampling function
        for a given seed."""
        low = -2
        high = 1
        mean_a = 0.5
        std_a = 2
        n_wires = 3
        n_if = n_wires * (n_wires - 1) // 2
        p = cvqnn_layer_uniform(n_wires=n_wires, low=low, high=high, mean_active=mean_a, std_active=std_a, seed=seed)
        np.random.seed(seed)
        theta_1 = np.random.uniform(low=low, high=high, size=(n_if,))
        phi_1 = np.random.uniform(low=low, high=high, size=(n_if,))
        varphi_1 = np.random.uniform(low=low, high=high, size=(n_wires,))
        r = np.random.normal(loc=mean_a, scale=std_a, size=(n_wires,))
        phi_r = np.random.uniform(low=low, high=high, size=(n_wires,))
        theta_2 = np.random.uniform(low=low, high=high, size=(n_if,))
        phi_2 = np.random.uniform(low=low, high=high, size=(n_if,))
        varphi_2 = np.random.uniform(low=low, high=high, size=(n_wires,))
        a = np.random.normal(loc=mean_a, scale=std_a, size=(n_wires,))
        phi_a = np.random.uniform(low=low, high=high, size=(n_wires,))
        k = np.random.normal(loc=mean_a, scale=std_a, size=(n_wires,))
        p_target = [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]
        assert np.allclose(p, p_target, atol=tol)

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

    def test_cvqnnlayer_normal_dimensions(self, n_subsystems):
        """Confirm that pennylane.init.cvqnn_layer_normal()
         returns an array with the right dimensions."""
        a = (n_subsystems, )
        b = (n_subsystems * (n_subsystems - 1) // 2, )
        p = cvqnn_layer_normal(n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a, a, a, b, b, a, a, a, a]

    def test_cvqnnlayer_normal_edgecase(self, seed, tol):
        """Test sampling edge case of pennylane.init.cvqnn_layer_normal()."""
        p = cvqnn_layer_normal(n_wires=10, mean=1, std=0, mean_active=1, std_active=0, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_cvqnnlayer_normal_range(self, seed, tol):
        """Confirm that pennylane.init.cvqnn_layer_normal() invokes the correct np.random sampling function."""
        mean = -0.5
        std = 1
        mean_a = 0.5
        std_a = 2
        n_wires = 3
        n_if = n_wires * (n_wires - 1) // 2
        p = cvqnn_layer_normal(n_wires=n_wires, mean=mean, std=std, mean_active=mean_a, std_active=std_a, seed=seed)
        np.random.seed(seed)
        theta_1 = np.random.normal(loc=mean, scale=std, size=(n_if,))
        phi_1 = np.random.normal(loc=mean, scale=std, size=(n_if,))
        varphi_1 = np.random.normal(loc=mean, scale=std, size=(n_wires,))
        r = np.random.normal(loc=mean_a, scale=std_a, size=(n_wires,))
        phi_r = np.random.normal(loc=mean, scale=std, size=(n_wires,))
        theta_2 = np.random.normal(loc=mean, scale=std, size=(n_if,))
        phi_2 = np.random.normal(loc=mean, scale=std, size=(n_if,))
        varphi_2 = np.random.normal(loc=mean, scale=std, size=(n_wires,))
        a = np.random.normal(loc=mean_a, scale=std_a, size=(n_wires,))
        phi_a = np.random.normal(loc=mean, scale=std, size=(n_wires,))
        k = np.random.normal(loc=mean_a, scale=std_a, size=(n_wires,))
        p_target = [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]
        assert np.allclose(p, p_target, atol=tol)


class TestParsInterferometer:
    """Tests the pennylane.init module for an interferometer."""

    def test_interferometer_uniform_dimensions(self, n_subsystems):
        """Confirm that pennylane.init.interferometer_uniform()
         returns an array with the right dimensions."""
        a = (n_subsystems, )
        b = (n_subsystems * (n_subsystems - 1) // 2, )
        p = interferometer_uniform(n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a]

    def test_interferometer_uniform_interval(self, seed):
        """Confirm that no uniform sample in pennylane.init.interferometer_uniform() lies outside of interval."""
        low = -2
        high = 1
        p = interferometer_uniform(n_wires=10, low=low, high=high, seed=seed)
        assert all([(p_ <= high).all() and (p_ >= low).all() for p_ in p])

    def test_interferometer_uniform_edgecase(self, seed, tol):
        """Test sampling edge case of pennylane.init.interferometer_uniform()."""
        p = interferometer_uniform(n_wires=10, low=1, high=1, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_interferometer_uniform_seed(self, seed, tol):
        """Confirm that pennylane.init.interferometer_uniform() invokes the correct np.random sampling function
        for a given seed."""
        low = -2
        high = 1
        n_wires = 3
        n_if = n_wires * (n_wires - 1) // 2
        p = interferometer_uniform(n_wires=n_wires, low=low, high=high, seed=seed)
        np.random.seed(seed)
        theta = np.random.uniform(low=low, high=high, size=(n_if,))
        phi = np.random.uniform(low=low, high=high, size=(n_if,))
        varphi = np.random.uniform(low=low, high=high, size=(n_wires,))
        p_target = [theta, phi, varphi]
        assert np.allclose(p, p_target, atol=tol, rtol=0.)

    def test_interferometer_normal_dimensions(self, n_subsystems):
        """Confirm that pennylane.init.interferometer_normal()
         returns an array with the right dimensions."""
        a = (n_subsystems, )
        b = (n_subsystems * (n_subsystems - 1) // 2, )
        p = interferometer_normal(n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a]

    def test_interferometer_normal_edgecase(self, seed, tol):
        """Test sampling edge case of pennylane.init.interferometer_normal()."""
        p = interferometer_normal(n_wires=10, mean=1, std=0, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_interferometer_normal_seed(self, seed, tol):
        """Confirm that pennylane.init.interferometer_normal() invokes the correct np.random sampling function
        for a given seed."""
        mean = 0.5
        std = 1
        n_wires = 3
        n_if = n_wires * (n_wires - 1) // 2
        p = interferometer_normal(n_wires=n_wires, mean=mean, std=std, seed=seed)
        np.random.seed(seed)
        theta = np.random.normal(loc=mean, scale=std, size=(n_if,))
        phi = np.random.normal(loc=mean, scale=std, size=(n_if,))
        varphi = np.random.normal(loc=mean, scale=std, size=(n_wires,))
        p_target = [theta, phi, varphi]
        assert np.allclose(p, p_target, atol=tol, rtol=0.)


class TestParsStronglyEntangling:
    """Tests the pennylane.init module for a strongly entangling circuit."""

    def test_stronglyentanglinglayers_uniform_dimensions(self, n_subsystems, n_layers):
        """Confirm that the pennylane.init.strong_ent_layers_uniform()
         returns an array with the right dimensions."""
        a = (n_layers, n_subsystems, 3)
        p = strong_ent_layers_uniform(n_layers=n_layers, n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_stronglyentanglinglayers_uniform_interval(self, seed):
        """Confirm that no uniform sample in pennylane.init.strong_ent_layers_uniform() lies outside of interval."""
        low = -2
        high = 1
        n_layers = 3
        p = strong_ent_layers_uniform(n_layers=n_layers, n_wires=10, low=low, high=high, seed=seed)
        assert all([(p_ <= high).all() and (p_ >= low).all() for p_ in p])

    def test_stronglyentanglinglayers_uniform_edgecase(self, seed, tol):
        """Test sampling edge case of pennylane.init.strong_ent_layers_uniform()."""
        n_layers = 3
        p = strong_ent_layers_uniform(n_layers=n_layers, n_wires=10, low=1, high=1, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_stronglyentanglinglayers_uniform_seed(self, seed, tol):
        """Confirm that pennylane.init.strong_ent_layers_uniform() invokes the correct np.random sampling function
        for a given seed."""
        low = -2
        high = 1
        n_wires = 3
        n_layers = 3
        p = strong_ent_layers_uniform(n_layers=n_layers, n_wires=n_wires, low=low, high=high, seed=seed)
        np.random.seed(seed)
        p_target = np.random.uniform(low=low, high=high, size=(n_layers, n_wires, 3))
        assert np.allclose(p[0], p_target, atol=tol, rtol=0.)

    def test_stronglyentanglinglayer_uniform_dimensions(self, n_subsystems):
        """Confirm that the pennylane.init.strong_ent_layer_uniform()
         returns an array with the right dimensions."""
        a = (n_subsystems, 3)
        p = strong_ent_layer_uniform(n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_stronglyentanglinglayer_uniform_interval(self, seed):
        """Confirm that no uniform sample in pennylane.init.strong_ent_layer_uniform() lies outside of interval."""
        low = -2
        high = 1
        p = strong_ent_layer_uniform(n_wires=10, low=low, high=high, seed=seed)
        assert all([(p_ <= high).all() and (p_ >= low).all() for p_ in p])

    def test_stronglyentanglinglayer_uniform_edgecase(self, seed, tol):
        """Test sampling edge case of pennylane.init.strong_ent_layer_uniform()."""
        p = strong_ent_layer_uniform(n_wires=10, low=1, high=1, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_stronglyentanglinglayer_uniform_seed(self, seed, tol):
        """Confirm that pennylane.init.strong_ent_layer_uniform() invokes the correct np.random sampling function
        for a given seed."""
        low = -2
        high = 1
        n_wires = 3
        p = strong_ent_layer_uniform(n_wires=n_wires, low=low, high=high, seed=seed)
        np.random.seed(seed)
        p_target = np.random.uniform(low=low, high=high, size=(n_wires, 3))
        assert np.allclose(p[0], p_target, atol=tol, rtol=0.)

    def test_stronglyentanglinglayers_normal_dimensions(self, n_subsystems, n_layers):
        """Confirm that the pennylane.init.strong_ent_layers_normal()
         returns an array with the right dimensions."""
        a = (n_layers, n_subsystems, 3)
        p = strong_ent_layers_normal(n_layers=n_layers, n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_stronglyentanglinglayers_normal_edgecase(self, seed, tol):
        """Test sampling edge case of pennylane.init.strong_ent_layers_normal()."""
        n_layers = 3
        p = strong_ent_layers_normal(n_layers=n_layers, n_wires=10, mean=1, std=0, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_stronglyentanglinglayers_normal_seed(self, seed, tol):
        """Confirm that pennylane.init.strong_ent_layers_normal() invokes the correct np.random sampling function
        for a given seed."""
        mean = -2
        std = 1
        n_wires = 3
        n_layers = 3
        p = strong_ent_layers_normal(n_layers=n_layers, n_wires=n_wires, mean=mean, std=std, seed=seed)
        np.random.seed(seed)
        p_target = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires, 3))
        assert np.allclose(p[0], p_target, atol=tol, rtol=0.)

    def test_stronglyentanglinglayer_normal_dimensions(self, n_subsystems):
        """Confirm that the pennylane.init.parameters_stronglyentanglinglayer_normalm()
         returns an array with the right dimensions."""
        a = (n_subsystems, 3)
        p = strong_ent_layer_normal(n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_stronglyentanglinglayer_normal_edgecase(self, seed, tol):
        """Test sampling edge case of pennylane.init.strong_ent_layer_normal()."""
        p = strong_ent_layer_normal(n_wires=10, mean=1, std=0, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_stronglyentanglinglayer_normal_seed(self, seed, tol):
        """Confirm that pennylane.init.strong_ent_layer_normal() invokes the correct np.random sampling function
        for a given seed."""
        mean = -2
        std = 1
        n_wires = 3
        p = strong_ent_layer_normal(n_wires=n_wires, mean=mean, std=std, seed=seed)
        np.random.seed(seed)
        p_target = np.random.normal(loc=mean, scale=std, size=(n_wires, 3))
        assert np.allclose(p[0], p_target, atol=tol, rtol=0.)


class TestParsRandom:
    """Tests the pennylane.init module for a random circuit."""

    def test_randomlayers_uniform_dimensions(self, n_subsystems, n_layers, n_rots):
        """Confirm that the pennylane.init.random_layers_uniform()
         returns an array with the right dimensions."""
        if n_rots is None:
            n_rots = n_subsystems
        a = (n_layers, n_rots)
        p = random_layers_uniform(n_layers=n_layers, n_wires=n_subsystems, n_rots=n_rots, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_randomlayers_uniform_interval(self, seed):
        """Confirm that no uniform sample in pennylane.init.random_layers_uniform() lies outside of interval."""
        low = -2
        high = 1
        n_layers = 3
        p = random_layers_uniform(n_layers=n_layers, n_wires=10, low=low, high=high, seed=seed)
        assert all([(p_ <= high).all() and (p_ >= low).all() for p_ in p])

    def test_randomlayers_uniform_edgecase(self, seed, tol):
        """Test sampling edge case of pennylane.init.random_layers_uniform()."""
        n_layers = 3
        p = random_layers_uniform(n_layers=n_layers, n_wires=10, low=1, high=1, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_randomlayers_uniform_seed(self, seed, tol):
        """Confirm that pennylane.init.random_layers_uniform() invokes the correct np.random sampling function
        for a given seed."""
        low = -2
        high = 1
        n_wires = 3
        n_rots = 5
        n_layers = 3
        p = random_layers_uniform(n_layers=n_layers, n_wires=n_wires, n_rots=n_rots, low=low, high=high, seed=seed)
        np.random.seed(seed)
        p_target = np.random.uniform(low=low, high=high, size=(n_layers, n_rots))
        assert np.allclose(p[0], p_target, atol=tol, rtol=0.)

    def test_randomlayer_uniform_dimensions(self, n_subsystems, n_rots):
        """Confirm that the pennylane.init.random_layer_uniform()
         returns an array with the right dimensions."""
        if n_rots is None:
            n_rots = n_subsystems
        a = (n_rots, )
        p = random_layer_uniform(n_wires=n_subsystems, n_rots=n_rots, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_randomlayer_uniform_interval(self, seed):
        """Confirm that no uniform sample in pennylane.init.random_layer_uniform() lies outside of interval."""
        low = -2
        high = 1
        p = random_layer_uniform(n_wires=10, low=low, high=high, seed=seed)
        assert all([(p_ <= high).all() and (p_ >= low).all() for p_ in p])

    def test_randomlayer_uniform_edgecase(self, seed, tol):
        """Test sampling edge case of pennylane.init.random_layer_uniform()."""
        p = random_layer_uniform(n_wires=10, low=1, high=1, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_randomlayer_uniform_seed(self, seed, tol):
        """Confirm that pennylane.init.random_layer_uniform() invokes the correct np.random sampling function
        for a given seed."""
        low = -2
        high = 1
        n_wires = 3
        n_rots = 5
        p = random_layer_uniform(n_wires=n_wires, n_rots=n_rots, low=low, high=high, seed=seed)
        np.random.seed(seed)
        p_target = np.random.uniform(low=low, high=high, size=(n_rots,))
        assert np.allclose(p[0], p_target, atol=tol, rtol=0.)

    def test_randomlayers_normal_dimensions(self, n_subsystems, n_layers, n_rots):
        """Confirm that the pennylane.init.random_layers_normal()
         returns an array with the right dimensions."""
        if n_rots is None:
            n_rots = n_subsystems
        a = (n_layers, n_rots)
        p = random_layers_normal(n_layers=n_layers, n_wires=n_subsystems, n_rots=n_rots, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_randomlayers_normal_edgecase(self, seed, tol):
        """Test sampling edge case of pennylane.init.random_layers_normal()."""
        n_layers = 3
        p = random_layers_normal(n_layers=n_layers, n_wires=10, mean=1, std=0, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_randomlayers_normal_seed(self, seed, tol):
        """Confirm that pennylane.init.random_layers_normal() invokes the correct np.random sampling function
        for a given seed."""
        mean = -2
        std = 1
        n_wires = 3
        n_rots = 5
        n_layers = 3
        p = random_layers_normal(n_layers=n_layers, n_wires=n_wires, n_rots=n_rots, mean=mean, std=std, seed=seed)
        np.random.seed(seed)
        p_target = np.random.normal(loc=mean, scale=std, size=(n_layers, n_rots))
        assert np.allclose(p[0], p_target, atol=tol, rtol=0.)

    def test_randomlayer_normal_dimensions(self, n_subsystems, n_rots):
        """Confirm that the pennylane.init.random_layer_normal()
         returns an array with the right dimensions."""
        if n_rots is None:
            n_rots = n_subsystems
        a = (n_rots, )
        p = random_layer_normal(n_wires=n_subsystems, n_rots=n_rots, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_randomlayer_normal_edgecase(self, seed, tol):
        """Test sampling edge case of pennylane.init.random_layer_normal()."""
        p = random_layer_normal(n_wires=10, mean=1, std=0, seed=seed)
        p_mean = np.mean(np.array([np.mean(pp) for p_ in p for pp in p_]))
        assert np.allclose(p_mean, 1, atol=tol, rtol=0.)

    def test_randomlayer_normal_seed(self, seed, tol):
        """Confirm that pennylane.init.random_layer_normal() invokes the correct np.random sampling function
        for a given seed."""
        mean = -2
        std = 1
        n_wires = 3
        n_rots = 5
        p = random_layer_normal(n_wires=n_wires, n_rots=n_rots, mean=mean, std=std, seed=seed)
        np.random.seed(seed)
        p_target = np.random.normal(loc=mean, scale=std, size=(n_rots,))
        assert np.allclose(p[0], p_target, atol=tol, rtol=0.)
