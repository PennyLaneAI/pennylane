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
from pennylane.templates.parameters import (parameters_cvqnn_layer,
                                            parameters_cvqnn_layers,
                                            parameters_interferometer,
                                            parameters_random_layer,
                                            parameters_random_layers,
                                            parameters_stronglyentangling_layer,
                                            parameters_stronglyentangling_layers)


@pytest.fixture(scope="module",
                params=[2, 3])
def n_subsystems(request):
    """Number of qubits or modes."""
    return request.param


@pytest.fixture(scope="module",
                params=[2, 3])
def n_layers(request):
    """Number of layers."""
    return request.param


@pytest.fixture(scope="module",
                params=[None, 2, 10])
def n_rots(request):
    """Number of rotations in random layer."""
    return request.param

@pytest.fixture(scope="module",
                params=[1, 2, 3])
def seed(request):
    """Different seeds."""
    return request.param


class TestParsCVQNN:
    """Tests the pennylane.templates.parameters methods for a cv-quantum neural network."""

    def test_pars_cvqnn_layer_dimensions(self, n_subsystems):
        """Confirm that pennylane.templates.utils.parameters_cvqnn_layer()
         returns an array with the right dimensions."""
        a = (n_subsystems, )
        b = (n_subsystems * (n_subsystems - 1) // 2, )
        p = parameters_cvqnn_layer(n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a, a, a, b, b, a, a, a, a]

    def test_pars_cvqnn_layer_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_cvqnn_layer() samples from the right distributions."""
        p = parameters_cvqnn_layer(n_wires=1000, uniform_min=-2, uniform_max=1, mean=0.5, std=2., seed=seed)
        p_av = np.array([np.mean(p_) for p_ in p])
        p_std = np.array([np.std(p_) for p_ in p])
        target_av = np.array([-0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5])
        target_std = np.array([0.86, 0.86, 0.86, 2, 0.86, 0.86, 0.86, 0.86, 2, 0.86, 2])
        assert np.allclose(p_av, target_av, atol=0.13)
        assert np.allclose(p_std, target_std, atol=0.13)

    def test_pars_cvqnn_layers_dimensions(self, n_subsystems, n_layers):
        """Confirm that pennylane.templates.utils.parameters_cvqnn_layers()
         returns an array with the right dimensions."""
        a = (n_layers, n_subsystems)
        b = (n_layers, n_subsystems * (n_subsystems - 1) // 2)
        p = parameters_cvqnn_layers(n_wires=n_subsystems, n_layers=n_layers, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a, a, a, b, b, a, a, a, a]

    def test_pars_cvqnn_layers_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_cvqnn_layers() samples from the right distributions."""
        p = parameters_cvqnn_layers(n_layers=2, n_wires=1000, uniform_min=-2, uniform_max=1, mean=0.5, std=2., seed=seed)
        p_av = np.array([np.mean(p_) for p_ in p])
        p_std = np.array([np.std(p_) for p_ in p])
        target_av = np.array([-0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5])
        target_std = np.array([0.86, 0.86, 0.86, 2, 0.86, 0.86, 0.86, 0.86, 2, 0.86, 2])
        assert np.allclose(p_av, target_av, atol=0.13)
        assert np.allclose(p_std, target_std, atol=0.13)


class TestParsInterferometer:
    """Tests the pennylane.templates.parameters method for an interferometer."""

    def test_pars_interferometer_dimensions(self, n_subsystems):
        """Confirm that pennylane.templates.utils.parameters_interferometer()
         returns an array with the right dimensions."""
        a = (n_subsystems, )
        b = (n_subsystems * (n_subsystems - 1) // 2, )
        p = parameters_interferometer(n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a]

    def test_pars_interferometer_layer_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_interferometer() samples from the right distributions."""
        p = parameters_interferometer(n_wires=1000, uniform_min=-2, uniform_max=1, seed=seed)
        p_av = np.array([np.mean(p_) for p_ in p])
        p_std = np.array([np.std(p_) for p_ in p])
        target_av = np.array([-0.5, -0.5, -0.5])
        target_std = np.array([0.86, 0.86, 0.86])
        assert np.allclose(p_av, target_av, atol=0.13)
        assert np.allclose(p_std, target_std, atol=0.13)


class TestParsStronglyEntangling:
    """Tests the pennylane.templates.parameters methods for a strongly entangling circuit."""

    def test_pars_stronglyentangling_layer_dimensions(self, n_subsystems):
        """Confirm that the pennylane.templates.utils.parameters_stronglyentangling_layer()
         returns an array with the right dimensions."""
        a = (n_subsystems, 3)
        p = parameters_stronglyentangling_layer(n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_pars_stronglyentangling_layer_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_stronglyentangling_layer()
        samples from the right distributions."""
        p = parameters_stronglyentangling_layer(n_wires=1000, uniform_min=-2, uniform_max=1, seed=seed)
        p_av = [np.mean(p_) for p_ in p]
        p_std = [np.std(p_) for p_ in p]
        assert np.isclose(p_av, [-0.5], atol=0.13)
        assert np.isclose(p_std, [0.86], atol=0.13)

    def test_pars_stronglyentangling_layers_dimensions(self, n_subsystems, n_layers):
        """Confirm that the pennylane.templates.utils.parameters_stronglyentangling_layers()
         returns an array with the right dimensions."""
        a = (n_layers, n_subsystems, 3)
        p = parameters_stronglyentangling_layers(n_layers=n_layers, n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_pars_stronglyentangling_layers_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_stronglyentangling_layers()
        samples from the right distributions."""
        p = parameters_stronglyentangling_layers(n_layers=2, n_wires=1000, uniform_min=-2, uniform_max=1, seed=seed)
        p_av = [np.mean(p_) for p_ in p]
        p_std = [np.std(p_) for p_ in p]
        assert np.isclose(p_av, [-0.5], atol=0.13)
        assert np.isclose(p_std, [0.86], atol=0.13)


class TestParsRandom:
    """Tests the pennylane.templates.parameters methods for a random circuit."""

    def test_pars_random_layer_dimensions(self, n_subsystems, n_rots):
        """Confirm that the pennylane.templates.utils.parameters_random_layer()
         returns an array with the right dimensions."""
        if n_rots is None:
            n_rots = n_subsystems
        a = (n_rots, )
        p = parameters_random_layer(n_wires=n_subsystems, n_rots=n_rots, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_pars_stronglyentangling_layer_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_random_layer()
        samples from the right distributions."""
        p = parameters_random_layer(n_wires=1000, uniform_min=-2, uniform_max=1, seed=seed)
        p_av = [np.mean(p_) for p_ in p]
        p_std = [np.std(p_) for p_ in p]
        assert np.isclose(p_av, [-0.5], atol=0.13)
        assert np.isclose(p_std, [0.86], atol=0.13)

    def test_pars_random_layers_dimensions(self, n_subsystems, n_layers, n_rots):
        """Confirm that the pennylane.templates.utils.parameters_random_layers()
         returns an array with the right dimensions."""
        if n_rots is None:
            n_rots = n_subsystems
        a = (n_layers, n_rots)
        p = parameters_random_layers(n_layers=n_layers, n_wires=n_subsystems, n_rots=n_rots, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_pars_stronglyentangling_layers_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_random_layers()
        samples from the right distributions."""
        p = parameters_random_layers(n_layers=2, n_wires=1000, uniform_min=-2, uniform_max=1, seed=seed)
        p_av = [np.mean(p_) for p_ in p]
        p_std = [np.std(p_) for p_ in p]
        assert np.isclose(p_av, [-0.5], atol=0.13)
        assert np.isclose(p_std, [0.86], atol=0.13)