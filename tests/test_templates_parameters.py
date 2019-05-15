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
from pennylane.templates.parameters import (parameters_cvqnnlayer_uniform,
                                            parameters_cvqnnlayers_uniform,
                                            parameters_cvqnnlayer_normal,
                                            parameters_cvqnnlayers_normal,
                                            parameters_interferometer_uniform,
                                            parameters_interferometer_normal,
                                            parameters_randomlayer_uniform,
                                            parameters_randomlayers_uniform,
                                            parameters_randomlayer_normal,
                                            parameters_randomlayers_normal,
                                            parameters_stronglyentanglinglayer_uniform,
                                            parameters_stronglyentanglinglayers_uniform,
                                            parameters_stronglyentanglinglayer_normal,
                                            parameters_stronglyentanglinglayers_normal)


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

    def test_pars_cvqnnlayer_uniform_dimensions(self, n_subsystems):
        """Confirm that pennylane.templates.utils.parameters_cvqnnlayer_uniform()
         returns an array with the right dimensions."""
        a = (n_subsystems, )
        b = (n_subsystems * (n_subsystems - 1) // 2, )
        p = parameters_cvqnnlayer_uniform(n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a, a, a, b, b, a, a, a, a]

    def test_pars_cvqnnlayer_uniform_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_cvqnnlayer_uniform() samples from the right
        distributions."""
        p = parameters_cvqnnlayer_uniform(n_wires=1000, uniform_min=-2, uniform_max=1, mean=0.5, std=2., seed=seed)
        p_av = np.array([np.mean(p_) for p_ in p])
        p_std = np.array([np.std(p_) for p_ in p])
        target_av = np.array([-0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5])
        target_std = np.array([0.86, 0.86, 0.86, 2, 0.86, 0.86, 0.86, 0.86, 2, 0.86, 2])
        assert np.allclose(p_av, target_av, atol=0.13)
        assert np.allclose(p_std, target_std, atol=0.13)

    def test_pars_cvqnnlayers_uniform_dimensions(self, n_subsystems, n_layers):
        """Confirm that pennylane.templates.utils.parameters_cvqnnlayers_uniform()
         returns an array with the right dimensions."""
        a = (n_layers, n_subsystems)
        b = (n_layers, n_subsystems * (n_subsystems - 1) // 2)
        p = parameters_cvqnnlayers_uniform(n_wires=n_subsystems, n_layers=n_layers, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a, a, a, b, b, a, a, a, a]

    def test_pars_cvqnnlayers_uniform_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_cvqnnlayers_uniform() samples from the right
        distributions."""
        p = parameters_cvqnnlayers_uniform(n_layers=2, n_wires=1000, uniform_min=-2, uniform_max=1, mean=0.5, std=2., seed=seed)
        p_av = np.array([np.mean(p_) for p_ in p])
        p_std = np.array([np.std(p_) for p_ in p])
        target_av = np.array([-0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5])
        target_std = np.array([0.86, 0.86, 0.86, 2, 0.86, 0.86, 0.86, 0.86, 2, 0.86, 2])
        assert np.allclose(p_av, target_av, atol=0.13)
        assert np.allclose(p_std, target_std, atol=0.13)

    def test_pars_cvqnnlayer_normal_dimensions(self, n_subsystems):
        """Confirm that pennylane.templates.utils.parameters_cvqnnlayer_normal()
         returns an array with the right dimensions."""
        a = (n_subsystems, )
        b = (n_subsystems * (n_subsystems - 1) // 2, )
        p = parameters_cvqnnlayer_normal(n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a, a, a, b, b, a, a, a, a]

    def test_pars_cvqnnlayer_normal_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_cvqnnlayer_normal() samples from the right
        distributions."""
        p = parameters_cvqnnlayer_normal(n_wires=1000, mean=-1, std=0.2, seed=seed)
        p_av = np.array([np.mean(p_) for p_ in p])
        p_std = np.array([np.std(p_) for p_ in p])
        target_av = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        target_std = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        assert np.allclose(p_av, target_av, atol=0.13)
        assert np.allclose(p_std, target_std, atol=0.13)

    def test_pars_cvqnnlayers_normal_dimensions(self, n_subsystems, n_layers):
        """Confirm that pennylane.templates.utils.parameters_cvqnnlayers_normal()
         returns an array with the right dimensions."""
        a = (n_layers, n_subsystems)
        b = (n_layers, n_subsystems * (n_subsystems - 1) // 2)
        p = parameters_cvqnnlayers_normal(n_wires=n_subsystems, n_layers=n_layers, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a, a, a, b, b, a, a, a, a]

    def test_pars_cvqnnlayers_normal_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_cvqnnlayers_normal() samples from the right
        distributions."""
        p = parameters_cvqnnlayers_normal(n_layers=2, n_wires=1000, mean=-1, std=0.2, seed=seed)
        p_av = np.array([np.mean(p_) for p_ in p])
        p_std = np.array([np.std(p_) for p_ in p])
        target_av = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        target_std = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        assert np.allclose(p_av, target_av, atol=0.13)
        assert np.allclose(p_std, target_std, atol=0.13)


class TestParsInterferometer:
    """Tests the pennylane.templates.parameters method for an interferometer."""

    def test_pars_interferometer_uniform_dimensions(self, n_subsystems):
        """Confirm that pennylane.templates.utils.parameters_interferometer_uniform()
         returns an array with the right dimensions."""
        a = (n_subsystems, )
        b = (n_subsystems * (n_subsystems - 1) // 2, )
        p = parameters_interferometer_uniform(n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a]

    def test_pars_interferometer_uniform_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_interferometer_uniform() samples from the right
        distributions."""
        p = parameters_interferometer_uniform(n_wires=1000, uniform_min=-2, uniform_max=1, seed=seed)
        p_av = np.array([np.mean(p_) for p_ in p])
        p_std = np.array([np.std(p_) for p_ in p])
        target_av = np.array([-0.5, -0.5, -0.5])
        target_std = np.array([0.86, 0.86, 0.86])
        assert np.allclose(p_av, target_av, atol=0.13)
        assert np.allclose(p_std, target_std, atol=0.13)

    def test_pars_interferometer_normal_dimensions(self, n_subsystems):
        """Confirm that pennylane.templates.utils.parameters_interferometer_normal()
         returns an array with the right dimensions."""
        a = (n_subsystems, )
        b = (n_subsystems * (n_subsystems - 1) // 2, )
        p = parameters_interferometer_normal(n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [b, b, a]

    def test_pars_interferometer_normal_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_interferometer_normal() samples from the right
        distributions."""
        p = parameters_interferometer_normal(n_wires=1000, mean=-1, std=0.2, seed=seed)
        p_av = np.array([np.mean(p_) for p_ in p])
        p_std = np.array([np.std(p_) for p_ in p])
        target_av = np.array([-1, -1, -1])
        target_std = np.array([0.2, 0.2, 0.2])
        assert np.allclose(p_av, target_av, atol=0.13)
        assert np.allclose(p_std, target_std, atol=0.13)


class TestParsStronglyEntangling:
    """Tests the pennylane.templates.parameters methods for a strongly entangling circuit."""

    def test_pars_stronglyentanglinglayer_uniform_dimensions(self, n_subsystems):
        """Confirm that the pennylane.templates.utils.parameters_stronglyentanglinglayer_uniform()
         returns an array with the right dimensions."""
        a = (n_subsystems, 3)
        p = parameters_stronglyentanglinglayer_uniform(n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_pars_stronglyentanglinglayer_uniform_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_stronglyentanglinglayer_uniform()
        samples from the right distributions."""
        p = parameters_stronglyentanglinglayer_uniform(n_wires=1000, uniform_min=-2, uniform_max=1, seed=seed)
        p_av = [np.mean(p_) for p_ in p]
        p_std = [np.std(p_) for p_ in p]
        assert np.isclose(p_av, [-0.5], atol=0.13)
        assert np.isclose(p_std, [0.86], atol=0.13)

    def test_pars_stronglyentanglinglayers_uniform_dimensions(self, n_subsystems, n_layers):
        """Confirm that the pennylane.templates.utils.parameters_stronglyentanglinglayers_uniform()
         returns an array with the right dimensions."""
        a = (n_layers, n_subsystems, 3)
        p = parameters_stronglyentanglinglayers_uniform(n_layers=n_layers, n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_pars_stronglyentanglinglayers_uniform_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_stronglyentanglinglayers_uniform()
        samples from the right distributions."""
        p = parameters_stronglyentanglinglayers_uniform(n_layers=2, n_wires=1000, uniform_min=-2, uniform_max=1,
                                                        seed=seed)
        p_av = [np.mean(p_) for p_ in p]
        p_std = [np.std(p_) for p_ in p]
        assert np.isclose(p_av, [-0.5], atol=0.13)
        assert np.isclose(p_std, [0.86], atol=0.13)

    def test_pars_stronglyentanglinglayer_normal_dimensions(self, n_subsystems):
        """Confirm that the pennylane.templates.utils.parameters_stronglyentanglinglayer_normalm()
         returns an array with the right dimensions."""
        a = (n_subsystems, 3)
        p = parameters_stronglyentanglinglayer_normal(n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_pars_stronglyentanglinglayer_normal_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_stronglyentanglinglayer_normal()
        samples from the right distributions."""
        p = parameters_stronglyentanglinglayer_normal(n_wires=1000, mean=-1, std=0.2, seed=seed)
        p_av = [np.mean(p_) for p_ in p]
        p_std = [np.std(p_) for p_ in p]
        assert np.isclose(p_av, [-1], atol=0.01)
        assert np.isclose(p_std, [0.2], atol=0.01)

    def test_pars_stronglyentanglinglayers_normal_dimensions(self, n_subsystems, n_layers):
        """Confirm that the pennylane.templates.utils.parameters_stronglyentanglinglayers_normal()
         returns an array with the right dimensions."""
        a = (n_layers, n_subsystems, 3)
        p = parameters_stronglyentanglinglayers_normal(n_layers=n_layers, n_wires=n_subsystems, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_pars_stronglyentanglinglayers_normal_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_stronglyentanglinglayers_normal()
        samples from the right distributions."""
        p = parameters_stronglyentanglinglayers_normal(n_layers=2, n_wires=1000, mean=-1, std=0.2, seed=seed)
        p_av = [np.mean(p_) for p_ in p]
        p_std = [np.std(p_) for p_ in p]
        assert np.isclose(p_av, [-1], atol=0.5)
        assert np.isclose(p_std, [0.2], atol=0.5)


class TestParsRandom:
    """Tests the pennylane.templates.parameters methods for a random circuit."""

    def test_pars_randomlayer_uniform_dimensions(self, n_subsystems, n_rots):
        """Confirm that the pennylane.templates.utils.parameters_randomlayer_uniform()
         returns an array with the right dimensions."""
        if n_rots is None:
            n_rots = n_subsystems
        a = (n_rots, )
        p = parameters_randomlayer_uniform(n_wires=n_subsystems, n_rots=n_rots, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_pars_randomlayer_uniform_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_randomlayer_uniform()
        samples from the right distributions."""
        p = parameters_randomlayer_uniform(n_wires=1000, uniform_min=-2, uniform_max=1, seed=seed)
        p_av = [np.mean(p_) for p_ in p]
        p_std = [np.std(p_) for p_ in p]
        assert np.isclose(p_av, [-0.5], atol=0.5)
        assert np.isclose(p_std, [0.86], atol=0.5)

    def test_pars_randomlayers_uniform_dimensions(self, n_subsystems, n_layers, n_rots):
        """Confirm that the pennylane.templates.utils.parameters_randomlayers_uniform()
         returns an array with the right dimensions."""
        if n_rots is None:
            n_rots = n_subsystems
        a = (n_layers, n_rots)
        p = parameters_randomlayers_uniform(n_layers=n_layers, n_wires=n_subsystems, n_rots=n_rots, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_pars_randomlayers_uniform_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_randomlayers_uniform()
        samples from the right distributions."""
        p = parameters_randomlayers_uniform(n_layers=2, n_wires=1000, uniform_min=-2, uniform_max=1, seed=seed)
        p_av = [np.mean(p_) for p_ in p]
        p_std = [np.std(p_) for p_ in p]
        assert np.isclose(p_av, [-0.5], atol=0.5)
        assert np.isclose(p_std, [0.86], atol=0.5)

    def test_pars_randomlayer_normal_dimensions(self, n_subsystems, n_rots):
        """Confirm that the pennylane.templates.utils.parameters_randomlayer_normal()
         returns an array with the right dimensions."""
        if n_rots is None:
            n_rots = n_subsystems
        a = (n_rots, )
        p = parameters_randomlayer_normal(n_wires=n_subsystems, n_rots=n_rots, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_pars_randomlayer_normal_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_randomlayer_normal()
        samples from the right distributions."""
        p = parameters_randomlayer_normal(n_wires=1000, mean=-1, std=0.2, seed=seed)
        p_av = [np.mean(p_) for p_ in p]
        p_std = [np.std(p_) for p_ in p]
        assert np.isclose(p_av, [-1], atol=0.5)
        assert np.isclose(p_std, [0.2], atol=0.5)

    def test_pars_randomlayers_normal_dimensions(self, n_subsystems, n_layers, n_rots):
        """Confirm that the pennylane.templates.utils.parameters_randomlayers_normal()
         returns an array with the right dimensions."""
        if n_rots is None:
            n_rots = n_subsystems
        a = (n_layers, n_rots)
        p = parameters_randomlayers_normal(n_layers=n_layers, n_wires=n_subsystems, n_rots=n_rots, seed=0)
        dims = [p_.shape for p_ in p]
        assert dims == [a]

    def test_pars_randomlayers_normal_range(self, seed):
        """Confirm that pennylane.templates.utils.parameters_randomlayers_normal()
        samples from the right distributions."""
        p = parameters_randomlayers_normal(n_layers=2, n_wires=1000, mean=-1, std=0.2, seed=seed)
        p_av = [np.mean(p_) for p_ in p]
        p_std = [np.std(p_) for p_ in p]
        assert np.isclose(p_av, [-1], atol=0.5)
        assert np.isclose(p_std, [0.2], atol=0.5)
