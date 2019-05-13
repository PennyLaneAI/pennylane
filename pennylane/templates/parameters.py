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
r"""
Parameters
==========

This module contains methods to create arrays of initial parameters that can
be used in :mod:`pennylane.templates.layers`.

.. autosummary::

    parameters_cvqnn_layers
    parameters_cvqnn_layer
    parameters_stronglyentangling_layers
    parameters_stronglyentangling_layer
    parameters_random_layers
    parameters_random_layer

"""
import numpy as np
from math import pi


def parameters_cvqnn_layers(n_layers, n_wires, uniform_min=0, uniform_max=2 * pi, mean=0, std=0.1, seed=None):
    r"""
    Create a list of randomly initialised parameter arrays for :func:`pennylane.templates.layers.CVNeuralNetLayers`.

    The number of parameters for each of the ``n_layers`` layers is either ``n_wires`` or ``n_wires*(n_wires-1)/2``,
    depending on the gate type.

    Rotation angles are initialised uniformly from the interval ``[uniform_min, uniform_max]``, while
    all other parameters are drawn from a normal distribution with mean ``mean`` and standard deviation ``std``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net
        uniform_min (float): minimum value of non-angle gate parameters
        uniform_max (float): maximum value of non-angle gate parameters
        mean (float): mean of angle gate parameters
        std (float): standard deviation of angle gate parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of eleven parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)
    n_if = n_wires * (n_wires - 1) // 2
    interval = uniform_max-uniform_min

    theta_1 = np.random.random(size=(n_layers, n_if)) * interval + uniform_min
    phi_1 = np.random.random(size=(n_layers, n_if)) * interval + uniform_min
    varphi_1 = np.random.random(size=(n_layers, n_wires)) * interval + uniform_min
    r = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    phi_r = np.random.random(size=(n_layers, n_wires)) * interval + uniform_min
    theta_2 = np.random.random(size=(n_layers, n_if)) * interval + uniform_min
    phi_2 = np.random.random(size=(n_layers, n_if)) * interval + uniform_min
    varphi_2 = np.random.random(size=(n_layers, n_wires)) * interval + uniform_min
    a = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    phi_a = np.random.random(size=(n_layers, n_wires)) * interval + uniform_min
    k = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))

    return [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]


def parameters_cvqnn_layer(n_wires, uniform_min=0, uniform_max=2 * pi, mean=0, std=0.1, seed=None):
    r"""
    Create a list of randomly initialised parameter arrays for :func:`pennylane.templates.layers.CVNeuralNetLayer`.

    The number of parameters is either ``n_wires`` or ``n_wires*(n_wires-1)/2``, depending on the gate type.

    Rotation angles are initialised uniformly from the interval ``[uniform_min, uniform_max]``, while
    all other parameters are drawn from a normal distribution with mean `mean` and standard deviation `std`.

    Args:
        n_wires (int): number of modes of the CV Neural Net
        uniform_min (float): minimum value of non-angle gate parameters
        uniform_max (float): maximum value of non-angle gate parameters
        mean (float): mean of angle gate parameters
        std (float): standard deviation of angle gate parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of eleven parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)

    n_if = n_wires * (n_wires - 1) // 2
    interval = uniform_max - uniform_min

    theta_1 = np.random.random(size=(n_if, )) * interval + uniform_min
    phi_1 = np.random.random(size=(n_if, )) * interval + uniform_min
    varphi_1 = np.random.random(size=(n_wires,)) * interval + uniform_min
    r = np.random.normal(loc=mean, scale=std, size=(n_wires,))
    phi_r = np.random.random(size=(n_wires,)) * interval + uniform_min
    theta_2 = np.random.random(size=(n_if, )) * interval + uniform_min
    phi_2 = np.random.random(size=(n_if, )) * interval + uniform_min
    varphi_2 = np.random.random(size=(n_wires,)) * interval + uniform_min
    a = np.random.normal(loc=mean, scale=std, size=(n_wires,))
    phi_a = np.random.random(size=(n_wires,)) * interval + uniform_min
    k = np.random.normal(loc=mean, scale=std, size=(n_wires,))

    return [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]


def parameters_stronglyentangling_layers(n_layers, n_wires, uniform_min=0, uniform_max=2 * pi, seed=None):
    r"""
    Create a list of randomly initialised parameter arrays for :func:`pennylane.templates.layers.StronglyEntanglingLayers`.

    The number of parameter array is ``(n_layers, n_qubits, 3)`` and each parameter is drawn uniformly at random
    from between ``uniform_min`` and ``uniform_max``. The parameters define the rotation angles in RX, RY and RZ rotations
    applied to each layer.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits
        uniform_min (float): minimum value of non-angle gate parameters
        uniform_max (float): maximum value of non-angle gate parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array of parameters
    """
    if seed is not None:
        np.random.seed(seed)

    interval = uniform_max - uniform_min
    params = np.random.random(size=(n_layers, n_wires, 3)) * interval + uniform_min
    return [params]


def parameters_stronglyentangling_layer(n_wires, uniform_min=0, uniform_max=2 * pi, seed=None):
    r"""
    Create a list of randomly initialised parameter arrays for :func:`pennylane.templates.layers.StronglyEntanglingLayers`.

    The number of parameter array is ``(n_qubits, 3)`` and each parameter is drawn uniformly at random
    from between ``uniform_min`` and ``uniform_max``. The parameters define the rotation angles in RX, RY and RZ rotations
    applied to each layer.

    Args:
        n_wires (int): number of qubits
        uniform_min (float): minimum value of non-angle gate parameters
        uniform_max (float): maximum value of non-angle gate parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array of parameters
    """
    if seed is not None:
        np.random.seed(seed)

    interval = uniform_max - uniform_min
    params = np.random.random(size=(n_wires, 3)) * interval + uniform_min
    return [params]


def parameters_random_layers(n_layers, n_wires, n_rots=None, uniform_min=0, uniform_max=2 * pi, seed=None):
    r"""
    Create a list of randomly initialised parameter arrays for :func:`pennylane.templates.layers.RandomLayers`.

    The number of parameter array is ``(n_layers, K)`` and each parameter is drawn uniformly at random
    from between ``uniform_min`` and ``uniform_max``. The parameters define the rotation angles in randomly
    positioned rotations applied in each layer.

    Args:
        n_layers (int): number of layers
        n_qubits (int): number of qubits
        n_rots (int): number of rotations, if None, n_rots = n_qubits
        uniform_min (float): minimum value of non-angle gate parameters
        uniform_max (float): maximum value of non-angle gate parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array of parameters
    """
    if seed is not None:
        np.random.seed(seed)

    if n_rots is None:
        n_rots = n_wires

    interval = uniform_max - uniform_min
    params = np.random.random(size=(n_layers, n_rots)) * interval + uniform_min
    return [params]


def parameters_random_layer(n_wires, n_rots=None, uniform_min=0, uniform_max=2 * pi, seed=None):
    r"""
    Create a list of randomly initialised parameter arrays for :func:`pennylane.templates.layers.StronglyEntanglingLayers`.

    The number of parameter array is ``(n_qubits, 3)`` and each parameter is drawn uniformly at random
    from between ``uniform_min`` and ``uniform_max``. The parameters define the rotation angles in RX, RY and RZ rotations
    applied to each layer.

    Args:
        n_wires (int): number of qubits
        n_rots (int): number of rotations, if None, n_rots = n_qubits
        uniform_min (float): minimum value of non-angle gate parameters
        uniform_max (float): maximum value of non-angle gate parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array of parameters
    """
    if seed is not None:
        np.random.seed(seed)

    if n_rots is None:
        n_rots = n_wires

    interval = uniform_max - uniform_min
    params = np.random.random(size=(n_rots,)) * interval + uniform_min
    return [params]
