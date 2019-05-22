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
Parameter Initializations
=========================

**Module name:** :mod:`pennylane.templates.parameters`

.. currentmodule:: pennylane.templates.parameters

This module contains methods to create arrays of initial parameters that can \
be used in trainable templates.

Qubit architectures
-------------------

Strongly entangling circuit
***************************

.. autosummary::

    strong_ent_layers_uniform
    strong_ent_layers_normal
    strong_ent_layer_uniform
    strong_ent_layer_normal

Random circuit
**************

.. autosummary::

    random_layers_uniform
    random_layers_normal
    random_layer_uniform
    random_layer_normal

Continuous-variable architectures
---------------------------------

Continuous-variable quantum neural network
******************************************

.. autosummary::

    cvqnn_layers_uniform
    cvqnn_layers_normal
    cvqnn_layer_uniform
    cvqnn_layer_normal

Interferometer
**************

.. autosummary::

    interferometer_uniform
    interferometer_normal

Code details
^^^^^^^^^^^^
"""
import numpy as np
from math import pi


def strong_ent_layers_uniform(n_layers, n_wires, uniform_min=0, uniform_max=2 * pi, seed=None):
    r"""
    Creates a list of one randomly initialized parameter array for \
    :func:`~.StronglyEntanglingLayers`, sampled uniformly.

    The shape of the parameter array is ``(n_layers, n_wires, 3)`` and each parameter is drawn uniformly at random \
    from between ``uniform_min`` and ``uniform_max``. The parameters define the three rotation angles
    applied in each layer.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits

    Keyword Args:
        uniform_min (float): minimum value of non-angle gate parameters
        uniform_max (float): maximum value of non-angle gate parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)

    interval = uniform_max - uniform_min
    params = np.random.random(size=(n_layers, n_wires, 3)) * interval + uniform_min
    return [params]


def strong_ent_layers_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""
    Creates a list of one randomly initialized parameter array for \
    :func:`~.StronglyEntanglingLayers`, sampled from a normal distribution.

    The shape of the parameter array is ``(n_layers, n_wires, 3)`` and each parameter is drawn
    from a normal distribution with mean ``mean`` and standard deviation ``std``.
    The parameters define the three rotation angles applied in each layer.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits

    Keyword Args:
        mean (float): mean of initial parameters
        std (float): standard deviation of initial parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)

    params = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires, 3))
    return [params]


def strong_ent_layer_uniform(n_wires, uniform_min=0, uniform_max=2 * pi, seed=None):
    r"""
    Creates a list of one randomly initialized parameter array for \
    :func:`~.StronglyEntanglingLayers`, sampled uniformly.

    The shape of the parameter array is ``(n_wires, 3)`` and each parameter is drawn uniformly at random \
    from between ``uniform_min`` and ``uniform_max``. The parameters define the three rotation angles
    applied to each layer.

    Args:
        n_wires (int): number of qubits

    Keyword Args:
        uniform_min (float): minimum value of non-angle gate parameters
        uniform_max (float): maximum value of non-angle gate parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)

    interval = uniform_max - uniform_min
    params = np.random.random(size=(n_wires, 3)) * interval + uniform_min
    return [params]


def strong_ent_layer_normal(n_wires, mean=0, std=0.1, seed=None):
    r"""
    Creates a list of one randomly initialized parameter array for \
    :func:`~.StronglyEntanglingLayers`, sampled from a normal distribution.

    The shape of the parameter array is ``(n_wires, 3)`` and each parameter is drawn
    from a normal distribution with mean ``mean`` and standard deviation ``std``.
    The parameters define the three rotation angles applied to each layer.

    Args:
        n_wires (int): number of qubits

    Keyword Args:
        mean (float): mean of initial parameters
        std (float): standard deviation of initial parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)

    params = np.random.normal(loc=mean, scale=std, size=(n_wires, 3))
    return [params]


def random_layers_uniform(n_layers, n_wires, n_rots=None, uniform_min=0, uniform_max=2 * pi, seed=None):
    r"""
    Creates a list of one randomly initialized parameter array for :func:`~.RandomLayers`, sampled uniformly.

    The shape of the parameter array is ``(n_layers, n_rots)`` and each parameter is drawn uniformly at random \
    from between ``uniform_min`` and ``uniform_max``. The parameters define the rotation angles of the randomly \
    positioned rotations applied in each layer.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits

    Keyword Args:
        n_rots (int): number of rotations, if ``None``, ``n_rots=n_wires``
        uniform_min (float): minimum value of non-angle gate parameters
        uniform_max (float): maximum value of non-angle gate parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)

    if n_rots is None:
        n_rots = n_wires

    interval = uniform_max - uniform_min
    params = np.random.random(size=(n_layers, n_rots)) * interval + uniform_min
    return [params]


def random_layers_normal(n_layers, n_wires, n_rots=None, mean=0, std=0.1, seed=None):
    r"""
    Creates a list of one randomly initialized parameter array for :func:`~.RandomLayers`, sampled from a normal distribution.

    The shape of the parameter array is ``(n_layers, n_rots)`` and each parameter is drawn
    from a normal distribution with mean ``mean`` and standard deviation ``std``.
    The parameters define the rotation angles of the randomly positioned rotations applied in each layer.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits

    Keyword Args:
        n_rots (int): number of rotations, if ``None``, ``n_rots=n_wires``
        mean (float): mean of initial parameters
        std (float): standard deviation of initial parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)

    if n_rots is None:
        n_rots = n_wires

    params = np.random.normal(loc=mean, scale=std, size=(n_layers, n_rots))
    return [params]


def random_layer_uniform(n_wires, n_rots=None, uniform_min=0, uniform_max=2 * pi, seed=None):
    r"""
    Creates a list of one randomly initialized parameter array for :func:`~.RandomLayer`, sampled uniformly.

    The number of parameter array is ``(n_rots,)`` and each parameter is drawn uniformly at random \
    from between ``uniform_min`` and ``uniform_max``. The parameters define the rotation angles of the randomly \
    positioned rotations applied in each layer.

    Args:
        n_wires (int): number of qubits

    Keyword Args:
        n_rots (int): number of rotations, if ``None``, ``n_rots=n_wires``
        uniform_min (float): minimum value of rotation angles
        uniform_max (float): maximum value of rotation angles
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)

    if n_rots is None:
        n_rots = n_wires

    interval = uniform_max - uniform_min
    params = np.random.random(size=(n_rots,)) * interval + uniform_min
    return [params]


def random_layer_normal(n_wires, n_rots=None, mean=0, std=0.1, seed=None):
    r"""
    Creates a list of one randomly initialized parameter array for :func:`~.RandomLayer`, sampled from a normal distribution.

    The number of parameter array is ``(n_rots,)`` and each parameter is drawn
    from a normal distribution with mean ``mean`` and standard deviation ``std``.
    The parameters define the rotation angles of the randomly positioned rotations applied in each layer.

    Args:
        n_wires (int): number of qubits

    Keyword Args:
        n_rots (int): number of rotations, if ``None``, ``n_rots=n_wires``
        mean (float): mean of initial parameters
        std (float): standard deviation of initial parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)

    if n_rots is None:
        n_rots = n_wires

    params = np.random.normal(loc=mean, scale=std, size=(n_rots,))
    return [params]


def cvqnn_layers_uniform(n_layers, n_wires, uniform_min=0, uniform_max=2 * pi, mean=0, std=0.1, seed=None):
    r"""
    Creates a list of eleven randomly initialized parameter arrays for the positional arguments in \
    :func:`~.CVNeuralNetLayers`, sampled uniformly.

    The shape of the arrays is either ``(n_layers, n_wires)`` or ``(n_layers, n_wires*(n_wires-1)/2)``.

    Rotation angles are initialized uniformly from the interval ``[uniform_min, uniform_max]``, while \
    all other parameters are drawn from a normal distribution with mean ``mean`` and standard deviation ``std``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        uniform_min (float): minimum value of uniformly drawn rotation angles
        uniform_max (float): maximum value of uniformly drawn rotation angles
        mean (float): mean of other gate parameters
        std (float): standard deviation of other gate parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
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

    return theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k


def cvqnn_layers_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""
    Creates a list of eleven randomly initialized parameter arrays for the positional arguments in \
    :func:`~.CVNeuralNetLayers`, sampled from a normal distribution.

    The shape of the arrays is either ``(n_layers, n_wires)`` or ``(n_layers, n_wires*(n_wires-1)/2)``.

    All parameters are drawn from a normal distribution with mean ``mean`` and standard deviation ``std``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        mean (float): mean of initial parameters
        std (float): standard deviation of initial parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)
    n_if = n_wires * (n_wires - 1) // 2

    theta_1 = np.random.normal(loc=mean, scale=std, size=(n_layers, n_if))
    phi_1 = np.random.normal(loc=mean, scale=std, size=(n_layers, n_if))
    varphi_1 = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    r = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    phi_r = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    theta_2 = np.random.normal(loc=mean, scale=std, size=(n_layers, n_if))
    phi_2 = np.random.normal(loc=mean, scale=std, size=(n_layers, n_if))
    varphi_2 = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    a = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    phi_a = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    k = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))

    return [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]


def cvqnn_layer_uniform(n_wires, uniform_min=0, uniform_max=2 * pi, mean=0, std=0.1, seed=None):
    r"""
    Creates a list of eleven randomly initialized parameter arrays for the positional arguments in \
    :func:`~.CVNeuralNetLayer`, sampled uniformly.

    The shape of the arrays is either ``(n_wires,)`` or ``(n_wires*(n_wires-1)/2,)``.

    Rotation angles are initialized uniformly from the interval ``[uniform_min, uniform_max]``, while \
    all other parameters are drawn from a normal distribution with mean ``mean`` and standard deviation ``std``.

    Args:
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        uniform_min (float): minimum value of uniformly drawn rotation angles
        uniform_max (float): maximum value of uniformly drawn rotation angles
        mean (float): mean of other gate parameters
        std (float): standard deviation of other gate parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
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


def cvqnn_layer_normal(n_wires, mean=0, std=0.1, seed=None):
    r"""
    Creates a list of eleven randomly initialized parameter arrays for the positional arguments in \
    :func:`~.CVNeuralNetLayer`, sampled from a normal distribution.

    The shape of the arrays is either ``(n_wires,)`` or ``(n_wires*(n_wires-1)/2,)``.

    All parameters are drawn from a normal distribution with mean ``mean`` and standard deviation ``std``.

    Args:
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        mean (float): mean of initial parameters
        std (float): standard deviation of initial parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)

    n_if = n_wires * (n_wires - 1) // 2

    theta_1 = np.random.normal(loc=mean, scale=std, size=(n_if,))
    phi_1 = np.random.normal(loc=mean, scale=std, size=(n_if,))
    varphi_1 = np.random.normal(loc=mean, scale=std, size=(n_wires,))
    r = np.random.normal(loc=mean, scale=std, size=(n_wires,))
    phi_r = np.random.normal(loc=mean, scale=std, size=(n_wires,))
    theta_2 = np.random.normal(loc=mean, scale=std, size=(n_if,))
    phi_2 = np.random.normal(loc=mean, scale=std, size=(n_if,))
    varphi_2 = np.random.normal(loc=mean, scale=std, size=(n_wires,))
    a = np.random.normal(loc=mean, scale=std, size=(n_wires,))
    phi_a = np.random.normal(loc=mean, scale=std, size=(n_wires,))
    k = np.random.normal(loc=mean, scale=std, size=(n_wires,))

    return [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]


def interferometer_uniform(n_wires, uniform_min=0, uniform_max=2 * pi, seed=None):
    r"""
    Creates a list of three randomly initialized parameter arrays for \
    :func:`~.Interferometer`, sampled uniformly.

    The shape of the arrays is either ``(n_wires,)`` or ``(n_wires*(n_wires-1)/2,)``.

    The parameters are initialized uniformly from the interval ``[uniform_min, uniform_max]``.

    Args:
        n_wires (int): number of modes that the interferometer acts on

    Keyword Args:
        uniform_min (float): minimum value of uniformly drawn rotation angles
        uniform_max (float): maximum value of uniformly drawn rotation angles
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)
    n_if = n_wires * (n_wires - 1) // 2
    interval = uniform_max-uniform_min

    theta = np.random.random(size=(n_if,)) * interval + uniform_min
    phi = np.random.random(size=(n_if,)) * interval + uniform_min
    varphi = np.random.random(size=(n_wires,)) * interval + uniform_min

    return [theta, phi, varphi]


def interferometer_normal(n_wires, mean=0, std=0.1, seed=None):
    r"""
    Creates a list of three randomly initialized parameter arrays for \
    :func:`~.Interferometer`, sampled from a normal distribution.

    The shape of the arrays is either ``(n_wires,)`` or ``(n_wires*(n_wires-1)/2,)``.

    All parameters are drawn from a normal distribution with mean ``mean`` and standard deviation ``std``.

    Args:
        n_wires (int): number of modes that the interferometer acts on

    Keyword Args:
        mean (float): mean of initial parameters
        std (float): standard deviation of initial parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)
    n_if = n_wires * (n_wires - 1) // 2

    theta = np.random.normal(loc=mean, scale=std, size=(n_if,))
    phi = np.random.normal(loc=mean, scale=std, size=(n_if,))
    varphi = np.random.normal(loc=mean, scale=std, size=(n_wires,))

    return [theta, phi, varphi]
