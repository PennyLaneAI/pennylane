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

**Module name:** :mod:`pennylane.init`

.. currentmodule:: pennylane.init

This module contains methods to create arrays of parameters that can \
be used as initial parameters of trainable templates.

The methods return lists of numpy arrays, and the arrays have the correct shape to be fed in as the first positional
arguments in the templates.

.. note::

    For the use of PennyLane in combination with PyTorch or TensorFlow, the numpy arrays have to be converted to
    *trainable* tensors.

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
# pylint: disable=too-many-arguments
from math import pi
import numpy as np


def strong_ent_layers_uniform(n_layers, n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a list of a single parameter array for :func:`~.StronglyEntanglingLayers`, drawn from a uniform
    distribution.

    The shape of the parameter array is ``(n_layers, n_wires, 3)`` and each parameter is drawn uniformly at random \
    from between ``low`` and ``high``. The parameters define the three rotation angles
    applied in each layer.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits

    Keyword Args:
        low (float): minimum value of non-angle gate parameters
        high (float): maximum value of non-angle gate parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    params = np.random.uniform(low=low, high=high, size=(n_layers, n_wires, 3))
    return [params]


def strong_ent_layers_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a list of a single parameter array for :func:`~.StronglyEntanglingLayers`, drawn from a normal
    distribution.

    The shape of the parameter array is ``(n_layers, n_wires, 3)`` and each parameter is drawn
    from a normal distribution with mean ``mean`` and standard deviation ``std``.
    The parameters define the three rotation angles applied in each layer.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits

    Keyword Args:
        mean (float): mean of parameters
        std (float): standard deviation of parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    params = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires, 3))
    return [params]


def strong_ent_layer_uniform(n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a list of a single parameter array for :func:`~.StronglyEntanglingLayer`,
    drawn from a uniform distribution.

    The shape of the parameter array is ``(n_wires, 3)`` and each parameter is drawn uniformly at random \
    from between ``low`` and ``high``. The parameters define the three rotation angles
    applied to each layer.

    Args:
        n_wires (int): number of qubits

    Keyword Args:
        low (float): minimum value of non-angle gate parameters
        high (float): maximum value of non-angle gate parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    params = np.random.uniform(low=low, high=high, size=(n_wires, 3))
    return [params]


def strong_ent_layer_normal(n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a list of a single parameter array for :func:`~.StronglyEntanglingLayer`,
    drawn from a normal distribution.

    The shape of the parameter array is ``(n_wires, 3)`` and each parameter is drawn
    from a normal distribution with mean ``mean`` and standard deviation ``std``.
    The parameters define the three rotation angles applied to each layer.

    Args:
        n_wires (int): number of qubits

    Keyword Args:
        mean (float): mean of parameters
        std (float): standard deviation of parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    params = np.random.normal(loc=mean, scale=std, size=(n_wires, 3))
    return [params]


def random_layers_uniform(n_layers, n_wires, n_rots=None, low=0, high=2 * pi, seed=None):
    r"""Creates a list of a single parameter array for :func:`~.RandomLayers`, drawn from a uniform distribution.

    The shape of the parameter array is ``(n_layers, n_rots)`` and each parameter is drawn uniformly at random \
    from between ``low`` and ``high``. The parameters define the rotation angles of the randomly \
    positioned rotations applied in each layer.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits

    Keyword Args:
        n_rots (int): number of rotations, if ``None``, ``n_rots=n_wires``
        low (float): minimum value of non-angle gate parameters
        high (float): maximum value of non-angle gate parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    if n_rots is None:
        n_rots = n_wires

    params = np.random.uniform(low=low, high=high, size=(n_layers, n_rots))
    return [params]


def random_layers_normal(n_layers, n_wires, n_rots=None, mean=0, std=0.1, seed=None):
    r"""Creates a list of a single parameter array for :func:`~.RandomLayers`, drawn from a normal distribution.

    The shape of the parameter array is ``(n_layers, n_rots)`` and each parameter is drawn
    from a normal distribution with mean ``mean`` and standard deviation ``std``.
    The parameters define the rotation angles of the randomly positioned rotations applied in each layer.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits

    Keyword Args:
        n_rots (int): number of rotations, if ``None``, ``n_rots=n_wires``
        mean (float): mean of parameters
        std (float): standard deviation of parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    if n_rots is None:
        n_rots = n_wires

    params = np.random.normal(loc=mean, scale=std, size=(n_layers, n_rots))
    return [params]


def random_layer_uniform(n_wires, n_rots=None, low=0, high=2 * pi, seed=None):
    r"""Creates a list of a single parameter array for :func:`~.RandomLayer`, drawn from a uniform distribution.

    The number of parameter array is ``(n_rots,)`` and each parameter is drawn uniformly at random \
    from between ``low`` and ``high``. The parameters define the rotation angles of the randomly \
    positioned rotations applied in each layer.

    Args:
        n_wires (int): number of qubits

    Keyword Args:
        n_rots (int): number of rotations, if ``None``, ``n_rots=n_wires``
        low (float): minimum value of rotation angles
        high (float): maximum value of rotation angles
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    if n_rots is None:
        n_rots = n_wires

    params = np.random.uniform(low=low, high=high, size=(n_rots,))
    return [params]


def random_layer_normal(n_wires, n_rots=None, mean=0, std=0.1, seed=None):
    r"""Creates a list of a single parameter array for :func:`~.RandomLayer`, drawn from a normal distribution.

    The number of parameter array is ``(n_rots,)`` and each parameter is drawn
    from a normal distribution with mean ``mean`` and standard deviation ``std``.
    The parameters define the rotation angles of the randomly positioned rotations applied in each layer.

    Args:
        n_wires (int): number of qubits

    Keyword Args:
        n_rots (int): number of rotations, if ``None``, ``n_rots=n_wires``
        mean (float): mean of parameters
        std (float): standard deviation of parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    if n_rots is None:
        n_rots = n_wires

    params = np.random.normal(loc=mean, scale=std, size=(n_rots,))
    return [params]


def cvqnn_layers_uniform(n_layers, n_wires, low=0, high=2*pi, mean_active=0, std_active=0.1, seed=None):
    r"""Creates a list of eleven parameter arrays for :func:`~.CVNeuralNetLayers`,
    where non-active gate parameters are drawn from a uniform distribution and active parameters
    from a normal distribution.

    The shape of the arrays is ``(n_layers, n_wires*(n_wires-1)/2)`` for the parameters used in an interferometer,
    and ``(n_layers, n_wires)``  else.

    All gate parameters are drawn uniformly from the interval ``[low, high]``, except from the three types of
    'active gate parameters': the displacement amplitude, squeezing amplitude and kerr parameter. These
    active gate parameters are sampled from a normal distribution with mean ``mean_active`` and standard
    deviation ``std_active``. Since they influence the mean photon number (or energy) of the quantum system,
    one typically wants to initialize them with values close to zero.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        low (float): minimum value of uniformly drawn rotation angles
        high (float): maximum value of uniformly drawn rotation angles
        mean_active (float): mean of active gate parameters
        std_active (float): standard deviation of active gate parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)
    n_if = n_wires * (n_wires - 1) // 2

    theta_1 = np.random.uniform(low=low, high=high, size=(n_layers, n_if))
    phi_1 = np.random.uniform(low=low, high=high, size=(n_layers, n_if))
    varphi_1 = np.random.uniform(low=low, high=high, size=(n_layers, n_wires))
    r = np.random.normal(loc=mean_active, scale=std_active, size=(n_layers, n_wires))
    phi_r = np.random.uniform(low=low, high=high, size=(n_layers, n_wires))
    theta_2 = np.random.uniform(low=low, high=high, size=(n_layers, n_if))
    phi_2 = np.random.uniform(low=low, high=high, size=(n_layers, n_if))
    varphi_2 = np.random.uniform(low=low, high=high, size=(n_layers, n_wires))
    a = np.random.normal(loc=mean_active, scale=std_active, size=(n_layers, n_wires))
    phi_a = np.random.uniform(low=low, high=high, size=(n_layers, n_wires))
    k = np.random.normal(loc=mean_active, scale=std_active, size=(n_layers, n_wires))

    return [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]


def cvqnn_layers_normal(n_layers, n_wires, mean=0, std=1, mean_active=0, std_active=0.1, seed=None):
    r"""Creates a list of eleven parameter arrays for :func:`~.CVNeuralNetLayers`, where both active and non-active
    gate parameters are drawn from normal distributions.

    The shape of the arrays is ``(n_layers, n_wires*(n_wires-1)/2)`` for the parameters used in an interferometer,
    and ``(n_layers, n_wires)``  else.

    All gate parameters are drawn from a normal distribution with mean ``mean`` and standard deviation ``std``,
    except from the three types of 'active gate parameters': the displacement amplitude, squeezing amplitude and kerr
    parameter. These active gate parameters are sampled from a normal distribution with mean ``mean_active`` and
    standard deviation ``std_active``. Since they influence the mean photon number (or energy) of the quantum system,
    one typically wants to initialize them with values close to zero.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        mean (float): mean of non-active parameters
        std (float): standard deviation of non-active parameters
        mean_active (float): mean of active gate parameters
        std_active (float): standard deviation of active gate parameters
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
    r = np.random.normal(loc=mean_active, scale=std_active, size=(n_layers, n_wires))
    phi_r = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    theta_2 = np.random.normal(loc=mean, scale=std, size=(n_layers, n_if))
    phi_2 = np.random.normal(loc=mean, scale=std, size=(n_layers, n_if))
    varphi_2 = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    a = np.random.normal(loc=mean_active, scale=std_active, size=(n_layers, n_wires))
    phi_a = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    k = np.random.normal(loc=mean_active, scale=std_active, size=(n_layers, n_wires))

    return [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]


def cvqnn_layer_uniform(n_wires, low=0, high=2 * pi, mean_active=0, std_active=0.1, seed=None):
    r"""Creates a list of eleven parameter arrays for :func:`~.CVNeuralNetLayer`,
    where non-active gate parameters are drawn from a uniform distribution and active parameters
    from a normal distribution.

    The shape of the arrays is ``(n_wires*(n_wires-1)/2)`` for the parameters used in an interferometer,
    and ``(n_wires)``  else.

    All gate parameters are drawn uniformly from the interval ``[low, high]``, except from the three types of
    'active gate parameters': the displacement amplitude, squeezing amplitude and kerr parameter. These
    active gate parameters are sampled from a normal distribution with mean ``mean_active`` and standard
    deviation ``std_active``. Since they influence the mean photon number (or energy) of the quantum system,
    one typically wants to initialize them with values close to zero.

    Args:
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        low (float): minimum value of uniformly drawn non-active gate parameters
        high (float): maximum value of uniformly drawn non-active gate parameters
        mean_active (float): mean of active gate parameters
        std_active (float): standard deviation of active gate parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)

    n_if = n_wires * (n_wires - 1) // 2

    theta_1 = np.random.uniform(low=low, high=high, size=(n_if, ))
    phi_1 = np.random.uniform(low=low, high=high, size=(n_if, ))
    varphi_1 = np.random.uniform(low=low, high=high, size=(n_wires,))
    r = np.random.normal(loc=mean_active, scale=std_active, size=(n_wires,))
    phi_r = np.random.uniform(low=low, high=high, size=(n_wires,))
    theta_2 = np.random.uniform(low=low, high=high, size=(n_if, ))
    phi_2 = np.random.uniform(low=low, high=high, size=(n_if, ))
    varphi_2 = np.random.uniform(low=low, high=high, size=(n_wires,))
    a = np.random.normal(loc=mean_active, scale=std_active, size=(n_wires,))
    phi_a = np.random.uniform(low=low, high=high, size=(n_wires,))
    k = np.random.normal(loc=mean_active, scale=std_active, size=(n_wires,))

    return [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]


def cvqnn_layer_normal(n_wires, mean=0, std=1, mean_active=0, std_active=0.1, seed=None):
    r"""Creates a list of eleven parameter arrays for :func:`~.CVNeuralNetLayer`, where both active and non-active
    gate parameters are drawn from normal distributions.

    The shape of the arrays is ``(n_wires*(n_wires-1)/2)`` for the parameters used in an interferometer,
    and ``(n_wires)``  else.

    All gate parameters are drawn from a normal distribution with mean ``mean`` and standard deviation ``std``,
    except from the three types of 'active gate parameters': the displacement amplitude, squeezing amplitude and kerr
    parameter. These active gate parameters are sampled from a normal distribution with mean ``mean_active`` and
    standard deviation ``std_active``. Since they influence the mean photon number (or energy) of the quantum system,
    one typically wants to initialize them with values close to zero.

    Args:
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        mean (float): mean of non-active parameters
        std (float): standard deviation of non-active parameters
        mean_active (float): mean of active gate parameters
        std_active (float): standard deviation of active gate parameters
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
    r = np.random.normal(loc=mean_active, scale=std_active, size=(n_wires,))
    phi_r = np.random.normal(loc=mean, scale=std, size=(n_wires,))
    theta_2 = np.random.normal(loc=mean, scale=std, size=(n_if,))
    phi_2 = np.random.normal(loc=mean, scale=std, size=(n_if,))
    varphi_2 = np.random.normal(loc=mean, scale=std, size=(n_wires,))
    a = np.random.normal(loc=mean_active, scale=std_active, size=(n_wires,))
    phi_a = np.random.normal(loc=mean, scale=std, size=(n_wires,))
    k = np.random.normal(loc=mean_active, scale=std_active, size=(n_wires,))

    return [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]


def interferometer_uniform(n_wires, low=0, high=2 * pi, seed=None):
    r"""Returns a list of three parameter arrays of the form ``[theta, phi, varphi]``, where:

    * ``theta`` is the array of beamsplitter transmittivity angles, of size ``(n_wires*(n_wires-1)/2, )``

    * ``phi`` is the array of beamsplitter phases, of size ``(n_wires*(n_wires-1)/2, )``

    * ``varphi`` is the array of local angles for the final rotation gates, of size ``(n_wires, )``
 
    All parameters are initialized uniformly from the interval ``[low, high]``.

    Args:
        n_wires (int): number of modes that the interferometer acts on

    Keyword Args:
        low (float): minimum value of uniformly drawn rotation angles
        high (float): maximum value of uniformly drawn rotation angles
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)
    n_if = n_wires * (n_wires - 1) // 2

    theta = np.random.uniform(low=low, high=high, size=(n_if,))
    phi = np.random.uniform(low=low, high=high, size=(n_if,))
    varphi = np.random.uniform(low=low, high=high, size=(n_wires,))

    return [theta, phi, varphi]


def interferometer_normal(n_wires, mean=0, std=0.1, seed=None):
    r"""Returns a list of three parameter arrays of the form ``[theta, phi, varphi]``, where:

    * ``theta`` is the array of beamsplitter transmittivity angles, of size ``(n_wires*(n_wires-1)/2, )``

    * ``phi`` is the array of beamsplitter phases, of size ``(n_wires*(n_wires-1)/2, )``

    * ``varphi`` is the array of local angles for the final rotation gates, of size ``(n_wires, )``
   
    All parameters are drawn from a normal distribution with mean ``mean`` and standard deviation ``std``.

    Args:
        n_wires (int): number of modes that the interferometer acts on

    Keyword Args:
        mean (float): mean of parameters
        std (float): standard deviation of parameters
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
