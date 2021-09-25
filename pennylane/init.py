# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains functions that generate initial parameters, for example
to use in templates.
"""
# pylint: disable=too-many-arguments
from math import pi
from pennylane import numpy as np


def particle_conserving_u2_uniform(n_layers, n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a parameter array for :func:`~.ParticleConservingU2`, drawn from a uniform
    distribution.
    Each parameter is drawn uniformly at random from the half-open interval [``low``, ``high``).
    The parameters define the trainable angles entering the Z rotation
    :math:`R_\mathrm{z}(\vec{\theta})` and particle-conserving gate :math:`U_{2,\mathrm{ex}}`
    implemented by the :func:`~.u2_ex_gate()`.
    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits
        low (float): lower endpoint of the parameter interval
        high (float): upper endpoint of the parameter interval
        seed (int): seed used in sampling the parameters, makes function call deterministic
    Returns:
        array: parameter array
    """

    if seed is not None:
        np.random.seed(seed)

    if n_wires < 2:
        raise ValueError(
            "The number of qubits must be greater than one; got 'n_wires' = {}".format(n_wires)
        )

    params = np.random.uniform(low=low, high=high, size=(n_layers, 2 * n_wires - 1))
    return params


def particle_conserving_u2_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for :func:`~.ParticleConservingU2`, drawn from a normal
    distribution.
    Each parameter is drawn from a normal distribution with ``mean`` and standard deviation ``std``.
    The parameters define the trainable angles entering the Z rotation
    :math:`R_\mathrm{z}(\vec{\theta})` and particle-conserving gate :math:`U_{2,\mathrm{ex}}`
    implemented by the :func:`~.u2_ex_gate()`.
    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits
        mean (float): mean of parameters
        std (float): standard deviation of parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic
    Returns:
        array: parameter array
    """

    if seed is not None:
        np.random.seed(seed)

    if n_wires < 2:
        raise ValueError(
            "The number of qubits must be greater than one; got 'n_wires' = {}".format(n_wires)
        )

    params = np.random.normal(loc=mean, scale=std, size=(n_layers, 2 * n_wires - 1))
    return params


def particle_conserving_u1_uniform(n_layers, n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a parameter array for :func:`~.ParticleConservingU1`, drawn from a uniform
    distribution.
    Each parameter is drawn uniformly at random from the half-open interval [``low``, ``high``).
    The parameters define the trainable angles entering the particle-conserving
    exchange gates :math:`U_{1,\mathrm{ex}}(\phi, \theta)` implemented by the
    :func:`~.u1_ex_gate()`.
    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits
        low (float): lower endpoint of the parameter interval
        high (float): upper endpoint of the parameter interval
        seed (int): seed used in sampling the parameters, makes function call deterministic
    Returns:
        array: parameter array
    """

    if seed is not None:
        np.random.seed(seed)

    if n_wires < 2:
        raise ValueError(
            "The number of qubits must be greater than one; got 'n_wires' = {}".format(n_wires)
        )

    params = np.random.uniform(low=low, high=high, size=(n_layers, n_wires - 1, 2))
    return params


def particle_conserving_u1_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for :func:`~.ParticleConservingU1`, drawn from a normal
    distribution.
    Each parameter is drawn from a normal distribution with ``mean`` and standard deviation ``std``.
    The parameters define the trainable angles entering the particle-conserving
    exchange gates :math:`U_{1,\mathrm{ex}}(\phi, \theta)` implemented by the
    :func:`~.u1_ex_gate()`.
    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits
        mean (float): mean of parameters
        std (float): standard deviation of parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic
    Returns:
        array: parameter array
    """

    if seed is not None:
        np.random.seed(seed)

    if n_wires < 2:
        raise ValueError(
            "The number of qubits must be greater than one; got 'n_wires' = {}".format(n_wires)
        )

    params = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires - 1, 2))
    return params


def qaoa_embedding_uniform(n_layers, n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a parameter array for :func:`~.QAOAEmbedding`, drawn from a uniform
    distribution.

    Each parameter is drawn uniformly at random
    from between ``low`` and ``high``. The parameters define the trainable angles of 'ZZ interactions' and
    the 'local fields'.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits
        low (float): minimum value of uniform distribution
        high (float): maximum value of uniform distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    if n_wires == 1:
        shp = (n_layers, 1)
    elif n_wires == 2:
        shp = (n_layers, 3)
    else:
        shp = (n_layers, 2 * n_wires)

    params = np.random.uniform(low=low, high=high, size=shp)
    return params


def qaoa_embedding_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for :func:`~.QAOAEmbedding`, drawn from a normal
    distribution.

    Each parameter is drawn from a normal
    distribution with ``mean`` and ``variance``. The parameters define the the trainable angles of
    'ZZ interactions' and the 'local fields' in the template.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits
        mean (float): mean of parameters
        std (float): standard deviation of parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    if n_wires == 1:
        shp = (n_layers, 1)
    elif n_wires == 2:
        shp = (n_layers, 3)
    else:
        shp = (n_layers, 2 * n_wires)

    params = np.random.normal(loc=mean, scale=std, size=shp)
    return params


def strong_ent_layers_uniform(n_layers, n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a parameter array for :func:`~.StronglyEntanglingLayers`, drawn from a uniform
    distribution.

    The shape of the parameter array is ``(n_layers, n_wires, 3)`` and each parameter is drawn uniformly at random \
    from between ``low`` and ``high``. The parameters define the three rotation angles
    applied in each layer.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits

    Keyword Args:
        low (float): minimum value of uniform distribution
        high (float): maximum value of uniform distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    params = np.random.uniform(low=low, high=high, size=(n_layers, n_wires, 3))
    return params


def strong_ent_layers_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for :func:`~.StronglyEntanglingLayers`, drawn from a normal
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
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    params = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires, 3))
    return params


def random_layers_uniform(n_layers, n_wires, n_rots=None, low=0, high=2 * pi, seed=None):
    r"""Creates a parameter array for :func:`~.RandomLayers`, drawn from a uniform distribution.

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
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    # set default
    if n_rots is None:
        n_rots = n_wires

    # no circuit if there are no wires
    if n_wires == 0:
        n_rots = 0

    params = np.random.uniform(low=low, high=high, size=(n_layers, n_rots))
    return params


def random_layers_normal(n_layers, n_wires, n_rots=None, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for :func:`~.RandomLayers`, drawn from a normal distribution.

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
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    # set default
    if n_rots is None:
        n_rots = n_wires

    # no circuit if there are no wires
    if n_wires == 0:
        n_rots = 0

    params = np.random.normal(loc=mean, scale=std, size=(n_layers, n_rots))
    return params


def cvqnn_layers_all(n_layers, n_wires, seed=None):
    r"""Creates a list of all eleven parameter arrays for :func:`~.CVNeuralNetLayers`.

    The template contains active gates (``Squeezing``, ``Displacement`` and ``Kerr`` gates), while
    all other gates are passive.
    Active gates change the photon number (and hence the energy) of the system, and are
    therefore drawn from a normal distribution with mean :math:`0` and a small
    standard deviation of :math:`0.1`.
    Non-active gate parameters are angles and drawn from a uniform distribution with interval :math:`[0, 2\pi]`.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)

    kwargs = {"n_layers": n_layers, "n_wires": n_wires, "seed": seed}

    theta_1 = cvqnn_layers_theta_uniform(**kwargs)
    phi_1 = cvqnn_layers_phi_uniform(**kwargs)
    varphi_1 = cvqnn_layers_varphi_uniform(**kwargs)
    r = cvqnn_layers_r_normal(**kwargs)
    phi_r = cvqnn_layers_phi_r_uniform(**kwargs)
    theta_2 = cvqnn_layers_theta_uniform(**kwargs)
    phi_2 = cvqnn_layers_phi_uniform(**kwargs)
    varphi_2 = cvqnn_layers_varphi_uniform(**kwargs)
    a = cvqnn_layers_a_normal(**kwargs)
    phi_a = cvqnn_layers_phi_a_uniform(**kwargs)
    k = cvqnn_layers_kappa_normal(**kwargs)

    return [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]


def cvqnn_layers_theta_uniform(n_layers, n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a parameter array for the ``theta`` input to the interferometers of :func:`~.CVNeuralNetLayers`.

    The parameters are drawn from a uniform distribution.

    The shape of the arrays is ``(n_layers, n_wires*(n_wires-1)/2)``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        low (float): minimum value of uniform distribution
        high (float): maximum value of uniform distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    n_if = n_wires * (n_wires - 1) // 2
    theta = np.random.uniform(low=low, high=high, size=(n_layers, n_if))
    return theta


def cvqnn_layers_theta_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for the ``theta`` input to the interferometers of :func:`~.CVNeuralNetLayers`.

    The parameters are drawn from a normal distribution.

    The shape of the array is ``(n_layers, n_wires*(n_wires-1)/2)``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        mean (float): mean of normal distribution
        std (float): standard deviation of normal distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    n_if = n_wires * (n_wires - 1) // 2
    theta = np.random.normal(loc=mean, scale=std, size=(n_layers, n_if))
    return theta


def cvqnn_layers_phi_uniform(n_layers, n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a parameter array for the ``phi`` input to the interferometers of :func:`~.CVNeuralNetLayers`.

    The parameters are drawn from a uniform distribution.

    The shape of the arrays is ``(n_layers, n_wires*(n_wires-1)/2)``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        low (float): minimum value of uniform distribution
        high (float): maximum value of uniform distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    n_if = n_wires * (n_wires - 1) // 2
    phi = np.random.uniform(low=low, high=high, size=(n_layers, n_if))
    return phi


def cvqnn_layers_phi_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for the ``phi`` input to the interferometers of :func:`~.CVNeuralNetLayers`.

    The parameters are drawn from a normal distribution.

    The shape of the array is ``(n_layers, n_wires*(n_wires-1)/2)``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        mean (float): mean of normal distribution
        std (float): standard deviation of normal distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    n_if = n_wires * (n_wires - 1) // 2
    phi = np.random.normal(loc=mean, scale=std, size=(n_layers, n_if))
    return phi


def cvqnn_layers_varphi_uniform(n_layers, n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a parameter array for the ``varphi`` input to the interferometers of :func:`~.CVNeuralNetLayers`.

    The parameters are drawn from a uniform distribution.

    The shape of the arrays is ``(n_layers, n_wires)``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        low (float): minimum value of uniform distribution
        high (float): maximum value of uniform distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    varphi = np.random.uniform(low=low, high=high, size=(n_layers, n_wires))
    return varphi


def cvqnn_layers_varphi_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for the ``varphi`` input to the interferometers of :func:`~.CVNeuralNetLayers`.

    The parameters are drawn from a normal distribution.

    The shape of the arrays is ``(n_layers, n_wires)``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        mean(float): mean of normal distribution
        std(float): standard deviation of normal distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    varphi = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    return varphi


def cvqnn_layers_r_uniform(n_layers, n_wires, low=0, high=0.1, seed=None):
    r"""Creates a parameter array for the squeezing amplitude ``r`` of :func:`~.CVNeuralNetLayers`.

    The parameters are drawn from a uniform distribution.

    The shape of the arrays is ``(n_layers, n_wires)``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        low (float): minimum value of uniform distribution
        high (float): maximum value of uniform distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    r = np.random.uniform(low=low, high=high, size=(n_layers, n_wires))
    return r


def cvqnn_layers_r_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for the squeezing amplitude ``r`` of :func:`~.CVNeuralNetLayers`.

    The parameters are drawn from a normal distribution.

    The shape of the arrays is ``(n_layers, n_wires)``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        mean(float): mean of normal distribution
        std(float): standard deviation of normal distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    r = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    return r


def cvqnn_layers_phi_r_uniform(n_layers, n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a parameter array for the squeezing phase ``phi_r`` of :func:`~.CVNeuralNetLayers`.

    The parameters are drawn from a uniform distribution.

    The shape of the arrays is ``(n_layers, n_wires)``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        low (float): minimum value of uniform distribution
        high (float): maximum value of uniform distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    phi_r = np.random.uniform(low=low, high=high, size=(n_layers, n_wires))
    return phi_r


def cvqnn_layers_phi_r_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for the squeezing phase ``phi_r`` of :func:`~.CVNeuralNetLayers`.

    The parameters are drawn from a normal distribution.

    The shape of the arrays is ``(n_layers, n_wires)``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        mean(float): mean of normal distribution
        std(float): standard deviation of normal distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    phi_r = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    return phi_r


def cvqnn_layers_a_uniform(n_layers, n_wires, low=0, high=0.1, seed=None):
    r"""Creates a parameter array for the displacement amplitude ``a`` of :func:`~.CVNeuralNetLayers`.

    The parameters are drawn from a uniform distribution.

    The shape of the arrays is ``(n_layers, n_wires)``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        low (float): minimum value of uniform distribution
        high (float): maximum value of uniform distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    a = np.random.uniform(low=low, high=high, size=(n_layers, n_wires))
    return a


def cvqnn_layers_a_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for the displacement amplitude ``a`` of :func:`~.CVNeuralNetLayers`.

    The parameters are drawn from a normal distribution.

    The shape of the arrays is ``(n_layers, n_wires)``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        mean(float): mean of normal distribution
        std(float): standard deviation of normal distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    a = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    return a


def cvqnn_layers_phi_a_uniform(n_layers, n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a parameter array for the displacement phase ``phi_a`` of :func:`~.CVNeuralNetLayers`.

    The parameters are drawn from a uniform distribution.

    The shape of the arrays is ``(n_layers, n_wires)``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        low (float): minimum value of uniform distribution
        high (float): maximum value of uniform distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    phi_a = np.random.uniform(low=low, high=high, size=(n_layers, n_wires))
    return phi_a


def cvqnn_layers_phi_a_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for the displacement phase ``phi_a`` of :func:`~.CVNeuralNetLayers`.

    The parameters are drawn from a normal distribution.

    The shape of the arrays is ``(n_layers, n_wires)``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        mean(float): mean of normal distribution
        std(float): standard deviation of normal distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    phi_a = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    return phi_a


def cvqnn_layers_kappa_uniform(n_layers, n_wires, low=0, high=0.1, seed=None):
    r"""Creates a parameter array for the kerr parameter ``kappa`` of :func:`~.CVNeuralNetLayers`.

    The parameters are drawn from a uniform distribution.

    The shape of the arrays is ``(n_layers, n_wires)``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        low (float): minimum value of uniform distribution
        high (float): maximum value of uniform distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    kappa = np.random.uniform(low=low, high=high, size=(n_layers, n_wires))
    return kappa


def cvqnn_layers_kappa_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for the kerr parameter ``kappa`` of :func:`~.CVNeuralNetLayers`.

    The parameters are drawn from a normal distribution.

    The shape of the arrays is ``(n_layers, n_wires)``.

    Args:
        n_layers (int): number of layers of the CV Neural Net
        n_wires (int): number of modes of the CV Neural Net

    Keyword Args:
        mean(float): mean of normal distribution
        std(float): standard deviation of normal distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    kappa = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))
    return kappa


def interferometer_all(n_wires, seed=None):
    r"""Creates a list of arrays for the three initial parameters of :func:`Interferometer`, all drawn from a uniform
    distribution with interval :math:`[0, 2\pi]`.

    * ``theta`` is the array of beamsplitter transmittivity angles, of size ``(n_wires*(n_wires-1)/2, )``

    * ``phi`` is the array of beamsplitter phases, of size ``(n_wires*(n_wires-1)/2, )``

    * ``varphi`` is the array of local angles for the final rotation gates, of size ``(n_wires, )``

    Args:
        n_wires (int): number of modes that the interferometer acts on

    Keyword Args:
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        list of parameter arrays
    """
    if seed is not None:
        np.random.seed(seed)

    kwargs = {"n_wires": n_wires, "seed": seed}

    theta = interferometer_theta_uniform(**kwargs)
    phi = interferometer_phi_uniform(**kwargs)
    varphi = interferometer_varphi_uniform(**kwargs)

    return [theta, phi, varphi]


def interferometer_theta_uniform(n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a parameter array for the ``theta`` input of :func:`Interferometer`, drawn from a uniform
    distribution.

    The array has shape ``(n_wires*(n_wires-1)/2, )``.

    Args:
        n_wires (int): number of modes that the interferometer acts on

    Keyword Args:
        low (float): minimum value of uniform distribution
        high (float): maximum value of uniform distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)
    n_if = n_wires * (n_wires - 1) // 2

    theta = np.random.uniform(low=low, high=high, size=(n_if,))
    return theta


def interferometer_phi_uniform(n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a parameter array for the ``phi`` input of :func:`Interferometer`, drawn from a uniform
    distribution.

    The array has shape ``(n_wires*(n_wires-1)/2, )``.

    Args:
        n_wires (int): number of modes that the interferometer acts on

    Keyword Args:
        low (float): minimum value of uniform distribution
        high (float): maximum value of uniform distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)
    n_if = n_wires * (n_wires - 1) // 2

    phi = np.random.uniform(low=low, high=high, size=(n_if,))
    return phi


def interferometer_varphi_uniform(n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a parameter array for the ``varphi`` input of :func:`Interferometer`, drawn from a uniform
    distribution.

    The array has shape ``(n_wires, )``.

    Args:
        n_wires (int): number of modes that the interferometer acts on

    Keyword Args:
        low (float): minimum value of uniform distribution
        high (float): maximum value of uniform distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    varphi = np.random.uniform(low=low, high=high, size=(n_wires,))
    return varphi


def interferometer_theta_normal(n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for the ``theta`` input of :func:`Interferometer`, drawn from a normal
    distribution.

    The array has shape ``(n_wires*(n_wires-1)/2, )``.

    Args:
        n_wires (int): number of modes that the interferometer acts on

    Keyword Args:
        mean(float): mean of normal distribution
        std(float): standard deviation of normal distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)
    n_if = n_wires * (n_wires - 1) // 2

    theta = np.random.normal(loc=mean, scale=std, size=(n_if,))
    return theta


def interferometer_phi_normal(n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for the ``phi`` input of :func:`Interferometer`, drawn from a normal
    distribution.

    The array has shape ``(n_wires*(n_wires-1)/2, )``.

    Args:
        n_wires (int): number of modes that the interferometer acts on

    Keyword Args:
        mean(float): mean of normal distribution
        std(float): standard deviation of normal distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)
    n_if = n_wires * (n_wires - 1) // 2

    phi = np.random.normal(loc=mean, scale=std, size=(n_if,))
    return phi


def interferometer_varphi_normal(n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for the ``varphi`` input of :func:`Interferometer`, drawn from a normal
    distribution.

    The array has shape ``(n_wires, )``.

    Args:
        n_wires (int): number of modes that the interferometer acts on

    Keyword Args:
        mean(float): mean of normal distribution
        std(float): standard deviation of normal distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    varphi = np.random.normal(loc=mean, scale=std, size=(n_wires,))
    return varphi


def simplified_two_design_initial_layer_uniform(n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a parameter array for the ``initial_layer`` argument of :func:`~.SimplifiedTwoDesign`,
    drawn from a uniform distribution.

    The shape of the parameter array is ``(n_wires,)`` and each parameter is drawn uniformly at random \
    from between ``low`` and ``high``. The parameters define the Pauli-Y rotation angles
    applied in the initial layer.

    Args:
        n_wires (int): number of qubits
        low (float): minimum value of uniform distribution
        high (float): maximum value of uniform distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    params = np.random.uniform(low=low, high=high, size=(n_wires,))
    return params


def simplified_two_design_initial_layer_normal(n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for the ``initial_layer`` argument of :func:`~.SimplifiedTwoDesign`,
    drawn from a uniform distribution.

    The shape of the parameter array is ``(n_wires,)`` and each parameter is drawn
    from a normal distribution with mean ``mean`` and standard deviation ``std``.
    The parameters define the Pauli-Y rotation angles
    applied in the initial layer.

    Args:
        n_wires (int): number of qubits
        mean (float): mean of parameters
        std (float): standard deviation of parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    params = np.random.normal(loc=mean, scale=std, size=(n_wires,))
    return params


def simplified_two_design_weights_uniform(n_layers, n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a parameter array for the ``weights`` argument of :func:`~.SimplifiedTwoDesign`,
    drawn from a uniform distribution.

    The shape of the parameter array is ``(n_layers, n_wires - 1, 2)``
    and each parameter is drawn uniformly at random \
    from between ``low`` and ``high``. The parameters define the Pauli-Y rotation angles
    applied in each layer.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits
        low (float): minimum value of uniform distribution
        high (float): maximum value of uniform distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    n_unitaries_per_layer = n_wires - 1

    if n_unitaries_per_layer in [0, -1]:
        params = np.array([])
    else:
        params = np.random.uniform(low=low, high=high, size=(n_layers, n_unitaries_per_layer, 2))

    return params


def simplified_two_design_weights_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for the ``weights`` argument of :func:`~.SimplifiedTwoDesign`,
    drawn from a uniform distribution.

    The shape of the parameter array is ``(n_layers, n_wires - 1, 2)``
    and each parameter is drawn
    from a normal distribution with mean ``mean`` and standard deviation ``std``.
    The parameters define the Pauli-Y rotation angles
    applied in each layer.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits
        mean (float): mean of parameters
        std (float): standard deviation of parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    n_unitaries_per_layer = n_wires - 1

    if n_unitaries_per_layer in [0, -1]:
        params = np.array([])
    else:
        params = np.random.normal(loc=mean, scale=std, size=(n_layers, n_unitaries_per_layer, 2))

    return params


def basic_entangler_layers_normal(n_layers, n_wires, mean=0, std=0.1, seed=None):
    r"""Creates a parameter array for :func:`~.BasicEntanglerLayers`, drawn from a normal
    distribution.

    The shape of the parameter array is ``(n_layers, n_wires)`` and each parameter is drawn
    from a normal distribution with mean ``mean`` and standard deviation ``std``.
    The parameters define the rotation angles applied in each layer.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits
        mean (float): mean of parameters
        std (float): standard deviation of parameters
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    params = np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires))

    return params


def basic_entangler_layers_uniform(n_layers, n_wires, low=0, high=2 * pi, seed=None):
    r"""Creates a parameter array for :func:`~.BasicEntanglerLayers`, drawn from a uniform
    distribution.

    The shape of the parameter array is ``(n_layers, n_wires)`` and each parameter is drawn uniformly at random
    from between ``low`` and ``high``. The parameters define the rotation angles
    applied in each layer.

    Args:
        n_layers (int): number of layers
        n_wires (int): number of qubits
        low (float): minimum value of uniform distribution
        high (float): maximum value of uniform distribution
        seed (int): seed used in sampling the parameters, makes function call deterministic

    Returns:
        array: parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    params = np.random.uniform(low=low, high=high, size=(n_layers, n_wires))

    return params
