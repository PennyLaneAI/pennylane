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
Layers are trainable templates that are typically repeated, using different adjustable parameters in each repetition.
They implement a transformation from a quantum state to another quantum state.
"""
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane import numpy as np
from pennylane.ops import CNOT, RX, RY, RZ, Rot, Squeezing, Displacement, Kerr
from pennylane.templates.subroutines import Interferometer
from pennylane.templates.utils import (_check_shape, _check_no_variable, _check_wires,
                                       _check_type)


def StronglyEntanglingLayers(weights, wires, repeat=1, ranges=None, imprimitive=CNOT):
    r"""A sequence of layers of type :func:`StronglyEntanglingLayer()`, as specified in
    `arXiv:1804.00633 <https://arxiv.org/abs/1804.00633>`_.

    The first dimension of ``weights`` has to be equal to ``repeat``. The template is applied to
    the qubits specified by the sequence ``wires``.

    Args:
        weights (array[float]): array of weights of shape ``(repeat, len(wires), 3)``
        wires (Sequence[int] or int): int or sequence of qubit indices that the template acts on

    Keyword Args:
        repeat (int): number of layers applied
        ranges (Sequence[int]): sequence determining the range hyperparameter for each subsequent layer
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`

    Raises:
        QuantumFunctionError if arguments do not have the correct format.
    """
    if ranges is None:
        ranges = [1] * repeat

    #############
    # Input checks
    _check_no_variable([repeat, ranges, imprimitive], ['repeat', 'ranges', 'imprimitive'])
    wires, n_wires = _check_wires(wires)
    _check_shape(weights, (repeat, n_wires, 3))
    _check_type(repeat, int)
    _check_type(ranges, list)
    _check_type(ranges[0], int)
    ###############

    for l in range(repeat):
        StronglyEntanglingLayer(weights[l], r=ranges[l], imprimitive=imprimitive, wires=wires)


def StronglyEntanglingLayer(weights, wires, r=None, imprimitive=None):
    r"""A layer applying rotations on each qubit followed by cascades of 2-qubit entangling gates.

    The 2-qubit or imprimitive gates act on each qubit :math:`i` chronologically. The second qubit
    is determined by :math:`(i+r)\mod n`, where :math:`n` is equal to the number of wires
    and :math:`r` is a layer hyperparameter called the *range*.

    This is an example of two 4-qubit strongly entangling layers (ranges :math:`r=1` and :math:`r=2`, respectively) with
    rotations :math:`R` and CNOTs as imprimitives:

    .. figure:: ../../_static/layer_sec.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    Args:
        weights (array[float]): array of weights of shape ``(len(wires), 3)``
        wires (Sequence[int]): sequence of qubit indices that the template acts on

    Keyword Args:
        r (int): range of the imprimitive gates of this layer, defaults to 1
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`

    Raises:
        ValueError: if less than 2 wires were specified
    """

    for i, wire in enumerate(wires):
        Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=wire)

    num_wires = len(wires)
    if num_wires > 1:
        for i in range(num_wires):
            imprimitive(wires=[wires[i], wires[(i + r) % num_wires]])


def RandomLayers(weights, wires, repeat=1, ratio_imprim=0.3, imprimitive=CNOT, n_rots=None, rotations=None, seed=42):
    r"""A sequence of layers of type :func:`RandomLayer()`.

    The imprimitive or two-qubit gate and the rotations are distributed randomly in the circuit,
     and their type can be chosen explicitly.

    See :func:`RandomLayer` for details on the randomised behaviour.

    Args:
        weights (array[float]): array of weights of shape ``(L, k)``,
        wires (Sequence[int]): sequence of qubit indices that the template acts on
        repeat (int): number of layers applied

    Keyword Args:
        repeat (int): number of layers applied
        ratio_imprim (float): value between 0 and 1 that determines the ratio of imprimitive to rotation
            gates (default 0.3)
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`
        n_rots (int): number of rotations per layer
        rotations (list[pennylane.ops.Operation]): List of Pauli-X, Pauli-Y and/or Pauli-Z gates. The frequency
            determines how often a particular rotation type is used. Defaults to the use of all three
            rotations with equal frequency.
        seed (int): seed to generate random architecture

    Raises:
        QuantumFunctionError if arguments do not have the correct format.
    """
    if rotations is None:
        rotations = [RX, RY, RZ]

    #############
    # Input checks
    hyperparams = [repeat, ratio_imprim, imprimitive, n_rots, rotations, seed]
    hyperparam_names = ['repeat', 'ratio_imprim', 'imprimitive', 'n_rots', 'rotations', 'seed']
    _check_no_variable(hyperparams, hyperparam_names)
    wires, n_wires = _check_wires(wires)
    if n_rots is None:
        n_rots = len(wires)
    _check_shape(inpt=weights, target_shp=(repeat, n_rots))
    _check_type(repeat, int)
    _check_type(ratio_imprim, float)
    _check_type(n_rots, int)
    _check_type(rotations, list)
    _check_type(seed, int)
    ###############

    n_layers = len(weights)
    for l in range(n_layers):
        RandomLayer(weights[l],
                    wires=wires,
                    ratio_imprim=ratio_imprim,
                    imprimitive=imprimitive,
                    n_rots=n_rots,
                    rotations=rotations,
                    seed=seed)


def RandomLayer(weights, wires, ratio_imprim=0.3, imprimitive=None, n_rots=None, rotations=None, seed=None):
    r"""A layer of randomly chosen single qubit rotations and 2-qubit entangling gates, acting
    on randomly chosen qubits.

    The number :math:`k` of single qubit rotations is inferred from the first dimension of ``weights``.

    This is an example of two 4-qubit random layers with four Pauli-y/Pauli-z rotations :math:`R_y, R_z`,
    controlled-Z gates as imprimitives, as well as ``ratio_imprim=0.3``:

    .. figure:: ../../_static/layer_rnd.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    Args:
        weights (array[float]): array of weights of shape ``(k,)``
        wires (Sequence[int]): sequence of qubit indices that the template acts on

    Keyword Args:
        ratio_imprim (float): value between 0 and 1 that determines the ratio of imprimitive to rotation gates
        imprimitive (pennylane.ops.Operation): two-qubit gate to use
        n_rots (int): number of rotations per layer
        rotations (list[pennylane.ops.Operation]): list of Pauli-X, Pauli-Y and/or Pauli-Z gates
        seed (int): seed to generate random architecture

    Raises:
        QuantumFunctionError if arguments do not have the correct format.
    """
    if rotations is None:
        rotations = [RX, RY, RZ]
    if n_rots is None:
        n_rots = len(wires)

    if seed is not None:
        np.random.seed(seed)

    i = 0
    while i < n_rots:
        if np.random.random() > ratio_imprim:
            gate = np.random.choice(rotations)
            wire = np.random.choice(wires)
            gate(weights[i], wires=wire)
            i += 1
        else:
            on_wires = np.random.permutation(wires)[:2]
            on_wires = list(on_wires)
            imprimitive(wires=on_wires)


def CVNeuralNetLayers(theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k, wires, repeat=1):
    r"""A sequence of layers of type :func:`CVNeuralNetLayer()`,
    as specified in `arXiv:1806.06871 <https://arxiv.org/abs/1806.06871>`_.

    The layers act on the :math:`M` modes given in ``wires``,
    and include interferometers of :math:`K=M(M-1)/2` beamsplitters.

    .. note::

       The CV neural network architecture includes :class:`~pennylane.ops.Kerr` operations.
       Make sure to use a suitable device, such as the :code:`strawberryfields.fock`
       device of the `PennyLane-SF <https://github.com/XanaduAI/pennylane-sf>`_ plugin.

    Args:
        theta_1 (array[float]): length :math:`(L, K)` array of transmittivity angles for first interferometer
        phi_1 (array[float]): length :math:`(L, K)` array of phase angles for first interferometer
        varphi_1 (array[float]): length :math:`(L, M)` array of rotation angles to apply after first interferometer
        r (array[float]): length :math:`(L, M)` array of squeezing amounts for :class:`~pennylane.ops.Squeezing`
            operations
        phi_r (array[float]): length :math:`(L, M)` array of squeezing angles for :class:`~pennylane.ops.Squeezing`
            operations
        theta_2 (array[float]): length :math:`(L, K)` array of transmittivity angles for second interferometer
        phi_2 (array[float]): length :math:`(L, K)` array of phase angles for second interferometer
        varphi_2 (array[float]): length :math:`(L, M)` array of rotation angles to apply after second interferometer
        a (array[float]): length :math:`(L, M)` array of displacement magnitudes for
            :class:`~pennylane.ops.Displacement` operations
        phi_a (array[float]): length :math:`(L, M)` array of displacement angles for
            :class:`~pennylane.ops.Displacement` operations
        k (array[float]): length :math:`(L, M)` array of kerr parameters for :class:`~pennylane.ops.Kerr` operations
        wires (Sequence[int]): sequence of mode indices that the template acts on

    Raises:
        QuantumFunctionError if arguments do not have the correct format.
    """

    #############
    # Input checks
    _check_no_variable([repeat], ['repeat'])
    wires, n_wires = _check_wires(wires)
    n_if = n_wires*(n_wires-1)//2
    weights = [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]
    shps = [(repeat, n_if), (repeat, n_if), (repeat, n_wires), (repeat, n_wires), (repeat, n_wires),
           (repeat, n_if), (repeat, n_if), (repeat, n_wires), (repeat, n_wires), (repeat, n_wires),
           (repeat, n_wires)]
    _check_shape(weights, shps)
    _check_type(repeat, int)
    ###############

    for l in range(repeat):
        CVNeuralNetLayer(theta_1[l], phi_1[l], varphi_1[l], r[l], phi_r[l],
                         theta_2[l], phi_2[l], varphi_2[l], a[l], phi_a[l], k[l], wires=wires)


def CVNeuralNetLayer(theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k, wires):
    r"""A layer of interferometers, displacement and squeezing gates mimicking a neural network,
    as well as a Kerr gate nonlinearity.

    The layer acts on the :math:`M` wires modes specified in ``wires``, and includes interferometers
    of :math:`K=M(M-1)/2` beamsplitters.

    This example shows a 4-mode CVNeuralNet layer with squeezing gates :math:`S`, displacement gates :math:`D` and
    Kerr gates :math:`K`. The two big blocks are interferometers of type
    :mod:`pennylane.templates.layers.Interferometer`:

    .. figure:: ../../_static/layer_cvqnn.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    .. note::

       The CV neural network architecture includes :class:`~pennylane.ops.Kerr` operations.
       Make sure to use a suitable device, such as the :code:`strawberryfields.fock`
       device of the `PennyLane-SF <https://github.com/XanaduAI/pennylane-sf>`_ plugin.

    Args:
        theta_1 (array[float]): length :math:`(K, )` array of transmittivity angles for first interferometer
        phi_1 (array[float]): length :math:`(K, )` array of phase angles for first interferometer
        varphi_1 (array[float]): length :math:`(M, )` array of rotation angles to apply after first interferometer
        r (array[float]): length :math:`(M, )` array of squeezing amounts for
            :class:`~pennylane.ops.Squeezing` operations
        phi_r (array[float]): length :math:`(M, )` array of squeezing angles for
            :class:`~pennylane.ops.Squeezing` operations
        theta_2 (array[float]): length :math:`(K, )` array of transmittivity angles for second interferometer
        phi_2 (array[float]): length :math:`(K, )` array of phase angles for second interferometer
        varphi_2 (array[float]): length :math:`(M, )` array of rotation angles to apply after second interferometer
        a (array[float]): length :math:`(M, )` array of displacement magnitudes for
            :class:`~pennylane.ops.Displacement` operations
        phi_a (array[float]): length :math:`(M, )` array of displacement angles for
            :class:`~pennylane.ops.Displacement` operations
        k (array[float]): length :math:`(M, )` array of kerr parameters for
            :class:`~pennylane.ops.Kerr` operations
        wires (Sequence[int]): sequence of mode indices that the template acts on
    """
    Interferometer(theta=theta_1, phi=phi_1, varphi=varphi_1, wires=wires)
    for i, wire in enumerate(wires):
        Squeezing(r[i], phi_r[i], wires=wire)
    Interferometer(theta=theta_2, phi=phi_2, varphi=varphi_2, wires=wires)
    for i, wire in enumerate(wires):
        Displacement(a[i], phi_a[i], wires=wire)
    for i, wire in enumerate(wires):
        Kerr(k[i], wires=wire)


layers = {"StronglyEntanglingLayers", "RandomLayers", "CVNeuralNetLayers"}

__all__ = list(layers)
