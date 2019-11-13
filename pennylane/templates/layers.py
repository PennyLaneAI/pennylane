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
from pennylane.ops import CNOT, RX, RY, RZ, Rot, Squeezing, Displacement, Kerr
from pennylane.templates.subroutines import Interferometer
import numpy as np


def StronglyEntanglingLayers(weights, wires, ranges=None, imprimitive=CNOT):
    r"""A sequence of layers of type :func:`StronglyEntanglingLayer()`, as specified in `arXiv:1804.00633 <https://arxiv.org/abs/1804.00633>`_.

    The number of layers :math:`L` is determined by the first dimension of ``weights``. The template is applied to
    the qubits specified by the sequence ``wires``.

    Args:
        weights (array[float]): array of weights of shape ``(L, len(wires), 3)``
        wires (Sequence[int]): sequence of qubit indices that the template acts on

    Keyword Args:
        ranges (Sequence[int]): sequence determining the range hyperparameter for each subsequent layer
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`
    """

    n_layers = len(weights)

    if ranges is None:
        ranges = [1] * n_layers

    n_layers = len(weights)
    for l in range(n_layers):
        StronglyEntanglingLayer(weights[l], r=ranges[l], imprimitive=imprimitive, wires=wires)


def StronglyEntanglingLayer(weights, wires, r=1, imprimitive=CNOT):
    r"""A layer applying rotations on each qubit followed by cascades of 2-qubit entangling gates.

    The 2-qubit or imprimitive gates act on each qubit :math:`i` chronologically. The second qubit for
    each gate is determined by :math:`(i+r)\mod n`, where :math:`n` is equal to `len(wires)`
    and :math:`range` a layer hyperparameter called the range.

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

    """

    for i, wire in enumerate(wires):
        Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=wire)

    n_wires = len(wires)
    if n_wires > 1:
        for i in range(n_wires):
            imprimitive(wires=[wires[i], wires[(i + r) % n_wires]])


def RandomLayers(weights, wires, ratio_imprim=0.3, imprimitive=CNOT, rotations=None, seed=42):
    r"""A sequence of layers of type :func:`RandomLayer()`.

    The number of layers :math:`L` and the number :math:`k` of rotations per layer is inferred from the first
    and second dimension of ``weights``. The type of imprimitive (two-qubit) gate and rotations distributed
    randomly in the circuit can be chosen explicitly.

    See :func:`RandomLayer` for details on the randomised behaviour.

    Args:
        weights (array[float]): array of weights of shape ``(L, k)``,
        wires (Sequence[int]): sequence of qubit indices that the template acts on

    Keyword Args:
        ratio_imprim (float): value between 0 and 1 that determines the ratio of imprimitive to rotation
            gates (default 0.3)
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`
        rotations (list[pennylane.ops.Operation]): List of Pauli-X, Pauli-Y and/or Pauli-Z gates. The frequency
            determines how often a particular rotation type is used. Defaults to the use of all three
            rotations with equal frequency.
        seed (int): seed to generate random architecture
    """
    if rotations is None:
        rotations = [RX, RY, RZ]

    n_layers = len(weights)
    for l in range(n_layers):
        RandomLayer(weights[l],
                    wires=wires,
                    ratio_imprim=ratio_imprim,
                    imprimitive=imprimitive,
                    rotations=rotations,
                    seed=seed)


def RandomLayer(weights, wires, ratio_imprim=0.3, imprimitive=CNOT, rotations=None, seed=42):
    r"""A layer of randomly chosen single qubit rotations and 2-qubit entangling gates, acting
    on randomly chosen qubits.

    The number :math:`k` of single qubit rotations is inferred from the first dimension of ``weights``.

    This is an example of two 4-qubit random layers with four Pauli-y/Pauli-z rotations :math:`R_y, R_z`,
    controlled-Z gates as imprimitives, as well as ``ratio_imprim=0.3``:

    .. figure:: ../../_static/layer_rnd.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    .. note::
        Using the default seed (or any other fixed integer seed) generates one and the same circuit in every
        quantum node. To generate different circuit architectures, either use a different random seed, or use ``seed=None``
        together with the ``cache=False`` option when creating a quantum node.

    .. warning::
        If you use a random number generator anywhere inside the quantum function without the ``cache=False`` option,
        a new random circuit architecture will be created every time the quantum node is evaluated.

    Args:
        weights (array[float]): array of weights of shape ``(k,)``
        wires (Sequence[int]): sequence of qubit indices that the template acts on

    Keyword Args:
        ratio_imprim (float): value between 0 and 1 that determines the ratio of imprimitive to rotation gates
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`
        rotations (list[pennylane.ops.Operation]): List of Pauli-X, Pauli-Y and/or Pauli-Z gates. The frequency
            determines how often a particular rotation type is used. Defaults to the use of all three
            rotations with equal frequency.
        seed (int): seed to generate random architecture

    Raises:
        ValueError: if less than 2 wires were specified
    """

    if len(wires) < 2:
        raise ValueError("RandomLayer requires at least two wires or subsystems to apply "
                         "the imprimitive gates.")
    if seed is not None:
        np.random.seed(seed)

    if rotations is None:
        rotations = [RX, RY, RZ]

    i = 0
    while i < len(weights):
        if np.random.random() > ratio_imprim:
            gate = np.random.choice(rotations)
            wire = np.random.choice(wires)
            gate(weights[i], wires=wire)
            i += 1
        else:
            on_wires = np.random.permutation(wires)[:2]
            on_wires = list(on_wires)
            imprimitive(wires=on_wires)


def CVNeuralNetLayers(theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k, wires):
    r"""A sequence of layers of type :func:`CVNeuralNetLayer()`, as specified in `arXiv:1806.06871 <https://arxiv.org/abs/1806.06871>`_.

    The number of layers :math:`L` is inferred from the first dimension of the eleven weight parameters. The layers
    act on the :math:`M` modes given in ``wires``, and include interferometers of :math:`K=M(M-1)/2` beamsplitters.

    .. note::

       The CV neural network architecture includes :class:`~pennylane.ops.Kerr` operations.
       Make sure to use a suitable device, such as the :code:`strawberryfields.fock`
       device of the `PennyLane-SF <https://github.com/XanaduAI/pennylane-sf>`_ plugin.

    Args:
        theta_1 (array[float]): length :math:`(L, K)` array of transmittivity angles for first interferometer
        phi_1 (array[float]): length :math:`(L, K)` array of phase angles for first interferometer
        varphi_1 (array[float]): length :math:`(L, M)` array of rotation angles to apply after first interferometer
        r (array[float]): length :math:`(L, M)` array of squeezing amounts for :class:`~pennylane.ops.Squeezing` operations
        phi_r (array[float]): length :math:`(L, M)` array of squeezing angles for :class:`~pennylane.ops.Squeezing` operations
        theta_2 (array[float]): length :math:`(L, K)` array of transmittivity angles for second interferometer
        phi_2 (array[float]): length :math:`(L, K)` array of phase angles for second interferometer
        varphi_2 (array[float]): length :math:`(L, M)` array of rotation angles to apply after second interferometer
        a (array[float]): length :math:`(L, M)` array of displacement magnitudes for :class:`~pennylane.ops.Displacement` operations
        phi_a (array[float]): length :math:`(L, M)` array of displacement angles for :class:`~pennylane.ops.Displacement` operations
        k (array[float]): length :math:`(L, M)` array of kerr parameters for :class:`~pennylane.ops.Kerr` operations
        wires (Sequence[int]): sequence of mode indices that the template acts on
    """

    inferred_layers = [len(theta_1), len(phi_1), len(varphi_1), len(r), len(phi_r), len(theta_2), len(phi_2),
                       len(varphi_2), len(a), len(phi_a), len(k)]
    if inferred_layers.count(inferred_layers[0]) != len(inferred_layers):
        raise ValueError("All parameter arrays need to have the same first dimension, from which the number "
                         "of layers is inferred; got first dimensions {}.".format(inferred_layers))

    n_layers = len(theta_1)
    for l in range(n_layers):
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
        r (array[float]): length :math:`(M, )` array of squeezing amounts for :class:`~pennylane.ops.Squeezing` operations
        phi_r (array[float]): length :math:`(M, )` array of squeezing angles for :class:`~pennylane.ops.Squeezing` operations
        theta_2 (array[float]): length :math:`(K, )` array of transmittivity angles for second interferometer
        phi_2 (array[float]): length :math:`(K, )` array of phase angles for second interferometer
        varphi_2 (array[float]): length :math:`(M, )` array of rotation angles to apply after second interferometer
        a (array[float]): length :math:`(M, )` array of displacement magnitudes for :class:`~pennylane.ops.Displacement` operations
        phi_a (array[float]): length :math:`(M, )` array of displacement angles for :class:`~pennylane.ops.Displacement` operations
        k (array[float]): length :math:`(M, )` array of kerr parameters for :class:`~pennylane.ops.Kerr` operations
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
