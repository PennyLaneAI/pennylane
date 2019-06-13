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
Layers
======

**Module name:** :mod:`pennylane.templates.layers`

.. currentmodule:: pennylane.templates.layers

This module contains templates for trainable `layers`. In contrast to other templates such as embeddings, layers
do typically only take trainable parameters, and get repeated in the circuit -- just like the layers of a
neural network. This makes the layer `learnable` within the limits of the architecture.

Most templates in this module have a ``Layer`` version that implements a single layer, as well as a ``Layers``
version which calls the single layer multiple times, possibly using different hyperparameters for the
sequence in each call.


Qubit architectures
-------------------

Strongly entangling circuit
***************************

.. autosummary::

    StronglyEntanglingLayers
    StronglyEntanglingLayer

Random circuit
**************

.. autosummary::

    RandomLayers
    RandomLayer

Continuous-variable architectures
---------------------------------

Continuous-variable quantum neural network
******************************************

.. autosummary::

    CVNeuralNetLayers
    CVNeuralNetLayer

Interferometer
**************

.. autosummary::

    Interferometer

Code details
^^^^^^^^^^^^
"""
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from collections.abc import Sequence

from pennylane.ops import CNOT, RX, RY, RZ, Rot, Squeezing, Displacement, Kerr, Beamsplitter, Rotation
from pennylane.qnode import QuantumFunctionError
from pennylane.variable import Variable
import numpy as np


def StronglyEntanglingLayers(weights, wires, ranges=None, imprimitive=CNOT):
    r"""A sequence of layers of type :func:`StronglyEntanglingLayer()`, as specified in :cite:`schuld2018circuit`.

    The number of layers :math:`L` is determined by the first dimension of ``weights``. The template is applied to
    the qubits specified by the sequence ``wires``.

    Args:
        weights (array[float]): array of weights of shape ``(L, len(wires), 3)``
        wires (Sequence[int]): sequence of qubit indices that the template acts on

    Keyword Args:
        ranges (Sequence[int]): sequence determining the range hyperparameter for each subsequent layer
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~.CNOT`
    """

    if ranges is None:
        ranges = [1]*len(weights)

    for block_weights, block_range in zip(weights, ranges):
        StronglyEntanglingLayer(block_weights, r=block_range, imprimitive=imprimitive, wires=wires)


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
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~.CNOT`
    """
    if len(wires) < 2:
        raise ValueError("StronglyEntanglingLayer requires at least two wires or subsystems to apply "
                         "the imprimitive gates.")

    for i, wire in enumerate(wires):
        Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=wire)

    num_wires = len(wires)
    for i in range(num_wires):
        imprimitive(wires=[wires[i], wires[(i + r) % num_wires]])


def RandomLayers(weights, wires, ratio_imprim=0.3, imprimitive=CNOT, rotations=None, seed=None):
    r"""A sequence of layers of type :func:`RandomLayer()`.

    The number of layers :math:`L` and the number :math:`k` of rotations per layer is inferred from the first
    and second dimension of ``weights``. The type of imprimitive (two-qubit) gate and rotations distributed
    randomly in the circuit can be chosen explicitly.

    Args:
        weights (array[float]): array of weights of shape ``(L, k)``,
        wires (Sequence[int]): sequence of qubit indices that the template acts on

    Keyword Args:
        ratio_imprim (float): value between 0 and 1 that determines the ratio of imprimitive to rotation
            gates (default 0.3)
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~.CNOT`
        rotations (list[pennylane.ops.Operation]): List of Pauli-X, Pauli-Y and/or Pauli-Z gates. The frequency
            determines how often a particular rotation type is used. Defaults to the use of all three
            rotations with equal frequency.
        seed (int): seed to generate random architecture
    """
    if rotations is None:
        rotations = [RX, RY, RZ]

    for layer_weights in weights:
        RandomLayer(layer_weights, wires=wires, ratio_imprim=ratio_imprim, imprimitive=imprimitive, rotations=rotations,
                    seed=seed)


def RandomLayer(weights, wires, ratio_imprim=0.3, imprimitive=CNOT, rotations=None, seed=None):
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
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~.CNOT`
        rotations (list[pennylane.ops.Operation]): List of Pauli-X, Pauli-Y and/or Pauli-Z gates. The frequency
            determines how often a particular rotation type is used. Defaults to the use of all three
            rotations with equal frequency.
        seed (int): seed to generate random architecture
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
    r"""A sequence of layers of type :func:`CVNeuralNetLayer()`, as specified in :cite:`killoran2018continuous`.

    The number of layers :math:`L` is inferred from the first dimension of the eleven weight parameters. The layers
    act on the :math:`M` modes given in ``wires``, and include interferometers of :math:`K=M(M-1)/2` beamsplitters.

    .. note::

       The CV neural network architecture includes :class:`~.Kerr` operations.
       Make sure to use a suitable device, such as the :code:`strawberryfields.fock`
       device of the `PennyLane-SF <https://github.com/XanaduAI/pennylane-sf>`_ plugin.

    Args:
        theta_1 (array[float]): length :math:`(L, K)` array of transmittivity angles for first interferometer
        phi_1 (array[float]): length :math:`(L, K)` array of phase angles for first interferometer
        varphi_1 (array[float]): length :math:`(L, M)` array of rotation angles to apply after first interferometer
        r (array[float]): length :math:`(L, M)` array of squeezing amounts for :class:`~.Squeezing` operations
        phi_r (array[float]): length :math:`(L, M)` array of squeezing angles for :class:`~.Squeezing` operations
        theta_2 (array[float]): length :math:`(L, K)` array of transmittivity angles for second interferometer
        phi_2 (array[float]): length :math:`(L, K)` array of phase angles for second interferometer
        varphi_2 (array[float]): length :math:`(L, M)` array of rotation angles to apply after second interferometer
        a (array[float]): length :math:`(L, M)` array of displacement magnitudes for :class:`~.Displacement` operations
        phi_a (array[float]): length :math:`(L, M)` array of displacement angles for :class:`~.Displacement` operations
        k (array[float]): length :math:`(L, M)` array of kerr parameters for :class:`~.Kerr` operations
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

       The CV neural network architecture includes :class:`~.Kerr` operations.
       Make sure to use a suitable device, such as the :code:`strawberryfields.fock`
       device of the `PennyLane-SF <https://github.com/XanaduAI/pennylane-sf>`_ plugin.

    Args:
        theta_1 (array[float]): length :math:`(K, )` array of transmittivity angles for first interferometer
        phi_1 (array[float]): length :math:`(K, )` array of phase angles for first interferometer
        varphi_1 (array[float]): length :math:`(M, )` array of rotation angles to apply after first interferometer
        r (array[float]): length :math:`(M, )` array of squeezing amounts for :class:`~.Squeezing` operations
        phi_r (array[float]): length :math:`(M, )` array of squeezing angles for :class:`~.Squeezing` operations
        theta_2 (array[float]): length :math:`(K, )` array of transmittivity angles for second interferometer
        phi_2 (array[float]): length :math:`(K, )` array of phase angles for second interferometer
        varphi_2 (array[float]): length :math:`(M, )` array of rotation angles to apply after second interferometer
        a (array[float]): length :math:`(M, )` array of displacement magnitudes for :class:`~.Displacement` operations
        phi_a (array[float]): length :math:`(M, )` array of displacement angles for :class:`~.Displacement` operations
        k (array[float]): length :math:`(M, )` array of kerr parameters for :class:`~.Kerr` operations
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


def Interferometer(theta, phi, varphi, wires, mesh='rectangular', beamsplitter='pennylane'):
    r"""General linear interferometer, an array of beamsplitters and phase shifters.

    For :math:`M` wires, the general interferometer is specified by
    providing :math:`M(M-1)/2` transmittivity angles :math:`\theta` and the same number of
    phase angles :math:`\phi`, as well as either :math:`M-1` or :math:`M` additional rotation
    parameters :math:`\varphi`.

    For the parametrization of a universal interferometer
    :math:`M-1` such rotation parameters are sufficient. If :math:`M` rotation
    parameters are given, the interferometer is over-parametrized, but the resulting
    circuit is more symmetric, which can be advantageous.

    By specifying the keyword argument ``mesh``, the scheme used to implement the interferometer
    may be adjusted:

    * ``mesh='rectangular'`` (default): uses the scheme described in
      :cite:`clements2016optimal`, resulting in a *rectangular* array of
      :math:`M(M-1)/2` beamsplitters arranged in :math:`M` slices and ordered from left
      to right and top to bottom in each slice. The first beamsplitter acts on
      wires :math:`0` and :math:`1`:

      .. figure:: ../../_static/clements.png
          :align: center
          :width: 30%
          :target: javascript:void(0);


    * ``mesh='triangular'``: uses the scheme described in :cite:`reck1994experimental`,
      resulting in a *triangular* array of :math:`M(M-1)/2` beamsplitters arranged in
      :math:`2M-3` slices and ordered from left to right and top to bottom. The
      first and fourth beamsplitters act on wires :math:`M-1` and :math:`M`, the second
      on :math:`M-2` and :math:`M-1`, and the third on :math:`M-3` and :math:`M-2`, and
      so on.

      .. figure:: ../../_static/reck.png
          :align: center
          :width: 30%
          :target: javascript:void(0);

    In both schemes, the network of :class:`~.Beamsplitter` operations is followed by
    :math:`M` (or :math:`M-1`) local :class:`Rotation` Operations. In the latter case, the
    rotation on the last wire is left out.

    The rectangular decomposition is generally advantageous, as it has a lower
    circuit depth (:math:`M` vs :math:`2M-3`) and optical depth than the triangular
    decomposition, resulting in reduced optical loss.

    This is an example of a 4-mode interferometer with beamsplitters :math:`B` and rotations :math:`R`,
    using ``mesh='rectangular'``:

    .. figure:: ../../_static/layer_interferometer.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    .. note::

        The decomposition as formulated in :cite:`clements2016optimal` uses a different
        convention for a beamsplitter :math:`T(\theta, \phi)` than PennyLane, namely:

        .. math:: T(\theta, \phi) = BS(\theta, 0) R(\phi)

        For the universality of the decomposition, the used convention is irrelevant, but
        for a given set of angles the resulting interferometers will be different.

        If an interferometer consistent with the convention from :cite:`clements2016optimal`
        is needed, the optional keyword argument ``beamsplitter='clements'`` can be specified. This
        will result in each :class:`~.Beamsplitter` being preceded by a :class:`Rotation` and
        thus increase the number of elementary operations in the circuit.

    Args:
        theta (array): length :math:`M(M-1)/2` array of transmittivity angles :math:`\theta`
        phi (array): length :math:`M(M-1)/2` array of phase angles :math:`\phi`
        varphi (array): length :math:`M` or :math:`M-1` array of rotation angles :math:`\varphi`
        wires (Sequence[int]): wires the interferometer should act on

    Keyword Args:
        mesh (string): the type of mesh to use
        beamsplitter (str): if ``clements``, the beamsplitter convention from
          Clements et al. 2016 (https://dx.doi.org/10.1364/OPTICA.3.001460) is used; if ``pennylane``, the
          beamsplitter is implemented via PennyLane's ``Beamsplitter`` operation.
    """
    if isinstance(beamsplitter, Variable):
        raise QuantumFunctionError("The beamsplitter parameter influences the "
                                   "circuit architecture and can not be passed as a QNode parameter.")

    if isinstance(mesh, Variable):
        raise QuantumFunctionError("The mesh parameter influences the circuit architecture "
                                   "and can not be passed as a QNode parameter.")

    if not isinstance(wires, Sequence):
        w = [wires]
    else:
        w = wires

    M = len(w)

    if M == 1:
        # the interferometer is a single rotation
        Rotation(varphi[0], wires=w[0])
        return

    n = 0 # keep track of free parameters

    if mesh == 'rectangular':
        # Apply the Clements beamsplitter array
        # The array depth is N
        for l in range(M):
            for k, (w1, w2) in enumerate(zip(w[:-1], w[1:])):
                #skip even or odd pairs depending on layer
                if (l+k)%2 != 1:
                    if beamsplitter == 'clements':
                        Rotation(phi[n], wires=[w1])
                        Beamsplitter(theta[n], 0, wires=[w1, w2])
                    else:
                        Beamsplitter(theta[n], phi[n], wires=[w1, w2])
                    n += 1

    elif mesh == 'triangular':
        # apply the Reck beamsplitter array
        # The array depth is 2*N-3
        for l in range(2*M-3):
            for k in range(abs(l+1-(M-1)), M-1, 2):
                if beamsplitter == 'clements':
                    Rotation(phi[n], wires=[w[k]])
                    Beamsplitter(theta[n], 0, wires=[w[k], w[k+1]])
                else:
                    Beamsplitter(theta[n], phi[n], wires=[w[k], w[k+1]])
                n += 1

    # apply the final local phase shifts to all modes
    for i, p in enumerate(varphi):
        Rotation(p, wires=[w[i]])
