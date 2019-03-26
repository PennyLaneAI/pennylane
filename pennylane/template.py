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
QML Templates
=============

**Module name:** :mod:`pennylane.template`

.. currentmodule:: pennylane.template

This module provides a growing library of templates of common quantum
machine learning circuit architectures that can be used to easily build,
evaluate, and train more complex quantum machine learning models.


.. note::

    The templates below are constructed out of **structured combinations**
    of the :mod:`quantum operations <pennylane.ops>` provided by PennyLane.

    As a result, you should follow all the rules of quantum operations
    when you use templates. For example **template functions can only be
    used within a valid** :mod:`pennylane.qnode`.


Examples
--------


For example, you can construct a circuit-centric quantum
classifier with the architecture from :cite:`schuld2018circuit` on an arbitrary
number of wires and with an arbitrary number of blocks by using the
template :class:`StronglyEntanglingCircuit` in the following way:

.. code-block:: python

    import pennylane as qml
    from pennylane.template import StronglyEntanglingCircuit
    from pennylane import numpy as np

    num_wires = 4
    num_blocks = 3
    dev = qml.device('default.qubit', wires=num_wires)

    @qml.qnode(dev)
    def circuit(weights, x=None):
        qml.BasisState(x, wires=range(num_wires))
        StronglyEntanglingCircuit(weights, periodic=True, wires=range(num_wires))
        return qml.expval.PauliZ(0)

    weights = np.random.randn(num_blocks, num_wires, 3)
    print(circuit(weights, x=np.array(np.random.randint(0, 1, num_wires))))


The handy :func:`Interferometer` function can be used to construct arbitrary
interferometers in terms of elementary :class:`~.Beamsplitter` operations,
by providing lists of transmittivity and phase angles. PennyLane can
then be used to easily differentiate and optimize these
parameters:

.. code-block:: python

    import pennylane as qml
    from pennylane.template import Interferometer
    from pennylane import numpy as np

    num_wires = 4
    dev = qml.device('default.gaussian', wires=num_wires)
    num_params = int(num_wires*(num_wires-1)/2)

    # initial parameters
    r = np.random.rand(num_wires, 2)
    theta = np.random.uniform(0, 2*np.pi, num_params)
    phi = np.random.uniform(0, 2*np.pi, num_params)

    @qml.qnode(dev)
    def circuit(theta, phi):
        for w in range(num_wires):
            qml.Squeezing(r[w][0], r[w][1], wires=w)

        Interferometer(theta=theta, phi=phi, wires=range(num_wires))
        return [qml.expval.MeanPhoton(wires=w) for w in range(num_wires)]

    print(qml.jacobian(circuit, 0)(theta, phi))

The function :func:`CVNeuralNet` implements the continuous-variable neural network architecture
from :cite:`killoran2018continuous`. Provided with a suitable array of weights, such neural
networks can be easily constructed and trained with PennyLane.

Provided templates
------------------

.. autosummary::

    StronglyEntanglingCircuit
    StronglyEntanglingCircuitBlock
    CVNeuralNet
    CVNeuralNetLayer
    Interferometer

Code details
^^^^^^^^^^^^
"""
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from collections.abc import Sequence

from pennylane.ops import CNOT, Rot, Squeezing, Displacement, Kerr, Beamsplitter, Rotation
from pennylane.qnode import QuantumFunctionError
from pennylane.variable import Variable


def StronglyEntanglingCircuit(weights, periodic=True, ranges=None, imprimitive=CNOT, wires=None):
    """pennylane.template.StronglyEntanglingCircuit(weights, periodic=True, ranges=None, imprimitive=qml.CNOT, wires)
    A strongly entangling circuit.

    Constructs the strongly entangling circuit used in the circuit-centric quantum
    classifier :cite:`schuld2018circuit`
    with ``len(weights)`` blocks on the :math:`N` wires with the provided weights.
    Each element of weights must be a an array of size :math:`3N`.

    Args:
        weights (array[float]): shape ``(len(weights), len(wires), 3)`` array of weights
        periodic (bool): whether to use periodic boundary conditions when
            applying imprimitive gates
        ranges (Sequence[int]): ranges of the imprimitive gates in the
            respective blocks
        imprimitive (pennylane.ops.Operation): imprimitive gate to use,
            defaults to :class:`~.CNOT`

    Keyword Args:
        wires (Sequence[int]): wires the strongly entangling circuit should act on
    """
    if ranges is None:
        ranges = [1]*len(weights)

    for block_weights, block_range in zip(weights, ranges):
        StronglyEntanglingCircuitBlock(block_weights, r=block_range, periodic=periodic, imprimitive=imprimitive, wires=wires)


def StronglyEntanglingCircuitBlock(weights, periodic=True, r=1, imprimitive=CNOT, wires=None):
    """pennylane.template.StronglyEntanglingCircuitBlock(weights, periodic=True, r=1, imprimitive=qml.CNOT, wires)
    An individual block of a strongly entangling circuit.

    Args:
        weights (array[float]): shape ``(len(wires), 3)`` array of weights
        periodic (bool): whether to use periodic boundary conditions when
            applying imprimitive gates
        r (Sequence[int]): range of the imprimitive gates of this block
        imprimitive (pennylane.ops.Operation): Imprimitive gate to use,
            defaults to :class:`~.CNOT`

    Keyword Args:
        wires (Sequence[int]): Wires the block should act on
    """
    for i, wire in enumerate(wires):
        Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=wire)

    num_wires = len(wires)
    for i in range(num_wires) if periodic else range(num_wires-1):
        imprimitive(wires=[wires[i], wires[(i+r) % num_wires]])


def CVNeuralNet(weights, wires=None):
    """pennylane.template.CVNeuralNet(weights, wires)
    A CV Quantum Neural Network

    Implements the CV Quantum Neural Network (CVQNN) architecture from
    :cite:`killoran2018continuous` for an arbitrary number of wires
    and layers.

    The weights parameter is a nested list. Its first dimension is equal to the number of layers. Each entry is again a
    list that contains the parameters feeding into :func:`CVNeuralNetLayer`.

    Args:
        weights (array[array]): array of arrays of weights for each
            layer of the CV neural network

    Keyword Args:
        wires (Sequence[int]): wires the CVQNN should act on
    """
    for layer_weights in weights:
        CVNeuralNetLayer(*layer_weights, wires=wires)


def CVNeuralNetLayer(theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k, wires=None):
    """pennylane.template.CVNeuralNetLayer(theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k, wires)
    A single layer of a CV Quantum Neural Network

    Implements a single layer from the the CV Quantum Neural Network (CVQNN)
    architecture of :cite:`killoran2018continuous` over :math:`N` wires.

    .. note::

       The CV neural network architecture includes :class:`~.Kerr` operations.
       Make sure to use a suitable device, such as the :code:`strawberryfields.fock`
       device of the `PennyLane-SF <https://github.com/XanaduAI/pennylane-sf>`_ plugin.

    Args:
        theta_1 (array[float]): length :math:`N(N-1)/2` array of transmittivity angles for first interferometer
        phi_1 (array[float]): length :math:`N(N-1)/2` array of phase angles for first interferometer
        varphi_1 (array[float]): length :math:`N` array of final rotation angles for first interferometer
        r (array[float]): length :math:`N` arrays of squeezing amounts for :class:`~.Squeezing` operations
        phi_r (array[float]): length :math:`N` arrays of squeezing angles for :class:`~.Squeezing` operations
        theta_2 (array[float]): length :math:`N(N-1)/2` array of transmittivity angles for second interferometer
        phi_2 (array[float]): length :math:`N(N-1)/2` array of phase angles for second interferometer
        varphi_2 (array[float]): length :math:`N` array of final rotation angles for second interferometer
        a (array[float]): length :math:`N` arrays of displacement magnitudes for :class:`~.Displacement` operations
        phi_a (array[float]): length :math:`N` arrays of displacement angles for :class:`~.Displacement` operations
        k (array[float]): length :math:`N` arrays of kerr parameters for :class:`~.Kerr` operations

    Keyword Args:
        wires (Sequence[int]): wires the layer should act on
    """
    Interferometer(theta=theta_1, phi=phi_1, varphi=varphi_1, wires=wires)
    for i, wire in enumerate(wires):
        Squeezing(r[i], phi_r[i], wires=wire)

    Interferometer(theta=theta_2, phi=phi_2, varphi=varphi_2, wires=wires)

    for i, wire in enumerate(wires):
        Displacement(a[i], phi_a[i], wires=wire)

    for i, wire in enumerate(wires):
        Kerr(k[i], wires=wire)


def Interferometer(theta, phi, varphi, wires=None, mesh='rectangular', beamsplitter='pennylane'):
    r"""pennylane.template.Interferometer(theta, phi, varphi, wires)
    General linear interferometer.

    For :math:`N` wires, the general interferometer is specified by
    providing :math:`N(N-1)` transmittivity angles :math:`\theta` and the same number of
    phase angles :math:`\phi`, as well as either :math:`N-1` or :math:`N` additional rotation
    parameters :math:`\varphi`.

    For the parametrization of a universal interferometer
    :math:`N-1` such rotation parameters are sufficient. If :math:`N` rotation
    parameters are given, the interferometer is over parametrized, but the resulting
    circuit is more symmetric, which can be advantageous.

    By specifying the keyword argument ``mesh``, the scheme used to implement the interferometer
    may be adjusted:

    * ``mesh='rectangular'`` (default): uses the scheme described in
      :cite:`clements2016optimal`, resulting in a *rectangular* array of
      :math:`N(N-1)/2` beamsplitters arranged in :math:`N` layers and numbered from left
      to right and top to bottom in each layer. The first beamsplitters acts on
      wires :math:`0` and :math:`1`.

      .. figure:: ../_static/clements.png
          :align: center
          :width: 30%
          :target: javascript:void(0);

      :html:`<br>`

    * ``mesh='triangular'``: uses the scheme described in :cite:`reck1994experimental`,
      resulting in a *triangular* array of :math:`N(N-1)/2` beamsplitters arranged in
      :math:`2N-3` layers and numbered from left to right and top to bottom. The
      first and forth beamsplitters act on wires :math:`N-1` and :math:`N`, the second
      on :math:`N-2` and :math:`N-1`, and the third on :math:`N-3` and :math:`N-2`, and
      so on.

      .. figure:: ../_static/reck.png
          :align: center
          :width: 30%
          :target: javascript:void(0);

    In both schemes, the network of :class:`~.Beamsplitter` operations is followed by
    :math:`N` (or :math:`N-1`) local :class:`Rotation` Operations. In the latter case, the
    rotation on the last wire is left out.

    The rectangular decomposition is generally advantageous, as it has a lower
    circuit depth (:math:`N` vs :math:`2N-3`) and optical depth than the triangular
    decomposition, resulting in reduced optical loss.

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
        theta (array): length :math:`N(N-1)/2` array of transmittivity angles :math:`\theta`
        phi (array): length :math:`N(N-1)/2` array of phase angles :math:`\phi`
        varphi (array): length :math:`N` or :math:`N-1` array of rotation angles :math:`\varphi`

    Keyword Args:
        mesh (string): the type of mesh to use
        beamsplitter (str): if ``clements``, the beamsplitter convention from
          Clements et al. 2016 (https://dx.doi.org/10.1364/OPTICA.3.001460) is used
        wires (Sequence[int]): wires the interferometer should act on
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

    N = len(w)

    if N == 1:
        # the interferometer is a single rotation
        Rotation(varphi[0], wires=w[0])
        return

    n = 0 # keep track of free parameters

    if mesh == 'rectangular':
        # Apply the Clements beamsplitter array
        # The array depth is N
        for l in range(N):
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
        for l in range(2*N-3):
            for k in range(abs(l+1-(N-1)), N-1, 2):
                if beamsplitter == 'clements':
                    Rotation(phi[n], wires=[w[k]])
                    Beamsplitter(theta[n], 0, wires=[w[k], w[k+1]])
                else:
                    Beamsplitter(theta[n], phi[n], wires=[w[k], w[k+1]])
                n += 1

    # apply the final local phase shifts to all modes
    for i, p in enumerate(varphi):
        Rotation(p, wires=[w[i]])
