# Copyright 2022 Xanadu Quantum Technologies Inc.

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
.. currentmodule:: pennylane

Measurements
------------

.. autosummary::
    :toctree: api

    ~classical_shadow
    ~shadow_expval

Shadow class for classical post-processing
------------------------------------------

.. autosummary::
    :toctree: api

    ~ClassicalShadow

QNode transforms
----------------

.. autosummary::
    :toctree: api

    ~shadows.shadow_expval
    ~shadows.shadow_state

Classical Shadows formalism
---------------------------

.. note:: As per `arXiv:2103.07510 <https://arxiv.org/abs/2103.07510>`_, when computing multiple expectation values it is advisable to directly estimate the desired observables by simultaneously measuring
    qubit-wise-commuting terms. One way of doing this in PennyLane is via :class:`~pennylane.Hamiltonian` and setting ``grouping_type="qwc"``. For more details on this topic, see the PennyLane demo
    on `estimating expectation values with classical shadows <https://pennylane.ai/qml/demos/tutorial_diffable_shadows.html>`_.

A :class:`ClassicalShadow` is a classical description of a quantum state that is capable of reproducing expectation values of local Pauli observables, see `arXiv:2002.08953 <https://arxiv.org/abs/2002.08953>`_.

The idea is to capture :math:`T` local snapshots (given by the ``shots`` set in the device) of the state by performing measurements in random Pauli bases at each qubit.
The measurement outcomes, denoted ``bits``, as well as the choices of measurement bases, ``recipes``, are recorded in two ``(T, len(wires))`` integer tensors, respectively.

From the :math:`t`-th measurement, we can reconstruct the ``local_snapshots`` (see :class:`ClassicalShadow` methods)

.. math:: \rho^{(t)} = \bigotimes_{i=1}^{n} 3 U^\dagger_i |b_i \rangle \langle b_i | U_i - \mathbb{I},

where :math:`U_i` is the rotation corresponding to the measurement (e.g. :math:`U_i=H` for measurement in :math:`X`) of qubit :math:`i` at snapshot :math:`t` and
:math:`|b_i\rangle = (1 - b_i, b_i)`
the corresponding computational basis state given the output bit :math:`b_i`.

From these local snapshots, one can compute expectation values of local Pauli strings, where locality refers to the number of non-Identity operators.
The accuracy of the procedure is determined by the number of measurements :math:`T` (``shots``).
To target an error :math:`\epsilon`, one needs of order :math:`T = \mathcal{O}\left( \log(M) 4^\ell/\epsilon^2 \right)` measurements to determine :math:`M` different,
:math:`\ell`-local observables.

One can in principle also reconstruct the global state :math:`\sum_t \rho^{(t)}/T`, though it is not advisable nor practical for larger systems due to its exponential scaling.

Basic usage
-----------

The easiest way of computing expectation values with classical shadows in PennyLane is to return :func:`shadow_expval` directly from the qnode.

.. code-block:: python3

    H = qml.Hamiltonian([1., 1.], [qml.Z(0) @ qml.Z(1), qml.X(0) @ qml.Z(1)])

    dev = qml.device("default.qubit", shots=10000)

    # shadow_expval + mid-circuit measurements require to defer measurements
    @qml.defer_measurements
    @qml.qnode(dev)
    def qnode(x):
        qml.Hadamard(0)
        qml.CNOT((0,1))
        qml.RX(x, wires=0)
        qml.measure(1)
        return qml.shadow_expval(H)

    x = np.array(0.5, requires_grad=True)

The big advantage of this way of computing expectation values is that it is differentiable.

>>> qnode(x)
array(0.8406)
>>> qml.grad(qnode)(x)
-0.49680000000000013

There are more options for post-processing classical shadows in :class:`ClassicalShadow`.
"""

from .classical_shadow import ClassicalShadow, median_of_means, pauli_expval

# allow aliasing in the module namespace
from .transforms import shadow_state, shadow_expval
