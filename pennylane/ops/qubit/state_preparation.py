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
"""
This submodule contains the discrete-variable quantum operations concerned
with preparing a certain state on the device.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access

import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.templates.state_preparations import BasisStatePreparation, MottonenStatePreparation


class BasisState(Operation):
    r"""BasisState(n, wires)
    Prepares a single computational basis state.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None (integer parameters not supported)

    .. note::

        If the ``BasisState`` operation is not supported natively on the
        target device, PennyLane will attempt to decompose the operation
        into :class:`~.PauliX` operations.

    Args:
        n (array): prepares the basis state :math:`\ket{n}`, where ``n`` is an
            array of integers from the set :math:`\{0, 1\}`, i.e.,
            if ``n = np.array([0, 1, 0])``, prepares the state :math:`|010\rangle`.
        wires (Sequence[int] or int): the wire(s) the operation acts on

    **Example**

    >>> dev = qml.device('default.qubit', wires=2)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.BasisState(np.array([1, 1]), wires=range(2))
    ...     return qml.state()
    >>> print(example_circuit())
    [0.+0.j 0.+0.j 0.+0.j 1.+0.j]
    """
    num_wires = AnyWires
    num_params = 1
    grad_method = None

    @staticmethod
    def decomposition(n, wires):
        return BasisStatePreparation(n, wires)

    def adjoint(self):
        raise qml.ops.AdjointError("No adjoint exists for BasisState operations.")


class QubitStateVector(Operation):
    r"""QubitStateVector(state, wires)
    Prepare subsystems using the given ket vector in the computational basis.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    .. note::

        If the ``QubitStateVector`` operation is not supported natively on the
        target device, PennyLane will attempt to decompose the operation
        using the method developed by Möttönen et al. (Quantum Info. Comput.,
        2005).

    Args:
        state (array[complex]): a state vector of size 2**len(wires)
        wires (Sequence[int] or int): the wire(s) the operation acts on

    **Example**

    >>> dev = qml.device('default.qubit', wires=2)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.QubitStateVector(np.array([1, 0, 0, 0]), wires=range(2))
    ...     return qml.state()
    >>> print(example_circuit())
    [1.+0.j 0.+0.j 0.+0.j 0.+0.j]
    """
    num_wires = AnyWires
    num_params = 1
    grad_method = None

    @staticmethod
    def decomposition(state, wires):
        return MottonenStatePreparation(state, wires)

    def adjoint(self):
        raise qml.ops.AdjointError("No adjoint exists for QubitStateVector operations.")


class QubitDensityMatrix(Operation):
    r"""QubitDensityMatrix(state, wires)
    Prepare subsystems using the given density matrix.
    If not all the wires are specified, remaining dimension is filled by :math:`\mathrm{tr}_{in}(\rho)`,
    where :math:`\rho` is the full system density matrix before this operation and :math:`\mathrm{tr}_{in}` is a
    partial trace over the subsystem to be replaced by input state.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    .. note::

        Exception raised if the ``QubitDensityMatrix`` operation is not supported natively on the
        target device.

    Args:
        state (array[complex]): a density matrix of size ``(2**len(wires), 2**len(wires))``
        wires (Sequence[int] or int): the wire(s) the operation acts on

    .. UsageDetails::

        Example:

        .. code-block:: python

            import pennylane as qml
            nr_wires = 2
            rho = np.zeros((2 ** nr_wires, 2 ** nr_wires), dtype=np.complex128)
            rho[0, 0] = 1  # initialize the pure state density matrix for the |0><0| state

            dev = qml.device("default.mixed", wires=2)
            @qml.qnode(dev)
            def circuit():
                qml.QubitDensityMatrix(rho, wires=[0, 1])
                return qml.state()

        Running this circuit:

        >>> circuit()
        [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
    """
    num_wires = AnyWires
    num_params = 1
    grad_method = None

    def adjoint(self):
        raise qml.ops.AdjointError("No adjoint exists for QubitDensityMatrix operations.")
