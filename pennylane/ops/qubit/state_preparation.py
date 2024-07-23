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
# pylint:disable=abstract-method,arguments-differ,protected-access,no-member
import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.operation import AnyWires, Operation, StatePrepBase
from pennylane.templates.state_preparations import MottonenStatePreparation
from pennylane.wires import WireError, Wires

state_prep_ops = {"BasisState", "StatePrep", "QubitDensityMatrix"}


class BasisState(StatePrepBase):
    r"""BasisState(features, wires)
    Prepares a single computational basis state.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None (integer parameters not supported)

    .. note::

        If the ``BasisState`` operation is not supported natively on the
        target device, PennyLane will attempt to decompose the operation
        into :class:`~.PauliX` operations.

    .. note::

        When called in the middle of a circuit, the action of the operation is defined
        as :math:`U|0\rangle = |\psi\rangle`

    Args:
        n (array): prepares the basis state :math:`\ket{n}`, where ``n`` is an
            array of integers from the set :math:`\{0, 1\}`, i.e.,
            if ``n = np.array([0, 1, 0])``, prepares the state :math:`|010\rangle`.
        wires (Sequence[int] or int): the wire(s) the operation acts on
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

    **Example**

    >>> dev = qml.device('default.qubit', wires=2)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.BasisState(np.array([1, 1]), wires=range(2))
    ...     return qml.state()
    >>> print(example_circuit())
    [0.+0.j 0.+0.j 0.+0.j 1.+0.j]
    """

    def __init__(self, features, wires, id=None):

        if isinstance(features, list):
            features = qml.math.stack(features)

        tracing = qml.math.is_abstract(features)

        if qml.math.shape(features) == ():
            if not tracing and features >= 2 ** len(wires):
                raise ValueError(
                    f"Features must be of length {len(wires)}, got features={features} which is >= {2 ** len(wires)}"
                )
            bin = 2 ** math.arange(len(wires))[::-1]
            features = qml.math.where((features & bin) > 0, 1, 0)

        wires = Wires(wires)
        shape = qml.math.shape(features)

        if len(shape) != 1:
            raise ValueError(f"Features must be one-dimensional; got shape {shape}.")

        n_features = shape[0]
        if n_features != len(wires):
            raise ValueError(
                f"Features must be of length {len(wires)}; got length {n_features} (features={features})."
            )

        if not tracing:
            features_list = list(qml.math.toarray(features))
            if not set(features_list).issubset({0, 1}):
                raise ValueError(f"Basis state must only consist of 0s and 1s; got {features_list}")

        super().__init__(features, wires=wires, id=id)

    def _flatten(self):
        features = self.parameters[0]
        features = tuple(features) if isinstance(features, list) else features
        return (features,), (self.wires,)

    @classmethod
    def _unflatten(cls, data, metadata) -> "BasisState":
        return cls(data[0], wires=metadata[0])

    @staticmethod
    def compute_decomposition(features, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.BasisState.decomposition`.

        Args:
            n (array): prepares the basis state :math:`\ket{n}`, where ``n`` is an
                array of integers from the set :math:`\{0, 1\}`
            wires (Iterable, Wires): the wire(s) the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.BasisState.compute_decomposition([1,0], wires=(0,1))
        [BasisStatePreparation([1, 0], wires=[0, 1])]

        """

        if not qml.math.is_abstract(features):
            op_list = []
            for wire, state in zip(wires, features):
                if state == 1:
                    op_list.append(qml.X(wire))
            return op_list

        op_list = []
        for wire, state in zip(wires, features):
            op_list.append(qml.PhaseShift(state * np.pi / 2, wire))
            op_list.append(qml.RX(state * np.pi, wire))
            op_list.append(qml.PhaseShift(state * np.pi / 2, wire))

        return op_list

    def state_vector(self, wire_order=None):
        """Returns a statevector of shape ``(2,) * num_wires``."""
        prep_vals = self.parameters[0]
        if qml.math.shape(prep_vals) == ():
            bin = 2 ** math.arange(len(self.wires))[::-1]
            prep_vals = qml.math.where((prep_vals & bin) > 0, 1, 0)

        prep_vals_int = math.cast(prep_vals, int)

        if wire_order is None:
            indices = prep_vals_int
            num_wires = len(indices)
        else:
            if not Wires(wire_order).contains_wires(self.wires):
                raise WireError("Custom wire_order must contain all BasisState wires")
            num_wires = len(wire_order)
            indices = [0] * num_wires
            for base_wire_label, value in zip(self.wires, prep_vals_int):
                indices[wire_order.index(base_wire_label)] = value

        if qml.math.get_interface(prep_vals_int) == "jax":
            ket = math.array(math.zeros((2,) * num_wires), like="jax")
            ket = ket.at[tuple(indices)].set(1)

        else:
            ket = math.zeros((2,) * num_wires)
            ket[tuple(indices)] = 1

        return math.convert_like(ket, prep_vals)


class StatePrep(StatePrepBase):
    r"""StatePrep(state, wires)
    Prepare subsystems using the given ket vector in the computational basis.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    .. note::

        If the ``StatePrep`` operation is not supported natively on the
        target device, PennyLane will attempt to decompose the operation
        using the method developed by Möttönen et al. (Quantum Info. Comput.,
        2005).

    .. note::

        When called in the middle of a circuit, the action of the operation is defined
        as :math:`U|0\rangle = |\psi\rangle`

    Args:
        state (array[complex]): a state vector of size 2**len(wires)
        wires (Sequence[int] or int): the wire(s) the operation acts on
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

    **Example**

    >>> dev = qml.device('default.qubit', wires=2)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.StatePrep(np.array([1, 0, 0, 0]), wires=range(2))
    ...     return qml.state()
    >>> print(example_circuit())
    [1.+0.j 0.+0.j 0.+0.j 0.+0.j]
    """

    num_wires = AnyWires
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (1,)
    """int: Number of dimensions per trainable parameter of the operator."""

    def __init__(self, state, wires, id=None):
        super().__init__(state, wires=wires, id=id)
        state = self.parameters[0]

        if len(state.shape) == 1:
            state = math.reshape(state, (1, state.shape[0]))
        if state.shape[1] != 2 ** len(self.wires):
            raise ValueError("State vector must have shape (2**wires,) or (batch_size, 2**wires).")

        param = math.cast(state, np.complex128)
        if not math.is_abstract(param):
            norm = math.linalg.norm(param, axis=-1, ord=2)
            if not math.allclose(norm, 1.0, atol=1e-10):
                raise ValueError("Sum of amplitudes-squared does not equal one.")

    @staticmethod
    def compute_decomposition(state, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.StatePrep.decomposition`.

        Args:
            state (array[complex]): a state vector of size 2**len(wires)
            wires (Iterable, Wires): the wire(s) the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.StatePrep.compute_decomposition(np.array([1, 0, 0, 0]), wires=range(2))
        [MottonenStatePreparation(tensor([1, 0, 0, 0], requires_grad=True), wires=[0, 1])]

        """
        return [MottonenStatePreparation(state, wires)]

    def state_vector(self, wire_order=None):
        num_op_wires = len(self.wires)
        op_vector_shape = (-1,) + (2,) * num_op_wires if self.batch_size else (2,) * num_op_wires
        op_vector = math.reshape(self.parameters[0], op_vector_shape)

        if wire_order is None or Wires(wire_order) == self.wires:
            return op_vector

        wire_order = Wires(wire_order)
        if not wire_order.contains_wires(self.wires):
            raise WireError(f"Custom wire_order must contain all {self.name} wires")

        # add zeros for each wire that isn't being set
        extra_wires = Wires(set(wire_order) - set(self.wires))
        for _ in extra_wires:
            op_vector = math.stack([op_vector, math.zeros_like(op_vector)], axis=-1)

        # transpose from operator wire order to provided wire order
        current_wires = self.wires + extra_wires
        transpose_axes = [current_wires.index(w) for w in wire_order]
        if self.batch_size:
            transpose_axes = [0] + [a + 1 for a in transpose_axes]
        return math.transpose(op_vector, transpose_axes)


# pylint: disable=missing-class-docstring
class QubitStateVector(StatePrep):
    pass  # QSV is still available


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
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

    .. details::
        :title: Usage Details

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
    """int: Number of trainable parameters that the operator depends on."""

    grad_method = None
