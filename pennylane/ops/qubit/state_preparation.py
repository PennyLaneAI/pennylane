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
# pylint:disable=too-many-branches,abstract-method,arguments-differ,protected-access,no-member
from typing import Optional

import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.operation import AnyWires, Operation, Operator, StatePrepBase
from pennylane.templates.state_preparations import MottonenStatePreparation
from pennylane.typing import TensorLike
from pennylane.wires import WireError, Wires, WiresLike

state_prep_ops = {"BasisState", "StatePrep", "QubitDensityMatrix"}

# TODO: Remove TOLERANCE as global variable
TOLERANCE = 1e-10


class BasisState(StatePrepBase):
    r"""BasisState(state, wires)
    Prepares a single computational basis state.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    .. note::

        If the ``BasisState`` operation is not supported natively on the
        target device, PennyLane will attempt to decompose the operation
        into :class:`~.PauliX` operations.

    .. note::

        When called in the middle of a circuit, the action of the operation is defined
        as :math:`U|0\rangle = |\psi\rangle`

    Args:
        state (tensor_like): binary input of shape ``(len(wires), )``, e.g., for ``state=np.array([0, 1, 0])`` or ``state=2`` (binary 010), the quantum system will be prepared in state :math:`|010 \rangle`.

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

    def __init__(self, state, wires, id=None):

        if isinstance(state, list):
            state = qml.math.stack(state)

        tracing = qml.math.is_abstract(state)

        if not qml.math.shape(state):
            if not tracing and state >= 2 ** len(wires):
                raise ValueError(
                    f"Integer state must be < {2 ** len(wires)} to have a feasible binary representation, got {state}"
                )
            bin = 2 ** math.arange(len(wires))[::-1]
            state = qml.math.where((state & bin) > 0, 1, 0)

        wires = Wires(wires)
        shape = qml.math.shape(state)

        if len(shape) != 1:
            raise ValueError(f"State must be one-dimensional; got shape {shape}.")

        n_states = shape[0]
        if n_states != len(wires):
            raise ValueError(
                f"State must be of length {len(wires)}; got length {n_states} (state={state})."
            )

        if not tracing:
            state_list = list(qml.math.toarray(state))
            if not set(state_list).issubset({0, 1}):
                raise ValueError(f"Basis state must only consist of 0s and 1s; got {state_list}")

        super().__init__(state, wires=wires, id=id)

    def _flatten(self):
        state = self.parameters[0]
        state = tuple(state) if isinstance(state, list) else state
        return (state,), (self.wires,)

    @classmethod
    def _unflatten(cls, data, metadata) -> "BasisState":
        return cls(data[0], wires=metadata[0])

    @staticmethod
    def compute_decomposition(state: TensorLike, wires: WiresLike) -> list[Operator]:
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.BasisState.decomposition`.

        Args:
            state (array): the basis state to be prepared
            wires (Iterable, Wires): the wire(s) the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.BasisState.compute_decomposition([1,0], wires=(0,1))
        [X(0)]

        """

        if not qml.math.is_abstract(state):
            return [qml.X(wire) for wire, basis in zip(wires, state) if basis == 1]

        op_list = []
        for wire, basis in zip(wires, state):
            op_list.append(qml.PhaseShift(basis * np.pi / 2, wire))
            op_list.append(qml.RX(basis * np.pi, wire))
            op_list.append(qml.PhaseShift(basis * np.pi / 2, wire))

        return op_list

    def state_vector(self, wire_order: Optional[WiresLike] = None) -> TensorLike:
        """Returns a statevector of shape ``(2,) * num_wires``."""
        prep_vals = self.parameters[0]
        prep_vals_int = math.cast(self.parameters[0], int)

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
    r"""StatePrep(state, wires, pad_with = None, normalize = False, validate_norm = True)
    Prepare subsystems using the given ket vector in the computational basis.

    By setting ``pad_with`` to a real or complex number, ``state`` is automatically padded to dimension
    :math:`2^n` where :math:`n` is the number of qubits used in the template.

    To represent a valid quantum state vector, the L2-norm of ``state`` must be one.
    The argument ``normalize`` can be set to ``True`` to automatically normalize the state.

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
        state (array[complex]): the state vector to prepare
        wires (Sequence[int] or int): the wire(s) the operation acts on
        pad_with (float or complex):  if not None, the input is padded with this constant to size :math:`2^n`
        normalize (bool): whether to normalize the state vector
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
        validate_norm (bool): whether to validate the norm of the input state


    Example:

        StatePrep encodes a normalized :math:`2^n`-dimensional state vector into a state
        of :math:`n` qubits:

        .. code-block:: python

            import pennylane as qml

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit(state=None):
                qml.StatePrep(state, wires=range(2))
                return qml.expval(qml.Z(0)), qml.state()

            res, state = circuit([1/2, 1/2, 1/2, 1/2])

        The final state of the device is - up to a global phase - equivalent to the input passed to the circuit:

        >>> state
        tensor([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j], requires_grad=True)

        **Differentiating with respect to the state**

        Due to non-trivial classical processing to construct the state preparation circuit,
        the state argument is in general **not differentiable**.

        **Normalization**

        The template will raise an error if the state input is not normalized.
        One can set ``normalize=True`` to automatically normalize it:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(state=None):
                qml.StatePrep(state, wires=range(2), normalize=True)
                return qml.expval(qml.Z(0)), qml.state()

            res, state = circuit([15, 15, 15, 15])

        >>> state
        tensor([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j], requires_grad=True)

        **Padding**

        If the dimension of the state vector is smaller than the number of amplitudes,
        one can automatically pad it with a constant for the missing dimensions using the ``pad_with`` option:

        .. code-block:: python

            from math import sqrt

            @qml.qnode(dev)
            def circuit(state=None):
                qml.StatePrep(state, wires=range(2), pad_with=0.)
                return qml.expval(qml.Z(0)), qml.state()

            res, state = circuit([1/sqrt(2), 1/sqrt(2)])

        >>> state
        tensor([0.70710678+0.j, 0.70710678+0.j, 0.        +0.j, 0.        +0.j], requires_grad=True)


    """

    num_wires = AnyWires
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (1,)
    """int: Number of dimensions per trainable parameter of the operator."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        state: TensorLike,
        wires: WiresLike,
        pad_with=None,
        normalize=False,
        id: Optional[str] = None,
        validate_norm: bool = True,
    ):

        state = self._preprocess(state, wires, pad_with, normalize, validate_norm)

        self._hyperparameters = {
            "pad_with": pad_with,
            "normalize": normalize,
            "validate_norm": validate_norm,
        }

        super().__init__(state, wires=wires, id=id)

    # pylint: disable=unused-argument
    @staticmethod
    def compute_decomposition(state: TensorLike, wires: WiresLike, **kwargs) -> list[Operator]:
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

    def _flatten(self):
        metadata = tuple((key, value) for key, value in self.hyperparameters.items())

        return tuple(
            self.parameters,
        ), (metadata, self.wires)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data, **dict(metadata[0]), wires=metadata[1])

    def state_vector(self, wire_order: Optional[WiresLike] = None):
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

    @staticmethod
    def _preprocess(state, wires, pad_with, normalize, validate_norm):
        """Validate and pre-process inputs as follows:

        * If state is batched, the processing that follows is applied to each state set in the batch.
        * Check that the state tensor is one-dimensional.
        * If pad_with is None, check that the last dimension of the state tensor
          has length :math:`2^n` where :math:`n` is the number of qubits. Else check that the
          last dimension of the state tensor is not larger than :math:`2^n` and pad state
          with value if necessary.
        * If normalize is false, check that last dimension of state is normalised to one. Else, normalise the
          state tensor.
        """
        if isinstance(state, (list, tuple)):
            state = math.array(state)

        shape = math.shape(state)

        # check shape
        if len(shape) not in (1, 2):
            raise ValueError(
                f"State must be a one-dimensional tensor, or two-dimensional with batching; got shape {shape}."
            )

        n_states = shape[-1]
        dim = 2 ** len(Wires(wires))
        if pad_with is None and n_states != dim:
            raise ValueError(
                f"State must be of length {dim}; got length {n_states}. "
                f"Use the 'pad_with' argument for automated padding."
            )

        if pad_with is not None:
            normalize = True
            if n_states > dim:
                raise ValueError(
                    f"Input state must be of length {dim} or "
                    f"smaller to be padded; got length {n_states}."
                )

            # pad
            if n_states < dim:
                padding = [pad_with] * (dim - n_states)
                if len(shape) > 1:
                    padding = [padding] * shape[0]
                padding = math.convert_like(padding, state)
                state = math.hstack([state, padding])

        if not validate_norm:
            return state

        # normalize
        if "int" in str(state.dtype):
            state = math.cast_like(state, 0.0)

        norm = math.linalg.norm(state, axis=-1)

        if math.is_abstract(norm):
            if normalize:
                state = state / math.reshape(norm, (*shape[:-1], 1))

        elif not math.allclose(norm, 1.0, atol=TOLERANCE):
            if normalize:
                state = state / math.reshape(norm, (*shape[:-1], 1))
            else:
                raise ValueError(
                    f"The state must be a vector of norm 1.0; got norm {norm}. "
                    "Use 'normalize=True' to automatically normalize."
                )

        return state


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
