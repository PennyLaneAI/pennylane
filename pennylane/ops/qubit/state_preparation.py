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
# pylint: disable=too-many-branches,arguments-differ
from warnings import warn

import numpy as np
import scipy as sp
from scipy.sparse import csr_array, csr_matrix

import pennylane as qml
from pennylane import math
from pennylane.decomposition import add_decomps, register_resources
from pennylane.exceptions import WireError
from pennylane.operation import Operation, Operator, StatePrepBase
from pennylane.templates.state_preparations import MottonenStatePreparation
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

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
        state (tensor_like): Binary input of shape ``(len(wires), )``. For example, if ``state=np.array([0, 1, 0])`` or ``state=2`` (equivalent to 010 in binary), the quantum system will be prepared in the state :math:`|010 \rangle`.

        wires (Sequence[int] or int): the wire(s) the operation acts on
        id (str): Custom label given to an operator instance. Can be useful for some applications where the instance has to be identified.

    **Example**

    >>> dev = qml.device('default.qubit', wires=2)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.BasisState(np.array([1, 1]), wires=range(2))
    ...     return qml.state()
    >>> print(example_circuit())
    [0.+0.j 0.+0.j 0.+0.j 1.+0.j]
    """

    resource_keys = {"num_wires"}

    @property
    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires)}

    def __init__(self, state, wires: WiresLike, id=None):

        wires = Wires(wires)
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
        state = qml.math.cast(state, int)
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

    def state_vector(self, wire_order: WiresLike | None = None) -> TensorLike:
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


def _basis_state_decomp_resources(num_wires):
    # Represent one of the X gates as an RX and a GlobalPhase because RX is
    # used when jax-jit is enabled without capture/qjit.
    return {qml.X: num_wires - 1 or num_wires, qml.RX: 1, qml.GlobalPhase: 1}


@register_resources(_basis_state_decomp_resources, exact=False)
def _basis_state_decomp(state, wires, **__):

    if qml.math.is_abstract(state) and not (qml.capture.enabled() or qml.compiler.active()):
        # This branch is for supporting jax-jit without capture/qjit.
        global_phase = 0.0
        for wire, basis in zip(wires, state):
            qml.RX(basis * np.pi, wires=wire)
            global_phase += basis * np.pi / 2
        qml.GlobalPhase(-global_phase)
        return

    @qml.for_loop(0, len(wires), 1)
    def _loop(i):
        qml.cond(qml.math.allclose(state[i], 1), qml.X)(wires[i])

    _loop()  # pylint: disable=no-value-for-parameter


add_decomps(BasisState, _basis_state_decomp)


class StatePrep(StatePrepBase):
    r"""StatePrep(state, wires, pad_with = None, normalize = False, validate_norm = False)
    Prepare subsystems using a state vector in the computational basis.

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
        state (array[complex] or csr_matrix): the state vector to prepare
        wires (Sequence[int] or int): the wire(s) the operation acts on
        pad_with (float or complex): if not ``None``, ``state`` is padded with this constant to be of size :math:`2^n`, where
            :math:`n` is the number of wires.
        normalize (bool): whether to normalize the state vector. To represent a valid quantum state vector, the L2-norm
            of ``state`` must be one. The argument ``normalize`` can be set to ``True`` to normalize the state automatically.
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
        array([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j])

    .. details::
        :title: Usage Details

        **Differentiating with respect to the state**

        Due to non-trivial classical processing to construct the state preparation circuit,
        the state argument is, in general, **not differentiable**.

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
        array([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j])

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
        array([0.70710678+0.j, 0.70710678+0.j, 0.        +0.j, 0.        +0.j])

        **Sparse state input**
        `state` can also be provided as a sparse matrix.  The state will be implicitly
        zero-padded to the full Hilbert space dimension.

        .. code-block:: pycon

            >>> init_state = sp.sparse.csr_matrix([0, 0, 1, 0])
            >>> qsv_op = qml.StatePrep(init_state, wires=[1, 2])
            >>> wire_order = [0, 1, 2]
            >>> ket = qsv_op.state_vector(wire_order=wire_order)
            >>> print(ket)  # Sparse representation
            <Compressed Sparse Row sparse array of dtype 'int64'
                with 1 stored elements and shape (1, 8)>
              Coords    Values
              (0, 2)    1
            >>> print(ket.toarray().flatten())  # Dense representation
            [0 0 1 0 0 0 0 0]

            # Normalization also works with sparse inputs:
            >>> init_state_sparse = sp.sparse.csr_matrix([1, 1, 1, 1]) # Unnormalized
            >>> qsv_op_norm = qml.StatePrep(init_state_sparse, wires=range(2), normalize=True)
            >>> ket_norm = qsv_op_norm.state_vector()
            >>> print(ket_norm.toarray().flatten()) # Normalized dense representation
            [0.5 0.5 0.5 0.5]


    """

    resource_keys = frozenset({"num_wires"})

    @property
    def resource_params(self):
        return {"num_wires": len(self.wires)}

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (1,)
    """int: Number of dimensions per trainable parameter of the operator."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        state: TensorLike | csr_matrix,
        wires: WiresLike,
        pad_with=None,
        normalize: bool = False,
        id: str | None = None,
        validate_norm: bool = False,
    ):
        self.is_sparse = False
        if sp.sparse.issparse(state):
            state = state.tocsr()
            state = self._preprocess_csr(
                state, wires, pad_with=pad_with, normalize=normalize, validate_norm=validate_norm
            )
            self.is_sparse = True
        else:
            state = self._preprocess(
                state, wires, pad_with=pad_with, normalize=normalize, validate_norm=validate_norm
            )

        self._hyperparameters = {
            "pad_with": pad_with,
            "normalize": normalize,
            "validate_norm": validate_norm,
        }

        super().__init__(state, wires=wires, id=id)

    def _check_batching(self):
        if self.is_sparse:
            self._batch_size = None
        else:
            super()._check_batching()

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
        [MottonenStatePreparation(array([1, 0, 0, 0]), wires=[0, 1])]

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

    def state_vector(self, wire_order: WiresLike | None = None):

        if self.is_sparse:
            op_vector = _sparse_statevec_permute_and_embed(
                self.parameters[0], self.wires, wire_order
            )
            return csr_array(op_vector)

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

        if not (validate_norm or normalize):
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

    @staticmethod
    def _preprocess_csr(state, wires, pad_with, normalize, validate_norm):
        """Validate and pre-process inputs as follows:

        * If the state is batched, the following processing is applied to each state set in the batch.
        * Check that the state tensor is one-dimensional.
        * pad_with has to be None.
        * If normalize is false, check that the last dimension of the state is normalized to one. Else, normalize the
          state tensor.
        """

        if pad_with:
            raise ValueError("Non-zero Padding is not supported for sparse states")
        shape = state.shape

        # Check shape. Note that csr_matrix is always 2D; scipy should have already checked that the input is a 2D array
        if len(shape) == 2 and shape[0] != 1:
            raise NotImplementedError(
                "StatePrep does not yet support parameter broadcasting with sparse state vectors."
            )

        n_states = shape[-1]
        dim = 2 ** len(Wires(wires))
        if n_states > dim:
            raise ValueError(
                f"State must be of length {dim} or smaller to be padded; got length {n_states}."
            )
        if n_states < dim:
            warn(
                f"State must be of length {dim}; got length {n_states}. "
                f"Automatically padding with zeros.",
                UserWarning,
            )
            # pad a csr_matrix with zeros
            state.resize((1, dim))

        if not (validate_norm or normalize):
            return state

        # normalize
        if np.issubdtype(state.dtype, np.integer):
            state = state.astype(float)

        norm = sp.sparse.linalg.norm(state)

        if normalize:
            state /= norm

        elif not math.allclose(norm, 1.0, atol=TOLERANCE):
            raise ValueError(
                f"The state must be a vector of norm 1.0; got norm {norm}. "
                "Use 'normalize=True' to automatically normalize."
            )
        return state


def _stateprep_resources(num_wires):
    return {qml.resource_rep(qml.MottonenStatePreparation, num_wires=num_wires): 1}


@register_resources(_stateprep_resources)
def _state_prep_decomp(state, wires, **_):
    qml.MottonenStatePreparation(state, wires)


add_decomps(StatePrep, _state_prep_decomp)


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
        array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
    """

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    grad_method = None


def _sparse_statevec_permute_and_embed(
    state: csr_matrix, wires: list, wire_order: list
) -> csr_matrix:
    """Permutes the wires of a statevector represented as a scipy.sparse.csr_matrix. If `wire_order` contains `wires`, then embed the `state` with corresponding orders, padding with bit 0 on other wires.

    Args:
        state (csr_matrix): the input statevector
        wires (Iterable[int]): the wires of the input statevector
        wire_order (Iterable[int]): the wires of the output statevector. E.g., [0, 2, 1] means the permutation of wires 0, 1, 2 to 0, 2, 1. wires=[2, 1] and wire_order=[1, 0, 2] means embedding the input state in a permuted order.

    Returns:
        csr_matrix: the permuted statevector
    """
    wires = Wires(wires)
    wire_order = Wires(wire_order) if wire_order else wires

    if not wire_order.contains_wires(wires):
        raise WireError(
            f"wire_order must contain all wires. Got wires {wires} and wire_order {wire_order}"
        )

    if wires == wire_order:
        return state

    index_map = _build_index_map(wires, wire_order)
    perm_pos = index_map[state.indices]
    new_csr = csr_matrix((state.data, perm_pos, state.indptr), shape=(1, 2 ** len(wire_order)))
    return new_csr


def _build_index_map(wires, wire_order):
    n_wires = len(wires)
    index_map = np.zeros(2**n_wires, dtype=int)
    for pos in range(2**n_wires):
        pos_bin = format(pos, f"0{n_wires}b")
        wire_values_map = {wire: pos_bin[i] for i, wire in enumerate(wires)}
        pos_bin_perm = [wire_values_map[wire] if wire in wires else "0" for wire in wire_order]
        index_map[pos] = int("".join(pos_bin_perm), 2)
    return index_map
