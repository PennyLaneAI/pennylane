# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transform function for the Clifford+T decomposition."""

import math
import warnings
from functools import lru_cache, partial
from itertools import product

import pennylane as qml
from pennylane.measurements.mid_measure import MeasurementValue
from pennylane.ops import Adjoint
from pennylane.ops.op_math.decompositions.ross_selinger import rs_decomposition
from pennylane.ops.op_math.decompositions.solovay_kitaev import sk_decomposition
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms.core import transform
from pennylane.transforms.optimization import (
    cancel_inverses,
    commute_controlled,
    merge_rotations,
    remove_barrier,
)
from pennylane.transforms.optimization.optimization_utils import _fuse_global_phases, find_next_gate
from pennylane.typing import PostprocessingFn

# Single qubits Clifford+T gates in PL
_CLIFFORD_T_ONE_GATES = [
    qml.Identity,
    qml.X,
    qml.Y,
    qml.Z,
    qml.Hadamard,
    qml.S,
    qml.SX,
    qml.T,
]

# Two qubits Clifford+T gates in PL
_CLIFFORD_T_TWO_GATES = [
    qml.CNOT,
    qml.CY,
    qml.CZ,
    qml.SWAP,
    qml.ISWAP,
]

# Single-parameter gates in PL
# note: _simplify_param makes use of their periodic nature,
# any additions to this, should be reflected there as well.
_PARAMETER_GATES = (qml.RX, qml.RY, qml.RZ, qml.Rot, qml.PhaseShift)

# Clifford+T gate set
_CLIFFORD_T_GATES = tuple(_CLIFFORD_T_ONE_GATES + _CLIFFORD_T_TWO_GATES) + (qml.GlobalPhase,)

# Gates to be skipped during decomposition
_SKIP_OP_TYPES = (qml.Barrier, qml.Snapshot, qml.WireCut, MeasurementValue)

# Stores the cache of a specified size for the decomposition function
# that is used to decompose the RZ gates in the Clifford+T basis.
_CLIFFORD_T_CACHE = None

_CATALYST_SKIP_OP_TYPES = ()


# pylint: disable=import-outside-toplevel, global-statement
def _add_catalyst_skip_op_types():
    """Delayed addition of PennyLane-Catalyst skip op types."""
    global _CATALYST_SKIP_OP_TYPES
    try:
        from catalyst.api_extensions.quantum_operators import MidCircuitMeasure

        _CATALYST_SKIP_OP_TYPES = (*_CATALYST_SKIP_OP_TYPES, MidCircuitMeasure)
    except (ModuleNotFoundError, ImportError):  # pragma: no cover
        pass


def _check_clifford_op(op, use_decomposition=False):
    r"""Checks if an operator is Clifford or not.

    For a given unitary operator :math:`U` acting on :math:`N` qubits, this method checks that the
    transformation :math:`UPU^{\dagger}` maps the Pauli tensor products :math:`P = {I, X, Y, Z}^{\otimes N}`
    to Pauli tensor products using the decomposition of the matrix for :math:`U` in the Pauli basis.

    Args:
        op (~pennylane.operation.Operation): the operator that needs to be tested
        use_decomposition (bool): if ``True``, use operator's decomposition to compute the matrix, in case
            it doesn't define a ``compute_matrix`` method. Default is ``False``.

    Returns:
        Bool that represents whether the provided operator is Clifford or not.
    """

    # Check if matrix can be calculated for the operator
    if (not op.has_matrix and not use_decomposition) or (
        use_decomposition and not qml.tape.QuantumScript(op.decomposition()).wires
    ):
        return False

    # Compute the LCUs for the operator in Pauli basis
    pauli_terms = qml.pauli_decompose(qml.matrix(op), wire_order=op.wires, check_hermitian=False)
    pauli_terms_adj = qml.Hamiltonian(qml.math.conj(pauli_terms.coeffs), pauli_terms.ops)

    # Build Pauli tensor products P in the Hamiltonian form
    def pauli_group(x):
        return [qml.Identity(x), qml.X(x), qml.Y(x), qml.Z(x)]

    # Build PauliSentence and Hamiltonians representations for set P
    pauli_sens = [
        qml.pauli.pauli_sentence(qml.prod(*pauli))
        for pauli in product(*(pauli_group(idx) for idx in op.wires))
    ]
    pauli_ops = (pauli_sen.operation(wire_order=op.wires) for pauli_sen in pauli_sens)

    # Perform U@P@U^\dagger and check if the result exists in set P
    for pauli_prod in product([pauli_terms], pauli_ops, [pauli_terms_adj]):
        # hopefully op_math.prod scales better than matrix multiplication, i.e., O((2^N)^3)
        upu = qml.pauli.pauli_sentence(qml.prod(*pauli_prod))
        upu.simplify()
        # Pauli sum always lie outside set P
        if len(upu) != 1:
            return False

    return True


def check_clifford_t(op, use_decomposition=False):
    r"""Checks whether the gate is in the standard Clifford+T basis.

    For a given unitary operator :math:`U` acting on :math:`N` qubits, which is not a T-gate,
    this method checks that the transformation :math:`UPU^{\dagger}` maps the Pauli tensor products
    :math:`P = {I, X, Y, Z}^{\otimes N}` to Pauli tensor products using the decomposition of the
    matrix for :math:`U` in the Pauli basis.

    Args:
        op (~pennylane.operation.Operation): the operator that needs to be checked
        use_decomposition (bool): if ``True``, use operator's decomposition to compute the matrix, in case
            it doesn't define a ``compute_matrix`` method. Default is ``False``.

    Returns:
        Bool that represents whether the provided operator is Clifford+T or not.
    """
    # Get the base operation for an adjointed operation
    if isinstance(op, Adjoint):
        base = op.base
    else:
        base = None

    # Save time and check from the pre-computed list
    if isinstance(op, _CLIFFORD_T_GATES) or isinstance(base, _CLIFFORD_T_GATES):
        return True

    # Save time and check from the parameter of rotation gates
    if isinstance(op, _PARAMETER_GATES) or isinstance(base, _PARAMETER_GATES):
        theta = op.data[0]
        return (
            False
            if qml.math.is_abstract(theta)
            else qml.math.allclose(qml.math.mod(theta, math.pi), 0.0)
        )

    return _check_clifford_op(op, use_decomposition=use_decomposition)


def _simplify_param(theta, gate):
    r"""Check if the parameter allows simplification for the rotation gate.

    For the cases where theta is an integer multiple of π: (a) returns a global phase
    when even, and (b) returns combination of provided gate with global phase when odd.
    In rest of the other cases it would return None.
    """
    if qml.math.is_abstract(theta):  # pragma: no cover
        return None

    if qml.math.allclose(theta, 0.0, atol=1e-6):
        return [qml.GlobalPhase(0.0)]

    rem_, mod_ = qml.math.divide(theta, math.pi), qml.math.mod(theta, math.pi)
    if qml.math.allclose(mod_, 0.0, atol=1e-6):
        ops = [qml.GlobalPhase(theta / 2)]
        if qml.math.allclose(qml.math.mod(rem_, 2), 1.0, atol=1e-6):
            ops.append(gate)
        return ops

    return None


# pylint: disable= too-many-branches,
def _rot_decompose(op):
    r"""Decomposes a rotation operation: :class:`~.Rot`, :class:`~.RX`, :class:`~.RY`, :class:`~.RZ`,
    :class:`~.PhaseShift` or a :class`~.QubitUnitary` into a basis composed of :class:`~.RZ`,
    :class:`~.S`, and :class:`~.Hadamard`.
    """
    d_ops = []

    if isinstance(op, qml.QubitUnitary):
        ops = op.decomposition()
        for o in ops[:-1]:
            d_ops.extend(_rot_decompose(o))
        d_ops.append(ops[-1])
        return d_ops

    # Extend for Rot operation with Rz.Ry.Rz decompositions
    if isinstance(op, qml.Rot):
        (phi, theta, omega), wires = op.parameters, op.wires
        for dec in [qml.RZ(phi, wires), qml.RY(theta, wires), qml.RZ(omega, wires)]:
            d_ops.extend(_rot_decompose(dec))
        return d_ops

    (theta,), wires = op.data, op.wires
    if isinstance(op, qml.ops.Adjoint):
        ops_ = _rot_decompose(op.base.adjoint())
    elif isinstance(op, qml.RX):
        ops_ = _simplify_param(theta, qml.X(wires))
        if ops_ is None:  # Use Rx = H @ Rz @ H
            ops_ = [qml.Hadamard(wires), qml.RZ(theta, wires), qml.Hadamard(wires)]
    elif isinstance(op, qml.RY):
        ops_ = _simplify_param(theta, qml.Y(wires))
        if ops_ is None:  # Use Ry = S @ H @ Rz @ H @ S.adjoint()
            ops_ = [
                qml.S(wires),
                qml.Hadamard(wires),
                qml.RZ(theta, wires),
                qml.Hadamard(wires),
                qml.adjoint(qml.S(wires)),
            ][::-1]
    elif isinstance(op, qml.RZ):
        ops_ = _simplify_param(theta, qml.Z(wires))
        if ops_ is None:
            ops_ = [qml.RZ(theta, wires)]
    elif isinstance(op, qml.PhaseShift):
        ops_ = _simplify_param(theta, qml.Z(wires))
        if ops_ is None:
            ops_ = [qml.RZ(theta, wires=wires), qml.GlobalPhase(-theta / 2)]
            if not qml.math.is_abstract(theta):
                # The following loop simplifies the two cases where for all odd intergers `k`,
                # `PhaseShift(k * pi / 2)` is S / S* and `PhaseShift(k * pi / 4)` is T / T*.
                for val_ in [2, 4]:
                    div_ = qml.math.divide(theta, math.pi / val_)
                    mod_ = qml.math.mod(theta, math.pi / val_)
                    if qml.math.allclose(mod_, 0.0, atol=1e-6) and qml.math.allclose(
                        qml.math.mod(div_, 2), 1.0, atol=1e-6
                    ):
                        vop_ = qml.S(wires) if val_ == 2 else qml.T(wires)
                        sign = qml.math.mod(qml.math.floor_divide(div_, 2), 2)
                        ops_ = [
                            vop_ if qml.math.allclose(sign, 0.0, atol=1e-6) else qml.adjoint(vop_)
                        ]
                        break
        else:
            ops_.append(qml.GlobalPhase(-theta / 2))

    else:
        raise ValueError(
            f"Operation {op} is not a supported Pauli rotation: qml.RX, qml.RY, qml.RZ, qml.Rot and qml.PhaseShift."
        )

    d_ops.extend(ops_)
    return d_ops


def _one_qubit_decompose(op):
    r"""Decomposition for single qubit operations using combination of :class:`~.RZ`, :class:`~.S`, and
    :class:`~.Hadamard`."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd_ops = qml.ops.one_qubit_decomposition(
            qml.matrix(op), op.wires, "ZXZ", return_global_phase=True
        )
    # Get the global phase
    gphase_op = sd_ops.pop()

    # Decompose the rotation gates
    d_ops = []
    for sd_op in sd_ops:
        d_ops.extend(_rot_decompose(sd_op) if sd_op.num_params else [sd_op])

    return d_ops, gphase_op


def _two_qubit_decompose(op):
    r"""Decomposition for two qubit operations using combination of :class:`~.RZ`, :class:`~.S`,
    :class:`~.Hadamard`, and :class:`~.CNOT`."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        td_ops = qml.ops.two_qubit_decomposition(qml.matrix(op), op.wires)

    d_ops = []
    for td_op in td_ops:
        d_ops.extend(
            _rot_decompose(td_op) if td_op.num_params and len(td_op.wires) == 1 else [td_op]
        )

    return d_ops


def _merge_param_gates(operations, merge_ops=None):
    """Merge the provided parametrized gates on the same wires that are adjacent to each other"""

    copied_ops = operations.copy()
    merged_ops, number_ops = [], 0

    while len(copied_ops) > 0:
        curr_gate = copied_ops.pop(0)

        # If gate is not to be merged, let it go
        if merge_ops is not None and curr_gate.name not in merge_ops:
            merged_ops.append(curr_gate)
            continue

        # If gate is in the merge_ops, update counter
        if curr_gate.name in merge_ops:
            number_ops += 1

        # Find the next gate that acts on the same wires
        next_gate_idx = find_next_gate(curr_gate.wires, copied_ops)
        if next_gate_idx is None:
            merged_ops.append(curr_gate)
            continue

        # Initialize the current angle and iterate until end of merge
        curr_params = curr_gate.parameters
        curr_intrfc = qml.math.get_deep_interface(curr_gate.parameters)
        cumulative_angles = qml.math.array(curr_params, dtype=float, like=curr_intrfc)
        next_gate = copied_ops[next_gate_idx]
        while curr_gate.name == next_gate.name and curr_gate.wires == next_gate.wires:
            cumulative_angles += qml.math.array(next_gate.parameters, like=curr_intrfc)
            # Check if the subsequent gate exists in the vicinity
            copied_ops.pop(next_gate_idx)
            next_gate_idx = find_next_gate(curr_gate.wires, copied_ops)
            if next_gate_idx is None:
                break
            next_gate = copied_ops[next_gate_idx]

        # Replace the current gate, add it to merged list and pop it from orginal one
        merged_ops.append(curr_gate.__class__(*cumulative_angles, wires=curr_gate.wires))

    return merged_ops, number_ops


# NOTE: The cache size is set to 2000, which is a reasonable size for the mapping
# decomposition in circuit with up to 500 wires. This is based on the fact that the
# decompositions being mapped would contain {H, S, T} (with or without {Sadj, Tadj}).
@lru_cache(maxsize=2000)
def _map_wires(op, wire):
    """Maps the operator to the provided wire."""
    return op.map_wires({0: wire})


class _CachedCallable:
    """A class to cache the decomposition of the operators.

    Args:
        method (str): The method to be used for decomposition.
        epsilon (float): The maximum permissible operator norm error for the decomposition.
        cache_size (int): The size of the cache built for the decomposition function based on the angle.
        is_qjit (bool): Whether the decomposition is being performed with QJIT enabled.
        **method_kwargs: Keyword argument to pass options for the ``method`` used for decompositions.
    """

    def __init__(self, method, epsilon, cache_size, is_qjit=False, **method_kwargs):
        match method:
            case "sk":
                self.decompose_fn = lru_cache(maxsize=cache_size)(
                    partial(sk_decomposition, epsilon=epsilon, **method_kwargs)
                )
            case "gridsynth":
                self.decompose_fn = lru_cache(maxsize=cache_size)(
                    partial(rs_decomposition, epsilon=epsilon, is_qjit=is_qjit, **method_kwargs)
                )
            case _:
                raise NotImplementedError(
                    f"Currently we only support Solovay-Kitaev ('sk') and Ross-Selinger ('gridsynth') decompositions, got {method}"
                )

        self.method = method
        self.epsilon = epsilon
        self.cache_size = cache_size
        self.is_qjit = is_qjit
        self.method_kwargs = method_kwargs
        self.query = lru_cache(maxsize=cache_size)(self.cached_decompose)

    # pylint: disable=too-many-arguments
    def compatible(self, method, epsilon, cache_size, cache_eps_rtol, is_qjit, **method_kwargs):
        """Check compatibility based on `method`, `epsilon`, `cache_eps_rtol` and `method_kwargs`."""
        return (
            self.method == method
            and self.epsilon <= epsilon
            and (
                qml.math.allclose(self.epsilon, epsilon, rtol=cache_eps_rtol, atol=0.0)
                if cache_eps_rtol is not None
                else True
            )
            and self.cache_size <= cache_size
            and self.is_qjit == is_qjit
            and self.method_kwargs == method_kwargs
        )

    def cached_decompose(self, op):
        """Decomposes the angle into a sequence of gates."""
        if not self.is_qjit:
            f_adj = op.data[0] < 2 * math.pi  # 4 * pi / 2
            cached_op = op if f_adj else op.adjoint()

            seq = self.decompose_fn(cached_op)
            if f_adj:
                return seq

            adj = [qml.adjoint(s, lazy=False) for s in reversed(seq)]
            return adj[1:] + adj[:1]

        return self.decompose_fn(op)


# pylint: disable=too-many-branches,too-many-statements
@transform
def clifford_t_decomposition(
    tape: QuantumScript,
    epsilon=1e-4,
    method="sk",
    cache_size=1000,
    cache_eps_rtol=None,
    **method_kwargs,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Decomposes a circuit into the Clifford+T basis.

    This method first decomposes the gate operations to a basis comprised of Clifford, :class:`~.T`, :class:`~.RZ` and
    :class:`~.GlobalPhase` operations (and their adjoints). The Clifford gates include the following PennyLane operations:

    - Single qubit gates - :class:`~.Identity`, :class:`~.PauliX`, :class:`~.PauliY`, :class:`~.PauliZ`,
      :class:`~.SX`, :class:`~.S`, and :class:`~.Hadamard`.
    - Two qubit gates - :class:`~.CNOT`, :class:`~.CY`, :class:`~.CZ`, :class:`~.SWAP`, and :class:`~.ISWAP`.

    Then, the leftover single qubit :class:`~.RZ` operations are approximated in the Clifford+T basis with
    :math:`\epsilon > 0` error. By default, we use the Solovay-Kitaev algorithm described in
    `Dawson and Nielsen (2005) <https://arxiv.org/abs/quant-ph/0505030>`_ for this.
    Alternatively, the Ross-Selinger algorithm described in `Ross and Selinger (2016) <https://arxiv.org/abs/1403.2975v3>`_
    can be used by setting the ``method`` to ``"gridsynth"``.

    Args:
        tape (QNode or QuantumTape or Callable): The quantum circuit to be decomposed.
        epsilon (float): The maximum permissible operator norm error of the complete circuit decomposition. Defaults to ``0.0001``.
        method (str): Method to be used for Clifford+T decomposition. Default value is ``"sk"`` for Solovay-Kitaev. Alternatively,
            the Ross-Selinger algorithm can be used with ``"gridsynth"``.
        cache_size (int): The size of the cache built for the decomposition function based on the angle. Defaults to ``1000``.
        cache_eps_rtol (Optional[float]): The relative tolerance for ``epsilon`` values between which the cache may be reused.
            Defaults to ``None``, which means that a cached decomposition will be used if it is `at least as precise` as the requested error.
        **method_kwargs: Keyword argument to pass options for the ``method`` used for decompositions.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described
        in the :func:`qml.transform <pennylane.transform>`.

    **Keyword Arguments**

    - Solovay-Kitaev decomposition --
        **max_depth** (int), **basis_set** (list[str]), **basis_length** (int) -- arguments for the ``"sk"`` method,
        where the decomposition is performed using the :func:`~.sk_decomposition` method.

    - Ross-Selinger (``gridsynth``) decomposition --
        **max_search_trials** (int), **max_factoring_trials** (int) -- arguments for the ``"gridsynth"`` method,
        where the decomposition is performed using the :func:`~.rs_decomposition` method.

    Raises:
        ValueError: If a gate operation does not have a decomposition when required.
        NotImplementedError: If chosen decomposition ``method`` is not supported.

    .. seealso:: :func:`~.rs_decomposition` and :func:`~.sk_decomposition` for Ross-Selinger and Solovay-Kitaev decomposition methods, respectively.

    **Example**

    .. code-block:: python3

        @qml.qnode(qml.device("default.qubit"))
        def circuit(x, y):
            qml.RX(x, 0)
            qml.CNOT([0, 1])
            qml.RY(y, 0)
            return qml.expval(qml.Z(0))

        x, y = 1.1, 2.2
        decomposed_circuit = qml.transforms.clifford_t_decomposition(circuit)
        result = circuit(x, y)
        approx = decomposed_circuit(x, y)

    >>> qml.math.allclose(result, approx, atol=1e-4)
    True
    """
    with QueuingManager.stop_recording():
        # Build the basis set and the pipeline for initial compilation pass
        basis_set = [op.__name__ for op in _PARAMETER_GATES + _CLIFFORD_T_GATES + _SKIP_OP_TYPES]
        pipelines = [remove_barrier, commute_controlled, cancel_inverses, merge_rotations]

        # Compile the tape according to depth provided by the user and expand it
        [compiled_tape], _ = qml.compile(tape, pipelines, basis_set=basis_set)

        if not _CATALYST_SKIP_OP_TYPES:
            _add_catalyst_skip_op_types()

        # Now iterate over the expanded tape operations
        decomp_ops, gphase_ops = [], []
        for op in compiled_tape.operations:
            # Check whether operation is to be skipped
            if isinstance(op, _SKIP_OP_TYPES + _CATALYST_SKIP_OP_TYPES):
                decomp_ops.append(op)

            # Check whether the operation is a global phase
            elif isinstance(op, qml.GlobalPhase):
                gphase_ops.append(op)

            # Check whether the operation is a Clifford or a T-gate
            elif op.name in basis_set and check_clifford_t(op):
                if op.num_params:
                    decomp_ops.extend(_rot_decompose(op))
                else:
                    decomp_ops.append(op)

            # Decompose and then iteratively go deeper via DFS
            else:
                # Single qubit unitary decomposition with ZXZ rotations
                if op.num_wires == 1:
                    if op.name in basis_set:
                        d_ops = _rot_decompose(op)
                    else:
                        d_ops, g_op = _one_qubit_decompose(op)
                        gphase_ops.append(g_op)
                    decomp_ops.extend(d_ops)

                # Two qubit unitary decomposition with SU(4) rotations
                elif op.num_wires == 2:
                    d_ops = _two_qubit_decompose(op)
                    decomp_ops.extend(d_ops)

                # If we don't know how to decompose the operation
                else:
                    raise ValueError(
                        f"Cannot unroll {op} into the Clifford+T basis as no rule exists for its decomposition"
                    )

        # Merge RZ rotations together
        merged_ops, number_ops = _merge_param_gates(decomp_ops, merge_ops=["RZ"])

        # Squeeze global phases into a single global phase
        new_operations = _fuse_global_phases(merged_ops + gphase_ops)

        # Compute the per-gate epsilon value
        epsilon /= number_ops or 1

        # _CACHED_DECOMPOSE is a global variable that caches the decomposition function,
        # where the implementation of each function should have the following signature:
        # def decompose_fn(op: Operator, epsilon: float, **method_kwargs) -> List[Operator]
        # note: the last operator in the decomposition must be a GlobalPhase

        is_qjit = qml.compiler.active_compiler() == "catalyst"

        # Build the decomposition cache based on the method
        global _CLIFFORD_T_CACHE  # pylint: disable=global-statement
        if _CLIFFORD_T_CACHE is None or not _CLIFFORD_T_CACHE.compatible(
            method, epsilon, cache_size, cache_eps_rtol, is_qjit, **method_kwargs
        ):
            _CLIFFORD_T_CACHE = _CachedCallable(
                method, epsilon, cache_size, is_qjit, **method_kwargs
            )

        decomp_ops = []
        phase = new_operations.pop().data[0]
        for op in new_operations:
            if isinstance(op, qml.RZ):
                # If simplifies to Identity, skip it
                if not (op_param := op.simplify().data):
                    continue
                wire = op.wires[0] if is_qjit else 0
                # Decompose the RZ operation with a default wire
                clifford_ops = _CLIFFORD_T_CACHE.query(qml.RZ(op_param[0], [wire]))
                op_wire = op.wires[0]
                # Extract the global phase from the last operation
                # Map the operations to the original wires
                if is_qjit:
                    phase += clifford_ops[-1].data[0]
                    decomp_ops.extend(clifford_ops[:-1])  # Already mapped
                else:
                    phase += qml.math.convert_like(clifford_ops[-1].data[0], phase)
                    decomp_ops.extend([_map_wires(cl_op, op_wire) for cl_op in clifford_ops[:-1]])
            else:
                decomp_ops.append(op)

        # check if phase is non-zero for non jax-jit cases
        if qml.math.is_abstract(phase) or not qml.math.allclose(phase, 0.0):
            decomp_ops.append(qml.GlobalPhase(phase))

    # Construct a new tape with the expanded set of operations
    # and then clear `decomp_ops` list to free up the memory
    new_tape = compiled_tape.copy(operations=decomp_ops)
    decomp_ops.clear()

    # Perform a final attempt of simplification before return
    if not is_qjit:
        # This is skipped for qjit because when qjit is enabled, the circuit may contain
        # higher-level operations such as Cond and ForLoop whose wires attribute does not
        # reflect the wires of all operators within its scope.
        [new_tape], _ = cancel_inverses(new_tape)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
