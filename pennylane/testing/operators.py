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
"""
This module contains ``assert_valid``, which checks that an operator is correctly defined.
"""

import copy
import itertools
import pickle
from collections import defaultdict
from functools import partial

import jax
import numpy as np
import scipy.sparse

from pennylane import capture, decomposition, math, ops, pytrees
from pennylane._grad import jacobian
from pennylane.allocation import Allocate
from pennylane.core import apply, queuing
from pennylane.core.operator import Operator, Operator1, Operator2, abstractify
from pennylane.decomposition import has_decomp, list_decomps
from pennylane.decomposition.decomposition_rule import DecompositionRule, _decomp_contains_mcm
from pennylane.decomposition.resources import CompressedResourceOp
from pennylane.decomposition.utils import _get_decomp_args, to_name
from pennylane.devices import device
from pennylane.exceptions import (
    DecompositionUndefinedError,
    DiagGatesUndefinedError,
    EigvalsUndefinedError,
    GeneratorUndefinedError,
    MatrixUndefinedError,
    SparseMatrixUndefinedError,
)
from pennylane.measurements import MeasurementProcess, probs
from pennylane.ops import functions as op_funcs
from pennylane.ops.mid_measure import MidMeasure, PauliMeasure
from pennylane.ops.op_math import (
    Adjoint2,
    CompositeOp2,
    Conditional,
    Controlled,
    ControlledOp2,
    Pow2,
    SymbolicOp2,
)
from pennylane.tape import QuantumScript
from pennylane.templates.core import SubroutineOp
from pennylane.templates.subroutines.qsvt import QSVT
from pennylane.templates.subroutines.select import Select
from pennylane.transforms import resolve_dynamic_wires
from pennylane.typing import AbstractArray, AbstractWires, TensorLike
from pennylane.wires import Wires
from pennylane.workflow import QNode

from .capture import plxpr_to_tape


def _assert_error_raised(func, expected_error_type, failure_comment):

    def inner_func(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except expected_error_type:
            return
        raise AssertionError(failure_comment)

    return inner_func


def _resolve_dynamic_wires(ops, num_zeroed):
    """Apply the transform resolve_dynamic_wires to a list of operations or tape."""

    if not isinstance(ops, QuantumScript):
        tape = QuantumScript(ops)
        return _resolve_dynamic_wires(tape, num_zeroed).operations

    zeroed = range(len(ops.wires), len(ops.wires) + num_zeroed)
    return resolve_dynamic_wires([ops], zeroed=zeroed)[0][0]


def _assert_equal(op0, op1):
    """Assert that two operators are equal using :func:`~.assert_equal`, except that mid-circuit
    measurements which differ only by their ``meas_uid`` are considered equal."""

    # strip the meas_uid from the arguments before comparing
    if isinstance(op0, (MidMeasure, PauliMeasure)) and isinstance(op1, type(op0)):
        op0 = type(op0)(**{**op0.arguments, "meas_uid": None})
        op1 = type(op1)(**{**op1.arguments, "meas_uid": None})

    op_funcs.assert_equal(op0, op1)


def _assert_equal_ops(ops0, ops1):
    """Assert that two sequences of operators are equal for the purpose of testing."""
    len0, len1 = len(ops0), len(ops1)
    assert len0 == len1, f"The two lists of ops have different lengths, got {len0} and {len1}"
    for op0, op1 in zip(ops0, ops1, strict=True):
        _assert_equal(op0, op1)


# pylint: disable=too-many-branches
def _check_decomposition_legacy(op, skip_wire_mapping):
    """Checks involving the decomposition."""

    if not op.has_decomposition:
        failure_comment = "DecompositionUndefinedError expected if op.has_decomposition is False."
        _assert_error_raised(
            op.decomposition,
            DecompositionUndefinedError,
            failure_comment=failure_comment,
        )()
        _, args, kwargs = _get_decomp_args(op)
        _assert_error_raised(
            op.compute_decomposition,
            DecompositionUndefinedError,
            failure_comment=failure_comment,
        )(*args, **kwargs)
        return

    with queuing.AnnotatedQueue() as queued_decomp:
        decomp_output = op.decomposition()

    queued_tape = QuantumScript.from_queue(queued_decomp)

    try:
        _, args, kwargs = _get_decomp_args(op)
        compute_decomp_output = type(op).compute_decomposition(*args, **kwargs)
    except (DecompositionUndefinedError, TypeError):
        # sometimes decomposition is defined but not compute_decomposition
        # Also sometimes compute_decomposition can have a different signature
        compute_decomp_output = None

    assert isinstance(decomp_output, list), "op.decomposition must return a list"
    if compute_decomp_output is not None:
        assert isinstance(compute_decomp_output, list), "op.compute_decomposition must be a list"

    allocations = [op for op in decomp_output if isinstance(op, Allocate)]
    if allocations:
        total_work_wires = sum(len(op.wires) for op in allocations)
        decomp_output = _resolve_dynamic_wires(decomp_output, total_work_wires)
        if compute_decomp_output is not None:
            compute_decomp_output = _resolve_dynamic_wires(compute_decomp_output, total_work_wires)
        queued_tape = _resolve_dynamic_wires(queued_tape, total_work_wires)

    assert op not in decomp_output, "an operator should not be included in its own decomposition"

    if compute_decomp_output is not None:
        try:
            _assert_equal_ops(decomp_output, compute_decomp_output)
        except AssertionError as e:
            raise AssertionError("decomposition must match compute_decomposition") from e

    try:
        _assert_equal_ops(decomp_output, queued_tape)
    except AssertionError as e:
        raise AssertionError("decomposition must match queued operators.") from e

    if op.has_matrix:
        op_matrix = op.matrix()
        decomp_mat = op_funcs.matrix(decomp_output, wire_order=op.wires)
        comment = f"The decomposition does not match op.matrix. Got \n{op_matrix}\n\n {decomp_mat}"
        assert math.allclose(op_matrix, decomp_mat), comment

    if not skip_wire_mapping:

        # Check that mapping wires transitions to the decomposition. We use 10 * len(op.wires)
        # here because op.wires often does not contain the work wires, which may cause the mapped
        # wires to overlap with the work wires, so we space it out to be safe.
        wire_map = {w: i + 10 * len(op.wires) for i, w in enumerate(op.wires)}
        mapped_op = op.map_wires(wire_map)

        # calling `map_wires` on a Controlled operator generates a new `op` from the controls and
        # base, so may return a different class of operator. We only compare decomps of `op` and
        # `mapped_op` if `mapped_op` **has** a decomposition.
        if not mapped_op.has_decomposition:
            return

        mapped_decomp = mapped_op.decomposition()
        if allocations:
            mapped_decomp = _resolve_dynamic_wires(mapped_decomp, total_work_wires)

        for mapped_op, orig_op in zip(mapped_decomp, decomp_output, strict=True):
            comment = "Operators in decomposition of wire-mapped operator must have mapped wires."
            # pylint: disable-next=no-member
            assert mapped_op.wires == op_funcs.map_wires(orig_op, wire_map).wires, comment


def _check_decomposition_new(op, skip_decomp_matrix_check=False):
    """Checks involving the new system of decompositions."""

    op_type = type(op)

    if isinstance(op, Operator1):

        err_msg = "resource_params must have the same keys as specified by resource_keys"
        assert set(op.resource_params.keys()) == set(op_type.resource_keys), err_msg

        if op_type.resource_params is Operator.resource_params:
            err_msg = "resource_params must be defined for operators with decompositions"
            assert not has_decomp(op_type), err_msg

    for rule in list_decomps(op):
        _test_decomposition_rule(op, rule, skip_decomp_matrix_check)

    for rule in list_decomps(f"Adjoint({to_name(op)})"):
        adj_op = ops.adjoint(op)
        _test_decomposition_rule(adj_op, rule, skip_decomp_matrix_check)

    for rule in list_decomps(f"Pow({to_name(op)})"):
        for z in [2, 3, 4, 8, 9]:
            pow_op = ops.pow(op, z)
            _test_decomposition_rule(pow_op, rule, skip_decomp_matrix_check)

    for rule in list_decomps(f"C({to_name(op)})"):
        for n_ctrl_wires, c_value, n_workers in itertools.product([1, 2, 3], [0, 1], [0, 1, 2]):
            ctrl = Controlled if isinstance(op, Operator1) else ControlledOp2
            int_wires = [w for w in op.wires if isinstance(w, int)]
            max_wire = max(int_wires) if int_wires else -1
            control_wires = [i + max_wire + 1 for i in range(n_ctrl_wires)]
            control_values = [c_value] * n_ctrl_wires
            work_wires = [i + max(control_wires) + 1 for i in range(n_workers)]
            ctrl_op = ctrl(op, control_wires, control_values, work_wires)
            _test_decomposition_rule(ctrl_op, rule, skip_decomp_matrix_check)


def _assert_counts_match(counts_0, counts_1):

    if counts_0 == counts_1:
        return

    miscounts = [
        (op, val, counts_0[op])
        for op, val in counts_1.items()
        if op in counts_0 and val != counts_0[op]
    ]
    if miscounts:
        op_len = max([8] + [len(str(op)) for op, *_ in miscounts])
        miscounts_str = (
            f"\nThe numbers are off for following ops:"
            f"\n{'Operator'.rjust(op_len)} : Actual  !=  Resource function\n"
        )
        miscounts_str += "\n".join(
            f"{str(op).rjust(op_len)} : {str(val0).rjust(6)}  !=  {val1}"
            for op, val0, val1 in miscounts
        )
    else:
        miscounts_str = ""
    assertion_error_string = (
        f"\nGate counts expected from resource function:\n{counts_0}"
        f"\nActual gate counts:\n{dict(counts_1)}"
        f"{miscounts_str}"
        "\nMissing entirely in gate counts from resource function:\n"
        f"{[op for op in counts_1 if op not in counts_0]}\n"
        "Missing entirely in actual gate counts:\n"
        f"{[op for op in counts_0 if op not in counts_1]}"
    )
    raise AssertionError(assertion_error_string)


def _decomp_rule_to_tape(rule, args, kwargs):
    with queuing.AnnotatedQueue() as q:
        rule(*args, **kwargs)
    return QuantumScript.from_queue(q)


def _capture_decomp_rule_to_tape(rule, op):

    # Match each operator model's capture boundary: legacy hyperparameters remain
    # closed over, while Operator2 exposes its dynamic, wire, and hybrid arguments.
    if isinstance(op, Operator1):
        decomposition = partial(rule, **op.hyperparameters)
        capture_args = op.data
        capture_kwargs = {"wires": op.wires}
    else:
        decomposition = partial(rule, **op.static_args, **op.compilable_args)
        capture_args = ()
        wire_args = {k: math.array(w, like="jax") for k, w in op.wire_args.items()}
        capture_kwargs = {**op.dynamic_args, **wire_args, **op.hybrid_args}

    plxpr = capture.make_plxpr(decomposition, autograph=False)(*capture_args, **capture_kwargs)
    flat_capture_args = jax.tree.leaves((capture_args, capture_kwargs))
    return plxpr_to_tape(plxpr.jaxpr, plxpr.consts, *flat_capture_args)


def _test_decomposition_rule(op, rule: DecompositionRule, skip_decomp_matrix_check: bool = False):
    """Tests that a decomposition rule is consistent with the operator."""

    params, args, kwargs = _get_decomp_args(op)

    if not rule.is_applicable(**params):
        return

    # Test that the resource function is correct
    resources = rule.compute_resources(**params)
    estimated_gate_counts = resources.gate_counts

    # Make sure all counts are int
    for gate, count in estimated_gate_counts.items():
        assert isinstance(count, int), (
            f"Resource count for '{gate}' in '{op.name}' decomp rule '{rule.name}' must be an integer, "
            f"but got {type(count)} ({count}). "
        )

    tape = (
        _capture_decomp_rule_to_tape(rule, op)
        if capture.enabled()
        else _decomp_rule_to_tape(rule, args, kwargs)
    )

    total_work_wires = rule.get_work_wire_spec(**params).total
    if total_work_wires:
        tape = _resolve_dynamic_wires(tape, total_work_wires)

    actual_gate_counts = defaultdict(int)
    for _op in tape.operations:
        if isinstance(_op, Conditional):
            _op = _op.base
        op_rep = abstractify(_op)
        actual_gate_counts[op_rep] += 1
    actual_gate_counts = dict(sorted(actual_gate_counts.items(), key=lambda item: str(item[0])))

    if rule.exact_resources and not (
        isinstance(op, SubroutineOp) and not op.subroutine.exact_resources
    ):
        non_zero_gate_counts = {k: v for k, v in estimated_gate_counts.items() if v > 0}
        _assert_counts_match(non_zero_gate_counts, actual_gate_counts)
    else:
        # If the resource estimate is not expected to match exactly to the actual
        # decomposition, at least make sure that all gates are accounted for.
        assert all(op in estimated_gate_counts for op in actual_gate_counts), (
            "\nGate counts expected from resource function to contain actual gates:\n"
            f"{list(estimated_gate_counts.keys())}\nActual gates:\n{list(actual_gate_counts.keys())}\n"
            "Missing in gate counts from resource function:\n"
            f"{[op for op in actual_gate_counts if op not in estimated_gate_counts]}"
        )

    # Tests that the decomposition produces the same matrix
    if op.has_matrix and not skip_decomp_matrix_check and not _decomp_contains_mcm(rule, params):
        # Add projector to the additional wires (work wires) on the tape
        work_wires = tape.wires - op.wires
        all_wires = op.wires + work_wires
        if work_wires:
            op = op @ ops.Projector([0] * len(work_wires), wires=work_wires)
            tape.operations.insert(0, ops.Projector([0] * len(work_wires), wires=work_wires))

        op_matrix = op.matrix(wire_order=all_wires)
        with capture.pause():
            decomp_matrix = op_funcs.matrix(tape, wire_order=all_wires)
        assert math.allclose(
            op_matrix, decomp_matrix
        ), "decomposition must produce the same matrix as the operator."


def _check_matrix(op):
    """Check that if the operation says it has a matrix, it does. Otherwise a ``MatrixUndefinedError`` should be raised."""
    if op.has_matrix:
        mat = op.matrix()
        assert isinstance(mat, TensorLike), "matrix must be a TensorLike"
        l = 2 ** len(op.wires)
        failure_comment = f"matrix must be two dimensional with shape ({l}, {l})"
        assert math.shape(mat) == (l, l), failure_comment
    else:
        failure_comment = (
            "If has_matrix is False, the matrix method must raise a ``MatrixUndefinedError``."
        )
        _assert_error_raised(op.matrix, MatrixUndefinedError, failure_comment=failure_comment)()


def _check_sparse_matrix(op):
    """Check that if the operation says it has a sparse matrix, it does. Otherwise a ``SparseMatrixUndefinedError`` should be raised."""
    if op.has_sparse_matrix:
        mat = op.sparse_matrix()
        assert isinstance(mat, scipy.sparse.csr_matrix), "matrix must be a TensorLike"
        l = 2 ** len(op.wires)
        failure_comment = f"matrix must be two dimensional with shape ({l}, {l})"
        assert math.shape(mat) == (l, l), failure_comment

        assert isinstance(
            op.sparse_matrix(), scipy.sparse.csr_matrix
        ), "sparse matrix should default to csr format"
        assert isinstance(
            op.sparse_matrix(format="csc"), scipy.sparse.csc_matrix
        ), "sparse matrix should be formatted as csc"
        assert isinstance(
            op.sparse_matrix(format="lil"), scipy.sparse.lil_matrix
        ), "sparse matrix should be formatted as lil"
        assert isinstance(
            op.sparse_matrix(format="coo"), scipy.sparse.coo_matrix
        ), "sparse matrix should be formatted as coo"
    else:
        failure_comment = "If has_sparse_matrix is False, the matrix method must raise a ``SparseMatrixUndefinedError``."
        _assert_error_raised(
            op.sparse_matrix,
            SparseMatrixUndefinedError,
            failure_comment=failure_comment,
        )()


def _check_eigendecomposition(op):
    """Checks involving diagonalizing gates and eigenvalues."""
    if op.has_diagonalizing_gates:
        dg = op.diagonalizing_gates()
        try:
            _, args, kwargs = _get_decomp_args(op)
            compute_dg = type(op).compute_diagonalizing_gates(*args, **kwargs)
        except (DiagGatesUndefinedError, TypeError):
            # sometimes diagonalizing gates is defined but not compute_diagonalizing_gates
            # compute_diagonalizing_gates might also have a different call signature
            compute_dg = dg

        for op1, op2 in zip(dg, compute_dg, strict=True):
            assert op1 == op2, "diagonalizing_gates and compute_diagonalizing_gates must match"
    else:
        failure_comment = "If has_diagonalizing_gates is False, diagonalizing_gates must raise a DiagGatesUndefinedError"
        _assert_error_raised(op.diagonalizing_gates, DiagGatesUndefinedError, failure_comment)()

    try:
        eg = op.eigvals()
    except EigvalsUndefinedError:
        eg = None

    has_eigvals = True
    try:
        _, args, kwargs = _get_decomp_args(op)
        if isinstance(op, Operator1):
            kwargs = {k: v for k, v in kwargs.items() if k != "wires"}
        compute_eg = type(op).compute_eigvals(*args, **kwargs)
    except EigvalsUndefinedError:
        compute_eg = eg
        has_eigvals = False

    if has_eigvals:
        assert math.allclose(eg, compute_eg), "eigvals and compute_eigvals must match"

    if (eg is not None or compute_eg is not None) and op.has_diagonalizing_gates:
        eg = eg if eg is not None else compute_eg
        dg = ops.prod(*dg[::-1]) if len(dg) > 0 else ops.Identity(op.wires)
        eg = ops.QubitUnitary(np.diag(eg), wires=op.wires)
        decomp = ops.prod(ops.adjoint(dg), eg, dg)
        # The decomposition's wires may be ordered differently than the operator's wires,
        # so both matrices must be computed in the same wire order.
        decomp_mat = op_funcs.matrix(decomp, wire_order=op.wires)
        original_mat = op_funcs.matrix(op, wire_order=op.wires)
        failure_comment = f"eigenvalues and diagonalizing gates must be able to reproduce the original operator. Got \n{decomp_mat}\n\n{original_mat}"
        assert math.allclose(decomp_mat, original_mat), failure_comment


def _check_generator(op):
    """Checks that if an operator's has_generator property is True, it has a generator."""

    if op.has_generator:
        gen = op.generator()
        assert isinstance(gen, Operator)
        new_op = ops.exp(gen, 1j * op.data[0])
        assert math.allclose(
            op_funcs.matrix(op, wire_order=op.wires), op_funcs.matrix(new_op, wire_order=op.wires)
        )
    else:
        failure_comment = (
            "If has_generator is False, the matrix method must raise a ``GeneratorUndefinedError``."
        )
        _assert_error_raised(
            op.generator, GeneratorUndefinedError, failure_comment=failure_comment
        )()


def _check_copy(op, skip_deepcopy):
    """Check that copies and deep copies give identical objects."""
    copied_op = copy.copy(op)
    assert op_funcs.equal(copied_op, op), "copied op must be equal with equal"
    assert copied_op == op, "copied op must be equivalent to original operation"
    assert copied_op is not op, "copied op must be a separate instance from original operation"
    if not skip_deepcopy:
        try:
            op_funcs.assert_equal(copy.deepcopy(op), op)
        except AssertionError as e:
            raise AssertionError("deep copied op must also be equal") from e


# pylint: disable=import-outside-toplevel, protected-access
def _check_pytree(op):
    """Check that the operator is a pytree."""

    data, metadata = op._flatten()
    try:
        assert hash(metadata), "metadata must be hashable"
    except Exception as e:
        raise AssertionError(
            f"metadata output from _flatten must be hashable. Got metadata {metadata}"
        ) from e

    try:
        new_op = type(op)._unflatten(data, metadata)
    except Exception as e:
        message = (
            f"{type(op).__name__}._unflatten must be able to reproduce the original operation from "
            f"{data} and {metadata}. You may need to override either the _unflatten or _flatten method. "
            f"\nFor local testing, try type(op)._unflatten(*op._flatten())"
        )
        raise AssertionError(message) from e

    try:
        op_funcs.assert_equal(op, new_op)
    except AssertionError as e:
        raise AssertionError(
            "metadata and data must be able to reproduce the original operation"
        ) from e

    leaves, struct = jax.tree_util.tree_flatten(op)
    unflattened = jax.tree_util.tree_unflatten(struct, leaves)
    assert unflattened == op, f"op must be a valid pytree. Got {unflattened} instead of {op}."

    leaves, struct = pytrees.flatten(op)
    unflattened = pytrees.unflatten(leaves, struct)
    assert unflattened == op, f"op must be a valid pytree. Got {unflattened} instead of {op}."

    if isinstance(op, Operator1):
        # Nested operators can contribute parameters or structural leaves (such as Operator2
        # wires) that are intentionally absent from the outer legacy ``data`` view. Stop at
        # nested PennyLane objects and expand only their numerical data. The outer ``data`` may
        # intentionally omit some nested parameters (for example, ``Evolution`` excludes its
        # generator's parameters), but every exposed parameter must still occur in pytree order.
        def nested_pl_object(obj):
            return obj is not op and isinstance(obj, (Operator, MeasurementProcess))

        legacy_leaves, _ = pytrees.flatten(op, is_leaf=nested_pl_object)
        ordered_data = []
        for leaf in legacy_leaves:
            if isinstance(leaf, Operator1):
                ordered_data.extend(leaf.data)
            elif isinstance(leaf, Operator2):
                ordered_data.extend(leaf.dynamic_args.values())
            elif not isinstance(leaf, MeasurementProcess):
                ordered_data.append(leaf)

        ordered_data = iter(ordered_data)
        for data_item in op.data:
            if not any(math.allclose(data_item, leaf) for leaf in ordered_data):
                raise AssertionError(
                    "data must be the terminal leaves of the pytree in the same order. "
                    f"Could not find {data_item} in the remaining leaves."
                )


def _check_capture(op):

    if not all(isinstance(w, int) for w in op.wires):
        return

    data, struct = jax.tree_util.tree_flatten(op)

    def test_fn(*args):
        op = jax.tree_util.tree_unflatten(struct, args)
        if isinstance(op, Operator2):
            op._bind_primitive()
            return op.tracer
        return op

    jaxpr = jax.make_jaxpr(test_fn)(*data)
    new_op = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *data)[0]
    op_funcs.assert_equal(op, new_op)

    leaves = jax.tree_util.tree_leaves(jaxpr.eqns[-1].params)
    error_msg = "capture params cannot contain tracers"
    assert not any(math.is_abstract(l) for l in leaves), error_msg


def _check_pickle(op):
    """Check that an operation can be dumped and reloaded with pickle."""
    pickled = pickle.dumps(op)
    unpickled = pickle.loads(pickled)
    assert unpickled == op, "operation must be able to be pickled and unpickled"


def _check_bind_new_parameters(op):
    """Check that bind new parameters can create a new op with different data."""
    new_data = [d * 0.0 for d in op.data]
    new_data_op = op_funcs.bind_new_parameters(op, new_data)
    failure_comment = "bind_new_parameters must be able to update the operator with new data."
    for d1, d2 in zip(new_data_op.data, new_data, strict=True):
        assert math.allclose(d1, d2), failure_comment


def _check_bind_new_parameters_op2(op):
    """Check that bind new parameters can create a new op with different bound arguments."""
    dyn_args = op.base.dynamic_args if isinstance(op, SymbolicOp2) else op.dynamic_args
    new_dyn_args = {k: math.cast_like(v * 0.0, v) for k, v in dyn_args.items()}
    new_data_op = op_funcs.bind_new_parameters(op, tuple(new_dyn_args.values()))
    failure_comment = "bind_new_parameters must be able to update the operator2 with new arguments."
    for name, val in new_dyn_args.items():
        op_to_check = new_data_op.base if isinstance(new_data_op, SymbolicOp2) else new_data_op
        assert math.allclose(op_to_check.arguments[name], val), failure_comment


def _check_bind_new_parameters_symbolicop2(op):
    """Check that bind new parameters can create a new op with different bound arguments."""
    new_dyn_args = {
        k: math.cast_like(v * 0.0, v)
        for k, v in op.base.arguments.items()
        if k in op.base.dynamic_argnames
    }
    new_data_op = op_funcs.bind_new_parameters(op, new_dyn_args.values())
    failure_comment = (
        "bind_new_parameters must be able to update the symbolicop2 with new arguments."
    )
    for name, val in new_dyn_args.items():
        assert math.allclose(new_data_op.base.arguments[name], val), failure_comment


def _check_differentiation(op):
    """Checks that the operator can be executed and differentiated correctly."""

    if op.num_params == 0:
        return

    data, struct = pytrees.flatten(op)

    def circuit(*args):
        apply(pytrees.unflatten(args, struct))
        return probs(wires=op.wires)

    qnode_ref = QNode(circuit, device("default.qubit"), diff_method="backprop")
    qnode_ps = QNode(circuit, device("default.qubit"), diff_method="parameter-shift")

    params = [x if isinstance(x, int) else pnp.array(x) for x in data]

    ps = jacobian(qnode_ps)(*params)
    expected_bp = jacobian(qnode_ref)(*params)

    error_msg = (
        "Parameter-shift does not produce the same Jacobian as with backpropagation. "
        "This might be a bug, or it might be expected due to the mathematical nature "
        "of backpropagation, in which case, this test can be skipped for this operator."
    )

    if isinstance(ps, tuple):
        for actual, expected in zip(ps, expected_bp, strict=True):
            assert math.allclose(actual, expected), error_msg
    else:
        assert math.allclose(ps, expected_bp), error_msg


def _check_wires(op, skip_wire_mapping):
    """Check that wires are a ``Wires`` class and can be mapped."""
    assert isinstance(op.wires, Wires), "wires must be a wires instance"
    if skip_wire_mapping:
        return
    wire_map = {w: i + 10 * len(op.wires) for i, w in enumerate(op.wires)}
    mapped_op = op.map_wires(wire_map)
    new_wires = Wires([i + 10 * len(op.wires) for i in range(len(op.wires))])
    assert mapped_op.wires == new_wires, "wires must be mappable with map_wires"


# pylint: disable=too-many-arguments
def _assert_valid_operator2(
    op: Operator2,
    skip_deepcopy=False,
    skip_differentiation=False,
    skip_new_decomp=False,
    skip_decomp_matrix_check=False,
    skip_pickle=False,
    skip_wire_mapping=False,
    skip_bind_new_parameters=False,
    skip_eigvals=False,
) -> None:
    """
    Runs basic validation checks on an :class:`~.core.Operator2` to make sure it has been correctly defined.

    Args:
        op: The operator to validate.
        skip_deepcopy: If ``True``, the deepcopy test will be skipped.
        skip_differentiation: If ``True``, the differentiation test will be skipped.
        skip_new_decomp: If ``True``, the new decomposition test will be skipped.
        skip_decomp_matrix_check: If ``True``, the decomposition matrix check will be skipped.
        skip_pickle: If ``True``, the pickle test will be skipped.
        skip_wire_mapping: If ``True``, the wire mapping test will be skipped.
        skip_bind_new_parameters: If ``True``, the ``bind_new_parameters`` test will be skipped.
        skip_eigvals: If ``True``, the eigendecomposition tests will be skipped.
    """

    # Note: these attributes are in the spec but not the implementation yet.
    # assert isinstance(op.data, tuple), "op.data must be a tuple"
    # assert isinstance(op.num_wires, int), "op.num_wires must be a int"

    assert isinstance(op.wires, Wires), "op.wires must be a Wires instance"
    assert isinstance(op.ndim_params, tuple), "ndim_params must be a tuple"
    assert isinstance(op.compilable_argnames, tuple), "compilable_argnames must be a tuple"
    assert isinstance(op.hybrid_argnames, tuple), "hybrid_argnames must be a tuple"
    assert isinstance(op.wire_argnames, tuple), "wire_argnames must be a tuple"
    assert isinstance(op.static_argnames, tuple), "static_argnames must be a tuple"
    assert isinstance(op.dynamic_argnames, tuple), "dynamic_argnames must be a tuple"
    op_funcs.assert_equal(type(op)(**op.arguments), op)

    # check abstractify
    abstractified_op = abstractify(op)
    leaves, _ = pytrees.flatten(abstractified_op, lambda l: isinstance(l, Wires))
    for l in leaves:
        if not isinstance(l, (AbstractArray, AbstractWires, CompressedResourceOp)):
            raise AssertionError(
                f"Op not properly abstractified. {abstractified_op} had non-abstract leaf {l}."
            )

    # Some operators (e.g. composites and ``Select``) hold their data inside operator-valued
    # arguments rather than dynamic arguments, so their ``data`` does not correspond to
    # ``dynamic_argnames`` and this check does not apply.
    if not isinstance(op, (Adjoint2, CompositeOp2, ControlledOp2, Pow2, Select, QSVT)):

        error_msg = "ndim_params must have the same length as dynamic_argnames"
        assert len(op.ndim_params) == len(op.dynamic_argnames), error_msg

        for (name, val), dim in zip(op.dynamic_args.items(), op.ndim_params, strict=True):
            # make sure that the bound args are not outside the allowed dimensions
            error_msg = f"shape of {name} is not equal to dimension in ndim_params"
            assert len(math.shape(val)) == dim, error_msg

    for (name, val), dim in zip(op.wire_args.items(), op.wire_sizes, strict=True):
        # make sure wires have the right sizes
        if op.wire_sizes:
            error_msg = f"Wires argument {name} has an invalid dimension."
            assert dim is None or len(val) == dim, error_msg

    for name, val in op.hybrid_args.items():
        leaves, _ = pytrees.flatten(val, is_leaf=lambda l: isinstance(l, Operator))
        for leaf in leaves:
            if isinstance(leaf, Operator):
                assert_valid(
                    leaf,
                    skip_deepcopy=skip_deepcopy,
                    skip_differentiation=skip_differentiation,
                    skip_new_decomp=skip_new_decomp,
                    skip_decomp_matrix_check=skip_decomp_matrix_check,
                    skip_pickle=skip_pickle,
                    skip_wire_mapping=skip_wire_mapping,
                    skip_bind_new_parameters=skip_bind_new_parameters,
                    skip_eigvals=skip_eigvals,
                )

    if not skip_bind_new_parameters:
        if isinstance(op, SymbolicOp2):
            _check_bind_new_parameters_symbolicop2(op)
        else:
            _check_bind_new_parameters_op2(op)


# pylint: disable=too-many-arguments
def assert_valid(
    op: Operator,
    *,
    skip_deepcopy=False,
    skip_differentiation=False,
    skip_new_decomp=False,
    skip_decomp_matrix_check=False,
    skip_pickle=False,
    skip_wire_mapping=False,
    skip_bind_new_parameters=False,
    skip_eigvals=False,
) -> None:
    """Runs basic validation checks on an :class:`~.core.Operator` or :class:`~.core.Operator2` to make
    sure it has been correctly defined.

    Args:
        op (.Operator): an operator instance to validate

    Keyword Args:
        skip_deepcopy=False: If ``True``, deepcopy tests are not run.
        skip_differentiation=False: If ``True``, differentiation tests are not run.
        skip_new_decomp: If ``True``, the operator will not be tested for its decomposition
            defined using the new system.
        skip_decomp_matrix_check: If ``True``, the decomposition rule check will only
            verify that the produced operators match the resource function, and does not
            test that the matrix of the decomposition matches the operator itself.
        skip_pickle=False : If ``True``, pickling tests are not run. Set to ``True`` when
            testing a locally defined operator, as pickle cannot handle local objects
        skip_wire_mapping : If ``True``, the operator will not be tested for wire mapping.
        skip_bind_new_parameters: If ``True``, the ``bind_new_parameters`` tests will be skipped.
        skip_eigvals: If ``True``, the eigendecomposition tests will be skipped.

    **Examples:**

    .. code-block:: python

        class MyOp(Operator):

            def __init__(self, data, wires):
                self._data = data
                super().__init__(wires=wires)

        op = MyOp(pnp.array(0.5), wires=0)

    >>> assert_valid(op)
    Traceback (most recent call last):
        ...
    AssertionError: MyOp._unflatten must be able to reproduce the original operation from () and (Wires([0]), ()). You may need to override either the _unflatten or _flatten method.
    For local testing, try type(op)._unflatten(*op._flatten())

    .. code-block:: python

        class MyOp(Operator):

            def __init__(self, wires):
                self.hyperparameters["unhashable_list"] = []
                super().__init__(wires=wires)

        op = MyOp(wires = 0)

    >>> assert_valid(op)
    Traceback (most recent call last):
        ...
    AssertionError: metadata output from _flatten must be hashable. Got metadata (Wires([0]), (('unhashable_list', []),))

    """
    if isinstance(op, Operator2):
        _assert_valid_operator2(
            op,
            skip_deepcopy,
            skip_differentiation,
            skip_new_decomp,
            skip_decomp_matrix_check,
            skip_pickle,
            skip_wire_mapping,
            skip_bind_new_parameters,
            skip_eigvals,
        )
    else:
        assert isinstance(op.data, tuple), "op.data must be a tuple"
        assert isinstance(op.parameters, list), "op.parameters must be a list"

        for d, p in zip(op.data, op.parameters, strict=True):
            assert isinstance(d, TensorLike), "each data element must be tensorlike"
            assert math.allclose(d, p), "data and parameters must match."

        if not skip_bind_new_parameters:
            _check_bind_new_parameters(op)

    _check_pytree(op)
    _check_copy(op, skip_deepcopy=skip_deepcopy)
    if len(op.wires) <= 26:
        _check_wires(op, skip_wire_mapping=skip_wire_mapping)
    if not skip_pickle:
        _check_pickle(op)
    if not capture.enabled():
        _check_decomposition_legacy(op, skip_wire_mapping=skip_wire_mapping)
    if not skip_new_decomp:
        _check_decomposition_new(op, skip_decomp_matrix_check=skip_decomp_matrix_check)
    _check_matrix(op)
    _check_sparse_matrix(op)
    if not skip_eigvals:
        _check_eigendecomposition(op)
    _check_generator(op)
    if not skip_differentiation and not capture.enabled():
        _check_differentiation(op)
    if capture.enabled():
        _check_capture(op)
