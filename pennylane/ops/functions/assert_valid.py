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
This module contains the qml.ops.functions.check_validity function for determining whether or not an
Operator class is correctly defined.
"""

import copy
import itertools
import pickle
from collections import defaultdict
from string import ascii_lowercase

import numpy as np
import scipy.sparse

import pennylane as qml
from pennylane.decomposition import DecompositionRule
from pennylane.exceptions import EigvalsUndefinedError

from .equal import assert_equal


def _assert_error_raised(func, error, failure_comment):
    def inner_func(*args, **kwargs):
        error_raised = False
        try:
            func(*args, **kwargs)
        except error:
            error_raised = True
        assert error_raised, failure_comment

    return inner_func


def _check_decomposition(op, skip_wire_mapping):
    """Checks involving the decomposition."""
    if op.has_decomposition:
        decomp = op.decomposition()
        try:
            compute_decomp = type(op).compute_decomposition(
                *op.data, wires=op.wires, **op.hyperparameters
            )
        except (qml.operation.DecompositionUndefinedError, TypeError):
            # sometimes decomposition is defined but not compute_decomposition
            # Also  sometimes compute_decomposition can have a different signature
            compute_decomp = decomp
        with qml.queuing.AnnotatedQueue() as queued_decomp:
            op.decomposition()
        processed_queue = qml.tape.QuantumTape.from_queue(queued_decomp)

        assert isinstance(decomp, list), "decomposition must be a list"
        assert isinstance(compute_decomp, list), "decomposition must be a list"
        assert op not in decomp, "an operator should not be included in its own decomposition"

        for o1, o2, o3 in zip(decomp, compute_decomp, processed_queue):
            assert o1 == o2, "decomposition must match compute_decomposition"
            assert o1 == o3, "decomposition must match queued operations"
            assert isinstance(o1, qml.operation.Operator), "decomposition must contain operators"

        if skip_wire_mapping:
            return
        # Check that mapping wires transitions to the decomposition
        wire_map = {w: ascii_lowercase[i] for i, w in enumerate(op.wires)}
        mapped_op = op.map_wires(wire_map)
        # calling `map_wires` on a Controlled operator generates a new `op` from the controls and
        # base, so may return a different class of operator. We only compare decomps of `op` and
        # `mapped_op` if `mapped_op` **has** a decomposition.
        # see MultiControlledX([0, 1]) and CNOT([0, 1]) as an example
        if mapped_op.has_decomposition:
            mapped_decomp = mapped_op.decomposition()
            orig_decomp = op.decomposition()
            for mapped_op, orig_op in zip(mapped_decomp, orig_decomp):
                assert (
                    mapped_op.wires == qml.map_wires(orig_op, wire_map).wires
                ), "Operators in decomposition of wire-mapped operator must have mapped wires."
    else:
        failure_comment = "If has_decomposition is False, then decomposition must raise a ``DecompositionUndefinedError``."
        _assert_error_raised(
            op.decomposition,
            qml.operation.DecompositionUndefinedError,
            failure_comment=failure_comment,
        )()
        _assert_error_raised(
            op.compute_decomposition,
            qml.operation.DecompositionUndefinedError,
            failure_comment=failure_comment,
        )(*op.data, wires=op.wires, **op.hyperparameters)


def _check_decomposition_new(op):
    """Checks involving the new system of decompositions."""
    op_type = type(op)
    if op_type.resource_params is qml.operation.Operator.resource_params:
        assert not qml.decomposition.has_decomp(
            op_type
        ), "resource_params must be defined for operators with decompositions"
        return

    assert set(op.resource_params.keys()) == set(
        op_type.resource_keys
    ), "resource_params must have the same keys as specified by resource_keys"

    for rule in qml.list_decomps(op_type):
        _test_decomposition_rule(op, rule)

    for rule in qml.list_decomps(f"Adjoint({op_type.__name__})"):
        adj_op = qml.ops.Adjoint(op)
        _test_decomposition_rule(adj_op, rule)

    for rule in qml.list_decomps(f"Pow({op_type.__name__})"):
        for z in [2, 3, 4, 8, 9]:
            pow_op = qml.ops.Pow(op, z)
            _test_decomposition_rule(pow_op, rule)

    for rule in qml.list_decomps(f"C({op_type.__name__})"):
        for n_ctrl_wires, c_value, n_workers in itertools.product([1, 2, 3], [0, 1], [0, 1, 2]):
            ctrl_op = qml.ops.Controlled(
                op,
                control_wires=[i + len(op.wires) for i in range(n_ctrl_wires)],
                control_values=[c_value] * n_ctrl_wires,
                work_wires=[i + len(op.wires) + n_ctrl_wires for i in range(n_workers)],
            )
            _test_decomposition_rule(ctrl_op, rule)


def _test_decomposition_rule(op, rule: DecompositionRule):
    """Tests that a decomposition rule is consistent with the operator."""

    if not rule.is_applicable(**op.resource_params):
        return

    # Test that the resource function is correct
    resources = rule.compute_resources(**op.resource_params)
    gate_counts = resources.gate_counts

    with qml.queuing.AnnotatedQueue() as q:
        rule(*op.data, wires=op.wires, **op.hyperparameters)
    tape = qml.tape.QuantumScript.from_queue(q)

    total_work_wires = rule.get_work_wire_spec(**op.resource_params).total
    if total_work_wires:
        [tape], _ = qml.transforms.resolve_dynamic_wires(
            [tape], zeroed=range(len(tape.wires), len(tape.wires) + total_work_wires)
        )

    actual_gate_counts = defaultdict(int)
    for _op in tape.operations:
        resource_rep = qml.resource_rep(type(_op), **_op.resource_params)
        actual_gate_counts[resource_rep] += 1

    if rule.exact_resources:
        non_zero_gate_counts = {k: v for k, v in gate_counts.items() if v > 0}
        assert non_zero_gate_counts == actual_gate_counts, (
            f"\nGate counts expected from resource function:\n{non_zero_gate_counts}"
            f"\nActual gate counts:\n{actual_gate_counts}"
        )
    else:
        # If the resource estimate is not expected to match exactly to the actual
        # decomposition, at least make sure that all gates are accounted for.
        assert all(op in gate_counts for op in actual_gate_counts)

    # Tests that the decomposition produces the same matrix
    if op.has_matrix:
        # Add projector to the additional wires (work wires) on the tape
        work_wires = tape.wires - op.wires
        all_wires = op.wires + work_wires
        if work_wires:
            op = op @ qml.Projector([0] * len(work_wires), wires=work_wires)
            tape.operations.insert(0, qml.Projector([0] * len(work_wires), wires=work_wires))

        op_matrix = op.matrix(wire_order=all_wires)
        decomp_matrix = qml.matrix(tape, wire_order=all_wires)
        assert qml.math.allclose(
            op_matrix, decomp_matrix
        ), "decomposition must produce the same matrix as the operator."


def _check_matrix(op):
    """Check that if the operation says it has a matrix, it does. Otherwise a ``MatrixUndefinedError`` should be raised."""
    if op.has_matrix:
        mat = op.matrix()
        assert isinstance(mat, qml.typing.TensorLike), "matrix must be a TensorLike"
        l = 2 ** len(op.wires)
        failure_comment = f"matrix must be two dimensional with shape ({l}, {l})"
        assert qml.math.shape(mat) == (l, l), failure_comment
    else:
        failure_comment = (
            "If has_matrix is False, the matrix method must raise a ``MatrixUndefinedError``."
        )
        _assert_error_raised(
            op.matrix, qml.operation.MatrixUndefinedError, failure_comment=failure_comment
        )()


def _check_sparse_matrix(op):
    """Check that if the operation says it has a sparse matrix, it does. Otherwise a ``SparseMatrixUndefinedError`` should be raised."""
    if op.has_sparse_matrix:
        mat = op.sparse_matrix()
        assert isinstance(mat, scipy.sparse.csr_matrix), "matrix must be a TensorLike"
        l = 2 ** len(op.wires)
        failure_comment = f"matrix must be two dimensional with shape ({l}, {l})"
        assert qml.math.shape(mat) == (l, l), failure_comment

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
            qml.operation.SparseMatrixUndefinedError,
            failure_comment=failure_comment,
        )()


def _check_matrix_matches_decomp(op):
    """Check that if both the matrix and decomposition are defined, they match."""
    if op.has_matrix and op.has_decomposition:
        mat = op.matrix()
        decomp_mat = qml.matrix(qml.tape.QuantumScript(op.decomposition()), wire_order=op.wires)
        failure_comment = (
            f"matrix and matrix from decomposition must match. Got \n{mat}\n\n {decomp_mat}"
        )
        assert qml.math.allclose(mat, decomp_mat), failure_comment


def _check_eigendecomposition(op):
    """Checks involving diagonalizing gates and eigenvalues."""
    if op.has_diagonalizing_gates:
        dg = op.diagonalizing_gates()
        try:
            compute_dg = type(op).compute_diagonalizing_gates(
                *op.data, wires=op.wires, **op.hyperparameters
            )
        except (qml.operation.DiagGatesUndefinedError, TypeError):
            # sometimes diagonalizing gates is defined but not compute_diagonalizing_gates
            # compute_diagonalizing_gates might also have a different call signature
            compute_dg = dg

        for op1, op2 in zip(dg, compute_dg):
            assert op1 == op2, "diagonalizing_gates and compute_diagonalizing_gates must match"
    else:
        failure_comment = "If has_diagonalizing_gates is False, diagonalizing_gates must raise a DiagGatesUndefinedError"
        _assert_error_raised(
            op.diagonalizing_gates, qml.operation.DiagGatesUndefinedError, failure_comment
        )()

    try:
        eg = op.eigvals()
    except EigvalsUndefinedError:
        eg = None

    has_eigvals = True
    try:
        compute_eg = type(op).compute_eigvals(*op.data, **op.hyperparameters)
    except EigvalsUndefinedError:
        compute_eg = eg
        has_eigvals = False

    if has_eigvals:
        assert qml.math.allclose(eg, compute_eg), "eigvals and compute_eigvals must match"

    if has_eigvals and op.has_diagonalizing_gates:
        dg = qml.prod(*dg[::-1]) if len(dg) > 0 else qml.Identity(op.wires)
        eg = qml.QubitUnitary(np.diag(eg), wires=op.wires)
        decomp = qml.prod(qml.adjoint(dg), eg, dg)
        decomp_mat = qml.matrix(decomp)
        original_mat = qml.matrix(op)
        failure_comment = f"eigenvalues and diagonalizing gates must be able to reproduce the original operator. Got \n{decomp_mat}\n\n{original_mat}"
        assert qml.math.allclose(decomp_mat, original_mat), failure_comment


def _check_generator(op):
    """Checks that if an operator's has_generator property is True, it has a generator."""

    if op.has_generator:
        gen = op.generator()
        assert isinstance(gen, qml.operation.Operator)
        new_op = qml.exp(gen, 1j * op.data[0])
        assert qml.math.allclose(
            qml.matrix(op, wire_order=op.wires), qml.matrix(new_op, wire_order=op.wires)
        )
    else:
        failure_comment = (
            "If has_generator is False, the matrix method must raise a ``GeneratorUndefinedError``."
        )
        _assert_error_raised(
            op.generator, qml.operation.GeneratorUndefinedError, failure_comment=failure_comment
        )()


def _check_copy(op, skip_deepcopy):
    """Check that copies and deep copies give identical objects."""
    copied_op = copy.copy(op)
    assert qml.equal(copied_op, op), "copied op must be equal with qml.equal"
    assert copied_op == op, "copied op must be equivalent to original operation"
    assert copied_op is not op, "copied op must be a separate instance from original operaiton"
    if not skip_deepcopy:
        assert qml.equal(copy.deepcopy(op), op), "deep copied op must also be equal"


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
    assert op == new_op, "metadata and data must be able to reproduce the original operation"

    try:
        import jax
    except ImportError:
        return
    leaves, struct = jax.tree_util.tree_flatten(op)
    unflattened_op = jax.tree_util.tree_unflatten(struct, leaves)
    assert unflattened_op == op, f"op must be a valid pytree. Got {unflattened_op} instead of {op}."

    for d1, d2 in zip(op.data, leaves):
        assert qml.math.allclose(
            d1, d2
        ), f"data must be the terminal leaves of the pytree. Got {d1}, {d2}"


def _check_capture(op):
    try:
        import jax
    except ImportError:
        return

    if not all(isinstance(w, int) for w in op.wires):
        return

    qml.capture.enable()
    try:
        data, struct = jax.tree_util.tree_flatten(op)

        def test_fn(*args):
            return jax.tree_util.tree_unflatten(struct, args)

        jaxpr = jax.make_jaxpr(test_fn)(*data)
        new_op = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *data)[0]
        assert_equal(op, new_op)

        leaves = jax.tree_util.tree_leaves(jaxpr.eqns[-1].params)
        assert not any(
            qml.math.is_abstract(l) for l in leaves
        ), "capture params cannot contain tracers"
    except Exception as e:
        raise ValueError(
            "The capture of the operation into jaxpr failed somehow."
            " This capture mechanism is currently experimental and not a core"
            " requirement, but will be necessary in the future."
            " Please see the capture module documentation for more information."
        ) from e
    finally:
        qml.capture.disable()


def _check_pickle(op):
    """Check that an operation can be dumped and reloaded with pickle."""
    pickled = pickle.dumps(op)
    unpickled = pickle.loads(pickled)
    assert unpickled == op, "operation must be able to be pickled and unpickled"


def _check_bind_new_parameters(op):
    """Check that bind new parameters can create a new op with different data."""
    new_data = [d * 0.0 for d in op.data]
    new_data_op = qml.ops.functions.bind_new_parameters(op, new_data)
    failure_comment = "bind_new_parameters must be able to update the operator with new data."
    for d1, d2 in zip(new_data_op.data, new_data):
        assert qml.math.allclose(d1, d2), failure_comment


def _check_differentiation(op):
    """Checks that the operator can be executed and differentiated correctly."""

    if op.num_params == 0:
        return

    data, struct = qml.pytrees.flatten(op)

    def circuit(*args):
        qml.apply(qml.pytrees.unflatten(args, struct))
        return qml.probs(wires=op.wires)

    qnode_ref = qml.QNode(circuit, qml.device("default.qubit"), diff_method="backprop")
    qnode_ps = qml.QNode(circuit, qml.device("default.qubit"), diff_method="parameter-shift")

    params = [x if isinstance(x, int) else qml.numpy.array(x) for x in data]

    ps = qml.jacobian(qnode_ps)(*params)
    expected_bp = qml.jacobian(qnode_ref)(*params)

    error_msg = (
        "Parameter-shift does not produce the same Jacobian as with backpropagation. "
        "This might be a bug, or it might be expected due to the mathematical nature "
        "of backpropagation, in which case, this test can be skipped for this operator."
    )

    if isinstance(ps, tuple):
        for actual, expected in zip(ps, expected_bp):
            assert qml.math.allclose(actual, expected), error_msg
    else:
        assert qml.math.allclose(ps, expected_bp), error_msg


def _check_wires(op, skip_wire_mapping):
    """Check that wires are a ``Wires`` class and can be mapped."""
    assert isinstance(op.wires, qml.wires.Wires), "wires must be a wires instance"

    if skip_wire_mapping:
        return
    wire_map = {w: ascii_lowercase[i] for i, w in enumerate(op.wires)}
    mapped_op = op.map_wires(wire_map)
    new_wires = qml.wires.Wires(list(ascii_lowercase[: len(op.wires)]))
    assert mapped_op.wires == new_wires, "wires must be mappable with map_wires"


# pylint: disable=too-many-arguments
def assert_valid(
    op: qml.operation.Operator,
    *,
    skip_deepcopy=False,
    skip_differentiation=False,
    skip_new_decomp=False,
    skip_pickle=False,
    skip_wire_mapping=False,
    skip_capture=False,
) -> None:
    """Runs basic validation checks on an :class:`~.operation.Operator` to make
    sure it has been correctly defined.

    Args:
        op (.Operator): an operator instance to validate

    Keyword Args:
        skip_deepcopy=False: If ``True``, deepcopy tests are not run.
        skip_differentiation=False: If ``True``, differentiation tests are not run.
        skip_new_decomp: If ``True``, the operator will not be tested for its decomposition
            defined using the new system.
        skip_pickle=False : If ``True``, pickling tests are not run. Set to ``True`` when
            testing a locally defined operator, as pickle cannot handle local objects
        skip_wire_mapping : If ``True``, the operator will not be tested for wire mapping.

    **Examples:**

    .. code-block:: python

        class MyOp(qml.operation.Operator):

            def __init__(self, data, wires):
                self.data = data
                super().__init__(wires=wires)

        op = MyOp(qml.numpy.array(0.5), wires=0)

    >>> assert_valid(op)
    Traceback (most recent call last):
        ...
    AssertionError: MyOp._unflatten must be able to reproduce the original operation from () and (Wires([0]), ()). You may need to override either the _unflatten or _flatten method.
    For local testing, try type(op)._unflatten(*op._flatten())

    .. code-block:: python

        class MyOp(qml.operation.Operator):

            def __init__(self, wires):
                self.hyperparameters["unhashable_list"] = []
                super().__init__(wires=wires)

        op = MyOp(wires = 0)

    >>> assert_valid(op)
    Traceback (most recent call last):
        ...
    AssertionError: metadata output from _flatten must be hashable. Got metadata (Wires([0]), (('unhashable_list', []),))

    """

    assert isinstance(op.data, tuple), "op.data must be a tuple"
    assert isinstance(op.parameters, list), "op.parameters must be a list"
    for d, p in zip(op.data, op.parameters):
        assert isinstance(d, qml.typing.TensorLike), "each data element must be tensorlike"
        assert qml.math.allclose(d, p), "data and parameters must match."

    if len(op.wires) <= 26:
        _check_wires(op, skip_wire_mapping=skip_wire_mapping)
    _check_copy(op, skip_deepcopy=skip_deepcopy)
    _check_pytree(op)
    if not skip_pickle:
        _check_pickle(op)
    _check_bind_new_parameters(op)
    _check_decomposition(op, skip_wire_mapping=skip_wire_mapping)
    if not skip_new_decomp:
        _check_decomposition_new(op)
    _check_matrix(op)
    _check_matrix_matches_decomp(op)
    _check_sparse_matrix(op)
    _check_eigendecomposition(op)
    _check_generator(op)
    if not skip_differentiation:
        _check_differentiation(op)
    if not skip_capture:
        _check_capture(op)
