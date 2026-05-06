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
This module contains the qp.ops.functions.check_validity function for determining whether or not an
Operator class is correctly defined.
"""

import copy
import itertools
import pickle
from collections import defaultdict
from string import ascii_lowercase

import numpy as np
import scipy.sparse

import pennylane as qp
from pennylane.decomposition import DecompositionRule
from pennylane.decomposition.reconstruct import get_decomp_kwargs, has_reconstructor, reconstruct
from pennylane.decomposition.resources import adjoint_resource_rep, pow_resource_rep, resource_rep
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
        except (qp.exceptions.DecompositionUndefinedError, TypeError):
            # sometimes decomposition is defined but not compute_decomposition
            # Also  sometimes compute_decomposition can have a different signature
            compute_decomp = decomp
        with qp.queuing.AnnotatedQueue() as queued_decomp:
            op.decomposition()
        processed_queue = qp.tape.QuantumTape.from_queue(queued_decomp)

        assert isinstance(decomp, list), "decomposition must be a list"
        assert isinstance(compute_decomp, list), "decomposition must be a list"
        assert op not in decomp, "an operator should not be included in its own decomposition"
        compute_decomp_msg = "decomposition must match compute_decomposition"
        queue_msg = "decomposition must match queued operations"
        assert len(decomp) == len(compute_decomp), compute_decomp_msg
        assert len(decomp) == len(processed_queue), queue_msg

        for o1, o2, o3 in zip(decomp, compute_decomp, processed_queue, strict=True):
            if isinstance(o1, qp.ops.MidMeasure):
                for other_op, msg in [(o2, compute_decomp_msg), (o3, queue_msg)]:
                    assert isinstance(other_op, qp.ops.MidMeasure), msg
                    assert o1.wires == other_op.wires, msg
                    assert o1.reset == other_op.reset, msg
                    assert o1.postselect == other_op.postselect, msg
            else:
                assert isinstance(o1, qp.operation.Operator), "decomposition must contain operators"
                for other_op, msg in [(o2, compute_decomp_msg), (o3, queue_msg)]:
                    try:
                        assert_equal(o1, other_op)
                    except AssertionError as e:
                        raise AssertionError(msg) from e

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
            for mapped_op, orig_op in zip(mapped_decomp, orig_decomp, strict=True):
                assert (
                    mapped_op.wires
                    == qp.map_wires(orig_op, wire_map).wires  # pylint: disable=no-member
                ), "Operators in decomposition of wire-mapped operator must have mapped wires."
    else:
        failure_comment = "If has_decomposition is False, then decomposition must raise a ``DecompositionUndefinedError``."
        _assert_error_raised(
            op.decomposition,
            qp.operation.DecompositionUndefinedError,
            failure_comment=failure_comment,
        )()
        _assert_error_raised(
            op.compute_decomposition,
            qp.operation.DecompositionUndefinedError,
            failure_comment=failure_comment,
        )(*op.data, wires=op.wires, **op.hyperparameters)


def _check_decomposition_new(op, skip_decomp_matrix_check=False):
    """Checks involving the new system of decompositions."""
    op_type = type(op)
    if op_type.resource_params is qp.operation.Operator.resource_params:
        assert not qp.decomposition.has_decomp(
            op_type
        ), "resource_params must be defined for operators with decompositions"
        return

    assert set(op.resource_params.keys()) == set(
        op_type.resource_keys
    ), "resource_params must have the same keys as specified by resource_keys"

    for rule in qp.list_decomps(op_type):
        _test_decomposition_rule(op, rule, skip_decomp_matrix_check)

    for rule in qp.list_decomps(f"Adjoint({op_type.__name__})"):
        adj_op = qp.ops.Adjoint(op)
        _test_decomposition_rule(adj_op, rule, skip_decomp_matrix_check)

    for rule in qp.list_decomps(f"Pow({op_type.__name__})"):
        for z in [2, 3, 4, 8, 9]:
            pow_op = qp.ops.Pow(op, z)
            _test_decomposition_rule(pow_op, rule, skip_decomp_matrix_check)

    for rule in qp.list_decomps(f"C({op_type.__name__})"):
        for n_ctrl_wires, c_value, n_workers in itertools.product([1, 2, 3], [0, 1], [0, 1, 2]):
            ctrl_op = qp.ops.Controlled(
                op,
                control_wires=[i + len(op.wires) for i in range(n_ctrl_wires)],
                control_values=[c_value] * n_ctrl_wires,
                work_wires=[i + len(op.wires) + n_ctrl_wires for i in range(n_workers)],
            )
            _test_decomposition_rule(ctrl_op, rule, skip_decomp_matrix_check)


def _check_reconstructor(op):
    """Checks that the op can be reconstructed."""

    op_rep = resource_rep(op.__class__, **op.resource_params)
    if not has_reconstructor(op_rep.op_type, op_rep.params):
        return  # skip ops that are not meant to be compatible

    if isinstance(op, (qp.ops.MidMeasure, qp.ops.PauliMeasure)):
        return

    reconstructed_op = reconstruct(op.data, op.wires, op_rep.op_type, op_rep.params)
    qp.assert_equal(reconstructed_op, op)

    adjoint_op = qp.adjoint(op)
    op_rep = adjoint_resource_rep(op.__class__, op.resource_params)
    assert has_reconstructor(op_rep.op_type, op_rep.params)

    reconstructed_op = reconstruct(adjoint_op.data, adjoint_op.wires, op_rep.op_type, op_rep.params)
    qp.assert_equal(reconstructed_op, adjoint_op)

    pow_op = qp.pow(op, z=2)
    op_rep = pow_resource_rep(op.__class__, op.resource_params, z=2)
    assert has_reconstructor(op_rep.op_type, op_rep.params)

    reconstructed_op = reconstruct(pow_op.data, pow_op.wires, op_rep.op_type, op_rep.params)
    qp.assert_equal(reconstructed_op, pow_op)


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


def _test_decomposition_rule(op, rule: DecompositionRule, skip_decomp_matrix_check: bool = False):
    """Tests that a decomposition rule is consistent with the operator."""

    if not rule.is_applicable(**op.resource_params):
        return

    # Test that the resource function is correct
    resources = rule.compute_resources(**op.resource_params)
    gate_counts = resources.gate_counts

    kwargs = get_decomp_kwargs(op)
    with qp.queuing.AnnotatedQueue() as q:
        rule(*op.data, wires=op.wires, **kwargs)
    tape = qp.tape.QuantumScript.from_queue(q)

    total_work_wires = rule.get_work_wire_spec(**op.resource_params).total
    if total_work_wires:
        [tape], _ = qp.transforms.resolve_dynamic_wires(
            [tape], zeroed=range(len(tape.wires), len(tape.wires) + total_work_wires)
        )

    actual_gate_counts = defaultdict(int)
    for _op in tape.operations:
        if isinstance(_op, qp.ops.Conditional):
            _op = _op.base
        op_rep = qp.resource_rep(type(_op), **_op.resource_params)
        actual_gate_counts[op_rep] += 1

    if rule.exact_resources and not (
        isinstance(op, qp.templates.SubroutineOp) and not op.subroutine.exact_resources
    ):
        non_zero_gate_counts = {k: v for k, v in gate_counts.items() if v > 0}
        _assert_counts_match(non_zero_gate_counts, actual_gate_counts)
    else:
        # If the resource estimate is not expected to match exactly to the actual
        # decomposition, at least make sure that all gates are accounted for.
        assert all(op in gate_counts for op in actual_gate_counts), (
            "\nGate counts expected from resource function to contain actual gates:\n"
            f"{list(gate_counts.keys())}\nActual gates:\n{list(actual_gate_counts.keys())}\n"
            "Missing in gate counts from resource function:\n"
            f"{[op for op in actual_gate_counts if op not in gate_counts]}"
        )

    # Tests that the decomposition produces the same matrix
    if op.has_matrix and not skip_decomp_matrix_check:
        # Add projector to the additional wires (work wires) on the tape
        work_wires = tape.wires - op.wires
        all_wires = op.wires + work_wires
        if work_wires:
            op = op @ qp.Projector([0] * len(work_wires), wires=work_wires)
            tape.operations.insert(0, qp.Projector([0] * len(work_wires), wires=work_wires))

        op_matrix = op.matrix(wire_order=all_wires)
        decomp_matrix = qp.matrix(tape, wire_order=all_wires)
        assert qp.math.allclose(
            op_matrix, decomp_matrix
        ), "decomposition must produce the same matrix as the operator."


def _check_matrix(op):
    """Check that if the operation says it has a matrix, it does. Otherwise a ``MatrixUndefinedError`` should be raised."""
    if op.has_matrix:
        mat = op.matrix()
        assert isinstance(mat, qp.typing.TensorLike), "matrix must be a TensorLike"
        l = 2 ** len(op.wires)
        failure_comment = f"matrix must be two dimensional with shape ({l}, {l})"
        assert qp.math.shape(mat) == (l, l), failure_comment
    else:
        failure_comment = (
            "If has_matrix is False, the matrix method must raise a ``MatrixUndefinedError``."
        )
        _assert_error_raised(
            op.matrix, qp.operation.MatrixUndefinedError, failure_comment=failure_comment
        )()


def _check_sparse_matrix(op):
    """Check that if the operation says it has a sparse matrix, it does. Otherwise a ``SparseMatrixUndefinedError`` should be raised."""
    if op.has_sparse_matrix:
        mat = op.sparse_matrix()
        assert isinstance(mat, scipy.sparse.csr_matrix), "matrix must be a TensorLike"
        l = 2 ** len(op.wires)
        failure_comment = f"matrix must be two dimensional with shape ({l}, {l})"
        assert qp.math.shape(mat) == (l, l), failure_comment

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
            qp.operation.SparseMatrixUndefinedError,
            failure_comment=failure_comment,
        )()


def _check_matrix_matches_decomp(op):
    """Check that if both the matrix and decomposition are defined, they match."""
    if op.has_matrix and op.has_decomposition:
        mat = op.matrix()
        decomp_mat = qp.matrix(qp.tape.QuantumScript(op.decomposition()), wire_order=op.wires)
        failure_comment = (
            f"matrix and matrix from decomposition must match. Got \n{mat}\n\n {decomp_mat}"
        )
        assert qp.math.allclose(mat, decomp_mat), failure_comment


def _check_eigendecomposition(op):
    """Checks involving diagonalizing gates and eigenvalues."""
    if op.has_diagonalizing_gates:
        dg = op.diagonalizing_gates()
        try:
            compute_dg = type(op).compute_diagonalizing_gates(
                *op.data, wires=op.wires, **op.hyperparameters
            )
        except (qp.operation.DiagGatesUndefinedError, TypeError):
            # sometimes diagonalizing gates is defined but not compute_diagonalizing_gates
            # compute_diagonalizing_gates might also have a different call signature
            compute_dg = dg

        for op1, op2 in zip(dg, compute_dg, strict=True):
            assert op1 == op2, "diagonalizing_gates and compute_diagonalizing_gates must match"
    else:
        failure_comment = "If has_diagonalizing_gates is False, diagonalizing_gates must raise a DiagGatesUndefinedError"
        _assert_error_raised(
            op.diagonalizing_gates, qp.operation.DiagGatesUndefinedError, failure_comment
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
        assert qp.math.allclose(eg, compute_eg), "eigvals and compute_eigvals must match"

    if has_eigvals and op.has_diagonalizing_gates:
        dg = qp.prod(*dg[::-1]) if len(dg) > 0 else qp.Identity(op.wires)
        eg = qp.QubitUnitary(np.diag(eg), wires=op.wires)
        decomp = qp.prod(qp.adjoint(dg), eg, dg)
        decomp_mat = qp.matrix(decomp)
        original_mat = qp.matrix(op)
        failure_comment = f"eigenvalues and diagonalizing gates must be able to reproduce the original operator. Got \n{decomp_mat}\n\n{original_mat}"
        assert qp.math.allclose(decomp_mat, original_mat), failure_comment


def _check_generator(op):
    """Checks that if an operator's has_generator property is True, it has a generator."""

    if op.has_generator:
        gen = op.generator()
        assert isinstance(gen, qp.operation.Operator)
        new_op = qp.exp(gen, 1j * op.data[0])
        assert qp.math.allclose(
            qp.matrix(op, wire_order=op.wires), qp.matrix(new_op, wire_order=op.wires)
        )
    else:
        failure_comment = (
            "If has_generator is False, the matrix method must raise a ``GeneratorUndefinedError``."
        )
        _assert_error_raised(
            op.generator, qp.operation.GeneratorUndefinedError, failure_comment=failure_comment
        )()


def _check_copy(op, skip_deepcopy):
    """Check that copies and deep copies give identical objects."""
    copied_op = copy.copy(op)
    assert qp.equal(copied_op, op), "copied op must be equal with qp.equal"
    assert copied_op == op, "copied op must be equivalent to original operation"
    assert copied_op is not op, "copied op must be a separate instance from original operaiton"
    if not skip_deepcopy:
        try:
            assert_equal(copy.deepcopy(op), op)
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
        assert_equal(op, new_op)
    except AssertionError as e:
        raise AssertionError(
            "metadata and data must be able to reproduce the original operation"
        ) from e
    try:
        import jax
    except ImportError:
        return
    leaves, struct = jax.tree_util.tree_flatten(op)
    unflattened_op = jax.tree_util.tree_unflatten(struct, leaves)
    assert unflattened_op == op, f"op must be a valid pytree. Got {unflattened_op} instead of {op}."

    for d1, d2 in zip(op.data, leaves, strict=True):
        assert qp.math.allclose(
            d1, d2
        ), f"data must be the terminal leaves of the pytree. Got {d1}, {d2}"


def _check_capture(op):
    if isinstance(op, qp.templates.SubroutineOp):
        return
    try:
        import jax
    except ImportError:
        return

    if not all(isinstance(w, int) for w in op.wires):
        return

    qp.capture.enable()
    try:
        data, struct = jax.tree_util.tree_flatten(op)

        def test_fn(*args):
            return jax.tree_util.tree_unflatten(struct, args)

        jaxpr = jax.make_jaxpr(test_fn)(*data)
        new_op = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *data)[0]
        assert_equal(op, new_op)

        leaves = jax.tree_util.tree_leaves(jaxpr.eqns[-1].params)
        assert not any(
            qp.math.is_abstract(l) for l in leaves
        ), "capture params cannot contain tracers"
    except Exception as e:
        raise ValueError(
            "The capture of the operation into jaxpr failed somehow."
            " This capture mechanism is currently experimental and not a core"
            " requirement, but will be necessary in the future."
            " Please see the capture module documentation for more information."
        ) from e
    finally:
        qp.capture.disable()


def _check_pickle(op):
    """Check that an operation can be dumped and reloaded with pickle."""
    pickled = pickle.dumps(op)
    unpickled = pickle.loads(pickled)
    assert unpickled == op, "operation must be able to be pickled and unpickled"


def _check_bind_new_parameters(op):
    """Check that bind new parameters can create a new op with different data."""
    new_data = [d * 0.0 for d in op.data]
    new_data_op = qp.ops.functions.bind_new_parameters(op, new_data)
    failure_comment = "bind_new_parameters must be able to update the operator with new data."
    for d1, d2 in zip(new_data_op.data, new_data, strict=True):
        assert qp.math.allclose(d1, d2), failure_comment


def _check_differentiation(op):
    """Checks that the operator can be executed and differentiated correctly."""

    if op.num_params == 0:
        return

    data, struct = qp.pytrees.flatten(op)

    def circuit(*args):
        qp.apply(qp.pytrees.unflatten(args, struct))
        return qp.probs(wires=op.wires)

    qnode_ref = qp.QNode(circuit, qp.device("default.qubit"), diff_method="backprop")
    qnode_ps = qp.QNode(circuit, qp.device("default.qubit"), diff_method="parameter-shift")

    params = [x if isinstance(x, int) else qp.numpy.array(x) for x in data]

    ps = qp.jacobian(qnode_ps)(*params)
    expected_bp = qp.jacobian(qnode_ref)(*params)

    error_msg = (
        "Parameter-shift does not produce the same Jacobian as with backpropagation. "
        "This might be a bug, or it might be expected due to the mathematical nature "
        "of backpropagation, in which case, this test can be skipped for this operator."
    )

    if isinstance(ps, tuple):
        for actual, expected in zip(ps, expected_bp, strict=True):
            assert qp.math.allclose(actual, expected), error_msg
    else:
        assert qp.math.allclose(ps, expected_bp), error_msg


def _check_wires(op, skip_wire_mapping):
    """Check that wires are a ``Wires`` class and can be mapped."""
    assert isinstance(op.wires, qp.wires.Wires), "wires must be a wires instance"
    if skip_wire_mapping:
        return
    wire_map = {w: ascii_lowercase[i] for i, w in enumerate(op.wires)}
    mapped_op = op.map_wires(wire_map)
    new_wires = qp.wires.Wires(list(ascii_lowercase[: len(op.wires)]))
    assert mapped_op.wires == new_wires, "wires must be mappable with map_wires"


# pylint: disable=too-many-arguments
def assert_valid(
    op: qp.operation.Operator,
    *,
    skip_deepcopy=False,
    skip_differentiation=False,
    skip_new_decomp=False,
    skip_decomp_matrix_check=False,
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
        skip_decomp_matrix_check: If ``True``, the decomposition rule check will only
            verify that the produced operators match the resource function, and does not
            test that the matrix of the decomposition matches the operator itself.
        skip_pickle=False : If ``True``, pickling tests are not run. Set to ``True`` when
            testing a locally defined operator, as pickle cannot handle local objects
        skip_wire_mapping : If ``True``, the operator will not be tested for wire mapping.
        skip_capture: If ``True``, the program capture tests will be skipped.

    **Examples:**

    .. code-block:: python

        class MyOp(qp.operation.Operator):

            def __init__(self, data, wires):
                self.data = data
                super().__init__(wires=wires)

        op = MyOp(qp.numpy.array(0.5), wires=0)

    >>> assert_valid(op)
    Traceback (most recent call last):
        ...
    AssertionError: MyOp._unflatten must be able to reproduce the original operation from () and (Wires([0]), ()). You may need to override either the _unflatten or _flatten method.
    For local testing, try type(op)._unflatten(*op._flatten())

    .. code-block:: python

        class MyOp(qp.operation.Operator):

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
    for d, p in zip(op.data, op.parameters, strict=True):
        assert isinstance(d, qp.typing.TensorLike), "each data element must be tensorlike"
        assert qp.math.allclose(d, p), "data and parameters must match."

    if len(op.wires) <= 26:
        _check_wires(op, skip_wire_mapping=skip_wire_mapping)
    _check_copy(op, skip_deepcopy=skip_deepcopy)
    _check_pytree(op)
    if not skip_pickle:
        _check_pickle(op)
    _check_bind_new_parameters(op)
    _check_decomposition(op, skip_wire_mapping=skip_wire_mapping)
    if not skip_new_decomp:
        _check_decomposition_new(op, skip_decomp_matrix_check=skip_decomp_matrix_check)
        _check_reconstructor(op)
    _check_matrix(op)
    _check_matrix_matches_decomp(op)
    _check_sparse_matrix(op)
    _check_eigendecomposition(op)
    _check_generator(op)
    if not skip_differentiation:
        _check_differentiation(op)
    if not skip_capture:
        _check_capture(op)
