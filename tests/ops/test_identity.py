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
"""Unit tests for the Identity Operator."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.core.operator.utils import abstractify
from pennylane.ops.functions import assert_valid
from pennylane.ops.identity import GlobalPhase, Identity, _ctrl_g_phase
from pennylane.typing import Float, Wire

op_wires = [[], [0], ["a"], [0, 1], ["a", "b", "c"], [100, "xasd", 12]]
op_repr = ["I()", "I(0)", "I('a')", "I([0, 1])", "I(['a', 'b', 'c'])", "I([100, 'xasd', 12])"]
op_params = tuple(zip(op_wires, op_repr))


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("phi", (0.0, 1.0, -1.0))
def test_global_phase_is_valid(phi):
    """Tests that the GlobalPhase operator is valid."""
    op = GlobalPhase(phi)
    assert_valid(op, skip_differentiation=True)


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("wires", ((), [0], [0, 1]))
def test_identity_is_valid(wires):
    """Tests that Identity ops are valid."""
    op = Identity(wires)
    assert_valid(op, skip_differentiation=True)


def test_abstractify_globalphase():
    """Test that globalphase can be abstractified."""

    assert abstractify(GlobalPhase(0.5)) == GlobalPhase(Float)
    assert abstractify(GlobalPhase(1)) == GlobalPhase(Float)


def test_abstractify_identity():
    """Test that identity can be abstractified."""

    assert abstractify(Identity(wires=[0])) == Identity(Wire[1])
    assert abstractify(Identity(wires=[0, 1])) == Identity(Wire[2])


class TestControlledGlobalPhase:
    """Tests for the custom controlled dispatch of GlobalPhase (``_ctrl_g_phase``)."""

    @staticmethod
    def _expected_matrix(phi, n_control):
        """A controlled GlobalPhase applies ``e^{-i phi}`` to the all-ones control state."""
        mat = np.eye(2**n_control, dtype=complex)
        mat[-1, -1] = np.exp(-1j * phi)
        return mat

    def test_single_control_returns_phase_shift(self):
        """A single control turns GlobalPhase into a PhaseShift on the control wire."""
        op = qp.ctrl(GlobalPhase(0.123), control=[0])
        qp.assert_equal(op, qp.PhaseShift(-0.123, wires=0))

    def test_two_controls_returns_controlled_phase_shift(self):
        """Two controls turn GlobalPhase into a ControlledPhaseShift."""
        op = qp.ctrl(GlobalPhase(0.123), control=[0, 1])
        qp.assert_equal(op, qp.ControlledPhaseShift(-0.123, wires=[0, 1]))

    @pytest.mark.parametrize("control", ([0], [0, 1], [0, 1, 2]))
    def test_matrix_matches_controlled_global_phase(self, control):
        """The dispatched op reproduces the controlled-GlobalPhase matrix."""
        phi = 0.123
        op = qp.ctrl(GlobalPhase(phi), control=control)
        mat = qp.matrix(op, wire_order=control)
        assert np.allclose(mat, self._expected_matrix(phi, len(control)))

    def test_not_all_true_control_values_not_implemented(self):
        """With a zero control value the dispatch declines, falling back to a generic Controlled."""
        res = _ctrl_g_phase(GlobalPhase(0.123), qp.wires.Wires([0]), [False])
        assert res is NotImplemented

        op = qp.ctrl(GlobalPhase(0.123), control=[0], control_values=[False])
        assert not isinstance(op, qp.PhaseShift)

    @pytest.mark.parametrize(
        "phi, control, work_wires, work_wire_type",
        [
            (0.123, [0, 1, 2], [3, 4], "zeroed"),
            (0.5, [0, 1, 2], [3], "borrowed"),
            (-1.0, [0, 1, 2, 3], [4, 5, 6], "zeroed"),
        ],
    )
    def test_work_wires_passed_through(self, phi, control, work_wires, work_wire_type):
        """Regression test that work wires are forwarded to the multi-control dispatch."""
        op = qp.ctrl(
            GlobalPhase(phi),
            control=control,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
        assert list(op.work_wires) == work_wires
        assert op.work_wire_type == work_wire_type
        qp.assert_equal(
            op,
            qp.ctrl(
                qp.PhaseShift(-phi, wires=control[-1]),
                control=control[:-1],
                work_wires=work_wires,
                work_wire_type=work_wire_type,
            ),
        )


def test_is_verified_hermitian():
    """Test that identity is verified to be hermitian."""
    assert Identity.is_verified_hermitian is True
    assert Identity(0).is_verified_hermitian is True


@pytest.mark.parametrize("wires", op_wires)
class TestIdentity:
    # pylint: disable=protected-access
    def test_flatten_unflatten(self, wires):
        """Test the flatten and unflatten methods of identity."""
        op = Identity(wires)
        data, metadata = op._flatten()
        assert data == ([], [qp.wires.Wires(wires)], [])
        assert hash(metadata)

        new_op = Identity._unflatten(*op._flatten())
        qp.assert_equal(op, new_op)

    def test_class_name(self, wires):
        """Test the class name of either I and Identity is by default 'Identity'"""
        assert qp.I.__name__ == "Identity"
        assert qp.Identity.__name__ == "Identity"

        assert qp.I(wires).name == "Identity"
        assert qp.Identity(wires).name == "Identity"

    @pytest.mark.jax
    def test_jax_pytree_integration(self, wires):
        """Test that identity round-trips through the jax pytree registry."""
        import jax

        op = qp.Identity(wires)

        leaves, tree_def = jax.tree_util.tree_flatten(op)
        qp.assert_equal(jax.tree_util.tree_unflatten(tree_def, leaves), op)

        if all(isinstance(w, int) for w in wires):
            # ``Operator2`` treats wires as dynamic pytree leaves, so only integer wire
            # labels can be traced by ``jax.jit``.
            adj_op = jax.jit(lambda op: qp.adjoint(op, lazy=False))(op)
            qp.assert_equal(op, adj_op)

    def test_identity_eigvals(self, wires, tol):
        """Test identity eigenvalues are correct"""
        res = Identity(wires).eigvals()
        expected = np.ones(2 ** len(wires))
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_decomposition(self, wires):
        """Test the decomposition of the identity operation."""

        assert Identity.compute_decomposition(wires=wires) == []
        assert Identity(wires=wires).decomposition() == []

    def test_label_method(self, wires):
        """Test the label method for the Identity Operator"""
        assert Identity(wires=wires).label() == "I"

    @pytest.mark.parametrize("n", (2, -3, 3.455, -1.29))
    def test_identity_pow(self, wires, n):
        """Test that the identity raised to any power is simply a single copy."""
        op = Identity(wires)
        pow_ops = op.pow(n)
        assert len(pow_ops) == 1
        assert pow_ops[0].__class__ is Identity
        assert pow_ops[0].wires == op.wires

    def test_matrix_representation(self, wires, tol):
        """Test the matrix representation"""
        res_static = Identity.compute_matrix(wires=wires)
        res_dynamic = Identity(wires=wires).matrix()
        expected = np.eye(int(2 ** len(wires)))
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)

    def test_sparse_matrix_format(self, wires):
        from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, lil_matrix

        op = qp.Identity(wires=wires)
        assert isinstance(op.sparse_matrix(), csr_matrix)
        assert isinstance(op.sparse_matrix(format="csc"), csc_matrix)
        assert isinstance(op.sparse_matrix(format="lil"), lil_matrix)
        assert isinstance(op.sparse_matrix(format="coo"), coo_matrix)
        assert qp.math.allclose(op.matrix(), op.sparse_matrix().toarray())


@pytest.mark.parametrize("wires, expected_repr", op_params)
def test_repr(wires, expected_repr):
    """Test the operator's repr"""
    op = Identity(wires=wires)
    assert repr(op) == expected_repr
