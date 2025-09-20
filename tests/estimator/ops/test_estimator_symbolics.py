# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Tests for symbolic resource operators.
"""
import pytest

import pennylane.estimator as qre
from pennylane.estimator.resource_operator import GateCount
from pennylane.estimator.wires_manager import Allocate, Deallocate
from pennylane.exceptions import ResourcesUndefinedError
from pennylane.queuing import AnnotatedQueue

# pylint: disable=no-self-use, too-few-public-methods


class TestAdjoint:
    """Tests for the Adjoint resource Op"""

    @pytest.mark.parametrize(
        "base_type, base_args",
        (
            (qre.S, {}),
            (qre.RZ, {"precision": 1e-3}),
            (qre.CNOT, {"wires": ["ctrl", "trgt"]}),
        ),
    )
    def test_init(self, base_type, base_args):
        """Test that the operator is instantiated correctly"""
        with AnnotatedQueue() as q:
            base_op = base_type(**base_args)
            adj_base_op = qre.Adjoint(base_op)

        assert base_op not in q.queue
        assert adj_base_op.num_wires == base_op.num_wires
        assert base_op.wires == adj_base_op.wires

    def test_resource_decomp(self):
        """Test that we can obtain the resources as expected"""
        op = qre.S()  # has default_adjoint_decomp defined
        adj_op = qre.Adjoint(op)
        assert adj_op.resource_decomp(**adj_op.resource_params) == op.adjoint_resource_decomp(
            **op.resource_params
        )

        class ResourceDummyS(qre.S):
            """Dummy class with no default adjoint decomp"""

            @classmethod
            def adjoint_resource_decomp(cls, target_resource_params=None) -> list[GateCount]:
                """No default resources"""
                raise ResourcesUndefinedError

        op = ResourceDummyS()  # no default_adjoint_decomp defined
        adj_op = qre.Adjoint(op)
        expected_res = [
            GateCount(
                qre.resource_rep(
                    qre.Adjoint,
                    {"base_cmpr_op": qre.resource_rep(qre.T)},
                ),
                2,
            )
        ]
        assert adj_op.resource_decomp(**adj_op.resource_params) == expected_res

    @pytest.mark.parametrize(
        "base_op",
        (
            qre.S(),
            qre.RZ(precision=1e-3),
            qre.CNOT(wires=["ctrl", "trgt"]),
        ),
    )
    def test_adj_resource_decomp(self, base_op):
        """Test that the adjoint of this operator produces resources as expected."""
        adj_op = qre.Adjoint(base_op)
        adj_adj_op = qre.Adjoint(adj_op)

        expected_res = [GateCount(base_op.resource_rep_from_op())]
        assert adj_adj_op.resource_decomp(**adj_adj_op.resource_params) == expected_res

    def test_tracking_name(self):
        """Test that the name of the operator is tracked correctly."""
        assert qre.Adjoint.tracking_name(qre.T.resource_rep()) == "Adjoint(T)"
        assert qre.Adjoint.tracking_name(qre.S.resource_rep()) == "Adjoint(S)"
        assert qre.Adjoint.tracking_name(qre.CNOT.resource_rep()) == "Adjoint(CNOT)"

    # pylint: disable=protected-access, import-outside-toplevel
    def test_apply_adj(self):
        """Test that the apply_adj method is working correctly."""
        from pennylane.estimator.ops.op_math.symbolic import _apply_adj

        assert _apply_adj(Allocate(1)) == Deallocate(1)
        assert _apply_adj(Deallocate(1)) == Allocate(1)

        expected_res = GateCount(qre.Adjoint.resource_rep(qre.T.resource_rep()), 1)
        assert _apply_adj(GateCount(qre.T.resource_rep(), 1)) == expected_res

    # pylint: disable=protected-access
    def test_apply_adj_raises_error_on_unknown_type(self):
        """Test that the apply_adj method is working correctly."""
        with pytest.raises(TypeError):
            qre.ops.op_math.symbolic._apply_adj(1)


class TestControlled:
    """Tests for the Controlled resource Op"""

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values",
        (
            (1, 0),
            (2, 0),
            (3, 2),
        ),
    )
    @pytest.mark.parametrize(
        "base_type, base_args",
        (
            (qre.S, {}),
            (qre.RZ, {"precision": 1e-3}),
            (qre.CNOT, {"wires": ["ctrl", "trgt"]}),
        ),
    )
    def test_init(self, ctrl_wires, ctrl_values, base_type, base_args):
        """Test that the operator is instantiated correctly"""
        wires = list(range(ctrl_wires))
        wires.extend(base_args.get("wires", [ctrl_wires]))
        with AnnotatedQueue() as q:
            base_op = base_type(**base_args)
            ctrl_base_op = qre.Controlled(base_op, ctrl_wires, ctrl_values, wires=wires)

        assert base_op not in q.queue
        assert ctrl_base_op.num_wires == base_op.num_wires + ctrl_wires

    def test_resource_decomp(self):
        """Test that we can obtain the resources as expected"""
        op = qre.Z()  # has default_ctrl_decomp defined
        ctrl_params_and_expected_res = (
            ((1, 0), op.controlled_resource_decomp(1, 0, **op.resource_params)),
            ((2, 0), op.controlled_resource_decomp(2, 0, **op.resource_params)),
            ((3, 2), op.controlled_resource_decomp(3, 2, **op.resource_params)),
        )

        for (ctrl_wires, ctrl_values), res in ctrl_params_and_expected_res:
            ctrl_op = qre.Controlled(op, ctrl_wires, ctrl_values)
            assert ctrl_op.resource_decomp(**ctrl_op.resource_params) == res

        class ResourceDummyZ(qre.Z):
            """Dummy class with no default ctrl decomp"""

            @classmethod
            def controlled_resource_decomp(
                cls, num_ctrl_wires, num_zero_ctrl, **kwargs
            ) -> list[GateCount]:
                """No default resources"""
                raise ResourcesUndefinedError

        op = ResourceDummyZ()  # no default_ctrl_decomp defined
        ctrl_op = qre.Controlled(op, num_ctrl_wires=3, num_zero_ctrl=2)
        expected_res = [
            GateCount(qre.resource_rep(qre.X), 4),
            GateCount(
                qre.Controlled.resource_rep(
                    qre.resource_rep(qre.S),
                    num_ctrl_wires=3,
                    num_zero_ctrl=0,
                ),
                2,
            ),
        ]
        assert ctrl_op.resource_decomp(**ctrl_op.resource_params) == expected_res

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values",
        (
            (1, 0),
            (2, 0),
            (3, 2),
        ),
    )
    @pytest.mark.parametrize(
        "base_op",
        (
            qre.S(),
            qre.RZ(precision=1e-3),
            qre.CNOT(wires=["ctrl", "trgt"]),
        ),
    )
    def test_ctrl_resource_decomp(self, ctrl_wires, ctrl_values, base_op):
        """Test that the control of this operator produces resources as expected."""
        ctrl_op = qre.Controlled(base_op, ctrl_wires, ctrl_values)
        ctrl_ctrl_op = qre.Controlled(ctrl_op, num_ctrl_wires=2, num_zero_ctrl=1)

        expected_res = [
            GateCount(
                qre.Controlled.resource_rep(
                    base_op.resource_rep_from_op(),
                    num_ctrl_wires=ctrl_wires + 2,
                    num_zero_ctrl=ctrl_values + 1,
                )
            )
        ]
        assert ctrl_ctrl_op.resource_decomp(**ctrl_ctrl_op.resource_params) == expected_res

    def test_tracking_name(self):
        """Test that the name of the operator is tracked correctly."""
        assert (
            qre.Controlled.tracking_name(qre.T.resource_rep(), 1, 0)
            == "C(T, num_ctrl_wires=1,num_zero_ctrl=0)"
        )
        assert (
            qre.Controlled.tracking_name(qre.S.resource_rep(), 2, 0)
            == "C(S, num_ctrl_wires=2,num_zero_ctrl=0)"
        )
        assert (
            qre.Controlled.tracking_name(qre.CNOT.resource_rep(), 3, 2)
            == "C(CNOT, num_ctrl_wires=3,num_zero_ctrl=2)"
        )
