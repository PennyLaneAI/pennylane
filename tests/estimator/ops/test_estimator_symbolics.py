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
from collections import defaultdict

import pytest

import pennylane.estimator as qre
from pennylane.estimator.ops.op_math.symbolic import apply_adj
from pennylane.estimator.resource_operator import GateCount, resource_rep
from pennylane.estimator.wires_manager import Allocate, Deallocate
from pennylane.queuing import AnnotatedQueue
from pennylane.wires import Wires

# pylint: disable=no-self-use, too-few-public-methods


class DummyOp(qre.ResourceOperator):
    resource_keys = {"num_wires"}

    def __init__(self, num_wires, wires=None):
        self.num_wires = num_wires
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        return {"num_wires": self.num_wires}

    @classmethod
    def resource_rep(cls, num_wires) -> qre.CompressedResourceOp:
        params = {"num_wires": num_wires}
        return qre.CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, num_wires) -> list[GateCount]:
        return [
            Allocate(num_wires),
            GateCount(qre.X.resource_rep()),
            Deallocate(num_wires),
        ]


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

        op = DummyOp(num_wires=1)  # no default_adjoint_decomp defined
        adj_op = qre.Adjoint(op)
        expected_res = [
            Allocate(1),
            GateCount(
                qre.resource_rep(
                    qre.Adjoint,
                    {"base_cmpr_op": qre.X.resource_rep()},
                ),
                1,
            ),
            Deallocate(1),
        ]
        assert adj_op.resource_decomp(**adj_op.resource_params) == expected_res

    @pytest.mark.parametrize(
        "base_op, adj_res",
        (
            (
                qre.SemiAdder(5),
                qre.Resources(
                    zeroed_wires=4,
                    any_state_wires=0,
                    algo_wires=10,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(qre.CNOT): 25,
                            resource_rep(qre.Toffoli, {"elbow": "left"}): 4,
                            resource_rep(qre.Hadamard): 12,
                        },
                    ),
                ),
            ),
            (
                qre.CRZ(precision=1e-3),
                qre.Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=2,
                    gate_types=defaultdict(
                        int, {resource_rep(qre.CNOT): 2, resource_rep(qre.T): 42}
                    ),
                ),
            ),
            (
                qre.CRZ(),
                qre.Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=2,
                    gate_types=defaultdict(
                        int, {resource_rep(qre.CNOT): 2, resource_rep(qre.T): 88}
                    ),
                ),
            ),
        ),
    )
    def test_estimate_for_adjoint(self, base_op, adj_res):
        """Test that the adjoint of this operator produces expected resources with estimate."""
        adj_op = qre.Adjoint(base_op)
        adj_adj_op = qre.Adjoint(adj_op)

        assert qre.estimate(adj_op) == adj_res
        assert qre.estimate(adj_adj_op) == adj_res

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

    def test_apply_adj(self):
        """Test that the apply_adj method is working correctly."""
        assert apply_adj(Allocate(1)) == Deallocate(1)
        assert apply_adj(Deallocate(1)) == Allocate(1)

        expected_res = GateCount(qre.Adjoint.resource_rep(qre.T.resource_rep()), 1)
        assert apply_adj(GateCount(qre.T.resource_rep(), 1)) == expected_res

    def test_raises_error_on_unknown_type(self):
        """Test that the apply_adj method is working correctly."""
        with pytest.raises(TypeError):
            qre.ops.op_math.symbolic.apply_adj(1)


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

        op = DummyOp(num_wires=1)  # no default_controlled_decomp defined
        ctrl_op = qre.Controlled(op, num_ctrl_wires=3, num_zero_ctrl=2)
        expected_res = [
            GateCount(qre.resource_rep(qre.X), 4),
            Allocate(1),
            GateCount(
                qre.Controlled.resource_rep(
                    qre.resource_rep(qre.X),
                    num_ctrl_wires=3,
                    num_zero_ctrl=0,
                ),
                1,
            ),
            Deallocate(1),
        ]
        assert ctrl_op.resource_decomp(**ctrl_op.resource_params) == expected_res

    def test_else_block_of_apply_controlled(self):
        """Test that the else block of the apply_controlled method for code coverage purposes."""

        base_op = DummyOp(num_wires=2)
        ctrl_op = qre.Controlled(base_op, num_ctrl_wires=2, num_zero_ctrl=1)

        expected_res = [
            GateCount(qre.X.resource_rep(), 2),
            qre.Allocate(2),
            GateCount(
                qre.Controlled.resource_rep(
                    qre.X.resource_rep(),
                    num_ctrl_wires=2,
                    num_zero_ctrl=0,
                ),
                1,
            ),
            qre.Deallocate(2),
        ]

        assert ctrl_op.resource_decomp(**ctrl_op.resource_params) == expected_res

    @pytest.mark.parametrize(
        "base_op, ctrl_res, ctrl_ctrl_res",
        (
            (
                qre.SemiAdder(5),
                qre.Resources(
                    zeroed_wires=4,
                    any_state_wires=0,
                    algo_wires=11,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(qre.CNOT): 32,
                            resource_rep(qre.Toffoli, {"elbow": "left"}): 8,
                            resource_rep(qre.Hadamard): 24,
                        },
                    ),
                ),
                qre.Resources(
                    zeroed_wires=5,
                    any_state_wires=0,
                    algo_wires=12,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(qre.CNOT): 32,
                            resource_rep(qre.Toffoli, {"elbow": "left"}): 8,
                            resource_rep(qre.Toffoli): 2,
                            resource_rep(qre.Hadamard): 24,
                        },
                    ),
                ),
            ),
            (
                qre.CRZ(precision=1e-3),
                qre.Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=3,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(qre.T): 42,
                            resource_rep(qre.Toffoli): 2,
                        },
                    ),
                ),
                qre.Resources(
                    zeroed_wires=1,
                    any_state_wires=0,
                    algo_wires=4,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(qre.CNOT): 2,
                            resource_rep(qre.T): 42,
                            resource_rep(qre.Toffoli): 2,
                            resource_rep(qre.Toffoli, {"elbow": "left"}): 2,
                            resource_rep(qre.Hadamard): 6,
                        },
                    ),
                ),
            ),
            (
                qre.CRZ(),
                qre.Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=3,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(qre.T): 88,
                            resource_rep(qre.Toffoli): 2,
                        },
                    ),
                ),
                qre.Resources(
                    zeroed_wires=1,
                    any_state_wires=0,
                    algo_wires=4,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(qre.CNOT): 2,
                            resource_rep(qre.T): 88,
                            resource_rep(qre.Toffoli): 2,
                            resource_rep(qre.Toffoli, {"elbow": "left"}): 2,
                            resource_rep(qre.Hadamard): 6,
                        },
                    ),
                ),
            ),
        ),
    )
    def test_estimate_for_controlled(self, base_op, ctrl_res, ctrl_ctrl_res):
        """Test that the controlled operator produces expected resources with estimate"""
        ctrl_op = qre.Controlled(base_op, num_ctrl_wires=1, num_zero_ctrl=0)
        ctrl_ctrl_op = qre.Controlled(ctrl_op, num_ctrl_wires=1, num_zero_ctrl=0)
        assert qre.estimate(ctrl_op) == ctrl_res
        assert qre.estimate(ctrl_ctrl_op) == ctrl_ctrl_res

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


class TestPow:
    """Tests for the Pow resource Op"""

    @pytest.mark.parametrize("z", (0, 1, 2, 3))
    @pytest.mark.parametrize(
        "base_type, base_args",
        (
            (qre.S, {}),
            (qre.RZ, {"precision": 1e-3}),
            (qre.CNOT, {"wires": ["ctrl", "trgt"]}),
        ),
    )
    def test_init(self, z, base_type, base_args):
        """Test that the operator is instantiated correctly"""
        with AnnotatedQueue() as q:
            base_op = base_type(**base_args)
            pow_base_op = qre.Pow(base_op, z)

        assert base_op not in q.queue
        assert pow_base_op.num_wires == base_op.num_wires
        assert base_op.wires == pow_base_op.wires

    def test_resource_decomp(self):
        """Test that we can obtain the resources as expected"""
        op = qre.X()  # has default_pow_decomp defined
        z_and_expected_res = (
            (0, [GateCount(qre.resource_rep(qre.Identity()))]),
            (1, [GateCount(op.resource_rep_from_op())]),
            (2, op.pow_resource_decomp(2, **op.resource_params)),
            (3, op.pow_resource_decomp(3, **op.resource_params)),
        )

        for z, res in z_and_expected_res:
            pow_op = qre.Pow(op, z)
            assert pow_op.resource_decomp(**pow_op.resource_params) == res

        op = DummyOp(num_wires=1)  # no default_pow_decomp defined
        z_and_expected_res_unitary = (
            (0, [GateCount(qre.resource_rep(qre.Identity()))]),
            (1, [GateCount(op.resource_rep_from_op())]),
            (2, [GateCount(op.resource_rep_from_op(), 2)]),
            (3, [GateCount(op.resource_rep_from_op(), 3)]),
        )

        for z, res in z_and_expected_res_unitary:
            pow_op = qre.Pow(op, z)
            assert pow_op.resource_decomp(**pow_op.resource_params) == res

    @pytest.mark.parametrize("z", (0, 1, 2, 3))
    @pytest.mark.parametrize(
        "base_op",
        (
            qre.S(),
            qre.RZ(precision=1e-3),
            qre.CNOT(wires=["ctrl", "trgt"]),
        ),
    )
    def test_pow_resource_decomp(self, base_op, z):
        """Test that the power of this operator produces resources as expected."""
        pow_op = qre.Pow(base_op, z)
        pow_pow_op = qre.Pow(pow_op, pow_z=5)

        expected_res = [
            GateCount(
                qre.Pow.resource_rep(
                    base_op.resource_rep_from_op(),
                    pow_z=5 * z,
                )
            )
        ]
        assert pow_pow_op.resource_decomp(**pow_pow_op.resource_params) == expected_res

    @pytest.mark.parametrize(
        "base_op, z, pow_res",
        (
            (
                qre.RX(),
                2,
                qre.Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=1,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(qre.T): 44,
                        },
                    ),
                ),
            ),
            (
                qre.RX(precision=1e-3),
                2,
                qre.Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=1,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(qre.T): 21,
                        },
                    ),
                ),
            ),
        ),
    )
    def test_estimate_for_pow(self, base_op, z, pow_res):
        """Test that the power of this operator produces expected resources with estimate."""
        pow_op = qre.Pow(base_op, z)
        pow_pow_op = qre.Pow(pow_op, pow_z=5)
        assert qre.estimate(pow_op) == pow_res
        assert (qre.estimate(pow_pow_op)) == pow_res

    def test_tracking_name(self):
        """Test that the name of the operator is tracked correctly."""
        assert qre.Pow.tracking_name(qre.T.resource_rep(), 1) == "Pow(T, 1)"
        assert qre.Pow.tracking_name(qre.S.resource_rep(), 2) == "Pow(S, 2)"
        assert qre.Pow.tracking_name(qre.CNOT.resource_rep(), 3) == "Pow(CNOT, 3)"


class TestProd:
    """Tests for the Prod resource Op"""

    def test_init(self):
        """Test that the operator is instantiated correctly"""
        with AnnotatedQueue() as q:
            ops = [
                qre.X(wires=0),
                qre.CZ(wires=[1, 2]),
                (qre.Hadamard(), 4),
                qre.RX(precision=1e-4, wires=3),
                (qre.CNOT(), 2),
            ]
            prod_op = qre.Prod(res_ops=ops)

        for op in ops:
            if isinstance(op, tuple):
                op = op[0]
            assert op not in q.queue

        assert prod_op.num_wires == 4
        assert prod_op.wires == Wires([0, 1, 2, 3])

    def test_resource_decomp(self):
        """Test that we can obtain the resources as expected"""
        ops = [
            qre.X(wires=0),
            qre.CZ(wires=[1, 2]),
            (qre.Hadamard(), 4),
            qre.RX(precision=1e-4, wires=3),
            [qre.CNOT(), 2],
        ]
        prod_op = qre.Prod(res_ops=ops)

        expected_res = [
            GateCount(qre.resource_rep(qre.X)),
            GateCount(qre.resource_rep(qre.CZ)),
            GateCount(qre.resource_rep(qre.Hadamard), 4),
            GateCount(qre.resource_rep(qre.RX, {"precision": 1e-4})),
            GateCount(qre.resource_rep(qre.CNOT), 2),
        ]
        assert prod_op.resource_decomp(**prod_op.resource_params) == expected_res

    def test_resource_init_wires(self):
        """Test that the operator initializes correctly with wires."""
        prod_op = qre.Prod([qre.X(), qre.Y()], wires=[0, 1])
        assert prod_op.num_wires == 2
        assert prod_op.wires == Wires([0, 1])

        prod_op = qre.Prod([qre.X(), qre.Y()])
        assert prod_op.wires is None

    def test_resource_init_raises(self):
        """Test that the operator raises an error if the resource operator is not a ResourceOperator."""
        with pytest.raises(ValueError, match="All factors of the Product must be"):
            qre.Prod([qre.X(), 3])


class TestChangeOpBasis:
    """Tests for the ChangeOpBasis resource Op"""

    def test_init(self):
        """Test that the operator is instantiated correctly"""
        with AnnotatedQueue() as q:
            compute_op = qre.S(wires=0)
            base_op = qre.Prod(((qre.Z(), 3),), wires=[0, 1, 2])
            uncompute_op = qre.Pow(qre.T(wires=0), 6)

            cb_op = qre.ChangeOpBasis(compute_op, base_op, uncompute_op)

        for op in [compute_op, base_op, uncompute_op]:
            assert op not in q.queue

        assert cb_op.num_wires == 3
        assert cb_op.wires == Wires([0, 1, 2])

    def test_resource_init_raises(self):
        """Test that the operator raises an error if the resource operator is not a ResourceOperator."""
        with pytest.raises(ValueError, match="All ops of the ChangeOpBasis must be"):
            qre.ChangeOpBasis(qre.X(), 3, qre.X())

    def test_resource_init_wires(self):
        """Test that the operator initializes correctly with wires."""
        cb_op = qre.ChangeOpBasis(qre.X(), qre.Y(), qre.X(), wires=[0, 1])
        assert cb_op.num_wires == 2
        assert cb_op.wires == Wires([0, 1])

        cb_op = qre.ChangeOpBasis(qre.X(), qre.Y(), qre.Z())
        assert cb_op.wires is None

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize(
        "compute_op, target_op, uncompute_op, expected_res",
        (
            (
                qre.S(wires=0),
                qre.Prod([(qre.Z(), 3)], wires=[0, 1, 2]),
                qre.Pow(qre.T(wires=0), 6),
                [
                    GateCount(qre.resource_rep(qre.S)),
                    GateCount(
                        qre.resource_rep(
                            qre.Prod,
                            {
                                "cmpr_factors_and_counts": ((qre.Z.resource_rep(), 3),),
                                "num_wires": 3,
                            },
                        ),
                    ),
                    GateCount(
                        qre.resource_rep(
                            qre.Pow,
                            {
                                "base_cmpr_op": (qre.T.resource_rep()),
                                "pow_z": 6,
                            },
                        ),
                    ),
                ],
            ),
            (
                qre.S(wires=0),
                qre.Prod([(qre.Z(), 3)], wires=[0, 1, 2]),
                None,
                [
                    GateCount(qre.resource_rep(qre.S)),
                    GateCount(
                        qre.resource_rep(
                            qre.Prod,
                            {
                                "cmpr_factors_and_counts": ((qre.Z.resource_rep(), 3),),
                                "num_wires": 3,
                            },
                        ),
                    ),
                    GateCount(
                        qre.resource_rep(
                            qre.Adjoint,
                            {
                                "base_cmpr_op": (qre.S.resource_rep()),
                            },
                        ),
                    ),
                ],
            ),
        ),
    )
    def test_resource_decomp(self, compute_op, target_op, uncompute_op, expected_res):
        """Test that we can obtain the resources as expected"""
        cb_op = qre.ChangeOpBasis(compute_op, target_op, uncompute_op)

        assert cb_op.resource_decomp(**cb_op.resource_params) == expected_res

    @pytest.mark.parametrize(
        "compute_op, target_op, uncompute_op, num_wires",
        (
            (
                qre.S(wires=0),
                qre.Prod([(qre.Z(), 3)], wires=[0, 1, 2]),
                qre.Pow(qre.T(wires=0), 6),
                None,
            ),
            (qre.S(wires=0), qre.Prod([(qre.Z(), 3)], wires=[0, 1, 2]), None, 3),
        ),
    )
    def test_resource_rep(self, compute_op, target_op, uncompute_op, num_wires):
        """Test that correct compressed representation is obtained."""
        op = qre.ChangeOpBasis(compute_op, target_op, uncompute_op)
        cmpr_compute_op = compute_op.resource_rep_from_op()
        cmpr_target_op = target_op.resource_rep_from_op()

        if uncompute_op:
            cmpr_uncompute_op = uncompute_op.resource_rep_from_op()
        else:
            cmpr_uncompute_op = resource_rep(qre.Adjoint, {"base_cmpr_op": cmpr_compute_op})

        expected = qre.CompressedResourceOp(
            qre.ChangeOpBasis,
            3,
            {
                "cmpr_compute_op": cmpr_compute_op,
                "cmpr_target_op": cmpr_target_op,
                "cmpr_uncompute_op": cmpr_uncompute_op,
                "num_wires": 3,
            },
        )

        assert (
            op.resource_rep(cmpr_compute_op, cmpr_target_op, cmpr_uncompute_op, num_wires)
            == expected
        )

    @pytest.mark.parametrize(
        "compute_op, target_op, uncompute_op, num_ctrl_wires, num_zero_ctrl, expected_res",
        (
            (
                qre.S(wires=0),
                qre.X(wires=0),
                qre.S(wires=0),
                1,
                0,
                [
                    qre.GateCount(qre.resource_rep(qre.S), 1),
                    qre.GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=qre.X.resource_rep(),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        ),
                        1,
                    ),
                    qre.GateCount(qre.resource_rep(qre.S), 1),
                ],
            ),
            (
                qre.Hadamard(wires=0),
                qre.Z(wires=0),
                None,
                2,
                1,
                [
                    qre.GateCount(qre.resource_rep(qre.Hadamard), 1),
                    qre.GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=qre.Z.resource_rep(),
                            num_ctrl_wires=2,
                            num_zero_ctrl=1,
                        ),
                        1,
                    ),
                    qre.GateCount(
                        qre.resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": qre.Hadamard.resource_rep()},
                        ),
                        1,
                    ),
                ],
            ),
        ),
    )
    def test_controlled_resource_decomp(
        self, compute_op, target_op, uncompute_op, num_ctrl_wires, num_zero_ctrl, expected_res
    ):
        """Test that the controlled resource decomposition is correct."""
        cb_op = qre.ChangeOpBasis(compute_op, target_op, uncompute_op)
        res = cb_op.controlled_resource_decomp(num_ctrl_wires, num_zero_ctrl, cb_op.resource_params)
        assert res == expected_res
