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

import pennylane.labs.resource_estimation as plre
from pennylane.labs.resource_estimation.resource_operator import GateCount
from pennylane.queuing import AnnotatedQueue
from pennylane.wires import Wires

# pylint: disable=no-self-use,


class TestResourceAdjoint:
    """Tests for the Adjoint resource Op"""

    @pytest.mark.parametrize(
        "base_type, base_args",
        (
            (plre.ResourceS, {}),
            (plre.ResourceRZ, {"precision": 1e-3}),
            (plre.ResourceCNOT, {"wires": ["ctrl", "trgt"]}),
        ),
    )
    def test_init(self, base_type, base_args):
        """Test that the operator is instantiated correctly"""
        with AnnotatedQueue() as q:
            base_op = base_type(**base_args)
            adj_base_op = plre.ResourceAdjoint(base_op)

        assert base_op not in q.queue
        assert adj_base_op.num_wires == base_op.num_wires
        assert base_op.wires == adj_base_op.wires

    def test_resource_decomp(self):
        """Test that we can obtain the resources as expected"""
        op = plre.ResourceS()  # has default_adjoint_decomp defined
        adj_op = plre.ResourceAdjoint(op)
        assert adj_op.resource_decomp(**adj_op.resource_params) == op.adjoint_resource_decomp(
            **op.resource_params
        )

        class ResourceDummyS(plre.ResourceS):
            """Dummy class with no default adjoint decomp"""

            @classmethod
            def adjoint_resource_decomp(cls, **kwargs) -> list[GateCount]:
                """No default resources"""
                raise plre.ResourcesNotDefined

        op = ResourceDummyS()  # no default_adjoint_decomp defined
        adj_op = plre.ResourceAdjoint(op)
        expected_res = [
            GateCount(
                plre.resource_rep(
                    plre.ResourceAdjoint,
                    {"base_cmpr_op": plre.resource_rep(plre.ResourceT)},
                ),
                2,
            )
        ]
        assert adj_op.resource_decomp(**adj_op.resource_params) == expected_res

    @pytest.mark.parametrize(
        "base_op",
        (
            plre.ResourceS(),
            plre.ResourceRZ(precision=1e-3),
            plre.ResourceCNOT(wires=["ctrl", "trgt"]),
        ),
    )
    def test_adj_resource_decomp(self, base_op):
        """Test that the adjoint of this operator produces resources as expected."""
        adj_op = plre.ResourceAdjoint(base_op)
        adj_adj_op = plre.ResourceAdjoint(adj_op)

        expected_res = [GateCount(base_op.resource_rep_from_op())]
        assert adj_adj_op.resource_decomp(**adj_adj_op.resource_params) == expected_res


class TestResourcePow:
    """Tests for the Pow resource Op"""

    @pytest.mark.parametrize("z", (0, 1, 2, 3))
    @pytest.mark.parametrize(
        "base_type, base_args",
        (
            (plre.ResourceS, {}),
            (plre.ResourceRZ, {"precision": 1e-3}),
            (plre.ResourceCNOT, {"wires": ["ctrl", "trgt"]}),
        ),
    )
    def test_init(self, z, base_type, base_args):
        """Test that the operator is instantiated correctly"""
        with AnnotatedQueue() as q:
            base_op = base_type(**base_args)
            pow_base_op = plre.ResourcePow(base_op, z)

        assert base_op not in q.queue
        assert pow_base_op.num_wires == base_op.num_wires
        assert base_op.wires == pow_base_op.wires

    def test_resource_decomp(self):
        """Test that we can obtain the resources as expected"""
        op = plre.ResourceX()  # has default_pow_decomp defined
        z_and_expected_res = (
            (0, [GateCount(plre.resource_rep(plre.ResourceIdentity()))]),
            (1, [GateCount(op.resource_rep_from_op())]),
            (2, op.pow_resource_decomp(2, **op.resource_params)),
            (3, op.pow_resource_decomp(3, **op.resource_params)),
        )

        for z, res in z_and_expected_res:
            pow_op = plre.ResourcePow(op, z)
            assert pow_op.resource_decomp(**pow_op.resource_params) == res

        class ResourceDummyX(plre.ResourceX):
            """Dummy class with no default pow decomp"""

            @classmethod
            def pow_resource_decomp(cls, pow_z, **kwargs) -> list[GateCount]:
                """No default resources"""
                raise plre.ResourcesNotDefined

        op = ResourceDummyX()  # no default_pow_decomp defined
        pow_op = plre.ResourcePow(op, 7)
        expected_res = [GateCount(op.resource_rep_from_op(), 7)]
        assert pow_op.resource_decomp(**pow_op.resource_params) == expected_res

    @pytest.mark.parametrize("z", (0, 1, 2, 3))
    @pytest.mark.parametrize(
        "base_op",
        (
            plre.ResourceS(),
            plre.ResourceRZ(precision=1e-3),
            plre.ResourceCNOT(wires=["ctrl", "trgt"]),
        ),
    )
    def test_pow_resource_decomp(self, base_op, z):
        """Test that the power of this operator produces resources as expected."""
        pow_op = plre.ResourcePow(base_op, z)
        pow_pow_op = plre.ResourcePow(pow_op, z=5)

        expected_res = [
            GateCount(
                plre.ResourcePow.resource_rep(
                    base_op.resource_rep_from_op(),
                    z=5 * z,
                )
            )
        ]
        assert pow_pow_op.resource_decomp(**pow_pow_op.resource_params) == expected_res


class TestResourceControlled:
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
            (plre.ResourceS, {}),
            (plre.ResourceRZ, {"precision": 1e-3}),
            (plre.ResourceCNOT, {"wires": ["ctrl", "trgt"]}),
        ),
    )
    def test_init(self, ctrl_wires, ctrl_values, base_type, base_args):
        """Test that the operator is instantiated correctly"""
        with AnnotatedQueue() as q:
            base_op = base_type(**base_args)
            ctrl_base_op = plre.ResourceControlled(base_op, ctrl_wires, ctrl_values)

        assert base_op not in q.queue
        assert ctrl_base_op.num_wires == base_op.num_wires + ctrl_wires

    def test_resource_decomp(self):
        """Test that we can obtain the resources as expected"""
        op = plre.ResourceZ()  # has default_ctrl_decomp defined
        ctrl_params_and_expected_res = (
            ((1, 0), op.controlled_resource_decomp(1, 0, **op.resource_params)),
            ((2, 0), op.controlled_resource_decomp(2, 0, **op.resource_params)),
            ((3, 2), op.controlled_resource_decomp(3, 2, **op.resource_params)),
        )

        for (ctrl_wires, ctrl_values), res in ctrl_params_and_expected_res:
            ctrl_op = plre.ResourceControlled(op, ctrl_wires, ctrl_values)
            assert ctrl_op.resource_decomp(**ctrl_op.resource_params) == res

        class ResourceDummyZ(plre.ResourceZ):
            """Dummy class with no default ctrl decomp"""

            @classmethod
            def controlled_resource_decomp(
                cls, ctrl_num_ctrl_wires, ctrl_num_ctrl_values, **kwargs
            ) -> list[GateCount]:
                """No default resources"""
                raise plre.ResourcesNotDefined

        op = ResourceDummyZ()  # no default_ctrl_decomp defined
        ctrl_op = plre.ResourceControlled(op, num_ctrl_wires=3, num_ctrl_values=2)
        expected_res = [
            GateCount(plre.resource_rep(plre.ResourceX), 4),
            GateCount(
                plre.ResourceControlled.resource_rep(
                    plre.resource_rep(plre.ResourceS),
                    num_ctrl_wires=3,
                    num_ctrl_values=0,
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
            plre.ResourceS(),
            plre.ResourceRZ(precision=1e-3),
            plre.ResourceCNOT(wires=["ctrl", "trgt"]),
        ),
    )
    def test_ctrl_resource_decomp(self, ctrl_wires, ctrl_values, base_op):
        """Test that the control of this operator produces resources as expected."""
        ctrl_op = plre.ResourceControlled(base_op, ctrl_wires, ctrl_values)
        ctrl_ctrl_op = plre.ResourceControlled(ctrl_op, num_ctrl_wires=2, num_ctrl_values=1)

        expected_res = [
            GateCount(
                plre.ResourceControlled.resource_rep(
                    base_op.resource_rep_from_op(),
                    num_ctrl_wires=ctrl_wires + 2,
                    num_ctrl_values=ctrl_values + 1,
                )
            )
        ]
        assert ctrl_ctrl_op.resource_decomp(**ctrl_ctrl_op.resource_params) == expected_res


class TestResourceProd:
    """Tests for the Prod resource Op"""

    def test_init(self):
        """Test that the operator is instantiated correctly"""
        with AnnotatedQueue() as q:
            ops = [
                plre.ResourceX(wires=0),
                plre.ResourceCZ(wires=[1, 2]),
                (plre.ResourceHadamard(), 4),
                plre.ResourceRX(precision=1e-4, wires=3),
                (plre.ResourceCNOT(), 2),
            ]
            prod_op = plre.ResourceProd(res_ops=ops)

        for op in ops:
            if isinstance(op, tuple):
                op = op[0]
            assert op not in q.queue

        assert prod_op.num_wires == 4
        assert prod_op.wires == Wires([0, 1, 2, 3])

    def test_resource_decomp(self):
        """Test that we can obtain the resources as expected"""
        ops = [
            plre.ResourceX(wires=0),
            plre.ResourceCZ(wires=[1, 2]),
            (plre.ResourceHadamard(), 4),
            plre.ResourceRX(precision=1e-4, wires=3),
            (plre.ResourceCNOT(), 2),
        ]
        prod_op = plre.ResourceProd(res_ops=ops)

        expected_res = [
            GateCount(plre.resource_rep(plre.ResourceX)),
            GateCount(plre.resource_rep(plre.ResourceCZ)),
            GateCount(plre.resource_rep(plre.ResourceHadamard), 4),
            GateCount(plre.resource_rep(plre.ResourceRX, {"precision": 1e-4})),
            GateCount(plre.resource_rep(plre.ResourceCNOT), 2),
        ]
        assert prod_op.resource_decomp(**prod_op.resource_params) == expected_res


class TestResourceChangeBasisOp:
    """Tests for the ChangeBasis resource Op"""

    def test_init(self):
        """Test that the operator is instantiated correctly"""
        with AnnotatedQueue() as q:
            compute_op = plre.ResourceS(wires=0)
            base_op = plre.ResourceProd(((plre.ResourceZ(), 3),), wires=[0, 1, 2])
            uncompute_op = plre.ResourcePow(plre.ResourceT(wires=0), 6)

            cb_op = plre.ResourceChangeBasisOp(compute_op, base_op, uncompute_op)

        for op in [compute_op, base_op, uncompute_op]:
            assert op not in q.queue

        assert cb_op.num_wires == 3
        assert cb_op.wires == Wires([0, 1, 2])

    def test_resource_decomp(self):
        """Test that we can obtain the resources as expected"""
        compute_op = plre.ResourceS(wires=0)
        base_op = plre.ResourceProd([(plre.ResourceZ(), 3)], wires=[0, 1, 2])
        uncompute_op = plre.ResourcePow(plre.ResourceT(wires=0), 6)

        cb_op = plre.ResourceChangeBasisOp(compute_op, base_op, uncompute_op)
        expected_res = [
            GateCount(plre.resource_rep(plre.ResourceS)),
            GateCount(
                plre.resource_rep(
                    plre.ResourceProd,
                    {
                        "cmpr_factors_and_counts": ((plre.ResourceZ.resource_rep(), 3),),
                        "num_wires": 3,
                    },
                ),
            ),
            GateCount(
                plre.resource_rep(
                    plre.ResourcePow,
                    {
                        "base_cmpr_op": (plre.ResourceT.resource_rep()),
                        "z": 6,
                    },
                ),
            ),
        ]

        assert cb_op.resource_decomp(**cb_op.resource_params) == expected_res
