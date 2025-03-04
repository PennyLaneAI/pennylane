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

import pennylane as qml
import pennylane.labs.resource_estimation as re
from pennylane.labs.resource_estimation.ops.op_math.symbolic import (
    _extract_exp_params,
    _resources_from_pauli_sentence,
)
from pennylane.operation import Operation
from pennylane.pauli import PauliSentence, PauliWord

# pylint: disable=protected-access,no-self-use,arguments-differ


class DummyOp(re.ResourceOperator, Operation):
    """Dummy ResourceOperator child class which implements the
    :code:`exp_resource_decomp` method."""

    def __init__(self, a, b, wires=(0,)):
        self.a = a
        self.b = b
        super().__init__(wires=wires)

    @staticmethod
    def _resource_decomp(a, b, **kwargs):
        h = re.ResourceHadamard.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        return {h: a, cnot: b}

    @classmethod
    def exp_resource_decomp(cls, coeff, num_steps, a, b):  # pylint: disable=unused-argument
        return cls.resources(a + 1, b + 1)

    @property
    def resource_params(self) -> dict:
        return {"a": self.a, "b": self.b}

    @classmethod
    def resource_rep(cls, a, b):
        return re.CompressedResourceOp(cls, {"a": a, "b": b})


z_pauli_rep = qml.Z(0).pauli_rep
lc_op = qml.ops.LinearCombination(
    [1.11, 0.12, -3.4, 5], [qml.X(0) @ qml.X(1), qml.Z(2), qml.Y(0) @ qml.Y(1), qml.I((0, 1, 2))]
)
exp_params_data = (
    (
        lc_op,
        {
            "base_class": qml.ops.LinearCombination,
            "base_params": {},
            "base_pauli_rep": lc_op.pauli_rep,
            "coeff": 1.2j,
            "num_steps": 3,
        },
    ),
    (
        re.ResourceQFT(range(10)),
        {
            "base_class": re.ResourceQFT,
            "base_params": {"num_wires": 10},
            "base_pauli_rep": None,
            "coeff": 1.2j,
            "num_steps": 3,
        },
    ),
)


class TestResourceAdjoint:
    """Tests for ResourceAdjoint"""

    adjoint_ops = [
        re.ResourceAdjoint(re.ResourceQFT([0, 1])),
        re.ResourceAdjoint(re.ResourceAdjoint(re.ResourceQFT([0, 1]))),
        re.ResourceAdjoint(re.ResourcePow(re.ResourceX(0), 5)),
    ]

    expected_params = [
        {"base_class": re.ResourceQFT, "base_params": {"num_wires": 2}},
        {
            "base_class": re.ResourceAdjoint,
            "base_params": {"base_class": re.ResourceQFT, "base_params": {"num_wires": 2}},
        },
        {
            "base_class": re.ResourcePow,
            "base_params": {"base_class": re.ResourceX, "base_params": {}, "z": 5},
        },
    ]

    @pytest.mark.parametrize("op, expected", zip(adjoint_ops, expected_params))
    def test_resource_params(self, op, expected):
        """Test that the resources are correct"""
        assert op.resource_params == expected

    expected_names = [
        "Adjoint(QFT(2))",
        "Adjoint(Adjoint(QFT(2)))",
        "Adjoint(Pow(X, 5))",
    ]

    @pytest.mark.parametrize("op, expected", zip(adjoint_ops, expected_names))
    def test_tracking_name(self, op, expected):
        """Test that the tracking name is correct"""
        name = op.tracking_name_from_op()
        assert name == expected

    @pytest.mark.parametrize(
        "nested_op, expected_op",
        [
            (
                re.ResourceAdjoint(re.ResourceAdjoint(re.ResourceQFT([0, 1, 2]))),
                re.ResourceQFT([0, 1, 2]),
            ),
            (
                re.ResourceAdjoint(
                    re.ResourceAdjoint(re.ResourceAdjoint(re.ResourceQFT([0, 1, 2])))
                ),
                re.ResourceAdjoint(re.ResourceQFT([0, 1, 2])),
            ),
            (
                re.ResourceAdjoint(
                    re.ResourceAdjoint(
                        re.ResourceAdjoint(re.ResourceAdjoint(re.ResourceQFT([0, 1, 2])))
                    )
                ),
                re.ResourceQFT([0, 1, 2]),
            ),
            (
                re.ResourceAdjoint(
                    re.ResourceAdjoint(
                        re.ResourceAdjoint(
                            re.ResourceAdjoint(re.ResourceAdjoint(re.ResourceQFT([0, 1, 2])))
                        )
                    )
                ),
                re.ResourceAdjoint(re.ResourceQFT([0, 1, 2])),
            ),
        ],
    )
    def test_nested_adjoints(self, nested_op, expected_op):
        """Test the resources of nested Adjoints."""
        assert re.get_resources(nested_op) == re.get_resources(expected_op)

    expected_resources = [
        re.Resources(gate_types={"Adjoint(QFT(2))": 1}, num_gates=1, num_wires=2),
        re.Resources(gate_types={"Adjoint(Adjoint(QFT(2)))": 1}, num_gates=1, num_wires=2),
        re.Resources(gate_types={"Adjoint(Pow(X, 5))": 1}, num_gates=1, num_wires=1),
    ]

    @pytest.mark.parametrize("op, expected", zip(adjoint_ops, expected_resources))
    def test_tracking(self, op, expected):
        """Test that adjoints can be tracked."""
        tracking_name = op.tracking_name_from_op()
        gate_set = {tracking_name}

        assert re.get_resources(op, gate_set=gate_set) == expected


class TestResourceControlled:
    """Tests for ResourceControlled"""

    controlled_ops = [
        re.ResourceControlled(re.ResourceQFT([0, 1]), control_wires=[2]),
        re.ResourceControlled(
            re.ResourceControlled(re.ResourceQFT([0, 1]), control_wires=[2]), control_wires=[3]
        ),
        re.ResourceControlled(re.ResourceQFT([0, 1]), control_wires=[2, 3], control_values=[0, 1]),
        re.ResourceControlled(
            re.ResourceAdjoint(re.ResourceQFT([0, 1])),
            control_wires=[2, 3],
            control_values=[0, 1],
            work_wires=[4],
        ),
    ]

    expected_params = [
        {
            "base_class": re.ResourceQFT,
            "base_params": {"num_wires": 2},
            "num_ctrl_wires": 1,
            "num_ctrl_values": 0,
            "num_work_wires": 0,
        },
        {
            "base_class": re.ResourceControlled,
            "base_params": {
                "base_class": re.ResourceQFT,
                "base_params": {"num_wires": 2},
                "num_ctrl_wires": 1,
                "num_ctrl_values": 0,
                "num_work_wires": 0,
            },
            "num_ctrl_wires": 1,
            "num_ctrl_values": 0,
            "num_work_wires": 0,
        },
        {
            "base_class": re.ResourceQFT,
            "base_params": {"num_wires": 2},
            "num_ctrl_wires": 2,
            "num_ctrl_values": 1,
            "num_work_wires": 0,
        },
        {
            "base_class": re.ResourceAdjoint,
            "base_params": {"base_class": re.ResourceQFT, "base_params": {"num_wires": 2}},
            "num_ctrl_wires": 2,
            "num_ctrl_values": 1,
            "num_work_wires": 1,
        },
    ]

    @pytest.mark.parametrize("op, expected", zip(controlled_ops, expected_params))
    def test_resource_params(self, op, expected):
        """Test that the resources are correct"""
        assert op.resource_params == expected

    expected_names = [
        "C(QFT(2),1,0,0)",
        "C(C(QFT(2),1,0,0),1,0,0)",
        "C(QFT(2),2,1,0)",
        "C(Adjoint(QFT(2)),2,1,1)",
    ]

    @pytest.mark.parametrize("op, expected", zip(controlled_ops, expected_names))
    def test_tracking_name(self, op, expected):
        """Test that the tracking name is correct"""
        name = op.tracking_name_from_op()
        assert name == expected

    @pytest.mark.parametrize(
        "nested_op, expected_op",
        [
            (
                re.ResourceControlled(
                    re.ResourceControlled(re.ResourceX(0), control_wires=[1]), control_wires=[2]
                ),
                re.ResourceToffoli([0, 1, 2]),
            ),
            (
                re.ResourceControlled(
                    re.ResourceControlled(re.ResourceX(0), control_wires=[1]), control_wires=[2]
                ),
                re.ResourceControlled(re.ResourceX(0), control_wires=[1, 2]),
            ),
        ],
    )
    def test_nested_controls(self, nested_op, expected_op):
        """Test the resources for nested Controlled operators."""
        assert re.get_resources(nested_op) == re.get_resources(expected_op)

    expected_resources = [
        re.Resources(gate_types={"C(QFT(2),1,0,0)": 1}, num_gates=1, num_wires=3),
        re.Resources(gate_types={"C(C(QFT(2),1,0,0),1,0,0)": 1}, num_gates=1, num_wires=4),
        re.Resources(gate_types={"C(QFT(2),2,1,0)": 1}, num_gates=1, num_wires=4),
        re.Resources(
            gate_types={"C(Adjoint(QFT(2)),2,1,1)": 1}, num_gates=1, num_wires=4
        ),  # PL does not count work wires for controlled operators
    ]

    @pytest.mark.parametrize("op, expected", zip(controlled_ops, expected_resources))
    def test_tracking(self, op, expected):
        """Test that adjoints can be tracked."""
        tracking_name = op.tracking_name_from_op()
        gate_set = {tracking_name}

        assert re.get_resources(op, gate_set=gate_set) == expected


class TestResourcePow:
    """Tests for ResourcePow"""

    pow_ops = [
        re.ResourcePow(re.ResourceQFT([0, 1]), 2),
        re.ResourcePow(re.ResourceAdjoint(re.ResourceQFT([0, 1])), 2),
        re.ResourcePow(re.ResourcePow(re.ResourceQFT([0, 1]), 2), 3),
    ]

    expected_params = [
        {"base_class": re.ResourceQFT, "base_params": {"num_wires": 2}, "z": 2},
        {
            "base_class": re.ResourceAdjoint,
            "base_params": {"base_class": re.ResourceQFT, "base_params": {"num_wires": 2}},
            "z": 2,
        },
        {
            "base_class": re.ResourcePow,
            "base_params": {"base_class": re.ResourceQFT, "base_params": {"num_wires": 2}, "z": 2},
            "z": 3,
        },
    ]

    @pytest.mark.parametrize("op, expected", zip(pow_ops, expected_params))
    def test_resource_params(self, op, expected):
        """Test that the resources are correct"""
        assert op.resource_params == expected

    expected_names = [
        "Pow(QFT(2), 2)",
        "Pow(Adjoint(QFT(2)), 2)",
        "Pow(Pow(QFT(2), 2), 3)",
    ]

    @pytest.mark.parametrize("op, expected", zip(pow_ops, expected_names))
    def test_tracking_name(self, op, expected):
        """Test that the tracking name is correct"""
        rep = op.resource_rep_from_op()
        name = rep.op_type.tracking_name(**rep.params)
        assert name == expected

    expected_resources = [
        re.Resources(gate_types={"Pow(QFT(2), 2)": 1}, num_gates=1, num_wires=2),
        re.Resources(gate_types={"Pow(Adjoint(QFT(2)), 2)": 1}, num_gates=1, num_wires=2),
        re.Resources(gate_types={"Pow(Pow(QFT(2), 2), 3)": 1}, num_gates=1, num_wires=2),
    ]

    @pytest.mark.parametrize("op, expected", zip(pow_ops, expected_resources))
    def test_tracking(self, op, expected):
        """Test that adjoints can be tracked."""
        tracking_name = op.tracking_name_from_op()
        gate_set = {tracking_name}

        assert re.get_resources(op, gate_set=gate_set) == expected

    @pytest.mark.parametrize(
        "nested_op, expected_op",
        [
            (
                re.ResourcePow(re.ResourcePow(re.ResourceQFT([0, 1]), 2), 2),
                re.ResourcePow(re.ResourceQFT([0, 1]), 4),
            ),
            (
                re.ResourcePow(re.ResourcePow(re.ResourcePow(re.ResourceQFT([0, 1]), 2), 2), 2),
                re.ResourcePow(re.ResourceQFT([0, 1]), 8),
            ),
            (
                re.ResourcePow(
                    re.ResourcePow(re.ResourcePow(re.ResourcePow(re.ResourceQFT([0, 1]), 2), 2), 2),
                    2,
                ),
                re.ResourcePow(re.ResourceQFT([0, 1]), 16),
            ),
        ],
    )
    def test_nested_pow(self, nested_op, expected_op):
        """Test the resources for nested Pow operators."""
        assert re.get_resources(nested_op) == re.get_resources(expected_op)


class TestResourceProd:
    """Test ResourceProd"""

    op_data = (
        re.ResourceProd(re.ResourceX(0), re.ResourceHadamard(1)),
        re.ResourceProd(
            re.ResourceQFT(wires=[0, 1, 2]),
            re.ResourceAdjoint(re.ResourceZ(0)),
            re.ResourceControlled(re.ResourcePhaseShift(1.23, 0), control_wires=["c1", "c2"]),
        ),
        re.ResourceProd(
            re.ResourceCNOT([0, 1]),
            re.ResourceExp(re.ResourceZ(0), 0.1j),
            re.ResourceProd(re.ResourceX(1), re.ResourceY(2)),
            re.ResourceZ(3),
        ),
    )

    resource_data = (
        {
            re.ResourceX.resource_rep(): 1,
            re.ResourceHadamard.resource_rep(): 1,
        },
        {
            re.ResourceQFT.resource_rep(3): 1,
            re.ResourceAdjoint.resource_rep(re.ResourceZ, {}): 1,
            re.ResourceControlled.resource_rep(re.ResourcePhaseShift, {}, 2, 0, 0): 1,
        },
        {
            re.ResourceCNOT.resource_rep(): 1,
            re.ResourceExp.resource_rep(re.ResourceZ, {}, z_pauli_rep, 0.1j, None): 1,
            re.ResourceProd.resource_rep(
                cmpr_factors=(re.ResourceX.resource_rep(), re.ResourceY.resource_rep())
            ): 1,
            re.ResourceZ.resource_rep(): 1,
        },
    )

    resource_params_data = (
        {
            "cmpr_factors": (
                re.ResourceX.resource_rep(),
                re.ResourceHadamard.resource_rep(),
            ),
        },
        {
            "cmpr_factors": (
                re.ResourceQFT.resource_rep(3),
                re.ResourceAdjoint.resource_rep(re.ResourceZ, {}),
                re.ResourceControlled.resource_rep(re.ResourcePhaseShift, {}, 2, 0, 0),
            ),
        },
        {
            "cmpr_factors": (
                re.ResourceCNOT.resource_rep(),
                re.ResourceExp.resource_rep(re.ResourceZ, {}, z_pauli_rep, 0.1j, None),
                re.ResourceProd.resource_rep(
                    cmpr_factors=(re.ResourceX.resource_rep(), re.ResourceY.resource_rep())
                ),
                re.ResourceZ.resource_rep(),
            ),
        },
    )

    @pytest.mark.parametrize(
        "op, params, expected_res", zip(op_data, resource_params_data, resource_data)
    )
    def test_resources(self, op, params, expected_res):
        """Test the resources method returns the correct dictionary"""
        res_from_op = op.resources(**op.resource_params)
        res_from_func = re.ResourceProd.resources(**params)

        assert res_from_op == expected_res
        assert res_from_func == expected_res

    @pytest.mark.parametrize("op, expected_params", zip(op_data, resource_params_data))
    def test_resource_params(self, op, expected_params):
        """Test that the resource params are correct"""
        assert op.resource_params == expected_params

    @pytest.mark.parametrize("expected_params", resource_params_data)
    def test_resource_rep(self, expected_params):
        """Test the resource_rep returns the correct CompressedResourceOp"""
        expected = re.CompressedResourceOp(re.ResourceProd, expected_params)
        assert re.ResourceProd.resource_rep(**expected_params) == expected


class TestResourceExp:
    """Test for ResourceExp"""

    @pytest.mark.parametrize("op, expected", exp_params_data)
    def test_resource_params(self, op, expected):
        """Test that the resource_params method produces the expected parameters."""
        exp_op = re.ResourceExp(op, 1.2j, num_steps=3)
        extracted_params = exp_op.resource_params
        assert extracted_params == expected

    @pytest.mark.parametrize("op, expected_params", exp_params_data)
    def test_resource_rep(self, op, expected_params):
        """Test that the resource_rep method produces the correct compressed representation."""
        exp_op = re.ResourceExp(op, 1.2j, num_steps=3)
        computed_rep = exp_op.resource_rep(**expected_params)
        expected_rep = re.CompressedResourceOp(
            re.ResourceExp, expected_params, name=exp_op.tracking_name_from_op()
        )

        assert expected_rep == computed_rep

    exp_res_data = (
        (
            re.ResourceExp(lc_op, 1.5j),
            {
                re.ResourcePauliRot.resource_rep("XX"): 1,
                re.ResourcePauliRot.resource_rep("YY"): 1,
                re.ResourcePauliRot.resource_rep("Z"): 1,
                re.ResourcePauliRot.resource_rep(""): 1,
            },
        ),
        (
            re.ResourceExp(DummyOp(2, 3, wires=[1, 2, 3]), 0.1j, num_steps=5),
            {
                re.ResourceHadamard.resource_rep(): 3,
                re.ResourceCNOT.resource_rep(): 4,
            },
        ),
    )

    @pytest.mark.parametrize("op, expected_resources", exp_res_data)
    def test_resources_decomp(self, op, expected_resources):
        """Test that the _resources_decomp method works as expected."""
        computed_resources = op._resource_decomp(**op.resource_params)
        assert computed_resources == expected_resources

    @pytest.mark.parametrize(
        "op, z, expected_resources",
        (
            (
                re.ResourceExp(lc_op, 1.5j),
                1,
                {re.ResourceExp.resource_rep(type(lc_op), {}, lc_op.pauli_rep, 1.5j, None): 1},
            ),
            (
                re.ResourceExp(lc_op, 1.5j),
                2,
                {re.ResourceExp.resource_rep(type(lc_op), {}, lc_op.pauli_rep, 3j, None): 1},
            ),
            (
                re.ResourceExp(lc_op, 1.5j),
                7,
                {re.ResourceExp.resource_rep(type(lc_op), {}, lc_op.pauli_rep, 10.5j, None): 1},
            ),
            (
                re.ResourceExp(DummyOp(2, 3, wires=[1, 2, 3]), 0.1j, num_steps=5),
                1,
                {re.ResourceExp.resource_rep(DummyOp, {"a": 2, "b": 3}, None, 0.1j, 5): 1},
            ),
            (
                re.ResourceExp(DummyOp(2, 3, wires=[1, 2, 3]), 0.1j, num_steps=5),
                4,
                {re.ResourceExp.resource_rep(DummyOp, {"a": 2, "b": 3}, None, 0.4j, 5): 1},
            ),
            (
                re.ResourceExp(DummyOp(2, 3, wires=[1, 2, 3]), 0.1j, num_steps=5),
                8,
                {re.ResourceExp.resource_rep(DummyOp, {"a": 2, "b": 3}, None, 0.8j, 5): 1},
            ),
        ),
    )
    def test_pow_resources(self, op, z, expected_resources):
        """Test that the pow resource decomp method works as expected."""
        params = op.resource_params
        computed_resources = op.pow_resource_decomp(z, **params)
        assert computed_resources == expected_resources


@pytest.mark.parametrize("base_op, expected_params", exp_params_data)
def test_extract_exp_params(base_op, expected_params):
    """Test the private _extract_exp_params method behaves as expected"""
    extracted_params = _extract_exp_params(base_op, scalar=1.2j, num_steps=3)
    assert extracted_params == expected_params


def test_extract_exp_params_raises_error():
    """Test that the private _extract_exp_params method raises an error if the base operator
    isnt compatible with ResourceExp."""
    with pytest.raises(ValueError, match="Cannot obtain resources for the exponential of"):
        _ = _extract_exp_params(qml.QFT(range(10)), 1j, 5)


@pytest.mark.parametrize(
    "ps, expected_res",
    (
        (PauliSentence({}), {}),
        (
            PauliSentence(
                {
                    PauliWord({0: "I", 2: "I", 3: "I"}): 0.12,
                    PauliWord({0: "X", 2: "I", 3: "I"}): -3.4,
                    PauliWord({0: "I", 2: "Y", 3: "I"}): 56,
                    PauliWord({0: "I", 2: "I", 3: "Z"}): 0.78,
                }
            ),
            {
                re.ResourcePauliRot.resource_rep(""): 1,
                re.ResourcePauliRot.resource_rep("X"): 1,
                re.ResourcePauliRot.resource_rep("Y"): 1,
                re.ResourcePauliRot.resource_rep("Z"): 1,
            },
        ),
        (
            PauliSentence(
                {
                    PauliWord({0: "X", 2: "X", 3: "X"}): 0.12,
                    PauliWord({0: "Y", 2: "Y", 3: "Y"}): -3.4,
                    PauliWord({0: "X", 2: "Y", 3: "Z"}): 56,
                }
            ),
            {
                re.ResourcePauliRot.resource_rep("XXX"): 1,
                re.ResourcePauliRot.resource_rep("YYY"): 1,
                re.ResourcePauliRot.resource_rep("XYZ"): 1,
            },
        ),
    ),
)
def test_resources_from_pauli_sentence(ps, expected_res):
    """Test that the private function resources_from_pauli_sentence works correcty"""
    extracted_res = _resources_from_pauli_sentence(ps)
    assert extracted_res == expected_res
