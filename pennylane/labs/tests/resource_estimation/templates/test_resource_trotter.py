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
Test the Resource classes associated with TrotterProduct and TrotterizedQfunc
"""
import pytest

import pennylane as qml
import pennylane.labs.resource_estimation as re
from pennylane.labs.resource_estimation.ops.op_math.symbolic import (
    _extract_exp_params,
    _resources_from_pauli_sentence,
)
from pennylane.operation import Operation
from pennylane.ops.op_math import LinearCombination, SProd
from pennylane.pauli import PauliSentence, PauliWord

# pylint: disable=no-self-use


class DummyOp(re.ResourceOperator, Operation):

    def __init__(self, a, b, wires=[0]):
        self.a = a
        self.b = b
        super().__init__(wires=wires)

    @staticmethod
    def _resource_decomp(a, b, **kwargs):
        h = re.ResourceHadamard.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        return {h: a, cnot: b}

    @classmethod
    def exp_resource_decomp(cls, coeff, num_steps, a, b):
        return cls.resources(a + 1, b + 1)

    def resource_params(self) -> dict:
        return {"a": self.a, "b": self.b}

    @classmethod
    def resource_rep(cls, a, b):
        return re.CompressedResourceOp(cls, {"a": a, "b": b})


class TestResourceTrotterProduct:
    """Test the ResourceTrotterProduct class"""

    op_data = (
        re.ResourceTrotterProduct(
            qml.sum(
                re.ResourceX(0),
                DummyOp(a=1, b=2),
                re.ResourceProd(re.ResourceZ(0), re.ResourceZ(1)),
            ),
            time=0.1,
            check_hermitian=False,
        ),
        re.ResourceTrotterProduct(
            LinearCombination(
                coeffs=[0.1, 0.2],
                observables=[
                    re.ResourceX(0),
                    re.ResourceProd(re.ResourceZ(0), re.ResourceZ(1)),
                ],
            ),
            time=0.1,
            n=5,
            order=2,
        ),
        re.ResourceTrotterProduct(
            LinearCombination(
                coeffs=[1.2, -3.4],
                observables=[re.ResourceZ(0), re.ResourceX(1)],
            ),
            time=0.1,
            n=10,
            order=4,
        ),
    )

    resource_params_data = (
        {
            "n": 1,
            "order": 1,
            "first_order_expansion": [
                re.ResourceExp.resource_rep(re.ResourceX, {}, qml.X(0).pauli_rep, 1j, 1),
                re.ResourceExp.resource_rep(DummyOp, {"a": 1, "b": 2}, None, 1j, 1),
                re.ResourceExp.resource_rep(
                    re.ResourceProd,
                    {"cmpr_factors": (re.ResourceZ.resource_rep(), re.ResourceZ.resource_rep())},
                    (qml.Z(0) @ qml.Z(1)).pauli_rep,
                    1j,
                    1,
                ),
            ],
        },
        {
            "n": 5,
            "order": 2,
            "first_order_expansion": [
                re.ResourceExp.resource_rep(SProd, {}, (0.1 * qml.X(0)).pauli_rep, 1j, 1),
                re.ResourceExp.resource_rep(
                    SProd, {}, (0.2 * qml.Z(0) @ qml.Z(1)).pauli_rep, 1j, 1
                ),
            ],
        },
        {
            "n": 10,
            "order": 4,
            "first_order_expansion": [
                re.ResourceExp.resource_rep(SProd, {}, (1.2 * qml.Z(0)).pauli_rep, 1j, 1),
                re.ResourceExp.resource_rep(SProd, {}, (-3.4 * qml.X(1)).pauli_rep, 1j, 1),
            ],
        },
    )

    resource_data = (
        {
            re.ResourceExp.resource_rep(re.ResourceX, {}, qml.X(0).pauli_rep, 1j, 1): 1,
            re.ResourceExp.resource_rep(DummyOp, {"a": 1, "b": 2}, None, 1j, 1): 1,
            re.ResourceExp.resource_rep(
                re.ResourceProd,
                {"cmpr_factors": (re.ResourceZ.resource_rep(), re.ResourceZ.resource_rep())},
                (qml.Z(0) @ qml.Z(1)).pauli_rep,
                1j,
                1,
            ): 1,
        },
        {
            re.ResourceExp.resource_rep(SProd, {}, (0.1 * qml.X(0)).pauli_rep, 1j, 1): 6,
            re.ResourceExp.resource_rep(SProd, {}, (0.2 * qml.Z(0) @ qml.Z(1)).pauli_rep, 1j, 1): 5,
        },
        {
            re.ResourceExp.resource_rep(SProd, {}, (1.2 * qml.Z(0)).pauli_rep, 1j, 1): 51,
            re.ResourceExp.resource_rep(SProd, {}, (-3.4 * qml.X(1)).pauli_rep, 1j, 1): 50,
        },
    )

    @pytest.mark.parametrize("op, expected_res", zip(op_data, resource_data))
    def test_resources(self, op, expected_res):
        """Test the resources method returns the correct dictionary"""
        op_rep = op.resource_rep_from_op()
        computed_res = op_rep.op_type.resources(**op_rep.params)
        assert computed_res == expected_res

    @pytest.mark.parametrize("op, expected_params", zip(op_data, resource_params_data))
    def test_resource_params(self, op, expected_params):
        """Test that the resource params are correct"""
        assert op.resource_params() == expected_params

    @pytest.mark.parametrize("params", resource_params_data)
    def test_resource_rep(self, params):
        """Test the resource_rep returns the correct CompressedResourceOp"""
        expected = re.CompressedResourceOp(re.ResourceTrotterProduct, params)
        assert re.ResourceTrotterProduct.resource_rep(**params) == expected
