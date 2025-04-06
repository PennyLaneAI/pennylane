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
from pennylane.operation import Operation
from pennylane.ops.op_math import LinearCombination, SProd

# pylint: disable=no-self-use,arguments-differ


def qfunc_1(time, wires):
    """A quantum function which queues operations to be trotterized."""
    re.ResourceRX(1.23 * time, wires[0])
    re.ResourceRY(-4.5 * time, wires[0])


def qfunc_2(time, arg1, wires):
    """A quantum function which queues operations to be trotterized."""
    for w in wires:
        re.ResourceHadamard(w)

    re.ResourceQFT(wires)
    re.ResourceRX(arg1 * time, wires[0])


def qfunc_3(time, arg1, arg2, kwarg1=False, wires=None):
    """A quantum function which queues operations to be trotterized."""
    re.ResourceControlled(
        re.ResourceRot(arg1 * time, arg2 * time, time * (arg1 + arg2), wires=wires[1]),
        control_wires=wires[0],
    )
    if kwarg1:
        for w in wires:
            re.ResourceHadamard(w)


trotterized_qfunc_op_data = (
    re.ResourceTrotterizedQfunc(1, qfunc=qfunc_1, n=5, order=2, reverse=False, wires=[0]),
    re.ResourceTrotterizedQfunc(
        1, *(1.23,), qfunc=qfunc_2, n=10, order=2, reverse=False, wires=[0, 1, 2]
    ),
    re.ResourceTrotterizedQfunc(
        1,
        *(1.23, -4.56),
        qfunc=qfunc_3,
        n=10,
        order=4,
        reverse=False,
        wires=[0, 1],
        **{"kwarg1": True},
    ),
)


class DummyOp(re.ResourceOperator, Operation):
    """A Dummy ResourceOperator child class which implements the
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


class TestResourceTrotterProduct:
    """Test the ResourceTrotterProduct class"""

    op_data = (
        re.ResourceTrotterProduct(
            qml.sum(
                DummyOp(a=1, b=2),
                re.ResourceProd(re.ResourceZ(0), re.ResourceZ(1)),
                LinearCombination(
                    [1.2, 3.4, 5.6], [re.ResourceX(0), re.ResourceY(0), re.ResourceZ(1)]
                ),
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
                re.ResourceExp.resource_rep(DummyOp, {"a": 1, "b": 2}, None, 1j, 1),
                re.ResourceExp.resource_rep(
                    re.ResourceProd,
                    {"cmpr_factors": (re.ResourceZ.resource_rep(), re.ResourceZ.resource_rep())},
                    (qml.Z(0) @ qml.Z(1)).pauli_rep,
                    1j,
                    1,
                ),
                re.ResourceExp.resource_rep(
                    LinearCombination,
                    {},
                    (1.2 * qml.X(0) + 3.4 * qml.Y(0) + 5.6 * qml.Z(1)).pauli_rep,
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
            re.ResourceExp.resource_rep(DummyOp, {"a": 1, "b": 2}, None, 1j, 1): 1,
            re.ResourceExp.resource_rep(
                re.ResourceProd,
                {"cmpr_factors": (re.ResourceZ.resource_rep(), re.ResourceZ.resource_rep())},
                (qml.Z(0) @ qml.Z(1)).pauli_rep,
                1j,
                1,
            ): 1,
            re.ResourceExp.resource_rep(
                LinearCombination,
                {},
                (1.2 * qml.X(0) + 3.4 * qml.Y(0) + 5.6 * qml.Z(1)).pauli_rep,
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
        assert op.resource_params == expected_params

    @pytest.mark.parametrize("params", resource_params_data)
    def test_resource_rep(self, params):
        """Test the resource_rep returns the correct CompressedResourceOp"""
        expected = re.CompressedResourceOp(re.ResourceTrotterProduct, params)
        assert re.ResourceTrotterProduct.resource_rep(**params) == expected


class TestResourceTrotterizedQfunc:
    """Test the ResourceTrotterizedQfunc class"""

    resource_params_data = (
        {
            "n": 5,
            "order": 2,
            "qfunc_compressed_reps": (
                re.ResourceRX.resource_rep(),
                re.ResourceRY.resource_rep(),
            ),
        },
        {
            "n": 10,
            "order": 2,
            "qfunc_compressed_reps": (
                re.ResourceHadamard.resource_rep(),
                re.ResourceHadamard.resource_rep(),
                re.ResourceHadamard.resource_rep(),
                re.ResourceQFT.resource_rep(num_wires=3),
                re.ResourceRX.resource_rep(),
            ),
        },
        {
            "n": 10,
            "order": 4,
            "qfunc_compressed_reps": (
                re.ResourceControlled.resource_rep(re.ResourceRot, {}, 1, 0, 0),
                re.ResourceHadamard.resource_rep(),
                re.ResourceHadamard.resource_rep(),
            ),
        },
    )

    resource_data = (
        {
            re.ResourceRX.resource_rep(): 10,
            re.ResourceRY.resource_rep(): 10,
        },
        {
            re.ResourceHadamard.resource_rep(): 60,
            re.ResourceQFT.resource_rep(num_wires=3): 20,
            re.ResourceRX.resource_rep(): 20,
        },
        {
            re.ResourceControlled.resource_rep(re.ResourceRot, {}, 1, 0, 0): 100,
            re.ResourceHadamard.resource_rep(): 200,
        },
    )

    @pytest.mark.parametrize("op, expected_res", zip(trotterized_qfunc_op_data, resource_data))
    def test_resources(self, op, expected_res):
        """Test the resources method returns the correct dictionary"""
        op_rep = op.resource_rep_from_op()
        computed_res = op_rep.op_type.resources(**op_rep.params)
        assert computed_res == expected_res

    @pytest.mark.parametrize(
        "op, expected_params", zip(trotterized_qfunc_op_data, resource_params_data)
    )
    def test_resource_params(self, op, expected_params):
        """Test that the resource params are correct"""
        assert op.resource_params == expected_params

    @pytest.mark.parametrize("params", resource_params_data)
    def test_resource_rep(self, params):
        """Test the resource_rep returns the correct CompressedResourceOp"""
        expected = re.CompressedResourceOp(re.ResourceTrotterizedQfunc, params)
        assert re.ResourceTrotterizedQfunc.resource_rep(**params) == expected


@pytest.mark.parametrize(
    "qfunc, args_n_kwargs, hyperparams, expected_op",
    zip(
        (qfunc_1, qfunc_2, qfunc_3),
        (
            ((1,), {"wires": [0]}),
            ((1, 1.23), {"wires": [0, 1, 2]}),
            ((1, 1.23, -4.56), {"kwarg1": True, "wires": [0, 1]}),
        ),
        (
            {"n": 5, "order": 2},
            {"n": 10, "order": 2},
            {"n": 10, "order": 4},
        ),
        trotterized_qfunc_op_data,
    ),
)
def test_resource_trotterize(qfunc, args_n_kwargs, hyperparams, expected_op):
    """Test that the resource_trotterize wrapper function generates an instance of
    ResourceTrotterizedQfunc."""
    args, kwargs = args_n_kwargs
    assert qml.equal(
        re.resource_trotterize(qfunc=qfunc, **hyperparams)(*args, **kwargs), expected_op
    )
