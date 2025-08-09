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

"""Unit tests for the data structures used for resource estimation in the decomposition module."""

import pytest

import pennylane as qml
from pennylane.decomposition.resources import (
    CompressedResourceOp,
    Resources,
    adjoint_resource_rep,
    controlled_resource_rep,
    custom_ctrl_op_to_base,
    pow_resource_rep,
    resource_rep,
)


@pytest.mark.unit
class TestResources:
    """Unit tests for the Resources data structure."""

    def test_resource_initialize(self):
        """Tests initializing a Resources object."""
        resources = Resources()
        assert resources.num_gates == 0
        assert resources.gate_counts == {}
        assert resources.weighted_cost == 0.0

    def test_negative_gate_counts(self):
        """Tests that an error is raised if the gate count is negative."""
        with pytest.raises(AssertionError):
            Resources(
                gate_counts={
                    CompressedResourceOp(qml.RX, {}): 2,
                    CompressedResourceOp(qml.RZ, {}): -1,
                }
            )

    def test_negative_weighted_cost(self):
        """Tests that an error is raised if the cost is negative."""
        with pytest.raises(AssertionError):
            Resources(
                gate_counts={
                    CompressedResourceOp(qml.RX, {}): 2,
                },
                weighted_cost=-2.0,
            )

    def test_add_resources(self):
        """Tests adding two Resources objects."""

        resources1 = Resources(
            gate_counts={CompressedResourceOp(qml.RX, {}): 2, CompressedResourceOp(qml.RZ, {}): 1},
            weighted_cost=6.0,
        )
        resources2 = Resources(
            gate_counts={CompressedResourceOp(qml.RX, {}): 1, CompressedResourceOp(qml.RY, {}): 1},
            weighted_cost=2.0,
        )

        resources = resources1 + resources2
        assert resources.num_gates == 5
        assert resources.gate_counts == {
            CompressedResourceOp(qml.RX, {}): 3,
            CompressedResourceOp(qml.RZ, {}): 1,
            CompressedResourceOp(qml.RY, {}): 1,
        }
        assert resources.weighted_cost == 8.0

    def test_mul_resource_with_scalar(self):
        """Tests multiplying a Resources object with a scalar."""

        resources = Resources(
            gate_counts={CompressedResourceOp(qml.RX, {}): 2, CompressedResourceOp(qml.RZ, {}): 1},
            weighted_cost=2.0,
        )

        resources = resources * 2
        assert resources.num_gates == 6
        assert resources.gate_counts == {
            CompressedResourceOp(qml.RX, {}): 4,
            CompressedResourceOp(qml.RZ, {}): 2,
        }
        assert resources.weighted_cost == 4

    def test_repr(self):
        """Tests the __repr__ of a Resources object."""

        resources = Resources(
            {CompressedResourceOp(qml.RX, {}): 2, CompressedResourceOp(qml.RZ, {}): 1}, 5.0
        )
        assert repr(resources) == "<num_gates=3, gate_counts={RX: 2, RZ: 1}, weighted_cost=5.0>"


class DummyOp(qml.operation.Operator):  # pylint: disable=too-few-public-methods
    resource_keys = {"foo", "bar"}


@pytest.mark.unit
class TestCompressedResourceOp:
    """Unit tests for the CompressedResourceOp data structure."""

    def test_initialization(self):
        """Tests creating a CompressedResourceOp object."""

        op = CompressedResourceOp(qml.QFT, {"num_wires": 5})
        assert op.op_type is qml.QFT
        assert op.params == {"num_wires": 5}

        op = CompressedResourceOp(qml.RX)
        assert op.op_type is qml.RX
        assert op.params == {}

    def test_invalid_op_type(self):
        """Tests that an error is raised if the op is invalid."""

        with pytest.raises(TypeError, match="op_type must be an Operator type"):
            CompressedResourceOp("RX", {})

        with pytest.raises(TypeError, match="op_type must be a subclass of Operator"):
            CompressedResourceOp(int, {})

    def test_hash(self):
        """Tests that a CompressedResourceOp object is hashable."""

        op = CompressedResourceOp(qml.RX, {})
        assert isinstance(hash(op), int)

        op = CompressedResourceOp(qml.QFT, {"num_wires": 5})
        assert isinstance(hash(op), int)

        op = CompressedResourceOp(
            qml.ops.Controlled,
            {
                "base_class": qml.QFT,
                "base_params": {"num_wires": 5},  # nested dictionary in params
                "num_control_wires": 1,
                "num_zero_control_values": 1,
                "num_work_wires": 1,
            },
        )
        assert isinstance(hash(op), int)

    def test_hash_unhashable_keys(self):
        """Tests that a CompressedResourceOp is hashable when the params contain unhashable keys."""

        op = CompressedResourceOp(
            qml.ops.Exp,
            {
                "base_class": qml.ops.LinearCombination,
                "base_params": {},
                "base_pauli_rep": qml.Hamiltonian(
                    [1.11, 0.12, -3.4, 5],
                    [qml.X(0) @ qml.X(1), qml.Z(2), qml.Y(0) @ qml.Y(1), qml.I((0, 1, 2))],
                ).pauli_rep,
                "coeff": 1.2j,
                "num_steps": 3,
            },
        )
        assert isinstance(hash(op), int)

    def test_hash_list_params(self):
        """Tests when the resource params contains a list."""

        class CustomOp(qml.operation.Operator):  # pylint: disable=too-few-public-methods

            resource_keys = {"foo", "bar"}

            @property
            def resource_params(self) -> dict:
                return {"foo": [1, 2, 3], "bar": [1, 2, [3, 4, 5]]}

        op = CompressedResourceOp(CustomOp, {"foo": [1, 2, 3], "bar": [1, 2, [3, 4, 5]]})
        assert isinstance(hash(op), int)

    def test_same_params_same_hash(self):
        """Tests that two ops with the same params have the same hash."""

        op1 = CompressedResourceOp(qml.RX, {"a": 1, "b": 2})
        op2 = CompressedResourceOp(qml.RX, {"b": 2, "a": 1})
        assert hash(op1) == hash(op2)

    def test_empty_params_same_hash(self):
        """Tests that CompressedResourceOp objects initialized with or without empty
        parameters have the same hash."""
        op1 = CompressedResourceOp(qml.RX)
        op2 = CompressedResourceOp(qml.RX, {})
        assert hash(op1) == hash(op2)

    def test_different_params_different_hash(self):
        """Tests that CompressedResourceOp objects initialized with different parameters
        have different hashes."""
        op1 = CompressedResourceOp(qml.MultiRZ, {"num_wires": 5})
        op2 = CompressedResourceOp(qml.MultiRZ, {"num_wires": 6})
        assert hash(op1) != hash(op2)

    def test_equal(self):
        """Tests comparing two CompressedResourceOp objects."""

        op1 = CompressedResourceOp(qml.RX, {})
        op2 = CompressedResourceOp(qml.RX, {})
        assert op1 == op2

        op1 = CompressedResourceOp(qml.RX, {})
        op2 = CompressedResourceOp(qml.RZ, {})
        assert op1 != op2

        op1 = CompressedResourceOp(qml.MultiRZ, {"num_wires": 3})
        op2 = CompressedResourceOp(qml.MultiRZ, {"num_wires": 3})
        assert op1 == op2

        op1 = CompressedResourceOp(qml.MultiRZ, {"num_wires": 5})
        op2 = CompressedResourceOp(qml.MultiRZ, {"num_wires": 6})
        assert op1 != op2

    def test_repr(self):
        """Tests the repr defined for debugging purposes."""

        op = CompressedResourceOp(qml.RX, {})
        assert repr(op) == "RX"

        op = CompressedResourceOp(qml.MultiRZ, {"num_wires": 5})
        assert repr(op) == "MultiRZ(num_wires=5)"

        op = CompressedResourceOp(DummyOp, {"foo": 2, "bar": 1})
        assert repr(op) == "DummyOp(foo=2, bar=1)"

    @pytest.mark.parametrize(
        "op, expected_name",
        [
            (resource_rep(qml.RX), "RX"),
            (adjoint_resource_rep(qml.RX, {}), "Adjoint(RX)"),
            (controlled_resource_rep(qml.T, {}, 1, 0, 0), "C(T)"),
            (pow_resource_rep(qml.RX, {}, 2), "Pow(RX)"),
        ],
    )
    def test_name(self, op, expected_name):
        """Tests the name property of a CompressedResourceOp object."""
        assert op.name == expected_name


@pytest.mark.unit
class TestResourceRep:
    """Tests the resource_rep utility function."""

    def test_resource_rep_fail(self):
        """Tests that an error is raised if the op is invalid."""

        with pytest.raises(TypeError, match="op_type must be a type of Operator"):
            resource_rep(int)

        class CustomOp(qml.operation.Operator):  # pylint: disable=too-few-public-methods

            resource_keys = {}

            @property
            def resource_params(self) -> dict:
                return {}

        with pytest.raises(TypeError, match="CustomOp.resource_keys must be a set"):
            resource_rep(CustomOp)

    def test_params_mismatch(self):
        """Tests that an error is raised when parameters are missing."""

        with pytest.raises(TypeError, match="Missing keyword arguments"):
            resource_rep(DummyOp, foo=2)

        with pytest.raises(TypeError, match="Unexpected keyword arguments"):
            resource_rep(DummyOp, foo=2, bar=1, hello=3)

    def test_resource_rep(self):
        """Tests creating a resource rep."""

        assert resource_rep(DummyOp, foo=2, bar=1) == CompressedResourceOp(
            DummyOp, {"foo": 2, "bar": 1}
        )


@pytest.mark.unit
class TestControlledResourceRep:
    """Tests the controlled_resource_rep function."""

    def test_controlled_resource_rep(self):
        """Tests creating the resource rep of a general controlled operation."""

        rep = controlled_resource_rep(DummyOp, {"foo": 2, "bar": 1}, 2, 1, 1)
        assert rep == CompressedResourceOp(
            qml.ops.Controlled,
            {
                "base_class": DummyOp,
                "base_params": {"foo": 2, "bar": 1},
                "num_control_wires": 2,
                "num_zero_control_values": 1,
                "num_work_wires": 1,
                "work_wire_type": "borrowed",
            },
        )

    def test_controlled_resource_rep_flatten(self):
        """Tests that nested controlled ops are flattened."""

        rep = controlled_resource_rep(
            qml.ops.Controlled,
            {
                "base_class": qml.CRX,
                "base_params": {},
                "num_control_wires": 2,
                "num_zero_control_values": 1,
                "num_work_wires": 1,
                "work_wire_type": "borrowed",
            },
            1,
            1,
            1,
        )
        assert rep == CompressedResourceOp(
            qml.ops.Controlled,
            {
                "base_class": qml.RX,
                "base_params": {},
                "num_control_wires": 4,
                "num_zero_control_values": 2,
                "num_work_wires": 2,
                "work_wire_type": "borrowed",
            },
        )

    def test_controlled_resource_op_base_param_mismatch(self):
        """Tests that an error is raised when base op and base params mismatch."""

        with pytest.raises(TypeError, match="Missing keyword arguments"):
            controlled_resource_rep(DummyOp, {}, 1, 1, 1)

    def test_controlled_resource_op_flatten_x(self):
        """Tests that nested X-based controlled ops are flattened."""

        rep = controlled_resource_rep(
            qml.ops.Controlled,
            {
                "base_class": qml.MultiControlledX,
                "base_params": {
                    "num_control_wires": 2,
                    "num_zero_control_values": 1,
                    "num_work_wires": 1,
                    "work_wire_type": "zeroed",
                },
                "num_control_wires": 1,
                "num_zero_control_values": 1,
                "num_work_wires": 1,
                "work_wire_type": "zeroed",
            },
            1,
            1,
            1,
            "zeroed",
        )
        assert rep == CompressedResourceOp(
            qml.ops.MultiControlledX,
            {
                "num_control_wires": 4,
                "num_zero_control_values": 3,
                "num_work_wires": 3,
                "work_wire_type": "zeroed",
            },
        )

    def test_controlled_qubit_unitary(self):
        """Tests that a controlled QubitUnitary is a ControlledQubitUnitary."""

        rep = controlled_resource_rep(
            qml.ops.Controlled,
            {
                "base_class": qml.QubitUnitary,
                "base_params": {"num_wires": 2},
                "num_control_wires": 1,
                "num_zero_control_values": 1,
                "num_work_wires": 1,
                "work_wire_type": "zeroed",
            },
            1,
            1,
            1,
            "zeroed",
        )
        assert rep == CompressedResourceOp(
            qml.ops.ControlledQubitUnitary,
            {
                "num_target_wires": 2,
                "num_control_wires": 2,
                "num_zero_control_values": 2,
                "num_work_wires": 2,
                "work_wire_type": "zeroed",
            },
        )

    def test_nested_controlled_qubit_unitary(self):
        """Tests that a nested controlled qubit unitary is flattened."""

        rep = controlled_resource_rep(
            qml.ops.Controlled,
            {
                "base_class": qml.ControlledQubitUnitary,
                "base_params": {
                    "num_target_wires": 1,
                    "num_control_wires": 2,
                    "num_zero_control_values": 1,
                    "num_work_wires": 1,
                    "work_wire_type": "borrowed",
                },
                "num_control_wires": 1,
                "num_zero_control_values": 1,
                "num_work_wires": 1,
                "work_wire_type": "borrowed",
            },
            1,
            1,
            1,
            "zeroed",
        )
        assert rep == CompressedResourceOp(
            qml.ops.ControlledQubitUnitary,
            {
                "num_target_wires": 1,
                "num_control_wires": 4,
                "num_zero_control_values": 3,
                "num_work_wires": 3,
                "work_wire_type": "borrowed",
            },
        )

    def test_custom_controlled_ops(self):
        """Tests that the resource rep of custom controlled ops remain as the custom version."""

        for op_type in custom_ctrl_op_to_base():
            rep = resource_rep(op_type)
            assert rep == CompressedResourceOp(op_type, {})


@pytest.mark.unit
class TestSymbolicResourceRep:
    """Tests resource reps of symbolic operators"""

    def test_adjoint_resource_rep(self):
        """Tests creating the resource rep of the adjoint of an operator."""

        rep = qml.decomposition.adjoint_resource_rep(DummyOp, {"foo": 2, "bar": 1})
        assert rep == CompressedResourceOp(
            qml.ops.Adjoint, {"base_class": DummyOp, "base_params": {"foo": 2, "bar": 1}}
        )

    def test_resource_rep_dispatch_to_adjoint_resource_rep(self, mocker):
        """Tests that resource_rep dispatches to adjoint_resource_rep for Adjoint."""

        expected_fn = mocker.patch("pennylane.decomposition.resources.adjoint_resource_rep")
        _ = resource_rep(
            qml.ops.Adjoint, **{"base_class": DummyOp, "base_params": {"foo": 2, "bar": 1}}
        )
        assert expected_fn.called

    def test_adjoint_resource_rep_base_param_mismatch(self):
        """Tests that an error is raised when base op and base params mismatch."""

        with pytest.raises(TypeError, match="Missing keyword arguments"):
            qml.decomposition.adjoint_resource_rep(DummyOp, {})

    def test_adjoint_custom_controlled_ops(self):
        """Tests that the adjoint of custom controlled ops remain as the custom version."""

        for op_type in custom_ctrl_op_to_base():
            rep = qml.decomposition.adjoint_resource_rep(base_class=op_type, base_params={})
            assert rep == CompressedResourceOp(
                qml.ops.Adjoint,
                {
                    "base_class": op_type,
                    "base_params": {},
                },
            )

    def test_pow_resource_rep(self):
        """Tests the pow_resource_rep utility function."""

        rep = qml.decomposition.pow_resource_rep(qml.MultiRZ, {"num_wires": 3}, 3)
        assert rep == CompressedResourceOp(
            qml.ops.Pow, {"base_class": qml.MultiRZ, "base_params": {"num_wires": 3}, "z": 3}
        )

        op = qml.pow(qml.MultiRZ(0.5, wires=[0, 1, 2]), 3)
        assert op.resource_params == rep.params
