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

import pennylane as qp
from pennylane.core.operator import Operator2, abstractify
from pennylane.decomposition.resources import (
    CompressedResourceOp,
    Resources,
    _op_type_and_params,
    _resource_rep_from_op,
    _resource_rep_from_type_and_params,
    adjoint_resource_rep,
    controlled_resource_rep,
    custom_ctrl_op_to_base,
    pow_resource_rep,
    resource_rep,
)
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.typing import Wire
from tests.core.operator.operator2_utils import DynOp, HybridOp, StaticOp


@pytest.mark.unit
class TestResources:
    """Unit tests for the Resources data structure."""

    def test_resource_initialize(self):
        """Tests initializing a Resources object."""
        resources = Resources()
        assert resources.num_gates == 0
        assert resources.gate_counts == {}
        assert resources.weighted_cost == 0.0

    def test_abstractify_compressed_resource_op_is_idempotent(self):
        """An abstract resource representation should not be abstractified a second time."""
        rep = resource_rep(qp.MultiRZ, num_wires=3)
        assert abstractify(rep) is rep

    @pytest.mark.parametrize("base_cls", (qp.Y, qp.Z))
    def test_zero_control_count_is_preserved(self, base_cls):
        """Zero-controlled Operator2 resources must encode the exact number of zeros."""
        one_zero = _ctrl_abstract(
            abstractify(base_cls),
            Wire[2],
            Wire[1],
            "zeroed",
            num_zero_control_values=1,
        )
        two_zeros = _ctrl_abstract(
            abstractify(base_cls),
            Wire[2],
            Wire[1],
            "zeroed",
            num_zero_control_values=2,
        )

        assert isinstance(one_zero, CompressedResourceOp)
        assert one_zero != two_zeros
        assert len({one_zero: 1, two_zeros: 1}) == 2
        assert one_zero.op_type is qp.ops.Controlled
        assert one_zero.params == {
            "base_class": base_cls,
            "base_params": {},
            "num_control_wires": 2,
            "num_zero_control_values": 1,
            "num_work_wires": 1,
            "work_wire_type": "zeroed",
        }
        assert two_zeros.params["num_zero_control_values"] == 2

    def test_concrete_zero_control_count_is_preserved_only_for_resources(self):
        """Resource normalization retains zero counts without changing abstractification."""
        op = qp.ctrl(
            qp.Z(4),
            control=[0, 1],
            control_values=[False, True],
            work_wires=[2],
            work_wire_type="zeroed",
        )
        expected = _ctrl_abstract(
            abstractify(qp.Z),
            Wire[2],
            Wire[1],
            "zeroed",
            num_zero_control_values=1,
        )

        assert isinstance(abstractify(op), Operator2)
        assert not isinstance(abstractify(op), CompressedResourceOp)
        assert _resource_rep_from_op(op) == expected

    def test_zero_control_non_fixed_operator2_is_tagged(self):
        """Exact zero-control resources retain a non-fixed Operator2's full abstract form."""
        base = abstractify(DynOp([0.1, 0.2], wires=[0, 1]))
        rep = _ctrl_abstract(base, Wire[1], num_zero_control_values=1)

        assert isinstance(rep, CompressedResourceOp)
        assert rep.params["base_class"] is DynOp
        assert (
            _resource_rep_from_type_and_params(rep.params["base_class"], rep.params["base_params"])
            == base
        )

    @pytest.mark.parametrize(
        "op",
        (
            DynOp([0.1, 0.2], wires=[0, 1]),
            StaticOp("label", wires=[0, 1]),
            HybridOp([qp.X(0)], wires=[1]),
        ),
    )
    def test_non_fixed_operator2_pack_roundtrip(self, op):
        """Packing nested Operator2 bases retains dynamic, wire, static, and hybrid metadata."""
        expected = abstractify(op)
        op_type, params = _op_type_and_params(expected)

        assert _resource_rep_from_type_and_params(op_type, params) == expected
        assert resource_rep(op_type, **params) == expected

    def test_recursive_adjoint_preserves_zero_control_count(self):
        """Resource normalization recursively retains zeros below an Adjoint2 wrapper."""
        one_zero = qp.adjoint(qp.ctrl(qp.Z(2), [0, 1], control_values=[False, True]))
        two_zeros = qp.adjoint(qp.ctrl(qp.Z(2), [0, 1], control_values=[False, False]))

        assert _resource_rep_from_op(one_zero) != _resource_rep_from_op(two_zeros)

    def test_negative_gate_counts(self):
        """Tests that an error is raised if the gate count is negative."""
        with pytest.raises(AssertionError):
            Resources(
                gate_counts={
                    CompressedResourceOp(qp.RX, {}): 2,
                    CompressedResourceOp(qp.RZ, {}): -1,
                }
            )

    def test_negative_weighted_cost(self):
        """Tests that an error is raised if the cost is negative."""
        with pytest.raises(AssertionError):
            Resources(
                gate_counts={
                    CompressedResourceOp(qp.RX, {}): 2,
                },
                weighted_cost=-2.0,
            )

    def test_add_resources(self):
        """Tests adding two Resources objects."""

        resources1 = Resources(
            gate_counts={CompressedResourceOp(qp.RX, {}): 2, CompressedResourceOp(qp.RZ, {}): 1},
            weighted_cost=6.0,
        )
        resources2 = Resources(
            gate_counts={CompressedResourceOp(qp.RX, {}): 1, CompressedResourceOp(qp.RY, {}): 1},
            weighted_cost=2.0,
        )

        resources = resources1 + resources2
        assert resources.num_gates == 5
        assert resources.gate_counts == {
            CompressedResourceOp(qp.RX, {}): 3,
            CompressedResourceOp(qp.RZ, {}): 1,
            CompressedResourceOp(qp.RY, {}): 1,
        }
        assert resources.weighted_cost == 8.0

    def test_mul_resource_with_scalar(self):
        """Tests multiplying a Resources object with a scalar."""

        resources = Resources(
            gate_counts={CompressedResourceOp(qp.RX, {}): 2, CompressedResourceOp(qp.RZ, {}): 1},
            weighted_cost=2.0,
        )

        resources = resources * 2
        assert resources.num_gates == 6
        assert resources.gate_counts == {
            CompressedResourceOp(qp.RX, {}): 4,
            CompressedResourceOp(qp.RZ, {}): 2,
        }
        assert resources.weighted_cost == 4

    def test_repr(self):
        """Tests the __repr__ of a Resources object."""

        resources = Resources(
            {CompressedResourceOp(qp.RX, {}): 2, CompressedResourceOp(qp.RZ, {}): 1}, 5.0
        )
        assert repr(resources) == "<num_gates=3, gate_counts={RX: 2, RZ: 1}, weighted_cost=5.0>"


class DummyOp(qp.operation.Operator):  # pylint: disable=too-few-public-methods
    resource_keys = {"foo", "bar"}


@pytest.mark.unit
class TestCompressedResourceOp:
    """Unit tests for the CompressedResourceOp data structure."""

    def test_initialization(self):
        """Tests creating a CompressedResourceOp object."""

        op = CompressedResourceOp(qp.QFT, {"num_wires": 5})
        assert op.op_type is qp.QFT
        assert op.params == {"num_wires": 5}

        op = CompressedResourceOp(qp.RX)
        assert op.op_type is qp.RX
        assert op.params == {}

    def test_invalid_op_type(self):
        """Tests that an error is raised if the op is invalid."""

        with pytest.raises(TypeError, match="op_type must be an Operator type"):
            CompressedResourceOp("RX", {})

        with pytest.raises(TypeError, match="op_type must be a subclass of Operator"):
            CompressedResourceOp(int, {})

        with pytest.raises(TypeError, match="cannot represent an Operator2 class directly"):
            CompressedResourceOp(qp.CNOT)

    def test_hash(self):
        """Tests that a CompressedResourceOp object is hashable."""

        op = CompressedResourceOp(qp.RX, {})
        assert isinstance(hash(op), int)

        op = CompressedResourceOp(qp.QFT, {"num_wires": 5})
        assert isinstance(hash(op), int)

        op = CompressedResourceOp(
            qp.ops.Controlled,
            {
                "base_class": qp.QFT,
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
            qp.ops.Exp,
            {
                "base_class": qp.ops.LinearCombination,
                "base_params": {},
                "base_pauli_rep": qp.Hamiltonian(
                    [1.11, 0.12, -3.4, 5],
                    [qp.X(0) @ qp.X(1), qp.Z(2), qp.Y(0) @ qp.Y(1), qp.I((0, 1, 2))],
                ).pauli_rep,
                "coeff": 1.2j,
            },
        )
        assert isinstance(hash(op), int)

    def test_hash_list_params(self):
        """Tests when the resource params contains a list."""

        class CustomOp(qp.operation.Operator):  # pylint: disable=too-few-public-methods
            resource_keys = {"foo", "bar"}

            @property
            def resource_params(self) -> dict:
                return {"foo": [1, 2, 3], "bar": [1, 2, [3, 4, 5]]}

        op = CompressedResourceOp(CustomOp, {"foo": [1, 2, 3], "bar": [1, 2, [3, 4, 5]]})
        assert isinstance(hash(op), int)

    def test_same_params_same_hash(self):
        """Tests that two ops with the same params have the same hash."""

        op1 = CompressedResourceOp(qp.RX, {"a": 1, "b": 2})
        op2 = CompressedResourceOp(qp.RX, {"b": 2, "a": 1})
        assert hash(op1) == hash(op2)

    def test_mixed_operator_representations_same_hash(self):
        """Mixed legacy and Operator2 resource keys are hashable independent of order."""

        legacy_rep = CompressedResourceOp(qp.Hadamard)
        operator2_rep = abstractify(qp.X(0))

        op1 = CompressedResourceOp(qp.ops.Prod, {"resources": {legacy_rep: 1, operator2_rep: 2}})
        op2 = CompressedResourceOp(qp.ops.Prod, {"resources": {operator2_rep: 2, legacy_rep: 1}})

        assert op1 == op2
        assert hash(op1) == hash(op2)

    def test_empty_params_same_hash(self):
        """Tests that CompressedResourceOp objects initialized with or without empty
        parameters have the same hash."""
        op1 = CompressedResourceOp(qp.RX)
        op2 = CompressedResourceOp(qp.RX, {})
        assert hash(op1) == hash(op2)

    def test_different_params_different_hash(self):
        """Tests that CompressedResourceOp objects initialized with different parameters
        have different hashes."""
        op1 = CompressedResourceOp(qp.MultiRZ, {"num_wires": 5})
        op2 = CompressedResourceOp(qp.MultiRZ, {"num_wires": 6})
        assert hash(op1) != hash(op2)

    def test_equal(self):
        """Tests comparing two CompressedResourceOp objects."""

        op1 = CompressedResourceOp(qp.RX, {})
        op2 = CompressedResourceOp(qp.RX, {})
        assert op1 == op2

        op1 = CompressedResourceOp(qp.RX, {})
        op2 = CompressedResourceOp(qp.RZ, {})
        assert op1 != op2

        op1 = CompressedResourceOp(qp.MultiRZ, {"num_wires": 3})
        op2 = CompressedResourceOp(qp.MultiRZ, {"num_wires": 3})
        assert op1 == op2

        op1 = CompressedResourceOp(qp.MultiRZ, {"num_wires": 5})
        op2 = CompressedResourceOp(qp.MultiRZ, {"num_wires": 6})
        assert op1 != op2

        op1 = CompressedResourceOp(
            qp.ops.Prod, {"resources": {CompressedResourceOp(DummyOp, {"foo": 1, "bar": 2}): 1}}
        )
        op2 = CompressedResourceOp(
            qp.ops.Prod, {"resources": {CompressedResourceOp(DummyOp, {"bar": 2, "foo": 1}): 1}}
        )
        assert op1 == op2

    def test_repr(self):
        """Tests the repr defined for debugging purposes."""

        op = CompressedResourceOp(qp.RX, {})
        assert repr(op) == "RX"

        op = CompressedResourceOp(qp.MultiRZ, {"num_wires": 5})
        assert repr(op) == "MultiRZ(num_wires=5)"

        op = CompressedResourceOp(DummyOp, {"bar": 1, "foo": 2})
        assert repr(op) == "DummyOp(bar=1, foo=2)"

        op = CompressedResourceOp(DummyOp, {"foo": 2, "bar": 1})
        assert repr(op) == "DummyOp(bar=1, foo=2)"

        op = adjoint_resource_rep(qp.MultiRZ, {"num_wires": 4})
        assert repr(op) == "Adjoint(MultiRZ(num_wires=4))"

        op = pow_resource_rep(qp.MultiRZ, {"num_wires": 4}, z=2)
        assert repr(op) == "Pow(MultiRZ(num_wires=4), z=2)"

        op = controlled_resource_rep(qp.MultiRZ, {"num_wires": 5}, num_control_wires=2)
        assert (
            repr(op)
            == "Controlled(MultiRZ(num_wires=5), num_control_wires=2, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed)"
        )

    @pytest.mark.parametrize(
        "op, expected_name",
        [
            (resource_rep(qp.RX), "RX"),
            (adjoint_resource_rep(qp.RX, {}), "Adjoint(RX)"),
            (controlled_resource_rep(qp.T, {}, 1, 0, 0), "C(T)"),
            (pow_resource_rep(qp.RX, {}, 2), "Pow(RX)"),
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

        class CustomOp(qp.operation.Operator):  # pylint: disable=too-few-public-methods
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

    def test_resource_rep_basis_embedding_normalization(self):
        """Tests that BasisEmbedding is normalized to BasisState in resource_rep."""

        rep = resource_rep(qp.BasisEmbedding, num_wires=3)
        assert rep == resource_rep(qp.BasisState, num_wires=3)
        assert rep.op_type is qp.BasisState


@pytest.mark.unit
class TestControlledResourceRep:
    """Tests the controlled_resource_rep function."""

    @pytest.mark.parametrize("wires", [[0, 1], [0, 1, 2]])
    def test_abstractify_explicit_multicontrolled_x(self, wires):
        """An explicit MultiControlledX should retain its resource identity."""

        op = qp.MultiControlledX(wires)

        assert abstractify(op) == resource_rep(qp.MultiControlledX, **op.resource_params)

    def test_controlled_resource_rep(self):
        """Tests creating the resource rep of a general controlled operation."""

        rep = controlled_resource_rep(DummyOp, {"foo": 2, "bar": 1}, 2, 1, 1)
        assert rep == CompressedResourceOp(
            qp.ops.Controlled,
            {
                "base_class": DummyOp,
                "base_params": {"foo": 2, "bar": 1},
                "num_control_wires": 2,
                "num_zero_control_values": 1,
                "num_work_wires": 1,
                "work_wire_type": "borrowed",
            },
        )

    def test_controlled_resource_rep_basis_embedding_normalization(self):
        """Tests that BasisEmbedding is normalized to BasisState in controlled_resource_rep."""

        rep = controlled_resource_rep(
            qp.BasisEmbedding, {"num_wires": 3}, num_control_wires=1, num_zero_control_values=0
        )
        expected = controlled_resource_rep(
            qp.BasisState, {"num_wires": 3}, num_control_wires=1, num_zero_control_values=0
        )
        assert rep == expected

        # Also verify consistency with the resource_rep path (from actual ops)
        actual_op = qp.ctrl(qp.BasisEmbedding(features=1, wires=[0, 1, 2]), control=3)
        from_actual = abstractify(actual_op)
        assert rep == from_actual

    def test_controlled_resource_rep_flatten(self):
        """Tests that nested controlled ops are flattened."""

        rep = controlled_resource_rep(
            qp.ops.Controlled,
            {
                "base_class": qp.CRX,
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
            qp.ops.Controlled,
            {
                "base_class": qp.RX,
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
            qp.ops.Controlled,
            {
                "base_class": qp.MultiControlledX,
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
            qp.ops.MultiControlledX,
            {
                "num_control_wires": 4,
                "num_zero_control_values": 3,
                "num_work_wires": 3,
                "work_wire_type": "zeroed",
            },
        )

    @pytest.mark.parametrize(
        "num_control_wires, num_zero_control_values, num_work_wires, work_wire_type, expected",
        [
            (1, 0, 0, "zeroed", resource_rep(qp.CNOT)),
            (1, 0, 1, "zeroed", resource_rep(qp.CNOT)),
            (
                1,
                1,
                0,
                "zeroed",
                CompressedResourceOp(
                    qp.MultiControlledX,
                    {
                        "num_control_wires": 1,
                        "num_zero_control_values": 1,
                        "num_work_wires": 0,
                        "work_wire_type": "zeroed",
                    },
                ),
            ),
            (
                1,
                1,
                1,
                "zeroed",
                CompressedResourceOp(
                    qp.MultiControlledX,
                    {
                        "num_control_wires": 1,
                        "num_zero_control_values": 1,
                        "num_work_wires": 1,
                        "work_wire_type": "zeroed",
                    },
                ),
            ),
            (2, 0, 0, "zeroed", resource_rep(qp.Toffoli)),
            (
                2,
                0,
                1,
                "zeroed",
                CompressedResourceOp(
                    qp.MultiControlledX,
                    {
                        "num_control_wires": 2,
                        "num_zero_control_values": 0,
                        "num_work_wires": 1,
                        "work_wire_type": "zeroed",
                    },
                ),
            ),
            (
                2,
                1,
                0,
                "zeroed",
                CompressedResourceOp(
                    qp.MultiControlledX,
                    {
                        "num_control_wires": 2,
                        "num_zero_control_values": 1,
                        "num_work_wires": 0,
                        "work_wire_type": "zeroed",
                    },
                ),
            ),
            (
                2,
                1,
                1,
                "zeroed",
                CompressedResourceOp(
                    qp.MultiControlledX,
                    {
                        "num_control_wires": 2,
                        "num_zero_control_values": 1,
                        "num_work_wires": 1,
                        "work_wire_type": "zeroed",
                    },
                ),
            ),
        ],
    )
    def test_controlled_x_rep_for_x_base(  # pylint: disable=too-many-arguments
        self, num_control_wires, num_zero_control_values, num_work_wires, work_wire_type, expected
    ):
        """Test that resources of controlled PauliX gates are mapped correctly"""
        rep = controlled_resource_rep(
            qp.X,
            {},
            num_control_wires=num_control_wires,
            num_zero_control_values=num_zero_control_values,
            num_work_wires=num_work_wires,
            work_wire_type=work_wire_type,
        )
        assert rep == expected

    def test_controlled_qubit_unitary(self):
        """Tests that a controlled QubitUnitary is a ControlledQubitUnitary."""

        rep = controlled_resource_rep(
            qp.ops.Controlled,
            {
                "base_class": qp.QubitUnitary,
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
            qp.ops.ControlledQubitUnitary,
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
            qp.ops.Controlled,
            {
                "base_class": qp.ControlledQubitUnitary,
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
            qp.ops.ControlledQubitUnitary,
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
            expected = (
                abstractify(op_type)
                if issubclass(op_type, Operator2)
                else CompressedResourceOp(op_type, {})
            )
            assert rep == expected


@pytest.mark.unit
class TestSymbolicResourceRep:
    """Tests resource reps of symbolic operators"""

    def test_adjoint_resource_rep(self):
        """Tests creating the resource rep of the adjoint of an operator."""

        rep = qp.decomposition.adjoint_resource_rep(DummyOp, {"foo": 2, "bar": 1})
        assert rep == CompressedResourceOp(
            qp.ops.Adjoint, {"base_class": DummyOp, "base_params": {"foo": 2, "bar": 1}}
        )

    def test_resource_rep_dispatch_to_adjoint_resource_rep(self, mocker):
        """Tests that resource_rep dispatches to adjoint_resource_rep for Adjoint."""

        expected_fn = mocker.patch("pennylane.decomposition.resources.adjoint_resource_rep")
        _ = resource_rep(
            qp.ops.Adjoint, **{"base_class": DummyOp, "base_params": {"foo": 2, "bar": 1}}
        )
        assert expected_fn.called

    def test_adjoint_resource_rep_base_param_mismatch(self):
        """Tests that an error is raised when base op and base params mismatch."""

        with pytest.raises(TypeError, match="Missing keyword arguments"):
            qp.decomposition.adjoint_resource_rep(DummyOp, {})

    def test_adjoint_custom_controlled_ops(self):
        """Tests that the adjoint of custom controlled ops remain as the custom version."""

        for op_type in custom_ctrl_op_to_base():
            rep = qp.decomposition.adjoint_resource_rep(base_class=op_type, base_params={})
            expected = (
                qp.adjoint(abstractify(op_type))
                if issubclass(op_type, Operator2)
                else CompressedResourceOp(
                    qp.ops.Adjoint,
                    {
                        "base_class": op_type,
                        "base_params": {},
                    },
                )
            )
            assert rep == expected

    def test_pow_resource_rep(self):
        """Tests the pow_resource_rep utility function."""

        rep = qp.decomposition.pow_resource_rep(qp.MultiRZ, {"num_wires": 3}, 3)
        assert rep == CompressedResourceOp(
            qp.ops.Pow, {"base_class": qp.MultiRZ, "base_params": {"num_wires": 3}, "z": 3}
        )

        op = qp.pow(qp.MultiRZ(0.5, wires=[0, 1, 2]), 3)
        assert op.resource_params == rep.params
