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

"""Unit tests for the DecompositionRule class."""

from textwrap import dedent

import pytest

import pennylane as qml
from pennylane.decomposition.decomposition_rule import (
    DecompCollection,
    DecompositionRule,
    WorkWireSpec,
    _decompositions_private,
    register_condition,
    register_resources,
)
from pennylane.decomposition.resources import CompressedResourceOp, Resources
from pennylane.operation import Operator


class CustomOp(Operator):  # pylint: disable=too-few-public-methods
    pass


@pytest.mark.unit
class TestDecompositionRule:
    """Unit tests for DecompositionRule."""

    @pytest.mark.parametrize("exact_resources", [False, True])
    def test_create_decomposition_rule(self, exact_resources):
        """Test that a DecompositionRule object can be created."""

        def multi_rz_decomposition(theta, wires, **__):
            for w0, w1 in zip(wires[-1:0:-1], wires[-2::-1]):
                qml.CNOT(wires=(w0, w1))
            qml.RZ(theta, wires=wires[0])
            for w0, w1 in zip(wires[1:], wires[:-1]):
                qml.CNOT(wires=(w0, w1))

        def _multi_rz_resources(num_wires):
            return {
                qml.RZ: 1,
                qml.CNOT: 2 * (num_wires - 1),
            }

        multi_rz_decomposition = register_resources(
            _multi_rz_resources, multi_rz_decomposition, exact=exact_resources
        )

        assert isinstance(multi_rz_decomposition, DecompositionRule)

        with qml.queuing.AnnotatedQueue() as q:
            multi_rz_decomposition(0.5, wires=[0, 1, 2])

        assert q.queue == [
            qml.CNOT(wires=[2, 1]),
            qml.CNOT(wires=[1, 0]),
            qml.RZ(0.5, wires=[0]),
            qml.CNOT(wires=[1, 0]),
            qml.CNOT(wires=[2, 1]),
        ]

        assert multi_rz_decomposition.compute_resources(num_wires=3) == Resources(
            gate_counts={CompressedResourceOp(qml.RZ): 1, CompressedResourceOp(qml.CNOT): 4},
        )
        assert multi_rz_decomposition.exact_resources is exact_resources

    @pytest.mark.parametrize("exact_resources", [False, True])
    def test_decomposition_decorator(self, exact_resources):
        """Tests creating a decomposition rule using the decorator syntax."""

        def _multi_rz_resources(num_wires):
            return {
                qml.RZ: 1,
                qml.CNOT: 2 * (num_wires - 1),
            }

        @register_resources(_multi_rz_resources, exact=exact_resources)
        def multi_rz_decomposition(theta, wires, **__):
            for w0, w1 in zip(wires[-1:0:-1], wires[-2::-1]):
                qml.CNOT(wires=(w0, w1))
            qml.RZ(theta, wires=wires[0])
            for w0, w1 in zip(wires[1:], wires[:-1]):
                qml.CNOT(wires=(w0, w1))

        assert isinstance(multi_rz_decomposition, DecompositionRule)
        assert multi_rz_decomposition.exact_resources is exact_resources

        with qml.queuing.AnnotatedQueue() as q:
            multi_rz_decomposition(0.5, wires=[0, 1, 2])

        assert q.queue == [
            qml.CNOT(wires=[2, 1]),
            qml.CNOT(wires=[1, 0]),
            qml.RZ(0.5, wires=[0]),
            qml.CNOT(wires=[1, 0]),
            qml.CNOT(wires=[2, 1]),
        ]

        assert multi_rz_decomposition.compute_resources(num_wires=3) == Resources(
            gate_counts={CompressedResourceOp(qml.RZ): 1, CompressedResourceOp(qml.CNOT): 4},
        )

    def test_decomposition_condition(self):
        """Tests that the register_condition works."""

        @register_resources({qml.H: 2, qml.Toffoli: 1})
        @register_condition(lambda num_wires: num_wires == 3)
        def rule_1(wires, **__):
            raise NotImplementedError

        assert isinstance(rule_1, DecompositionRule)
        assert rule_1.is_applicable(num_wires=3)
        assert not rule_1.is_applicable(num_wires=2)
        assert rule_1.compute_resources(num_wires=3) == Resources(
            {
                CompressedResourceOp(qml.H): 2,
                CompressedResourceOp(qml.Toffoli): 1,
            }
        )

        @register_condition(lambda num_wires: num_wires == 3)
        @register_resources({qml.H: 2, qml.Toffoli: 1})
        def rule_2(wires, **__):
            raise NotImplementedError

        assert isinstance(rule_2, DecompositionRule)
        assert rule_2.is_applicable(num_wires=3)
        assert not rule_2.is_applicable(num_wires=2)
        assert rule_2.compute_resources(num_wires=3) == Resources(
            {
                CompressedResourceOp(qml.H): 2,
                CompressedResourceOp(qml.Toffoli): 1,
            }
        )

        def _resource_fn(**_):
            return {qml.H: 2, qml.Toffoli: 1}

        @register_resources(_resource_fn)
        @register_condition(lambda num_wires: num_wires == 3)
        def rule_3(wires, **__):
            raise NotImplementedError

        assert isinstance(rule_3, DecompositionRule)
        assert rule_3.is_applicable(num_wires=3)
        assert not rule_3.is_applicable(num_wires=2)
        assert rule_3.compute_resources(num_wires=3) == Resources(
            {
                CompressedResourceOp(qml.H): 2,
                CompressedResourceOp(qml.Toffoli): 1,
            }
        )

    @pytest.mark.parametrize("exact_resources", [False, True])
    def test_inspect_decomposition_rule(self, exact_resources):
        """Tests that the source code for a decomposition rule can be inspected."""

        @register_resources({qml.H: 2, qml.CNOT: 1}, exact=exact_resources)
        def my_cz(wires):
            qml.H(wires[0])
            qml.CNOT(wires=wires)
            qml.H(wires[0])

        assert str(my_cz) == dedent("""
                @register_resources({qml.H: 2, qml.CNOT: 1}, exact=exact_resources)
                def my_cz(wires):
                    qml.H(wires[0])
                    qml.CNOT(wires=wires)
                    qml.H(wires[0])
                """).strip()

    @pytest.mark.parametrize("use_custom_name", [True, False])
    def test_decomposition_rule_name(self, use_custom_name):
        """Tests the name attribute of a decomposition rule."""

        name = "custom_cz" if use_custom_name else ""

        # Test that the name is correctly set when creating a fresh rule.

        @register_resources({qml.H: 2, qml.CNOT: 1}, name=name)
        def my_cz(wires):
            qml.H(wires[0])
            qml.CNOT(wires=wires)
            qml.H(wires[0])

        expected_name = name or "my_cz"
        assert my_cz.name == expected_name
        assert repr(my_cz) == f"DecompositionRule(name={expected_name})"

        # Test that the name is correctly set when decorating an existing
        # rule that was previously created by `register_condition`

        @register_resources({qml.H: 2, qml.CNOT: 1}, name=name)
        @register_condition(lambda **_: True)
        def my_cz_second(wires):
            qml.H(wires[0])
            qml.CNOT(wires=wires)
            qml.H(wires[0])

        expected_name = name or "my_cz_second"
        assert my_cz_second.name == expected_name
        assert repr(my_cz_second) == f"DecompositionRule(name={expected_name})"

    def test_error_raised_with_no_resource_fn(self):
        """Tests that an error is raised when no resource fn is provided."""

        def multi_rz_decomposition(theta, wires, **__):
            for w0, w1 in zip(wires[-1:0:-1], wires[-2::-1]):
                qml.CNOT(wires=(w0, w1))
            qml.RZ(theta, wires=wires[0])
            for w0, w1 in zip(wires[1:], wires[:-1]):
                qml.CNOT(wires=(w0, w1))

        multi_rz_decomposition = register_resources(None, multi_rz_decomposition)

        with pytest.raises(NotImplementedError, match="No resource estimation found"):
            multi_rz_decomposition.compute_resources(num_wires=3)

    def test_decomposition_dictionary(self):
        """Tests that decomposition rules can be registered for an operator."""

        class SomeOtherOp(Operator):  # pylint: disable=too-few-public-methods
            pass

        assert not qml.decomposition.has_decomp(SomeOtherOp)

        @register_resources({qml.RZ: 2, qml.CNOT: 1})
        def custom_decomp(theta, wires, **__):
            qml.RZ(theta, wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RZ(theta, wires=wires[0])

        @register_resources({qml.RX: 2, qml.CZ: 1})
        def custom_decomp2(theta, wires, **__):
            qml.RX(theta, wires=wires[0])
            qml.CZ(wires=[wires[0], wires[1]])
            qml.RX(theta, wires=wires[0])

        @register_resources({qml.RY: 2, qml.CNOT: 1})
        def custom_decomp3(theta, wires, **__):
            qml.RY(theta, wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RY(theta, wires=wires[0])

        qml.add_decomps(SomeOtherOp, custom_decomp)
        qml.add_decomps(SomeOtherOp, custom_decomp2, custom_decomp3)

        assert qml.decomposition.has_decomp(SomeOtherOp)
        assert qml.decomposition.has_decomp(SomeOtherOp(wires=[0, 1]))
        assert list(qml.list_decomps(SomeOtherOp)) == [
            custom_decomp,
            custom_decomp2,
            custom_decomp3,
        ]
        assert list(qml.list_decomps(SomeOtherOp(wires=[0, 1]))) == [
            custom_decomp,
            custom_decomp2,
            custom_decomp3,
        ]

        def custom_decomp4(theta, wires, **__):
            qml.RZ(theta, wires=wires[0])
            qml.CZ(wires=[wires[0], wires[1]])
            qml.RZ(theta, wires=wires[0])

        with pytest.raises(TypeError, match="decomposition rule must be a qfunc with a resource"):
            qml.add_decomps(SomeOtherOp, custom_decomp4)

        _decompositions_private.pop("SomeOtherOp")  # cleanup

    def test_add_decomp_duplicate_names(self):
        """Tests that you cannot add decomposition rules with duplicate names."""

        class AnotherOp(Operator):  # pylint: disable=too-few-public-methods
            pass

        @register_resources({qml.RZ: 2, qml.CNOT: 1})
        def custom_decomp(theta, wires, **_):
            qml.RZ(theta, wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RZ(theta, wires=wires[0])

        @register_resources({}, name="custom_decomp")
        def some_decomp(theta, wires, **_):
            raise NotImplementedError

        with pytest.raises(ValueError, match="multiple decompositions with the same name"):
            qml.add_decomps(AnotherOp, custom_decomp, some_decomp)

        qml.add_decomps(AnotherOp, custom_decomp)

        with pytest.raises(ValueError, match="name: custom_decomp already exists"):
            qml.add_decomps(AnotherOp, some_decomp)

        _decompositions_private.pop("AnotherOp")  # cleanup

    def test_local_decomp_context(self):
        """Tests the local context manager for decompositions."""

        assert not qml.decomposition.has_decomp(CustomOp)

        @register_resources({qml.RZ: 2, qml.CNOT: 1})
        def custom_decomp(theta, wires, **__):
            qml.RZ(theta, wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RZ(theta, wires=wires[0])

        @register_resources({qml.RX: 2, qml.CZ: 1})
        def custom_decomp2(theta, wires, **__):
            qml.RX(theta, wires=wires[0])
            qml.CZ(wires=[wires[0], wires[1]])
            qml.RX(theta, wires=wires[0])

        @register_resources({qml.RY: 2, qml.CNOT: 1})
        def custom_decomp3(theta, wires, **__):
            qml.RY(theta, wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RY(theta, wires=wires[0])

        with qml.decomposition.local_decomps():

            qml.add_decomps(CustomOp, custom_decomp)
            qml.add_decomps(CustomOp, custom_decomp2, custom_decomp3)
            qml.add_decomps(qml.CRX, custom_decomp)

            assert qml.decomposition.has_decomp(CustomOp)
            assert qml.decomposition.has_decomp(CustomOp(wires=[0, 1]))
            assert list(qml.list_decomps(CustomOp)) == [
                custom_decomp,
                custom_decomp2,
                custom_decomp3,
            ]
            assert custom_decomp in qml.list_decomps(qml.CRX)

        # test that the context properly cleans up.
        assert list(qml.list_decomps(CustomOp)) == []
        assert not qml.decomposition.has_decomp(CustomOp)
        assert custom_decomp not in qml.list_decomps(qml.CRX)

    def test_custom_symbolic_decomposition(self):
        """Tests that custom decomposition rules for symbolic operators can be registered."""

        @register_resources({qml.RX: 1, qml.RZ: 1})
        def my_adjoint_custom_op(theta, wires, **__):
            qml.RX(theta, wires=wires[0])
            qml.RZ(theta, wires=wires[1])

        qml.add_decomps("Adjoint(CustomOp)", my_adjoint_custom_op)
        assert qml.decomposition.has_decomp("Adjoint(CustomOp)")
        assert list(qml.list_decomps("Adjoint(CustomOp)")) == [my_adjoint_custom_op]
        assert qml.decomposition.has_decomp(qml.adjoint(CustomOp(wires=[0, 1])))
        assert list(qml.list_decomps("Adjoint(CustomOp)")) == [my_adjoint_custom_op]

    def test_auto_wrap_in_resource_op(self):
        """Tests that simply classes can be auto-wrapped in a ``CompressionResourceOp``."""

        class DummyOp(Operator):  # pylint: disable=too-few-public-methods

            resource_keys = set()

        @register_resources({DummyOp: 1})
        def custom_decomp(*_, **__):
            raise NotImplementedError

        assert custom_decomp.compute_resources() == Resources(
            gate_counts={CompressedResourceOp(DummyOp): 1}
        )

        def custom_decomp_2(*_, **__):
            raise NotImplementedError

        custom_decomp_2 = register_resources({CompressedResourceOp(DummyOp): 1}, custom_decomp_2)

        assert custom_decomp_2.compute_resources() == Resources(
            gate_counts={CompressedResourceOp(DummyOp): 1}
        )

    def test_auto_wrap_fails(self):
        """Tests that an op with non-empty resource_keys cannot be auto-wrapped."""

        class DummyOp(Operator):  # pylint: disable=too-few-public-methods

            resource_keys = {"foo"}

        @register_resources({DummyOp: 1})
        def custom_decomp(*_, **__):
            raise NotImplementedError

        with pytest.raises(TypeError, match="Operator DummyOp has non-empty resource_keys"):
            custom_decomp.compute_resources()

        def custom_decomp_2(*_, **__):
            raise NotImplementedError

        with pytest.raises(TypeError, match="must be a subclass of Operator"):
            custom_decomp_2 = register_resources({int: 1}, custom_decomp_2)
            custom_decomp_2.compute_resources()

    def test_register_work_wires(self):
        """Tests that a decomposition can register work wire requirements"""

        @register_resources(
            {qml.CNOT: 3}, work_wires={"zeroed": 1, "garbage": 2, "borrowed": 3, "burnable": 4}
        )
        def custom_decomp(*_, **__):
            raise NotImplementedError

        assert custom_decomp.get_work_wire_spec() == WorkWireSpec(1, 3, 4, 2)

        @register_resources(
            lambda num_wires: {qml.CNOT: num_wires},
            work_wires=lambda num_wires: {
                "zeroed": num_wires // 2,
                "borrowed": num_wires - num_wires // 2,
            },
        )
        @register_condition(lambda num_wires: num_wires > 2)
        def custom_decomp_2(*_, **__):
            raise NotImplementedError

        assert custom_decomp_2.get_work_wire_spec(num_wires=5) == WorkWireSpec(zeroed=2, borrowed=3)

    @pytest.mark.parametrize("exact_resources", [False, True])
    def test_set_resources(self, exact_resources):
        """Test that a DecompositionRule object can be assigned new resources."""

        def multi_rz_decomposition(theta, wires, **__):
            for w0, w1 in zip(wires[-1:0:-1], wires[-2::-1]):
                qml.CNOT(wires=(w0, w1))
            qml.RZ(theta, wires=wires[0])
            for w0, w1 in zip(wires[1:], wires[:-1]):
                qml.CNOT(wires=(w0, w1))

        def _multi_rz_resources_old(num_wires):
            return {
                qml.RZ: 500,
                qml.CNOT: 2 * (num_wires - 1),
            }

        def _multi_rz_resources_new(num_wires):
            return {
                qml.RZ: 1,
                qml.CNOT: 2 * (num_wires - 1),
            }

        multi_rz_decomposition = register_resources(
            _multi_rz_resources_old, multi_rz_decomposition, exact=exact_resources
        )

        assert isinstance(multi_rz_decomposition, DecompositionRule)
        assert multi_rz_decomposition.compute_resources(num_wires=3) == Resources(
            gate_counts={CompressedResourceOp(qml.RZ): 500, CompressedResourceOp(qml.CNOT): 4},
        )
        assert multi_rz_decomposition.exact_resources is exact_resources

        # Overwrite resources
        multi_rz_decomposition.set_resources(
            _multi_rz_resources_new, exact_resources=not exact_resources
        )

        assert multi_rz_decomposition.compute_resources(num_wires=3) == Resources(
            gate_counts={CompressedResourceOp(qml.RZ): 1, CompressedResourceOp(qml.CNOT): 4},
        )
        assert multi_rz_decomposition.exact_resources is not exact_resources


class TestDecompCollection:
    """Tests the DecompCollection class."""

    @pytest.fixture(autouse=True, scope="class")
    def setup(self):
        """Sets up decomposition rules for CustomOp."""

        @register_resources({qml.RZ: 2, qml.CNOT: 1})
        def custom_decomp(theta, wires, **_):
            qml.RZ(theta, wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RZ(theta, wires=wires[0])

        @register_resources({qml.RX: 2, qml.CZ: 1}, name="custom2")
        def custom_decomp2(theta, wires, **_):
            qml.RX(theta, wires=wires[0])
            qml.CZ(wires=[wires[0], wires[1]])
            qml.RX(theta, wires=wires[0])

        @register_resources({qml.RY: 2, qml.CNOT: 1})
        def custom_decomp3(theta, wires, **_):
            qml.RY(theta, wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RY(theta, wires=wires[0])

        with qml.decomposition.local_decomps():
            qml.add_decomps(CustomOp, custom_decomp, custom_decomp2, custom_decomp3)
            yield

    def test_list_decomps_return_collection(self):
        """Tests that list_decomps returns a DecompCollection."""

        collection = qml.list_decomps(CustomOp)
        assert isinstance(collection, DecompCollection)
        assert len(collection) == 3
        assert repr(collection) == dedent("""
            DecompCollection([
                DecompositionRule(name=custom_decomp),
                DecompositionRule(name=custom2),
                DecompositionRule(name=custom_decomp3)
            ])
            """).strip()
        assert str(collection) == dedent("""
            Available Decomposition Rules:
            0: custom_decomp
            1: custom2
            2: custom_decomp3
            """).strip()

    def test_decomp_collection_access(self):
        """Tests that decomposition rules are accessible by name or index."""

        collection = qml.list_decomps(CustomOp)
        assert collection[0].name == "custom_decomp"
        assert collection["custom_decomp3"].name == "custom_decomp3"
        assert len(collection) == 3
        assert all(isinstance(rule, DecompositionRule) for rule in collection)
        assert [r.name for r in collection] == ["custom_decomp", "custom2", "custom_decomp3"]

        assert collection[0] in collection
        assert "custom_decomp" in collection
        assert "hello" not in collection
        assert 42 not in collection

        with pytest.raises(KeyError, match="Cannot find a decomposition with the given name: abc"):
            collection["abc"]  # pylint: disable=pointless-statement

    def test_append(self):
        """Tests the append method."""

        @register_resources({qml.RZ: 2, qml.CNOT: 1})
        def custom_decomp(*_, **__):
            raise NotImplementedError

        collection = qml.list_decomps(CustomOp)
        with pytest.raises(ValueError, match="name: custom_decomp already exists!"):
            collection.append(custom_decomp)

        custom_decomp.name = "custom3"
        collection.append(custom_decomp)
        assert "custom3" in collection
        assert len(collection) == 4

    def test_concatenate(self):
        """Tests adding DecompCollection objects."""

        @register_resources({qml.RZ: 2, qml.CNOT: 1}, name="custom3")
        def custom_decomp(*_, **__):
            raise NotImplementedError

        collection = qml.list_decomps(CustomOp)
        other = [custom_decomp]

        new_collection = collection + other
        assert len(new_collection) == 4
        assert "custom3" in new_collection

        collection += other  # uses iadd
        assert len(collection) == 4
        assert "custom3" in collection


class CustomParametrizedOp(Operator):
    """A custom parametrized op for testing."""

    resource_keys = {"num_wires"}

    def __init__(self, theta, wires):
        super().__init__(theta, wires=wires)

    @property
    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires)}


class TestShowDecomps:
    """Tests inspecting decomposition rules."""

    @pytest.fixture(autouse=True, scope="class")
    def setup(self):
        """Sets up decomposition rules for CustomOp."""

        @register_condition(lambda num_wires: num_wires == 2)
        @register_resources({qml.RZ: 2, qml.CNOT: 1}, name="simple")
        def two_wires_decomp(theta, wires, **_):
            qml.RZ(theta, wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RZ(theta, wires=wires[0])

        @register_resources(lambda num_wires: {qml.RX: 2, qml.CZ: 2 * (num_wires - 1), qml.H: 1})
        def general_decomp(theta, wires, **_):

            @qml.for_loop(len(wires) - 1)
            def _loop(i):
                qml.CZ(wires=[wires[i], wires[i + 1]])

            @qml.for_loop(len(wires) - 1, 0, -1)
            def _loop_back(i):
                qml.CZ(wires=[wires[i - 1], wires[i]])

            qml.RX(theta, wires=wires[0])
            _loop()
            qml.H(wires[-1])
            _loop_back()
            qml.RX(theta, wires=wires[0])

        @register_condition(lambda num_wires: num_wires > 3)
        @register_resources(
            lambda num_wires: {qml.CZ: 6, qml.Toffoli: 2 * (num_wires - 1), qml.H: 1, qml.RX: 1},
            work_wires={"zeroed": 2},
            name="with-aux",
        )
        def another_decomp(theta, wires, **_):

            @qml.for_loop(len(wires) - 2)
            def _loop(i):
                qml.Toffoli(wires=[wires[i], wires[i + 1], wires[i + 2]])

            @qml.for_loop(len(wires) - 1, 1, -1)
            def _loop_back(i):
                qml.Toffoli(wires=[wires[i - 2], wires[i - 1], wires[i]])

            with qml.allocate(2, "zero") as aux_wires:
                qml.CZ([wires[0], aux_wires[0]])
                qml.CZ(aux_wires)
                qml.CZ([aux_wires[1], wires[0]])
                _loop()
                qml.H(aux_wires[1])
                qml.RX(theta, aux_wires[0])
                _loop_back()
                qml.CZ([aux_wires[1], wires[0]])
                qml.CZ(aux_wires)
                qml.CZ([wires[0], aux_wires[0]])

        with qml.decomposition.local_decomps():
            qml.add_decomps(CustomParametrizedOp, two_wires_decomp, general_decomp, another_decomp)
            yield

    def test_show_all_decomps(self, capsys):
        """Tests showing all decomposition rules associated with an operator."""

        qml.show_decomps(CustomParametrizedOp(0.5, wires=[0, 1]))
        captured = capsys.readouterr()
        assert captured.out == dedent("""
            Decomposition 0 (name: simple)
            0: ──RZ(0.50)─╭●──RZ(0.50)─┤  
            1: ───────────╰X───────────┤  
            Gate Count: {RZ: 2, CNOT: 1}

            Decomposition 1 (name: general_decomp)
            0: ──RX(0.50)─╭●────╭●──RX(0.50)─┤  
            1: ───────────╰Z──H─╰Z───────────┤  
            Gate Count: {RX: 2, CZ: 2, Hadamard: 1}

            Decomposition 2 (name: with-aux)
            Not applicable to the provided operator instance!
            """).lstrip()

        qml.show_decomps(CustomParametrizedOp(0.5, wires=[0, 1, 2, 3, 4]))
        captured = capsys.readouterr()
        assert captured.out == dedent("""
            Decomposition 0 (name: simple)
            Not applicable to the provided operator instance!

            Decomposition 1 (name: general_decomp)
            0: ──RX(0.50)─╭●──────────────────────╭●──RX(0.50)─┤  
            1: ───────────╰Z─╭●────────────────╭●─╰Z───────────┤  
            2: ──────────────╰Z─╭●──────────╭●─╰Z──────────────┤  
            3: ─────────────────╰Z─╭●────╭●─╰Z─────────────────┤  
            4: ────────────────────╰Z──H─╰Z────────────────────┤  
            Gate Count: {RX: 2, CZ: 8, Hadamard: 1}

            Decomposition 2 (name: with-aux)
            <DynamicWire>: ─╭Allocate─╭Z─╭●──RX(0.50)──────────────────────╭●─╭Z─╭Deallocate─┤  
            <DynamicWire>: ─╰Allocate─│──╰Z─╭●─────────H────────────────╭●─╰Z─│──╰Deallocate─┤  
                        0: ───────────╰●────╰Z────────╭●─────────────╭●─╰Z────╰●─────────────┤  
                        1: ───────────────────────────├●─╭●───────╭●─├●──────────────────────┤  
                        2: ───────────────────────────╰X─├●─╭●─╭●─├●─╰X──────────────────────┤  
                        3: ──────────────────────────────╰X─├●─├●─╰X─────────────────────────┤  
                        4: ─────────────────────────────────╰X─╰X────────────────────────────┤  
            Estimated Gate Count: {CZ: 6, Toffoli: 8, Hadamard: 1, RX: 1}
            Actual Gate Count: {CZ: 6, Toffoli: 6, Hadamard: 1, RX: 1}
            Wire Allocations: {'zero': 1}
            """).lstrip()

    def test_show_decomp_by_name(self, capsys):
        """Tests inspecting a particular decomp by name."""

        qml.show_decomps(CustomParametrizedOp(0.5, wires=[0, 1]), "simple")
        captured = capsys.readouterr()
        assert captured.out == dedent("""
            Name: simple
            0: ──RZ(0.50)─╭●──RZ(0.50)─┤  
            1: ───────────╰X───────────┤  
            Gate Count: {RZ: 2, CNOT: 1}
            """).lstrip()

    def test_show_decomp_with_rule(self, capsys):
        """Tests inspecting a particular decomposition rule."""

        rule = qml.list_decomps(CustomParametrizedOp)["general_decomp"]
        qml.show_decomps(CustomParametrizedOp(0.5, wires=[0, 1, 2, 3, 4]), rule)
        captured = capsys.readouterr()
        assert captured.out == dedent("""
            Name: general_decomp
            0: ──RX(0.50)─╭●──────────────────────╭●──RX(0.50)─┤  
            1: ───────────╰Z─╭●────────────────╭●─╰Z───────────┤  
            2: ──────────────╰Z─╭●──────────╭●─╰Z──────────────┤  
            3: ─────────────────╰Z─╭●────╭●─╰Z─────────────────┤  
            4: ────────────────────╰Z──H─╰Z────────────────────┤  
            Gate Count: {RX: 2, CZ: 8, Hadamard: 1}
            """).lstrip()

    def test_show_multiple_decomps(self, capsys):
        """Tests showing multiple decomposition rules."""

        rule = qml.list_decomps(CustomParametrizedOp)["general_decomp"]
        qml.show_decomps(CustomParametrizedOp(0.5, wires=[0, 1, 2, 3, 4]), rule, "with-aux")
        captured = capsys.readouterr()
        assert captured.out == dedent("""
            Decomposition 0 (name: general_decomp)
            0: ──RX(0.50)─╭●──────────────────────╭●──RX(0.50)─┤  
            1: ───────────╰Z─╭●────────────────╭●─╰Z───────────┤  
            2: ──────────────╰Z─╭●──────────╭●─╰Z──────────────┤  
            3: ─────────────────╰Z─╭●────╭●─╰Z─────────────────┤  
            4: ────────────────────╰Z──H─╰Z────────────────────┤  
            Gate Count: {RX: 2, CZ: 8, Hadamard: 1}

            Decomposition 1 (name: with-aux)
            <DynamicWire>: ─╭Allocate─╭Z─╭●──RX(0.50)──────────────────────╭●─╭Z─╭Deallocate─┤  
            <DynamicWire>: ─╰Allocate─│──╰Z─╭●─────────H────────────────╭●─╰Z─│──╰Deallocate─┤  
                        0: ───────────╰●────╰Z────────╭●─────────────╭●─╰Z────╰●─────────────┤  
                        1: ───────────────────────────├●─╭●───────╭●─├●──────────────────────┤  
                        2: ───────────────────────────╰X─├●─╭●─╭●─├●─╰X──────────────────────┤  
                        3: ──────────────────────────────╰X─├●─├●─╰X─────────────────────────┤  
                        4: ─────────────────────────────────╰X─╰X────────────────────────────┤  
            Estimated Gate Count: {CZ: 6, Toffoli: 8, Hadamard: 1, RX: 1}
            Actual Gate Count: {CZ: 6, Toffoli: 6, Hadamard: 1, RX: 1}
            Wire Allocations: {'zero': 1}
            """).lstrip()
