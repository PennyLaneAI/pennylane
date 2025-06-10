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
Test the base and abstract Resource class
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, List

import numpy as np
import pytest

import pennylane as qml
from pennylane.labs.resource_estimation import (
    CompressedResourceOp,
    QubitManager,
    ResourceOperator,
    Resources,
    set_adj_decomp,
    set_ctrl_decomp,
    set_decomp,
    set_pow_decomp,
)
from pennylane.labs.resource_estimation.resource_operator import (
    GateCount,
    _make_hashable,
    resource_rep,
)
from pennylane.queuing import AnnotatedQueue
from pennylane.wires import Wires

# pylint: disable=protected-access, too-few-public-methods, no-self-use, unused-argument, arguments-differ


class ResourceDummyX(ResourceOperator):
    """Dummy testing class representing X gate"""


class ResourceDummyQFT(ResourceOperator):
    """Dummy testing class representing QFT gate"""


class ResourceDummyQSVT(ResourceOperator):
    """Dummy testing class representing QSVT gate"""


class ResourceDummyTrotterProduct(ResourceOperator):
    """Dummy testing class representing TrotterProduct gate"""


class ResourceDummyAdjoint(ResourceOperator):
    """Dummy testing class representing the Adjoint symbolic operator"""


class TestCompressedResourceOp:
    """Testing the methods and attributes of the CompressedResourceOp class"""

    test_hamiltonian = qml.dot([1, -1, 0.5], [qml.X(0), qml.Y(1), qml.Z(0) @ qml.Z(1)])
    compressed_ops_and_params_lst = (
        ("DummyX", ResourceDummyX, {"num_wires": 1}, None),
        ("DummyQFT", ResourceDummyQFT, {"num_wires": 5}, None),
        ("DummyQSVT", ResourceDummyQSVT, {"num_wires": 3, "num_angles": 5}, None),
        (
            "DummyTrotterProduct",
            ResourceDummyTrotterProduct,
            {"Hamiltonian": test_hamiltonian, "num_steps": 5, "order": 2},
            None,
        ),
        ("X", ResourceDummyX, {"num_wires": 1}, "X"),
    )

    compressed_op_names = (
        "DummyX",
        "DummyQFT",
        "DummyQSVT",
        "DummyTrotterProduct",
        "X",
    )

    @pytest.mark.parametrize("name, op_type, parameters, name_param", compressed_ops_and_params_lst)
    def test_init(self, name, op_type, parameters, name_param):
        """Test that we can correctly instantiate CompressedResourceOp"""
        cr_op = CompressedResourceOp(op_type, parameters, name=name_param)

        assert cr_op._name == name
        assert cr_op.op_type is op_type
        assert cr_op.params == parameters
        assert sorted(cr_op._hashable_params) == sorted(tuple(parameters.items()))

    def test_hash(self):
        """Test that the hash method behaves as expected"""
        CmprssedQSVT1 = CompressedResourceOp(ResourceDummyQSVT, {"num_wires": 3, "num_angles": 5})
        CmprssedQSVT2 = CompressedResourceOp(ResourceDummyQSVT, {"num_wires": 3, "num_angles": 5})
        Other = CompressedResourceOp(ResourceDummyQFT, {"num_wires": 3})

        assert hash(CmprssedQSVT1) == hash(CmprssedQSVT1)  # compare same object
        assert hash(CmprssedQSVT1) == hash(CmprssedQSVT2)  # compare identical instance
        assert hash(CmprssedQSVT1) != hash(Other)

        # test dictionary as parameter
        CmprssedAdjoint1 = CompressedResourceOp(
            ResourceDummyAdjoint, {"base_class": ResourceDummyQFT, "base_params": {"num_wires": 1}}
        )
        CmprssedAdjoint2 = CompressedResourceOp(
            ResourceDummyAdjoint, {"base_class": ResourceDummyQFT, "base_params": {"num_wires": 1}}
        )
        Other = CompressedResourceOp(
            ResourceDummyAdjoint, {"base_class": ResourceDummyQFT, "base_params": {"num_wires": 2}}
        )

        assert hash(CmprssedAdjoint1) == hash(CmprssedAdjoint1)
        assert hash(CmprssedAdjoint1) == hash(CmprssedAdjoint2)
        assert hash(CmprssedAdjoint1) != hash(Other)

    def test_equality(self):
        """Test that the equality methods behaves as expected"""
        CmprssedQSVT1 = CompressedResourceOp(ResourceDummyQSVT, {"num_wires": 3, "num_angles": 5})
        CmprssedQSVT2 = CompressedResourceOp(ResourceDummyQSVT, {"num_wires": 3, "num_angles": 5})
        CmprssedQSVT3 = CompressedResourceOp(ResourceDummyQSVT, {"num_angles": 5, "num_wires": 3})
        Other = CompressedResourceOp(ResourceDummyQFT, {"num_wires": 3})

        assert CmprssedQSVT1 == CmprssedQSVT2  # compare identical instance
        assert CmprssedQSVT1 == CmprssedQSVT3  # compare swapped parameters
        assert CmprssedQSVT1 != Other

    @pytest.mark.parametrize("args, name", zip(compressed_ops_and_params_lst, compressed_op_names))
    def test_name(self, args, name):
        """Test that the name method behaves as expected."""
        _, op_type, parameters, name_param = args
        cr_op = CompressedResourceOp(op_type, parameters, name=name_param)

        assert cr_op.name == name

    @pytest.mark.parametrize("args", compressed_ops_and_params_lst)
    def test_repr(self, args):
        """Test that the name method behaves as expected."""
        _, op_type, parameters, name_param = args

        cr_op = CompressedResourceOp(op_type, parameters, name=name_param)

        op_name = op_type.__name__
        expected_params_str_parts = [f"{k!r}:{v!r}" for k, v in sorted(parameters.items())]
        expected_params_str = ", ".join(expected_params_str_parts)

        expected_repr_string = f"CompressedResourceOp({op_name}, params={{{expected_params_str}}})"
        assert str(cr_op) == expected_repr_string

    def test_type_error(self):
        """Test that an error is raised if wrong type is provided for op_type."""
        with pytest.raises(TypeError, match="op_type must be a subclass of ResourceOperator."):
            CompressedResourceOp(type(1))


@dataclass(frozen=True)
class DummyCmprsRep:
    """A dummy class to populate the gate types dictionary for testing."""

    name: str
    param: int = 0


class DummyOp(ResourceOperator):
    """A dummy class to test ResourceOperator instantiation."""

    resource_keys = {"x"}

    def __init__(self, x=None, wires=None):
        self.x = x
        super().__init__(wires=wires)

    def __eq__(self, other: object) -> bool:
        return (self.__class__.__name__ == other.__class__.__name__) and (self.x == other.x)

    @property
    def resource_params(self):
        return {"x": self.x}

    @classmethod
    def resource_rep(cls, x):
        return DummyCmprsRep(cls.__name__, param=x)

    @classmethod
    def default_resource_decomp(cls, x) -> List:
        return [x]


class DummyOp_no_resource_rep(ResourceOperator):
    """A dummy class to test ResourceOperator instantiation."""

    def __init__(self, x, wires=None):
        self.x = x
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        return DummyCmprsRep({"x": self.x})

    @classmethod
    def default_resource_decomp(cls, x) -> List:
        return [x]


class DummyOp_no_resource_params(ResourceOperator):
    """A dummy class to test ResourceOperator instantiation."""

    def __init__(self, x, wires=None):
        self.x = x
        super().__init__(wires=wires)

    @classmethod
    def resource_rep(cls, x):
        return DummyCmprsRep(cls.__name__, param=x)

    @classmethod
    def default_resource_decomp(cls, x) -> List:
        return [x]


class DummyOp_no_resource_decomp(ResourceOperator):
    """A dummy class to test ResourceOperator instatiation."""

    def __init__(self, x, wires=None):
        self.x = x
        super().__init__(wires=wires)

    @classmethod
    def resource_rep(cls, x):
        return DummyCmprsRep(cls.__name__, param=x)

    @property
    def resource_params(self):
        return DummyCmprsRep({"x": self.x})


class ResourceRX(DummyOp):
    """Dummy op representing RX"""

    num_wires = 1


class ResourceHadamard(DummyOp):
    """Dummy op representing Hadamard"""

    num_wires = 1


class ResourceCNOT(DummyOp):
    """Dummy op representing CNOT"""

    num_wires = 2


class TestResourceOperator:
    """Tests for the ResourceOperator class"""

    res_op_error_lst = [
        DummyOp_no_resource_rep,
        DummyOp_no_resource_params,
        DummyOp_no_resource_decomp,
    ]

    @pytest.mark.parametrize("res_op", res_op_error_lst)
    def test_init_error_abstract_methods(self, res_op):
        """Test that errors are raised when the resource operator
        is implemented without specifying the abstract methods."""

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            res_op(x=1)

    def test_init_queuing(self):
        """Test that instantiating a resource operator correctly sets its arguments
        and queues it."""

        with AnnotatedQueue() as q:
            ResourceHadamard(wires=[0])
            ResourceHadamard(wires=[1])
            ResourceCNOT(wires=[0, 1])
            ResourceRX(x=1.23, wires=[0])

        expected_queue = [
            ResourceHadamard(wires=[0]),
            ResourceHadamard(wires=[1]),
            ResourceCNOT(wires=[0, 1]),
            ResourceRX(x=1.23, wires=[0]),
        ]

        assert q.queue == expected_queue

    def test_init_wire_override(self):
        """Test that setting the wires correctly overrides the num_wires argument."""
        dummy_op1 = DummyOp()
        assert dummy_op1.wires is None
        assert dummy_op1.num_wires == 0

        dummy_op2 = DummyOp(wires=[0, 1, 2])
        assert dummy_op2.wires == Wires([0, 1, 2])
        assert dummy_op2.num_wires == 3

    @pytest.mark.parametrize("s", [1, 2, 3])
    def test_mul(self, s):
        """Test multiply dunder method"""
        op = ResourceRX(1.23)
        resources = s * op

        gt = defaultdict(int, {DummyCmprsRep("ResourceRX", 1.23): s})
        qm = QubitManager(work_wires=0, algo_wires=1)
        expected_resources = Resources(qubit_manager=qm, gate_types=gt)
        assert resources == expected_resources

    @pytest.mark.parametrize("s", [1, 2, 3])
    def test_mat_mul(self, s):
        """Test matrix-multiply dunder method"""
        op = ResourceCNOT()
        resources = s @ op

        gt = defaultdict(int, {DummyCmprsRep("ResourceCNOT", None): s})
        qm = QubitManager(work_wires=0, algo_wires=s * 2)
        expected_resources = Resources(qubit_manager=qm, gate_types=gt)
        assert resources == expected_resources

    def test_add(self):
        """Test addition dunder method between two ResourceOperator classes"""
        op1 = ResourceRX(1.23)
        op2 = ResourceCNOT()
        resources = op1 + op2

        gt = defaultdict(
            int,
            {
                DummyCmprsRep("ResourceRX", 1.23): 1,
                DummyCmprsRep("ResourceCNOT", None): 1,
            },
        )
        qm = QubitManager(work_wires=0, algo_wires=2)
        expected_resources = Resources(qubit_manager=qm, gate_types=gt)
        assert resources == expected_resources

    def test_add_resources(self):
        """Test addition dunder method between a ResourceOperator and a Resources object"""
        op1 = ResourceRX(1.23)
        gt2 = defaultdict(int, {DummyCmprsRep("ResourceCNOT", None): 1})
        qm2 = QubitManager(work_wires=0, algo_wires=2)
        res2 = Resources(qubit_manager=qm2, gate_types=gt2)
        resources = op1 + res2

        gt = defaultdict(
            int,
            {
                DummyCmprsRep("ResourceRX", 1.23): 1,
                DummyCmprsRep("ResourceCNOT", None): 1,
            },
        )
        qm = QubitManager(work_wires=0, algo_wires=2)
        expected_resources = Resources(qubit_manager=qm, gate_types=gt)
        assert resources == expected_resources

    def test_add_error(self):
        """Test addition dunder method raises error when adding with unsupported type"""
        with pytest.raises(TypeError, match="Cannot add resource operator"):
            op1 = ResourceRX(1.23)
            _ = op1 + True

    def test_and(self):
        """Test and dunder method between two ResourceOperator classes"""
        op1 = ResourceRX(1.23)
        op2 = ResourceCNOT()
        resources = op1 & op2

        gt = defaultdict(
            int,
            {
                DummyCmprsRep("ResourceRX", 1.23): 1,
                DummyCmprsRep("ResourceCNOT", None): 1,
            },
        )
        qm = QubitManager(work_wires=0, algo_wires=3)
        expected_resources = Resources(qubit_manager=qm, gate_types=gt)
        assert resources == expected_resources

    def test_and_resources(self):
        """Test and dunder method between a ResourceOperator and a Resources object"""
        op1 = ResourceRX(1.23)
        gt2 = defaultdict(int, {DummyCmprsRep("ResourceCNOT", None): 1})
        qm2 = QubitManager(work_wires=0, algo_wires=2)
        res2 = Resources(qubit_manager=qm2, gate_types=gt2)
        resources = op1 & res2

        gt = defaultdict(
            int,
            {
                DummyCmprsRep("ResourceRX", 1.23): 1,
                DummyCmprsRep("ResourceCNOT", None): 1,
            },
        )
        qm = QubitManager(work_wires=0, algo_wires=3)
        expected_resources = Resources(qubit_manager=qm, gate_types=gt)
        assert resources == expected_resources

    def test_and_error(self):
        """Test and dunder method raises error when adding with unsupported type"""
        with pytest.raises(TypeError, match="Cannot add resource operator"):
            op1 = ResourceRX(1.23)
            _ = op1 & True


@pytest.mark.parametrize(
    "input_obj, expected_hashable",
    [
        (123, 123),
        ("hello", "hello"),
        (3.14, 3.14),
        (None, None),
        ((1, 2, 3), (1, 2, 3)),
        ([], ()),
        ([1, 2, 3], (1, 2, 3)),
        ([[1, 2], [3, 4]], ((1, 2), (3, 4))),
        (set(), ()),
        ({3, 1, 2}, (1, 2, 3)),
        ({}, ()),
        ({"b": 2, "a": 1}, (("a", 1), ("b", 2))),
        ({"key": [1, 2]}, (("key", (1, 2)),)),
        ({"nested": {"x": 1, "y": [2, 3]}}, (("nested", (("x", 1), ("y", (2, 3)))),)),
        (np.array([1, 2, 3]), (1, 2, 3)),
        (np.array([["a", "b"], ["c", "d"]]), (("a", "b"), ("c", "d"))),
        ([{"a": 1}, {2, 3}], ((("a", 1),), (2, 3))),
        (
            {"list_key": [1, {"nested_dict": "val"}]},
            (("list_key", (1, (("nested_dict", "val"),))),),
        ),
    ],
)
def test_make_hashable(input_obj, expected_hashable):
    """Test that _make_hashable function works as expected"""
    result = _make_hashable(input_obj)
    assert result == expected_hashable
    assert isinstance(result, Hashable)
    assert hash(result) is not None


def test_set_decomp():
    """Test that the set_decomp function works as expected."""
    op1 = DummyOp(x=5)
    assert DummyOp.resource_decomp(**op1.resource_params) == [5]

    def custom_res_decomp(x, **kwargs):
        return [x + 1]

    set_decomp(DummyOp, custom_res_decomp)

    assert DummyOp.resource_decomp(**op1.resource_params) == [6]

    def custom_res_decomp_error(y):  # must match signature of default_resource_decomp
        return [y + 1]

    with pytest.raises(ValueError):
        set_decomp(DummyOp, custom_res_decomp_error)


def test_set_adj_decomp():
    """Test that the set_decomp function works as expected."""

    class DummyAdjOp(DummyOp):
        """Dummy Adjoint Op class"""

        @classmethod
        def default_adjoint_resource_decomp(cls, x):
            return cls.default_resource_decomp(x=x)

    op1 = DummyAdjOp(x=5)
    assert DummyAdjOp.adjoint_resource_decomp(**op1.resource_params) == [5]

    def custom_res_decomp(x, **kwargs):
        return [x + 1]

    set_adj_decomp(DummyAdjOp, custom_res_decomp)

    assert DummyAdjOp.adjoint_resource_decomp(**op1.resource_params) == [6]

    def custom_res_decomp_error(y):  # must match signature of default_adjoint_resource_decomp
        return [y + 1]

    with pytest.raises(ValueError):
        set_adj_decomp(DummyAdjOp, custom_res_decomp_error)


def test_set_ctrl_decomp():
    """Test that the set_decomp function works as expected."""

    class DummyCtrlOp(DummyOp):
        """Dummy Controlled Op class"""

        @classmethod
        def default_controlled_resource_decomp(cls, ctrl_num_ctrl_wires, ctrl_num_ctrl_values, x):
            return cls.default_resource_decomp(x=x + ctrl_num_ctrl_values)

    op1 = DummyCtrlOp(x=5)
    assert DummyCtrlOp.controlled_resource_decomp(1, 0, **op1.resource_params) == [5]

    def custom_res_decomp(ctrl_num_ctrl_wires, ctrl_num_ctrl_values, x, **kwargs):
        return [x + ctrl_num_ctrl_wires]

    set_ctrl_decomp(DummyCtrlOp, custom_res_decomp)

    assert DummyCtrlOp.controlled_resource_decomp(1, 0, **op1.resource_params) == [6]

    def custom_res_decomp_error(x):  # must match signature of default_controlled_resource_decomp
        return [x + 1]

    with pytest.raises(ValueError):
        set_ctrl_decomp(DummyCtrlOp, custom_res_decomp_error)


def test_set_pow_decomp():
    """Test that the set_decomp function works as expected."""

    class DummyPowOp(DummyOp):
        """Dummy Pow Op class"""

        @classmethod
        def default_pow_resource_decomp(cls, pow_z, x):
            return cls.default_resource_decomp(x=x)

    op1 = DummyPowOp(x=5)
    assert DummyPowOp.pow_resource_decomp(pow_z=3, **op1.resource_params) == [5]

    def custom_res_decomp(pow_z, x, **kwargs):
        return [x * pow_z]

    set_pow_decomp(DummyPowOp, custom_res_decomp)

    assert DummyPowOp.pow_resource_decomp(pow_z=3, **op1.resource_params) == [15]

    def custom_res_decomp_error(x):  # must match signature of default_pow_resource_decomp
        return [x + 1]

    with pytest.raises(ValueError):
        set_pow_decomp(DummyPowOp, custom_res_decomp_error)


class TestGateCount:
    """Tests for the GateCount class."""

    @pytest.mark.parametrize("n", (1, 2, 3, 5))
    def test_init(self, n):
        """Test that we can correctly instantiate a GateCount object"""
        op = DummyOp(x=5)
        gate_counts = GateCount(op) if n == 1 else GateCount(op, n)

        assert gate_counts.gate == op
        assert gate_counts.count == n

    def test_equality(self):
        """Test that the equality method works as expected"""
        op = DummyOp(x=5)
        op1 = DummyOp(x=6)

        gc1 = GateCount(op, count=5)
        gc2 = GateCount(op, count=5)
        gc3 = GateCount(op, count=3)
        gc4 = GateCount(op1, count=5)

        assert gc1 == gc1
        assert gc1 == gc2
        assert gc1 != gc3
        assert gc1 != gc4

    def test_add_method(self):
        """Test that the arimethic methods work as expected"""
        op = DummyOp(x=5)
        op1 = DummyOp(x=6)

        gc = GateCount(op, count=3)
        gc1 = GateCount(op, count=2)
        assert gc + gc1 == GateCount(op, count=5)

        with pytest.raises(NotImplementedError):
            gc2 = GateCount(op1, count=2)
            _ = gc + gc2

    def test_mul_method(self):
        """Test that the arimethic methods work as expected"""
        op = DummyOp(x=5)
        gc = GateCount(op, count=3)

        assert gc * 5 == GateCount(op, count=15)

        with pytest.raises(NotImplementedError):
            _ = gc * 0.5


def test_resource_rep():
    """Test that the resource_rep method works as expected"""

    class ResourceOpA(ResourceOperator):
        """Test resource op class"""

        resource_keys = {"num_wires", "continuous_param", "bool_param"}

        @property
        def resource_params(self):
            """resource params method"""
            return {
                "num_wires": self.num_wires,
                "continuous_param": self.continuous_param,
                "bool_param": self.bool_param,
            }

        @classmethod
        def resource_rep(cls, num_wires, continuous_param, bool_param):
            """resource rep method"""
            params = {
                "num_wires": num_wires,
                "continuous_param": continuous_param,
                "bool_param": bool_param,
            }
            return CompressedResourceOp(cls, params)

        @classmethod
        def default_resource_decomp(cls, num_wires, continuous_param, bool_param):
            raise NotImplementedError

    class ResourceOpB(ResourceOperator):
        """Test resource op class"""

        resource_keys = {}

        @property
        def resource_params(self):
            """resource params method"""
            return {}

        @classmethod
        def resource_rep(cls):
            """resource rep method"""
            return CompressedResourceOp(cls, {})

        @classmethod
        def default_resource_decomp(cls, **kwargs):
            raise NotImplementedError

    expected_resource_rep_A = CompressedResourceOp(
        ResourceOpA,
        {
            "num_wires": 1,
            "continuous_param": 2.34,
            "bool_param": False,
        },
    )
    actual_resource_rep_A = resource_rep(
        ResourceOpA,
        {
            "num_wires": 1,
            "continuous_param": 2.34,
            "bool_param": False,
        },
    )
    assert expected_resource_rep_A == actual_resource_rep_A

    expected_resource_rep_B = CompressedResourceOp(ResourceOpB, {})
    actual_resource_rep_B = resource_rep(ResourceOpB)
    assert expected_resource_rep_B == actual_resource_rep_B
