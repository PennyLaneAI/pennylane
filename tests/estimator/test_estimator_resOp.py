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
This submodule tests the base classes for resource operators.
"""
from collections import defaultdict
from collections.abc import Hashable

import numpy as np
import pytest

import pennylane as qml
import pennylane.estimator.ops as qre_ops
from pennylane.estimator import ResourceOperator, Resources
from pennylane.estimator.resource_operator import GateCount, _dequeue, _make_hashable, resource_rep
from pennylane.queuing import AnnotatedQueue

# pylint: disable=protected-access, too-few-public-methods, no-self-use, unused-argument, disable=arguments-differ, no-member, comparison-with-itself, too-many-arguments, too-many-public-methods


class DummyX(ResourceOperator):
    """Dummy testing class representing X gate"""


class DummyQFT(ResourceOperator):
    """Dummy testing class representing QFT gate"""


class DummyQSVT(ResourceOperator):
    """Dummy testing class representing QSVT gate"""


class DummyTrotterProduct(ResourceOperator):
    """Dummy testing class representing TrotterProduct gate"""


class DummyAdjoint(ResourceOperator):
    """Dummy testing class representing the Adjoint symbolic operator"""


class DummyOp(ResourceOperator):
    """A dummy class to test ResourceOperator instantiation."""

    resource_keys = {"x"}

    def __init__(self, x=None, wires=None, num_wires=None):
        self.x = x
        if num_wires:
            self.num_wires = num_wires
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        """dummy resource params method"""
        return {"x": self.x}

    @classmethod
    def resource_rep(cls, x):
        """dummy resource rep method"""
        return cls(x=x)

    @classmethod
    def resource_decomp(cls, x) -> list:
        """dummy resources"""
        return [x]


class DummyOp_no_resource_rep(ResourceOperator):
    """A dummy class to test ResourceOperator instantiation."""

    def __init__(self, x, wires=None):
        self.x = x
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        """dummy resource params method"""
        return {"x": self.x}

    @classmethod
    def resource_decomp(cls, x) -> list:
        """dummy resources"""
        return [x]


class DummyOp_no_resource_params(ResourceOperator):
    """A dummy class to test ResourceOperator instantiation."""

    def __init__(self, x, wires=None):
        self.x = x
        super().__init__(wires=wires)

    @classmethod
    def resource_rep(cls, x):
        """dummy resource rep method"""
        return cls(x=x)

    @classmethod
    def resource_decomp(cls, x) -> list:
        """dummy resources"""
        return [x]


class DummyOp_no_resource_decomp(ResourceOperator):
    """A dummy class to test ResourceOperator instatiation."""

    def __init__(self, x, wires=None):
        self.x = x
        super().__init__(wires=wires)

    @classmethod
    def resource_rep(cls, x):
        """dummy resource rep method"""
        return cls(x=x)

    @property
    def resource_params(self):
        """dummy resource params method"""
        return {"x": self.x}


class DummyOp_no_resource_keys(ResourceOperator):
    """A dummy class to test ResourceOperator instantiation."""

    def __init__(self, x=None, wires=None):
        self.x = x
        super().__init__(wires=wires)

    def __eq__(self, other: object) -> bool:
        return (self.__class__.__name__ == other.__class__.__name__) and (self.x == other.x)

    @property
    def resource_params(self):
        """dummy resource params method"""
        return {}

    @classmethod
    def resource_rep(cls, x):
        """dummy resource rep method"""
        return cls(x=x)

    @classmethod
    def resource_decomp(cls, x) -> list:
        """dummy resources"""
        return [x]


class X(DummyOp_no_resource_keys):
    """Dummy op representing X"""

    num_wires = 1


class RX(DummyOp):
    """Dummy op representing RX"""

    num_wires = 1


class Hadamard(DummyOp):
    """Dummy op representing Hadamard"""

    num_wires = 1


class CNOT(DummyOp):
    """Dummy op representing CNOT"""

    num_wires = 2


class DummyOp_decomps(ResourceOperator):
    """Class for testing the resource_decomp methods"""

    resource_keys = {"max_register_size"}

    def __init__(self, max_register_size, wires=None):
        self.max_register_size = max_register_size
        self.num_wires = 2 * max_register_size
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources."""
        return {"max_register_size": self.max_register_size}

    @classmethod
    def resource_rep(cls, max_register_size):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.
        """
        return cls(max_register_size=max_register_size)

    @classmethod
    def resource_decomp(cls, max_register_size):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.
        """
        rx = resource_rep(RX, {"x": 1})
        return [GateCount(rx, max_register_size)]

    @classmethod
    def controlled_resource_decomp(cls, num_ctrl_wires, num_zero_ctrl, target_resource_params):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.
        """
        max_register_size = target_resource_params["max_register_size"]
        cnot = resource_rep(CNOT, {"x": None})
        rx = resource_rep(RX, {"x": 1})
        return [GateCount(cnot, max_register_size), GateCount(rx, max_register_size)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.
        """
        max_register_size = target_resource_params["max_register_size"]
        rx = resource_rep(RX, {"x": 1})
        h = resource_rep(Hadamard, {"x": None})
        return [GateCount(rx, max_register_size), GateCount(h, 2 * max_register_size)]

    @classmethod
    def pow_resource_decomp(cls, pow_z, target_resource_params):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.
        """
        max_register_size = target_resource_params["max_register_size"]
        rx = resource_rep(RX, {"x": 1})
        return [GateCount(rx, max_register_size * pow_z)]


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

    def test_equality_method(self):
        """Test that the __eq__ method for the ResourceOperator is correct."""

        dop1 = DummyOp(1.1)
        dop2 = DummyOp(2.2)
        dop3 = DummyOp(1.1)
        dop1.num_wires = 1
        dop3.num_wires = 3

        assert qre_ops.X() == qre_ops.X()
        assert qre_ops.SWAP() == qre_ops.SWAP()
        assert qre_ops.X() != qre_ops.SWAP()
        assert dop1 != dop2
        assert dop1 != dop3

    def test_equality_false(self):
        """Test that the __eq__ method returns False if the input operator is not ResourceOperator."""
        assert not qre_ops.X() == qml.X(0)

    def test_hash(self):
        """Test that the hash method behaves as expected"""
        QSVT1 = DummyOp(x={"num_wires": 3, "num_angles": 5})
        QSVT2 = DummyOp(x={"num_wires": 3, "num_angles": 5})
        QSVT3 = DummyOp(x={"num_angles": 5, "num_wires": 3})
        Other = DummyOp(x={"num_wires": 3})

        QSVT1.num_wires = 3
        QSVT2.num_wires = 3
        QSVT3.num_wires = 3
        Other.num_wires = 3

        assert hash(QSVT1) == hash(QSVT1)  # compare same object
        assert hash(QSVT1) == hash(QSVT2)  # compare identical instance
        assert hash(QSVT1) == hash(QSVT3)  # compare identical instance with different key order
        assert hash(QSVT1) != hash(Other)

    ops_to_queue = [
        Hadamard(wires=[0]),
        Hadamard(wires=[1]),
        CNOT(wires=[0, 1]),
        RX(x=1.23, wires=[0]),
    ]

    def test_init_queuing(self):
        """Test that instantiating a resource operator correctly sets its arguments
        and queues it."""
        with AnnotatedQueue() as q:
            Hadamard(wires=[0])
            Hadamard(wires=[1])
            CNOT(wires=[0, 1])
            RX(x=1.23, wires=[0])

        assert q.queue == self.ops_to_queue

    def test_dequeue(self):
        """Test that we can remove a resource operator correctly."""
        ops_to_remove = (
            self.ops_to_queue[0],
            [self.ops_to_queue[2]],
            self.ops_to_queue[0:3],
        )

        expected_queues = (
            self.ops_to_queue[1:],
            self.ops_to_queue[:2] + self.ops_to_queue[3:],
            self.ops_to_queue[3:],
        )

        for op_to_remove, expected_queue in zip(ops_to_remove, expected_queues):
            with AnnotatedQueue() as q:
                for op in self.ops_to_queue:
                    op.queue()

                _dequeue(op_to_remove)

            assert q.queue == expected_queue

    @pytest.mark.parametrize("s", [1, 2, 3])
    def test_mul(self, s):
        """Test multiply dunder method"""
        op = RX(1.23)
        resources = s * op

        gt = defaultdict(int, {RX(1.23): s})
        expected_resources = Resources(0, algo_wires=1, gate_types=gt)
        assert resources == expected_resources

    @pytest.mark.parametrize("s", [1, 2, 3])
    def test_mat_mul(self, s):
        """Test matrix-multiply dunder method"""
        op = CNOT()
        resources = s @ op

        gt = defaultdict(int, {CNOT(): s})
        expected_resources = Resources(0, algo_wires=s * 2, gate_types=gt)
        assert resources == expected_resources

    def test_add_series(self):
        """Test addition dunder method between two ResourceOperator classes"""
        op1 = RX(1.23)
        op2 = CNOT()
        resources = op1.add_series(op2)

        gt = defaultdict(
            int,
            {
                RX(1.23): 1,
                CNOT(): 1,
            },
        )
        expected_resources = Resources(zeroed_wires=0, algo_wires=2, gate_types=gt)
        assert resources == expected_resources

    def test_add_series_resources(self):
        """Test addition dunder method between a ResourceOperator and a Resources object"""
        op1 = RX(1.23)
        gt2 = defaultdict(int, {CNOT(): 1})
        res2 = Resources(zeroed_wires=0, algo_wires=2, gate_types=gt2)
        resources = op1.add_series(res2)

        gt = defaultdict(
            int,
            {
                RX(1.23): 1,
                CNOT(): 1,
            },
        )
        expected_resources = Resources(zeroed_wires=0, algo_wires=2, gate_types=gt)
        assert resources == expected_resources

    def test_add_series_error(self):
        """Test addition dunder method raises error when adding with unsupported type"""
        with pytest.raises(TypeError, match="Cannot add resource operator"):
            op1 = RX(1.23)
            _ = op1.add_series(True)

    def test_add_parallel(self):
        """Test add_parallel method between two ResourceOperator classes"""
        op1 = RX(1.23)
        op2 = CNOT()
        resources = op1.add_parallel(op2)

        gt = defaultdict(
            int,
            {
                RX(1.23): 1,
                CNOT(): 1,
            },
        )
        expected_resources = Resources(zeroed_wires=0, algo_wires=3, gate_types=gt)
        assert resources == expected_resources

    def test_add_parallel_resources(self):
        """Test and dunder method between a ResourceOperator and a Resources object"""
        op1 = RX(1.23)
        gt2 = defaultdict(int, {CNOT(): 1})
        res2 = Resources(zeroed_wires=0, any_state_wires=0, algo_wires=2, gate_types=gt2)
        resources = op1.add_parallel(res2)

        gt = defaultdict(
            int,
            {
                RX(1.23): 1,
                CNOT(): 1,
            },
        )
        expected_resources = Resources(zeroed_wires=0, algo_wires=3, gate_types=gt)
        assert resources == expected_resources

    def test_parallel_add_error(self):
        """Test add_parallel method raises error when adding with unsupported type"""
        with pytest.raises(TypeError, match="Cannot add resource operator"):
            op1 = RX(1.23)
            _ = op1.add_parallel(True)

    def test_mul_error(self):
        """Test multiply dunder method raises error when multiplying with unsupported type"""
        with pytest.raises(TypeError, match="Cannot multiply resource operator"):
            op1 = RX(1.23)
            _ = op1 * 0.2

    def test_matmul_error(self):
        """Test multiply dunder method raises error when multiplying with unsupported type"""
        with pytest.raises(TypeError, match="Cannot multiply resource operator"):
            op1 = RX(1.23)
            _ = op1 @ 0.2

    def test_default_resource_keys(self):
        """Test that default resource keys returns the correct result."""
        op1 = X
        assert op1.resource_keys == set()  # pylint: disable=comparison-with-callable

    def test_adjoint_resource_decomp(self):
        """Test that default adjoint operator returns the correct error."""
        dummy_params = {"max_register_size": 10}
        assert DummyOp_decomps.adjoint_resource_decomp(target_resource_params=dummy_params) == [
            GateCount(RX(1), 10),
            GateCount(Hadamard(None), 20),
        ]

    def test_controlled_resource_decomp(self):
        """Test that default controlled operator returns the correct error."""
        dummy_params = {"max_register_size": 10}
        assert DummyOp_decomps.controlled_resource_decomp(
            num_ctrl_wires=2, num_zero_ctrl=0, target_resource_params=dummy_params
        ) == [GateCount(CNOT(None), 10), GateCount(RX(1), 10)]

    def test_pow_resource_decomp(self):
        """Test that default power operator returns the correct error."""

        dummy_params = {"max_register_size": 10}
        assert DummyOp_decomps.pow_resource_decomp(
            pow_z=2, target_resource_params=dummy_params
        ) == [GateCount(RX(1), 20)]

    def test_tracking_name(self):
        """Test that correct tracking name is returned."""
        assert X().tracking_name() == "X"


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


def test_make_hashable_error():
    """Test that _make_hashable raises the correct error"""
    with pytest.raises(
        TypeError,
        match="Object of type <class 'dict_keys'> is not hashable and cannot be converted.",
    ):
        _make_hashable({"a": 1}.keys())


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
        assert gc1 != op

    def test_add_method(self):
        """Test that the arithmetic methods work as expected"""
        op = DummyOp(x=5)
        op1 = DummyOp(x=6)

        gc = GateCount(op, count=3)
        gc1 = GateCount(op, count=2)
        assert gc + gc1 == GateCount(op, count=5)

        with pytest.raises(NotImplementedError):
            gc2 = GateCount(op1, count=2)
            _ = gc + gc2

    def test_mul_method(self):
        """Test that the arithmetic methods work as expected"""
        op = DummyOp(x=5)
        gc = GateCount(op, count=3)

        assert gc * 5 == GateCount(op, count=15)

        with pytest.raises(NotImplementedError):
            _ = gc * 0.5

    def test_repr(self):
        """Test that the repr works as expected"""
        op = DummyX(1, {"num_wires": 1})
        gc = GateCount(op, count=3)

        assert repr(gc) == "(3 x DummyX)"


def test_resource_rep():
    """Test that the resource_rep method works as expected"""

    class ResourceOpA(ResourceOperator):
        """Test resource op class"""

        resource_keys = {"num_wires", "continuous_param", "bool_param"}

        def __init__(self, num_wires, continuous_param, bool_param, wires=None):
            self.num_wires = num_wires
            self.continuous_param = continuous_param
            self.bool_param = bool_param
            super().__init__(wires=wires)

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
            return cls(num_wires, continuous_param, bool_param)

        @classmethod
        def resource_decomp(cls, num_wires, continuous_param, bool_param):
            """dummy default resource decomp method"""
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
            return cls()

        @classmethod
        def resource_decomp(cls, **kwargs):
            """dummy default resource decomp method"""
            raise NotImplementedError

    expected_resource_rep_A = ResourceOpA(
        num_wires=1,
        continuous_param=2.34,
        bool_param=False,
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

    expected_resource_rep_B = ResourceOpB()
    actual_resource_rep_B = resource_rep(ResourceOpB)
    assert expected_resource_rep_B == actual_resource_rep_B
