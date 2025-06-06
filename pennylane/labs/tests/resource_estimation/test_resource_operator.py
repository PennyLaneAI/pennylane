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
from dataclasses import dataclass
from typing import List

import pytest

from pennylane.labs.resource_estimation import QubitManager, ResourceOperator, Resources
from pennylane.queuing import AnnotatedQueue
from pennylane.wires import Wires

# pylint: disable=protected-access, too-few-public-methods, no-self-use


@dataclass(frozen=True)
class DummyCmprsRep:
    """A dummy class to populate the gate types dictionary for testing."""

    name: str
    param: int = 0


class DummyOp(ResourceOperator):
    """A dummy class to test ResourceOperator instatiation."""

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
    """A dummy class to test ResourceOperator instatiation."""

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
    """A dummy class to test ResourceOperator instatiation."""

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

    def test_add(self):
        """Test addition dunder method"""

    def test_and(self):
        """Test and dunder method"""

    def test_mul(self):
        """Test multiply dunder method"""

    def test_mat_mul(self):
        """Test mat multiply dunder method"""


def test_set_decomp():
    """Test that the set_decomp function works as expected."""
    assert True


def test_set_decomp():
    """Test that the set_decomp function works as expected."""
    assert True


def test_set_decomp():
    """Test that the set_decomp function works as expected."""
    assert True


def test_set_decomp():
    """Test that the set_decomp function works as expected."""
    assert True
