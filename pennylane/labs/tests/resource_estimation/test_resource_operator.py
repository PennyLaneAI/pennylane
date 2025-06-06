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
from typing import List

import pytest

from pennylane.labs.resource_estimation import (
    QubitManager,
    ResourceOperator,
    Resources,
    set_adj_decomp,
    set_ctrl_decomp,
    set_decomp,
    set_pow_decomp,
)
from pennylane.queuing import AnnotatedQueue
from pennylane.wires import Wires

# pylint: disable=protected-access, too-few-public-methods, no-self-use, unused-argument, arguments-differ


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
