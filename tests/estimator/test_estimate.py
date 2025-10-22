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
Test the core resource estimation functionality.
"""
from collections import defaultdict

import pytest

import pennylane as qml
from pennylane.estimator.estimate import estimate
from pennylane.estimator.ops.qubit.non_parametric_ops import Hadamard, X
from pennylane.estimator.ops.qubit.parametric_ops_single_qubit import RX
from pennylane.estimator.resource_config import ResourceConfig
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    ResourcesUndefinedError,
    resource_rep,
)
from pennylane.estimator.resources_base import Resources
from pennylane.estimator.wires_manager import Allocate, Deallocate

# pylint: disable= no-self-use, arguments-differ


def _circuit_w_expval(circ):
    circ()
    return qml.expval(qml.Z(0))


def _circuit_w_sample(circ):
    circ()
    return qml.sample(wires=[0])


def _circuit_w_probs(circ):
    circ()
    return qml.probs()


def _circuit_w_state(circ):
    circ()
    return qml.state()


class TestCNOT(ResourceOperator):
    """Dummy class for testing"""

    num_wires = 2
    resource_keys = {}

    @classmethod
    def resource_rep(cls):
        return CompressedResourceOp(cls, 2, {})

    @property
    def resource_params(self):
        return {}

    @classmethod
    def resource_decomp(cls, **kwargs):
        raise ResourcesUndefinedError


class TestHadamard(ResourceOperator):
    """Dummy class for testing"""

    num_wires = 1
    resource_keys = {}

    @classmethod
    def resource_rep(cls):
        return CompressedResourceOp(cls, 1, {})

    @property
    def resource_params(self):
        return {}

    @classmethod
    def resource_decomp(cls, **kwargs):
        raise ResourcesUndefinedError


class TestT(ResourceOperator):
    """Dummy class for testing"""

    num_wires = 1
    resource_keys = {}

    @classmethod
    def resource_rep(cls):
        return CompressedResourceOp(cls, 1, {})

    @property
    def resource_params(self):
        return {}

    @classmethod
    def resource_decomp(cls, **kwargs):
        raise ResourcesUndefinedError


class TestZ(ResourceOperator):
    """Dummy class for testing"""

    num_wires = 1
    resource_keys = {}

    @classmethod
    def resource_rep(cls):
        return CompressedResourceOp(cls, 1, {})

    @property
    def resource_params(self):
        return {}

    @classmethod
    def resource_decomp(cls):
        t = resource_rep(TestT)
        return [GateCount(t, count=4)]


class TestRZ(ResourceOperator):
    """Dummy class for testing"""

    num_wires = 1
    resource_keys = {"precision"}

    def __init__(self, precision=None, wires=None) -> None:
        self.precision = precision
        super().__init__(wires=wires)

    @classmethod
    def resource_rep(cls, precision=None):
        return CompressedResourceOp(cls, 1, {"precision": precision})

    @property
    def resource_params(self):
        return {"precision": self.precision}

    @classmethod
    def resource_decomp(cls, precision):
        t = resource_rep(TestT)
        t_counts = round(1 / precision)
        return [GateCount(t, count=t_counts)]


class TestAlg1(ResourceOperator):
    """Dummy class for testing"""

    num_wires = 2
    resource_keys = {"num_iter"}

    def __init__(self, num_iter, wires=None) -> None:
        self.num_iter = num_iter
        super().__init__(wires=wires)

    @classmethod
    def resource_rep(cls, num_iter):
        return CompressedResourceOp(cls, 2, {"num_iter": num_iter})

    @property
    def resource_params(self):
        return {"num_iter": self.num_iter}

    @classmethod
    def resource_decomp(cls, num_iter):
        cnot = resource_rep(TestCNOT)
        h = resource_rep(TestHadamard)

        return [
            Allocate(num_wires=num_iter),
            GateCount(h, num_iter),
            GateCount(cnot, num_iter),
            Deallocate(num_wires=num_iter - 1),
        ]


class TestAlg2(ResourceOperator):
    """Dummy class for testing"""

    resource_keys = {"num_wires"}

    def __init__(self, num_wires, wires=None) -> None:
        self.num_wires = num_wires
        super().__init__(wires=wires)

    @classmethod
    def resource_rep(cls, num_wires):
        return CompressedResourceOp(cls, num_wires, {"num_wires": num_wires})

    @property
    def resource_params(self):
        return {"num_wires": self.num_wires}

    @classmethod
    def resource_decomp(cls, num_wires):
        rz = resource_rep(TestRZ, {"precision": 1e-2})
        alg1 = resource_rep(TestAlg1, {"num_iter": 3})

        return [
            Allocate(num_wires=num_wires),
            GateCount(rz, num_wires),
            GateCount(alg1, num_wires // 2),
            Deallocate(num_wires=num_wires),
        ]


def mock_rotation_decomp(precision):
    """A mock decomposition for rotation gates returning TestT gates for testing."""
    t = resource_rep(TestT)
    t_counts = round(1 / precision)
    return [GateCount(t, count=t_counts)]


class TestEstimateResources:
    """Test that core resource estimation functionality"""

    def test_estimate_with_unsupported_dispatch(self):
        """Test that a TypeError is raised when an unsupported type is passed to the estimate function."""
        with pytest.raises(TypeError, match="Could not obtain resources for workflow of type"):
            estimate(({1, 2, 3}))

    def test_estimate_qnode(self):
        """Test that a QNode can be estimated."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(precision):
            qml.Hadamard(wires=0)
            X()
            RX(precision=precision, wires=0)
            return qml.expval(qml.Z(0))

        resources = estimate(circuit, gate_set={"Hadamard", "X", "RX"})(1e-3)

        assert resources.algo_wires == 2
        assert resources.gate_counts["Hadamard"] == 1
        assert resources.gate_counts["X"] == 1
        assert resources.gate_counts["RX"] == 1
        assert resources.gate_types[RX.resource_rep(precision=1e-3)] == 1

    def test_qfunc_with_num_wires(self):
        """Test that the number of wires is correctly inferred from a qfunc
        when the num_wires argument is used."""

        def my_circuit():
            TestAlg2(num_wires=4)

        res = estimate(my_circuit, gate_set={"TestAlg2"})()
        expected = Resources(
            zeroed_wires=0,
            any_state_wires=0,
            algo_wires=4,
            gate_types={resource_rep(TestAlg2, {"num_wires": 4}): 1},
        )
        assert res == expected

    def test_gate_set_is_none(self):
        """Test that the default gate set is used when gate_set is None."""
        op = X()
        res = estimate(op, gate_set=None)
        expected = Resources(
            zeroed_wires=0,
            any_state_wires=0,
            algo_wires=1,
            gate_types={resource_rep(X): 1},
        )
        assert res == expected

    def test_estimate_resources_from_qfunc(self):
        """Test that we can accurately obtain resources from qfunc"""

        def my_circuit():
            for w in range(5):
                TestHadamard(wires=[w])
            TestCNOT(wires=[0, 1])
            TestRZ(wires=[1])
            TestRZ(precision=1e-2, wires=[2])
            TestCNOT(wires=[3, 4])
            TestAlg1(num_iter=5, wires=[5, 6])

        # See implementation of TestRZ to see how it decomposes down to
        # TestT based on the precision value, hence why
        # round(1/1e-2) + round(1/1e-9)
        expected_gates = defaultdict(
            int,
            {
                resource_rep(TestT): round(1 / 1e-2) + round(1 / 1e-9),
                resource_rep(TestCNOT): 7,
                resource_rep(TestHadamard): 10,
            },
        )
        expected_resources = Resources(
            zeroed_wires=4, any_state_wires=1, algo_wires=7, gate_types=expected_gates
        )

        gate_set = {"TestCNOT", "TestT", "TestHadamard"}
        custom_config = ResourceConfig()
        custom_config.resource_op_precisions[TestRZ] = {"precision": 1e-9}
        computed_resources = estimate(my_circuit, gate_set=gate_set, config=custom_config)()
        assert computed_resources == expected_resources

    def test_estimate_resources_from_resource_operator(self):
        """Test that we can accurately obtain resources from resource operator"""
        op = TestAlg2(num_wires=4)
        actual_resources = estimate(op, gate_set={"TestRZ", "TestAlg1"})

        expected_gates = defaultdict(
            int,
            {
                resource_rep(TestRZ, {"precision": 1e-2}): 4,
                resource_rep(TestAlg1, {"num_iter": 3}): 2,
            },
        )
        expected_resources = Resources(zeroed_wires=4, algo_wires=4, gate_types=expected_gates)
        assert actual_resources == expected_resources

    def test_estimate_resources_from_scaled_resource_operator(self):
        """Test that we can accurately obtain resources from resource operator"""
        op = 2 * TestAlg2(num_wires=4)
        actual_resources = estimate(op, gate_set={"TestRZ", "TestAlg1"})

        expected_gates = defaultdict(
            int,
            {
                resource_rep(TestRZ, {"precision": 1e-2}): 8,
                resource_rep(TestAlg1, {"num_iter": 3}): 4,
            },
        )
        expected_resources = Resources(zeroed_wires=4, algo_wires=4, gate_types=expected_gates)
        assert actual_resources == expected_resources

    def test_estimate_resources_from_resources_obj(self):
        """Test that we can accurately obtain resources from resources workflow"""
        gates = defaultdict(
            int,
            {
                resource_rep(TestRZ, {"precision": 1e-2}): 4,
                resource_rep(TestAlg1, {"num_iter": 3}): 2,
            },
        )
        resources = Resources(zeroed_wires=0, algo_wires=4, gate_types=gates)

        gate_set = {"TestCNOT", "TestT", "TestHadamard"}
        actual_resources = estimate(resources, gate_set=gate_set)

        expected_gates = defaultdict(
            int,
            {
                resource_rep(TestT): 4 * round(1 / 1e-2),
                resource_rep(TestCNOT): 6,
                resource_rep(TestHadamard): 6,
            },
        )
        # TODO: optimize allocation
        expected_resources = Resources(
            zeroed_wires=4, any_state_wires=2, algo_wires=4, gate_types=expected_gates
        )

        assert actual_resources == expected_resources

    def test_unsupported_object_in_queue_raises_error(self):
        """Test that a ValueError is raised for unsupported objects in the queue."""

        def my_circuit():
            qml.QueuingManager.append(0)  # Arbitrarily queue something

        with pytest.raises(
            ValueError,
            match="Queued object.*is not a ResourceOperator or Operator, and cannot be processed.",
        ):
            estimate(my_circuit)()

    def test_estimate_resources_from_qfunc_with_pl_op(self):
        """Test that PennyLane operators are correctly mapped to resource operators
        when processing a qfunc."""

        def my_circuit():
            qml.Hadamard(0)
            qml.PauliX(1)

        actual_resources = estimate(my_circuit, gate_set={"Hadamard", "X"})()

        expected_gates = defaultdict(
            int,
            {
                resource_rep(Hadamard): 1,
                resource_rep(X): 1,
            },
        )
        expected_resources = Resources(
            zeroed_wires=0, any_state_wires=0, algo_wires=2, gate_types=expected_gates
        )
        assert actual_resources == expected_resources

    def test_estimate_resources_from_pl_op_dispatch(self):
        """Test that the dispatch for PennyLane operators correctly maps to a
        ResourceOperator without further decomposition."""
        op = qml.PauliX(0)

        actual_resources = estimate(op)

        expected_gates = defaultdict(int, {resource_rep(X): 1})
        expected_resources = Resources(
            zeroed_wires=0, any_state_wires=0, algo_wires=1, gate_types=expected_gates
        )

        assert actual_resources == expected_resources

    @pytest.mark.parametrize(
        "gate_set, expected_resources",
        (
            (
                {"TestRZ", "TestAlg1", "TestZ"},
                Resources(
                    zeroed_wires=4,
                    algo_wires=4,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(TestRZ, {"precision": 1e-2}): 4,
                            resource_rep(TestAlg1, {"num_iter": 3}): 2,
                            resource_rep(TestZ): 4,
                        },
                    ),
                ),
            ),
            (
                {"TestCNOT", "TestT", "TestHadamard"},
                Resources(
                    zeroed_wires=8,
                    any_state_wires=2,
                    algo_wires=4,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(TestT): 416,
                            resource_rep(TestCNOT): 6,
                            resource_rep(TestHadamard): 6,
                        },
                    ),
                ),
            ),
        ),
    )
    def test_varying_gate_sets(self, gate_set, expected_resources):
        """Test that changing the gate_set correctly updates the resources"""

        def my_circ(num_wires):
            TestAlg2(num_wires, wires=range(num_wires))
            for w in range(num_wires):
                TestZ(wires=w)

        actual_resources = estimate(my_circ, gate_set=gate_set)(num_wires=4)
        assert actual_resources == expected_resources

    @pytest.mark.parametrize("error_val", (0.1, 0.01, 0.001))
    def test_varying_config(self, error_val):
        """Test that changing the resource_config correctly updates the resources"""
        custom_config = ResourceConfig()
        custom_config.resource_op_precisions[TestRZ] = {"precision": error_val}

        op = TestRZ()  # don't specify precision
        computed_resources = estimate(op, gate_set={"TestT"}, config=custom_config)

        expected_resources = Resources(
            zeroed_wires=0,
            algo_wires=1,
            gate_types=defaultdict(int, {resource_rep(TestT): round(1 / error_val)}),
        )

        assert computed_resources == expected_resources

    measurement_circuits = (
        _circuit_w_expval,
        _circuit_w_sample,
        _circuit_w_probs,
        _circuit_w_state,
    )

    @pytest.mark.parametrize("circ_w_measurement", measurement_circuits)
    def test_estimate_ignores_measurementprocess(self, circ_w_measurement):
        """Test that the estimate function ignores measurement processes"""

        def circ():
            qml.Hadamard(wires=[0])
            qml.X(wires=[1])
            qml.RX(1.23, wires=[0])
            qml.CNOT(wires=[0, 1])

        assert estimate(circ)() == estimate(circ_w_measurement)(circ)
