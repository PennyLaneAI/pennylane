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
Test the core resource tracking functionality.
"""
from collections import defaultdict

import pytest

from pennylane.labs.resource_estimation.ops.qubit.parametric_ops_single_qubit import (
    ResourceRX,
    ResourceRY,
    ResourceRZ,
)
from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires, QubitManager
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    ResourcesNotDefined,
    resource_rep,
)
from pennylane.labs.resource_estimation.resource_tracking import ResourceConfig, estimate
from pennylane.labs.resource_estimation.resources_base import Resources

# pylint: disable= no-self-use, arguments-differ


class ResourceTestCNOT(ResourceOperator):
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
        raise ResourcesNotDefined


class ResourceTestHadamard(ResourceOperator):
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
        raise ResourcesNotDefined


class ResourceTestT(ResourceOperator):
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
        raise ResourcesNotDefined


class ResourceTestZ(ResourceOperator):
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
        t = resource_rep(ResourceTestT)
        return [GateCount(t, count=4)]


class ResourceTestRZ(ResourceOperator):
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
        t = resource_rep(ResourceTestT)
        t_counts = round(1 / precision)
        return [GateCount(t, count=t_counts)]


class ResourceTestAlg1(ResourceOperator):
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
    def resource_decomp(cls, num_iter, **kwargs):
        cnot = resource_rep(ResourceTestCNOT)
        h = resource_rep(ResourceTestHadamard)

        return [
            AllocWires(num_wires=num_iter),
            GateCount(h, num_iter),
            GateCount(cnot, num_iter),
            FreeWires(num_wires=num_iter - 1),
        ]


class ResourceTestAlg2(ResourceOperator):
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
    def resource_decomp(cls, num_wires, **kwargs):
        rz = resource_rep(ResourceTestRZ, {"precision": 1e-2})
        alg1 = resource_rep(ResourceTestAlg1, {"num_iter": 3})

        return [
            AllocWires(num_wires=num_wires),
            GateCount(rz, num_wires),
            GateCount(alg1, num_wires // 2),
            FreeWires(num_wires=num_wires),
        ]


def mock_rotation_decomp(precision):
    """A mock decomposition for rotation gates returning TestT gates for testing."""
    t = resource_rep(ResourceTestT)
    t_counts = round(1 / precision)
    return [GateCount(t, count=t_counts)]


class TestEstimateResources:
    """Test that core resource estimation functionality"""

    def test_estimate_resources_from_qfunc(self):
        """Test that we can accurately obtain resources from qfunc"""

        def my_circuit():
            for w in range(5):
                ResourceTestHadamard(wires=[w])
            ResourceTestCNOT(wires=[0, 1])
            ResourceTestRZ(wires=[1])
            ResourceTestRZ(precision=1e-2, wires=[2])
            ResourceTestCNOT(wires=[3, 4])
            ResourceTestAlg1(num_iter=5, wires=[5, 6])

        expected_gates = defaultdict(
            int,
            {
                resource_rep(ResourceTestT): round(1 / 1e-2) + round(1 / 1e-9),
                resource_rep(ResourceTestCNOT): 7,
                resource_rep(ResourceTestHadamard): 10,
            },
        )
        expected_qubits = QubitManager(work_wires={"clean": 4, "dirty": 1}, algo_wires=7)
        expected_resources = Resources(qubit_manager=expected_qubits, gate_types=expected_gates)

        gate_set = {"TestCNOT", "TestT", "TestHadamard"}
        custom_config = ResourceConfig()
        custom_config.resource_op_precisions[ResourceTestRZ] = {"precision": 1e-9}
        computed_resources = estimate(my_circuit, gate_set=gate_set, config=custom_config)()
        assert computed_resources == expected_resources

    def test_estimate_resources_from_resource_operator(self):
        """Test that we can accurately obtain resources from qfunc"""
        op = ResourceTestAlg2(num_wires=4)
        actual_resources = estimate(op, gate_set={"TestRZ", "TestAlg1"})

        expected_gates = defaultdict(
            int,
            {
                resource_rep(ResourceTestRZ, {"precision": 1e-2}): 4,
                resource_rep(ResourceTestAlg1, {"num_iter": 3}): 2,
            },
        )
        expected_qubits = QubitManager(work_wires=4, algo_wires=4)
        expected_resources = Resources(qubit_manager=expected_qubits, gate_types=expected_gates)

        assert actual_resources == expected_resources

    def test_estimate_resources_from_resources_obj(self):
        """Test that we can accurately obtain resources from qfunc"""
        gates = defaultdict(
            int,
            {
                resource_rep(ResourceTestRZ, {"precision": 1e-2}): 4,
                resource_rep(ResourceTestAlg1, {"num_iter": 3}): 2,
            },
        )
        qubits = QubitManager(work_wires=0, algo_wires=4)
        resources = Resources(qubit_manager=qubits, gate_types=gates)

        gate_set = {"TestCNOT", "TestT", "TestHadamard"}
        actual_resources = estimate(resources, gate_set=gate_set)

        expected_gates = defaultdict(
            int,
            {
                resource_rep(ResourceTestT): 4 * round(1 / 1e-2),
                resource_rep(ResourceTestCNOT): 6,
                resource_rep(ResourceTestHadamard): 6,
            },
        )
        expected_qubits = QubitManager(
            work_wires={"clean": 4, "dirty": 2}, algo_wires=4
        )  # TODO: optimize allocation
        expected_resources = Resources(qubit_manager=expected_qubits, gate_types=expected_gates)

        assert actual_resources == expected_resources

    def test_estimate_resources_from_pl_operator(self):
        """Test that we can accurately obtain resources from qfunc"""
        assert True

    @pytest.mark.parametrize(
        "gate_set, expected_resources",
        (
            (
                {"TestRZ", "TestAlg1", "TestZ"},
                Resources(
                    qubit_manager=QubitManager(work_wires=4, algo_wires=4),
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(ResourceTestRZ, {"precision": 1e-2}): 4,
                            resource_rep(ResourceTestAlg1, {"num_iter": 3}): 2,
                            resource_rep(ResourceTestZ): 4,
                        },
                    ),
                ),
            ),
            (
                {"TestCNOT", "TestT", "TestHadamard"},
                Resources(
                    qubit_manager=QubitManager(work_wires={"clean": 8, "dirty": 2}, algo_wires=4),
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(ResourceTestT): 416,
                            resource_rep(ResourceTestCNOT): 6,
                            resource_rep(ResourceTestHadamard): 6,
                        },
                    ),
                ),
            ),
        ),
    )
    def test_varying_gate_sets(self, gate_set, expected_resources):
        """Test that changing the gate_set correctly updates the resources"""

        def my_circ(num_wires):
            ResourceTestAlg2(num_wires, wires=range(num_wires))
            for w in range(num_wires):
                ResourceTestZ(wires=w)

        actual_resources = estimate(my_circ, gate_set=gate_set)(num_wires=4)
        assert actual_resources == expected_resources

    @pytest.mark.parametrize("error_val", (0.1, 0.01, 0.001))
    def test_varying_config(self, error_val):
        """Test that changing the resource_config correctly updates the resources"""
        custom_config = ResourceConfig()
        custom_config.resource_op_precisions[ResourceTestRZ] = {"precision": error_val}

        op = ResourceTestRZ()  # don't specify precision
        computed_resources = estimate(op, gate_set={"TestT"}, config=custom_config)

        expected_resources = Resources(
            qubit_manager=QubitManager(work_wires=0, algo_wires=1),
            gate_types=defaultdict(int, {resource_rep(ResourceTestT): round(1 / error_val)}),
        )

        assert computed_resources == expected_resources

    @pytest.mark.parametrize("error_val", (0.1, 0.01, 0.001))
    def test_varying_single_qubit_rotation_precision(self, error_val):
        """Test that setting the single_qubit_rotation_precision correctly updates the resources"""
        custom_config = ResourceConfig()
        custom_config.set_single_qubit_rot_precision(error_val)

        custom_config.set_decomp(ResourceRX, mock_rotation_decomp)
        custom_config.set_decomp(ResourceRY, mock_rotation_decomp)
        custom_config.set_decomp(ResourceRZ, mock_rotation_decomp)

        def my_circuit():
            ResourceRX(wires=0)
            ResourceRY(wires=1)
            ResourceRZ(wires=2)

        computed_resources = estimate(my_circuit, gate_set={"TestT"}, config=custom_config)()

        expected_t_count = 3 * round(1 / error_val)
        expected_resources = Resources(
            qubit_manager=QubitManager(work_wires=0, algo_wires=3),
            gate_types=defaultdict(int, {resource_rep(ResourceTestT): expected_t_count}),
        )

        assert computed_resources == expected_resources
