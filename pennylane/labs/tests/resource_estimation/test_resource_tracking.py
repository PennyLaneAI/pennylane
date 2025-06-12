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
import copy
from collections import defaultdict

import pytest

from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires, QubitManager
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    ResourcesNotDefined,
    resource_rep,
)
from pennylane.labs.resource_estimation.resource_tracking import estimate_resources, resource_config
from pennylane.labs.resource_estimation.resources_base import Resources

# pylint: disable= no-self-use, arguments-differ
<<<<<<< HEAD


class ResourceTestCNOT(ResourceOperator):
    """Dummy class for testing"""

    num_wires = 2
    resource_keys = {}

    @classmethod
    def resource_rep(cls):
        return CompressedResourceOp(cls, {})

    @property
    def resource_params(self):
        return {}

    @classmethod
    def default_resource_decomp(cls, **kwargs):
        raise ResourcesNotDefined


class ResourceTestHadamard(ResourceOperator):
    """Dummy class for testing"""

    num_wires = 1
    resource_keys = {}

    @classmethod
    def resource_rep(cls):
        return CompressedResourceOp(cls, {})

    @property
    def resource_params(self):
        return {}

    @classmethod
    def default_resource_decomp(cls, **kwargs):
        raise ResourcesNotDefined


class ResourceTestT(ResourceOperator):
    """Dummy class for testing"""

    num_wires = 1
    resource_keys = {}

    @classmethod
    def resource_rep(cls):
        return CompressedResourceOp(cls, {})

    @property
    def resource_params(self):
        return {}

    @classmethod
    def default_resource_decomp(cls, **kwargs):
        raise ResourcesNotDefined


class ResourceTestZ(ResourceOperator):
    """Dummy class for testing"""

    num_wires = 1
    resource_keys = {}

    @classmethod
    def resource_rep(cls):
        return CompressedResourceOp(cls, {})

    @property
    def resource_params(self):
        return {}

    @classmethod
    def default_resource_decomp(cls, **kwargs):
        t = resource_rep(ResourceTestT)
        return [GateCount(t, count=4)]


class ResourceTestRZ(ResourceOperator):
    """Dummy class for testing"""

    num_wires = 1
    resource_keys = {"epsilon"}

    def __init__(self, epsilon=None, wires=None) -> None:
        self.epsilon = epsilon
        super().__init__(wires=wires)

    @classmethod
    def resource_rep(cls, epsilon=None):
        return CompressedResourceOp(cls, {"epsilon": epsilon})

    @property
    def resource_params(self):
        return {"epsilon": self.epsilon}

    @classmethod
    def default_resource_decomp(cls, epsilon, **kwargs):
        if epsilon is None:
            epsilon = kwargs["config"]["error_rz"]

        t = resource_rep(ResourceTestT)
        t_counts = round(1 / epsilon)
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
        return CompressedResourceOp(cls, {"num_iter": num_iter})

    @property
    def resource_params(self):
        return {"num_iter": self.num_iter}

    @classmethod
    def default_resource_decomp(cls, num_iter, **kwargs):
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
        return CompressedResourceOp(cls, {"num_wires": num_wires})

    @property
    def resource_params(self):
        return {"num_wires": self.num_wires}

    @classmethod
    def default_resource_decomp(cls, num_wires, **kwargs):
        rz = resource_rep(ResourceTestRZ, {"epsilon": 1e-2})
        alg1 = resource_rep(ResourceTestAlg1, {"num_iter": 3})

        return [
            AllocWires(num_wires=num_wires),
            GateCount(rz, num_wires),
            GateCount(alg1, num_wires // 2),
            FreeWires(num_wires=num_wires),
        ]
=======
>>>>>>> f3bb59979 (code review comments + changelog)


class ResourceTestCNOT(ResourceOperator):
    """Dummy class for testing"""

    num_wires = 2
    resource_keys = {}

    @classmethod
    def resource_rep(cls):
        return CompressedResourceOp(cls, {})

    @property
    def resource_params(self):
        return {}

    @classmethod
    def default_resource_decomp(cls, **kwargs):
        raise ResourcesNotDefined


class ResourceTestHadamard(ResourceOperator):
    """Dummy class for testing"""

    num_wires = 1
    resource_keys = {}

    @classmethod
    def resource_rep(cls):
        return CompressedResourceOp(cls, {})

    @property
    def resource_params(self):
        return {}

    @classmethod
    def default_resource_decomp(cls, **kwargs):
        raise ResourcesNotDefined


class ResourceTestT(ResourceOperator):
    """Dummy class for testing"""

    num_wires = 1
    resource_keys = {}

    @classmethod
    def resource_rep(cls):
        return CompressedResourceOp(cls, {})

    @property
    def resource_params(self):
        return {}

    @classmethod
    def default_resource_decomp(cls, **kwargs):
        raise ResourcesNotDefined


class ResourceTestZ(ResourceOperator):
    """Dummy class for testing"""

    num_wires = 1
    resource_keys = {}

    @classmethod
    def resource_rep(cls):
        return CompressedResourceOp(cls, {})

    @property
    def resource_params(self):
        return {}

    @classmethod
    def default_resource_decomp(cls, **kwargs):
        t = resource_rep(ResourceTestT)
        return [GateCount(t, count=4)]


class ResourceTestRZ(ResourceOperator):
    """Dummy class for testing"""

    num_wires = 1
    resource_keys = {"epsilon"}

    def __init__(self, epsilon=None, wires=None) -> None:
        self.epsilon = epsilon
        super().__init__(wires=wires)

    @classmethod
    def resource_rep(cls, epsilon=None):
        return CompressedResourceOp(cls, {"epsilon": epsilon})

    @property
    def resource_params(self):
        return {"epsilon": self.epsilon}

    @classmethod
    def default_resource_decomp(cls, epsilon, **kwargs):
        if epsilon is None:
            epsilon = kwargs["config"]["error_rz"]

        t = resource_rep(ResourceTestT)
        t_counts = round(1 / epsilon)
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
        return CompressedResourceOp(cls, {"num_iter": num_iter})

    @property
    def resource_params(self):
        return {"num_iter": self.num_iter}

    @classmethod
    def default_resource_decomp(cls, num_iter, **kwargs):
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
        return CompressedResourceOp(cls, {"num_wires": num_wires})

    @property
    def resource_params(self):
        return {"num_wires": self.num_wires}

    @classmethod
    def default_resource_decomp(cls, num_wires, **kwargs):
        rz = resource_rep(ResourceTestRZ, {"epsilon": 1e-2})
        alg1 = resource_rep(ResourceTestAlg1, {"num_iter": 3})

        return [
            AllocWires(num_wires=num_wires),
            GateCount(rz, num_wires),
            GateCount(alg1, num_wires // 2),
            FreeWires(num_wires=num_wires),
        ]


class TestEstimateResources:
    """Test that core resource estimation functionality"""

    def test_estimate_resources_from_qfunc(self):
        """Test that we can accurately obtain resources from qfunc"""

        def my_circuit():
            for w in range(5):
                ResourceTestHadamard(wires=[w])
            ResourceTestCNOT(wires=[0, 1])
            ResourceTestRZ(wires=[1])
            ResourceTestRZ(epsilon=1e-2, wires=[2])
            ResourceTestCNOT(wires=[3, 4])
            ResourceTestAlg1(num_iter=5, wires=[5, 6])
<<<<<<< HEAD
<<<<<<< HEAD
=======
            return
>>>>>>> 5a1c24570 (added tests and updated docs)
=======
>>>>>>> f3bb59979 (code review comments + changelog)

        expected_gates = defaultdict(
            int,
            {
                resource_rep(ResourceTestT): round(1 / 1e-2) + round(1 / 1e-5),
                resource_rep(ResourceTestCNOT): 7,
                resource_rep(ResourceTestHadamard): 10,
            },
        )
        expected_qubits = QubitManager(work_wires={"clean": 4, "dirty": 1}, algo_wires=7)
        expected_resources = Resources(qubit_manager=expected_qubits, gate_types=expected_gates)

        gate_set = {"TestCNOT", "TestT", "TestHadamard"}
        computed_resources = estimate_resources(my_circuit, gate_set=gate_set)()
        assert computed_resources == expected_resources

    def test_estimate_resources_from_resource_operator(self):
        """Test that we can accurately obtain resources from qfunc"""
        op = ResourceTestAlg2(num_wires=4)
        actual_resources = estimate_resources(op, gate_set={"TestRZ", "TestAlg1"})

        expected_gates = defaultdict(
            int,
            {
                resource_rep(ResourceTestRZ, {"epsilon": 1e-2}): 4,
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
                resource_rep(ResourceTestRZ, {"epsilon": 1e-2}): 4,
                resource_rep(ResourceTestAlg1, {"num_iter": 3}): 2,
            },
        )
        qubits = QubitManager(work_wires=0, algo_wires=4)
        resources = Resources(qubit_manager=qubits, gate_types=gates)
<<<<<<< HEAD
=======

        gate_set = {"TestCNOT", "TestT", "TestHadamard"}
        actual_resources = estimate_resources(resources, gate_set=gate_set)

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
>>>>>>> 5a1c24570 (added tests and updated docs)

        gate_set = {"TestCNOT", "TestT", "TestHadamard"}
        actual_resources = estimate_resources(resources, gate_set=gate_set)

<<<<<<< HEAD
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

=======
>>>>>>> 5a1c24570 (added tests and updated docs)
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
                            resource_rep(ResourceTestRZ, {"epsilon": 1e-2}): 4,
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
<<<<<<< HEAD
<<<<<<< HEAD
=======
            return
>>>>>>> 5a1c24570 (added tests and updated docs)
=======
>>>>>>> f3bb59979 (code review comments + changelog)

        actual_resources = estimate_resources(my_circ, gate_set=gate_set)(num_wires=4)
        assert actual_resources == expected_resources

    @pytest.mark.parametrize("error_val", (0.1, 0.01, 0.001))
    def test_varying_config(self, error_val):
        """Test that changing the resource_config correctly updates the resources"""
        custom_config = copy.copy(resource_config)
        custom_config["error_rz"] = error_val

        op = ResourceTestRZ()  # don't specify epsilon
        computed_resources = estimate_resources(op, gate_set={"TestT"}, config=custom_config)

        expected_resources = Resources(
            qubit_manager=QubitManager(work_wires=0, algo_wires=1),
            gate_types=defaultdict(int, {resource_rep(ResourceTestT): round(1 / error_val)}),
        )

        assert computed_resources == expected_resources

    @pytest.mark.parametrize("error_val", (0.1, 0.01, 0.001))
    def test_varying_single_qubit_rotation_error(self, error_val):
        """Test that setting the single_qubit_rotation_error correctly updates the resources"""
        op = ResourceTestRZ()  # don't specify epsilon
        computed_resources = estimate_resources(
            op, gate_set={"TestT"}, single_qubit_rotation_error=error_val
        )

        expected_resources = Resources(
            qubit_manager=QubitManager(work_wires=0, algo_wires=1),
            gate_types=defaultdict(int, {resource_rep(ResourceTestT): round(1 / error_val)}),
        )

        assert computed_resources == expected_resources
