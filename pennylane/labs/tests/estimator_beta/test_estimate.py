# Copyright 2026 Xanadu Quantum Technologies Inc.

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

import pennylane as qp
import pennylane.estimator as qre
from pennylane.estimator.ops.op_math.symbolic import Controlled, Pow
from pennylane.estimator.resource_config import ResourceConfig
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.estimator.resources_base import Resources
from pennylane.exceptions import ResourcesUndefinedError
from pennylane.labs.estimator_beta import Allocate, Deallocate, estimate

# pylint: disable= no-self-use, arguments-differ, too-many-public-methods


def _circuit_w_expval(circ):
    circ()
    return qp.expval(qp.Z(0))


def _circuit_w_sample(circ):
    circ()
    return qp.sample(wires=[0])


def _circuit_w_probs(circ):
    circ()
    return qp.probs()


def _circuit_w_state(circ):
    circ()
    return qp.state()


class DummyCNOT(ResourceOperator):
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

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params=None):
        return [GateCount(cls.resource_rep())]


class DummyHadamard(ResourceOperator):
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

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params=None):
        return [GateCount(cls.resource_rep())]


class DummyT(ResourceOperator):
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


class DummyZ(ResourceOperator):
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
        t = resource_rep(DummyT)
        return [GateCount(t, count=4)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params=None):
        return [GateCount(cls.resource_rep())]


class DummyRZ(ResourceOperator):
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
        t = resource_rep(DummyT)
        t_counts = round(1 / precision)
        return [GateCount(t, count=t_counts)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params=None):
        return [GateCount(cls.resource_rep())]


class DummyAlg1(ResourceOperator):
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
        cnot = resource_rep(DummyCNOT)
        h = resource_rep(DummyHadamard)

        return [
            Allocate(num_wires=num_iter),
            GateCount(h, num_iter),
            GateCount(cnot, num_iter),
            Deallocate(num_wires=num_iter - 1),
        ]


class DummyAlg2(ResourceOperator):
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
        rz = resource_rep(DummyRZ, {"precision": 1e-2})
        alg1 = resource_rep(DummyAlg1, {"num_iter": 3})

        return [
            Allocate(num_wires=num_wires),
            GateCount(rz, num_wires),
            GateCount(alg1, num_wires // 2),
            Deallocate(num_wires=num_wires),
        ]


class DummyAlg3(ResourceOperator):
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
        borrowed_qubits = Allocate(3, state="any", restored=True)
        free_borrowed_qubits = Deallocate(allocated_register=borrowed_qubits)

        return [
            Allocate(num_wires=2),
            borrowed_qubits,
            GateCount(qre.X.resource_rep(), num_wires + 5),
            free_borrowed_qubits,
            Deallocate(num_wires=2),
        ]


def mock_rotation_decomp(precision):
    """A mock decomposition for rotation gates returning DummyT gates for testing."""
    t = resource_rep(DummyT)
    t_counts = round(1 / precision)
    return [GateCount(t, count=t_counts)]


class TestEstimateResources:
    """Test that core resource estimation functionality"""

    @staticmethod
    def circ():
        """Dummy Circuit"""
        qp.Hadamard(wires=[0])
        qp.X(wires=[1])
        qp.RX(1.23, wires=[0])
        qp.CNOT(wires=[0, 1])

    def test_estimate_with_unsupported_dispatch_labs(self):
        """Test that a TypeError is raised when an unsupported type is passed to the estimate function."""
        with pytest.raises(TypeError, match="Could not obtain resources for workflow of type"):
            estimate({1, 2, 3})

    def test_estimate_with_tight_budget_qfunc_labs(self):
        """Test that an error is raised if a user provides `tight_budget=True` and wire allocation
        goes beyond the budget."""

        def circ_wires():
            DummyAlg1(num_iter=3)
            DummyCNOT()
            DummyAlg1(num_iter=3)

        with pytest.raises(
            ValueError, match="Allocated more wires than the prescribed wire budget."
        ):
            estimate(
                circ_wires,
                gate_set={"DummyCNOT", "DummyHadamard"},
                zeroed_wires=2,
                any_state_wires=2,
                tight_wires_budget=True,
            )()

    def test_estimate_with_tight_budget_resources_labs(self):
        """Test that an error is raised if a user provides `tight_budget=True` and wire allocation
        goes beyond the budget."""
        gates = {DummyAlg1.resource_rep(num_iter=3): 2, DummyCNOT.resource_rep(): 1}
        resource_wires = Resources(
            zeroed_wires=0,
            any_state_wires=0,
            algo_wires=2,
            gate_types=gates,
        )

        with pytest.raises(
            ValueError, match="Allocated more wires than the prescribed wire budget."
        ):
            estimate(
                resource_wires,
                gate_set={"DummyCNOT", "DummyHadamard"},
                zeroed_wires=2,
                any_state_wires=2,
                tight_wires_budget=True,
            )

    def test_estimate_qnode_labs(self):
        """Test that a QNode can be estimated."""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev)
        def circuit(precision):
            qp.Hadamard(wires=0)
            qre.X()
            qre.RX(precision=precision, wires=0)
            return qp.expval(qp.Z(0))

        resources = estimate(circuit, gate_set={"Hadamard", "X", "RX"})(1e-3)

        assert resources.algo_wires == 2
        assert resources.gate_counts["Hadamard"] == 1
        assert resources.gate_counts["X"] == 1
        assert resources.gate_counts["RX"] == 1
        assert resources.gate_types[qre.RX.resource_rep(precision=1e-3)] == 1

    def test_qfunc_with_num_wires_labs(self):
        """Test that the number of wires is correctly inferred from a qfunc
        when the num_wires argument is used."""

        def my_circuit():
            DummyAlg2(num_wires=4)

        res = estimate(my_circuit, gate_set={"DummyAlg2"})()
        expected = Resources(
            zeroed_wires=0,
            any_state_wires=0,
            algo_wires=4,
            gate_types={resource_rep(DummyAlg2, {"num_wires": 4}): 1},
        )
        assert res == expected

    def test_gate_set_is_none_labs(self):
        """Test that the default gate set is used when gate_set is None."""
        op = qre.X()
        res = estimate(op, gate_set=None)
        expected = Resources(
            zeroed_wires=0,
            any_state_wires=0,
            algo_wires=1,
            gate_types={resource_rep(qre.X): 1},
        )
        assert res == expected

    def test_estimate_resources_from_qfunc_labs(self):
        """Test that we can accurately obtain resources from qfunc"""

        def my_circuit():
            for w in range(5):
                DummyHadamard(wires=[w])
            DummyCNOT(wires=[0, 1])
            DummyRZ(wires=[1])
            DummyRZ(precision=1e-2, wires=[2])
            DummyCNOT(wires=[3, 4])
            DummyAlg1(num_iter=5, wires=[5, 6])

        # See implementation of DummyRZ to see how it decomposes down to
        # DummyT based on the precision value, hence why
        # round(1/1e-2) + round(1/1e-9)
        expected_gates = defaultdict(
            int,
            {
                resource_rep(DummyT): round(1 / 1e-2) + round(1 / 1e-9),
                resource_rep(DummyCNOT): 7,
                resource_rep(DummyHadamard): 10,
            },
        )
        expected_resources = Resources(
            zeroed_wires=4, any_state_wires=1, algo_wires=7, gate_types=expected_gates
        )

        gate_set = {"DummyCNOT", "DummyT", "DummyHadamard"}
        custom_config = ResourceConfig()
        custom_config.resource_op_precisions[DummyRZ] = {"precision": 1e-9}
        computed_resources = estimate(my_circuit, gate_set=gate_set, config=custom_config)()
        assert computed_resources == expected_resources

    def test_estimate_resources_from_resource_operator_labs(self):
        """Test that we can accurately obtain resources from resource operator"""
        op = DummyAlg2(num_wires=4)
        actual_resources = estimate(op, gate_set={"DummyRZ", "DummyAlg1"})

        expected_gates = defaultdict(
            int,
            {
                resource_rep(DummyRZ, {"precision": 1e-2}): 4,
                resource_rep(DummyAlg1, {"num_iter": 3}): 2,
            },
        )
        expected_resources = Resources(zeroed_wires=4, algo_wires=4, gate_types=expected_gates)
        assert actual_resources == expected_resources

    def test_estimate_resources_from_scaled_resource_operator_labs(self):
        """Test that we can accurately obtain resources from resource operator"""
        op = 2 * DummyAlg2(num_wires=4)
        actual_resources = estimate(op, gate_set={"DummyRZ", "DummyAlg1"})

        expected_gates = defaultdict(
            int,
            {
                resource_rep(DummyRZ, {"precision": 1e-2}): 8,
                resource_rep(DummyAlg1, {"num_iter": 3}): 4,
            },
        )
        expected_resources = Resources(zeroed_wires=4, algo_wires=4, gate_types=expected_gates)
        assert actual_resources == expected_resources

    def test_estimate_resources_from_resources_obj_labs(self):
        """Test that we can accurately obtain resources from resources workflow"""
        gates = defaultdict(
            int,
            {
                resource_rep(DummyRZ, {"precision": 1e-2}): 4,
                resource_rep(DummyAlg1, {"num_iter": 3}): 2,
            },
        )
        resources = Resources(zeroed_wires=0, algo_wires=4, gate_types=gates)

        gate_set = {"DummyCNOT", "DummyT", "DummyHadamard"}
        actual_resources = estimate(resources, gate_set=gate_set)

        expected_gates = defaultdict(
            int,
            {
                resource_rep(DummyT): 4 * round(1 / 1e-2),
                resource_rep(DummyCNOT): 6,
                resource_rep(DummyHadamard): 6,
            },
        )
        # TODO: optimize allocation
        expected_resources = Resources(
            zeroed_wires=2, any_state_wires=2, algo_wires=4, gate_types=expected_gates
        )

        assert actual_resources == expected_resources

    def test_unsupported_object_in_queue_raises_error_labs(self):
        """Test that a ValueError is raised for unsupported objects in the queue."""

        def my_circuit():
            qp.QueuingManager.append(0)  # Arbitrarily queue something

        with pytest.raises(
            ValueError,
            match="Circuit must contain only instances of 'ResourceOperator', 'Operator', 'MeasurementProcess' and 'MarkQubits',",
        ):
            estimate(my_circuit)()

    def test_estimate_resources_from_qfunc_with_pl_op_labs(self):
        """Test that PennyLane operators are correctly mapped to resource operators
        when processing a qfunc."""

        def my_circuit():
            qp.Hadamard(0)
            qp.PauliX(1)

        actual_resources = estimate(my_circuit, gate_set={"Hadamard", "X"})()

        expected_gates = defaultdict(
            int,
            {
                resource_rep(qre.Hadamard): 1,
                resource_rep(qre.X): 1,
            },
        )
        expected_resources = Resources(
            zeroed_wires=0, any_state_wires=0, algo_wires=2, gate_types=expected_gates
        )
        assert actual_resources == expected_resources

    def test_estimate_resources_from_pl_op_dispatch_labs(self):
        """Test that the dispatch for PennyLane operators correctly maps to a
        ResourceOperator without further decomposition."""
        op = qp.PauliX(0)

        actual_resources = estimate(op)

        expected_gates = defaultdict(int, {resource_rep(qre.X): 1})
        expected_resources = Resources(
            zeroed_wires=0, any_state_wires=0, algo_wires=1, gate_types=expected_gates
        )

        assert actual_resources == expected_resources

    @pytest.mark.parametrize(
        "gate_set, expected_resources",
        (
            (
                {"DummyRZ", "DummyAlg1", "DummyZ"},
                Resources(
                    zeroed_wires=4,
                    algo_wires=4,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(DummyRZ, {"precision": 1e-2}): 4,
                            resource_rep(DummyAlg1, {"num_iter": 3}): 2,
                            resource_rep(DummyZ): 4,
                        },
                    ),
                ),
            ),
            (
                {"DummyCNOT", "DummyT", "DummyHadamard"},
                Resources(
                    zeroed_wires=6,
                    any_state_wires=2,
                    algo_wires=4,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(DummyT): 416,
                            resource_rep(DummyCNOT): 6,
                            resource_rep(DummyHadamard): 6,
                        },
                    ),
                ),
            ),
        ),
    )
    def test_varying_gate_sets_labs(self, gate_set, expected_resources):
        """Test that changing the gate_set correctly updates the resources"""

        def my_circ(num_wires):
            DummyAlg2(num_wires, wires=range(num_wires))
            for w in range(num_wires):
                DummyZ(wires=w)

        actual_resources = estimate(my_circ, gate_set=gate_set)(num_wires=4)
        assert actual_resources == expected_resources

    @pytest.mark.parametrize("error_val", (0.1, 0.01, 0.001))
    def test_varying_config_labs(self, error_val):
        """Test that changing the resource_config correctly updates the resources"""
        custom_config = ResourceConfig()
        custom_config.resource_op_precisions[DummyRZ] = {"precision": error_val}

        op = DummyRZ()  # don't specify precision
        computed_resources = estimate(op, gate_set={"DummyT"}, config=custom_config)

        expected_resources = Resources(
            zeroed_wires=0,
            algo_wires=1,
            gate_types=defaultdict(int, {resource_rep(DummyT): round(1 / error_val)}),
        )

        assert computed_resources == expected_resources

    measurement_circuits = (
        _circuit_w_expval,
        _circuit_w_sample,
        _circuit_w_probs,
        _circuit_w_state,
    )

    @pytest.mark.parametrize("circ_w_measurement", measurement_circuits)
    def test_estimate_ignores_measurementprocess_labs(self, circ_w_measurement):
        """Test that the estimate function ignores measurement processes"""

        assert estimate(self.circ)() == estimate(circ_w_measurement)(self.circ)

    def test_custom_adjoint_decomposition_labs(self):
        """Test that a custom adjoint decomposition can be set and used."""

        def custom_adj_RZ(target_resource_params):  # pylint: disable=unused-argument
            return [GateCount(resource_rep(qre.Z))]

        rc = ResourceConfig()
        rc.set_decomp(qre.RZ, custom_adj_RZ, decomp_type="adj")

        res = estimate(qre.Adjoint(qre.RZ(0.1, wires=0)), config=rc)
        pl_res = estimate(qp.adjoint(qp.RZ(0.1, wires=0)), config=rc)

        expected_gates = defaultdict(int, {resource_rep(qre.Z): 1})
        expected_resources = Resources(
            zeroed_wires=0, any_state_wires=0, algo_wires=1, gate_types=expected_gates
        )

        assert res == expected_resources
        assert pl_res == expected_resources

    def test_custom_pow_decomposition_labs(self):
        """Test that a custom pow decomposition can be set and used."""

        def custom_pow_RZ(pow_z, target_resource_params):  # pylint: disable=unused-argument
            return [GateCount(resource_rep(qre.Hadamard), count=2)]

        rc = ResourceConfig()
        rc.set_decomp(qre.RZ, custom_pow_RZ, decomp_type="pow")

        res = estimate(Pow(qre.RZ(0.1, wires=0), pow_z=3), config=rc)
        pl_res = estimate(qp.pow(qp.RZ(0.1, wires=0)), config=rc)

        expected_gates = defaultdict(int, {resource_rep(qre.Hadamard): 2})
        expected_resources = Resources(
            zeroed_wires=0, any_state_wires=0, algo_wires=1, gate_types=expected_gates
        )

        assert res == expected_resources
        assert pl_res == expected_resources

    def test_custom_controlled_decomposition_labs(self):
        """Test that a custom controlled decomposition can be set and used."""

        def custom_ctrl_RZ(
            num_ctrl_wires, num_zero_ctrl, target_resource_params
        ):  # pylint: disable=unused-argument
            return [GateCount(resource_rep(qre.X), count=3)]

        rc = ResourceConfig()
        rc.set_decomp(qre.RZ, custom_ctrl_RZ, decomp_type="ctrl")

        res = estimate(
            Controlled(qre.RZ(0.1, wires=0), num_ctrl_wires=1, num_zero_ctrl=0), config=rc
        )
        pl_res = estimate(qp.ctrl(qp.RZ(0.1, wires=0), control=1, control_values=0), config=rc)

        expected_gates = defaultdict(int, {resource_rep(qre.X): 3})
        expected_resources = Resources(
            zeroed_wires=0, any_state_wires=0, algo_wires=2, gate_types=expected_gates
        )

        assert res == expected_resources
        assert pl_res == expected_resources

    @pytest.mark.parametrize(
        "op, gate_set, expected_resources",
        (
            (
                qre.Adjoint(DummyAlg3(num_wires=3)),
                {"X"},
                Resources(
                    zeroed_wires=5,
                    algo_wires=3,
                    gate_types={qre.X.resource_rep(): 8},
                ),
            ),
            (
                qre.Adjoint(
                    qre.Prod(
                        (DummyAlg1(3), qre.Adjoint(DummyAlg1(3))),
                    ),
                ),
                {"DummyCNOT", "DummyHadamard"},
                Resources(
                    zeroed_wires=3,
                    algo_wires=2,
                    gate_types={DummyCNOT.resource_rep(): 6, DummyHadamard.resource_rep(): 6},
                ),
            ),
            (
                qre.Controlled(DummyAlg3(num_wires=3), 3, 2),
                {"MultiControlledX", "X"},
                Resources(
                    zeroed_wires=5,
                    algo_wires=6,
                    gate_types={
                        qre.MultiControlledX.resource_rep(3, 0): 8,
                        qre.X.resource_rep(): 4,
                    },
                ),
            ),
        ),
    )
    def test_estimator_symbolic_ops_labs(self, op, gate_set, expected_resources):
        """Test that using symbolic ops works with Allocate and Deallocate"""
        actual_resources = estimate(op, gate_set)
        assert actual_resources == expected_resources
