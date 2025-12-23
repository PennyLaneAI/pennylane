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
from pennylane.estimator.ops.op_math.symbolic import Adjoint
from pennylane.estimator.ops.qubit.non_parametric_ops import Hadamard, X, Z
from pennylane.estimator.ops.qubit.parametric_ops_single_qubit import RX, RZ
from pennylane.estimator.resource_config import ResourceConfig
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.estimator.resources_base import Resources
from pennylane.estimator.wires_manager import Allocate, Deallocate
from pennylane.exceptions import ResourcesUndefinedError

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


def mock_rotation_decomp(precision):
    """A mock decomposition for rotation gates returning DummyT gates for testing."""
    t = resource_rep(DummyT)
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
            DummyAlg2(num_wires=4)

        res = estimate(my_circuit, gate_set={"DummyAlg2"})()
        expected = Resources(
            zeroed_wires=0,
            any_state_wires=0,
            algo_wires=4,
            gate_types={resource_rep(DummyAlg2, {"num_wires": 4}): 1},
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

    def test_estimate_resources_from_resource_operator(self):
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

    def test_estimate_resources_from_scaled_resource_operator(self):
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

    def test_estimate_resources_from_resources_obj(self):
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
                    zeroed_wires=8,
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
    def test_varying_gate_sets(self, gate_set, expected_resources):
        """Test that changing the gate_set correctly updates the resources"""

        def my_circ(num_wires):
            DummyAlg2(num_wires, wires=range(num_wires))
            for w in range(num_wires):
                DummyZ(wires=w)

        actual_resources = estimate(my_circ, gate_set=gate_set)(num_wires=4)
        assert actual_resources == expected_resources

    @pytest.mark.parametrize("error_val", (0.1, 0.01, 0.001))
    def test_varying_config(self, error_val):
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
    def test_estimate_ignores_measurementprocess(self, circ_w_measurement):
        """Test that the estimate function ignores measurement processes"""

        def circ():
            qml.Hadamard(wires=[0])
            qml.X(wires=[1])
            qml.RX(1.23, wires=[0])
            qml.CNOT(wires=[0, 1])

        assert estimate(circ)() == estimate(circ_w_measurement)(circ)

    def test_custom_adjoint_decomposition(self):
        """Test that a custom adjoint decomposition can be set and used."""

        def custom_adj_RZ(target_resource_params):  # pylint: disable=unused-argument
            return [GateCount(resource_rep(Z))]

        rc = ResourceConfig()
        rc.set_decomp(RZ, custom_adj_RZ, decomp_type="adj")

        res = estimate(Adjoint(RZ(0.1, wires=0)), config=rc)
        pl_res = estimate(qml.adjoint(qml.RZ(0.1, wires=0)), config=rc)

        expected_gates = defaultdict(int, {resource_rep(Z): 1})
        expected_resources = Resources(
            zeroed_wires=0, any_state_wires=0, algo_wires=1, gate_types=expected_gates
        )

        assert res == expected_resources
        assert pl_res == expected_resources

    def test_custom_pow_decomposition(self):
        """Test that a custom pow decomposition can be set and used."""
        from pennylane.estimator.ops.op_math.symbolic import Pow

        def custom_pow_RZ(pow_z, target_resource_params):  # pylint: disable=unused-argument
            return [GateCount(resource_rep(Hadamard), count=2)]

        rc = ResourceConfig()
        rc.set_decomp(RZ, custom_pow_RZ, decomp_type="pow")

        res = estimate(Pow(RZ(0.1, wires=0), pow_z=3), config=rc)
        pl_res = estimate(qml.pow(qml.RZ(0.1, wires=0)), config=rc)

        expected_gates = defaultdict(int, {resource_rep(Hadamard): 2})
        expected_resources = Resources(
            zeroed_wires=0, any_state_wires=0, algo_wires=1, gate_types=expected_gates
        )

        assert res == expected_resources
        assert pl_res == expected_resources

    def test_custom_controlled_decomposition(self):
        """Test that a custom controlled decomposition can be set and used."""
        from pennylane.estimator.ops.op_math.symbolic import Controlled

        def custom_ctrl_RZ(
            num_ctrl_wires, num_zero_ctrl, target_resource_params
        ):  # pylint: disable=unused-argument
            return [GateCount(resource_rep(X), count=3)]

        rc = ResourceConfig()
        rc.set_decomp(RZ, custom_ctrl_RZ, decomp_type="ctrl")

        res = estimate(Controlled(RZ(0.1, wires=0), num_ctrl_wires=1, num_zero_ctrl=0), config=rc)
        pl_res = estimate(qml.ctrl(qml.RZ(0.1, wires=0), control=1, control_values=0), config=rc)

        expected_gates = defaultdict(int, {resource_rep(X): 3})
        expected_resources = Resources(
            zeroed_wires=0, any_state_wires=0, algo_wires=2, gate_types=expected_gates
        )

        assert res == expected_resources
        assert pl_res == expected_resources
