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

import math
from collections import defaultdict

import pytest

import pennylane as qp
from pennylane.estimator.compact_hamiltonian import VibrationalHamiltonian
from pennylane.estimator.estimate import (
    _default_adjoint_decomp,
    _default_controlled_decomp,
    _get_resource_decomposition,
    _get_symbolic_resource_decomposition,
    _update_params_from_config,
    apply_default_symbolic_decomp,
    estimate,
)
from pennylane.estimator.ops.op_math.controlled_ops import CNOT
from pennylane.estimator.ops.op_math.symbolic import Adjoint, Controlled, Pow
from pennylane.estimator.ops.qubit.matrix_ops import QubitUnitary
from pennylane.estimator.ops.qubit.non_parametric_ops import Hadamard, T, X, Z
from pennylane.estimator.ops.qubit.parametric_ops_multi_qubit import MultiRZ
from pennylane.estimator.ops.qubit.parametric_ops_single_qubit import RX, RY, RZ
from pennylane.estimator.resource_config import ResourceConfig
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.estimator.resources_base import Resources
from pennylane.estimator.templates.subroutines import QFT
from pennylane.estimator.templates.trotter import TrotterVibrational
from pennylane.estimator.wires_manager import Allocate, Deallocate
from pennylane.exceptions import ResourcesUndefinedError

# pylint: disable= no-self-use, arguments-differ


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
        raise ResourcesUndefinedError(f"{cls} does not have a resource decomposition defined.")


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
        raise ResourcesUndefinedError(f"{cls} does not have a resource decomposition defined.")


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
        raise ResourcesUndefinedError(f"{cls} does not have a resource decomposition defined.")


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


symbolic_decomp_data = [
    (  # base symbolic decomp + default params
        Adjoint(RZ()),
        [GateCount(RZ.resource_rep(precision=1e-9))],
    ),
    (  # default symbolic decomp + default params
        Controlled(QubitUnitary(2), 2, 1),
        [
            GateCount(X.resource_rep(), 2),
            GateCount(Controlled.resource_rep(resource_rep(RZ, {"precision": 1e-9}), 2, 0)),
            GateCount(Controlled.resource_rep(resource_rep(RY, {"precision": 1e-9}), 2, 0), 2),
            GateCount(
                Controlled.resource_rep(
                    QubitUnitary.resource_rep(num_wires=1, precision=1e-9), 2, 0
                ),
                4,
            ),
            GateCount(Controlled.resource_rep(resource_rep(CNOT), 2, 0), 3),
        ],
    ),
    (  # default symbolic decomp + custom resource decomp
        Adjoint(QFT(4)),
        [
            GateCount(Adjoint.resource_rep(CNOT.resource_rep()), 2),
            GateCount(Adjoint.resource_rep(Hadamard.resource_rep()), 4),
        ],
    ),
    (  # custom symbolic decomp + default params
        Pow(RX(), 3),
        [
            GateCount(Hadamard.resource_rep(), 2),
            GateCount(RZ.resource_rep(1e-9), 3),
        ],
    ),
    (  # (nested) default symbolic decomp + default symbolic decomp + default params
        Adjoint(Controlled(QubitUnitary(1), 2, 2)),
        [
            GateCount(
                Adjoint.resource_rep(Controlled.resource_rep(RZ.resource_rep(1e-9), 2, 0)), 2
            ),
            GateCount(Adjoint.resource_rep(Controlled.resource_rep(RY.resource_rep(1e-9), 2, 0))),
            GateCount(Adjoint.resource_rep(X.resource_rep()), 4),
        ],
    ),
    (  # (nested) default symbolic decomp + base symbolic decomp + default params
        Adjoint(Controlled(MultiRZ(4), 3, 2)),
        [
            GateCount(
                Adjoint.resource_rep(Controlled.resource_rep(RZ.resource_rep(precision=None), 3, 2))
            ),
            GateCount(Adjoint.resource_rep(CNOT.resource_rep()), 6),
        ],
    ),
    (  # (nested) default symbolic decomp + custom symbolic decomp + default params
        Controlled(Pow(RX(), 5), 3, 0),
        [
            GateCount(Controlled.resource_rep(Hadamard.resource_rep(), 3, 0), 2),
            GateCount(Controlled.resource_rep(RZ.resource_rep(1e-9), 3, 0), 5),
        ],
    ),
    (  # base symbolic decomp + custom params
        Adjoint(RZ(1e-3)),
        [GateCount(RZ.resource_rep(precision=1e-3))],
    ),
    (  # default symbolic decomp + custom params
        Controlled(QubitUnitary(2, 1e-3), 2, 1),
        [
            GateCount(X.resource_rep(), 2),
            GateCount(Controlled.resource_rep(resource_rep(RZ, {"precision": 1e-3}), 2, 0)),
            GateCount(Controlled.resource_rep(resource_rep(RY, {"precision": 1e-3}), 2, 0), 2),
            GateCount(
                Controlled.resource_rep(
                    QubitUnitary.resource_rep(num_wires=1, precision=1e-3), 2, 0
                ),
                4,
            ),
            GateCount(Controlled.resource_rep(resource_rep(CNOT), 2, 0), 3),
        ],
    ),
    (  # custom symbolic decomp + custom params
        Pow(RX(1e-3), 3),
        [
            GateCount(Hadamard.resource_rep(), 2),
            GateCount(RZ.resource_rep(1e-3), 3),
        ],
    ),
    (  # (nested) default symbolic decomp + default symbolic decomp + custom params
        Adjoint(Controlled(QubitUnitary(1, 1e-3), 2, 2)),
        [
            GateCount(
                Adjoint.resource_rep(Controlled.resource_rep(RZ.resource_rep(1e-3), 2, 0)), 2
            ),
            GateCount(Adjoint.resource_rep(Controlled.resource_rep(RY.resource_rep(1e-3), 2, 0))),
            GateCount(Adjoint.resource_rep(X.resource_rep()), 4),
        ],
    ),
    (  # (nested) default symbolic decomp + base symbolic decomp + custom params
        Adjoint(Controlled(MultiRZ(4, 1e-3), 3, 2)),
        [
            GateCount(
                Adjoint.resource_rep(Controlled.resource_rep(RZ.resource_rep(precision=1e-3), 3, 2))
            ),
            GateCount(Adjoint.resource_rep(CNOT.resource_rep()), 6),
        ],
    ),
    (  # (nested) default symbolic decomp + custom symbolic decomp + custom params
        Controlled(Pow(RX(1e-3), 5), 3, 0),
        [
            GateCount(Controlled.resource_rep(Hadamard.resource_rep(), 3, 0), 2),
            GateCount(Controlled.resource_rep(RZ.resource_rep(1e-3), 3, 0), 5),
        ],
    ),
    (  # (multi-nested) default symbolic decomp + default symbolic decomp + custom resource decomp
        Controlled(Adjoint(QFT(4)), 3, 2),
        [
            GateCount(X.resource_rep(), 4),
            GateCount(Controlled.resource_rep(Adjoint.resource_rep(CNOT.resource_rep()), 3, 0), 2),
            GateCount(
                Controlled.resource_rep(Adjoint.resource_rep(Hadamard.resource_rep()), 3, 0), 4
            ),
        ],
    ),
]

general_decomp_data = [
    (RZ(), [GateCount(T.resource_rep(), 44)]),  # base decomp + default params
    (RY(), [GateCount(T.resource_rep(), 30)]),  # custom decomp + default params
    (RZ(1e-3), [GateCount(T.resource_rep(), 21)]),  # base decomp + default params
    (RY(1e-3), [GateCount(T.resource_rep(), 10)]),  # custom decomp + default params
] + symbolic_decomp_data


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
            estimate({1, 2, 3})

    def test_estimate_qnode(self):
        """Test that a QNode can be estimated."""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev)
        def circuit(precision):
            qp.Hadamard(wires=0)
            X()
            RX(precision=precision, wires=0)
            return qp.expval(qp.Z(0))

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

    def test_estimate_resources_from_resource_operator_no_decomp(self):
        """Test that a ResourcesUndefinedError is raised when obtaining resources for
        a resource operator which has no resource_decomp defined"""
        with pytest.raises(
            ResourcesUndefinedError, match=".* does not have a resource decomposition defined"
        ):
            estimate(workflow=DummyT(), gate_set={DummyCNOT})

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
            qp.QueuingManager.append(0)  # Arbitrarily queue something

        with pytest.raises(
            ValueError,
            match="Queued object.*is not a ResourceOperator or Operator, and cannot be processed.",
        ):
            estimate(my_circuit)()

    def test_estimate_resources_from_qfunc_with_pl_op(self):
        """Test that PennyLane operators are correctly mapped to resource operators
        when processing a qfunc."""

        def my_circuit():
            qp.Hadamard(0)
            qp.PauliX(1)

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
        op = qp.PauliX(0)

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
            qp.Hadamard(wires=[0])
            qp.X(wires=[1])
            qp.RX(1.23, wires=[0])
            qp.CNOT(wires=[0, 1])

        assert estimate(circ)() == estimate(circ_w_measurement)(circ)

    def test_custom_adjoint_decomposition(self):
        """Test that a custom adjoint decomposition can be set and used."""

        def custom_adj_RZ(target_resource_params):  # pylint: disable=unused-argument
            return [GateCount(resource_rep(Z))]

        rc = ResourceConfig()
        rc.set_decomp(RZ, custom_adj_RZ, decomp_type="adj")

        res = estimate(Adjoint(RZ(0.1, wires=0)), config=rc)
        pl_res = estimate(qp.adjoint(qp.RZ(0.1, wires=0)), config=rc)

        expected_gates = defaultdict(int, {resource_rep(Z): 1})
        expected_resources = Resources(
            zeroed_wires=0, any_state_wires=0, algo_wires=1, gate_types=expected_gates
        )

        assert res == expected_resources
        assert pl_res == expected_resources

    def test_custom_pow_decomposition(self):
        """Test that a custom pow decomposition can be set and used."""

        def custom_pow_RZ(pow_z, target_resource_params):  # pylint: disable=unused-argument
            return [GateCount(resource_rep(Hadamard), count=2)]

        rc = ResourceConfig()
        rc.set_decomp(RZ, custom_pow_RZ, decomp_type="pow")

        res = estimate(Pow(RZ(0.1, wires=0), pow_z=3), config=rc)
        pl_res = estimate(qp.pow(qp.RZ(0.1, wires=0)), config=rc)

        expected_gates = defaultdict(int, {resource_rep(Hadamard): 2})
        expected_resources = Resources(
            zeroed_wires=0, any_state_wires=0, algo_wires=1, gate_types=expected_gates
        )

        assert res == expected_resources
        assert pl_res == expected_resources

    def test_custom_controlled_decomposition(self):
        """Test that a custom controlled decomposition can be set and used."""

        def custom_ctrl_RZ(
            num_ctrl_wires, num_zero_ctrl, target_resource_params
        ):  # pylint: disable=unused-argument
            return [GateCount(resource_rep(X), count=3)]

        rc = ResourceConfig()
        rc.set_decomp(RZ, custom_ctrl_RZ, decomp_type="ctrl")

        res = estimate(Controlled(RZ(0.1, wires=0), num_ctrl_wires=1, num_zero_ctrl=0), config=rc)
        pl_res = estimate(qp.ctrl(qp.RZ(0.1, wires=0), control=1, control_values=0), config=rc)

        expected_gates = defaultdict(int, {resource_rep(X): 3})
        expected_resources = Resources(
            zeroed_wires=0, any_state_wires=0, algo_wires=2, gate_types=expected_gates
        )

        assert res == expected_resources
        assert pl_res == expected_resources


@pytest.mark.parametrize(
    "res_op, expected_params",
    (
        (RZ(precision=None), {"precision": 1e-9}),  # Update None --> default value
        (RZ(precision=1e-3), {"precision": 1e-3}),  # Keep input values
        (RX(precision=None), {"precision": 1e-5}),  # Update None --> custom value
        (
            TrotterVibrational(
                vibration_ham=VibrationalHamiltonian(num_modes=2, grid_size=4, taylor_degree=2),
                num_steps=10,
                order=2,
                phase_grad_precision=1e-5,
            ),
            {
                "vibration_ham": VibrationalHamiltonian(num_modes=2, grid_size=4, taylor_degree=2),
                "num_steps": 10,
                "order": 2,
                "phase_grad_precision": 1e-5,  # Keep input values
                "coeff_precision": 1e-3,  # Only update None arguments --> default value
            },
        ),
        (
            TrotterVibrational(
                vibration_ham=VibrationalHamiltonian(num_modes=2, grid_size=4, taylor_degree=2),
                num_steps=10,
                order=2,
                coeff_precision=1e-5,
            ),
            {
                "vibration_ham": VibrationalHamiltonian(num_modes=2, grid_size=4, taylor_degree=2),
                "num_steps": 10,
                "order": 2,
                "phase_grad_precision": 1e-2,  # Only update None arguments --> default value
                "coeff_precision": 1e-5,  # Keep input values
            },
        ),
    ),
)
def test_update_params_from_config(res_op, expected_params):
    """Test that the params are correctly updated using the config"""
    config = ResourceConfig()
    config.set_precision(op_type=RX, precision=1e-5)
    config.set_precision(
        op_type=TrotterVibrational,
        precision=1e-2,
        resource_key="phase_grad_precision",
    )
    comp_res_op = res_op.resource_rep_from_op()
    assert _update_params_from_config(comp_res_op, config) == expected_params


def test_default_adjoint_decomp():
    """Test that the default adjoint decomposition is applied as expected"""
    base_resource_decomp = [
        GateCount(RX.resource_rep(), 5),
        Allocate(num_wires=3),
        GateCount(Hadamard.resource_rep(), 2),
        Deallocate(num_wires=1),
        GateCount(RZ.resource_rep(1e-3), 4),
    ]

    expected_resource_decomp = [
        GateCount(Adjoint.resource_rep(RZ.resource_rep(1e-3)), 4),
        Allocate(num_wires=1),
        GateCount(Adjoint.resource_rep(Hadamard.resource_rep()), 2),
        Deallocate(num_wires=3),
        GateCount(Adjoint.resource_rep(RX.resource_rep()), 5),
    ]

    assert _default_adjoint_decomp(base_resource_decomp) == expected_resource_decomp


@pytest.mark.parametrize(
    "num_ctrl, num_zero",
    ((1, 0), (2, 0), (2, 1), (5, 3)),
)
def test_default_controlled_decomp(num_ctrl, num_zero):
    """Test that the default controlled decomposition is applied as expected"""
    base_resource_decomp = [
        GateCount(RX.resource_rep(), 5),
        Allocate(num_wires=3),
        GateCount(Hadamard.resource_rep(), 2),
        Deallocate(num_wires=1),
        GateCount(RZ.resource_rep(1e-3), 4),
    ]

    expected_resource_decomp = [
        GateCount(Controlled.resource_rep(RX.resource_rep(), num_ctrl, 0), 5),
        Allocate(num_wires=3),
        GateCount(Controlled.resource_rep(Hadamard.resource_rep(), num_ctrl, 0), 2),
        Deallocate(num_wires=1),
        GateCount(Controlled.resource_rep(RZ.resource_rep(1e-3), num_ctrl, 0), 4),
    ]

    x_gates = [GateCount(X.resource_rep(), 2 * num_zero)] if num_zero > 0 else []
    expected_resource_decomp = x_gates + expected_resource_decomp

    assert (
        _default_controlled_decomp(num_ctrl, num_zero, base_resource_decomp)
        == expected_resource_decomp
    )


@pytest.mark.parametrize(
    "res_op, res_decomp, sym_type, params, expected_decomp",
    (
        (
            Hadamard.resource_rep(),
            [
                GateCount(RX.resource_rep(), 5),
                Allocate(num_wires=3),
                GateCount(Hadamard.resource_rep(), 2),
                Deallocate(num_wires=1),
                GateCount(RZ.resource_rep(1e-3), 4),
            ],
            Adjoint,
            {},
            [
                GateCount(Adjoint.resource_rep(RZ.resource_rep(1e-3)), 4),
                Allocate(num_wires=1),
                GateCount(Adjoint.resource_rep(Hadamard.resource_rep()), 2),
                Deallocate(num_wires=3),
                GateCount(Adjoint.resource_rep(RX.resource_rep()), 5),
            ],
        ),
        (
            RZ.resource_rep(1e-3),
            [GateCount(Hadamard.resource_rep(), 2)],
            Pow,
            {"pow_z": 2},
            [GateCount(RZ.resource_rep(1e-3), 2)],
        ),
        (
            RZ.resource_rep(1e-3),
            [GateCount(Hadamard.resource_rep(), 2)],
            Pow,
            {"pow_z": 5},
            [GateCount(RZ.resource_rep(1e-3), 5)],
        ),
        (
            Hadamard.resource_rep(),
            [
                GateCount(RX.resource_rep(), 5),
                Allocate(num_wires=3),
                GateCount(Hadamard.resource_rep(), 2),
                Deallocate(num_wires=1),
                GateCount(RZ.resource_rep(1e-3), 4),
            ],
            Controlled,
            {
                "num_ctrl_wires": 2,
                "num_zero_ctrl": 0,
            },
            [
                GateCount(Controlled.resource_rep(RX.resource_rep(), 2, 0), 5),
                Allocate(num_wires=3),
                GateCount(Controlled.resource_rep(Hadamard.resource_rep(), 2, 0), 2),
                Deallocate(num_wires=1),
                GateCount(Controlled.resource_rep(RZ.resource_rep(1e-3), 2, 0), 4),
            ],
        ),
        (
            Hadamard.resource_rep(),
            [
                GateCount(RX.resource_rep(), 5),
                Allocate(num_wires=3),
                GateCount(Hadamard.resource_rep(), 2),
                Deallocate(num_wires=1),
                GateCount(RZ.resource_rep(1e-3), 4),
            ],
            Controlled,
            {
                "num_ctrl_wires": 5,
                "num_zero_ctrl": 3,
            },
            [
                GateCount(X.resource_rep(), 6),
                GateCount(Controlled.resource_rep(RX.resource_rep(), 5, 0), 5),
                Allocate(num_wires=3),
                GateCount(Controlled.resource_rep(Hadamard.resource_rep(), 5, 0), 2),
                Deallocate(num_wires=1),
                GateCount(Controlled.resource_rep(RZ.resource_rep(1e-3), 5, 0), 4),
            ],
        ),
    ),
)
def test_apply_default_symbolic_decomp(res_op, res_decomp, sym_type, params, expected_decomp):
    """Test that the apply_default_symbolic_decomp function works as expected"""
    assert apply_default_symbolic_decomp(res_op, res_decomp, sym_type, **params) == expected_decomp


@pytest.mark.parametrize(
    "params, error, error_message",
    (
        (
            {
                "base_compr_resource_op": X.resource_rep(),
                "base_resource_decomp": [GateCount(X.resource_rep())],
                "symbolic_type": ResourceOperator,
                "target_symbolic_params": {},
            },
            ValueError,
            "Unexpected symbolic type",
        ),
        (
            {
                "base_compr_resource_op": X.resource_rep(),
                "base_resource_decomp": [GateCount(X.resource_rep())],
                "symbolic_type": Pow,
                "pow_z": 1.23,
            },
            NotImplementedError,
            "No default decomposition for fractional or negative powers",
        ),
        (
            {
                "base_compr_resource_op": X.resource_rep(),
                "base_resource_decomp": [GateCount(X.resource_rep())],
                "symbolic_type": Pow,
                "pow_z": -2,
            },
            NotImplementedError,
            "No default decomposition for fractional or negative powers",
        ),
    ),
)
def test_apply_default_symbolic_decomp_raises_error(params, error, error_message):
    """Test that an error is raised if `symbolic_type` is not one of `Adjoint`, `Pow`, `Controlled`"""
    with pytest.raises(error, match=error_message):
        apply_default_symbolic_decomp(**params)


@pytest.mark.parametrize("res_op, expected_decomp", symbolic_decomp_data)
def test_get_symbolic_resource_decomposition(res_op, expected_decomp):
    """Test that we get the correct decomposition with the correct parameters as expected"""

    def custom_RX_pow(pow_z, target_resource_params):
        precision = target_resource_params["precision"]
        return [
            GateCount(Hadamard.resource_rep(), 2),
            GateCount(RZ.resource_rep(precision), pow_z),
        ]

    def custom_QFT(num_wires):
        return [
            GateCount(Hadamard.resource_rep(), num_wires),
            GateCount(CNOT.resource_rep(), num_wires // 2),
        ]

    cfg = ResourceConfig()
    cfg.set_decomp(RX, custom_RX_pow, "pow")
    cfg.set_decomp(QFT, custom_QFT)

    computed_decomp = _get_symbolic_resource_decomposition(res_op.resource_rep_from_op(), cfg)
    assert computed_decomp == expected_decomp


@pytest.mark.parametrize("res_op, expected_decomp", general_decomp_data)
def test_get_resource_decomposition(res_op, expected_decomp):
    def custom_RX_pow(pow_z, target_resource_params):
        precision = target_resource_params["precision"]
        return [
            GateCount(Hadamard.resource_rep(), 2),
            GateCount(RZ.resource_rep(precision), pow_z),
        ]

    def custom_QFT(num_wires):
        return [
            GateCount(Hadamard.resource_rep(), num_wires),
            GateCount(CNOT.resource_rep(), num_wires // 2),
        ]

    def custom_RY(precision):
        return [GateCount(T.resource_rep(), round(math.log2(1 / precision)))]

    cfg = ResourceConfig()
    cfg.set_decomp(RY, custom_RY)
    cfg.set_decomp(QFT, custom_QFT)
    cfg.set_decomp(RX, custom_RX_pow, "pow")

    computed_decomp = _get_resource_decomposition(res_op.resource_rep_from_op(), cfg)
    assert computed_decomp == expected_decomp
