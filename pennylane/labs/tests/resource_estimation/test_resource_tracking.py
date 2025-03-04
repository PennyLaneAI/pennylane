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
"""Test the core resource tracking pipeline."""
from collections import defaultdict
from copy import copy

# pylint:disable=protected-access, no-self-use
import pytest

import pennylane as qml
import pennylane.labs.resource_estimation as re
from pennylane.labs.resource_estimation.resource_tracking import (
    DefaultGateSet,
    _clean_gate_counts,
    _counts_from_compressed_res_op,
    _operations_to_compressed_reps,
    _StandardGateSet,
    get_resources,
    resource_config,
    resources_from_operation,
    resources_from_qfunc,
    resources_from_tape,
)


class DummyOperation(qml.operation.Operation):
    """Dummy class to test _operations_to_compressed_reps function."""

    def __init__(self, wires=None):
        super().__init__(wires=wires)

    def decomposition(self):
        decomp = [
            re.ResourceHadamard(wires=self.wires[0]),
            re.ResourceHadamard(wires=self.wires[1]),
            re.ResourceCNOT(wires=[self.wires[0], self.wires[1]]),
        ]

        return decomp


class TestGetResources:
    """Test the core resource tracking pipeline"""

    compressed_rep_data = (
        (
            [
                re.ResourceHadamard(0),
                re.ResourceRX(1.23, 1),
                re.ResourceCNOT(wires=[1, 2]),
            ],
            [
                re.CompressedResourceOp(re.ResourceHadamard, {}),
                re.CompressedResourceOp(re.ResourceRX, {}),
                re.CompressedResourceOp(re.ResourceCNOT, {}),
            ],
        ),
        (
            [
                re.ResourceQFT(wires=[1, 2, 3]),
                re.ResourceIdentity(wires=[1, 2, 3]),
                re.ResourceRot(1.23, 0.45, -6, wires=0),
                re.ResourceQFT(wires=[1, 2]),
            ],
            [
                re.CompressedResourceOp(re.ResourceQFT, {"num_wires": 3}),
                re.CompressedResourceOp(re.ResourceIdentity, {}),
                re.CompressedResourceOp(re.ResourceRot, {}),
                re.CompressedResourceOp(re.ResourceQFT, {"num_wires": 2}),
            ],
        ),
        (
            [
                re.ResourceQFT(wires=[0, 1]),
                DummyOperation(wires=["a", "b"]),
                re.ResourceRY(-0.5, wires=1),
            ],
            [
                re.CompressedResourceOp(re.ResourceQFT, {"num_wires": 2}),
                re.CompressedResourceOp(re.ResourceHadamard, {}),
                re.CompressedResourceOp(re.ResourceHadamard, {}),
                re.CompressedResourceOp(re.ResourceCNOT, {}),
                re.CompressedResourceOp(re.ResourceRY, {}),
            ],
        ),  # Test decomposition logic
    )

    @pytest.mark.parametrize("ops_lst, compressed_reps", compressed_rep_data)
    def test_operations_to_compressed_reps(self, ops_lst, compressed_reps):
        """Test that we can transform a list of operations into compressed reps"""
        computed_compressed_reps = _operations_to_compressed_reps(ops_lst)
        for computed_cr, expected_cr in zip(computed_compressed_reps, compressed_reps):
            assert computed_cr == expected_cr

    compressed_rep_counts = (
        (
            re.ResourceHadamard(wires=0).resource_rep_from_op(),
            defaultdict(int, {re.CompressedResourceOp(re.ResourceHadamard, {}): 1}),
        ),
        (
            re.ResourceRX(1.23, wires=0).resource_rep_from_op(),
            defaultdict(int, {re.CompressedResourceOp(re.ResourceT, {}): 17}),
        ),
        (
            re.ResourceIdentity(wires=[1, 2, 3]).resource_rep_from_op(),
            defaultdict(int, {}),  # Identity has no resources
        ),
        (
            re.ResourceControlledPhaseShift(1.23, wires=[0, 1]).resource_rep_from_op(),
            defaultdict(
                int,
                {
                    re.CompressedResourceOp(re.ResourceT, {}): 51,
                    re.CompressedResourceOp(re.ResourceCNOT, {}): 2,
                },
            ),
        ),
        (
            re.ResourceQFT(wires=[1, 2, 3, 4]).resource_rep_from_op(),
            defaultdict(
                int,
                {
                    re.CompressedResourceOp(re.ResourceT, {}): 306,
                    re.CompressedResourceOp(re.ResourceCNOT, {}): 18,
                    re.CompressedResourceOp(re.ResourceHadamard, {}): 4,
                },
            ),
        ),
    )

    @pytest.mark.parametrize("op_in_gate_set", [True, False])
    @pytest.mark.parametrize("scalar", [1, 2, 5])
    @pytest.mark.parametrize("compressed_rep, expected_counts", compressed_rep_counts)
    def test_counts_from_compressed_res(
        self, scalar, compressed_rep, expected_counts, op_in_gate_set
    ):
        """Test that we can obtain counts from a compressed resource op"""

        if op_in_gate_set:
            # Test that we add op directly to counts if its in the gate_set
            custom_gate_set = {compressed_rep._name}

            base_gate_counts = defaultdict(int)
            _counts_from_compressed_res_op(
                compressed_rep,
                gate_counts_dict=base_gate_counts,
                gate_set=custom_gate_set,
                scalar=scalar,
            )

            assert base_gate_counts == defaultdict(int, {compressed_rep: scalar})

        else:
            expected_counts = copy(expected_counts)
            for resource_op, counts in expected_counts.items():  # scale expected counts
                expected_counts[resource_op] = scalar * counts

            base_gate_counts = defaultdict(int)
            _counts_from_compressed_res_op(
                compressed_rep,
                gate_counts_dict=base_gate_counts,
                gate_set=DefaultGateSet,
                scalar=scalar,
            )

            assert base_gate_counts == expected_counts

    @pytest.mark.parametrize(
        "custom_config, num_T_gates",
        (
            (
                {
                    "error_rz": 10e-2,
                },
                13,
            ),
            (
                {
                    "error_rz": 10e-3,
                },
                17,
            ),
            (
                {
                    "error_rz": 10e-4,
                },
                21,
            ),
        ),
    )
    def test_counts_from_compressed_res_custom_config(self, custom_config, num_T_gates):
        """Test that the function works with custom configs and a non-empty gate_counts_dict"""
        base_gate_counts = defaultdict(
            int, {re.ResourceT.resource_rep(): 3, re.ResourceS.resource_rep(): 5}
        )

        _counts_from_compressed_res_op(
            re.ResourceRZ.resource_rep(),
            base_gate_counts,
            gate_set=DefaultGateSet,
            config=custom_config,
        )
        expected_counts = defaultdict(
            int, {re.ResourceT.resource_rep(): 3 + num_T_gates, re.ResourceS.resource_rep(): 5}
        )

        assert base_gate_counts == expected_counts

    def test_clean_gate_counts(self):
        """Test that the function groups operations by name instead of compressed representation."""

        gate_counts = defaultdict(
            int,
            {
                re.ResourceQFT.resource_rep(5): 1,
                re.ResourceHadamard.resource_rep(): 3,
                re.ResourceCNOT.resource_rep(): 1,
                re.ResourceQFT.resource_rep(3): 4,
            },
        )

        expected_clean_counts = defaultdict(
            int, {"CNOT": 1, "Hadamard": 3, "QFT(5)": 1, "QFT(3)": 4}
        )

        assert _clean_gate_counts(gate_counts) == expected_clean_counts

    @pytest.mark.parametrize(
        "op, expected_resources",
        (
            (re.ResourceHadamard(wires=0), re.Resources(1, 1, defaultdict(int, {"Hadamard": 1}))),
            (re.ResourceRX(1.23, wires=1), re.Resources(1, 17, defaultdict(int, {"T": 17}))),
            (
                re.ResourceQFT(wires=range(5)),
                re.Resources(5, 541, defaultdict(int, {"Hadamard": 5, "CNOT": 26, "T": 510})),
            ),
        ),
    )
    def test_resources_from_operation(self, op, expected_resources):
        """Test that we can extract the resources from an Operation."""
        computed_resources = resources_from_operation(
            op
        )  # add tests that don't use default gate_set and config
        assert computed_resources == expected_resources

    @staticmethod
    def my_qfunc():
        """Dummy qfunc used to test resources_from_qfunc function."""
        for w in range(2):
            re.ResourceHadamard(w)

        re.ResourceCNOT([0, 1])
        re.ResourceRX(1.23, 0)
        re.ResourceRY(-4.56, 1)

        re.ResourceQFT(wires=[0, 1, 2])
        return qml.expval(re.ResourceHadamard(2))

    def test_resources_from_qfunc(self):
        """Test the we can extract the resources from a quantum function."""
        expected_resources_standard = re.Resources(
            num_wires=3,
            num_gates=24,
            gate_types=defaultdict(
                int, {"Hadamard": 5, "CNOT": 7, "SWAP": 1, "RX": 1, "RY": 1, "RZ": 9}
            ),
        )

        computed_resources = resources_from_qfunc(self.my_qfunc, gate_set=_StandardGateSet)()
        assert computed_resources == expected_resources_standard

        expected_resources_custom = re.Resources(
            num_wires=3,
            num_gates=190,
            gate_types=defaultdict(int, {"Hadamard": 5, "CNOT": 10, "T": 175}),
        )

        my_resource_config = copy(resource_config)
        my_resource_config["error_rx"] = 10e-1
        my_resource_config["error_ry"] = 10e-2
        computed_resources = resources_from_qfunc(
            self.my_qfunc, gate_set=DefaultGateSet, config=my_resource_config
        )()

        assert computed_resources == expected_resources_custom

    my_tape = qml.tape.QuantumScript(
        ops=[
            re.ResourceHadamard(0),
            re.ResourceHadamard(1),
            re.ResourceCNOT([0, 1]),
            re.ResourceRX(1.23, 0),
            re.ResourceRY(-4.56, 1),
            re.ResourceQFT(wires=[0, 1, 2]),
        ],
        measurements=[qml.expval(re.ResourceHadamard(2))],
    )

    def test_resources_from_tape(self):
        """Test that we can extract the resources from a quantum tape"""
        expected_resources_standard = re.Resources(
            num_wires=3,
            num_gates=24,
            gate_types=defaultdict(
                int, {"Hadamard": 5, "CNOT": 7, "SWAP": 1, "RX": 1, "RY": 1, "RZ": 9}
            ),
        )

        computed_resources = resources_from_tape(self.my_tape, gate_set=_StandardGateSet)
        assert computed_resources == expected_resources_standard

        expected_resources_custom = re.Resources(
            num_wires=3,
            num_gates=190,
            gate_types=defaultdict(int, {"Hadamard": 5, "CNOT": 10, "T": 175}),
        )

        my_resource_config = copy(resource_config)
        my_resource_config["error_rx"] = 10e-1
        my_resource_config["error_ry"] = 10e-2
        computed_resources = resources_from_tape(
            self.my_tape, gate_set=DefaultGateSet, config=my_resource_config
        )

        assert computed_resources == expected_resources_custom

    def test_get_resources(self):
        """Test that we can dispatch between each of the implementations above"""
        op = re.ResourceControlledPhaseShift(1.23, wires=[0, 1])
        tape = qml.tape.QuantumScript(ops=[op], measurements=[qml.expval(re.ResourceHadamard(0))])

        def circuit():
            re.ResourceControlledPhaseShift(1.23, wires=[0, 1])
            return qml.expval(re.ResourceHadamard(0))

        res_from_op = get_resources(op)
        res_from_tape = get_resources(tape)
        res_from_circuit = get_resources(circuit)()

        expected_resources = re.Resources(
            num_wires=2, num_gates=53, gate_types=defaultdict(int, {"CNOT": 2, "T": 51})
        )

        assert res_from_op == expected_resources
        assert res_from_tape == expected_resources
        assert res_from_circuit == expected_resources
