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
"""Tests for controlled resource operators."""

from collections import defaultdict

import pennylane.labs.estimator_beta as qre
from pennylane.estimator.resource_operator import GateCount, resource_rep

# pylint: disable= no-self-use


class TestCH:
    """Test the Resource CH operation"""

    op = qre.CH(wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = [
            GateCount(qre.Hadamard.resource_rep(), 2),
            GateCount(qre.T.resource_rep(), 1),
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.T)}), 1),
            GateCount(qre.S.resource_rep(), 1),
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1),
            GateCount(qre.CNOT.resource_rep(), 1),
        ]
        assert qre.ch_resource_decomp(**self.op.resource_params) == expected_resources

    def test_resources_estimate(self):
        """Test that correct resources are produced when using estimate."""
        expected_resources = qre.Resources(
            zeroed_wires=0,
            any_state_wires=0,
            algo_wires=2,
            gate_types=defaultdict(
                int,
                {
                    resource_rep(qre.CNOT): 1,
                    resource_rep(qre.T): 2,
                    resource_rep(qre.Z): 2,
                    resource_rep(qre.S): 3,
                    resource_rep(qre.Hadamard): 2,
                },
            ),
        )
        assert qre.estimate(self.op) == expected_resources

    def test_toffoli_based_resources(self):
        """Test that the resources method produces the expected resources when using a Toffoli-based decomposition."""

        expected_resources = [
            qre.Allocate(1),
            GateCount(qre.Hadamard.resource_rep(), 5),
            GateCount(qre.S.resource_rep(), 2),
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1),
            GateCount(resource_rep(qre.Toffoli), 1),
            GateCount(resource_rep(qre.CNOT), 5),
            GateCount(resource_rep(qre.CZ), 1),
            GateCount(resource_rep(qre.X), 4),
            qre.Deallocate(1),
        ]
        result = qre.ch_toffoli_based_resource_decomp(**self.op.resource_params)
        for r, e in zip(result, expected_resources):
            if hasattr(r, "equal"):
                assert r.equal(e)
            else:
                assert r == e
