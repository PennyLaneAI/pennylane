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
Tests for qchem resource operators.
"""
import pytest

import pennylane.estimator as qre

# pylint: disable=no-self-use,too-many-arguments


class TestSingleExcitation:
    """Test the Resource SingleExcitation class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 2 wires, got 3"):
            qre.SingleExcitation(wires=[0, 1, 2])

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resource_params(self, precision):
        """Test that the resource params are correct."""
        if precision:
            op = qre.SingleExcitation(precision=precision)
        else:
            op = qre.SingleExcitation()

        assert op.resource_params == {"precision": precision}

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resource_rep(self, precision):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(qre.SingleExcitation, 2, {"precision": precision})
        assert qre.SingleExcitation.resource_rep(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources(self, precision):
        """Test that the resources are correct."""
        t_dag = qre.Adjoint.resource_rep(qre.resource_rep(qre.T))
        s_dag = qre.Adjoint.resource_rep(qre.resource_rep(qre.S))

        expected = [
            qre.GateCount(t_dag, 2),
            qre.GateCount(qre.resource_rep(qre.Hadamard), 4),
            qre.GateCount(qre.resource_rep(qre.S), 2),
            qre.GateCount(s_dag, 2),
            qre.GateCount(qre.resource_rep(qre.CNOT), 2),
            qre.GateCount(qre.resource_rep(qre.RZ, {"precision": precision})),
            qre.GateCount(qre.resource_rep(qre.RY, {"precision": precision})),
            qre.GateCount(qre.resource_rep(qre.T), 2),
        ]
        assert qre.SingleExcitation.resource_decomp(precision=precision) == expected
