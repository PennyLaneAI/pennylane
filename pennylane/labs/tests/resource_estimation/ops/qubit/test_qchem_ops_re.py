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

import pennylane.labs.resource_estimation as plre

# pylint: disable=no-self-use,too-many-arguments


class TestResourceSingleExcitation:
    """Test the SingleExcitation class."""

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resource_params(self, precision):
        """Test that the resource params are correct."""
        if precision:
            op = plre.ResourceSingleExcitation(precision=precision)
        else:
            op = plre.ResourceSingleExcitation()

        assert op.resource_params == {"precision": precision}

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resource_rep(self, precision):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceSingleExcitation, 2, {"precision": precision}
        )
        assert plre.ResourceSingleExcitation.resource_rep(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources(self, precision):
        """Test that the resources are correct."""
        t_dag = plre.ResourceAdjoint.resource_rep(plre.resource_rep(plre.ResourceT))
        s_dag = plre.ResourceAdjoint.resource_rep(plre.resource_rep(plre.ResourceS))

        expected = [
            plre.GateCount(t_dag, 2),
            plre.GateCount(plre.resource_rep(plre.ResourceHadamard), 4),
            plre.GateCount(plre.resource_rep(plre.ResourceS), 2),
            plre.GateCount(s_dag, 2),
            plre.GateCount(plre.resource_rep(plre.ResourceCNOT), 2),
            plre.GateCount(plre.resource_rep(plre.ResourceRZ, {"precision": precision})),
            plre.GateCount(plre.resource_rep(plre.ResourceRY, {"precision": precision})),
            plre.GateCount(plre.resource_rep(plre.ResourceT), 2),
        ]
        assert plre.ResourceSingleExcitation.resource_decomp(precision=precision) == expected
