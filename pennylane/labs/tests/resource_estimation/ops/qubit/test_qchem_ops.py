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


class TestResourceSingleExcitation:
    """Test the SingleExcitation class."""

    @pytest.mark.parametrize("eps", (None, 1e-3))
    def test_resource_params(self, eps):
        """Test that the resource params are correct."""
        if eps:
            op = plre.ResourceSingleExcitation(eps=eps)
        else:
            op = plre.ResourceSingleExcitation()

        assert op.resource_params == {"eps": eps}

    @pytest.mark.parametrize("eps", (None, 1e-3))
    def test_resource_rep(self, eps):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(plre.ResourceSingleExcitation, {"eps": eps})
        assert plre.ResourceSingleExcitation.resource_rep(eps=eps) == expected

    @pytest.mark.parametrize("eps", (None, 1e-3))
    def test_resources(self, eps):
        """Test that the resources are correct."""
        t_dag = plre.ResourceAdjoint.resource_rep(plre.resource_rep(plre.ResourceT))
        s_dag = plre.ResourceAdjoint.resource_rep(plre.resource_rep(plre.ResourceS))

        expected = [
            plre.GateCount(t_dag, 2),
            plre.GateCount(plre.resource_rep(plre.ResourceHadamard), 4),
            plre.GateCount(plre.resource_rep(plre.ResourceS), 2),
            plre.GateCount(s_dag, 2),
            plre.GateCount(plre.resource_rep(plre.ResourceCNOT), 2),
            plre.GateCount(plre.resource_rep(plre.ResourceRZ, {"eps": eps})),
            plre.GateCount(plre.resource_rep(plre.ResourceRY, {"eps": eps})),
            plre.GateCount(plre.resource_rep(plre.ResourceT), 2),
        ]
        assert plre.ResourceSingleExcitation.resource_decomp(eps=eps) == expected
