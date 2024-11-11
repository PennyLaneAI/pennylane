# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Tests for controlled resource operators.
"""
import pytest

import pennylane.labs.resource_estimation as re

# pylint: disable=no-self-use


class TestControlledPhaseShift:
    """Test ResourceControlledPhaseShift"""

    params = [(1.2, [0, 1]), (2.4, [2, 3])]

    @pytest.mark.parametrize("phi, wires", params)
    def test_resources(self, phi, wires):
        """Test the resources method"""

        op = re.ResourceControlledPhaseShift(phi, wires)

        expected = {
            re.CompressedResourceOp(re.ResourceCNOT, {}): 2,
            re.CompressedResourceOp(re.ResourceRZ, {}): 3,
        }

        assert op.resources() == expected

    @pytest.mark.parametrize("phi, wires", params)
    def test_resource_params(self, phi, wires):
        """Test the resource parameters"""

        op = re.ResourceControlledPhaseShift(phi, wires)
        assert op.resource_params() == {}  # pylint: disable=use-implicit-booleaness-not-comparison

    @pytest.mark.parametrize("phi, wires", params)
    def test_resource_rep(self, phi, wires):
        """Test the compressed representation"""

        op = re.ResourceControlledPhaseShift(phi, wires)
        expected = re.CompressedResourceOp(re.ResourceControlledPhaseShift, {})

        assert op.resource_rep() == expected

    @pytest.mark.parametrize("phi, wires", params)
    def test_resource_rep_from_op(self, phi, wires):
        """Test resource_rep_from_op method"""

        op = re.ResourceControlledPhaseShift(phi, wires)
        assert op.resource_rep_from_op() == re.ResourceControlledPhaseShift.resource_rep(
            **op.resource_params()
        )

    @pytest.mark.parametrize("phi, wires", params)
    def test_resources_from_rep(self, phi, wires):
        """Compute the resources from the compressed representation"""

        op = re.ResourceControlledPhaseShift(phi, wires)

        expected = {
            re.CompressedResourceOp(re.ResourceCNOT, {}): 2,
            re.CompressedResourceOp(re.ResourceRZ, {}): 3,
        }

        assert (
            op.resources(
                **re.ResourceControlledPhaseShift.resource_rep(**op.resource_params()).params
            )
            == expected
        )


class TestCNOT:
    """Test ResourceCNOT"""

    def test_resources(self):
        """Test that the resources method is not implemented"""
        op = re.ResourceCNOT([0, 1])
        with pytest.raises(re.ResourcesNotDefined):
            op.resources()

    def test_resource_rep(self):
        """Test the compressed representation"""
        op = re.ResourceCNOT([0, 1])
        expected = re.CompressedResourceOp(re.ResourceCNOT, {})
        assert op.resource_rep() == expected
