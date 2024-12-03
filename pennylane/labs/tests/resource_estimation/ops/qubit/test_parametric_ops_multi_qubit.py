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
"""Tests for parametric multi qubit resource operators."""

import pytest

import pennylane.labs.resource_estimation as re

# pylint: disable=use-implicit-booleaness-not-comparison,no-self-use


class TestMultiRZ:
    """Test the ResourceMultiRZ class."""

    @pytest.mark.parametrize("num_wires", range(1, 10))
    def test_resource_params(self, num_wires):
        """Test that the resource params are correct."""
        op = re.ResourceMultiRZ(0.5, range(num_wires))
        assert op.resource_params() == {"num_wires": num_wires}

    @pytest.mark.parametrize("num_wires", range(1, 10))
    def test_resource_rep(self, num_wires):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceMultiRZ, {"num_wires": num_wires})
        assert re.ResourceMultiRZ.resource_rep(num_wires) == expected

    @pytest.mark.parametrize("num_wires", range(1, 10))
    def test_resources(self, num_wires):
        """Test that the resources are correct."""
        expected = {
            re.ResourceCNOT.resource_rep(): 2 * (num_wires - 1),
            re.ResourceRZ.resource_rep(): 1,
        }
        assert re.ResourceMultiRZ.resources(num_wires) == expected

    @pytest.mark.parametrize("num_wires", range(1, 10))
    def test_resources_from_rep(self, num_wires):
        """Test that the resources can be computed from the compressed representation and params."""
        op = re.ResourceMultiRZ(0.5, wires=range(num_wires))
        expected = {
            re.ResourceCNOT.resource_rep(): 2 * (num_wires - 1),
            re.ResourceRZ.resource_rep(): 1,
        }

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestPauliRot:
    """Test the ResourcePauliRot class."""

    pauli_words = ("I", "XYZ", "XXX", "XIYIZIX", "III")

    @pytest.mark.parametrize("pauli_word", pauli_words)
    def test_resource_params(self, pauli_word):
        """Test that the resource params are correct."""
        op = re.ResourcePauliRot(theta=0.5, pauli_word=pauli_word, wires=range(len(pauli_word)))
        assert op.resource_params() == {"pauli_word": pauli_word}

    @pytest.mark.parametrize("pauli_word", pauli_words)
    def test_resource_rep(self, pauli_word):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourcePauliRot, {"pauli_word": pauli_word})
        assert re.ResourcePauliRot.resource_rep(pauli_word) == expected

    expected_h_count = (0, 4, 6, 6, 0)
    expected_s_count = (0, 1, 0, 1, 0)
    params = zip(pauli_words, expected_h_count, expected_s_count)

    @pytest.mark.parametrize("pauli_word, expected_h_count, expected_s_count", params)
    def test_resources(self, pauli_word, expected_h_count, expected_s_count):
        """Test that the resources are correct."""
        active_wires = len(pauli_word.replace("I", ""))

        if set(pauli_word) == {"I"}:
            expected = {re.ResourceGlobalPhase.resource_rep(): 1}
        else:
            expected = {
                re.ResourceHadamard.resource_rep(): expected_h_count,
                re.ResourceS.resource_rep(): 4 * expected_s_count,
                re.ResourceRZ.resource_rep(): 1,
                re.ResourceCNOT.resource_rep(): 2 * (active_wires - 1),
            }

        assert re.ResourcePauliRot.resources(pauli_word) == expected

    @pytest.mark.parametrize("pauli_word, expected_h_count, expected_rx_count", params)
    def test_resources_from_rep(self, pauli_word, expected_h_count, expected_rx_count):
        """Test that the resources can be computed from the compressed representation and params."""
        op = re.ResourcePauliRot(0.5, pauli_word, wires=range(len(pauli_word)))
        active_wires = len(pauli_word.replace("I", ""))

        if set(pauli_word) == {"I"}:
            expected = {re.ResourceGlobalPhase.resource_rep(): 1}
        else:
            expected = {
                re.ResourceHadamard.resource_rep(): expected_h_count,
                re.ResourceRX.resource_rep(): expected_rx_count,
                re.ResourceMultiRZ.resource_rep(active_wires): 1,
            }

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestIsingXX:
    """Test the IsingXX class."""

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourceIsingXX(0.5, wires=[0, 1])
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceIsingXX, {})
        assert re.ResourceIsingXX.resource_rep() == expected

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceCNOT.resource_rep(): 2,
            re.ResourceRX.resource_rep(): 1,
        }
        assert re.ResourceIsingXX.resources() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation and params."""
        op = re.ResourceIsingXX(0.5, wires=[0, 1])
        expected = {
            re.ResourceCNOT.resource_rep(): 2,
            re.ResourceRX.resource_rep(): 1,
        }
        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestIsingXY:
    """Test the IsingXY class."""

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourceIsingXY(0.5, wires=[0, 1])
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceIsingXY, {})
        assert re.ResourceIsingXY.resource_rep() == expected

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceHadamard.resource_rep(): 2,
            re.ResourceCY.resource_rep(): 2,
            re.ResourceRY.resource_rep(): 1,
            re.ResourceRX.resource_rep(): 1,
        }
        assert re.ResourceIsingXY.resources() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation and params."""
        op = re.ResourceIsingXY(0.5, wires=[0, 1])
        expected = {
            re.ResourceHadamard.resource_rep(): 2,
            re.ResourceCY.resource_rep(): 2,
            re.ResourceRY.resource_rep(): 1,
            re.ResourceRX.resource_rep(): 1,
        }
        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestIsingYY:
    """Test the IsingYY class."""

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourceIsingYY(0.5, wires=[0, 1])
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceIsingYY, {})
        assert re.ResourceIsingYY.resource_rep() == expected

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceCY.resource_rep(): 2,
            re.ResourceRY.resource_rep(): 1,
        }
        assert re.ResourceIsingYY.resources() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation and params."""
        op = re.ResourceIsingYY(0.5, wires=[0, 1])
        expected = {
            re.ResourceCY.resource_rep(): 2,
            re.ResourceRY.resource_rep(): 1,
        }
        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestIsingZZ:
    """Test the IsingZZ class."""

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourceIsingZZ(0.5, wires=[0, 1])
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceIsingZZ, {})
        assert re.ResourceIsingZZ.resource_rep() == expected

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceCNOT.resource_rep(): 2,
            re.ResourceRZ.resource_rep(): 1,
        }
        assert re.ResourceIsingZZ.resources() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation and params."""
        op = re.ResourceIsingZZ(0.5, wires=[0, 1])
        expected = {
            re.ResourceCNOT.resource_rep(): 2,
            re.ResourceRZ.resource_rep(): 1,
        }
        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestPSWAP:
    """Test the PSWAP class."""

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourcePSWAP(0.5, wires=[0, 1])
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourcePSWAP, {})
        assert re.ResourcePSWAP.resource_rep() == expected

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceSWAP.resource_rep(): 1,
            re.ResourceCNOT.resource_rep(): 2,
            re.ResourcePhaseShift.resource_rep(): 1,
        }
        assert re.ResourcePSWAP.resources() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation and params."""
        op = re.ResourcePSWAP(0.5, wires=[0, 1])
        expected = {
            re.ResourceSWAP.resource_rep(): 1,
            re.ResourceCNOT.resource_rep(): 2,
            re.ResourcePhaseShift.resource_rep(): 1,
        }
        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected
