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
Tests for quantum algorithmic subroutines resource operators.
"""
import pytest

from pennylane.labs import resource_estimation as plre
from pennylane.labs.resource_estimation import AllocWires, GateCount, resource_rep

# pylint: disable=no-self-use


class TestUniformStatePrep:
    """Test the ResourceUniformStatePrep class."""

    @pytest.mark.parametrize(
        "num_states",
        (10, 6, 4),
    )
    def test_resource_params(self, num_states):
        """Test that the resource params are correct."""
        op = plre.ResourceUniformStatePrep(num_states)
        assert op.resource_params == {"num_states": num_states}

    @pytest.mark.parametrize(
        "num_states",
        (10, 6, 4),
    )
    def test_resource_rep(self, num_states):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceUniformStatePrep, {"num_states": num_states}
        )
        assert plre.ResourceUniformStatePrep.resource_rep(num_states) == expected

    @pytest.mark.parametrize(
        "num_states, expected_res",
        (
            (
                10,
                [
                    GateCount(resource_rep(plre.ResourceHadamard), 10),
                    GateCount(
                        resource_rep(
                            plre.ResourceIntegerComparator, {"val": 5, "register_size": 3}
                        ),
                        1,
                    ),
                    GateCount(resource_rep(plre.ResourceRZ), 2),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {
                                "base_cmpr_op": resource_rep(
                                    plre.ResourceIntegerComparator, {"val": 5, "register_size": 3}
                                )
                            },
                        ),
                        1,
                    ),
                ],
            ),
            (
                6,
                [
                    GateCount(resource_rep(plre.ResourceHadamard), 7),
                    GateCount(
                        resource_rep(
                            plre.ResourceIntegerComparator, {"val": 3, "register_size": 2}
                        ),
                        1,
                    ),
                    GateCount(resource_rep(plre.ResourceRZ), 2),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {
                                "base_cmpr_op": resource_rep(
                                    plre.ResourceIntegerComparator, {"val": 3, "register_size": 2}
                                )
                            },
                        ),
                        1,
                    ),
                ],
            ),
            (
                4,
                [
                    GateCount(resource_rep(plre.ResourceHadamard), 2),
                ],
            ),
        ),
    )
    def test_resources(self, num_states, expected_res):
        """Test that the resources are correct."""
        print(plre.ResourceUniformStatePrep.resource_decomp(num_states))
        assert plre.ResourceUniformStatePrep.resource_decomp(num_states) == expected_res


class TestAliasSampling:
    """Test the ResourceAliasSampling class."""

    @pytest.mark.parametrize(
        "num_coeffs, precision",
        (
            (10, None),
            (6, None),
            (4, 1e-6),
        ),
    )
    def test_resource_params(self, num_coeffs, precision):
        """Test that the resource params are correct."""
        op = plre.ResourceAliasSampling(num_coeffs, precision=precision)
        assert op.resource_params == {"num_coeffs": num_coeffs, "precision": precision}

    @pytest.mark.parametrize(
        "num_coeffs, precision",
        (
            (10, None),
            (6, None),
            (4, 1e-6),
        ),
    )
    def test_resource_rep(self, num_coeffs, precision):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceAliasSampling, {"num_coeffs": num_coeffs, "precision": precision}
        )
        assert plre.ResourceAliasSampling.resource_rep(num_coeffs, precision) == expected

    @pytest.mark.parametrize(
        "num_coeffs, precision, expected_res",
        (
            (
                10,
                None,
                [
                    AllocWires(65),
                    GateCount(resource_rep(plre.ResourceUniformStatePrep, {"num_states": 10})),
                    GateCount(resource_rep(plre.ResourceHadamard), 30),
                    GateCount(
                        resource_rep(
                            plre.ResourceQROM, {"num_bitstrings": 10, "size_bitstring": 34}
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceRegisterComparator,
                            {"a_num_qubits": 30, "b_num_qubits": 30},
                        ),
                        1,
                    ),
                    GateCount(resource_rep(plre.ResourceCSWAP), 4),
                ],
            ),
            (
                6,
                None,
                [
                    AllocWires(64),
                    GateCount(resource_rep(plre.ResourceUniformStatePrep, {"num_states": 6})),
                    GateCount(resource_rep(plre.ResourceHadamard), 30),
                    GateCount(
                        resource_rep(
                            plre.ResourceQROM, {"num_bitstrings": 6, "size_bitstring": 33}
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceRegisterComparator,
                            {"a_num_qubits": 30, "b_num_qubits": 30},
                        ),
                        1,
                    ),
                    GateCount(resource_rep(plre.ResourceCSWAP), 3),
                ],
            ),
            (
                4,
                1e-6,
                [
                    AllocWires(43),
                    GateCount(resource_rep(plre.ResourceUniformStatePrep, {"num_states": 4})),
                    GateCount(resource_rep(plre.ResourceHadamard), 20),
                    GateCount(
                        resource_rep(
                            plre.ResourceQROM, {"num_bitstrings": 4, "size_bitstring": 22}
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceRegisterComparator,
                            {"a_num_qubits": 20, "b_num_qubits": 20},
                        ),
                        1,
                    ),
                    GateCount(resource_rep(plre.ResourceCSWAP), 2),
                ],
            ),
        ),
    )
    def test_resources(self, num_coeffs, precision, expected_res):
        """Test that the resources are correct."""
        if precision is None:
            config = {"precision_alias_sampling": 1e-9}
            assert (
                plre.ResourceAliasSampling.resource_decomp(num_coeffs, precision, config=config)
                == expected_res
            )
        else:
            assert plre.ResourceAliasSampling.resource_decomp(num_coeffs, precision) == expected_res
