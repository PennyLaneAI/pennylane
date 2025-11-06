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
"""Tests for parametric multi qubit resource operators."""

import pytest

import pennylane.labs.resource_estimation as re

# pylint: disable=use-implicit-booleaness-not-comparison,no-self-use,too-many-arguments


class TestMultiRZ:
    """Test the ResourceMultiRZ class."""

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("num_wires", range(1, 5))
    def test_resource_params(self, num_wires, precision):
        """Test that the resource params are correct."""
        if precision:
            op = re.ResourceMultiRZ(num_wires, precision=precision)
        else:
            op = re.ResourceMultiRZ(num_wires)

        assert op.resource_params == {"num_wires": num_wires, "precision": precision}

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("num_wires", range(1, 5))
    def test_resource_rep(self, num_wires, precision):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(
            re.ResourceMultiRZ, num_wires, {"num_wires": num_wires, "precision": precision}
        )
        assert re.ResourceMultiRZ.resource_rep(num_wires, precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("num_wires", range(1, 5))
    def test_resources(self, num_wires, precision):
        """Test that the resources are correct."""
        expected = [
            re.GateCount(re.ResourceCNOT.resource_rep(), 2 * (num_wires - 1)),
            re.GateCount(re.ResourceRZ.resource_rep(precision=precision)),
        ]
        assert re.ResourceMultiRZ.resource_decomp(num_wires, precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("num_wires", range(1, 5))
    def test_resources_from_rep(self, num_wires, precision):
        """Test that the resources can be computed from the compressed representation and params."""
        op = re.ResourceMultiRZ(num_wires, precision)
        expected = [
            re.GateCount(re.ResourceCNOT.resource_rep(), 2 * (num_wires - 1)),
            re.GateCount(re.ResourceRZ.resource_rep(precision=precision)),
        ]

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("num_wires", range(1, 5))
    def test_adjoint_decomp(self, num_wires, precision):
        """Test that the adjoint decomposition is correct."""
        expected = [
            re.GateCount(re.ResourceMultiRZ.resource_rep(num_wires=num_wires, precision=precision))
        ]
        assert (
            re.ResourceMultiRZ.adjoint_resource_decomp(num_wires=num_wires, precision=precision)
            == expected
        )

    ctrl_data = (
        (
            1,
            0,
            [
                re.GateCount(re.resource_rep(re.ResourceCNOT()), 4),
                re.GateCount(
                    re.ResourceControlled.resource_rep(re.ResourceRZ.resource_rep(1e-3), 1, 0)
                ),
            ],
        ),
        (
            1,
            1,
            [
                re.GateCount(re.resource_rep(re.ResourceCNOT()), 4),
                re.GateCount(
                    re.ResourceControlled.resource_rep(re.ResourceRZ.resource_rep(1e-3), 1, 1)
                ),
            ],
        ),
        (
            2,
            0,
            [
                re.GateCount(re.resource_rep(re.ResourceCNOT()), 4),
                re.GateCount(
                    re.ResourceControlled.resource_rep(re.ResourceRZ.resource_rep(1e-3), 2, 0)
                ),
            ],
        ),
        (
            3,
            2,
            [
                re.GateCount(re.resource_rep(re.ResourceCNOT()), 4),
                re.GateCount(
                    re.ResourceControlled.resource_rep(re.ResourceRZ.resource_rep(1e-3), 3, 2)
                ),
            ],
        ),
    )

    @pytest.mark.parametrize("num_ctrl_wires, num_ctrl_values, expected_res", ctrl_data)
    def test_resource_controlled(self, num_ctrl_wires, num_ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""

        op = re.ResourceMultiRZ(num_wires=3, precision=1e-3)
        op2 = re.ResourceControlled(
            op,
            num_ctrl_wires,
            num_ctrl_values,
        )

        assert (
            op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, **op.resource_params)
            == expected_res
        )
        assert op2.resource_decomp(**op2.resource_params) == expected_res

    @pytest.mark.parametrize("z", range(1, 5))
    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("num_wires", range(1, 5))
    def test_pow_decomp(self, z, num_wires, precision):
        """Test that the pow decomposition is correct."""
        op = re.ResourceMultiRZ(num_wires, precision=precision)
        expected_res = [re.GateCount(re.ResourceMultiRZ.resource_rep(num_wires, precision))]
        assert op.pow_resource_decomp(z, **op.resource_params) == expected_res


class TestPauliRot:
    """Test the ResourcePauliRot class."""

    pauli_words = ("I", "XYZ", "XXX", "XIYIZIX", "III")

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("pauli_string", pauli_words)
    def test_resource_params(self, pauli_string, precision):
        """Test that the resource params are correct."""
        op = re.ResourcePauliRot(pauli_string=pauli_string, precision=precision)
        assert op.resource_params == {"pauli_string": pauli_string, "precision": precision}

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("pauli_string", pauli_words)
    def test_resource_rep(self, pauli_string, precision):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(
            re.ResourcePauliRot,
            len(pauli_string),
            {"pauli_string": pauli_string, "precision": precision},
        )
        assert re.ResourcePauliRot.resource_rep(pauli_string, precision) == expected

    expected_h_count = (0, 4, 6, 6, 0)
    expected_s_count = (0, 1, 0, 1, 0)
    params = zip(pauli_words, expected_h_count, expected_s_count)

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("pauli_string, expected_h_count, expected_s_count", params)
    def test_resources(self, pauli_string, expected_h_count, expected_s_count, precision):
        """Test that the resources are correct."""
        active_wires = len(pauli_string.replace("I", ""))

        if set(pauli_string) == {"I"}:
            expected = [re.GateCount(re.ResourceGlobalPhase.resource_rep())]
        else:
            expected = []

            if expected_h_count:
                expected.append(re.GateCount(re.ResourceHadamard.resource_rep(), expected_h_count))

            if expected_s_count:
                expected.append(re.GateCount(re.ResourceS.resource_rep(), expected_s_count))
                expected.append(
                    re.GateCount(
                        re.ResourceAdjoint.resource_rep(re.ResourceS.resource_rep()),
                        expected_s_count,
                    )
                )

            expected.append(re.GateCount(re.ResourceRZ.resource_rep(precision=precision)))
            expected.append(re.GateCount(re.ResourceCNOT.resource_rep(), 2 * (active_wires - 1)))

        assert re.ResourcePauliRot.resource_decomp(pauli_string, precision=precision) == expected

    def test_resources_empty_pauli_string(self):
        """Test that the resources method produces the correct result for an empty pauli string."""
        expected = [re.GateCount(re.ResourceGlobalPhase.resource_rep())]
        assert re.ResourcePauliRot.resource_decomp(pauli_string="") == expected

    @pytest.mark.parametrize("pauli_string, expected_h_count, expected_s_count", params)
    def test_resources_from_rep(self, pauli_string, expected_h_count, expected_s_count):
        """Test that the resources can be computed from the compressed representation and params."""
        op = re.ResourcePauliRot(0.5, pauli_string, wires=range(len(pauli_string)))
        active_wires = len(pauli_string.replace("I", ""))

        if set(pauli_string) == {"I"}:
            expected = [re.GateCount(re.ResourceGlobalPhase.resource_rep())]
        else:
            expected = [
                re.GateCount(re.ResourceRZ.resource_rep()),
                re.GateCount(re.ResourceCNOT.resource_rep(), 2 * (active_wires - 1)),
            ]

            if expected_h_count:
                expected.append(re.GateCount(re.ResourceHadamard.resource_rep(), expected_h_count))

            if expected_s_count:
                expected.append(re.GateCount(re.ResourceS.resource_rep(), expected_s_count))
                expected.append(
                    re.GateCount(
                        re.ResourceAdjoint.resource_rep(re.ResourceS.resource_rep()),
                        expected_s_count,
                    )
                )

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("pauli_word", pauli_words)
    def test_adjoint_decomp(self, pauli_word, precision):
        """Test that the adjoint decomposition is correct."""
        expected = [
            re.GateCount(
                re.ResourcePauliRot.resource_rep(pauli_string=pauli_word, precision=precision)
            )
        ]
        assert (
            re.ResourcePauliRot.adjoint_resource_decomp(
                pauli_string=pauli_word, precision=precision
            )
            == expected
        )

    ctrl_data = (
        (
            "XXX",
            1,
            0,
            [
                re.GateCount(re.ResourceHadamard.resource_rep(), 6),
                re.GateCount(
                    re.ResourceControlled.resource_rep(
                        re.ResourceRZ.resource_rep(precision=1e-5), 1, 0
                    ),
                    1,
                ),
                re.GateCount(re.ResourceCNOT.resource_rep(), 4),
            ],
        ),
        (
            "XXX",
            1,
            1,
            [
                re.GateCount(re.ResourceHadamard.resource_rep(), 6),
                re.GateCount(
                    re.ResourceControlled.resource_rep(
                        re.ResourceRZ.resource_rep(precision=1e-5), 1, 1
                    ),
                    1,
                ),
                re.GateCount(re.ResourceCNOT.resource_rep(), 4),
            ],
        ),
        (
            "XXX",
            2,
            0,
            [
                re.GateCount(re.ResourceHadamard.resource_rep(), 6),
                re.GateCount(
                    re.ResourceControlled.resource_rep(
                        re.ResourceRZ.resource_rep(precision=1e-5), 2, 0
                    ),
                    1,
                ),
                re.GateCount(re.ResourceCNOT.resource_rep(), 4),
            ],
        ),
        (
            "XIYIZIX",
            1,
            0,
            [
                re.GateCount(re.ResourceHadamard.resource_rep(), 6),
                re.GateCount(re.ResourceS.resource_rep(), 1),
                re.GateCount(re.ResourceAdjoint.resource_rep(re.ResourceS.resource_rep()), 1),
                re.GateCount(
                    re.ResourceControlled.resource_rep(
                        re.ResourceRZ.resource_rep(precision=1e-5), 1, 0
                    ),
                    1,
                ),
                re.GateCount(re.ResourceCNOT.resource_rep(), 6),
            ],
        ),
        (
            "XIYIZIX",
            1,
            1,
            [
                re.GateCount(re.ResourceHadamard.resource_rep(), 6),
                re.GateCount(re.ResourceS.resource_rep(), 1),
                re.GateCount(re.ResourceAdjoint.resource_rep(re.ResourceS.resource_rep()), 1),
                re.GateCount(
                    re.ResourceControlled.resource_rep(
                        re.ResourceRZ.resource_rep(precision=1e-5), 1, 1
                    ),
                    1,
                ),
                re.GateCount(re.ResourceCNOT.resource_rep(), 6),
            ],
        ),
        (
            "XIYIZIX",
            2,
            0,
            [
                re.GateCount(re.ResourceHadamard.resource_rep(), 6),
                re.GateCount(re.ResourceS.resource_rep(), 1),
                re.GateCount(re.ResourceAdjoint.resource_rep(re.ResourceS.resource_rep()), 1),
                re.GateCount(
                    re.ResourceControlled.resource_rep(
                        re.ResourceRZ.resource_rep(precision=1e-5), 2, 0
                    ),
                    1,
                ),
                re.GateCount(re.ResourceCNOT.resource_rep(), 6),
            ],
        ),
        (
            "III",
            1,
            0,
            [
                re.GateCount(
                    re.ResourceControlled.resource_rep(re.ResourceGlobalPhase.resource_rep(), 1, 0)
                )
            ],
        ),
        (
            "X",
            1,
            1,
            [
                re.GateCount(
                    re.ResourceControlled.resource_rep(
                        re.ResourceRX.resource_rep(precision=1e-5), 1, 1
                    )
                )
            ],
        ),
        (
            "Y",
            2,
            0,
            [
                re.GateCount(
                    re.ResourceControlled.resource_rep(
                        re.ResourceRY.resource_rep(precision=1e-5), 2, 0
                    )
                )
            ],
        ),
    )

    @pytest.mark.parametrize("pauli_word, num_ctrl_wires, num_ctrl_values, expected_res", ctrl_data)
    def test_resource_controlled(self, num_ctrl_wires, num_ctrl_values, pauli_word, expected_res):
        """Test that the controlled resources are as expected"""

        op = re.ResourcePauliRot(pauli_word, precision=1e-5)
        op2 = re.ResourceControlled(op, num_ctrl_wires, num_ctrl_values)

        assert (
            op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, **op.resource_params)
            == expected_res
        )
        assert op2.resource_decomp(**op2.resource_params) == expected_res

    @pytest.mark.parametrize("z", range(1, 5))
    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("pauli_word", pauli_words)
    def test_pow_decomp(self, z, pauli_word, precision):
        """Test that the pow decomposition is correct."""
        op = re.ResourcePauliRot(pauli_string=pauli_word, precision=precision)
        expected_res = [
            re.GateCount(
                re.ResourcePauliRot.resource_rep(pauli_string=pauli_word, precision=precision)
            )
        ]
        assert op.pow_resource_decomp(z, **op.resource_params) == expected_res

        op2 = re.ResourcePow(op, z)
        assert op2.resource_decomp(**op2.resource_params) == expected_res


class TestIsingXX:
    """Test the IsingXX class."""

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resource_params(self, precision):
        """Test that the resource params are correct."""
        if precision:
            op = re.ResourceIsingXX(precision=precision)
        else:
            op = re.ResourceIsingXX()

        assert op.resource_params == {"precision": precision}

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resource_rep(self, precision):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceIsingXX, 2, {"precision": precision})
        assert re.ResourceIsingXX.resource_rep(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources(self, precision):
        """Test that the resources are correct."""
        expected = [
            re.GateCount(re.ResourceCNOT.resource_rep(), 2),
            re.GateCount(re.ResourceRX.resource_rep(precision=precision)),
        ]
        assert re.ResourceIsingXX.resource_decomp(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources_pow(self, precision):
        """Test that the pow resources are correct."""
        expected = [re.GateCount(re.ResourceIsingXX.resource_rep(precision=precision))]
        assert re.ResourceIsingXX.pow_resource_decomp(pow_z=3, precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources_adjoint(self, precision):
        """Test that the adjoint resources are correct."""
        expected = [re.GateCount(re.ResourceIsingXX.resource_rep(precision=precision))]
        assert re.ResourceIsingXX.adjoint_resource_decomp(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources_controlled(self, precision):
        """Test that the controlled resources are correct."""
        expected = [
            re.GateCount(re.ResourceCNOT.resource_rep(), 2),
            re.GateCount(
                re.ResourceControlled.resource_rep(
                    re.ResourceRX.resource_rep(precision=precision),
                    3,
                    2,
                )
            ),
        ]
        op = re.ResourceControlled(
            re.ResourceIsingXX(precision=precision), num_ctrl_wires=3, num_ctrl_values=2
        )
        assert op.resource_decomp(**op.resource_params) == expected


class TestIsingXY:
    """Test the IsingXY class."""

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resource_params(self, precision):
        """Test that the resource params are correct."""
        if precision:
            op = re.ResourceIsingXY(precision=precision)
        else:
            op = re.ResourceIsingXY()

        assert op.resource_params == {"precision": precision}

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resource_rep(self, precision):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceIsingXY, 2, {"precision": precision})
        assert re.ResourceIsingXY.resource_rep(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources(self, precision):
        """Test that the resources are correct."""
        expected = [
            re.GateCount(re.ResourceHadamard.resource_rep(), 2),
            re.GateCount(re.ResourceCY.resource_rep(), 2),
            re.GateCount(re.ResourceRY.resource_rep(precision=precision)),
            re.GateCount(re.ResourceRX.resource_rep(precision=precision)),
        ]
        assert re.ResourceIsingXY.resource_decomp(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources_pow(self, precision):
        """Test that the pow resources are correct."""
        expected = [re.GateCount(re.ResourceIsingXY.resource_rep(precision=precision))]
        assert re.ResourceIsingXY.pow_resource_decomp(pow_z=3, precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources_adjoint(self, precision):
        """Test that the adjoint resources are correct."""
        expected = [re.GateCount(re.ResourceIsingXY.resource_rep(precision=precision))]
        assert re.ResourceIsingXY.adjoint_resource_decomp(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources_controlled(self, precision):
        """Test that the controlled resources are correct."""
        expected = [
            re.GateCount(re.ResourceHadamard.resource_rep(), 2),
            re.GateCount(re.ResourceCY.resource_rep(), 2),
            re.GateCount(
                re.ResourceControlled.resource_rep(
                    re.ResourceRY.resource_rep(precision=precision),
                    3,
                    2,
                )
            ),
            re.GateCount(
                re.ResourceControlled.resource_rep(
                    re.ResourceRX.resource_rep(precision=precision),
                    3,
                    2,
                )
            ),
        ]
        op = re.ResourceControlled(
            re.ResourceIsingXY(precision=precision), num_ctrl_wires=3, num_ctrl_values=2
        )
        assert op.resource_decomp(**op.resource_params) == expected


class TestIsingYY:
    """Test the IsingYY class."""

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resource_params(self, precision):
        """Test that the resource params are correct."""
        if precision:
            op = re.ResourceIsingYY(precision=precision)
        else:
            op = re.ResourceIsingYY()

        assert op.resource_params == {"precision": precision}

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resource_rep(self, precision):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceIsingYY, 2, {"precision": precision})
        assert re.ResourceIsingYY.resource_rep(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources(self, precision):
        """Test that the resources are correct."""
        expected = [
            re.GateCount(re.ResourceCY.resource_rep(), 2),
            re.GateCount(re.ResourceRY.resource_rep(precision=precision)),
        ]
        assert re.ResourceIsingYY.resource_decomp(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources_pow(self, precision):
        """Test that the pow resources are correct."""
        expected = [re.GateCount(re.ResourceIsingYY.resource_rep(precision=precision))]
        assert re.ResourceIsingYY.pow_resource_decomp(pow_z=3, precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources_adjoint(self, precision):
        """Test that the adjoint resources are correct."""
        expected = [re.GateCount(re.ResourceIsingYY.resource_rep(precision=precision))]
        assert re.ResourceIsingYY.adjoint_resource_decomp(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources_controlled(self, precision):
        """Test that the controlled resources are correct."""
        expected = [
            re.GateCount(re.ResourceCY.resource_rep(), 2),
            re.GateCount(
                re.ResourceControlled.resource_rep(
                    re.ResourceRY.resource_rep(precision=precision),
                    3,
                    2,
                )
            ),
        ]
        op = re.ResourceControlled(
            re.ResourceIsingYY(precision=precision), num_ctrl_wires=3, num_ctrl_values=2
        )
        assert op.resource_decomp(**op.resource_params) == expected


class TestIsingZZ:
    """Test the IsingZZ class."""

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resource_params(self, precision):
        """Test that the resource params are correct."""
        if precision:
            op = re.ResourceIsingZZ(precision=precision)
        else:
            op = re.ResourceIsingZZ()

        assert op.resource_params == {"precision": precision}

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resource_rep(self, precision):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceIsingZZ, 2, {"precision": precision})
        assert re.ResourceIsingZZ.resource_rep(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources(self, precision):
        """Test that the resources are correct."""
        expected = [
            re.GateCount(re.ResourceCNOT.resource_rep(), 2),
            re.GateCount(re.ResourceRZ.resource_rep(precision=precision)),
        ]
        assert re.ResourceIsingZZ.resource_decomp(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources_pow(self, precision):
        """Test that the pow resources are correct."""
        expected = [re.GateCount(re.ResourceIsingZZ.resource_rep(precision=precision))]
        assert re.ResourceIsingZZ.pow_resource_decomp(pow_z=3, precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources_adjoint(self, precision):
        """Test that the adjoint resources are correct."""
        expected = [re.GateCount(re.ResourceIsingZZ.resource_rep(precision=precision))]
        assert re.ResourceIsingZZ.adjoint_resource_decomp(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources_controlled(self, precision):
        """Test that the controlled resources are correct."""
        expected = [
            re.GateCount(re.ResourceCNOT.resource_rep(), 2),
            re.GateCount(
                re.ResourceControlled.resource_rep(
                    re.ResourceRZ.resource_rep(precision=precision),
                    3,
                    2,
                )
            ),
        ]
        op = re.ResourceControlled(
            re.ResourceIsingZZ(precision=precision), num_ctrl_wires=3, num_ctrl_values=2
        )
        assert op.resource_decomp(**op.resource_params) == expected


class TestPSWAP:
    """Test the PSWAP class."""

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resource_params(self, precision):
        """Test that the resource params are correct."""
        if precision:
            op = re.ResourcePSWAP(precision=precision)
        else:
            op = re.ResourcePSWAP()

        assert op.resource_params == {"precision": precision}

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resource_rep(self, precision):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourcePSWAP, 2, {"precision": precision})
        assert re.ResourcePSWAP.resource_rep(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources(self, precision):
        """Test that the resources are correct."""
        expected = [
            re.GateCount(re.ResourceSWAP.resource_rep()),
            re.GateCount(re.ResourcePhaseShift.resource_rep(precision=precision)),
            re.GateCount(re.ResourceCNOT.resource_rep(), 2),
        ]
        assert re.ResourcePSWAP.resource_decomp(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources_adjoint(self, precision):
        """Test that the adjoint resources are correct."""
        expected = [re.GateCount(re.ResourcePSWAP.resource_rep(precision=precision))]
        assert re.ResourcePSWAP.adjoint_resource_decomp(precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    def test_resources_controlled(self, precision):
        """Test that the controlled resources are correct."""
        expected = [
            re.GateCount(
                re.ResourceControlled.resource_rep(
                    re.ResourceSWAP.resource_rep(),
                    3,
                    2,
                )
            ),
            re.GateCount(re.ResourceCNOT.resource_rep(), 2),
            re.GateCount(
                re.ResourceControlled.resource_rep(
                    re.ResourcePhaseShift.resource_rep(precision=precision),
                    3,
                    2,
                )
            ),
        ]
        op = re.ResourceControlled(
            re.ResourcePSWAP(precision=precision), num_ctrl_wires=3, num_ctrl_values=2
        )
        assert op.resource_decomp(**op.resource_params) == expected
