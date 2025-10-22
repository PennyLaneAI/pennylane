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

import pennylane.estimator as qre

# pylint: disable=use-implicit-booleaness-not-comparison,no-self-use,too-many-arguments


class TestMultiRZ:
    """Test the Resource MultiRZ class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 4 wires, got 2"):
            qre.MultiRZ(num_wires=4, wires=[0, 1])

    def test_init_no_num_wires(self):
        """Test that we can instantiate the operator without providing num_wires"""
        op = qre.MultiRZ(wires=range(3))
        assert op.resource_params == {"num_wires": 3, "precision": None}

    def test_init_raises_error(self):
        """Test that an error is raised when wires and num_wires are both not provided"""
        with pytest.raises(ValueError, match="Must provide atleast one of"):
            qre.MultiRZ()

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("num_wires", range(1, 5))
    def test_resource_params(self, num_wires, precision):
        """Test that the resource params are correct."""
        if precision:
            op = qre.MultiRZ(num_wires, precision=precision)
        else:
            op = qre.MultiRZ(num_wires)

        assert op.resource_params == {"num_wires": num_wires, "precision": precision}

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("num_wires", range(1, 5))
    def test_resource_rep(self, num_wires, precision):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.MultiRZ, num_wires, {"num_wires": num_wires, "precision": precision}
        )
        assert qre.MultiRZ.resource_rep(num_wires, precision=precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("num_wires", range(1, 5))
    def test_resources(self, num_wires, precision):
        """Test that the resources are correct."""
        expected = [
            qre.GateCount(qre.CNOT.resource_rep(), 2 * (num_wires - 1)),
            qre.GateCount(qre.RZ.resource_rep(precision=precision)),
        ]
        assert qre.MultiRZ.resource_decomp(num_wires, precision) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("num_wires", range(1, 5))
    def test_resources_from_rep(self, num_wires, precision):
        """Test that the resources can be computed from the compressed representation and params."""
        op = qre.MultiRZ(num_wires, precision)
        expected = [
            qre.GateCount(qre.CNOT.resource_rep(), 2 * (num_wires - 1)),
            qre.GateCount(qre.RZ.resource_rep(precision=precision)),
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
            qre.GateCount(qre.MultiRZ.resource_rep(num_wires=num_wires, precision=precision))
        ]
        assert (
            qre.MultiRZ.adjoint_resource_decomp({"num_wires": num_wires, "precision": precision})
            == expected
        )

    ctrl_data = (
        (
            1,
            0,
            [
                qre.GateCount(qre.resource_rep(qre.CNOT()), 4),
                qre.GateCount(qre.Controlled.resource_rep(qre.RZ.resource_rep(1e-3), 1, 0)),
            ],
        ),
        (
            1,
            1,
            [
                qre.GateCount(qre.resource_rep(qre.CNOT()), 4),
                qre.GateCount(qre.Controlled.resource_rep(qre.RZ.resource_rep(1e-3), 1, 1)),
            ],
        ),
        (
            2,
            0,
            [
                qre.GateCount(qre.resource_rep(qre.CNOT()), 4),
                qre.GateCount(qre.Controlled.resource_rep(qre.RZ.resource_rep(1e-3), 2, 0)),
            ],
        ),
        (
            3,
            2,
            [
                qre.GateCount(qre.resource_rep(qre.CNOT()), 4),
                qre.GateCount(qre.Controlled.resource_rep(qre.RZ.resource_rep(1e-3), 3, 2)),
            ],
        ),
    )

    @pytest.mark.parametrize("num_ctrl_wires, num_zero_ctrl, expected_res", ctrl_data)
    def test_resource_controlled(self, num_ctrl_wires, num_zero_ctrl, expected_res):
        """Test that the controlled resources are as expected"""

        op = qre.MultiRZ(num_wires=3, precision=1e-3)
        op2 = qre.Controlled(
            op,
            num_ctrl_wires,
            num_zero_ctrl,
        )

        assert (
            op.controlled_resource_decomp(num_ctrl_wires, num_zero_ctrl, op.resource_params)
            == expected_res
        )
        assert op2.resource_decomp(**op2.resource_params) == expected_res

    @pytest.mark.parametrize("z", range(1, 5))
    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("num_wires", range(1, 5))
    def test_pow_decomp(self, z, num_wires, precision):
        """Test that the pow decomposition is correct."""
        op = qre.MultiRZ(num_wires, precision=precision)
        expected_res = [qre.GateCount(qre.MultiRZ.resource_rep(num_wires, precision))]
        assert op.pow_resource_decomp(z, op.resource_params) == expected_res


class TestPauliRot:
    """Test the Resource PauliRot class."""

    pauli_words = ("I", "XYZ", "XXX", "XIYIZIX", "III")

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 3 wires, got 2"):
            qre.PauliRot(pauli_string="XYZ", precision=1e-3, wires=[0, 1])

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("pauli_string", pauli_words)
    def test_resource_params(self, pauli_string, precision):
        """Test that the resource params are correct."""
        op = qre.PauliRot(pauli_string=pauli_string, precision=precision)
        assert op.resource_params == {"pauli_string": pauli_string, "precision": precision}

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("pauli_string", pauli_words)
    def test_resource_rep(self, pauli_string, precision):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.PauliRot,
            len(pauli_string),
            {"pauli_string": pauli_string, "precision": precision},
        )
        assert qre.PauliRot.resource_rep(pauli_string, precision) == expected

    expected_h_count = (0, 4, 6, 6, 0)
    expected_s_count = (0, 1, 0, 1, 0)
    params = zip(pauli_words, expected_h_count, expected_s_count)

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("pauli_string, expected_h_count, expected_s_count", params)
    def test_resources(self, pauli_string, expected_h_count, expected_s_count, precision):
        """Test that the resources are correct."""
        active_wires = len(pauli_string.replace("I", ""))

        if set(pauli_string) == {"I"}:
            expected = [qre.GateCount(qre.GlobalPhase.resource_rep())]
        else:
            expected = []

            if expected_h_count:
                expected.append(qre.GateCount(qre.Hadamard.resource_rep(), expected_h_count))

            if expected_s_count:
                expected.append(qre.GateCount(qre.S.resource_rep(), expected_s_count))
                expected.append(
                    qre.GateCount(
                        qre.Adjoint.resource_rep(qre.S.resource_rep()),
                        expected_s_count,
                    )
                )

            expected.append(qre.GateCount(qre.RZ.resource_rep(precision=precision)))
            expected.append(qre.GateCount(qre.CNOT.resource_rep(), 2 * (active_wires - 1)))

        assert qre.PauliRot.resource_decomp(pauli_string, precision=precision) == expected

    def test_resources_empty_pauli_string(self):
        """Test that the resources method produces the correct result for an empty pauli string."""
        expected = [qre.GateCount(qre.GlobalPhase.resource_rep())]
        assert qre.PauliRot.resource_decomp(pauli_string="") == expected

    @pytest.mark.parametrize("pauli_string, expected_h_count, expected_s_count", params)
    def test_resources_from_rep(self, pauli_string, expected_h_count, expected_s_count):
        """Test that the resources can be computed from the compressed representation and params."""
        op = qre.PauliRot(0.5, pauli_string, wires=range(len(pauli_string)))
        active_wires = len(pauli_string.replace("I", ""))

        if set(pauli_string) == {"I"}:
            expected = [qre.GateCount(qre.GlobalPhase.resource_rep())]
        else:
            expected = [
                qre.GateCount(qre.RZ.resource_rep()),
                qre.GateCount(qre.CNOT.resource_rep(), 2 * (active_wires - 1)),
            ]

            if expected_h_count:
                expected.append(qre.GateCount(qre.Hadamard.resource_rep(), expected_h_count))

            if expected_s_count:
                expected.append(qre.GateCount(qre.S.resource_rep(), expected_s_count))
                expected.append(
                    qre.GateCount(
                        qre.Adjoint.resource_rep(qre.S.resource_rep()),
                        expected_s_count,
                    )
                )

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    @pytest.mark.parametrize(
        "pauli_string, expected", (("X", qre.RX), ("Y", qre.RY), ("Z", qre.RZ))
    )
    def test_resource_decomp_single_pauli_string(self, pauli_string, expected):
        """Test that the resources method produces the correct result for a single pauli string."""
        expected = [qre.GateCount(expected.resource_rep(precision=1e-3))]
        assert qre.PauliRot.resource_decomp(pauli_string=pauli_string, precision=1e-3) == expected

    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("pauli_word", pauli_words)
    def test_adjoint_decomp(self, pauli_word, precision):
        """Test that the adjoint decomposition is correct."""
        expected = [
            qre.GateCount(qre.PauliRot.resource_rep(pauli_string=pauli_word, precision=precision))
        ]
        assert (
            qre.PauliRot.adjoint_resource_decomp(
                target_resource_params={"pauli_string": pauli_word, "precision": precision}
            )
            == expected
        )

    ctrl_data = (
        (
            "XXX",
            1,
            0,
            [
                qre.GateCount(qre.Hadamard.resource_rep(), 6),
                qre.GateCount(
                    qre.Controlled.resource_rep(qre.RZ.resource_rep(precision=1e-5), 1, 0),
                    1,
                ),
                qre.GateCount(qre.CNOT.resource_rep(), 4),
            ],
        ),
        (
            "XXX",
            1,
            1,
            [
                qre.GateCount(qre.Hadamard.resource_rep(), 6),
                qre.GateCount(
                    qre.Controlled.resource_rep(qre.RZ.resource_rep(precision=1e-5), 1, 1),
                    1,
                ),
                qre.GateCount(qre.CNOT.resource_rep(), 4),
            ],
        ),
        (
            "XXX",
            2,
            0,
            [
                qre.GateCount(qre.Hadamard.resource_rep(), 6),
                qre.GateCount(
                    qre.Controlled.resource_rep(qre.RZ.resource_rep(precision=1e-5), 2, 0),
                    1,
                ),
                qre.GateCount(qre.CNOT.resource_rep(), 4),
            ],
        ),
        (
            "XIYIZIX",
            1,
            0,
            [
                qre.GateCount(qre.Hadamard.resource_rep(), 6),
                qre.GateCount(qre.S.resource_rep(), 1),
                qre.GateCount(qre.Adjoint.resource_rep(qre.S.resource_rep()), 1),
                qre.GateCount(
                    qre.Controlled.resource_rep(qre.RZ.resource_rep(precision=1e-5), 1, 0),
                    1,
                ),
                qre.GateCount(qre.CNOT.resource_rep(), 6),
            ],
        ),
        (
            "XIYIZIX",
            1,
            1,
            [
                qre.GateCount(qre.Hadamard.resource_rep(), 6),
                qre.GateCount(qre.S.resource_rep(), 1),
                qre.GateCount(qre.Adjoint.resource_rep(qre.S.resource_rep()), 1),
                qre.GateCount(
                    qre.Controlled.resource_rep(qre.RZ.resource_rep(precision=1e-5), 1, 1),
                    1,
                ),
                qre.GateCount(qre.CNOT.resource_rep(), 6),
            ],
        ),
        (
            "XIYIZIX",
            2,
            0,
            [
                qre.GateCount(qre.Hadamard.resource_rep(), 6),
                qre.GateCount(qre.S.resource_rep(), 1),
                qre.GateCount(qre.Adjoint.resource_rep(qre.S.resource_rep()), 1),
                qre.GateCount(
                    qre.Controlled.resource_rep(qre.RZ.resource_rep(precision=1e-5), 2, 0),
                    1,
                ),
                qre.GateCount(qre.CNOT.resource_rep(), 6),
            ],
        ),
        (
            "III",
            1,
            0,
            [qre.GateCount(qre.Controlled.resource_rep(qre.GlobalPhase.resource_rep(), 1, 0))],
        ),
        (
            "X",
            1,
            1,
            [qre.GateCount(qre.Controlled.resource_rep(qre.RX.resource_rep(precision=1e-5), 1, 1))],
        ),
        (
            "Y",
            2,
            0,
            [qre.GateCount(qre.Controlled.resource_rep(qre.RY.resource_rep(precision=1e-5), 2, 0))],
        ),
        (
            "Z",
            2,
            1,
            [qre.GateCount(qre.Controlled.resource_rep(qre.RZ.resource_rep(precision=1e-5), 2, 1))],
        ),
    )

    @pytest.mark.parametrize("pauli_word, num_ctrl_wires, num_zero_ctrl, expected_res", ctrl_data)
    def test_resource_controlled(self, num_ctrl_wires, num_zero_ctrl, pauli_word, expected_res):
        """Test that the controlled resources are as expected"""

        op = qre.PauliRot(pauli_word, precision=1e-5)
        op2 = qre.Controlled(op, num_ctrl_wires, num_zero_ctrl)

        assert (
            op.controlled_resource_decomp(num_ctrl_wires, num_zero_ctrl, op.resource_params)
            == expected_res
        )
        assert op2.resource_decomp(**op2.resource_params) == expected_res

    @pytest.mark.parametrize("z", range(1, 5))
    @pytest.mark.parametrize("precision", (None, 1e-3))
    @pytest.mark.parametrize("pauli_word", pauli_words)
    def test_pow_decomp(self, z, pauli_word, precision):
        """Test that the pow decomposition is correct."""
        op = qre.PauliRot(pauli_string=pauli_word, precision=precision)
        expected_res = [
            qre.GateCount(qre.PauliRot.resource_rep(pauli_string=pauli_word, precision=precision))
        ]
        assert op.pow_resource_decomp(z, op.resource_params) == expected_res
