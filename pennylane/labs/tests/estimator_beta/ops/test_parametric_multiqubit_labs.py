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

r"""Unit tests for the parametric multi-qubit operators."""

import pytest

import pennylane.labs.estimator_beta as qre


# pylint: disable = no-self-use
class TestPauliRot:
    """Test the Resource PauliRot class."""

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

    @pytest.mark.parametrize(
        "pauli_string, precision, expected",
        (
            (
                "X",
                None,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RX.resource_rep(precision=None), 1, 0)
                    )
                ],
            ),
            (
                "Y",
                None,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RY.resource_rep(precision=None), 1, 0)
                    )
                ],
            ),
            (
                "Z",
                None,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RZ.resource_rep(precision=None), 1, 0)
                    )
                ],
            ),
            (
                "XX",
                None,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RX.resource_rep(precision=None), 1, 0)
                    ),
                    qre.GateCount(qre.CNOT.resource_rep(), 2),
                ],
            ),
            (
                "YY",
                None,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RY.resource_rep(precision=None), 1, 0)
                    ),
                    qre.GateCount(qre.CY.resource_rep(), 2),
                ],
            ),
            (
                "X",
                1e-3,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RX.resource_rep(precision=1e-3), 1, 0)
                    )
                ],
            ),
            (
                "Y",
                1e-3,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RY.resource_rep(precision=1e-3), 1, 0)
                    )
                ],
            ),
            (
                "Z",
                1e-3,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RZ.resource_rep(precision=1e-3), 1, 0)
                    )
                ],
            ),
            (
                "XX",
                1e-3,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RX.resource_rep(precision=1e-3), 1, 0)
                    ),
                    qre.GateCount(qre.CNOT.resource_rep(), 2),
                ],
            ),
            (
                "YY",
                1e-3,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RY.resource_rep(precision=1e-3), 1, 0)
                    ),
                    qre.GateCount(qre.CY.resource_rep(), 2),
                ],
            ),
        ),
    )
    def test_controlled_resource_decomp_special_cases(self, pauli_string, expected, precision):
        """Test that the controlled resources method produces the correct result for all special cases."""
        assert (
            qre.PauliRot.controlled_resource_decomp(
                target_resource_params={"pauli_string": pauli_string, "precision": precision},
                num_ctrl_wires=1,
                num_zero_ctrl=0,
            )
            == expected
        )
