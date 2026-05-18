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

from collections import defaultdict

import pytest

import pennylane.labs.estimator_beta as qre
from pennylane.estimator.resource_operator import resource_rep


# pylint: disable = no-self-use, too-many-arguments
class TestPauliRot:
    """Test the alternate controlled decomposition for PauliRot class."""

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

        assert (
            qre.paulirot_controlled_resource_decomp(
                num_ctrl_wires=num_ctrl_wires,
                num_zero_ctrl=num_zero_ctrl,
                target_resource_params=op.resource_params,
            )
            == expected_res
        )

    @pytest.mark.parametrize(
        "pauli_string, num_ctrl_wires, num_zero_ctrl, precision, expected",
        (
            (
                "X",
                1,
                0,
                None,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RX.resource_rep(precision=None), 1, 0)
                    )
                ],
            ),
            (
                "Y",
                2,
                2,
                None,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RY.resource_rep(precision=None), 2, 2)
                    )
                ],
            ),
            (
                "Z",
                1,
                0,
                None,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RZ.resource_rep(precision=None), 1, 0)
                    )
                ],
            ),
            (
                "XX",
                3,
                2,
                None,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RX.resource_rep(precision=None), 3, 2)
                    ),
                    qre.GateCount(qre.CNOT.resource_rep(), 2),
                ],
            ),
            (
                "YY",
                1,
                1,
                None,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RY.resource_rep(precision=None), 1, 1)
                    ),
                    qre.GateCount(qre.CY.resource_rep(), 2),
                ],
            ),
            (
                "X",
                5,
                3,
                1e-3,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RX.resource_rep(precision=1e-3), 5, 3)
                    )
                ],
            ),
            (
                "Y",
                1,
                0,
                1e-3,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RY.resource_rep(precision=1e-3), 1, 0)
                    )
                ],
            ),
            (
                "Z",
                4,
                1,
                1e-3,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RZ.resource_rep(precision=1e-3), 4, 1)
                    )
                ],
            ),
            (
                "XX",
                2,
                1,
                1e-3,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RX.resource_rep(precision=1e-3), 2, 1)
                    ),
                    qre.GateCount(qre.CNOT.resource_rep(), 2),
                ],
            ),
            (
                "YY",
                3,
                2,
                1e-3,
                [
                    qre.GateCount(
                        qre.Controlled.resource_rep(qre.RY.resource_rep(precision=1e-3), 3, 2)
                    ),
                    qre.GateCount(qre.CY.resource_rep(), 2),
                ],
            ),
        ),
    )
    def test_controlled_resource_decomp_special_cases(
        self, num_ctrl_wires, num_zero_ctrl, pauli_string, expected, precision
    ):
        """Test that the controlled resources method produces the correct result for all special cases."""
        op = qre.PauliRot(pauli_string, precision=precision)
        assert (
            qre.paulirot_controlled_resource_decomp(
                num_ctrl_wires=num_ctrl_wires,
                num_zero_ctrl=num_zero_ctrl,
                target_resource_params=op.resource_params,
            )
            == expected
        )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, pauli_string, precision, expected",
        (
            (
                1,
                0,
                "X",
                None,
                qre.Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=2,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(qre.CNOT): 2,
                            resource_rep(qre.T): 88,
                            resource_rep(qre.Hadamard): 2,
                        },
                    ),
                ),
            ),
            (
                2,
                1,
                "XX",
                1e-3,
                qre.Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=4,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(qre.Toffoli, {"elbow": None}): 2,
                            resource_rep(qre.CNOT): 2,
                            resource_rep(qre.T): 42,
                            resource_rep(qre.X): 4,
                            resource_rep(qre.Hadamard): 2,
                        },
                    ),
                ),
            ),
            (
                5,
                3,
                "YY",
                1e-3,
                qre.Resources(
                    zeroed_wires=1,
                    any_state_wires=0,
                    algo_wires=7,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(qre.Toffoli, {"elbow": None}): 14,
                            resource_rep(qre.CNOT): 2,
                            resource_rep(qre.T): 42,
                            resource_rep(qre.Z): 2,
                            resource_rep(qre.S): 4,
                            resource_rep(qre.X): 12,
                        },
                    ),
                ),
            ),
        ),
    )
    def test_controlled_decomp_estimate(
        self, num_ctrl_wires, num_zero_ctrl, pauli_string, expected, precision
    ):
        """Test that the controlled resources method produces the correct result when estimate is used."""
        op = qre.Controlled(
            qre.PauliRot(pauli_string, precision=precision),
            num_ctrl_wires=num_ctrl_wires,
            num_zero_ctrl=num_zero_ctrl,
        )
        assert qre.estimate(op) == expected
