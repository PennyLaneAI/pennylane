# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the available built-in quantum channels.
"""
import functools
import pytest
import numpy as np
import pennylane as qml
from pennylane.ops import channel
from pennylane.wires import WireError

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

ch_list = [
    channel.AmplitudeDamping,
    channel.GeneralizedAmplitudeDamping,
    channel.PhaseDamping,
    channel.BitFlip,
    channel.PhaseFlip,
    channel.DepolarizingChannel,
    channel.ResetError,
    channel.PauliError,
    channel.ThermalRelaxationError,
]


class TestChannels:
    """Tests for the quantum channels"""

    @pytest.mark.parametrize("ops", ch_list)
    @pytest.mark.parametrize("p", [0, 0.1, 1])
    @pytest.mark.parametrize("tr_args", [[100e-6, 100e-6, 20e-9], [100e-6, 120e-6, 20e-9]])
    def test_kraus_matrices_sum_identity(self, ops, p, tr_args, tol):
        """Test channels are trace-preserving"""
        if ops.__name__ == "GeneralizedAmplitudeDamping":
            op = [ops(p, p, wires=0)]
        elif ops.__name__ == "ResetError":
            op = [ops(p / 2, p / 3, wires=0)]
        elif ops.__name__ == "PauliError":
            op = [ops("X", p, wires=0), ops("XX", p, wires=[0, 1])]
        elif ops.__name__ == "ThermalRelaxationError":
            op = [ops(p, *tr_args, wires=0)]
        else:
            op = [ops(p, wires=0)]
        for operation in op:
            K_list = operation.kraus_matrices
            K_arr = np.array(K_list)
            Kraus_sum = np.einsum("ajk,ajl->kl", K_arr.conj(), K_arr)
            assert np.allclose(Kraus_sum, np.eye(K_list[0].shape[0]), atol=tol, rtol=0)


class TestAmplitudeDamping:
    """Tests for the quantum channel AmplitudeDamping"""

    def test_gamma_zero(self, tol):
        """Test gamma=0 gives correct Kraus matrices"""
        op = channel.AmplitudeDamping
        assert np.allclose(op(0, wires=0).kraus_matrices[0], np.eye(2), atol=tol, rtol=0)
        assert np.allclose(op(0, wires=0).kraus_matrices[1], np.zeros((2, 2)), atol=tol, rtol=0)

    def test_gamma_arbitrary(self, tol):
        """Test gamma=0.1 gives correct Kraus matrices"""
        op = channel.AmplitudeDamping
        expected = [
            np.array([[1.0, 0.0], [0.0, 0.9486833]]),
            np.array([[0.0, 0.31622777], [0.0, 0.0]]),
        ]
        assert np.allclose(op(0.1, wires=0).kraus_matrices, expected, atol=tol, rtol=0)

    def test_gamma_invalid_parameter(self):
        with pytest.raises(ValueError, match="gamma must be between"):
            channel.AmplitudeDamping(1.5, wires=0).kraus_matrices


class TestGeneralizedAmplitudeDamping:
    """Tests for the quantum channel GeneralizedAmplitudeDamping"""

    def test_gamma_p_zero(self, tol):
        """Test p=0, gamma=0 gives correct Kraus matrices"""
        op = channel.GeneralizedAmplitudeDamping
        assert np.allclose(op(0, 0, wires=0).kraus_matrices[0], np.zeros((2, 2)), atol=tol, rtol=0)
        assert np.allclose(op(0, 0, wires=0).kraus_matrices[2], np.eye(2), atol=tol, rtol=0)

    def test_gamma_p_arbitrary(self, tol):
        """Test arbitrary p and gamma values give correct first Kraus matrix"""

        op = channel.GeneralizedAmplitudeDamping
        # check K0 for gamma=0.1, p =0.1
        expected_K0 = np.array([[0.31622777, 0.0], [0.0, 0.3]])
        assert np.allclose(op(0.1, 0.1, wires=0).kraus_matrices[0], expected_K0, atol=tol, rtol=0)

        # check K3 for gamma=0.1, p=0.5
        expected_K3 = np.array([[0.0, 0.0], [0.2236068, 0.0]])
        assert np.allclose(op(0.1, 0.5, wires=0).kraus_matrices[3], expected_K3, atol=tol, rtol=0)

    def test_gamma_invalid_parameter(self):
        with pytest.raises(ValueError, match="gamma must be between"):
            channel.GeneralizedAmplitudeDamping(1.5, 0.0, wires=0).kraus_matrices

    def test_p_invalid_parameter(self):
        with pytest.raises(ValueError, match="p must be between"):
            channel.GeneralizedAmplitudeDamping(0.0, 1.5, wires=0).kraus_matrices


class TestPhaseDamping:
    """Tests for the quantum channel PhaseDamping"""

    def test_gamma_zero(self, tol):
        """Test gamma=0 gives correct Kraus matrices"""
        op = channel.PhaseDamping
        assert np.allclose(op(0, wires=0).kraus_matrices[0], np.eye(2), atol=tol, rtol=0)
        assert np.allclose(op(0, wires=0).kraus_matrices[1], np.zeros((2, 2)), atol=tol, rtol=0)

    def test_gamma_arbitrary(self, tol):
        """Test gamma=0.1 gives correct Kraus matrices"""
        op = channel.PhaseDamping
        expected = [
            np.array([[1.0, 0.0], [0.0, 0.9486833]]),
            np.array([[0.0, 0.0], [0.0, 0.31622777]]),
        ]
        assert np.allclose(op(0.1, wires=0).kraus_matrices, expected, atol=tol, rtol=0)

    def test_gamma_invalid_parameter(self):
        with pytest.raises(ValueError, match="gamma must be between"):
            channel.PhaseDamping(1.5, wires=0).kraus_matrices


class TestBitFlip:
    """Tests for the quantum channel BitFlipChannel"""

    @pytest.mark.parametrize("p", [0, 0.1, 0.5, 1])
    def test_p_arbitrary(self, p, tol):
        """Test that various values of p give correct Kraus matrices"""
        op = channel.BitFlip

        expected_K0 = np.sqrt(1 - p) * np.eye(2)
        assert np.allclose(op(p, wires=0).kraus_matrices[0], expected_K0, atol=tol, rtol=0)

        expected_K1 = np.sqrt(p) * X
        assert np.allclose(op(p, wires=0).kraus_matrices[1], expected_K1, atol=tol, rtol=0)

    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7))
    def test_grad_bitflip(self, angle, tol):
        """Test that analytical gradient is computed correctly for different states. Channel
        grad recipes are independent of channel parameter"""

        dev = qml.device("default.mixed", wires=1)
        prob = 0.5

        @qml.qnode(dev)
        def circuit(p):
            qml.RX(angle, wires=0)
            qml.BitFlip(p, wires=0)
            return qml.expval(qml.PauliZ(0))

        gradient = np.squeeze(qml.grad(circuit)(prob))
        assert gradient == circuit(1) - circuit(0)
        assert np.allclose(gradient, (-2 * np.cos(angle)))

    def test_p_invalid_parameter(self):
        with pytest.raises(ValueError, match="p must be between"):
            channel.BitFlip(1.5, wires=0).kraus_matrices


class TestPhaseFlip:
    """Test that various values of p give correct Kraus matrices"""

    @pytest.mark.parametrize("p", [0, 0.1, 0.5, 1])
    def test_p_arbitrary(self, p, tol):
        """Test p=0.1 gives correct Kraus matrices"""
        op = channel.PhaseFlip

        expected_K0 = np.sqrt(1 - p) * np.eye(2)
        assert np.allclose(op(p, wires=0).kraus_matrices[0], expected_K0, atol=tol, rtol=0)

        expected_K1 = np.sqrt(p) * Z
        assert np.allclose(op(p, wires=0).kraus_matrices[1], expected_K1, atol=tol, rtol=0)

    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7))
    def test_grad_phaseflip(self, angle, tol):
        """Test that analytical gradient is computed correctly for different states. Channel
        grad recipes are independent of channel parameter"""

        dev = qml.device("default.mixed", wires=1)
        prob = 0.5

        @qml.qnode(dev)
        def circuit(p):
            qml.RX(angle, wires=0)
            qml.PhaseFlip(p, wires=0)
            return qml.expval(qml.PauliZ(0))

        gradient = np.squeeze(qml.grad(circuit)(prob))
        assert gradient == circuit(1) - circuit(0)
        assert np.allclose(gradient, 0.0)

    def test_p_invalid_parameter(self):
        with pytest.raises(ValueError, match="p must be between"):
            channel.PhaseFlip(1.5, wires=0).kraus_matrices


class TestDepolarizingChannel:
    """Tests for the quantum channel DepolarizingChannel"""

    def test_p_zero(self, tol):
        """Test p=0 gives correct Kraus matrices"""
        op = channel.DepolarizingChannel
        assert np.allclose(op(0, wires=0).kraus_matrices[0], np.eye(2), atol=tol, rtol=0)
        assert np.allclose(op(0, wires=0).kraus_matrices[1], np.zeros((2, 2)), atol=tol, rtol=0)

    def test_p_arbitrary(self, tol):
        """Test p=0.1 gives correct Kraus matrices"""
        p = 0.1
        op = channel.DepolarizingChannel
        expected = np.sqrt(p / 3) * X
        assert np.allclose(op(0.1, wires=0).kraus_matrices[1], expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7))
    def test_grad_depolarizing(self, angle, tol):
        """Test that analytical gradient is computed correctly for different states. Channel
        grad recipes are independent of channel parameter"""

        dev = qml.device("default.mixed", wires=1)
        prob = 0.5

        @qml.qnode(dev)
        def circuit(p):
            qml.RX(angle, wires=0)
            qml.DepolarizingChannel(p, wires=0)
            return qml.expval(qml.PauliZ(0))

        gradient = np.squeeze(qml.grad(circuit)(prob))
        assert gradient == circuit(1) - circuit(0)
        assert np.allclose(gradient, -(4 / 3) * np.cos(angle))

    def test_p_invalid_parameter(self):
        with pytest.raises(ValueError, match="p must be between"):
            channel.DepolarizingChannel(1.5, wires=0).kraus_matrices


class TestResetError:
    """Tests for the quantum channel ResetError"""

    @pytest.mark.parametrize("p_0,p_1", list(zip([0.5, 0.1, 0.0, 0.0], [0, 0.1, 0.5, 0.0])))
    def test_p0_p1_arbitrary(self, p_0, p_1, tol):
        """Test that various values of p_0 and p_1 give correct Kraus matrices"""
        op = channel.ResetError

        expected_K0 = np.sqrt(1 - p_0 - p_1) * np.eye(2)
        assert np.allclose(op(p_0, p_1, wires=0).kraus_matrices[0], expected_K0, atol=tol, rtol=0)

        expected_K1 = np.sqrt(p_0) * np.array([[1, 0], [0, 0]])
        assert np.allclose(op(p_0, p_1, wires=0).kraus_matrices[1], expected_K1, atol=tol, rtol=0)

        expected_K2 = np.sqrt(p_0) * np.array([[0, 1], [0, 0]])
        assert np.allclose(op(p_0, p_1, wires=0).kraus_matrices[2], expected_K2, atol=tol, rtol=0)

        expected_K3 = np.sqrt(p_1) * np.array([[0, 0], [1, 0]])
        assert np.allclose(op(p_0, p_1, wires=0).kraus_matrices[3], expected_K3, atol=tol, rtol=0)

        expected_K4 = np.sqrt(p_1) * np.array([[0, 0], [0, 1]])
        assert np.allclose(op(p_0, p_1, wires=0).kraus_matrices[4], expected_K4, atol=tol, rtol=0)

    def test_p0_invalid_parameter(self):
        with pytest.raises(ValueError, match="p_0 must be between"):
            channel.ResetError(1.5, 0.0, wires=0).kraus_matrices

    def test_p1_invalid_parameter(self):
        with pytest.raises(ValueError, match="p_1 must be between"):
            channel.ResetError(0.0, 1.5, wires=0).kraus_matrices

    def test_p0_p1_sum_not_normalized(self):
        with pytest.raises(ValueError, match="must be between"):
            channel.ResetError(1.0, 1.0, wires=0).kraus_matrices

    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7))
    def test_grad_reset_error(self, angle, tol):
        """Test that gradient is computed correctly for different states. Channel
        grad recipes are independent of channel parameter"""

        dev = qml.device("default.mixed", wires=1)
        p_0, p_1 = 0.0, 0.5

        @qml.qnode(dev)
        def circuit(p_0, p_1):
            qml.RX(angle, wires=0)
            qml.ResetError(p_0, p_1, wires=0)
            return qml.expval(qml.PauliZ(0))

        gradient = np.squeeze(qml.grad(circuit)(p_0, p_1))
        assert np.allclose(
            gradient,
            np.array(
                [
                    (1 / 0.1) * (circuit(0.1, p_1) - circuit(0.0, p_1)),
                    (1 / 0.1) * (circuit(p_0, 0.1) - circuit(p_0, 0.0)),
                ]
            ),
        )
        assert np.allclose(
            gradient,
            np.array(
                [
                    (2 * np.sin(angle / 2) * np.sin(angle / 2)),
                    (-2 * np.cos(angle / 2) * np.cos(angle / 2)),
                ]
            ),
        )


class TestPauliError:
    """Tests for the quantum channel PauliError"""

    OPERATORS_WRONG_PARAMS = ["XXX", "XXX", "ABC", "XXX"]
    P_WRONG_PARAMS = [0.5, 1.5, 0.5, 0.5]
    WIRES_WRONG_PARAMS = [[0], [0, 1, 2], [0, 1, 2], [1, 1, 2]]
    EXPECTED_ERRORS = [ValueError, ValueError, ValueError, WireError]
    EXPECTED_MESSAGES = [
        "The number of operators must match the number of wires",
        "p must be between \\[0,1\\]",
        "The specified operators need to be either of 'X', 'Y' or 'Z'",
        "Wires must be unique",
    ]

    @pytest.mark.parametrize(
        "operators, p, wires, error, message",
        list(
            zip(
                OPERATORS_WRONG_PARAMS,
                P_WRONG_PARAMS,
                WIRES_WRONG_PARAMS,
                EXPECTED_ERRORS,
                EXPECTED_MESSAGES,
            )
        ),
    )
    def test_wrong_parameters(self, operators, p, wires, error, message):
        """Test wrong parametrizations of PauliError"""
        with pytest.raises(error, match=message):
            Ks = channel.PauliError(operators, p, wires=wires)

    def test_warning_many_qubits(self):
        """Test if warning is thrown when huge matrix"""
        with pytest.warns(UserWarning):
            Ks = channel.PauliError("X" * 512, 0.5, wires=list(range(512)))

    def test_p_zero(self, tol):
        """Test resulting Kraus matrices for p=0"""
        expected_Ks = [np.eye(2 ** 5), np.zeros((2 ** 5, 2 ** 5))]
        c = channel.PauliError("XXXXX", 0, wires=[0, 1, 2, 3, 4])

        assert np.allclose(c.kraus_matrices, expected_Ks, atol=tol, rtol=0)

    def test_p_one(self, tol):
        """Test resulting Kraus matrices for p=1"""
        expected_Ks = [np.zeros((2 ** 5, 2 ** 5)), np.flip(np.eye(2 ** 5), axis=1)]
        c = channel.PauliError("XXXXX", 1, wires=[0, 1, 2, 3, 4])

        assert np.allclose(c.kraus_matrices, expected_Ks, atol=tol, rtol=0)

    OPERATORS = ["X", "XY", "ZX"]
    WIRES = [[1], [0, 1], [3, 1]]
    EXPECTED_KS = [
        [
            np.sqrt(0.5) * np.eye(2),
            np.array(
                [
                    [0.0, 0.70710678],
                    [0.70710678, 0.0],
                ]
            ),
        ],
        [
            np.sqrt(0.5) * np.eye(4),
            np.array(
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 0.70710678j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.70710678j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 - 0.70710678j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.70710678j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ]
            ),
        ],
        [
            np.sqrt(0.5) * np.eye(4),
            np.array(
                [
                    [0.0, 0.70710678, 0.0, 0.0],
                    [0.70710678, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -0.0, -0.70710678],
                    [0.0, 0.0, -0.70710678, -0.0],
                ]
            ),
        ],
    ]

    @pytest.mark.parametrize(
        "operators, wires, expected_Ks", list(zip(OPERATORS, WIRES, EXPECTED_KS))
    )
    def test_kraus_matrix(self, tol, operators, wires, expected_Ks):
        """Test sevaral resulting kraus matrices for sevaral configurations"""
        c = channel.PauliError(operators, 0.5, wires=wires)

        assert np.allclose(c.kraus_matrices, expected_Ks, atol=tol, rtol=0)


class TestQubitChannel:
    """Tests for the quantum channel QubitChannel"""

    def test_input_correctly_handled(self, tol):
        """Test that Kraus matrices are correctly processed"""
        K_list1 = [
            np.array([[1.0, 0.0], [0.0, 0.9486833]]),
            np.array([[0.0, 0.31622777], [0.0, 0.0]]),
        ]
        out = channel.QubitChannel(K_list1, wires=0).kraus_matrices

        # verify equivalent to input matrices
        assert np.allclose(out, K_list1, atol=tol, rtol=0)

    def test_kraus_matrices_valid(self):
        """Tests that the given Kraus matrices are valid"""

        # check all Kraus matrices are square matrices
        K_list1 = [np.zeros((2, 2)), np.zeros((2, 3))]
        with pytest.raises(
            ValueError, match="Only channels with the same input and output Hilbert space"
        ):
            channel.QubitChannel(K_list1, wires=0)

        # check all Kraus matrices have the same shape
        K_list2 = [np.eye(2), np.eye(4)]
        with pytest.raises(ValueError, match="All Kraus matrices must have the same shape."):
            channel.QubitChannel(K_list2, wires=0)

        # check the dimension of all Kraus matrices are valid
        K_list3 = [np.array([np.eye(2), np.eye(2)]), np.array([np.eye(2), np.eye(2)])]
        with pytest.raises(ValueError, match="Dimension of all Kraus matrices must be "):
            channel.QubitChannel(K_list3, wires=0)

    def test_channel_trace_preserving(self):
        """Tests that the channel represents a trace-preserving map"""

        # real Kraus matrices
        K_list1 = [
            np.array([[1.0, 0.0], [0.0, 0.9486833]]),
            np.array([[0.0, 0.31622777], [0.0, 0.0]]),
        ]
        with pytest.raises(ValueError, match="Only trace preserving channels can be applied."):
            channel.QubitChannel(K_list1 * 2, wires=0)

        # complex Kraus matrices
        p = 0.1
        K_list2 = [np.sqrt(p) * Y, np.sqrt(1 - p) * np.eye(2)]
        with pytest.raises(ValueError, match="Only trace preserving channels can be applied."):
            channel.QubitChannel(K_list2 * 2, wires=0)


class TestThermalRelaxationError:
    """Tests for the quantum channel ThermalRelaxationError"""

    @pytest.mark.parametrize(
        "pe,t1,t2,tg",
        list(
            zip(
                [0.2, 0.4, 0.6, 0.0],
                [100e-6, 50e-6, 80e-6, np.inf],
                [80e-6, 40e-6, 80e-6, 50e-6],
                [20e-9, 40e-9, 40e-6, 40e-9],
            )
        ),
    )
    def test_t2_le_t1_arbitrary(self, pe, t1, t2, tg, tol):
        """Test that various values of pe, t1, t2, and tg  for t2 <= t1 give correct Kraus matrices"""

        op = channel.ThermalRelaxationError

        eT1 = np.exp(-tg / t1)
        p_reset = 1 - eT1
        eT2 = np.exp(-tg / t2)
        pz = (1 - p_reset) * (1 - eT2 / eT1) / 2
        pr0 = (1 - pe) * p_reset
        pr1 = pe * p_reset
        pid = 1 - pz - pr0 - pr1

        expected_K0 = np.sqrt(pid) * np.eye(2)
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices[0], expected_K0, atol=tol, rtol=0
        )

        expected_K1 = np.sqrt(pz) * np.array([[1, 0], [0, -1]])
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices[1], expected_K1, atol=tol, rtol=0
        )

        expected_K2 = np.sqrt(pr0) * np.array([[1, 0], [0, 0]])
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices[2], expected_K2, atol=tol, rtol=0
        )

        expected_K3 = np.sqrt(pr0) * np.array([[0, 1], [0, 0]])
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices[3], expected_K3, atol=tol, rtol=0
        )

        expected_K4 = np.sqrt(pr1) * np.array([[0, 0], [1, 0]])
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices[4], expected_K4, atol=tol, rtol=0
        )

        expected_K5 = np.sqrt(pr1) * np.array([[0, 0], [0, 1]])
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices[5], expected_K5, atol=tol, rtol=0
        )

    @pytest.mark.parametrize(
        "pe,t1,t2,tg",
        list(
            zip(
                [0.8, 0.5, 0.0],
                [100e-6, 50e-6, 80e-6],
                [120e-6, 100e-6, 90e-6],
                [20e-9, 40e-9, 90e-6],
            )
        ),
    )
    def test_t2_g_t1_arbitrary(self, pe, t1, t2, tg, tol):
        """Test that various values of pe, t1, t2, and tg  for t2 > t1 give correct Kraus matrices"""

        op = channel.ThermalRelaxationError

        if t1 == np.inf:
            eT1 = 0
            p_reset = 0
        else:
            eT1 = np.exp(-tg / t1)
            p_reset = 1 - eT1
        if t2 == np.inf:
            eT2 = 1
        else:
            eT2 = np.exp(-tg / t2)

        e0 = p_reset * pe
        v0 = np.array([[0], [1], [0], [0]])
        e1 = -p_reset * pe + p_reset
        v1 = np.array([[0], [0], [1], [0]])
        common_term = np.sqrt(
            4 * eT2 ** 2 + 4 * p_reset ** 2 * pe ** 2 - 4 * p_reset ** 2 * pe + p_reset ** 2
        )
        e2 = 1 - p_reset / 2 - common_term / 2
        term2 = 2 * eT2 / (2 * p_reset * pe - p_reset - common_term)
        v2 = np.array([[term2], [0], [0], [1]]) / np.sqrt(term2 ** 2 + 1 ** 2)
        term3 = 2 * eT2 / (2 * p_reset * pe - p_reset + common_term)
        e3 = 1 - p_reset / 2 + common_term / 2
        v3 = np.array([[term3], [0], [0], [1]]) / np.sqrt(term3 ** 2 + 1 ** 2)

        expected_K0 = np.sqrt(e0) * v0.reshape(2, 2, order="F")
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices[0], expected_K0, atol=tol, rtol=0
        )

        expected_K1 = np.sqrt(e1) * v1.reshape(2, 2, order="F")
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices[1], expected_K1, atol=tol, rtol=0
        )

        expected_K2 = np.sqrt(e2) * v2.reshape(2, 2, order="F")
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices[2], expected_K2, atol=tol, rtol=0
        )

        expected_K3 = np.sqrt(e3) * v3.reshape(2, 2, order="F")
        assert np.allclose(
            op(pe, t1, t2, tg, wires=0).kraus_matrices[3], expected_K3, atol=tol, rtol=0
        )

    def test_pe_invalid_parameter(self):
        with pytest.raises(ValueError, match="pe must be between"):
            channel.ThermalRelaxationError(1.5, 100e-6, 100e-6, 20e-9, wires=0).kraus_matrices

    def test_T2_g_2T1_invalid_parameter(self):
        with pytest.raises(ValueError, match="Invalid T_2 relaxation time parameter"):
            channel.ThermalRelaxationError(0.3, 100e-6, np.inf, 20e-9, wires=0).kraus_matrices

    def test_T1_le_0_invalid_parameter(self):
        with pytest.raises(ValueError, match="Invalid T_1 relaxation time parameter"):
            channel.ThermalRelaxationError(0.3, -50e-6, np.inf, 20e-9, wires=0).kraus_matrices

    def test_T2_le_0_invalid_parameter(self):
        with pytest.raises(ValueError, match="Invalid T_2 relaxation time parameter"):
            channel.ThermalRelaxationError(0.3, 100e-6, 0, 20e-9, wires=0).kraus_matrices

    def test_tg_le_0_invalid_parameter(self):
        with pytest.raises(ValueError, match="Invalid gate_time"):
            channel.ThermalRelaxationError(0.3, 100e-6, 100e-6, -20e-9, wires=0).kraus_matrices

    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7))
    def test_grad_thermal_relaxation_error(self, angle, tol):
        """Test that gradient is computed correctly for different states. Channel
        grad recipes are independent of channel parameter"""

        dev = qml.device("default.mixed", wires=1)
        pe = 0.0

        @qml.qnode(dev)
        def circuit(pe):
            qml.RX(angle, wires=0)
            qml.ThermalRelaxationError(pe, 120e-6, 100e-6, 20e-9, wires=0)
            return qml.expval(qml.PauliZ(0))

        gradient = np.squeeze(qml.grad(circuit)(pe))
        assert np.allclose(
            gradient,
            np.array(
                [
                    (1 / 0.1) * (circuit(0.1) - circuit(0.0)),
                ]
            ),
        )
