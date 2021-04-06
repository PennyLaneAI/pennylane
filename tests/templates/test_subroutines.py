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
Unit tests for the :mod:`pennylane.template.subroutines` module.
Integration tests should be placed into ``test_templates.py``.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
from scipy.stats import unitary_group
import pennylane as qml
from pennylane import numpy as np
from pennylane.wires import Wires

from pennylane.templates.subroutines import (
    Interferometer,
    ArbitraryUnitary,
    SingleExcitationUnitary,
    DoubleExcitationUnitary,
    UCCSD,
    ApproxTimeEvolution,
    Permute,
    QuantumPhaseEstimation,
)

from pennylane.templates.subroutines.arbitrary_unitary import (
    _all_pauli_words_but_identity,
    _tuple_to_word,
    _n_k_gray_code,
)


# fmt: off
PAULI_WORD_TEST_DATA = [
    (1, ["X", "Y", "Z"]),
    (
        2,
        ["XI", "YI", "ZI", "ZX", "IX", "XX", "YX", "YY", "ZY", "IY", "XY", "XZ", "YZ", "ZZ", "IZ"],
    ),
    (
        3,
        [
            "XII", "YII", "ZII", "ZXI", "IXI", "XXI", "YXI", "YYI", "ZYI", "IYI", "XYI", "XZI", "YZI",
            "ZZI", "IZI", "IZX", "XZX", "YZX", "ZZX", "ZIX", "IIX", "XIX", "YIX", "YXX", "ZXX", "IXX",
            "XXX", "XYX", "YYX", "ZYX", "IYX", "IYY", "XYY", "YYY", "ZYY", "ZZY", "IZY", "XZY", "YZY",
            "YIY", "ZIY", "IIY", "XIY", "XXY", "YXY", "ZXY", "IXY", "IXZ", "XXZ", "YXZ", "ZXZ", "ZYZ",
            "IYZ", "XYZ", "YYZ", "YZZ", "ZZZ", "IZZ", "XZZ", "XIZ", "YIZ", "ZIZ", "IIZ",
        ]
    ),
]

GRAY_CODE_TEST_DATA = [
    (2, 2, [[0, 0], [1, 0], [1, 1], [0, 1]]),
    (2, 3, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1]]),
    (4, 2, [
        [0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [0, 1], [1, 1], [2, 1],
        [2, 2], [3, 2], [0, 2], [1, 2], [1, 3], [2, 3], [3, 3], [0, 3]
    ]),
    (3, 3, [
        [0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [1, 1, 0], [1, 2, 0], [2, 2, 0], [0, 2, 0],
        [0, 2, 1], [1, 2, 1], [2, 2, 1], [2, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [2, 1, 1], [0, 1, 1],
        [0, 1, 2], [1, 1, 2], [2, 1, 2], [2, 2, 2], [0, 2, 2], [1, 2, 2], [1, 0, 2], [2, 0, 2], [0, 0, 2]
    ]),
]


# fmt: on


class TestHelperFunctions:
    """Test the helper functions used in the layers."""

    @pytest.mark.parametrize("n,k,expected_code", GRAY_CODE_TEST_DATA)
    def test_n_k_gray_code(self, n, k, expected_code):
        """Test that _n_k_gray_code produces the Gray code correctly."""
        for expected_word, word in zip(expected_code, _n_k_gray_code(n, k)):
            assert expected_word == word

    @pytest.mark.parametrize("num_wires,expected_pauli_words", PAULI_WORD_TEST_DATA)
    def test_all_pauli_words_but_identity(self, num_wires, expected_pauli_words):
        """Test that the correct Pauli words are returned."""
        for expected_pauli_word, pauli_word in zip(
            expected_pauli_words, _all_pauli_words_but_identity(num_wires)
        ):
            assert expected_pauli_word == pauli_word

    @pytest.mark.parametrize(
        "tuple,expected_word",
        [
            ((0,), "I"),
            ((1,), "X"),
            ((2,), "Y"),
            ((3,), "Z"),
            ((0, 0, 0), "III"),
            ((1, 2, 3), "XYZ"),
            ((1, 2, 3, 0, 0, 3, 2, 1), "XYZIIZYX"),
        ],
    )
    def test_tuple_to_word(self, tuple, expected_word):
        assert _tuple_to_word(tuple) == expected_word


class TestInterferometer:
    """Tests for the Interferometer from the pennylane.template.layers module."""

    def test_invalid_mesh_exception(self):
        """Test that Interferometer() raises correct exception when mesh not recognized."""
        dev = qml.device("default.gaussian", wires=2)
        varphi = [0.42342, 0.234]

        @qml.qnode(dev)
        def circuit(varphi, mesh=None):
            Interferometer(theta=[0.21], phi=[0.53], varphi=varphi, mesh=mesh, wires=[0, 1])
            return qml.expval(qml.NumberOperator(0))

        with pytest.raises(ValueError, match="did not recognize mesh"):
            circuit(varphi, mesh="a")

    @pytest.mark.parametrize("mesh", ["rectangular", "triangular"])
    def test_invalid_beamsplitter_exception(self, mesh):
        """Test that Interferometer() raises correct exception when beamsplitter not recognized."""
        dev = qml.device("default.gaussian", wires=2)
        varphi = [0.42342, 0.234]

        @qml.qnode(dev)
        def circuit(varphi, bs=None):
            Interferometer(theta=[0.21], phi=[0.53], varphi=varphi, beamsplitter=bs, mesh=mesh, wires=[0, 1])
            return qml.expval(qml.NumberOperator(0))

        with pytest.raises(ValueError, match="did not recognize beamsplitter"):
            circuit(varphi, bs="a")

    def test_clements_beamsplitter_convention(self, tol):
        """test the beamsplitter convention"""
        N = 2
        wires = range(N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342, 0.1121]

        with qml.tape.OperationRecorder() as rec_rect:
            Interferometer(
                theta, phi, varphi, mesh="rectangular", beamsplitter="clements", wires=wires
            )

        with qml.tape.OperationRecorder() as rec_tria:
            Interferometer(
                theta, phi, varphi, mesh="triangular", beamsplitter="clements", wires=wires
            )

        for rec in [rec_rect, rec_tria]:
            assert len(rec.queue) == 4

            assert isinstance(rec.queue[0], qml.Rotation)
            assert rec.queue[0].parameters == phi

            assert isinstance(rec.queue[1], qml.Beamsplitter)
            assert rec.queue[1].parameters == [theta[0], 0]

            assert isinstance(rec.queue[2], qml.Rotation)
            assert rec.queue[2].parameters == [varphi[0]]

            assert isinstance(rec.queue[3], qml.Rotation)
            assert rec.queue[3].parameters == [varphi[1]]

    def test_one_mode(self, tol):
        """Test that a one mode interferometer correctly gives a rotation gate"""
        varphi = [0.42342]

        with qml.tape.OperationRecorder() as rec:
            Interferometer(theta=[], phi=[], varphi=varphi, wires=0)

        assert len(rec.queue) == 1
        assert isinstance(rec.queue[0], qml.Rotation)
        assert np.allclose(rec.queue[0].parameters, varphi, atol=tol)

    def test_two_mode_rect(self, tol):
        """Test that a two mode interferometer using the rectangular mesh
        correctly gives a beamsplitter+rotation gate"""
        N = 2
        wires = range(N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342, 0.1121]

        with qml.tape.OperationRecorder() as rec:
            Interferometer(theta, phi, varphi, wires=wires)

        isinstance(rec.queue[0], qml.Beamsplitter)
        assert rec.queue[0].parameters == theta + phi

        assert isinstance(rec.queue[1], qml.Rotation)
        assert rec.queue[1].parameters == [varphi[0]]

        assert isinstance(rec.queue[2], qml.Rotation)
        assert rec.queue[2].parameters == [varphi[1]]

    def test_two_mode_triangular(self, tol):
        """Test that a two mode interferometer using the triangular mesh
        correctly gives a beamsplitter+rotation gate"""
        N = 2
        wires = range(N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342, 0.1121]

        with qml.tape.OperationRecorder() as rec:
            Interferometer(theta, phi, varphi, mesh="triangular", wires=wires)

        assert len(rec.queue) == 3

        assert isinstance(rec.queue[0], qml.Beamsplitter)
        assert rec.queue[0].parameters == theta + phi

        assert isinstance(rec.queue[1], qml.Rotation)
        assert rec.queue[1].parameters == [varphi[0]]

        assert isinstance(rec.queue[2], qml.Rotation)
        assert rec.queue[2].parameters == [varphi[1]]

    def test_three_mode(self, tol):
        """Test that a three mode interferometer using either mesh gives the correct gates"""
        N = 3
        wires = range(N)

        theta = [0.321, 0.4523, 0.21321]
        phi = [0.234, 0.324, 0.234]
        varphi = [0.42342, 0.234, 0.1121]

        with qml.tape.OperationRecorder() as rec_rect:
            Interferometer(theta, phi, varphi, wires=wires)

        with qml.tape.OperationRecorder() as rec_tria:
            Interferometer(theta, phi, varphi, wires=wires)

        for rec in [rec_rect, rec_tria]:
            # test both meshes (both give identical results for the 3 mode case).
            assert len(rec.queue) == 6

            expected_bs_wires = [[0, 1], [1, 2], [0, 1]]

            for idx, op in enumerate(rec_rect.queue[:3]):
                assert isinstance(op, qml.Beamsplitter)
                assert op.parameters == [theta[idx], phi[idx]]
                assert op.wires == Wires(expected_bs_wires[idx])

            for idx, op in enumerate(rec.queue[3:]):
                assert isinstance(op, qml.Rotation)
                assert op.parameters == [varphi[idx]]
                assert op.wires == Wires([idx])

    def test_four_mode_rect(self, tol):
        """Test that a 4 mode interferometer using rectangular mesh gives the correct gates"""
        N = 4
        wires = range(N)

        theta = [0.321, 0.4523, 0.21321, 0.123, 0.5234, 1.23]
        phi = [0.234, 0.324, 0.234, 1.453, 1.42341, -0.534]
        varphi = [0.42342, 0.234, 0.4523, 0.1121]

        with qml.tape.OperationRecorder() as rec:
            Interferometer(theta, phi, varphi, wires=wires)

        assert len(rec.queue) == 10

        expected_bs_wires = [[0, 1], [2, 3], [1, 2], [0, 1], [2, 3], [1, 2]]

        for idx, op in enumerate(rec.queue[:6]):
            assert isinstance(op, qml.Beamsplitter)
            assert op.parameters == [theta[idx], phi[idx]]
            assert op.wires == Wires(expected_bs_wires[idx])

        for idx, op in enumerate(rec.queue[6:]):
            assert isinstance(op, qml.Rotation)
            assert op.parameters == [varphi[idx]]
            assert op.wires == Wires([idx])

    def test_four_mode_triangular(self, tol):
        """Test that a 4 mode interferometer using triangular mesh gives the correct gates"""
        N = 4
        wires = range(N)

        theta = [0.321, 0.4523, 0.21321, 0.123, 0.5234, 1.23]
        phi = [0.234, 0.324, 0.234, 1.453, 1.42341, -0.534]
        varphi = [0.42342, 0.234, 0.4523, 0.1121]

        with qml.tape.OperationRecorder() as rec:
            Interferometer(theta, phi, varphi, mesh="triangular", wires=wires)

        assert len(rec.queue) == 10

        expected_bs_wires = [[2, 3], [1, 2], [0, 1], [2, 3], [1, 2], [2, 3]]

        for idx, op in enumerate(rec.queue[:6]):
            assert isinstance(op, qml.Beamsplitter)
            assert op.parameters == [theta[idx], phi[idx]]
            assert op.wires == Wires(expected_bs_wires[idx])

        for idx, op in enumerate(rec.queue[6:]):
            assert isinstance(op, qml.Rotation)
            assert op.parameters == [varphi[idx]]
            assert op.wires == Wires([idx])

    def test_integration(self, tol):
        """test integration with PennyLane and gradient calculations"""
        N = 4
        wires = range(N)
        dev = qml.device("default.gaussian", wires=N)

        sq = np.array(
            [
                [0.8734294, 0.96854066],
                [0.86919454, 0.53085569],
                [0.23272833, 0.0113988],
                [0.43046882, 0.40235136],
            ]
        )

        theta = np.array([3.28406182, 3.0058243, 3.48940764, 3.41419504, 4.7808479, 4.47598146])
        phi = np.array([3.89357744, 2.67721355, 1.81631197, 6.11891294, 2.09716418, 1.37476761])
        varphi = np.array([0.4134863, 6.17555778, 0.80334114, 2.02400747])

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            for w in wires:
                qml.Squeezing(sq[w][0], sq[w][1], wires=w)

            Interferometer(theta=theta, phi=phi, varphi=varphi, wires=wires)
            return [qml.expval(qml.NumberOperator(w)) for w in wires]

        res = circuit(theta, phi, varphi)
        expected = np.array([0.96852694, 0.23878521, 0.82310606, 0.16547786])
        assert np.allclose(res, expected, atol=tol)

    def test_interferometer_wrong_dim(self):
        """Integration test for the CVNeuralNetLayers method."""
        dev = qml.device("default.gaussian", wires=4)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            Interferometer(theta=theta, phi=phi, varphi=varphi, wires=range(4))
            return qml.expval(qml.X(0))

        theta = np.array([3.28406182, 3.0058243, 3.48940764, 3.41419504, 4.7808479, 4.47598146])
        phi = np.array([3.89357744, 2.67721355, 1.81631197, 6.11891294, 2.09716418, 1.37476761])
        varphi = np.array([0.4134863, 6.17555778, 0.80334114, 2.02400747])

        with pytest.raises(ValueError, match=r"Theta must be of shape \(6,\)"):
            wrong_theta = np.array([0.1, 0.2])
            circuit(wrong_theta, phi, varphi)

        with pytest.raises(ValueError, match=r"Phi must be of shape \(6,\)"):
            wrong_phi = np.array([0.1, 0.2])
            circuit(theta, wrong_phi, varphi)

        with pytest.raises(ValueError, match=r"Varphi must be of shape \(4,\)"):
            wrong_varphi = np.array([0.1, 0.2])
            circuit(theta, phi, wrong_varphi)


class TestSingleExcitationUnitary:
    """Tests for the SingleExcitationUnitary template from the
    pennylane.templates.subroutine module."""

    @pytest.mark.parametrize(
        ("single_wires", "ref_gates"),
        [
            (
                [0, 1, 2],
                [
                    [0, qml.RX, [0], [-np.pi / 2]],
                    [1, qml.Hadamard, [2], []],
                    [7, qml.RX, [0], [np.pi / 2]],
                    [8, qml.Hadamard, [2], []],
                    [9, qml.Hadamard, [0], []],
                    [10, qml.RX, [2], [-np.pi / 2]],
                    [16, qml.Hadamard, [0], []],
                    [17, qml.RX, [2], [np.pi / 2]],
                    [4, qml.RZ, [2], [np.pi / 6]],
                    [13, qml.RZ, [2], [-np.pi / 6]],
                ],
            ),
            (
                [10, 11],
                [
                    [0, qml.RX, [10], [-np.pi / 2]],
                    [1, qml.Hadamard, [11], []],
                    [12, qml.Hadamard, [10], []],
                    [13, qml.RX, [11], [np.pi / 2]],
                    [3, qml.RZ, [11], [np.pi / 6]],
                    [10, qml.RZ, [11], [-np.pi / 6]],
                ],
            ),
            (
                [1, 2, 3, 4],
                [
                    [2, qml.CNOT, [1, 2], []],
                    [3, qml.CNOT, [2, 3], []],
                    [4, qml.CNOT, [3, 4], []],
                    [6, qml.CNOT, [3, 4], []],
                    [7, qml.CNOT, [2, 3], []],
                    [8, qml.CNOT, [1, 2], []],
                    [13, qml.CNOT, [1, 2], []],
                    [14, qml.CNOT, [2, 3], []],
                    [15, qml.CNOT, [3, 4], []],
                    [17, qml.CNOT, [3, 4], []],
                    [18, qml.CNOT, [2, 3], []],
                    [19, qml.CNOT, [1, 2], []],
                ],
            ),
            (
                [10, 11],
                [
                    [2, qml.CNOT, [10, 11], []],
                    [4, qml.CNOT, [10, 11], []],
                    [9, qml.CNOT, [10, 11], []],
                    [11, qml.CNOT, [10, 11], []],
                ],
            ),
        ],
    )
    def test_single_ex_unitary_operations(self, single_wires, ref_gates):
        """Test the correctness of the SingleExcitationUnitary template including the gate count
        and order, the wires each operation acts on and the correct use of parameters
        in the circuit."""

        sqg = 10
        cnots = 4 * (len(single_wires) - 1)
        weight = np.pi / 3
        with qml.tape.OperationRecorder() as rec:
            SingleExcitationUnitary(weight, wires=single_wires)

        assert len(rec.queue) == sqg + cnots

        for gate in ref_gates:
            idx = gate[0]

            exp_gate = gate[1]
            res_gate = rec.queue[idx]
            assert isinstance(res_gate, exp_gate)

            exp_wires = gate[2]
            res_wires = rec.queue[idx]._wires
            assert res_wires == Wires(exp_wires)

            exp_weight = gate[3]
            res_weight = rec.queue[idx].parameters
            assert res_weight == exp_weight

    @pytest.mark.parametrize(
        ("weight", "single_wires", "msg_match"),
        [
            (0.2, [0], "expected at least two wires"),
            (0.2, [], "expected at least two wires"),
            ([0.2, 1.1], [0, 1, 2], "Weight must be a scalar"),
        ],
    )
    def test_single_excitation_unitary_exceptions(self, weight, single_wires, msg_match):
        """Test that SingleExcitationUnitary throws an exception if ``weight`` or
        ``single_wires`` parameter has illegal shapes, types or values."""
        dev = qml.device("default.qubit", wires=5)

        def circuit(weight=weight):
            SingleExcitationUnitary(weight=weight, wires=single_wires)
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev)

        with pytest.raises(ValueError, match=msg_match):
            qnode(weight=weight)

    @pytest.mark.parametrize(
        ("weight", "single_wires", "expected"),
        [
            (2.21375586, [0, 2], [-0.59956665, 1.0, 0.59956665, -1.0]),
            (-5.93892805, [1, 3], [1.0, 0.94132639, -1.0, -0.94132639]),
        ],
    )
    def test_integration(self, weight, single_wires, expected, tol):
        """Test integration with PennyLane and gradient calculations"""

        N = 4
        wires = range(N)
        dev = qml.device("default.qubit", wires=N)

        @qml.qnode(dev)
        def circuit(weight):
            init_state = np.array([0, 0, 1, 1], requires_grad=False)
            qml.BasisState(init_state, wires=wires)
            SingleExcitationUnitary(weight, wires=single_wires)
            return [qml.expval(qml.PauliZ(w)) for w in range(N)]

        res = circuit(weight)
        assert np.allclose(res, np.array(expected), atol=tol)


class TestArbitraryUnitary:
    """Test the ArbitraryUnitary template."""

    def test_correct_gates_single_wire(self):
        """Test that the correct gates are applied on a single wire."""
        weights = np.arange(3, dtype=float)

        with qml.tape.OperationRecorder() as rec:
            ArbitraryUnitary(weights, wires=[0])

        assert all(op.name == "PauliRot" and op.wires == Wires([0]) for op in rec.queue)

        pauli_words = ["X", "Y", "Z"]

        for i, op in enumerate(rec.queue):
            assert op.data[0] == weights[i]
            assert op.data[1] == pauli_words[i]

    def test_correct_gates_two_wires(self):
        """Test that the correct gates are applied on two wires."""
        weights = np.arange(15, dtype=float)

        with qml.tape.OperationRecorder() as rec:
            ArbitraryUnitary(weights, wires=[0, 1])

        assert all(op.name == "PauliRot" and op.wires == Wires([0, 1]) for op in rec.queue)

        pauli_words = [
            "XI",
            "YI",
            "ZI",
            "ZX",
            "IX",
            "XX",
            "YX",
            "YY",
            "ZY",
            "IY",
            "XY",
            "XZ",
            "YZ",
            "ZZ",
            "IZ",
        ]

        for i, op in enumerate(rec.queue):
            assert op.data[0] == weights[i]
            assert op.data[1] == pauli_words[i]

    def test_exception_wrong_dim(self):
        """Verifies that exception is raised if the
        number of dimensions of features is incorrect."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            ArbitraryUnitary(weights, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Weights tensor must be of shape"):
            weights = np.array([0, 1])
            circuit(weights)


class TestDoubleExcitationUnitary:
    """Tests for the DoubleExcitationUnitary template from the
    pennylane.templates.subroutine module."""

    @pytest.mark.parametrize(
        ("wires1", "wires2", "ref_gates"),
        [
            (
                [0, 1, 2],
                [4, 5, 6],
                [
                    [0, qml.Hadamard, [0], []],
                    [1, qml.Hadamard, [2], []],
                    [2, qml.RX, [4], [-np.pi / 2]],
                    [3, qml.Hadamard, [6], []],
                    [9, qml.RZ, [6], [np.pi / 24]],
                    [15, qml.Hadamard, [0], []],
                    [16, qml.Hadamard, [2], []],
                    [17, qml.RX, [4], [np.pi / 2]],
                    [18, qml.Hadamard, [6], []],
                ],
            ),
            (
                [0, 1],
                [4, 5],
                [
                    [15, qml.RX, [0], [-np.pi / 2]],
                    [16, qml.Hadamard, [1], []],
                    [17, qml.RX, [4], [-np.pi / 2]],
                    [18, qml.RX, [5], [-np.pi / 2]],
                    [22, qml.RZ, [5], [np.pi / 24]],
                    [26, qml.RX, [0], [np.pi / 2]],
                    [27, qml.Hadamard, [1], []],
                    [28, qml.RX, [4], [np.pi / 2]],
                    [29, qml.RX, [5], [np.pi / 2]],
                ],
            ),
            (
                [1, 2, 3],
                [7, 8, 9, 10, 11],
                [
                    [46, qml.Hadamard, [1], []],
                    [47, qml.RX, [3], [-np.pi / 2]],
                    [48, qml.RX, [7], [-np.pi / 2]],
                    [49, qml.RX, [11], [-np.pi / 2]],
                    [57, qml.RZ, [11], [np.pi / 24]],
                    [65, qml.Hadamard, [1], []],
                    [66, qml.RX, [3], [np.pi / 2]],
                    [67, qml.RX, [7], [np.pi / 2]],
                    [68, qml.RX, [11], [np.pi / 2]],
                ],
            ),
            (
                [2, 3, 4],
                [8, 9, 10],
                [
                    [57, qml.Hadamard, [2], []],
                    [58, qml.Hadamard, [4], []],
                    [59, qml.Hadamard, [8], []],
                    [60, qml.RX, [10], [-np.pi / 2]],
                    [66, qml.RZ, [10], [np.pi / 24]],
                    [72, qml.Hadamard, [2], []],
                    [73, qml.Hadamard, [4], []],
                    [74, qml.Hadamard, [8], []],
                    [75, qml.RX, [10], [np.pi / 2]],
                ],
            ),
            (
                [3, 4, 5],
                [11, 12, 13, 14, 15],
                [
                    [92, qml.RX, [3], [-np.pi / 2]],
                    [93, qml.Hadamard, [5], []],
                    [94, qml.Hadamard, [11], []],
                    [95, qml.Hadamard, [15], []],
                    [103, qml.RZ, [15], [-np.pi / 24]],
                    [111, qml.RX, [3], [np.pi / 2]],
                    [112, qml.Hadamard, [5], []],
                    [113, qml.Hadamard, [11], []],
                    [114, qml.Hadamard, [15], []],
                ],
            ),
            (
                [4, 5, 6, 7],
                [9, 10],
                [
                    [95, qml.Hadamard, [4], []],
                    [96, qml.RX, [7], [-np.pi / 2]],
                    [97, qml.Hadamard, [9], []],
                    [98, qml.Hadamard, [10], []],
                    [104, qml.RZ, [10], [-np.pi / 24]],
                    [110, qml.Hadamard, [4], []],
                    [111, qml.RX, [7], [np.pi / 2]],
                    [112, qml.Hadamard, [9], []],
                    [113, qml.Hadamard, [10], []],
                ],
            ),
            (
                [5, 6],
                [10, 11, 12],
                [
                    [102, qml.RX, [5], [-np.pi / 2]],
                    [103, qml.RX, [6], [-np.pi / 2]],
                    [104, qml.RX, [10], [-np.pi / 2]],
                    [105, qml.Hadamard, [12], []],
                    [110, qml.RZ, [12], [-np.pi / 24]],
                    [115, qml.RX, [5], [np.pi / 2]],
                    [116, qml.RX, [6], [np.pi / 2]],
                    [117, qml.RX, [10], [np.pi / 2]],
                    [118, qml.Hadamard, [12], []],
                ],
            ),
            (
                [3, 4, 5, 6],
                [17, 18, 19],
                [
                    [147, qml.RX, [3], [-np.pi / 2]],
                    [148, qml.RX, [6], [-np.pi / 2]],
                    [149, qml.Hadamard, [17], []],
                    [150, qml.RX, [19], [-np.pi / 2]],
                    [157, qml.RZ, [19], [-np.pi / 24]],
                    [164, qml.RX, [3], [np.pi / 2]],
                    [165, qml.RX, [6], [np.pi / 2]],
                    [166, qml.Hadamard, [17], []],
                    [167, qml.RX, [19], [np.pi / 2]],
                ],
            ),
            (
                [6, 7],
                [8, 9],
                [
                    [4, qml.CNOT, [6, 7], []],
                    [5, qml.CNOT, [7, 8], []],
                    [6, qml.CNOT, [8, 9], []],
                    [8, qml.CNOT, [8, 9], []],
                    [9, qml.CNOT, [7, 8], []],
                    [10, qml.CNOT, [6, 7], []],
                ],
            ),
            (
                [4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13],
                [
                    [58, qml.CNOT, [4, 5], []],
                    [59, qml.CNOT, [5, 6], []],
                    [60, qml.CNOT, [6, 7], []],
                    [61, qml.CNOT, [7, 8], []],
                    [62, qml.CNOT, [8, 9], []],
                    [63, qml.CNOT, [9, 10], []],
                    [64, qml.CNOT, [10, 11], []],
                    [65, qml.CNOT, [11, 12], []],
                    [66, qml.CNOT, [12, 13], []],
                    [122, qml.CNOT, [12, 13], []],
                    [123, qml.CNOT, [11, 12], []],
                    [124, qml.CNOT, [10, 11], []],
                    [125, qml.CNOT, [9, 10], []],
                    [126, qml.CNOT, [8, 9], []],
                    [127, qml.CNOT, [7, 8], []],
                    [128, qml.CNOT, [6, 7], []],
                    [129, qml.CNOT, [5, 6], []],
                    [130, qml.CNOT, [4, 5], []],
                ],
            ),
        ],
    )
    def test_double_ex_unitary_operations(self, wires1, wires2, ref_gates):
        """Test the correctness of the DoubleExcitationUnitary template including the gate count
        and order, the wires each operation acts on and the correct use of parameters
        in the circuit."""

        sqg = 72
        cnots = 16 * (len(wires1) - 1 + len(wires2) - 1 + 1)
        weight = np.pi / 3
        with qml.tape.OperationRecorder() as rec:
            DoubleExcitationUnitary(weight, wires1=wires1, wires2=wires2)

        assert len(rec.queue) == sqg + cnots

        for gate in ref_gates:
            idx = gate[0]

            exp_gate = gate[1]
            res_gate = rec.queue[idx]
            assert isinstance(res_gate, exp_gate)

            exp_wires = gate[2]
            res_wires = rec.queue[idx]._wires
            assert res_wires == Wires(exp_wires)

            exp_weight = gate[3]
            res_weight = rec.queue[idx].parameters
            assert res_weight == exp_weight

    @pytest.mark.parametrize(
        ("weight", "wires1", "wires2", "msg_match"),
        [
            (0.2, [0], [1, 2], "expected at least two wires representing the occupied"),
            (0.2, [0, 1], [2], "expected at least two wires representing the unoccupied"),
            (0.2, [0], [1], "expected at least two wires representing the occupied"),
            ([0.2, 1.1], [0, 2], [4, 6], "Weight must be a scalar"),
        ],
    )
    def test_double_excitation_unitary_exceptions(self, weight, wires1, wires2, msg_match):
        """Test that DoubleExcitationUnitary throws an exception if ``weight`` or
        ``pphh`` parameter has illegal shapes, types or values."""
        dev = qml.device("default.qubit", wires=10)

        def circuit(weight=weight):
            DoubleExcitationUnitary(weight=weight, wires1=wires1, wires2=wires2)
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev)

        with pytest.raises(ValueError, match=msg_match):
            qnode(weight=weight)

    @pytest.mark.parametrize(
        ("weight", "wires1", "wires2", "expected"),
        [
            (1.34817, [0, 1], [3, 4], [0.22079189, 0.22079189, 1.0, -0.22079189, -0.22079189]),
            (0.84817, [1, 2], [3, 4], [1.0, 0.66135688, 0.66135688, -0.66135688, -0.66135688]),
        ],
    )
    def test_integration(self, weight, wires1, wires2, expected, tol):
        """Test integration with PennyLane and gradient calculations"""

        N = 5
        dev = qml.device("default.qubit", wires=N)

        @qml.qnode(dev)
        def circuit(weight):
            init_state = np.array([0, 0, 0, 1, 1], requires_grad=False)
            qml.BasisState(init_state, wires=range(N))
            DoubleExcitationUnitary(weight, wires1=wires1, wires2=wires2)
            return [qml.expval(qml.PauliZ(w)) for w in range(N)]

        res = circuit(weight)
        assert np.allclose(res, np.array(expected), atol=tol)


class TestUCCSDUnitary:
    """Tests for the UCCSD template from the pennylane.templates.subroutine module."""

    @pytest.mark.parametrize(
        ("s_wires", "d_wires", "weights", "ref_gates"),
        [
            (
                [[0, 1, 2]],
                [],
                np.array([3.815]),
                [
                    [0, qml.BasisState, [0, 1, 2, 3, 4, 5], [np.array([0, 0, 0, 0, 1, 1])]],
                    [1, qml.RX, [0], [-np.pi / 2]],
                    [5, qml.RZ, [2], [1.9075]],
                    [6, qml.CNOT, [1, 2], []],
                ],
            ),
            (
                [[0, 1, 2], [1, 2, 3]],
                [],
                np.array([3.815, 4.866]),
                [
                    [2, qml.Hadamard, [2], []],
                    [8, qml.RX, [0], [np.pi / 2]],
                    [12, qml.CNOT, [0, 1], []],
                    [23, qml.RZ, [3], [2.433]],
                    [24, qml.CNOT, [2, 3], []],
                    [26, qml.RX, [1], [np.pi / 2]],
                ],
            ),
            (
                [],
                [[[0, 1], [2, 3, 4, 5]]],
                np.array([3.815]),
                [
                    [3, qml.RX, [2], [-np.pi / 2]],
                    [29, qml.RZ, [5], [0.476875]],
                    [73, qml.Hadamard, [0], []],
                    [150, qml.RX, [1], [np.pi / 2]],
                    [88, qml.CNOT, [3, 4], []],
                    [121, qml.CNOT, [2, 3], []],
                ],
            ),
            (
                [],
                [[[0, 1], [2, 3]], [[0, 1], [4, 5]]],
                np.array([3.815, 4.866]),
                [
                    [4, qml.Hadamard, [3], []],
                    [16, qml.RX, [0], [-np.pi / 2]],
                    [38, qml.RZ, [3], [0.476875]],
                    [78, qml.Hadamard, [2], []],
                    [107, qml.RX, [1], [-np.pi / 2]],
                    [209, qml.Hadamard, [4], []],
                    [218, qml.RZ, [5], [-0.60825]],
                    [82, qml.CNOT, [2, 3], []],
                    [159, qml.CNOT, [4, 5], []],
                ],
            ),
            (
                [[0, 1, 2, 3, 4], [1, 2, 3]],
                [[[0, 1], [2, 3]], [[0, 1], [4, 5]]],
                np.array([3.815, 4.866, 1.019, 0.639]),
                [
                    [16, qml.RX, [0], [-np.pi / 2]],
                    [47, qml.Hadamard, [1], []],
                    [74, qml.Hadamard, [2], []],
                    [83, qml.RZ, [3], [-0.127375]],
                    [134, qml.RX, [4], [np.pi / 2]],
                    [158, qml.RZ, [5], [0.079875]],
                    [188, qml.RZ, [5], [-0.079875]],
                    [96, qml.CNOT, [1, 2], []],
                    [235, qml.CNOT, [1, 4], []],
                ],
            ),
        ],
    )
    def test_uccsd_operations(self, s_wires, d_wires, weights, ref_gates):
        """Test the correctness of the UCCSD template including the gate count
        and order, the wires the operation acts on and the correct use of parameters
        in the circuit."""

        sqg = 10 * len(s_wires) + 72 * len(d_wires)

        cnots = 0
        for s_wires_ in s_wires:
            cnots += 4 * (len(s_wires_) - 1)

        for d_wires_ in d_wires:
            cnots += 16 * (len(d_wires_[0]) - 1 + len(d_wires_[1]) - 1 + 1)
        N = 6
        wires = range(N)

        ref_state = np.array([1, 1, 0, 0, 0, 0])

        with qml.tape.OperationRecorder() as rec:
            UCCSD(weights, wires, s_wires=s_wires, d_wires=d_wires, init_state=ref_state)

        assert len(rec.queue) == sqg + cnots + 1

        for gate in ref_gates:
            idx = gate[0]

            exp_gate = gate[1]
            res_gate = rec.queue[idx]
            assert isinstance(res_gate, exp_gate)

            exp_wires = gate[2]
            res_wires = rec.queue[idx]._wires
            assert res_wires == Wires(exp_wires)

            exp_weight = gate[3]
            res_weight = rec.queue[idx].parameters
            if exp_gate != qml.BasisState:
                assert res_weight == exp_weight
            else:
                assert np.allclose(res_weight, exp_weight)

    @pytest.mark.parametrize(
        ("weights", "s_wires", "d_wires", "init_state", "msg_match"),
        [
            (
                np.array([-2.8]),
                [[0, 1, 2]],
                [],
                np.array([1.2, 1, 0, 0]),
                "Elements of 'init_state' must be integers",
            ),
            (
                np.array([-2.8]),
                [],
                [],
                np.array([1, 1, 0, 0]),
                "s_wires and d_wires lists can not be both empty",
            ),
            (
                np.array([-2.8]),
                [],
                [[[0, 1, 2, 3]]],
                np.array([1, 1, 0, 0]),
                "expected entries of d_wires to be of size 2",
            ),
            (
                np.array([-2.8]),
                [[0, 2]],
                [],
                np.array([1, 1, 0, 0, 0]),
                "BasisState parameter and wires",
            ),
            (
                np.array([-2.8, 1.6]),
                [[0, 1, 2]],
                [],
                np.array([1, 1, 0, 0]),
                "Weights tensor must be of",
            ),
            (
                np.array([-2.8, 1.6]),
                [],
                [[[0, 1], [2, 3]]],
                np.array([1, 1, 0, 0]),
                "Weights tensor must be of",
            ),
            (
                np.array([-2.8, 1.6]),
                [[0, 1, 2], [1, 2, 3]],
                [[[0, 1], [2, 3]]],
                np.array([1, 1, 0, 0]),
                "Weights tensor must be of",
            ),
        ],
    )
    def test_uccsd_xceptions(self, weights, s_wires, d_wires, init_state, msg_match):
        """Test that UCCSD throws an exception if the parameters have illegal
        shapes, types or values."""
        N = 4
        wires = range(4)
        dev = qml.device("default.qubit", wires=N)

        def circuit(
            weights=weights, wires=wires, s_wires=s_wires, d_wires=d_wires, init_state=init_state
        ):
            UCCSD(
                weights=weights,
                wires=wires,
                s_wires=s_wires,
                d_wires=d_wires,
                init_state=init_state,
            )
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev)

        with pytest.raises(ValueError, match=msg_match):
            qnode(
                weights=weights,
                wires=wires,
                s_wires=s_wires,
                d_wires=d_wires,
                init_state=init_state,
            )

    @pytest.mark.parametrize(
        ("weights", "s_wires", "d_wires", "expected"),
        [
            (
                np.array([3.90575761, -1.89772083, -1.36689032]),
                [[0, 1, 2], [1, 2, 3]],
                [[[0, 1], [2, 3]]],
                [-0.14619406, -0.06502792, 0.14619406, 0.06502792],
            )
        ],
    )
    def test_integration(self, weights, s_wires, d_wires, expected, tol):
        """Test integration with PennyLane and gradient calculations"""

        N = 4
        wires = range(N)
        dev = qml.device("default.qubit", wires=N)

        w0 = weights[0]
        w1 = weights[1]
        w2 = weights[2]

        @qml.qnode(dev)
        def circuit(w0, w1, w2):
            UCCSD(
                [w0, w1, w2],
                wires,
                s_wires=s_wires,
                d_wires=d_wires,
                init_state=np.array([1, 1, 0, 0]),
            )
            return [qml.expval(qml.PauliZ(w)) for w in range(N)]

        res = circuit(w0, w1, w2)
        assert np.allclose(res, np.array(expected), atol=tol)


class TestApproxTimeEvolution:
    """Tests for the ApproxTimeEvolution template from the pennylane.templates.subroutine module."""

    def test_hamiltonian_error(self):
        """Tests if the correct error is thrown when hamiltonian is not a pennylane.Hamiltonian object"""

        n_wires = 2
        dev = qml.device("default.qubit", wires=n_wires)

        hamiltonian = np.array([[1, 1], [1, 1]])

        @qml.qnode(dev)
        def circuit():
            ApproxTimeEvolution(hamiltonian, 2, 3)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="hamiltonian must be of type pennylane.Hamiltonian"):
            circuit()

    @pytest.mark.parametrize(
        ("hamiltonian", "output"),
        [
            (qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.Hadamard(0)]), "Hadamard"),
            (
                qml.Hamiltonian(
                    [1, 1],
                    [qml.PauliX(0) @ qml.Hermitian(np.array([[1, 1], [1, 1]]), 1), qml.PauliX(0)],
                ),
                "Hermitian",
            ),
        ],
    )
    def test_non_pauli_error(self, hamiltonian, output):
        """Tests if the correct errors are thrown when the user attempts to input a matrix with non-Pauli terms"""

        n_wires = 2
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            ApproxTimeEvolution(hamiltonian, 2, 3)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        with pytest.raises(
            ValueError, match="hamiltonian must be written in terms of Pauli matrices"
        ):
            circuit()

    @pytest.mark.parametrize(
        ("time", "hamiltonian", "steps", "gates"),
        [
            (
                2,
                qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(1)]),
                2,
                [
                    qml.PauliRot(2.0, "X", wires=[0]),
                    qml.PauliRot(2.0, "X", wires=[1]),
                    qml.PauliRot(2.0, "X", wires=[0]),
                    qml.PauliRot(2.0, "X", wires=[1]),
                ],
            ),
            (
                2,
                qml.Hamiltonian([2, 0.5], [qml.PauliX("a"), qml.PauliZ("b") @ qml.PauliX("a")]),
                2,
                [
                    qml.PauliRot(4.0, "X", wires=["a"]),
                    qml.PauliRot(1.0, "ZX", wires=["b", "a"]),
                    qml.PauliRot(4.0, "X", wires=["a"]),
                    qml.PauliRot(1.0, "ZX", wires=["b", "a"]),
                ],
            ),
            (
                2,
                qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.Identity(0) @ qml.Identity(1)]),
                2,
                [qml.PauliRot(2.0, "X", wires=[0]), qml.PauliRot(2.0, "X", wires=[0])],
            ),
            (
                2,
                qml.Hamiltonian(
                    [2, 0.5, 0.5],
                    [
                        qml.PauliX("a"),
                        qml.PauliZ(-15) @ qml.PauliX("a"),
                        qml.Identity(0) @ qml.PauliY(-15),
                    ],
                ),
                1,
                [
                    qml.PauliRot(8.0, "X", wires=["a"]),
                    qml.PauliRot(2.0, "ZX", wires=[-15, "a"]),
                    qml.PauliRot(2.0, "IY", wires=[0, -15]),
                ],
            ),
        ],
    )
    def test_evolution_operations(self, time, hamiltonian, steps, gates):
        """Tests that the sequence of gates implemented in the ApproxTimeEvolution template is correct"""

        n_wires = 2

        with qml.tape.OperationRecorder() as rec:
            ApproxTimeEvolution(hamiltonian, time, steps)

        for i, gate in enumerate(rec.operations):
            prep = [gate.parameters, gate.wires]
            target = [gates[i].parameters, gates[i].wires]

            assert prep == target

    @pytest.mark.parametrize(
        ("time", "hamiltonian", "steps", "expectation"),
        [
            (np.pi, qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(1)]), 2, [1.0, 1.0]),
            (
                np.pi / 2,
                qml.Hamiltonian([0.5, 1], [qml.PauliY(0), qml.Identity(0) @ qml.PauliX(1)]),
                1,
                [0.0, -1.0],
            ),
            (
                np.pi / 4,
                qml.Hamiltonian(
                    [1, 1, 1], [qml.PauliX(0), qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(1)]
                ),
                1,
                [0.0, 0.0],
            ),
            (
                1,
                qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(1)]),
                2,
                [-0.41614684, -0.41614684],
            ),
            (
                2,
                qml.Hamiltonian(
                    [1, 1, 1, 1],
                    [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliY(1)],
                ),
                2,
                [-0.87801124, 0.51725747],
            ),
        ],
    )
    def test_evolution_output(self, time, hamiltonian, steps, expectation):
        """Tests that the output from the ApproxTimeEvolution template is correct"""

        n_wires = 2
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            ApproxTimeEvolution(hamiltonian, time, steps)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        assert np.allclose(circuit(), expectation)


class TestPermute:
    """Tests for the Permute template from the pennylane.templates.subroutine module."""

    @pytest.mark.parametrize(
        "permutation_order,expected_error_message",
        [
            ([0], "Permutations must involve at least 2 qubits."),
            ([0, 1, 2], "Permutation must specify outcome of all wires."),
            ([0, 1, 1, 3], "Values in a permutation must all be unique"),
            ([4, 3, 2, 1], "not present in wire set"),
        ],
    )
    def test_invalid_inputs_qnodes(self, permutation_order, expected_error_message):
        """Tests if errors are thrown for invalid permutations with QNodes."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def permute_qubits():
            Permute(permutation_order, wires=dev.wires)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match=expected_error_message):
            permute_qubits()

    @pytest.mark.parametrize(
        "permutation_order,expected_error_message",
        [
            ([0], "Permutations must involve at least 2 qubits."),
            ([2, "c", "a", 0], "Permutation must specify outcome of all wires."),
            ([2, "a", "c", "c", 1], "Values in a permutation must all be unique"),
            ([2, "a", "d", "c", 1], r"not present in wire set"),
        ],
    )
    def test_invalid_inputs_tape(self, permutation_order, expected_error_message):
        """Tests if errors are thrown for invalid permutations with tapes."""

        wire_labels = [0, 2, "a", "c", 1]

        with qml.tape.QuantumTape() as tape:
            with pytest.raises(ValueError, match=expected_error_message):
                Permute(permutation_order, wires=wire_labels)

    def test_identity_permutation_qnode(self):
        """ Test that identity permutations have no effect on QNodes. """

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def identity_permutation():
            ops = Permute([0, 1, 2, 3], wires=dev.wires)
            return qml.expval(qml.PauliZ(0))

        identity_permutation()

        assert len(identity_permutation.qtape.operations) == 0

    def test_identity_permutation_tape(self):
        """ Test that identity permutations have no effect on tapes. """

        with qml.tape.QuantumTape() as tape:
            Permute([0, "a", "c", "d"], wires=[0, "a", "c", "d"])

        assert len(tape.operations) == 0

    @pytest.mark.parametrize(
        "permutation_order,expected_wires",
        [
            ([1, 0], [(0, 1)]),
            ([1, 0, 2], [(0, 1)]),
            ([1, 0, 2, 3], [(0, 1)]),
            ([0, 2, 1, 3], [(1, 2)]),
            ([2, 3, 0, 1], [(0, 2), (1, 3)]),
        ],
    )
    def test_two_cycle_permutations_qnode(self, permutation_order, expected_wires):
        """ Test some two-cycles on QNodes. """

        dev = qml.device("default.qubit", wires=len(permutation_order))

        @qml.qnode(dev)
        def two_cycle():
            Permute(permutation_order, wires=dev.wires)
            return qml.expval(qml.PauliZ(0))

        two_cycle()

        # Ensure all operations are SWAPs, and that the wires are the same
        assert all(op.name == "SWAP" for op in two_cycle.qtape.operations)
        assert [op.wires.labels for op in two_cycle.qtape.operations] == expected_wires

    @pytest.mark.parametrize(
        # For tape need to specify the wire labels
        "permutation_order,wire_order,expected_wires",
        [
            ([1, 0], [0, 1], [(0, 1)]),
            ([1, 0, 2], [0, 1, 2], [(0, 1)]),
            ([1, 0, 2, 3], [0, 1, 2, 3], [(0, 1)]),
            ([0, 2, 1, 3], [0, 1, 2, 3], [(1, 2)]),
            ([2, 3, 0, 1], [0, 1, 2, 3], [(0, 2), (1, 3)]),
            (["a", "b", 0, 1], [0, 1, "a", "b"], [(0, "a"), (1, "b")]),
        ],
    )
    def test_two_cycle_permutations_tape(self, permutation_order, wire_order, expected_wires):
        """ Test some two-cycles on tapes. """

        with qml.tape.QuantumTape() as tape:
            Permute(permutation_order, wire_order)

        # Ensure all operations are SWAPs, and that the wires are the same
        assert all(op.name == "SWAP" for op in tape.operations)
        assert [op.wires.labels for op in tape.operations] == expected_wires

    @pytest.mark.parametrize(
        "permutation_order,expected_wires",
        [
            ([1, 2, 0], [(0, 1), (1, 2)]),
            ([3, 0, 1, 2], [(0, 3), (1, 3), (2, 3)]),
            ([1, 2, 3, 0], [(0, 1), (1, 2), (2, 3)]),
        ],
    )
    def test_cyclic_permutations_qnode(self, permutation_order, expected_wires):
        """ Test more general cycles on QNodes. """

        dev = qml.device("default.qubit", wires=len(permutation_order))

        @qml.qnode(dev)
        def cycle():
            Permute(permutation_order, wires=dev.wires)
            return qml.expval(qml.PauliZ(0))

        cycle()

        # Ensure all operations are SWAPs, and that the wires are the same
        assert all(op.name == "SWAP" for op in cycle.qtape.operations)
        assert [op.wires.labels for op in cycle.qtape.operations] == expected_wires

    @pytest.mark.parametrize(
        "permutation_order,wire_order,expected_wires",
        [
            ([1, 2, 0], [0, 1, 2], [(0, 1), (1, 2)]),
            (["d", "a", "b", "c"], ["a", "b", "c", "d"], [("a", "d"), ("b", "d"), ("c", "d")]),
            (["b", 0, "d", "a"], ["a", "b", 0, "d"], [("a", "b"), ("b", 0), (0, "d")]),
        ],
    )
    def test_cyclic_permutations_tape(self, permutation_order, wire_order, expected_wires):
        """ Test more general cycles on tapes. """

        with qml.tape.QuantumTape() as tape:
            Permute(permutation_order, wire_order)

        # Ensure all operations are SWAPs, and that the wires are the same
        assert all(op.name == "SWAP" for op in tape.operations)
        assert [op.wires.labels for op in tape.operations] == expected_wires

    @pytest.mark.parametrize(
        "permutation_order,expected_wires",
        [
            ([3, 0, 2, 1], [(0, 3), (1, 3)]),
            ([1, 3, 0, 4, 2], [(0, 1), (1, 3), (2, 3), (3, 4)]),
            ([5, 1, 4, 2, 3, 0], [(0, 5), (2, 4), (3, 4)]),
        ],
    )
    def test_arbitrary_permutations_qnode(self, permutation_order, expected_wires):
        """ Test arbitrarily generated permutations on QNodes. """

        dev = qml.device("default.qubit", wires=len(permutation_order))

        @qml.qnode(dev)
        def arbitrary_perm():
            Permute(permutation_order, wires=dev.wires)
            return qml.expval(qml.PauliZ(0))

        arbitrary_perm()

        # Ensure all operations are SWAPs, and that the wires are the same
        assert all(op.name == "SWAP" for op in arbitrary_perm.qtape.operations)
        assert [op.wires.labels for op in arbitrary_perm.qtape.operations] == expected_wires

    @pytest.mark.parametrize(
        "permutation_order,wire_order,expected_wires",
        [
            ([1, 3, 0, 2], [0, 1, 2, 3], [(0, 1), (1, 3), (2, 3)]),
            (
                ["d", "a", "e", "b", "c"],
                ["a", "b", "c", "d", "e"],
                [("a", "d"), ("b", "d"), ("c", "e")],
            ),
            (
                ["p", "f", 4, "q", "z", 0, "c", "d"],
                ["z", 0, "d", "c", 4, "f", "q", "p"],
                [("z", "p"), (0, "f"), ("d", 4), ("c", "q"), (4, "p")],
            ),
        ],
    )
    def test_arbitrary_permutations_tape(self, permutation_order, wire_order, expected_wires):
        """ Test arbitrarily generated permutations on tapes. """

        with qml.tape.QuantumTape() as tape:
            Permute(permutation_order, wire_order)

        # Ensure all operations are SWAPs, and that the wires are the same
        assert all(op.name == "SWAP" for op in tape.operations)
        assert [op.wires.labels for op in tape.operations] == expected_wires

    @pytest.mark.parametrize(
        "num_wires,permutation_order,wire_subset,expected_wires",
        [
            (3, [1, 0], [0, 1], [(0, 1)]),
            (4, [3, 0, 2], [0, 2, 3], [(0, 3), (2, 3)]),
            (6, [4, 2, 1, 3], [1, 2, 3, 4], [(1, 4), (3, 4)]),
        ],
    )
    def test_subset_permutations_qnode(
        self, num_wires, permutation_order, wire_subset, expected_wires
    ):
        """ Test permutation of wire subsets on QNodes. """

        dev = qml.device("default.qubit", wires=num_wires)

        @qml.qnode(dev)
        def subset_perm():
            Permute(permutation_order, wires=wire_subset)
            return qml.expval(qml.PauliZ(0))

        subset_perm()

        # Ensure all operations are SWAPs, and that the wires are the same
        assert all(op.name == "SWAP" for op in subset_perm.qtape.operations)
        assert [op.wires.labels for op in subset_perm.qtape.operations] == expected_wires

    @pytest.mark.parametrize(
        "wire_labels,permutation_order,wire_subset,expected_wires",
        [
            ([0, 1, 2], [1, 0], [0, 1], [(0, 1)]),
            ([0, 1, 2, 3], [3, 0, 2], [0, 2, 3], [(0, 3), (2, 3)]),
            (
                [0, 2, "a", "c", 1, 4],
                [4, "c", 2, "a"],
                [2, "a", "c", 4],
                [(2, 4), ("a", "c"), ("c", 4)],
            ),
        ],
    )
    def test_subset_permutations_tape(
        self, wire_labels, permutation_order, wire_subset, expected_wires
    ):
        """ Test permutation of wire subsets on tapes. """

        with qml.tape.QuantumTape() as tape:
            # Make sure all the wires are actually there
            for wire in wire_labels:
                qml.RZ(0, wires=wire)
            Permute(permutation_order, wire_subset)

        # Make sure to start comparison after the set of RZs have been applied
        assert all(op.name == "SWAP" for op in tape.operations[len(wire_labels) :])
        assert [op.wires.labels for op in tape.operations[len(wire_labels) :]] == expected_wires


class TestQuantumPhaseEstimation:
    """Tests for the QuantumPhaseEstimation template from the pennylane.templates.subroutine
    module."""

    def test_same_wires(self):
        """Tests if a QuantumFunctionError is raised if target_wires and estimation_wires contain a
        common element"""

        with pytest.raises(qml.QuantumFunctionError, match="The target wires and estimation wires"):
            QuantumPhaseEstimation(np.eye(2), target_wires=[0, 1], estimation_wires=[1, 2])

    def test_expected_tape(self):
        """Tests if QuantumPhaseEstimation populates the tape as expected for a fixed example"""

        m = qml.RX(0.3, wires=0).matrix

        op = QuantumPhaseEstimation(m, target_wires=[0], estimation_wires=[1, 2])
        tape = op.expand()

        with qml.tape.QuantumTape() as tape2:
            qml.Hadamard(1),
            qml.ControlledQubitUnitary(m @ m, control_wires=[1], wires=[0]),
            qml.Hadamard(2),
            qml.ControlledQubitUnitary(m, control_wires=[2], wires=[0]),
            qml.QFT(wires=[1, 2]).inv()

        assert len(tape2.queue) == len(tape.queue)
        assert all([op1.name == op2.name for op1, op2 in zip(tape.queue, tape2.queue)])
        assert all([op1.wires == op2.wires for op1, op2 in zip(tape.queue, tape2.queue)])
        assert np.allclose(tape.queue[1].matrix, tape2.queue[1].matrix)
        assert np.allclose(tape.queue[3].matrix, tape2.queue[3].matrix)

    @pytest.mark.parametrize("phase", [2, 3, 6, np.pi])
    def test_phase_estimated(self, phase):
        """Tests that the QPE circuit can correctly estimate the phase of a simple RX rotation."""
        estimates = []
        wire_range = range(2, 10)

        for wires in wire_range:
            dev = qml.device("default.qubit", wires=wires)
            m = qml.RX(phase, wires=0).matrix
            target_wires = [0]
            estimation_wires = range(1, wires)

            with qml.tape.QuantumTape() as tape:
                # We want to prepare an eigenstate of RX, in this case |+>
                qml.Hadamard(wires=target_wires)

                qml.templates.QuantumPhaseEstimation(
                    m, target_wires=target_wires, estimation_wires=estimation_wires
                )
                qml.probs(estimation_wires)

            tape = tape.expand()
            res = tape.execute(dev).flatten()
            initial_estimate = np.argmax(res) / 2 ** (wires - 1)

            # We need to rescale because RX is exp(- i theta X / 2) and we expect a unitary of the
            # form exp(2 pi i theta X)
            rescaled_estimate = (1 - initial_estimate) * np.pi * 4
            estimates.append(rescaled_estimate)

        # Check that the error is monotonically decreasing
        for i in range(len(estimates) - 1):
            err1 = np.abs(estimates[i] - phase)
            err2 = np.abs(estimates[i + 1] - phase)
            assert err1 >= err2

        # This is quite a large error, but we'd need to push the qubit number up more to get it
        # lower
        assert np.allclose(estimates[-1], phase, rtol=1e-2)

    def test_phase_estimated_two_qubit(self):
        """Tests that the QPE circuit can correctly estimate the phase of a random two-qubit
        unitary."""

        unitary = unitary_group.rvs(4, random_state=1967)
        eigvals, eigvecs = np.linalg.eig(unitary)

        state = eigvecs[:, 0]
        eigval = eigvals[0]
        phase = np.real_if_close(np.log(eigval) / (2 * np.pi * 1j))

        estimates = []
        wire_range = range(3, 11)

        for wires in wire_range:
            dev = qml.device("default.qubit", wires=wires)

            target_wires = [0, 1]
            estimation_wires = range(2, wires)

            with qml.tape.QuantumTape() as tape:
                # We want to prepare an eigenstate of RX, in this case |+>
                qml.QubitStateVector(state, wires=target_wires)

                qml.templates.QuantumPhaseEstimation(
                    unitary, target_wires=target_wires, estimation_wires=estimation_wires
                )
                qml.probs(estimation_wires)

            tape = tape.expand(depth=2)
            res = tape.execute(dev).flatten()

            if phase < 0:
                estimate = np.argmax(res) / 2 ** (wires - 2) - 1
            else:
                estimate = np.argmax(res) / 2 ** (wires - 2)
            estimates.append(estimate)

        # Check that the error is monotonically decreasing
        for i in range(len(estimates) - 1):
            err1 = np.abs(estimates[i] - phase)
            err2 = np.abs(estimates[i + 1] - phase)
            assert err1 >= err2

        # This is quite a large error, but we'd need to push the qubit number up more to get it
        # lower
        assert np.allclose(estimates[-1], phase, rtol=1e-2)
