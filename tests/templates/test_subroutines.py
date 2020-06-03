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
import pennylane as qml
from pennylane import numpy as np

from pennylane.templates.subroutines import (
    Interferometer, 
    ArbitraryUnitary,
    SingleExcitationUnitary,
    DoubleExcitationUnitary,
    UCCSD
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
        for expected_pauli_word, pauli_word in zip(expected_pauli_words, _all_pauli_words_but_identity(num_wires)):
            assert expected_pauli_word == pauli_word

    @pytest.mark.parametrize("tuple,expected_word", [
        ((0,), "I"),
        ((1,), "X"),
        ((2,), "Y"),
        ((3,), "Z"),
        ((0, 0, 0), "III"),
        ((1, 2, 3), "XYZ"),
        ((1, 2, 3, 0, 0, 3, 2, 1), "XYZIIZYX"),
    ])
    def test_tuple_to_word(self, tuple, expected_word):
        assert _tuple_to_word(tuple) == expected_word

class TestInterferometer:
    """Tests for the Interferometer from the pennylane.template.layers module."""

    def test_invalid_mesh_exception(self):
        """Test that Interferometer() raises correct exception when mesh not recognized."""
        dev = qml.device('default.gaussian', wires=1)
        varphi = [0.42342]

        @qml.qnode(dev)
        def circuit(varphi, mesh=None):
            Interferometer(theta=[], phi=[], varphi=varphi, mesh=mesh, wires=0)
            return qml.expval(qml.NumberOperator(0))

        with pytest.raises(ValueError, match="Mesh option"):
            circuit(varphi, mesh='a')

    def test_invalid_mesh_exception(self):
        """Test that Interferometer() raises correct exception when beamsplitter not recognized."""
        dev = qml.device('default.gaussian', wires=1)
        varphi = [0.42342]

        @qml.qnode(dev)
        def circuit(varphi, bs=None):
            Interferometer(theta=[], phi=[], varphi=varphi, beamsplitter=bs, wires=0)
            return qml.expval(qml.NumberOperator(0))

        with pytest.raises(ValueError, match="did not recognize option"):
            circuit(varphi, bs='a')

    def test_clements_beamsplitter_convention(self, tol):
        """test the beamsplitter convention"""
        N = 2
        wires = range(N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342, 0.1121]

        with qml.utils.OperationRecorder() as rec_rect:
            Interferometer(theta, phi, varphi, mesh='rectangular', beamsplitter='clements', wires=wires)

        with qml.utils.OperationRecorder() as rec_tria:
            Interferometer(theta, phi, varphi, mesh='triangular', beamsplitter='clements', wires=wires)

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

        with qml.utils.OperationRecorder() as rec:
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

        with qml.utils.OperationRecorder() as rec:
            Interferometer(theta, phi, varphi, wires=wires)

        isinstance(rec.queue[0], qml.Beamsplitter)
        assert rec.queue[0].parameters == theta+phi

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

        with qml.utils.OperationRecorder() as rec:
            Interferometer(theta, phi, varphi, mesh='triangular', wires=wires)

        assert len(rec.queue) == 3

        assert isinstance(rec.queue[0], qml.Beamsplitter)
        assert rec.queue[0].parameters == theta+phi

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

        with qml.utils.OperationRecorder() as rec_rect:
            Interferometer(theta, phi, varphi, wires=wires)

        with qml.utils.OperationRecorder() as rec_tria:
            Interferometer(theta, phi, varphi, wires=wires)

        for rec in [rec_rect, rec_tria]:
            # test both meshes (both give identical results for the 3 mode case).
            assert len(rec.queue) == 6

            expected_bs_wires = [[0, 1], [1, 2], [0, 1]]

            for idx, op in enumerate(rec_rect.queue[:3]):
                assert isinstance(op, qml.Beamsplitter)
                assert op.parameters == [theta[idx], phi[idx]]
                assert op.wires.tolist() == expected_bs_wires[idx]

            for idx, op in enumerate(rec.queue[3:]):
                assert isinstance(op, qml.Rotation)
                assert op.parameters == [varphi[idx]]
                assert op.wires.tolist() == [idx]

    def test_four_mode_rect(self, tol):
        """Test that a 4 mode interferometer using rectangular mesh gives the correct gates"""
        N = 4
        wires = range(N)

        theta = [0.321, 0.4523, 0.21321, 0.123, 0.5234, 1.23]
        phi = [0.234, 0.324, 0.234, 1.453, 1.42341, -0.534]
        varphi = [0.42342, 0.234, 0.4523, 0.1121]

        with qml.utils.OperationRecorder() as rec:
            Interferometer(theta, phi, varphi, wires=wires)

        assert len(rec.queue) == 10

        expected_bs_wires = [[0, 1], [2, 3], [1, 2], [0, 1], [2, 3], [1, 2]]

        for idx, op in enumerate(rec.queue[:6]):
            assert isinstance(op, qml.Beamsplitter)
            assert op.parameters == [theta[idx], phi[idx]]
            assert op.wires.tolist() == expected_bs_wires[idx]

        for idx, op in enumerate(rec.queue[6:]):
            assert isinstance(op, qml.Rotation)
            assert op.parameters == [varphi[idx]]
            assert op.wires.tolist() == [idx]

    def test_four_mode_triangular(self, tol):
        """Test that a 4 mode interferometer using triangular mesh gives the correct gates"""
        N = 4
        wires = range(N)

        theta = [0.321, 0.4523, 0.21321, 0.123, 0.5234, 1.23]
        phi = [0.234, 0.324, 0.234, 1.453, 1.42341, -0.534]
        varphi = [0.42342, 0.234, 0.4523, 0.1121]

        with qml.utils.OperationRecorder() as rec:
            Interferometer(theta, phi, varphi, mesh='triangular', wires=wires)

        assert len(rec.queue) == 10

        expected_bs_wires = [[2, 3], [1, 2], [0, 1], [2, 3], [1, 2], [2, 3]]

        for idx, op in enumerate(rec.queue[:6]):
            assert isinstance(op, qml.Beamsplitter)
            assert op.parameters == [theta[idx], phi[idx]]
            assert op.wires.tolist() == expected_bs_wires[idx]

        for idx, op in enumerate(rec.queue[6:]):
            assert isinstance(op, qml.Rotation)
            assert op.parameters == [varphi[idx]]
            assert op.wires.tolist() == [idx]

    def test_integration(self, tol):
        """test integration with PennyLane and gradient calculations"""
        N = 4
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        sq = np.array([[0.8734294, 0.96854066],
                       [0.86919454, 0.53085569],
                       [0.23272833, 0.0113988 ],
                       [0.43046882, 0.40235136]])

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

        # compare the two methods of computing the Jacobian
        jac_A = circuit.jacobian((theta, phi, varphi), method="A")
        jac_F = circuit.jacobian((theta, phi, varphi), method="F")
        assert jac_A == pytest.approx(jac_F, abs=tol)


class TestSingleExcitationUnitary:
    """Tests for the SingleExcitationUnitary template from the pennylane.templates.subroutine module."""

    @pytest.mark.parametrize(
        ("ph", "ref_gates"),
        [
        ([0,2],   [[0 , qml.RX      , [0]  , [-np.pi/2]] , [1 , qml.Hadamard, [2], []],
                   [7 , qml.RX      , [0]  , [ np.pi/2]] , [8 , qml.Hadamard, [2], []],
                   [9 , qml.Hadamard, [0]  , []]         , [10, qml.RX      , [2], [-np.pi/2]],
                   [16, qml.Hadamard, [0]  , []]         , [17, qml.RX      , [2], [ np.pi/2]],
                   [4 , qml.RZ      , [2]  , [np.pi/6]]  , [13, qml.RZ      , [2], [-np.pi/6]]]
                   ),

        ([10,11], [[0 , qml.RX      , [10]  , [-np.pi/2]] , [1 , qml.Hadamard, [11], []],
                   [12, qml.Hadamard, [10]  , []]         , [13, qml.RX      , [11], [ np.pi/2]],
                   [3 , qml.RZ      , [11], [np.pi/6]]    , [10, qml.RZ      , [11], [-np.pi/6]]]
                   ),        

        ([1,4],   [[2 , qml.CNOT, [1,2], []], [3 , qml.CNOT, [2,3], []], [4 , qml.CNOT, [3,4], []],
                   [6 , qml.CNOT, [3,4], []], [7 , qml.CNOT, [2,3], []], [8 , qml.CNOT, [1,2], []],
                   [13, qml.CNOT, [1,2], []], [14, qml.CNOT, [2,3], []], [15, qml.CNOT, [3,4], []],
                   [17, qml.CNOT, [3,4], []], [18, qml.CNOT, [2,3], []], [19, qml.CNOT, [1,2], []]]
                   ),

        ([10,11], [[2 , qml.CNOT, [10,11] , []], [4  , qml.CNOT, [10,11], []],
                   [9 , qml.CNOT, [10,11] , []], [11 , qml.CNOT, [10,11], []]]
                   )        
        ]
    )
    def test_single_ex_unitary_operations(self, ph, ref_gates):
        """Test the correctness of the SingleExcitationUnitary template including the gate count
        and order, the wires each operation acts on and the correct use of parameters 
        in the circuit."""

        sqg = 10
        cnots = 4*(ph[1]-ph[0])
        weight = np.pi/3
        with qml.utils.OperationRecorder() as rec:
            SingleExcitationUnitary(weight, wires=ph)

        assert len(rec.queue) == sqg + cnots            

        for gate in ref_gates:
            idx = gate[0]

            exp_gate = gate[1]
            res_gate = rec.queue[idx]
            assert isinstance(res_gate, exp_gate)

            exp_wires = gate[2]
            res_wires = rec.queue[idx]._wires
            assert res_wires == qml.wires.Wires(exp_wires)

            exp_weight = gate[3]
            res_weight = rec.queue[idx].parameters
            assert res_weight == exp_weight

    @pytest.mark.parametrize(
        ("weight", "ph", "msg_match"),
        [
            ( 0.2      , [0]         , "expected 2 wires"),
            ( 0.2      , []          , "expected 2 wires"),
            ([0.2, 1.1], [0,2]       , "'weight' must be of shape"),
            ( 0.2      , [3, 1]      , "wires_1 must be greater than wires_0")
        ]
    )
    def test_single_excitation_unitary_exceptions(self, weight, ph, msg_match):
        """Test that SingleExcitationUnitary throws an exception if ``weight`` or 
        ``ph`` parameter has illegal shapes, types or values."""
        dev = qml.device("default.qubit", wires=5)

        def circuit(weight=weight, wires=ph):
            SingleExcitationUnitary(weight=weight, wires=ph)
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev)

        with pytest.raises(ValueError, match=msg_match):
            qnode(weight=weight, wires=ph)

    @pytest.mark.parametrize(
        ("weight", "ph", "expected"),
        [
            ( 2.21375586 , [0, 2], [-0.59956665, 1.        , 0.59956665, -1.]),
            ( -5.93892805, [1, 3], [ 1.        , 0.94132639, -1.       , -0.94132639])
        ]
    )
    def test_integration(self, weight, ph, expected, tol):
        """Test integration with PennyLane and gradient calculations"""

        N = 4
        wires = range(N)
        dev = qml.device('default.qubit', wires=N)

        @qml.qnode(dev)
        def circuit(weight):
            init_state = np.flip(np.array([1,1,0,0]))
            qml.BasisState(init_state, wires=wires)
            SingleExcitationUnitary(weight, wires=ph)

        return [qml.expval(qml.PauliZ(w)) for w in range(N)]

        res = circuit(weight)
        assert np.allclose(res, np.array(expected), atol=tol)

        # compare the two methods of computing the Jacobian
        jac_A = circuit.jacobian((weight), method="A")
        jac_F = circuit.jacobian((weight), method="F")
        assert jac_A == pytest.approx(jac_F, abs=tol)


class TestArbitraryUnitary:
    """Test the ArbitraryUnitary template."""

    def test_correct_gates_single_wire(self):
        """Test that the correct gates are applied on a single wire."""
        weights = np.arange(3, dtype=float)

        with qml.utils.OperationRecorder() as rec:
            ArbitraryUnitary(weights, wires=[0])

        assert all(op.name == "PauliRot" and op.wires.tolist() == [0] for op in rec.queue)

        pauli_words = ["X", "Y", "Z"]

        for i, op in enumerate(rec.queue):
            assert op.params[0] == weights[i]
            assert op.params[1] == pauli_words[i]

    def test_correct_gates_two_wires(self):
        """Test that the correct gates are applied on two wires."""
        weights = np.arange(15, dtype=float)

        with qml.utils.OperationRecorder() as rec:
            ArbitraryUnitary(weights, wires=[0, 1])

        assert all(op.name == "PauliRot" and op.wires.tolist() == [0, 1] for op in rec.queue)

        pauli_words = ["XI", "YI", "ZI", "ZX", "IX", "XX", "YX", "YY", "ZY", "IY", "XY", "XZ", "YZ", "ZZ", "IZ"]

        for i, op in enumerate(rec.queue):
            assert op.params[0] == weights[i]
            assert op.params[1] == pauli_words[i]


class TestDoubleExcitationUnitary:
    """Tests for the DoubleExcitationUnitary template from the pennylane.templates.subroutine module."""

    @pytest.mark.parametrize(
        ("pphh", "ref_gates"),
        [
        ([0,2,4,6],   [[0,  qml.Hadamard, [0], []]        , [1, qml.Hadamard, [2], []],
                       [2,  qml.RX,       [4], [-np.pi/2]], [3, qml.Hadamard, [6], []],
                       [9,  qml.RZ, [6], [np.pi/24]]      ,
                       [15, qml.Hadamard, [0], []]        , [16, qml.Hadamard, [2], []],
                       [17, qml.RX,       [4], [np.pi/2]] , [18, qml.Hadamard, [6], []]]
                   ),
        ([0,1,4,5],   [[15, qml.RX, [0], [-np.pi/2]], [16, qml.Hadamard, [1], []],
                       [17, qml.RX, [4], [-np.pi/2]], [18, qml.RX,       [5], [-np.pi/2]],
                       [22, qml.RZ, [5], [np.pi/24]],
                       [26, qml.RX, [0], [np.pi/2]] , [27, qml.Hadamard, [1], []],
                       [28, qml.RX, [4], [np.pi/2]] , [29, qml.RX,       [5], [np.pi/2]]]
                   ),
        ([1,3,7,11],  [[46, qml.Hadamard, [1], []]         , [47, qml.RX, [3],  [-np.pi/2]],
                       [48, qml.RX      , [7], [-np.pi/2]] , [49, qml.RX, [11], [-np.pi/2]],
                       [57, qml.RZ, [11], [np.pi/24]]      ,
                       [65, qml.Hadamard, [1], []]         , [66, qml.RX, [3] , [np.pi/2]],
                       [67, qml.RX      , [7], [np.pi/2]]  , [68, qml.RX, [11], [np.pi/2]]]
                   ),
        ([2,4,8,10],  [[57, qml.Hadamard, [2], []], [58, qml.Hadamard, [4] , []],
                       [59, qml.Hadamard, [8], []], [60, qml.RX,       [10], [-np.pi/2]],
                       [66, qml.RZ, [10], [np.pi/24]]  ,
                       [72, qml.Hadamard, [2], []], [73, qml.Hadamard, [4], []],
                       [74, qml.Hadamard, [8], []], [75, qml.RX,      [10], [np.pi/2]]]
                   ),
        ([3,5,11,15], [[92,  qml.RX,       [3],  [-np.pi/2]], [93, qml.Hadamard, [5] , []],
                       [94,  qml.Hadamard, [11], []]        , [95, qml.Hadamard, [15], []],
                       [103, qml.RZ, [15], [-np.pi/24]]     ,
                       [111, qml.RX,       [3],  [np.pi/2]] , [112, qml.Hadamard, [5] , []],
                       [113, qml.Hadamard, [11], []]        , [114, qml.Hadamard, [15], []]]
                   ),
        ([4,7,9,10] , [[95,  qml.Hadamard, [4], []]     , [96, qml.RX,       [7],  [-np.pi/2]],
                       [97,  qml.Hadamard, [9], []]     , [98, qml.Hadamard, [10], []],
                       [104, qml.RZ, [10], [-np.pi/24]] ,
                       [110, qml.Hadamard, [4], []]     , [111, qml.RX,       [7] , [np.pi/2]],
                       [112, qml.Hadamard, [9], []]     , [113, qml.Hadamard, [10], []]]
                   ),
        ([5,6,10,12], [[102, qml.RX, [5],  [-np.pi/2]]  , [103, qml.RX,       [6],  [-np.pi/2]],
                       [104, qml.RX, [10], [-np.pi/2]]  , [105, qml.Hadamard, [12], []],
                       [110, qml.RZ, [12], [-np.pi/24]] ,
                       [115, qml.RX, [5],  [np.pi/2]]   , [116, qml.RX,       [6],  [np.pi/2]],
                       [117, qml.RX, [10], [np.pi/2]]   , [118, qml.Hadamard, [12], []]]
                   ),
        ([3,6,17,19], [[147, qml.RX,       [3],  [-np.pi/2]], [148, qml.RX, [6],  [-np.pi/2]],
                       [149, qml.Hadamard, [17], []]        , [150, qml.RX, [19], [-np.pi/2]],
                       [157, qml.RZ, [19], [-np.pi/24]]     ,
                       [164, qml.RX, [3],  [np.pi/2]]       , [165, qml.RX, [6],  [np.pi/2]],
                       [166, qml.Hadamard, [17], []]        , [167, qml.RX, [19], [np.pi/2]]]
                   ),
        ([6,7,8,9]  , [[4, qml.CNOT, [6, 7], []], [5,  qml.CNOT, [7, 8], []], 
                       [6, qml.CNOT, [8, 9], []], [8,  qml.CNOT, [8, 9], []],
                       [9, qml.CNOT, [7, 8], []], [10, qml.CNOT, [6, 7], []]]
                   ),
        ([4,7,8,13] , [[58,  qml.CNOT, [4, 5],   []], [59, qml.CNOT, [5, 6],   []], 
                       [60,  qml.CNOT, [6, 7],   []], [61, qml.CNOT, [7, 8],   []],
                       [62,  qml.CNOT, [8, 9],   []], [63, qml.CNOT, [9, 10],  []],
                       [64,  qml.CNOT, [10, 11], []], [65, qml.CNOT, [11, 12], []],
                       [66,  qml.CNOT, [12,13],  []],
                       [122, qml.CNOT, [12, 13], []], [123, qml.CNOT, [11, 12], []], 
                       [124, qml.CNOT, [10, 11], []], [125, qml.CNOT, [9, 10],  []],
                       [126, qml.CNOT, [8, 9],   []], [127, qml.CNOT, [7, 8],   []],
                       [128, qml.CNOT, [6, 7],   []], [129, qml.CNOT, [5, 6],   []],
                       [130, qml.CNOT, [4,5],    []]]
                   ),
        ]
    )
    def test_double_ex_unitary_operations(self, pphh, ref_gates):
        """Test the correctness of the DoubleExcitationUnitary template including the gate count
        and order, the wires each operation acts on and the correct use of parameters 
        in the circuit."""

        sqg = 72
        cnots = 16*(pphh[1]-pphh[0] + pphh[3]-pphh[2] + 1)
        weight = np.pi/3
        with qml.utils.OperationRecorder() as rec:
            DoubleExcitationUnitary(weight, wires=pphh)

        assert len(rec.queue) == sqg + cnots

        for gate in ref_gates:
            idx = gate[0]

            exp_gate = gate[1]
            res_gate = rec.queue[idx]
            assert isinstance(res_gate, exp_gate)

            exp_wires = gate[2]
            res_wires = rec.queue[idx]._wires
            assert res_wires == qml.wires.Wires(exp_wires)

            exp_weight = gate[3]
            res_weight = rec.queue[idx].parameters
            assert res_weight == exp_weight

    @pytest.mark.parametrize(
        ("weight", "pphh", "msg_match"),
        [
            ( 0.2      , [0]                  , "expected 4 wires"),
            ( 0.2      , [0, 1]               , "expected 4 wires"),
            ( 0.2      , [0, 1, 2, 3, 4]      , "expected 4 wires"),
            ( 0.2      , []                   , "expected 4 wires"),
            ([0.2, 1.1], [0, 2, 4, 6]         , "'weight' must be of shape"),
            ( 0.2      , [1, 0, 6, 3]         , "wires_3 > wires_2 > wires_1 > wires_0"),
            ( 0.2      , [1, 0, 3, 6]         , "wires_3 > wires_2 > wires_1 > wires_0")
        ]
    )
    def test_double_excitation_unitary_exceptions(self, weight, pphh, msg_match):
        """Test that DoubleExcitationUnitary throws an exception if ``weight`` or 
        ``pphh`` parameter has illegal shapes, types or values."""
        dev = qml.device("default.qubit", wires=10)

        def circuit(weight=weight, wires=None):
            DoubleExcitationUnitary(weight=weight, wires=wires)
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev)

        with pytest.raises(ValueError, match=msg_match):
            qnode(weight=weight, wires=pphh)

    @pytest.mark.parametrize(
        ("weight", "pphh", "expected"),
        [
            (1.34817, [0, 1, 3, 4], [0.22079189, 0.22079189, 1.,         -0.22079189, -0.22079189]),
            (0.84817, [1, 2, 3, 4], [1.,         0.66135688, 0.66135688, -0.66135688, -0.66135688])
        ]
    )
    def test_integration(self, weight, pphh, expected, tol):
        """Test integration with PennyLane and gradient calculations"""

        N = 5
        wires = range(N)
        dev = qml.device('default.qubit', wires=N)

        @qml.qnode(dev)
        def circuit(weight):
            init_state = np.flip(np.array([1,1,0,0,0]))
            qml.BasisState(init_state, wires=wires)
            DoubleExcitationUnitary(weight, wires=pphh)

            return [qml.expval(qml.PauliZ(w)) for w in range(N)]

        res = circuit(weight)
        assert np.allclose(res, np.array(expected), atol=tol)

        # compare the two methods of computing the Jacobian
        jac_A = circuit.jacobian((weight), method="A")
        jac_F = circuit.jacobian((weight), method="F")
        assert jac_A == pytest.approx(jac_F, abs=tol)


class TestUCCSDUnitary:
    """Tests for the UCCSD template from the pennylane.templates.subroutine module."""

    @pytest.mark.parametrize(
        ("ph", "pphh", "weights", "ref_gates"),
        [
          ([[0, 2]], [], np.array([3.815]),
             [ [0, qml.BasisState, [0, 1, 2, 3, 4, 5], [np.array([0, 0, 0, 0, 1, 1])]],
               [1, qml.RX,         [0],        [-np.pi/2]],
               [5, qml.RZ,         [2],        [1.9075]],
               [6, qml.CNOT,       [1, 2],     []] ]),

          ([[0, 2], [1, 3]], [], np.array([3.815, 4.866]),
             [ [2,  qml.Hadamard, [2],    []],
               [8,  qml.RX,       [0],    [np.pi/2]],
               [12, qml.CNOT,     [0, 1], []],
               [23, qml.RZ,       [3],    [2.433]],
               [24, qml.CNOT,     [2, 3], []],
               [26, qml.RX,       [1],    [np.pi/2]] ]),

          ([], [[0, 1, 2, 5]], np.array([3.815]),
             [ [3,   qml.RX,       [2],    [-np.pi/2]],
               [29,  qml.RZ,       [5],    [0.476875]],
               [73,  qml.Hadamard, [0],    []],
               [150, qml.RX,       [1],    [np.pi/2]],
               [88,  qml.CNOT,     [3, 4], []],
               [121, qml.CNOT,     [2, 3], []] ]),

          ([], [[0, 1, 2, 3], [0, 1, 4, 5]], np.array([3.815, 4.866]),
             [ [4,   qml.Hadamard, [3],    []],
               [16,  qml.RX,       [0],    [-np.pi/2]],
               [38,  qml.RZ,       [3],    [0.476875]],
               [78,  qml.Hadamard, [2],    []],
               [107, qml.RX,       [1],    [-np.pi/2]],
               [209, qml.Hadamard, [4],    []],
               [218, qml.RZ,       [5],    [-0.60825]],
               [82,  qml.CNOT,     [2, 3], []],
               [159, qml.CNOT,     [4, 5], []] ]),

          ([[0, 4], [1, 3]], [[0, 1, 2, 3], [0, 1, 4, 5]], np.array([3.815, 4.866, 1.019, 0.639]),
             [ [16,  qml.RX,       [0],    [-np.pi/2]],
               [47,  qml.Hadamard, [1],    []],
               [74,  qml.Hadamard, [2],    []],
               [83,  qml.RZ,       [3],    [-0.127375]],
               [134, qml.RX,       [4],    [np.pi/2]],
               [158, qml.RZ,       [5],    [0.079875]],
               [188, qml.RZ,       [5],    [-0.079875]],
               [96,  qml.CNOT,     [1, 2], []],
               [235, qml.CNOT,     [1, 4], []] ])
        ]
    )
    def test_uccsd_operations(self, ph, pphh, weights, ref_gates):
        """Test the correctness of the UCCSD template including the gate count
        and order, the wires the operation acts on and the correct use of parameters 
        in the circuit."""

        sqg = 10*len(ph) + 72*len(pphh)

        cnots = 0
        for i_ph in ph:
            cnots += 4*(i_ph[1]-i_ph[0])

        for i_pphh in pphh:
            cnots += 16*(i_pphh[1]-i_pphh[0] + i_pphh[3]-i_pphh[2] + 1)
        N = 6
        wires = range(N)

        ref_state = np.array([1, 1, 0, 0, 0, 0])

        with qml.utils.OperationRecorder() as rec:
            UCCSD(weights, wires, ph=ph, pphh=pphh, init_state=ref_state)

        assert len(rec.queue) == sqg + cnots + 1

        for gate in ref_gates:
            idx = gate[0]

            exp_gate = gate[1]
            res_gate = rec.queue[idx]
            assert isinstance(res_gate, exp_gate)

            exp_wires = gate[2]
            res_wires = rec.queue[idx]._wires
            assert res_wires == qml.wires.Wires(exp_wires)

            exp_weight = gate[3]
            res_weight = rec.queue[idx].parameters
            if exp_gate != qml.BasisState:
                assert res_weight == exp_weight
            else:
                assert np.allclose(res_weight, exp_weight)

    @pytest.mark.parametrize(
        ("weights", "ph", "pphh", "init_state", "msg_match"),
        [
            ( np.array([-2.8]), [[0, 2]], [], [1, 1, 0, 0],
             "'init_state' must be a Numpy array"),

            ( np.array([-2.8]), [[0, 2]], [], (1, 1, 0, 0),
             "'init_state' must be a Numpy array"),

            ( np.array([-2.8]), [[0, 2]], [], np.array([1.2, 1, 0, 0]),
             "Elements of 'init_state' must be integers"),

            ( np.array([-2.8]), [], [], np.array([1, 1, 0, 0]),
             "'ph' and 'pphh' lists can not be both empty"),

            ( np.array([-2.8]), None, None, np.array([1, 1, 0, 0]),
             "'ph' and 'pphh' lists can not be both empty"),

            ( np.array([-2.8]), None, [[0, 1, 2, 3]], np.array([1, 1, 0, 0]),
             "'ph' must be a list"),

            ( np.array([-2.8, 1.6]), [0, [1, 2]], [], np.array([1, 1, 0, 0]),
             "Each element of 'ph' must be a list"),

            ( np.array([-2.8, 1.6]), [["a", 3], [1, 2]], [], np.array([1, 1, 0, 0]),
             "Each element of 'ph' must be a list of integers"),

            ( np.array([-2.8, 1.6]), [[1.4, 3], [1, 2]], [], np.array([1, 1, 0, 0]),
             "Each element of 'ph' must be a list of integers"),

            ( np.array([-2.8]), [[0, 2]], None, np.array([1, 1, 0, 0]),
             "'pphh' must be a list"),

            ( np.array([-2.8, 1.6]), [], [0, [1, 2, 3, 4]], np.array([1, 1, 0, 0]),
             "Each element of 'pphh' must be a list"),

            ( np.array([-2.8, 1.6]), [], [[0, 1, "a", 3], [1, 2, 3, 4]], np.array([1, 1, 0, 0]),
             "Each element of 'pphh' must be a list of integers"),

            ( np.array([-2.8, 1.6]), [], [[0, 1, 1.4, 3], [1, 2, 3, 4]], np.array([1, 1, 0, 0]),
             "Each element of 'pphh' must be a list of integers"),

            ( np.array([-2.8]), [[0, 2]], [], np.array([1, 1, 0, 0, 0]),
             "'init_state' must be of shape"),

            ( np.array([-2.8, 1.6]), [[0, 2], [1, 3, 4]], [], np.array([1, 1, 0, 0]),
             "Elements of 'ph' must be of shape"),

            ( np.array([-2.8, 1.6]), [[0, 2]], [[0, 1, 2,]], np.array([1, 1, 0, 0]),
             "Elements of 'pphh' must be of shape"),

            ( np.array([-2.8, 1.6]), [[0, 2]], [], np.array([1, 1, 0, 0]),
             "'weights' must be of shape"),

            ( np.array([-2.8, 1.6]), [], [[0, 1, 2, 3]], np.array([1, 1, 0, 0]),
             "'weights' must be of shape"),

            ( np.array([-2.8, 1.6]), [[0, 2], [1, 3]], [[0, 1, 2, 3]], np.array([1, 1, 0, 0]),
             "'weights' must be of shape")
        
        ]
    )
    def test_uccsd_xceptions(self, weights, ph, pphh, init_state, msg_match):
        """Test that UCCSD throws an exception if the parameters have illegal
        shapes, types or values."""
        N=4
        wires = range(4)
        dev = qml.device("default.qubit", wires=N)

        def circuit(weights=weights, wires=wires, ph=ph, pphh=pphh, init_state=init_state):
            UCCSD(weights=weights, wires=wires, ph=ph, pphh=pphh, init_state=init_state)
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev)

        with pytest.raises(ValueError, match=msg_match):
            qnode(weights=weights, wires=wires, ph=ph, pphh=pphh, init_state=init_state)

    @pytest.mark.parametrize(
        ("weights", "ph", "pphh", "expected"),
        [
            (np.array([3.90575761, -1.89772083, -1.36689032]),
             [[0, 2], [1, 3]], [[0, 1, 2, 3]],
             [-0.14619406, -0.06502792, 0.14619406, 0.06502792])
        ]
    )
    def test_integration(self, weights, ph, pphh, expected, tol):
        """Test integration with PennyLane and gradient calculations"""

        N = 4
        wires = range(N)
        dev = qml.device('default.qubit', wires=N)

        w_ph_0 = weights[0]
        w_ph_1 = weights[1]
        w_pphh = weights[2]

        @qml.qnode(dev)
        def circuit(w_ph_0, w_ph_1, w_pphh):
            UCCSD([w_ph_0, w_ph_1, w_pphh], wires, ph=ph, pphh=pphh, init_state=np.array([1, 1, 0, 0]))
            return [qml.expval(qml.PauliZ(w)) for w in range(N)]

        res = circuit(w_ph_0, w_ph_1, w_pphh)
        assert np.allclose(res, np.array(expected), atol=tol)

        # compare the two methods of computing the Jacobian
        jac_A = circuit.jacobian((w_ph_0, w_ph_1, w_pphh), method="A")
        jac_F = circuit.jacobian((w_ph_0, w_ph_1, w_pphh), method="F")
        assert jac_A == pytest.approx(jac_F, abs=tol)
