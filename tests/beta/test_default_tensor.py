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
Unit tests for the :mod:`pennylane.plugin.Tensornet` device.
"""
import cmath
# pylint: disable=protected-access,cell-var-from-loop
import math

import pytest

# TODO: remove the following skip when Tensornet has been ported to
# Qubit device, and the gate imports above are removed.
tensorflow = pytest.importorskip("tensorflow", minversion="2.0")

import pennylane as qml
from pennylane import numpy as np, QuantumFunctionError
from pennylane.beta.plugins.numpy_ops import (
    CNOT,
    CSWAP,
    CZ,
    SWAP,
    CRot3,
    CRotx,
    CRoty,
    CRotz,
    H,
    Rot3,
    Rotx,
    Roty,
    Rotz,
    Rphi,
    S,
    T,
    X,
    Y,
    Z,
    hermitian,
    identity,
    Toffoli,
    spectral_decomposition,
    unitary,
)

tensornetwork = pytest.importorskip("tensornetwork", minversion="0.3")


U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ]
)


U2 = np.array(
    [
        [
            -0.07843244 - 3.57825948e-01j,
            0.71447295 - 5.38069384e-02j,
            0.20949966 + 6.59100734e-05j,
            -0.50297381 + 2.35731613e-01j,
        ],
        [
            -0.26626692 + 4.53837083e-01j,
            0.27771991 - 2.40717436e-01j,
            0.41228017 - 1.30198687e-01j,
            0.01384490 - 6.33200028e-01j,
        ],
        [
            -0.69254712 - 2.56963068e-02j,
            -0.15484858 + 6.57298384e-02j,
            -0.53082141 + 7.18073414e-02j,
            -0.41060450 - 1.89462315e-01j,
        ],
        [
            -0.09686189 - 3.15085273e-01j,
            -0.53241387 - 1.99491763e-01j,
            0.56928622 + 3.97704398e-01j,
            -0.28671074 - 6.01574497e-02j,
        ],
    ]
)


U_toffoli = np.diag([1 for i in range(8)])
U_toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])

U_swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

U_cswap = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1]])


H = np.array(
    [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
)


THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


def prep_par(par, op):
    "Convert par into a list of parameters that op expects."
    if op.par_domain == "A":
        return [np.diag([x, 1]) for x in par]
    return par


def edges_valid(dev, num_nodes):
    """Returns True if the edges in a device are properly accounted for, when there are num_nodes in tensor network"""
    node_edges = [dev._nodes['state'][idx].edges for idx in range(num_nodes)]
    node_edges_set = set([edge for sublist in node_edges for edge in sublist])
    return node_edges_set == set(dev._terminal_edges)


class TestAuxiliaryFunctions:
    """Test auxiliary functions."""

    def test_spectral_decomposition(self, tol):
        """Test that the correct spectral decomposition is returned."""

        a, P = spectral_decomposition(H)

        # verify that H = \sum_k a_k P_k
        assert np.allclose(H, np.einsum("i,ijk->jk", a, P), atol=tol, rtol=0)

    def test_phase_shift(self, tol):
        """Test phase shift is correct"""

        # test identity for theta=0
        assert np.allclose(Rphi(0), np.identity(2), atol=tol, rtol=0)

        # test arbitrary phase shift
        phi = 0.5432
        expected = np.array([[1, 0], [0, np.exp(1j * phi)]])
        assert np.allclose(Rphi(phi), expected, atol=tol, rtol=0)

    def test_x_rotation(self, tol):
        """Test x rotation is correct"""

        # test identity for theta=0
        assert np.allclose(Rotx(0), np.identity(2), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)
        assert np.allclose(Rotx(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = -1j * np.array([[0, 1], [1, 0]])
        assert np.allclose(Rotx(np.pi), expected, atol=tol, rtol=0)

    def test_y_rotation(self, tol):
        """Test y rotation is correct"""

        # test identity for theta=0
        assert np.allclose(Roty(0), np.identity(2), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
        assert np.allclose(Roty(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = np.array([[0, -1], [1, 0]])
        assert np.allclose(Roty(np.pi), expected, atol=tol, rtol=0)

    def test_z_rotation(self, tol):
        """Test z rotation is correct"""

        # test identity for theta=0
        assert np.allclose(Rotz(0), np.identity(2), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.diag(np.exp([-1j * np.pi / 4, 1j * np.pi / 4]))
        assert np.allclose(Rotz(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        assert np.allclose(Rotz(np.pi), -1j * Z, atol=tol, rtol=0)

    def test_arbitrary_rotation(self, tol):
        """Test arbitrary single qubit rotation is correct"""

        # test identity for phi,theta,omega=0
        assert np.allclose(Rot3(0, 0, 0), np.identity(2), atol=tol, rtol=0)

        # expected result
        def arbitrary_rotation(x, y, z):
            """arbitrary single qubit rotation"""
            c = np.cos(y / 2)
            s = np.sin(y / 2)
            return np.array(
                [
                    [np.exp(-0.5j * (x + z)) * c, -np.exp(0.5j * (x - z)) * s],
                    [np.exp(-0.5j * (x - z)) * s, np.exp(0.5j * (x + z)) * c],
                ]
            )

        a, b, c = 0.432, -0.152, 0.9234
        assert np.allclose(Rot3(a, b, c), arbitrary_rotation(a, b, c), atol=tol, rtol=0)

    def test_C_x_rotation(self, tol):
        """Test controlled x rotation is correct"""

        # test identity for theta=0
        assert np.allclose(CRotx(0), np.identity(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1/np.sqrt(2), -1j/np.sqrt(2)], [0, 0, -1j/np.sqrt(2), 1/np.sqrt(2)]])
        assert np.allclose(CRotx(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]])
        assert np.allclose(CRotx(np.pi), expected, atol=tol, rtol=0)

    def test_C_y_rotation(self, tol):
        """Test controlled y rotation is correct"""

        # test identity for theta=0
        assert np.allclose(CRoty(0), np.identity(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1/np.sqrt(2), -1/np.sqrt(2)], [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]])
        assert np.allclose(CRoty(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
        assert np.allclose(CRoty(np.pi), expected, atol=tol, rtol=0)

    def test_C_z_rotation(self, tol):
        """Test controlled z rotation is correct"""

        # test identity for theta=0
        assert np.allclose(CRotz(0), np.identity(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.exp(-1j * np.pi / 4), 0], [0, 0, 0, np.exp(1j * np.pi / 4)]])
        assert np.allclose(CRotz(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1j, 0], [0, 0, 0, 1j]])
        assert np.allclose(CRotz(np.pi), expected, atol=tol, rtol=0)

    def test_controlled_arbitrary_rotation(self, tol):
        """Test controlled arbitrary rotation is correct"""

        # test identity for phi,theta,omega=0
        assert np.allclose(CRot3(0, 0, 0), np.identity(4), atol=tol, rtol=0)

        # test identity for phi,theta,omega=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
        assert np.allclose(CRot3(np.pi, np.pi, np.pi), expected, atol=tol, rtol=0)

        def arbitrary_Crotation(x, y, z):
            """controlled arbitrary single qubit rotation"""
            c = np.cos(y / 2)
            s = np.sin(y / 2)
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, np.exp(-0.5j * (x + z)) * c, -np.exp(0.5j * (x - z)) * s],
                    [0, 0, np.exp(-0.5j * (x - z)) * s, np.exp(0.5j * (x + z)) * c]
                ]
            )

        a, b, c = 0.432, -0.152, 0.9234
        assert np.allclose(CRot3(a, b, c), arbitrary_Crotation(a, b, c), atol=tol, rtol=0)


class TestMatrixOperations:
    """Tests for unitary and hermitian functions."""

    def test_unitary(self, tol):
        """Test that the unitary function produces the correct output."""

        out = unitary(U)

        # verify output type
        assert isinstance(out, np.ndarray)

        # verify equivalent to input state
        assert np.allclose(out, U, atol=tol, rtol=0)

    def test_unitary_exceptions(self):
        """Tests that the unitary function raises the proper errors."""

        # test non-square matrix
        with pytest.raises(ValueError, match="must be a square matrix"):
            unitary(U[1:])

        # test non-unitary matrix
        U3 = U.copy()
        U3[0, 0] += 0.5
        with pytest.raises(ValueError, match="must be unitary"):
            unitary(U3)

    def test_hermitian(self, tol):
        """Test that the hermitian function produces the correct output."""

        out = hermitian(H)

        # verify output type
        assert isinstance(out, np.ndarray)

        # verify equivalent to input state
        assert np.allclose(out, H, atol=tol, rtol=0)

    def test_hermitian_exceptions(self):
        """Tests that the hermitian function raises the proper errors."""

        # test non-square matrix
        with pytest.raises(ValueError, match="must be a square matrix"):
            hermitian(H[1:])

        # test non-Hermitian matrix
        H2 = H.copy()
        H2[0, 1] = H2[0, 1].conj()
        with pytest.raises(ValueError, match="must be Hermitian"):
            hermitian(H2)

@pytest.mark.parametrize("rep", ("exact", "mps"))
class TestDefaultTensorNetwork:
    """Tests of the basic tensor network functionality of default.tensor plugin."""

    def test_clear_network_data(self, rep):
        """Tests that the _clear_network method clears the relevant bookkeeping data."""

        dev = qml.device('default.tensor', wires=2, representation=rep)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            qml.PauliY(wires=1)
            return qml.expval(qml.PauliZ(0)), qml.sample(qml.PauliY(1))

        circuit()
        dev._clear_network_data()

        assert dev._nodes == {}
        assert not dev._contracted
        assert dev._terminal_edges == []


    def test_reset(self, rep, tol):
        """Tests that the `reset` method clears relevant bookkeeping data and re-initializes the initial state."""

        dev = qml.device('default.tensor', wires=2, representation=rep)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            qml.PauliY(wires=1)
            return qml.expval(qml.PauliZ(0)), qml.sample(qml.PauliY(1))

        circuit()
        dev.reset()

        assert 'state' in dev._nodes and len(dev._nodes) == 1
        assert len(dev._nodes['state']) == 2
        assert all([dev._nodes['state'][idx].name == "ZeroState({},)".format(idx) for idx in range(2)])
        assert np.allclose([dev._nodes['state'][idx].tensor for idx in range(2)], dev._zero_state, atol=tol, rtol=0)
        assert not dev._contracted
        assert len(dev._terminal_edges) == 2
        assert edges_valid(dev, num_nodes=2)


    def test_add_initial_state_nodes_2_wires_factorized(self, rep):
        """Tests that factorized initial states are properly created for a 2 wire device."""

        dev = qml.device('default.tensor', wires=2, representation=rep)
        dev._clear_network_data()

        # factorized state
        tensors = [np.array([1., 0.]), np.array([np.sqrt(0.5), -1j * np.sqrt(0.5)])]
        wires = [[0], [1]]
        names = ["AliceState", "BobState"]
        dev._add_initial_state_nodes(tensors, wires, names)

        assert "state" in dev._nodes and len(dev._nodes) == 1
        assert len(dev._nodes["state"]) == 2
        assert dev._nodes["state"][0].name == "AliceState(0,)"
        assert dev._nodes["state"][1].name == "BobState(1,)"
        assert edges_valid(dev, num_nodes=2)


        def test_add_initial_state_nodes_2_wires_entangled(self, rep):
            """Tests that entangled initial states are properly created for a 2 wire device."""
        # entangled state

        dev = qml.device('default.tensor', wires=2, representation=rep)
        dev._clear_network_data()

        tensors = [np.array([[1., 0.], [0., 1.]]) / np.sqrt(2)]
        wires = [[0, 1]]
        names = ["BellState"]
        dev._add_initial_state_nodes(tensors, wires, names)

        assert "state" in dev._nodes and len(dev._nodes) == 1
        assert len(dev._nodes["state"]) == 1
        assert dev._nodes["state"][0].name == "BellState(0, 1)"
        assert edges_valid(dev, num_nodes=1)


    def test_add_initial_state_nodes_3_wires_completely_factorized(self, rep):
        """Tests that completely factorized initial states are properly created for a 3 wire device."""

        dev = qml.device('default.tensor', wires=3, representation=rep)
        dev._clear_network_data()

        tensors = [np.array([1., 0.]), np.array([1, -1j]) / np.sqrt(2), np.array([0., 1.])]
        wires = [[0], [1], [2]]
        names = ["AliceState", "BobState", "CharlieState"]
        dev._add_initial_state_nodes(tensors, wires, names)

        assert "state" in dev._nodes and len(dev._nodes) == 1
        assert len(dev._nodes["state"]) == 3
        assert dev._nodes["state"][0].name == "AliceState(0,)"
        assert dev._nodes["state"][1].name == "BobState(1,)"
        assert dev._nodes["state"][2].name == "CharlieState(2,)"
        assert edges_valid(dev, num_nodes=3)


        def test_add_initial_state_nodes_3_wires_biseparable_AB_C(self, rep):
            """Tests that biseparable AB|C initial states are properly created for a 3 wire device."""

        dev = qml.device('default.tensor', wires=3, representation=rep)
        dev._clear_network_data()

        tensors = [np.array([[1., 0.], [0., 1.]]) / np.sqrt(2), np.array([1., 1.]) / np.sqrt(2)]
        wires = [[0, 1], [2]]
        names = ["AliceBobState", "CharlieState"]
        dev._add_initial_state_nodes(tensors, wires, names)

        assert "state" in dev._nodes and len(dev._nodes) == 1
        assert len(dev._nodes["state"]) == 2
        assert dev._nodes["state"][0].name == "AliceBobState(0, 1)"
        assert dev._nodes["state"][1].name == "CharlieState(2,)"
        assert edges_valid(dev, num_nodes=2)


        def test_add_initial_state_nodes_3_wires_biseparable_A_BC(self, rep):
            """Tests that biseparable A|BC initial states are properly created for a 3 wire device."""

        dev = qml.device('default.tensor', wires=3, representation=rep)
        dev._clear_network_data()

        tensors = [np.array([[1., 0.], [0., 1.]]) / np.sqrt(2), np.array([1., 1.]) / np.sqrt(2)]
        wires = [[0], [1, 2]]
        names = ["AliceState", "BobCharlieState"]
        dev._add_initial_state_nodes(tensors, wires, names)

        assert "state" in dev._nodes and len(dev._nodes) == 1
        assert len(dev._nodes["state"]) == 2
        assert dev._nodes["state"][0].name == "AliceState(0,)"
        assert dev._nodes["state"][1].name == "BobCharlieState(1, 2)"
        assert edges_valid(dev, num_nodes=2)


        def test_add_initial_state_nodes_3_wires_biseparable_AC_B(self, rep):
            """Tests that biseparable AC|B initial states are properly created for a 3 wire device."""

        dev = qml.device('default.tensor', wires=3, representation=rep)
        dev._clear_network_data()

        tensors = [np.array([[1., 0.], [0., 1.]]) / np.sqrt(2), np.array([1., 1.]) / np.sqrt(2)]
        wires = [[0, 2], [1]]
        names = ["AliceCharlieState", "BobState"]
        dev._add_initial_state_nodes(tensors, wires, names)

        assert "state" in dev._nodes and len(dev._nodes) == 1
        assert len(dev._nodes["state"]) == 2
        assert dev._nodes["state"][0].name == "AliceCharlieState(0, 2)"
        assert dev._nodes["state"][1].name == "BobState(1,)"
        assert edges_valid(dev, num_nodes=2)


        def test_add_initial_state_nodes_3_wires_tripartite_entangled(self, rep):
            """Tests that tripartite entangled initial states are properly created for a 3 wire device."""

        dev = qml.device('default.tensor', wires=3, representation=rep)
        dev._clear_network_data()

        tensors = [np.array([[1., 0., 0., 0.],
                             [0., 0., 0., 0.],
                             [0., 0., 0., 0.],
                             [0., 0., 0., 1.]]) / np.sqrt(2)]
        wires = [[0, 1, 2]]
        names = ["GHZState"]
        dev._add_initial_state_nodes(tensors, wires, names)

        assert "state" in dev._nodes and len(dev._nodes) == 1
        assert len(dev._nodes["state"]) == 1
        assert dev._nodes["state"][0].name == "GHZState(0, 1, 2)"
        assert edges_valid(dev, num_nodes=1)


    @pytest.mark.parametrize("tensors,wires,names", [
        ([np.array([[1., 0.], [0., 1.]]) / np.sqrt(2)], [[0,1]], ["A", "B"]),
        ([np.array([[1., 0.], [0., 1.]]) / np.sqrt(2)], [[0], [1]], ["A"]),
        ([np.array([1., 0.]), np.array([1., 1.]) / np.sqrt(2)], [[0]], ["A"]),
    ])
    def test_add_initial_state_nodes_exception(self, rep, tensors, wires, names):
        """Tests that an exception is given if the method _add_initial_state_nodes
        receives arguments with incompatible lengths."""

        dev = qml.device('default.tensor', wires=2, representation=rep)
        dev._clear_network_data()

        with pytest.raises(ValueError, match="must all be the same length"):
            dev._add_initial_state_nodes(tensors, wires, names)


    def test_add_node(self, rep, tol):
        """Tests that the _add_node method adds nodes with the correct attributes."""

        dev = qml.device('default.tensor', wires=2, representation=rep)

        assert len(dev._nodes["state"]) == 2

        zero_state = np.array([1., 0])
        one_qubit_gate = np.array([[0, 1], [1, 0]])
        two_qubit_gate = np.eye(4)
        dev._add_node(one_qubit_gate, wires=[0], name="NewNodeX")
        dev._add_node(one_qubit_gate, wires=[1], name="NewNodeY")
        dev._add_node(two_qubit_gate, wires=[0, 1], name="NewNodeZ")
        assert len(dev._nodes["state"]) == 5
        node_names = [n.name for n in dev._nodes["state"]]
        assert set(node_names) == set(["ZeroState(0,)",
                                       "ZeroState(1,)",
                                       "NewNodeX(0,)",
                                       "NewNodeY(1,)",
                                       "NewNodeZ(0, 1)"])
        tensors = [n.tensor for n in dev._nodes["state"]]
        assert all([np.allclose(t, zero_state, atol=tol, rtol=0) for t in tensors[:2]])
        assert all([np.allclose(t, one_qubit_gate, atol=tol, rtol=0) for t in tensors[2:4]])
        assert np.allclose(tensors[4], two_qubit_gate, atol=tol, rtol=0)


    def test_add_node_creates_keys(self, rep, tol):
        """Tests that the _add_node method is able to create new keys in dev._nodes."""

        dev = qml.device('default.tensor', wires=2, representation=rep)

        assert "state" in dev._nodes and len(dev._nodes) == 1
        dev._add_node(np.array([[0, 1], [1, 0]]), wires=[0], key="junk")
        assert "junk" in dev._nodes and len(dev._nodes) == 2


    def test_create_nodes_from_tensors(self, rep):
        """Tests that the create_nodes_from_tensors method adds nodes to the tensor
        network properly."""

        dev = qml.device('default.tensor', wires=2, representation=rep)

        assert len(dev._nodes["state"]) == 2
        A = np.array([[0, 1], [1, 0]])
        new_node = dev._create_nodes_from_tensors([A], [[0]], ["GateA"], key="state")
        assert new_node[0] in dev._nodes["state"]
        assert len(dev._nodes["state"]) == 3

        new_nodes = dev._create_nodes_from_tensors([A, A], [[0], [1]], ["GateA", "GateB"], key="state")
        assert all([node in dev._nodes["state"] for node in new_nodes])

        obs_nodes = dev._create_nodes_from_tensors([A, A], [[0], [1]], ["ObsA", "ObsB"], key="observables")
        assert all(node in dev._nodes["observables"] for node in obs_nodes)


    @pytest.mark.parametrize("method", ["auto", "greedy", "branch", "optimal"])
    def test_contract_to_ket_correct(self, rep, method, tol):
        """Tests that the _contract_to_ket method contracts down to a single node with the correct tensor."""

        dev = qml.device('default.tensor', wires=3, representation=rep, contraction_method=method)

        dev.apply("PauliX", [0], [])
        dev.apply("Hadamard", [1], [])

        assert "contracted_state" not in dev._nodes
        dev._contract_to_ket()
        assert "contracted_state" in dev._nodes
        cont_state = dev._nodes["contracted_state"]
        dev._contract_to_ket()  # should not change anything
        assert dev._nodes["contracted_state"] == cont_state

        expected = np.outer(np.outer([0., 1.], [1 / np.sqrt(2), 1 / np.sqrt(2)]), [1., 0.]).reshape([2,2,2])

        assert np.allclose(cont_state.tensor, expected, atol=tol, rtol=0)
        assert cont_state.name == "Ket"

    @pytest.mark.parametrize("method", ["auto", "greedy", "branch", "optimal"])
    def test_state(self, rep, method, tol):
        """Tests that the _state method produces the correct state after contraction."""

        dev = qml.device('default.tensor', wires=3, representation=rep)

        dev.apply("PauliX", [0], [])
        dev.apply("Hadamard", [1], [])

        assert "contracted_state" not in dev._nodes
        ket = dev._state()
        assert "contracted_state" in dev._nodes

        expected = np.outer(np.outer([0., 1.], [1 / np.sqrt(2), 1 / np.sqrt(2)]), [1., 0.]).reshape([2,2,2])
        assert np.allclose(ket, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("rep", ("exact", "mps"))
class TestDefaultTensorIntegration:
    """Integration tests for default.tensor. This test ensures it integrates
    properly with the PennyLane interface, in particular QNode."""

    def test_load_tensornet_device(self, rep):
        """Test that the tensor network plugin loads correctly"""

        dev = qml.device("default.tensor", wires=2, representation=rep)
        assert dev.num_wires == 2
        assert dev.shots == 1000
        assert dev.analytic
        assert dev.short_name == "default.tensor"

    def test_args(self, rep):
        """Test that the plugin requires correct arguments"""

        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'wires'"
        ):
            qml.device("default.tensor")

    @pytest.mark.parametrize("gate", set(qml.ops.cv.ops))
    def test_unsupported_gate_error(self, rep, gate):
        """Tests that an error is raised if an unsupported gate is applied"""
        op = getattr(qml.ops, gate)

        if op.num_wires is qml.operation.Wires.Any or qml.operation.Wires.All:
            wires = [0]
        else:
            wires = list(range(op.num_wires))

        dev = qml.device('default.tensor', wires=3, representation=rep)

        @qml.qnode(dev)
        def circuit(*x):
            """Test quantum function"""
            x = prep_par(x, op)
            op(*x, wires=wires)

            return qml.expval(qml.X(0))

        with pytest.raises(
            QuantumFunctionError, match="Device default.tensor is a qubit device; CV operations are not allowed."
        ):
            x = np.random.random([op.num_params])
            circuit(*x)

    @pytest.mark.parametrize("observable", set(qml.ops.cv.obs))
    def test_unsupported_observable_error(self, rep, observable):
        """Test error is raised with unsupported observables"""

        op = getattr(qml.ops, observable)

        if op.num_wires is qml.operation.Wires.Any or qml.operation.Wires.All:
            wires = [0]
        else:
            wires = list(range(op.num_wires))

        dev = qml.device('default.tensor', wires=3, representation=rep)

        @qml.qnode(dev)
        def circuit(*x):
            """Test quantum function"""
            x = prep_par(x, op)
            return qml.expval(op(*x, wires=wires))

        with pytest.raises(
            QuantumFunctionError, match="Device default.tensor is a qubit device; CV operations are not allowed."
        ):
            x = np.random.random([op.num_params])
            circuit(*x)

    def test_qubit_circuit(self, rep, tol):
        """Test that the tensor network plugin provides correct result for a simple circuit"""

        p = 0.543

        dev = qml.device('default.tensor', wires=1, representation=rep)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -np.sin(p)

        assert np.isclose(circuit(p), expected, atol=tol, rtol=0)

    def test_qubit_identity(self, rep, tol):
        """Test that the tensor network plugin provides correct result for the Identity expectation"""

        p = 0.543

        dev = qml.device('default.tensor', wires=1, representation=rep)

        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.Identity(0))

        assert np.isclose(circuit(p), 1, atol=tol, rtol=0)

    # This test is ran against the state |0> with one Z expval
    @pytest.mark.parametrize("name,expected_output", [
        ("PauliX", -1),
        ("PauliY", -1),
        ("PauliZ", 1),
        ("Hadamard", 0),
    ])
    def test_supported_gate_single_wire_no_parameters(self, rep, tol, name, expected_output):
        """Tests supported gates that act on a single wire that are not parameterized"""

        op = getattr(qml.ops, name)

        dev = qml.device('default.tensor', wires=1, representation=rep)

        assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            op(wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran against the state |Phi+> with two Z expvals
    @pytest.mark.parametrize("name,expected_output", [
        ("CNOT", [-1/2, 1]),
        ("SWAP", [-1/2, -1/2]),
        ("CZ", [-1/2, -1/2]),
    ])
    def test_supported_gate_two_wires_no_parameters(self, rep, tol, name, expected_output):
        """Tests supported gates that act on two wires that are not parameterized"""

        op = getattr(qml.ops, name)

        dev = qml.device('default.tensor', wires=2, representation=rep)

        assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(np.array([1/2, 0, 0, math.sqrt(3)/2]), wires=[0, 1])
            op(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,expected_output", [
        ("CSWAP", [-1, -1, 1]),
    ])
    def test_supported_gate_three_wires_no_parameters(self, rep, tol, name, expected_output):
        """Tests supported gates that act on three wires that are not parameterized"""

        if rep == "mps":
            pytest.skip("Three-qubit gates are not supported with MPS representation.")
        dev = qml.device('default.tensor', wires=3, representation=rep)
        op = getattr(qml.ops, name)

        assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2])
            op(wires=[0, 1, 2])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran with two Z expvals
    @pytest.mark.parametrize("name,par,expected_output", [
        ("BasisState", [0, 0], [1, 1]),
        ("BasisState", [1, 0], [-1, 1]),
        ("BasisState", [0, 1], [1, -1]),
        ("QubitStateVector", [1, 0, 0, 0], [1, 1]),
        ("QubitStateVector", [0, 0, 1, 0], [-1, 1]),
        ("QubitStateVector", [0, 1, 0, 0], [1, -1]),
    ])
    def test_supported_state_preparation(self, rep, tol, name, par, expected_output):
        """Tests supported state preparations"""

        op = getattr(qml.ops, name)

        dev = qml.device('default.tensor', wires=2, representation=rep)

        assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            op(np.array(par), wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran on the state |0> with one Z expvals
    @pytest.mark.parametrize("name,par,expected_output", [
        ("PhaseShift", [math.pi/2], 1),
        ("PhaseShift", [-math.pi/4], 1),
        ("RX", [math.pi/2], 0),
        ("RX", [-math.pi/4], 1/math.sqrt(2)),
        ("RY", [math.pi/2], 0),
        ("RY", [-math.pi/4], 1/math.sqrt(2)),
        ("RZ", [math.pi/2], 1),
        ("RZ", [-math.pi/4], 1),
        ("Rot", [math.pi/2, 0, 0], 1),
        ("Rot", [0, math.pi/2, 0], 0),
        ("Rot", [0, 0, math.pi/2], 1),
        ("Rot", [math.pi/2, -math.pi/4, -math.pi/4], 1/math.sqrt(2)),
        ("Rot", [-math.pi/4, math.pi/2, math.pi/4], 0),
        ("Rot", [-math.pi/4, math.pi/4, math.pi/2], 1/math.sqrt(2)),
        ("QubitUnitary", [np.array([[1j/math.sqrt(2), 1j/math.sqrt(2)], [1j/math.sqrt(2), -1j/math.sqrt(2)]])], 0),
        ("QubitUnitary", [np.array([[-1j/math.sqrt(2), 1j/math.sqrt(2)], [1j/math.sqrt(2), 1j/math.sqrt(2)]])], 0),
    ])
    def test_supported_gate_single_wire_with_parameters(self, rep, tol, name, par, expected_output):
        """Tests supported gates that act on a single wire that are parameterized"""

        op = getattr(qml.ops, name)

        dev = qml.device('default.tensor', wires=1, representation=rep)

        assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            op(*par, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran against the state 1/2|00>+sqrt(3)/2|11> with two Z expvals
    @pytest.mark.parametrize("name,par,expected_output", [
        ("CRX", [0], [-1/2, -1/2]),
        ("CRX", [-math.pi], [-1/2, 1]),
        ("CRX", [math.pi/2], [-1/2, 1/4]),
        ("CRY", [0], [-1/2, -1/2]),
        ("CRY", [-math.pi], [-1/2, 1]),
        ("CRY", [math.pi/2], [-1/2, 1/4]),
        ("CRZ", [0], [-1/2, -1/2]),
        ("CRZ", [-math.pi], [-1/2, -1/2]),
        ("CRZ", [math.pi/2], [-1/2, -1/2]),
        ("CRot", [math.pi/2, 0, 0], [-1/2, -1/2]),
        ("CRot", [0, math.pi/2, 0], [-1/2, 1/4]),
        ("CRot", [0, 0, math.pi/2], [-1/2, -1/2]),
        ("CRot", [math.pi/2, 0, -math.pi], [-1/2, -1/2]),
        ("CRot", [0, math.pi/2, -math.pi], [-1/2, 1/4]),
        ("CRot", [-math.pi, 0, math.pi/2], [-1/2, -1/2]),
        ("QubitUnitary", [np.array([[1, 0, 0, 0], [0, 1/math.sqrt(2), 1/math.sqrt(2), 0], [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], [0, 0, 0, 1]])], [-1/2, -1/2]),
        ("QubitUnitary", [np.array([[-1, 0, 0, 0], [0, 1/math.sqrt(2), 1/math.sqrt(2), 0], [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], [0, 0, 0, -1]])], [-1/2, -1/2]),
    ])
    def test_supported_gate_two_wires_with_parameters(self, rep, tol, name, par, expected_output):
        """Tests supported gates that act on two wires wires that are parameterized"""

        op = getattr(qml.ops, name)

        dev = qml.device('default.tensor', wires=2, representation=rep)

        assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(np.array([1/2, 0, 0, math.sqrt(3)/2]), wires=[0, 1])
            op(*par, wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,state,expected_output", [
        ("PauliX", [1/math.sqrt(2), 1/math.sqrt(2)], 1),
        ("PauliX", [1/math.sqrt(2), -1/math.sqrt(2)], -1),
        ("PauliX", [1, 0], 0),
        ("PauliY", [1/math.sqrt(2), 1j/math.sqrt(2)], 1),
        ("PauliY", [1/math.sqrt(2), -1j/math.sqrt(2)], -1),
        ("PauliY", [1, 0], 0),
        ("PauliZ", [1, 0], 1),
        ("PauliZ", [0, 1], -1),
        ("PauliZ", [1/math.sqrt(2), 1/math.sqrt(2)], 0),
        ("Hadamard", [1, 0], 1/math.sqrt(2)),
        ("Hadamard", [0, 1], -1/math.sqrt(2)),
        ("Hadamard", [1/math.sqrt(2), 1/math.sqrt(2)], 1/math.sqrt(2)),
    ])
    def test_supported_observable_single_wire_no_parameters(self, rep, tol, name, state, expected_output):
        """Tests supported observables on single wires without parameters."""

        obs = getattr(qml.ops, name)

        dev = qml.device('default.tensor', wires=1, representation=rep)

        assert dev.supports_observable(name)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0])
            return qml.expval(obs(wires=[0]))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,state,expected_output,par", [
        ("Identity", [1, 0], 1, []),
        ("Identity", [0, 1], 1, []),
        ("Identity", [1/math.sqrt(2), -1/math.sqrt(2)], 1, []),
        ("Hermitian", [1, 0], 1, [np.array([[1, 1j], [-1j, 1]])]),
        ("Hermitian", [0, 1], 1, [np.array([[1, 1j], [-1j, 1]])]),
        ("Hermitian", [1/math.sqrt(2), -1/math.sqrt(2)], 1, [np.array([[1, 1j], [-1j, 1]])]),
    ])
    def test_supported_observable_single_wire_with_parameters(self, rep, tol, name, state, expected_output, par):
        """Tests supported observables on single wires with parameters."""

        obs = getattr(qml.ops, name)

        dev = qml.device('default.tensor', wires=1, representation=rep)

        assert dev.supports_observable(name)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0])
            return qml.expval(obs(*par, wires=[0]))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,state,expected_output,par", [
        ("Hermitian", [1/math.sqrt(3), 0, 1/math.sqrt(3), 1/math.sqrt(3)], 5/3, [np.array([[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]])]),
        ("Hermitian", [0, 0, 0, 1], 0, [np.array([[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])]),
        ("Hermitian", [1/math.sqrt(2), 0, -1/math.sqrt(2), 0], 1, [np.array([[1, 1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, 1j, 1]])]),
        ("Hermitian", [1/math.sqrt(3), -1/math.sqrt(3), 1/math.sqrt(6), 1/math.sqrt(6)], 1, [np.array([[1, 1j, 0, .5j], [-1j, 1, 0, 0], [0, 0, 1, -1j], [-.5j, 0, 1j, 1]])]),
        ("Hermitian", [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], 1, [np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])]),
        ("Hermitian", [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], -1, [np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])]),
    ])
    def test_supported_observable_two_wires_with_parameters(self, rep, tol, name, state, expected_output, par):
        """Tests supported observables on two wires with parameters."""

        obs = getattr(qml.ops, name)

        dev = qml.device('default.tensor', wires=2, representation=rep)

        assert dev.supports_observable(name)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0, 1])
            return qml.expval(obs(*par, wires=[0, 1]))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    def test_expval_warnings(self, rep):
        """Tests that expval raises a warning if the given observable is complex."""

        dev = qml.device("default.tensor", wires=1, representation=rep)

        A = np.array([[2j, 1j], [-3j, 1j]])
        obs_node = dev._create_nodes_from_tensors([A], [[0]], "ComplexObservable", key="observables")

        # text warning raised if matrix is complex
        with pytest.warns(RuntimeWarning, match='Nonvanishing imaginary part'):
            dev.ev(obs_node, wires=[[0]])

    @pytest.mark.parametrize("method", ["auto", "greedy", "branch", "optimal"])
    def test_correct_state_no_params(self, rep, method):
        """Tests that if different QNodes are used with the same device,
        then the contracted state is correct for each one."""
        dev = qml.device('default.tensor', wires=2, representation=rep)
        state = dev._state()

        expected = np.array([[1, 0],
                             [0, 0]])
        assert np.allclose(state, expected)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit()
        state = dev._state()

        expected = np.array([[1, 0],
                             [1, 0]]) / np.sqrt(2)
        assert np.allclose(state, expected)


    @pytest.mark.parametrize("method", ["auto", "greedy", "branch", "optimal"])
    def test_correct_state_diff_params(self, rep, method, tol):
        """Tests that if different inputs are fed to the same QNode,
        then the contracted state is updated correctly."""
        dev = qml.device('default.tensor', wires=2, representation=rep, contraction_method=method)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        def expected(theta):
            vec = np.outer(np.array([np.cos(theta / 2), -1j * np.sin(theta / 2)]), [1., 0.])
            return vec.reshape([2, 2])

        theta = np.pi / 4
        out1 = circuit(theta)
        ket1 = dev._state()
        assert "contracted_state" in dev._nodes
        assert np.allclose(ket1, expected(theta), atol=tol, rtol=0)
        assert out1 == np.cos(theta / 2) ** 2 - np.sin(theta / 2) ** 2

        theta = -0.1234
        out2 = circuit(theta)
        ket2 = dev._state()
        assert "contracted_state" in dev._nodes
        assert np.allclose(ket2, expected(theta), atol=tol, rtol=0)
        assert out2 == np.cos(theta / 2) ** 2 - np.sin(theta / 2) ** 2


@pytest.mark.parametrize("rep", ("exact", "mps"))
@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorExpval:
    """Test tensor expectation values"""

    def test_paulix_pauliy(self, theta, phi, varphi, rep, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.tensor", wires=3, representation=rep)
        dev.reset()

        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("RX", wires=[2], par=[varphi])
        dev.apply("CNOT", wires=[0, 1], par=[])
        dev.apply("CNOT", wires=[1, 2], par=[])

        res = dev.expval(["PauliX", "PauliY"], [[0], [2]], [[], []])
        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_identity(self, theta, phi, varphi, rep, tol):
        """Test that a tensor product involving PauliZ and Identity works correctly"""
        dev = qml.device("default.tensor", wires=3, representation=rep)
        dev.reset()
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("RX", wires=[2], par=[varphi])
        dev.apply("CNOT", wires=[0, 1], par=[])
        dev.apply("CNOT", wires=[1, 2], par=[])

        res = dev.expval(["PauliZ", "Identity", "PauliZ"], [[0], [1], [2]], [[], [], []])
        expected = np.cos(varphi)*np.cos(phi)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_hadamard(self, theta, phi, varphi, rep, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qml.device("default.tensor", wires=3, representation=rep)
        dev.reset()
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("RX", wires=[2], par=[varphi])
        dev.apply("CNOT", wires=[0, 1], par=[])
        dev.apply("CNOT", wires=[1, 2], par=[])

        res = dev.expval(["PauliZ", "Hadamard", "PauliY"], [[0], [1], [2]], [[], [], []])
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian(self, theta, phi, varphi, rep, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("default.tensor", wires=3, representation=rep)
        dev.reset()
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("RX", wires=[2], par=[varphi])
        dev.apply("CNOT", wires=[0, 1], par=[])
        dev.apply("CNOT", wires=[1, 2], par=[])

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        res = dev.expval(["PauliZ", "Hermitian"], [[0], [1, 2]], [[], [A]])
        expected = 0.5 * (
            -6 * np.cos(theta) * (np.cos(varphi) + 1)
            - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
            + 3 * np.cos(varphi) * np.sin(phi)
            + np.sin(phi)
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian_hermitian(self, theta, phi, varphi, rep, tol):
        """Test that a tensor product involving two Hermitian matrices works correctly"""
        dev = qml.device("default.tensor", wires=3, representation=rep)
        dev.reset()
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("RX", wires=[2], par=[varphi])
        dev.apply("CNOT", wires=[0, 1], par=[])
        dev.apply("CNOT", wires=[1, 2], par=[])

        A1 = np.array([[1, 2],
                       [2, 4]])

        A2 = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        res = dev.expval(["Hermitian", "Hermitian"], [[0], [1, 2]], [[A1], [A2]])
        expected = 0.25 * (
            -30
            + 4 * np.cos(phi) * np.sin(theta)
            + 3 * np.cos(varphi) * (-10 + 4 * np.cos(phi) * np.sin(theta) - 3 * np.sin(phi))
            - 3 * np.sin(phi)
            - 2 * (5 + np.cos(phi) * (6 + 4 * np.sin(theta)) + (-3 + 8 * np.sin(theta)) * np.sin(phi))
            * np.sin(varphi)
            + np.cos(theta)
            * (
                18
                + 5 * np.sin(phi)
                + 3 * np.cos(varphi) * (6 + 5 * np.sin(phi))
                + 2 * (3 + 10 * np.cos(phi) - 5 * np.sin(phi)) * np.sin(varphi)
            )
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian_identity_expectation(self, theta, phi, varphi, rep, tol):
        """Test that a tensor product involving an Hermitian matrix and the identity works correctly"""
        dev = qml.device("default.tensor", wires=2, representation=rep)
        dev.reset()
        dev.apply("RY", wires=[0], par=[theta])
        dev.apply("RY", wires=[1], par=[phi])
        dev.apply("CNOT", wires=[0, 1], par=[])

        A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])

        res = dev.expval(["Hermitian", "Identity"], [[0], [1]], [[A], []])

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        expected = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2

        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("rep", ("exact", "mps"))
@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorVar:
    """Test tensor variances"""

    def test_paulix_pauliy(self, theta, phi, varphi, rep, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.tensor", wires=3, representation=rep)
        dev.reset()
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("RX", wires=[2], par=[varphi])
        dev.apply("CNOT", wires=[0, 1], par=[])
        dev.apply("CNOT", wires=[1, 2], par=[])

        res = dev.var(["PauliX", "PauliY"], [[0], [2]], [[], [], []])

        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_hadamard(self, theta, phi, varphi, rep, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qml.device("default.tensor", wires=3, representation=rep)
        dev.reset()
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("RX", wires=[2], par=[varphi])
        dev.apply("CNOT", wires=[0, 1], par=[])
        dev.apply("CNOT", wires=[1, 2], par=[])

        res = dev.var(["PauliZ", "Hadamard", "PauliY"], [[0], [1], [2]], [[], [], []])

        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian(self, theta, phi, varphi, rep, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("default.tensor", wires=3, representation=rep)
        dev.reset()
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("RX", wires=[2], par=[varphi])
        dev.apply("CNOT", wires=[0, 1], par=[])
        dev.apply("CNOT", wires=[1, 2], par=[])

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        res = dev.var(["PauliZ", "Hermitian"], [[0], [1, 2]], [[], [A]])

        expected = (
            1057
            - np.cos(2 * phi)
            + 12 * (27 + np.cos(2 * phi)) * np.cos(varphi)
            - 2 * np.cos(2 * varphi) * np.sin(phi) * (16 * np.cos(phi) + 21 * np.sin(phi))
            + 16 * np.sin(2 * phi)
            - 8 * (-17 + np.cos(2 * phi) + 2 * np.sin(2 * phi)) * np.sin(varphi)
            - 8 * np.cos(2 * theta) * (3 + 3 * np.cos(varphi) + np.sin(varphi)) ** 2
            - 24 * np.cos(phi) * (np.cos(phi) + 2 * np.sin(phi)) * np.sin(2 * varphi)
            - 8
            * np.cos(theta)
            * (
                4
                * np.cos(phi)
                * (
                    4
                    + 8 * np.cos(varphi)
                    + np.cos(2 * varphi)
                    - (1 + 6 * np.cos(varphi)) * np.sin(varphi)
                )
                + np.sin(phi)
                * (
                    15
                    + 8 * np.cos(varphi)
                    - 11 * np.cos(2 * varphi)
                    + 42 * np.sin(varphi)
                    + 3 * np.sin(2 * varphi)
                )
            )
        ) / 16

        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("rep", ("exact", "mps"))
class TestSample:
    """Tests that samples are properly calculated."""

    def test_sample_dimensions(self, rep):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """

        dev = qml.device('default.tensor', wires=2, representation=rep)

        dev.apply('RX', wires=[0], par=[1.5708])
        dev.apply('RX', wires=[1], par=[1.5708])

        dev.shots = 10
        s1 = dev.sample('PauliZ', [0], [])
        assert np.array_equal(s1.shape, (10,))

        dev.shots = 12
        s2 = dev.sample('PauliZ', [1], [])
        assert np.array_equal(s2.shape, (12,))

        dev.shots = 17
        s3 = dev.sample('CZ', [0, 1], [])
        assert np.array_equal(s3.shape, (17,))

    def test_sample_values(self, rep, tol):
        """Tests if the samples returned by sample have
        the correct values
        """
        dev = qml.device('default.tensor', wires=2, representation=rep)
        dev.reset()

        dev.apply('RX', wires=[0], par=[1.5708])

        s1 = dev.sample('PauliZ', [0], [])

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)


@pytest.mark.parametrize("rep", ("exact", "mps"))
@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorSample:
    """Test samped values from tensor product observables"""

    def test_paulix_pauliy(self, theta, phi, varphi, monkeypatch, rep, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.tensor", wires=3, shots=10000, representation=rep)
        dev.reset()
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("RX", wires=[2], par=[varphi])
        dev.apply("CNOT", wires=[0, 1], par=[])
        dev.apply("CNOT", wires=[1, 2], par=[])

        with monkeypatch.context() as m:
            m.setattr("numpy.random.choice", lambda x, y, p: (x, p))
            s1, prob = dev.sample(["PauliX", "PauliY"], [[0], [2]], [[], [], []])

        # s1 should only contain 1 and -1
        assert np.allclose(s1 ** 2, 1, atol=tol, rtol=0)

        mean = s1 @ prob
        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)
        assert np.allclose(mean, expected, atol=tol, rtol=0)

        var = (s1 ** 2) @ prob - (s1 @ prob).real ** 2
        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16
        assert np.allclose(var, expected, atol=tol, rtol=0)

    def test_pauliz_hadamard(self, theta, phi, varphi, monkeypatch, rep, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qml.device("default.tensor", wires=3, representation=rep)
        dev.reset()
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("RX", wires=[2], par=[varphi])
        dev.apply("CNOT", wires=[0, 1], par=[])
        dev.apply("CNOT", wires=[1, 2], par=[])

        with monkeypatch.context() as m:
            m.setattr("numpy.random.choice", lambda x, y, p: (x, p))
            s1, p = dev.sample(["PauliZ", "Hadamard", "PauliY"], [[0], [1], [2]], [[], [], []])

        # s1 should only contain 1 and -1
        assert np.allclose(s1 ** 2, 1, atol=tol, rtol=0)

        mean = s1 @ p
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)
        assert np.allclose(mean, expected, atol=tol, rtol=0)

        var = (s1 ** 2) @ p - (s1 @ p).real ** 2
        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4
        assert np.allclose(var, expected, atol=tol, rtol=0)

    def test_hermitian(self, theta, phi, varphi, monkeypatch, rep, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("default.tensor", wires=3, representation=rep)
        dev.reset()
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("RX", wires=[2], par=[varphi])
        dev.apply("CNOT", wires=[0, 1], par=[])
        dev.apply("CNOT", wires=[1, 2], par=[])

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        with monkeypatch.context() as m:
            m.setattr("numpy.random.choice", lambda x, y, p: (x, p))
            s1, p = dev.sample(["PauliZ", "Hermitian"], [[0], [1, 2]], [[], [A]])

        # s1 should only contain the eigenvalues of
        # the hermitian matrix tensor product Z
        Z = np.diag([1, -1])
        eigvals = np.linalg.eigvalsh(np.kron(Z, A))
        assert set(np.round(s1, 8)).issubset(set(np.round(eigvals, 8)))

        mean = s1 @ p
        expected = 0.5 * (
            -6 * np.cos(theta) * (np.cos(varphi) + 1)
            - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
            + 3 * np.cos(varphi) * np.sin(phi)
            + np.sin(phi)
        )
        assert np.allclose(mean, expected, atol=tol, rtol=0)

        var = (s1 ** 2) @ p - (s1 @ p).real ** 2
        expected = (
            1057
            - np.cos(2 * phi)
            + 12 * (27 + np.cos(2 * phi)) * np.cos(varphi)
            - 2 * np.cos(2 * varphi) * np.sin(phi) * (16 * np.cos(phi) + 21 * np.sin(phi))
            + 16 * np.sin(2 * phi)
            - 8 * (-17 + np.cos(2 * phi) + 2 * np.sin(2 * phi)) * np.sin(varphi)
            - 8 * np.cos(2 * theta) * (3 + 3 * np.cos(varphi) + np.sin(varphi)) ** 2
            - 24 * np.cos(phi) * (np.cos(phi) + 2 * np.sin(phi)) * np.sin(2 * varphi)
            - 8
            * np.cos(theta)
            * (
                4
                * np.cos(phi)
                * (
                    4
                    + 8 * np.cos(varphi)
                    + np.cos(2 * varphi)
                    - (1 + 6 * np.cos(varphi)) * np.sin(varphi)
                )
                + np.sin(phi)
                * (
                    15
                    + 8 * np.cos(varphi)
                    - 11 * np.cos(2 * varphi)
                    + 42 * np.sin(varphi)
                    + 3 * np.sin(2 * varphi)
                )
            )
        ) / 16
        assert np.allclose(var, expected, atol=tol, rtol=0)
