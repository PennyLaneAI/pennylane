# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the quantum_monte_carlo transform"""
import itertools

import numpy as np
import pytest
from scipy.stats import unitary_group, norm

import pennylane as qml
from pennylane.transforms.qmc import (
    _apply_controlled_z,
    _apply_controlled_v,
    apply_controlled_Q,
    quantum_monte_carlo,
)
from pennylane.templates.subroutines.qmc import _make_V, _make_Z, make_Q
from pennylane.wires import Wires


def r_unitary(gate, alpha, control_wires, target_wire):
    r"""Applies a uniformly-controlled rotation to the target qubit.

    A uniformly-controlled rotation is a sequence of multi-controlled
    rotations, each of which is conditioned on the control qubits being in a different state.

    For example, a uniformly-controlled rotation with two control qubits describes a sequence of
    four multi-controlled rotations, each applying the rotation only if the control qubits
    are in states :math:`|00\rangle`, :math:`|01\rangle`, :math:`|10\rangle`, and :math:`|11\rangle`, respectively.

    To implement a uniformly-controlled rotation using single qubit rotations and CNOT gates,
    a decomposition based on Gray codes is used. For this purpose, the multi-controlled rotation
    angles alpha have to be converted into a set of non-controlled rotation angles theta.
    For more details, see `Möttönen and Vartiainen (2005), Fig 7a<https://arxiv.org/pdf/quant-ph/0504100.pdf>`_.

    Args:
        gate (.Operation): gate to be applied, needs to have exactly one parameter
        alpha (tensor_like): angles to decompose the uniformly-controlled rotation into multi-controlled rotations
        control_wires (array[int]): wires that act as control
        target_wire (int): wire that acts as target
    """

    theta = qml.templates.state_preparations.mottonen.compute_theta(alpha)

    gray_code_rank = len(control_wires)

    if gray_code_rank == 0:
        if qml.math.all(theta[..., 0] != 0.0):
            gate(theta[..., 0], wires=[target_wire])
        return

    code = qml.templates.state_preparations.mottonen.gray_code(gray_code_rank)
    num_selections = len(code)

    control_indices = [
        int(np.log2(int(code[i], 2) ^ int(code[(i + 1) % num_selections], 2)))
        for i in range(num_selections)
    ]

    for i, control_index in enumerate(control_indices):
        if qml.math.all(theta[..., i] != 0.0):
            gate(theta[..., i], wires=[target_wire])
        qml.CNOT(wires=[control_wires[control_index], target_wire])


def get_unitary(circ, n_wires):
    """Helper function to find unitary of a circuit"""
    dev = qml.device("default.qubit", wires=range(n_wires))

    @qml.qnode(dev)
    def unitary_z(basis_state):
        qml.BasisState(basis_state, wires=range(n_wires))
        circ()
        return qml.state()

    bitstrings = list(itertools.product([0, 1], repeat=n_wires))
    u = [unitary_z(np.array(bitstring)).numpy() for bitstring in bitstrings]
    u = np.array(u).T
    return u


@pytest.mark.parametrize("n_wires", range(2, 5))
def test_apply_controlled_z(n_wires):
    """Test if the _apply_controlled_z performs the correct transformation by reconstructing the
    unitary and comparing against the one provided in _make_Z."""
    n_all_wires = n_wires + 1

    wires = Wires(range(n_wires))
    control_wire = n_wires
    work_wires = None

    circ = lambda: _apply_controlled_z(
        wires=wires, control_wire=control_wire, work_wires=work_wires
    )
    u = get_unitary(circ, n_all_wires)

    # Note the sign flip in the following. The sign does not matter when performing the Q unitary
    # because two Zs are used.
    z_ideal = -_make_Z(2**n_wires)

    circ = lambda: qml.ControlledQubitUnitary(z_ideal, wires=wires, control_wires=control_wire)
    u_ideal = get_unitary(circ, n_all_wires)

    assert np.allclose(u, u_ideal)


@pytest.mark.parametrize("n_wires", range(2, 5))
def test_apply_controlled_v(n_wires):
    """Test if the _apply_controlled_v performs the correct transformation by reconstructing the
    unitary and comparing against the one provided in _make_V."""
    n_all_wires = n_wires + 1

    wires = Wires(range(n_wires))
    control_wire = Wires(n_wires)

    circ = lambda: _apply_controlled_v(target_wire=Wires([n_wires - 1]), control_wire=control_wire)
    u = get_unitary(circ, n_all_wires)

    # Note the sign flip in the following. The sign does not matter when performing the Q unitary
    # because two Vs are used.
    v_ideal = -_make_V(2**n_wires)

    circ = lambda: qml.ControlledQubitUnitary(v_ideal, wires=wires, control_wires=control_wire)
    u_ideal = get_unitary(circ, n_all_wires)

    assert np.allclose(u, u_ideal)


class TestApplyControlledQ:
    """Tests for the apply_controlled_Q function"""

    @pytest.mark.slow
    @pytest.mark.parametrize("n_wires", range(2, 5))
    def test_apply(self, n_wires):
        """Test if the apply_controlled_Q performs the correct transformation by reconstructing the
        unitary and comparing against the one provided in make_Q. Random unitaries are chosen for
        a_mat and r_mat."""
        n_all_wires = n_wires + 1

        wires = range(n_wires)
        target_wire = n_wires - 1
        control_wire = n_wires

        a_mat = unitary_group.rvs(2 ** (n_wires - 1), random_state=1967)
        r_mat = unitary_group.rvs(2**n_wires, random_state=1967)
        q_mat = make_Q(a_mat, r_mat)

        def fn():
            qml.QubitUnitary(a_mat, wires=wires[:-1])
            qml.QubitUnitary(r_mat, wires=wires)

        circ = apply_controlled_Q(
            fn, wires=wires, target_wire=target_wire, control_wire=control_wire, work_wires=None
        )

        u = get_unitary(circ, n_all_wires)

        circ = lambda: qml.ControlledQubitUnitary(q_mat, wires=wires, control_wires=control_wire)
        u_ideal = get_unitary(circ, n_all_wires)

        assert np.allclose(u_ideal, u)

    def test_raises(self):
        """Tests if a ValueError is raised when the target wire is not contained within wires"""
        with pytest.raises(ValueError, match="The target wire must be contained within wires"):
            apply_controlled_Q(
                lambda: ..., wires=range(3), target_wire=4, control_wire=5, work_wires=None
            )


class TestQuantumMonteCarlo:
    """Tests for the quantum_monte_carlo function"""

    @pytest.mark.slow
    @pytest.mark.parametrize("n_wires", range(2, 4))
    def test_apply(self, n_wires):
        """Test if the quantum_monte_carlo performs the correct transformation by reconstructing the
        unitary and comparing against the one provided in the QuantumPhaseEstimation template.
        Random unitaries are chosen for a_mat and r_mat."""
        n_all_wires = 2 * n_wires

        wires = range(n_wires)
        target_wire = n_wires - 1
        estimation_wires = range(n_wires, 2 * n_wires)

        a_mat = unitary_group.rvs(2 ** (n_wires - 1), random_state=1967)
        r_mat = unitary_group.rvs(2**n_wires, random_state=1967)
        q_mat = make_Q(a_mat, r_mat)

        def fn():
            qml.QubitUnitary(a_mat, wires=wires[:-1])
            qml.QubitUnitary(r_mat, wires=wires)

        circ = quantum_monte_carlo(
            fn, wires=wires, target_wire=target_wire, estimation_wires=estimation_wires
        )

        u = get_unitary(circ, n_all_wires)

        def circ_ideal():
            fn()
            qml.templates.QuantumPhaseEstimation(
                q_mat, target_wires=wires, estimation_wires=estimation_wires
            )

        u_ideal = get_unitary(circ_ideal, n_all_wires)
        assert np.allclose(u_ideal, u)

    def test_shared_wires(self):
        """Test if a ValueError is raised when the wires and estimation_wires share a common wire"""
        wires = range(2)
        estimation_wires = range(1, 3)

        with pytest.raises(ValueError, match="No wires can be shared between the wires"):
            quantum_monte_carlo(
                lambda: None, wires=wires, target_wire=0, estimation_wires=estimation_wires
            )

    @pytest.mark.slow
    def test_integration(self):
        """Test if quantum_monte_carlo generates the correct circuit by comparing it to the
        QuantumMonteCarlo template on the practical example specified in the usage details. Custom
        wire labels are also used."""

        m = 5  # number of wires in A
        M = 2**m

        xmax = np.pi  # bound to region [-pi, pi]
        xs = np.linspace(-xmax, xmax, M)

        probs = np.array([norm().pdf(x) for x in xs])
        probs /= np.sum(probs)

        func = lambda i: np.sin(xs[i]) ** 2
        r_rotations = np.array([2 * np.arcsin(np.sqrt(func(i))) for i in range(M)])

        A_wires = [0, "a", -1.1, -10, "bbb"]
        target_wire = "Ancilla"
        wires = A_wires + [target_wire]
        estimation_wires = ["bob", -3, 42, "penny", "lane"]

        def fn():
            qml.templates.MottonenStatePreparation(np.sqrt(probs), wires=A_wires)
            r_unitary(qml.RY, r_rotations, control_wires=A_wires[::-1], target_wire=target_wire)

        qmc_circuit = qml.quantum_monte_carlo(
            fn, wires=wires, target_wire=target_wire, estimation_wires=estimation_wires
        )

        with qml.tape.QuantumTape() as tape:
            qmc_circuit()
            qml.probs(estimation_wires)

        tape = tape.expand()

        for op in tape.operations:
            unexpanded = (
                isinstance(op, qml.MultiControlledX)
                or isinstance(op, qml.templates.QFT)
                or isinstance(op, qml.tape.QuantumTape)
            )
            assert not unexpanded

        dev = qml.device("default.qubit", wires=wires + estimation_wires)
        res = dev.execute(tape)

        @qml.qnode(dev)
        def circuit():
            qml.templates.QuantumMonteCarlo(
                probs, func, target_wires=wires, estimation_wires=estimation_wires
            )
            return qml.probs(estimation_wires)

        res_expected = circuit()
        assert np.allclose(res, res_expected)
