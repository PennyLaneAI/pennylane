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
from scipy.stats import unitary_group

import pennylane as qml
from pennylane.transforms.qmc import _apply_controlled_z, _apply_controlled_v, apply_controlled_Q
from pennylane.templates.subroutines.qmc import _make_V, _make_Z, make_Q


def get_unitary(circ, n_wires):
    """Helper function to find unitary of a circuit"""
    dev = qml.device("default.qubit", wires=range(n_wires))

    @qml.qnode(dev)
    def unitary_z(basis_state):
        qml.BasisState(basis_state, wires=range(n_wires))
        circ()
        return qml.state()

    bitstrings = list(itertools.product([0, 1], repeat=n_wires))
    u = [unitary_z(bitstring).numpy() for bitstring in bitstrings]
    u = np.array(u).T
    return u


@pytest.mark.parametrize("n_wires", range(2, 5))
def test_apply_controlled_z(n_wires):
    """Test if the _apply_controlled_z performs the correct transformation by reconstructing the
    unitary and comparing against the one provided in _make_Z."""
    n_all_wires = n_wires + 1

    wires = range(n_wires)
    control_wire = n_wires
    work_wires = None

    circ = lambda: _apply_controlled_z(
        wires=wires, control_wire=control_wire, work_wires=work_wires
    )
    u = get_unitary(circ, n_all_wires)

    # Note the sign flip in the following. The sign does not matter when performing the Q unitary
    # because two Zs are used.
    z_ideal = -_make_Z(2 ** n_wires)

    circ = lambda: qml.ControlledQubitUnitary(z_ideal, wires=wires, control_wires=control_wire)
    u_ideal = get_unitary(circ, n_all_wires)

    assert np.allclose(u, u_ideal)


@pytest.mark.parametrize("n_wires", range(2, 5))
def test_apply_controlled_v(n_wires):
    """Test if the _apply_controlled_v performs the correct transformation by reconstructing the
    unitary and comparing against the one provided in _make_V."""
    n_all_wires = n_wires + 1

    wires = range(n_wires)
    control_wire = n_wires

    circ = lambda: _apply_controlled_v(target_wire=n_wires - 1, control_wire=control_wire)
    u = get_unitary(circ, n_all_wires)

    # Note the sign flip in the following. The sign does not matter when performing the Q unitary
    # because two Vs are used.
    v_ideal = -_make_V(2 ** n_wires)

    circ = lambda: qml.ControlledQubitUnitary(v_ideal, wires=wires, control_wires=control_wire)
    u_ideal = get_unitary(circ, n_all_wires)

    assert np.allclose(u, u_ideal)


@pytest.mark.parametrize("n_wires", range(2, 5))
def test_apply_controlled_Q(n_wires):
    """Test if the apply_controlled_Q performs the correct transformation by reconstructing the
    unitary and comparing against the one provided in make_Q. Random unitaries are chosen for
    a_mat and r_mat."""
    n_all_wires = n_wires + 1

    wires = range(n_wires)
    target_wire = n_wires - 1
    control_wire = n_wires

    a_mat = unitary_group.rvs(2 ** (n_wires - 1), random_state=1967)
    r_mat = unitary_group.rvs(2 ** n_wires, random_state=1967)
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


def test_apply_controlled_Q_raises():
    """Tests if a ValueError is raised when the target wire is not contained within wires"""
    with pytest.raises(ValueError, match="The target wire must be contained within wires"):
        apply_controlled_Q(
            lambda: ..., wires=range(3), target_wire=4, control_wire=5, work_wires=None
        )
