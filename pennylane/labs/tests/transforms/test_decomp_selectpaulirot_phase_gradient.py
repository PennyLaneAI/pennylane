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

"""Tests for the decomposition rule qp.labs.transforms.make_selectpaulirot_to_phase_gradient_decomp"""

import numpy as np

# pylint: disable=no-value-for-parameter
import pytest

import pennylane as qp
from pennylane.labs.transforms.decomp_selectpaulirot_phase_gradient import (
    make_selectpaulirot_to_phase_gradient_decomp,
)
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.wires import WireError


@pytest.mark.parametrize(
    "n_angle_wires, n_phase_grad_wires, n_work_wires, msg_match",
    [
        [5, 3, 2, "angle_wires and phase_grad wires must be of same size"],
        [3, 4, 2, "angle_wires and phase_grad wires must be of same size"],
        [4, 4, 2, "work_wires need to be at least of size phase_grad_wires - 1"],
    ],
)
def test_wires_error(n_angle_wires, n_phase_grad_wires, n_work_wires, msg_match):
    """Test WireError is raised correctly"""
    angle_wires = qp.wires.Wires([f"ang_{i}" for i in range(n_angle_wires)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(n_phase_grad_wires)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(n_work_wires)])

    with pytest.raises(WireError, match=msg_match):
        _ = make_selectpaulirot_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires)


@pytest.mark.parametrize("prec", [2, 3, 5])
@pytest.mark.parametrize("num_controls", [1, 2])
def test_valid_decomp(prec, num_controls):
    """Test that the decomposition rule from make_selectpaulirot_to_phase_gradient_decomp works as expected
    as a fixed decomposition and yields the correct resources"""

    angles = (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
        @ np.array([1 / 2, 1 / 4, 1 / 8])
        * 4
        * np.pi
    )[: 2**num_controls]

    # If precision is very low, the number of control wires of the multiplexer dictate the
    # required number of work wires.
    num_work_wires = max(prec, num_controls + 1) - 1

    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(num_work_wires)])

    custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires
    )

    op = qp.SelectPauliRot(angles, control_wires=range(num_controls), target_wire=num_controls)
    _test_decomposition_rule(op, custom_decomp)


# @pytest.mark.usefixtures("enable_graph_decomposition") # fixture doesnt exist in labs tests
@pytest.mark.parametrize("prec", [2, 3, 5])
@pytest.mark.parametrize("num_controls", [1, 2])
def test_as_fixed_decomps(prec, num_controls):
    """Test that the decomposition rule from make_selectpaulirot_to_phase_gradient_decomp works as expected
    as a fixed decomposition and yields the correct resources"""
    with qp.decomposition.toggle_graph_ctx(
        True
    ):  # safe alternative to avoid enabling graph globally on the labs test runner

        angles = (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
            @ np.array([1 / 2, 1 / 4, 1 / 8])
            * 4
            * np.pi
        )[: 2**num_controls]

        # If precision is very low, the number of control wires of the multiplexer dictate the
        # required number of work wires.
        num_work_wires = max(prec, num_controls + 1) - 1

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
        phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(num_work_wires)])

        custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
            angle_wires, phase_grad_wires, work_wires
        )

        @qp.transforms.decompose(
            gate_set={
                "QROM",
                "Adjoint(QROM)",
                "SemiAdder",
                "CNOT",
                "X",
                "Adjoint(X)",
                "GlobalPhase",
            },
            fixed_decomps={qp.SelectPauliRot: custom_decomp},
        )
        @qp.qnode(qp.device("null.qubit"))
        def circuit(angles):
            qp.SelectPauliRot(angles, control_wires=range(num_controls), target_wire=num_controls)
            return qp.state()

        specs = qp.specs(circuit)(angles)["resources"].gate_types
        expected_specs = {
            "QROM": 1,
            "Adjoint(QROM)": 1,
            "CNOT": 2 * prec,
            "PauliX": 2 * prec,
            "SemiAdder": 1,
        }
        assert expected_specs == specs


def test_integration_multi_wire(seed):
    """
    Tests that the decomposition correctly realizes the phase gradient decomposition of SelectPauliRot as described in
    https://pennylane.ai/compilation/phase-gradient/d-multiplex-rotations
    """

    with qp.decomposition.toggle_graph_ctx(
        True
    ):  # safe alternative to avoid enabling graph globally on the labs test runner
        prec = 3

        wires = [0, 1, 2]
        angles = (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
            @ np.array([1 / 2, 1 / 4, 1 / 8])
            * 4
            * np.pi
        )

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
        phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(prec - 1)])

        phase_grad_state = np.exp(-1j * 2 * np.pi * np.arange(2**3) / 2**3) / np.sqrt(2**3)

        all_wires = angle_wires + phase_grad_wires + work_wires + wires

        custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
            angle_wires, phase_grad_wires, work_wires
        )

        @qp.transforms.decompose(
            gate_set={
                "QROM",
                "Adjoint(QROM)",
                "SemiAdder",
                "CNOT",
                "X",
                "Adjoint(X)",
                "StatePrep",
                "Adjoint(StatePrep)",
                "GlobalPhase",
            },
            fixed_decomps={qp.SelectPauliRot: custom_decomp},
        )
        @qp.qnode(qp.device("default.qubit", wires=all_wires))
        def circuit(angles, in_state):
            qp.StatePrep(in_state, wires=wires)  # input state
            qp.StatePrep(phase_grad_state, wires=phase_grad_wires)  # phase gradient state
            qp.SelectPauliRot(angles, control_wires=wires[:2], target_wire=wires[2])
            qp.adjoint(
                qp.StatePrep(phase_grad_state, wires=phase_grad_wires)
            )  # uncompute phase gradient state
            return qp.state()

        # random input state
        rng = np.random.default_rng(seed=seed)
        in_state = rng.random(2 ** len(wires))
        in_state /= np.linalg.norm(in_state)

        # returned output state
        out_state = circuit(angles, in_state)

        # expected output state
        zeros = np.eye(2 ** (prec * 3 - 1))[0]  # |000> on all the aux wires
        out_state_expected = (
            qp.matrix(qp.SelectPauliRot(angles, control_wires=wires[:2], target_wire=wires[2]))
            @ in_state
        )
        out_state_expected = np.kron(zeros, out_state_expected)

        assert np.allclose(out_state, out_state_expected)
