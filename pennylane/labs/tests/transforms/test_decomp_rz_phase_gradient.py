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

"""Tests for the transform ``qp.transform.rz_phase_gradient_decompmake_rz_to_phase_gradient_decomp``"""
import numpy as np

# pylint: disable=no-value-for-parameter
import pytest

import pennylane as qp
from pennylane.labs.transforms.decomp_rz_phase_gradient import make_rz_to_phase_gradient_decomp


# @pytest.mark.usefixtures("enable_graph_decomposition") # fixture doesnt exist in labs tests
@pytest.mark.parametrize("phi", [0.5, 0.3, 1 / 2 + 1 / 4 + 1 / 8, 1.0])
@pytest.mark.parametrize("p", [2, 3, 4])
def test_as_fixed_decomps(phi, p):
    """Test that the decomposition rule from make_rz_to_phase_gradient_decomp works as expected
    as a fixed decomposition and yields the correct resources"""
    with qp.decomposition.toggle_graph_ctx(
        True
    ):  # safe alternative to avoid enabling graph globally on the labs test runner

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(p)])
        phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(p)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(p - 1)])

        kwargs = {
            "angle_wires": angle_wires,
            "phase_grad_wires": phase_grad_wires,
            "work_wires": work_wires,
        }

        custom_decomp = make_rz_to_phase_gradient_decomp(**kwargs)
        gate_set = {"SemiAdder", "C(BasisEmbedding)", "GlobalPhase"}

        @qp.transforms.decompose(gate_set=gate_set, fixed_decomps={qp.RZ: custom_decomp})
        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            qp.RZ(phi, 0)
            return qp.state()

        specs = qp.specs(circuit)()["resources"].gate_types
        expected_specs = {"GlobalPhase": 1, "SemiAdder": 1, "C(BasisEmbedding)": 2}
        assert expected_specs == specs


@pytest.mark.parametrize("phi", [0.5, 0.3, 1 / 2 + 1 / 4 + 1 / 8, 1.0])
@pytest.mark.parametrize("p", [2, 3, 4])
def test_as_alt_decomps(phi, p):
    """Test that the decomposition rule from make_rz_to_phase_gradient_decomp works as expected as an alternative decomposition and yields the correct resources"""
    with qp.decomposition.toggle_graph_ctx(
        True
    ):  # safe alternative to avoid enabling graph globally on the labs test runner

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(p)])
        phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(p)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(p - 1)])

        kwargs = {
            "angle_wires": angle_wires,
            "phase_grad_wires": phase_grad_wires,
            "work_wires": work_wires,
        }

        custom_decomp = make_rz_to_phase_gradient_decomp(**kwargs)
        gate_set = {"SemiAdder", "C(BasisEmbedding)", "GlobalPhase"}

        @qp.transforms.decompose(gate_set=gate_set, alt_decomps={qp.RZ: [custom_decomp]})
        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            qp.RZ(phi, 0)
            return qp.state()

        specs = qp.specs(circuit)()["resources"].gate_types
        expected_specs = {"GlobalPhase": 1, "SemiAdder": 1, "C(BasisEmbedding)": 2}
        assert expected_specs == specs


def test_integration_multi_wire(seed):
    """
    Tests that the decomposition correctly realizes the phase gradient decomposition of SelectPauliRot as described in
    https://pennylane.ai/compilation/phase-gradient/d-multiplex-rotations
    """
    # This test compares the exact output state after applying the operator to a random input state
    # In particular, in confirms the following circuit identity
    #
    # |ψ>   ╭: ─╭◻──────────────────────────╭◻──────────┤ ╮ ≈MUX-R_Z(θ_j)|ψ>
    #       │: ─├◻──────────────────────────├◻──────────┤ │
    #       ╰: ─│──────────╭○────────────╭○─│───────────┤ ╯
    # |0>    : ─├load(θ_j)─│──╭SemiAdder─│──├load†(θ_j)─┤   |0>
    # |0>    : ─├load(θ_j)─│──├SemiAdder─│──├load†(θ_j)─┤   |0>
    # |0>    : ─╰load(θ_j)─├──├SemiAdder─├──╰load†(θ_j)─┤   |0>
    #       ╭: ────────────├X─├SemiAdder─├X─────────────┤ ╮
    # |∇_b> ┤: ────────────├X─├SemiAdder─├X─────────────┤ ├ |∇_b>
    #       ╰: ────────────╰X─╰SemiAdder─╰X─────────────┤ ╯

    with qp.decomposition.toggle_graph_ctx(
        True
    ):  # safe alternative to avoid enabling graph globally on the labs test runner
        prec = 3

        phi = (1 / 2 + 0 / 4 + 1 / 8) * 4 * np.pi
        wires = [0]

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
        phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(prec - 1)])

        phase_grad_state = np.exp(-1j * 2 * np.pi * np.arange(2**3) / 2**3) / np.sqrt(2**3)

        all_wires = angle_wires + phase_grad_wires + work_wires + wires

        custom_decomp = make_rz_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires)

        @qp.transforms.decompose(
            gate_set={
                "StatePrep",
                "Adjoint(StatePrep)",
                "C(BasisEmbedding)",
                "SemiAdder",
                "CNOT",
                "GlobalPhase",
            },
            fixed_decomps={qp.RZ: custom_decomp},
        )
        @qp.qnode(qp.device("default.qubit", wires=all_wires))
        def circuit(phi, in_state):
            qp.StatePrep(in_state, wires=wires)  # input state
            qp.StatePrep(phase_grad_state, wires=phase_grad_wires)  # phase gradient state
            qp.RZ(phi, wires)
            qp.adjoint(
                qp.StatePrep(phase_grad_state, wires=phase_grad_wires)
            )  # uncompute phase gradient state
            return qp.state()

        # random input state
        rng = np.random.default_rng(seed=seed)
        in_state = rng.random(2 ** len(wires))
        in_state /= np.linalg.norm(in_state)

        # returned output state
        out_state = circuit(phi, in_state)

        # expected output state
        zeros = np.eye(2 ** (prec * 3 - 1))[0]  # |000> on all the aux wires
        out_state_expected = qp.matrix(qp.RZ(phi, wires)) @ in_state
        out_state_expected = np.kron(zeros, out_state_expected)

        assert np.allclose(out_state, out_state_expected)
