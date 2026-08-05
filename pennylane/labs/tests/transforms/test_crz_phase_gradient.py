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

"""Tests for ``labs.transforms.make_crz_to_phase_gradient_decomp``"""

import numpy as np

# pylint: disable=no-value-for-parameter, disable=too-many-arguments
import pytest

import pennylane as qp
from pennylane.labs.transforms.decomp_crz_phase_gradient import make_crz_to_phase_gradient_decomp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.tape.plxpr_conversion import CollectOpsandMeas
from pennylane.transforms.decompose import DecomposeInterpreter


@pytest.mark.parametrize("phi", [0.5, 0.3, 1 / 2 + 1 / 4 + 1 / 8, 1.0])
@pytest.mark.parametrize("p", [2, 3, 4])
def test_valid_decomp(phi, p):
    """Test that ``make_crz_to_phase_gradient_decomp`` yields a valid decomposition"""
    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(p)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(p)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(p - 1)])

    kwargs = {
        "angle_wires": angle_wires,
        "phase_grad_wires": phase_grad_wires,
        "work_wires": work_wires,
    }

    custom_decomp = make_crz_to_phase_gradient_decomp(**kwargs)
    op = qp.CRZ(phi, [0, 1])
    _test_decomposition_rule(op, custom_decomp, skip_decomp_matrix_check=True)


# @pytest.mark.usefixtures("enable_graph_decomposition") # fixture doesnt exist in labs tests
@pytest.mark.parametrize("phi", [0.5, 0.3, 1 / 2 + 1 / 4 + 1 / 8, 1.0])
@pytest.mark.parametrize("p", [2, 3, 4])
def test_as_fixed_decomps(phi, p):
    """Test that the decomposition rule from ``make_crz_to_phase_gradient_decomp`` works as
    expected as a fixed decomposition and yields the correct resources"""
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

        custom_decomp = make_crz_to_phase_gradient_decomp(**kwargs)
        gate_set = {"SemiAdder", "C(BasisState)"}

        @qp.transforms.decompose(gate_set=gate_set, fixed_decomps={qp.CRZ: custom_decomp})
        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            qp.CRZ(phi, [0, 1])
            return qp.state()

        specs = qp.specs(circuit)()["resources"].quantum_operations
        expected_specs = {"SemiAdder": 1, "C(BasisState)": 4}
        assert specs == expected_specs


@pytest.mark.parametrize("phi", [0.5, 0.3, 1 / 2 + 1 / 4 + 1 / 8, 1.0])
@pytest.mark.parametrize("p", [2, 3, 4])
def test_as_alt_decomps(phi, p):
    """Test that the decomposition rule from ``make_crz_to_phase_gradient_decomp`` works as
    expected as an alternative decomposition and yields the correct resources"""
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

        custom_decomp = make_crz_to_phase_gradient_decomp(**kwargs)
        gate_set = {"SemiAdder", "C(BasisState)"}

        @qp.transforms.decompose(gate_set=gate_set, alt_decomps={qp.CRZ: [custom_decomp]})
        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            qp.CRZ(phi, [0, 1])
            return qp.state()

        specs = qp.specs(circuit)()["resources"].quantum_operations
        expected_specs = {"SemiAdder": 1, "C(BasisState)": 4}
        assert specs == expected_specs


def test_integration_multi_wire(seed):
    """
    Tests that the decomposition correctly realizes the phase gradient decomposition of CRZ as
    described in https://pennylane.ai/compilation/phase-gradient/c-control-rotations
    """

    with qp.decomposition.toggle_graph_ctx(
        True
    ):  # safe alternative to avoid enabling graph globally on the labs test runner
        prec = 3

        phi = (1 / 2 + 0 / 4 + 1 / 8) * 4 * np.pi
        wires = [0, 1]

        angle_wires = [f"aux_{i}" for i in range(prec)]
        phase_grad_wires = [f"qft_{i}" for i in range(prec)]
        work_wires = [f"work_{i}" for i in range(prec - 1)]

        phase_grad_state = np.exp(-1j * 2 * np.pi * np.arange(2**3) / 2**3) / np.sqrt(2**3)

        all_wires = angle_wires + phase_grad_wires + work_wires + wires

        custom_decomp = make_crz_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires)

        @qp.transforms.decompose(
            gate_set={
                "StatePrep",
                "Adjoint(StatePrep)",
                "C(BasisState)",
                "SemiAdder",
            },
            fixed_decomps={qp.CRZ: custom_decomp},
        )
        @qp.qnode(qp.device("default.qubit", wires=all_wires))
        def circuit(phi, in_state):
            # Prepare input state
            qp.StatePrep(in_state, wires=wires)
            # Prepare phase gradient state
            qp.StatePrep(phase_grad_state, wires=phase_grad_wires)
            qp.CRZ(phi, wires)
            # uncompute phase gradient state
            qp.adjoint(qp.StatePrep(phase_grad_state, wires=phase_grad_wires))
            return qp.state()

        # random input state
        rng = np.random.default_rng(seed=seed)
        in_state = rng.random(2 ** len(wires))
        in_state /= np.linalg.norm(in_state)

        # returned output state
        out_state = circuit(phi, in_state)

        # expected output state
        zeros = np.eye(2 ** (prec * 3 - 1), 1)[:, 0]  # |000> on all the aux wires
        out_state_expected = qp.matrix(qp.CRZ(phi, wires)) @ in_state
        out_state_expected = np.kron(zeros, out_state_expected)

        assert np.allclose(out_state, out_state_expected)


@pytest.mark.jax
def test_capture_compatibility():
    """Ensures capture compatibility."""

    # pylint: disable=import-outside-toplevel
    import jax
    import jax.numpy as jnp

    qp.capture.enable()
    try:
        with qp.decomposition.toggle_graph_ctx(True):
            first_free = 2  # 0, 1 used by CRZ

            precision = 3
            angle_wires = jnp.array(list(range(first_free, first_free + precision)))
            phase_grad_wires = jnp.array(
                list(range(first_free + precision, first_free + 2 * precision))
            )
            work_wires = jnp.array(
                list(range(first_free + 2 * precision, first_free + 3 * precision - 1))
            )

            custom_decomp = make_crz_to_phase_gradient_decomp(
                angle_wires, phase_grad_wires, work_wires
            )

            gate_set = {"C(BasisState)", "SemiAdder"}

            @DecomposeInterpreter(gate_set=gate_set, fixed_decomps={qp.CRZ: custom_decomp})
            def f(phi):
                qp.CRZ(phi, [0, 1])
                return qp.state()

            phi_val = jnp.pi

            cjaxpr = jax.make_jaxpr(f)(phi_val)

            collector = CollectOpsandMeas()
            collector.eval(cjaxpr.jaxpr, cjaxpr.consts, phi_val)

            op_names = {op.name for op in collector.state["ops"]}
            assert op_names.issubset(
                gate_set
            ), f"Following ops are present but not in gateset: {op_names - gate_set}"
    finally:
        qp.capture.disable()
