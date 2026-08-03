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

<<<<<<< HEAD
<<<<<<< HEAD
"""Tests for the decomposition rule qp.transforms.decompositions.make_selectpaulirot_to_phase_gradient_decomp"""

import warnings
=======
"""Tests for the decomposition rule qp.labs.transforms.make_selectpaulirot_to_phase_gradient_decomp"""
>>>>>>> 2d4eb0439e (renaming)
=======
"""Tests for the decomposition rule qp.transforms.decompositions.make_selectpaulirot_to_phase_gradient_decomp"""
>>>>>>> 2eb312e97c (fix docstring for test files)

import numpy as np

# pylint: disable=no-value-for-parameter
import pytest

import pennylane as qp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
<<<<<<< HEAD
=======
from pennylane.transforms.decompose import DecomposeInterpreter
>>>>>>> 2d4eb0439e (renaming)
from pennylane.transforms.decompositions import (
    make_selectpaulirot_to_phase_gradient_decomp,
)
from pennylane.wires import WireError


def test_wires_error_decomp_fun():
    """Test WireError is raised correctly when calling the decomposition function on a large
    SelectPauliRot that needs more work wires for its QROM than are available."""
    registers = {
        "angle_wires": 3,
        "phase_grad_wires": 3,
        "work_wires": 2,
        "control_wires": 4,
        "target_wires": 1,
    }
    registers = qp.registers(registers)
    control_wires = registers.pop("control_wires")
    target_wire = registers.pop("target_wires")[0]
    rule = make_selectpaulirot_to_phase_gradient_decomp(**registers)
    angles = np.random.random(2**4)
    with pytest.raises(WireError, match=r"work_wires need to be at least of size len\(control"):
        rule(angles, control_wires, target_wire, "X")


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


<<<<<<< HEAD
<<<<<<< HEAD
@pytest.mark.usefixtures("enable_graph_decomposition")
=======
# @pytest.mark.usefixtures("enable_graph_decomposition") # fixture doesnt exist in labs tests
>>>>>>> 2d4eb0439e (renaming)
=======
@pytest.mark.usefixtures("enable_graph_decomposition")
>>>>>>> 09a29c6c19 (get rid of graph ctx in favour of marker)
@pytest.mark.parametrize("prec", [2, 3, 5])
@pytest.mark.parametrize("num_controls", [1, 2])
def test_as_fixed_decomps(prec, num_controls):
    """Test that the decomposition rule from make_selectpaulirot_to_phase_gradient_decomp works as expected
    as a fixed decomposition and yields the correct resources"""
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 09a29c6c19 (get rid of graph ctx in favour of marker)
    angles = (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
        @ np.array([1 / 2, 1 / 4, 1 / 8])
        * 4
        * np.pi
    )[: 2**num_controls]
<<<<<<< HEAD

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

    specs = qp.specs(circuit)(angles)["resources"].quantum_operations
    expected_specs = {
        "QROM": 2,
        "CNOT": 2 * prec,
        "PauliX": 2 * prec,
        "SemiAdder": 1,
    }
    assert expected_specs == specs


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize("rot_axis", ["X", "Y", "Z"])
def test_integration_multi_wire(rot_axis, seed):
    """Numerically verify the decomposition reproduces SelectPauliRot's
    action on the system wires.

    Mirrors ``test_integration_multi_wire`` in test_rz_phase_gradient.py.
    """
    prec = 3
    num_controls = 2
    angles = np.array([0.5, 1.5, 2.0, 2.5]) * np.pi  # 2**num_controls entries

    ctrl_wires = list(range(num_controls))
    target_wire = num_controls
    sys_wires = ctrl_wires + [target_wire]

    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
    num_work = max(prec, num_controls + 1) - 1
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(num_work)])

    phase_grad_state = np.exp(-1j * 2 * np.pi * np.arange(2**prec) / 2**prec) / np.sqrt(2**prec)
    all_wires = angle_wires + phase_grad_wires + work_wires + qp.wires.Wires(sys_wires)

    custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires
    )

    gs = {
        "QROM",
        "SemiAdder",
        "CNOT",
        "PauliX",
        "GlobalPhase",
        "StatePrep",
        "Adjoint(StatePrep)",
    }

    # Depending on the rot_axis, additional operators
    # are expected in the decomposition
    if rot_axis == "Y":
        gs |= {
            "S",
            "Adjoint(S)",
        }
    if rot_axis in ("X", "Y"):
        gs |= {"Hadamard"}

    @qp.decompose(
        gate_set=gs,
        fixed_decomps={qp.SelectPauliRot: custom_decomp},
    )
    @qp.qnode(qp.device("default.qubit", wires=all_wires))
    def circuit(in_state):
        qp.StatePrep(in_state, wires=sys_wires)  # input state
        qp.StatePrep(phase_grad_state, wires=phase_grad_wires)  # phase gradient state
        qp.SelectPauliRot(
            angles, control_wires=ctrl_wires, target_wire=target_wire, rot_axis=rot_axis
        )
        qp.adjoint(
            qp.StatePrep(phase_grad_state, wires=phase_grad_wires)
        )  # uncompute phase gradient state
        return qp.state()

    # random input state
    rng = np.random.default_rng(seed)
    in_state = rng.random(2 ** len(sys_wires))
    in_state /= np.linalg.norm(in_state)

    # returned output state

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out_state = circuit(in_state)

    # Assert no warnings are raised from the decomposition
    assert len(record) == 0

    # expected: SelectPauliRot is applied on the system wires
    out_state_expected = (
        qp.matrix(
            qp.SelectPauliRot(
                angles, control_wires=ctrl_wires, target_wire=target_wire, rot_axis=rot_axis
            ),
            wire_order=sys_wires,
        )
        @ in_state
    )
    n_aux = len(angle_wires) + len(phase_grad_wires) + len(work_wires)
    # and aux registers back in |0...0>
    zeros = np.eye(2**n_aux, 1)[:, 0]  # |0...0> on all aux wires
    expected = np.kron(zeros, out_state_expected)

    assert np.allclose(out_state, expected), f"decomposition wrong for rot_axis={rot_axis}"
=======
    with qp.decomposition.toggle_graph_ctx(
        True
    ):  # safe alternative to avoid enabling graph globally on the labs test runner
        angles = (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
            @ np.array([1 / 2, 1 / 4, 1 / 8])
            * 4
            * np.pi
        )[: 2**num_controls]
=======
>>>>>>> 09a29c6c19 (get rid of graph ctx in favour of marker)

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

    specs = qp.specs(circuit)(angles)["resources"].quantum_operations
    expected_specs = {
        "QROM": 2,
        "CNOT": 2 * prec,
        "PauliX": 2 * prec,
        "SemiAdder": 1,
    }
    assert expected_specs == specs


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize("rot_axis", ["X", "Y", "Z"])
def test_integration_multi_wire(rot_axis, seed):
    """Numerically verify the decomposition reproduces SelectPauliRot's
    action on the system wires.

    Mirrors ``test_integration_multi_wire`` in test_rz_phase_gradient.py.
    """
    prec = 3
    num_controls = 2
    angles = np.array([0.5, 1.5, 2.0, 2.5]) * np.pi  # 2**num_controls entries

    ctrl_wires = list(range(num_controls))
    target_wire = num_controls
    sys_wires = ctrl_wires + [target_wire]

    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
    num_work = max(prec, num_controls + 1) - 1
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(num_work)])

    phase_grad_state = np.exp(-1j * 2 * np.pi * np.arange(2**prec) / 2**prec) / np.sqrt(2**prec)
    all_wires = angle_wires + phase_grad_wires + work_wires + qp.wires.Wires(sys_wires)

    custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires
    )

    @qp.decompose(
        gate_set={
            "QROM",
            "SemiAdder",
            "CNOT",
            "PauliX",
            "GlobalPhase",
        },
        fixed_decomps={qp.SelectPauliRot: custom_decomp},
    )
    @qp.qnode(qp.device("default.qubit", wires=all_wires))
    def circuit(in_state):
        qp.StatePrep(in_state, wires=sys_wires)  # input state
        qp.StatePrep(phase_grad_state, wires=phase_grad_wires)  # phase gradient state
        qp.SelectPauliRot(
            angles, control_wires=ctrl_wires, target_wire=target_wire, rot_axis=rot_axis
        )
        qp.adjoint(
            qp.StatePrep(phase_grad_state, wires=phase_grad_wires)
        )  # uncompute phase gradient state
        return qp.state()

    # random input state
    rng = np.random.default_rng(seed)
    in_state = rng.random(2 ** len(sys_wires))
    in_state /= np.linalg.norm(in_state)

    # returned output state
    out_state = circuit(in_state)

    # expected: SelectPauliRot is applied on the system wires
    out_state_expected = (
        qp.matrix(
            qp.SelectPauliRot(
                angles, control_wires=ctrl_wires, target_wire=target_wire, rot_axis=rot_axis
            ),
            wire_order=sys_wires,
        )
        @ in_state
    )
    n_aux = len(angle_wires) + len(phase_grad_wires) + len(work_wires)
    # and aux registers back in |0...0>
    zeros = np.eye(2**n_aux, 1)[:, 0]  # |0...0> on all aux wires
    expected = np.kron(zeros, out_state_expected)

<<<<<<< HEAD
<<<<<<< HEAD
        assert np.allclose(out_state, out_state_expected)
>>>>>>> 2d4eb0439e (renaming)
=======
    assert np.allclose(out_state, out_state_expected)
>>>>>>> 09a29c6c19 (get rid of graph ctx in favour of marker)
=======
    assert np.allclose(out_state, expected), f"decomposition wrong for rot_axis={rot_axis}"
>>>>>>> 6318150b9b (upgrade other test)


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.capture
def test_capture_compatibility():
    """Ensures capture compatibility."""

<<<<<<< HEAD
    prec = 3
    num_controls = 2
    control_wires = list(range(num_controls))
    target_wire = num_controls
    first_aux = num_controls + 1

    angle_wires = list(range(first_aux, first_aux + prec))
    phase_grad_wires = list(range(first_aux + prec, first_aux + 2 * prec))
    num_work_wires = max(prec, num_controls + 1) - 1
    work_wires = list(range(first_aux + 2 * prec, first_aux + 2 * prec + num_work_wires))

    angles = np.array(
        [
            (1 / 2 + 1 / 4 + 1 / 8) * 4 * np.pi,
            (1 / 2 + 1 / 4 + 0 / 8) * 4 * np.pi,
            (1 / 2 + 0 / 4 + 1 / 8) * 4 * np.pi,
            (0 / 2 + 1 / 4 + 1 / 8) * 4 * np.pi,
        ]
    )

    custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires
    )

    op = qp.SelectPauliRot(angles, control_wires=control_wires, target_wire=target_wire)
    _test_decomposition_rule(op, custom_decomp)
=======
    # pylint: disable=import-outside-toplevel
    import jax

    from pennylane.tape.plxpr_conversion import CollectOpsandMeas

    prec = 3
    num_controls = 2
    control_wires = list(range(num_controls))
    target_wire = num_controls
    first_aux = num_controls + 1

    angle_wires = list(range(first_aux, first_aux + prec))
    phase_grad_wires = list(range(first_aux + prec, first_aux + 2 * prec))
    num_work_wires = max(prec, num_controls + 1) - 1
    work_wires = list(range(first_aux + 2 * prec, first_aux + 2 * prec + num_work_wires))

    angles = np.array(
        [
            (1 / 2 + 1 / 4 + 1 / 8) * 4 * np.pi,
            (1 / 2 + 1 / 4 + 0 / 8) * 4 * np.pi,
            (1 / 2 + 0 / 4 + 1 / 8) * 4 * np.pi,
            (0 / 2 + 1 / 4 + 1 / 8) * 4 * np.pi,
        ]
    )

    custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires
    )

    gate_set = {
        "QROM",
        "SemiAdder",
        "CNOT",
        "PauliX",
        "GlobalPhase",
    }

    @DecomposeInterpreter(gate_set=gate_set, fixed_decomps={qp.SelectPauliRot: custom_decomp})
    def f(angles):
        qp.SelectPauliRot(angles, control_wires=control_wires, target_wire=target_wire)
        return qp.state()

    cjaxpr = jax.make_jaxpr(f)(angles)

    collector = CollectOpsandMeas()
    collector.eval(cjaxpr.jaxpr, cjaxpr.consts, angles)

<<<<<<< HEAD
            op_names = {op.name for op in collector.state["ops"]}
            # NOTE: Because `adjoint` is lazy in ChangeOpBasis,
            # unsimplified operators will be collected.
            gate_set |= {"Adjoint(CNOT)", "Adjoint(PauliX)"}
            assert op_names.issubset(
                gate_set
            ), f"Following ops are present but not in gateset: {op_names - gate_set}"
    finally:
        qp.capture.disable()
>>>>>>> 2d4eb0439e (renaming)
=======
    op_names = {op.name for op in collector.state["ops"]}
    # NOTE: Because `adjoint` is lazy in ChangeOpBasis,
    # unsimplified operators will be collected.
    gate_set |= {"Adjoint(CNOT)", "Adjoint(PauliX)"}
    assert op_names.issubset(
        gate_set
    ), f"Following ops are present but not in gateset: {op_names - gate_set}"
>>>>>>> 09a29c6c19 (get rid of graph ctx in favour of marker)


@pytest.mark.parametrize(
    "rot_axis, expected_op",
    [
        ("X", qp.RX),
        ("Y", qp.RY),
        ("Z", qp.RZ),
    ],
)
def test_rot_axis_zero_controls(rot_axis, expected_op):
    """Test the 0-control-wire edge case for all rotation axes."""
    angle_wires = qp.wires.Wires(["aux_0"])
    phase_grad_wires = qp.wires.Wires(["qft_0"])
    work_wires = qp.wires.Wires(["work_0"])

    custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires
    )

    angles = np.array([1.23])
    op = qp.SelectPauliRot(angles, control_wires=[], target_wire=0, rot_axis=rot_axis)

    _test_decomposition_rule(op, custom_decomp)

    with qp.queuing.AnnotatedQueue() as q:
        custom_decomp(angles, [], 0, rot_axis=rot_axis)

    assert len(q.queue) == 1
    assert isinstance(q.queue[0], expected_op)
<<<<<<< HEAD
<<<<<<< HEAD
=======


@pytest.mark.parametrize("rot_axis", ["X", "Y", "Z"])
def test_rot_axis_basis_changes(rot_axis):
    """Test that the custom decomposition rule correctly applies basis changes (Hadamard/S)
    for different rotation axes when control wires are present."""
    prec = 2
    num_controls = 1
    num_work_wires = max(prec, num_controls + 1) - 1

    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(num_work_wires)])

    custom_decomp = make_selectpaulirot_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires
    )

    angles = np.array([0.5, 1.5]) * np.pi
    op = qp.SelectPauliRot(angles, control_wires=[0], target_wire=1, rot_axis=rot_axis)

    # This should verify the decomp structure annd make sure that the resources match the
    # decomposition, giving us enough coverage
    _test_decomposition_rule(op, custom_decomp)
>>>>>>> 2d4eb0439e (renaming)
=======
>>>>>>> 6318150b9b (upgrade other test)
