# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Contains functions to convert a PennyLane tape to the textbook MBQC formalism
"""
from functools import partial, singledispatch

import networkx as nx

from pennylane import math
from pennylane.decomposition import enabled_graph, register_resources
from pennylane.devices.preprocess import null_postprocessing
from pennylane.measurements import SampleMP, sample
from pennylane.ops import CNOT, CZ, RZ, GlobalPhase, H, Identity, Rot, S, X, Y, Z, cond
from pennylane.queuing import AnnotatedQueue
from pennylane.tape import QuantumScript
from pennylane.transforms import decompose, transform

from .conditional_measure import cond_measure
from .graph_state_preparation import make_graph_state
from .operations import RotXZX
from .parametric_midmeasure import measure_arbitrary_basis, measure_x, measure_y
from .utils import QubitMgr, parity

mbqc_gate_set = frozenset({CNOT, H, S, RotXZX, RZ, X, Y, Z, Identity, GlobalPhase})


@register_resources({RotXZX: 1})
def _rot_to_xzx(phi, theta, omega, wires, **__):
    mat = Rot.compute_matrix(phi, theta, omega)
    lam, theta, phi = math.decomposition.xzx_rotation_angles(mat)
    RotXZX(lam, theta, phi, wires)


@transform
def convert_to_mbqc_gateset(tape):
    """Converts a circuit expressed in arbitrary gates to the limited gate set that we can
    convert to the textbook MBQC formalism"""
    if not enabled_graph():
        raise RuntimeError(
            "Using `convert_to_mbqc_gateset` requires the graph-based decomposition"
            " method. This can be toggled by calling `qml.decomposition.enable_graph()`"
        )
    tapes, fn = decompose(tape, gate_set=mbqc_gate_set, alt_decomps={Rot: [_rot_to_xzx]})
    return tapes, fn


@transform
def convert_to_mbqc_formalism(tape):
    """Convert a circuit to the textbook MBQC formalism based on the procedures outlined in
    Raussendorf et al. 2003, https://doi.org/10.1103/PhysRevA.68.022312. The circuit must
    be decomposed to the gate set {CNOT, H, S, RotXZX, RZ, X, Y, Z, Identity, GlobalPhase}
    before applying the transform.

    Note that this transform leaves all Paulis and Identities as physical gates, and applies
    all byproduct operations online immediately after their respective measurement procedures."""

    if len(tape.measurements) != 1 or not isinstance(tape.measurements[0], (SampleMP)):
        raise NotImplementedError(
            "Transforming to the MBQC formalism is not implemented for circuits where the "
            "final measurements have not been converted to a single samples measurement"
        )

    mp = tape.measurements[0]
    meas_wires = mp.wires if mp.wires else tape.wires

    # we include 13 auxillary wires - the largest number needed is 13 (for CNOT)
    num_qubits = len(tape.wires) + 13
    q_mgr = QubitMgr(num_qubits=num_qubits, start_idx=0)

    wire_map = {w: q_mgr.acquire_qubit() for w in tape.wires}

    with AnnotatedQueue() as q:
        for op in tape.operations:
            if isinstance(op, GlobalPhase):  # no wires
                GlobalPhase(*op.data)
            elif isinstance(op, CNOT):  # two wires
                ctrl, tgt = op.wires[0], op.wires[1]
                wire_map[ctrl], wire_map[tgt], measurements = queue_cnot(
                    q_mgr, wire_map[ctrl], wire_map[tgt]
                )
                cnot_corrections(measurements)(wire_map[ctrl], wire_map[tgt])
            else:  # one wire
                # pylint: disable=isinstance-second-argument-not-valid-type
                if isinstance(op, (X, Y, Z, Identity)):
                    # else branch because Identity may not have wires
                    wire = wire_map[op.wires[0]] if op.wires else ()
                    op.__class__(wire)
                else:
                    w = op.wires[0]
                    wire_map[w], measurements = queue_single_qubit_gate(
                        q_mgr, op, in_wire=wire_map[w]
                    )
                    queue_corrections(op, measurements)(wire_map[w])

    temp_tape = QuantumScript.from_queue(q)

    new_wires = [wire_map[w] for w in meas_wires]

    new_tape = tape.copy(operations=temp_tape.operations, measurements=[sample(wires=new_wires)])

    return (new_tape,), null_postprocessing


def queue_single_qubit_gate(q_mgr, op, in_wire):
    """Queue the resource state preparation, measurements and byproducts
    to execute the operation in the MBQC formalism. This implementation
    follows the procedures defined in Raussendorf et al. 2003,
    https://doi.org/10.1103/PhysRevA.68.022312, see Fig. 2"""

    graph_wires = q_mgr.acquire_qubits(4)
    wires = [in_wire] + graph_wires

    make_graph_state(nx.grid_graph((4,)), wires=graph_wires)
    CZ([wires[0], wires[1]])

    measurements = queue_measurements(op, wires)

    # release input qubit and intermediate graph qubits
    q_mgr.release_qubits(wires[0:-1])
    return wires[-1], measurements


@singledispatch
def queue_measurements(op, wires):
    """Queue the measurements needed to execute the operation in the MBQC formalism"""
    raise NotImplementedError(f"Received unsupported gate of type {op}")


@queue_measurements.register(RotXZX)
def _rot_measurements(op: RotXZX, wires):
    """Queue the measurements needed to execute RotXZX in the MBQC formalism"""

    phi, theta, omega = op.data

    m1 = measure_x(wires[0], reset=True)
    m2 = cond_measure(
        m1,
        partial(measure_arbitrary_basis, angle=phi),
        partial(measure_arbitrary_basis, angle=-phi),
    )(plane="XY", wires=wires[1], reset=True)
    m3 = cond_measure(
        m2,
        partial(measure_arbitrary_basis, angle=theta),
        partial(measure_arbitrary_basis, angle=-theta),
    )(plane="XY", wires=wires[2], reset=True)
    m4 = cond_measure(
        m1 ^ m3,
        partial(measure_arbitrary_basis, angle=omega),
        partial(measure_arbitrary_basis, angle=-omega),
    )(plane="XY", wires=wires[3], reset=True)

    return [m1, m2, m3, m4]


@queue_measurements.register(RZ)
def _rz_measurements(op: RZ, wires):
    """Queue the measurements needed to execute RZ in the MBQC formalism"""

    angle = op.parameters[0]

    m1 = measure_x(wires[0], reset=True)
    m2 = measure_x(wires[1], reset=True)
    m3 = cond_measure(
        m2,
        partial(measure_arbitrary_basis, angle=angle, reset=True),
        partial(measure_arbitrary_basis, angle=-angle, reset=True),
    )(plane="XY", wires=wires[2])
    m4 = measure_x(wires[3], reset=True)

    return [m1, m2, m3, m4]


@queue_measurements.register(H)
def _hadamard_measurements(op: H, wires):
    """Queue the measurements needed to execute Hadamard in the MBQC formalism"""
    m1 = measure_x(wires[0], reset=True)
    m2 = measure_y(wires[1], reset=True)
    m3 = measure_y(wires[2], reset=True)
    m4 = measure_y(wires[3], reset=True)

    return [m1, m2, m3, m4]


@queue_measurements.register(S)
def _s_measurements(op: S, wires):
    """Queue the measurements needed to execute S in the MBQC formalism"""
    m1 = measure_x(wires[0], reset=True)
    m2 = measure_x(wires[1], reset=True)
    m3 = measure_y(wires[2], reset=True)
    m4 = measure_x(wires[3], reset=True)

    return [m1, m2, m3, m4]


@singledispatch
def queue_corrections(op, measurements):
    """Queue the byproduct corrections associated with the operation in the
    MBQC formalism, based on the operation and the measurement results"""
    raise NotImplementedError(f"Received unsupported gate of type {op}")


@queue_corrections.register(RotXZX)
@queue_corrections.register(RZ)
def _rotation_corrections(op, measurements):
    """Queue the byproduct corrections associated with the rotation
    gate RotXZX in the MBQC formalism, based on the operation and the
    measurement results. Note that these corrections also apply in the
    more specific rotation case, RZ = RotXZX(0, Z, 0)"""

    m1, m2, m3, m4 = measurements

    def correction_func(wire):
        cond(m1 ^ m3, Z)(wire)
        cond(m2 ^ m4, X)(wire)

    return correction_func


@queue_corrections.register(H)
def _hadamard_corrections(op, measurements):
    """Queue the byproduct corrections associated with the Hadamard gate in
    the MBQC formalism, based on the operation and the measurement results"""
    m1, m2, m3, m4 = measurements

    def correction_func(wire):
        cond(m2 ^ m3, Z)(wire)
        cond(parity(m1, m3, m4), X)(wire)

    return correction_func


@queue_corrections.register(S)
def _s_corrections(op, measurements):
    """Queue the byproduct corrections associated with the S gate in
    the MBQC formalism, based on the operation and the measurement results"""

    m1, m2, m3, m4 = measurements

    def correction_func(wire):
        cond(parity(m1, m2, m3, 1), Z)(wire)
        cond(m2 ^ m4, X)(wire)

    return correction_func


def queue_cnot(q_mgr, ctrl_idx, target_idx):
    """Queue the resource state preparation, measurements and byproducts to execute
    the operation in the MBQC formalism. This is the 15-qubit procedure from
    Raussendorf et al. 2003, https://doi.org/10.1103/PhysRevA.68.022312, Fig. 2"""

    graph_wires = q_mgr.acquire_qubits(13)

    # Denote the index for the final output state
    output_ctrl_idx = graph_wires[5]
    output_target_idx = graph_wires[12]

    # Prepare the state
    make_graph_state(_generate_cnot_graph(), wires=graph_wires)

    # entangle input and graph using first qubit
    CZ([ctrl_idx, graph_wires[0]])
    CZ([target_idx, graph_wires[7]])

    measurements = cnot_measurements((ctrl_idx, target_idx, graph_wires))

    q_mgr.release_qubit(ctrl_idx)
    q_mgr.release_qubit(target_idx)

    # We can now free all but the last qubit, which has become the new input_idx
    q_mgr.release_qubits(graph_wires[0:5] + graph_wires[6:-1])
    return output_ctrl_idx, output_target_idx, measurements


def cnot_measurements(wires):
    """Queue the measurements needed to execute CNOT in the MBQC formalism.
    Numbering convention follows the procedure in Raussendorf et al. 2003,
    https://doi.org/10.1103/PhysRevA.68.022312, see Fig. 2"""
    ctrl_idx, target_idx, graph_wires = wires

    m1 = measure_x(ctrl_idx, reset=True)
    m2 = measure_y(graph_wires[0], reset=True)
    m3 = measure_y(graph_wires[1], reset=True)
    m4 = measure_y(graph_wires[2], reset=True)
    m5 = measure_y(graph_wires[3], reset=True)
    m6 = measure_y(graph_wires[4], reset=True)

    m8 = measure_y(graph_wires[6], reset=True)

    m9 = measure_x(target_idx, reset=True)
    m10 = measure_x(graph_wires[7], reset=True)
    m11 = measure_x(graph_wires[8], reset=True)
    m12 = measure_y(graph_wires[9], reset=True)
    m13 = measure_x(graph_wires[10], reset=True)
    m14 = measure_x(graph_wires[11], reset=True)

    return [m1, m2, m3, m4, m5, m6, m8, m9, m10, m11, m12, m13, m14]


def cnot_corrections(measurements):
    """Queue the byproduct corrections associated with the CNOT gate in
    the MBQC formalism, based on measurement results"""

    # Numbering convention follows the procedure in Raussendorf et al. 2003,
    # https://doi.org/10.1103/PhysRevA.68.022312, Fig 2
    m1, m2, m3, m4, m5, m6, m8, m9, m10, m11, m12, m13, m14 = measurements

    # corrections on control
    x_cor_ctrl = parity(m2, m3, m5, m6)
    z_cor_ctrl = parity(m1, m3, m4, m5, m8, m9, m11, 1)

    # corrections on target
    x_cor_tgt = parity(m2, m3, m8, m10, m12, m14)
    z_cor_tgt = parity(m9, m11, m13)

    def correction_func(ctrl_wire, target_wire):
        cond(z_cor_ctrl, Z)(ctrl_wire)
        cond(x_cor_ctrl, X)(ctrl_wire)
        cond(z_cor_tgt, Z)(target_wire)
        cond(x_cor_tgt, X)(target_wire)

    return correction_func


def _generate_cnot_graph():
    """Generate a graph for creating the resource state for a
    CNOT gate. Raussendorf et al. 2003, Fig. 2a.
    https://doi.org/10.1103/PhysRevA.68.022312"""
    wires = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    g = nx.Graph()
    g.add_nodes_from(wires)
    g.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (2, 6),
            (6, 9),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
        ]
    )
    return g
