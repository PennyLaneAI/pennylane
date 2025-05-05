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
from functools import partial

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
from .graph_state_preparation import GraphStatePrep
from .operations import RotXZX
from .parametric_midmeasure import measure_arbitrary_basis, measure_x, measure_y
from .utils import QubitMgr

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
    """docstring goes here"""

    if len(tape.measurements) != 1 or not isinstance(tape.measurements[0], SampleMP):
        raise NotImplementedError(
            "Transforming to the MBQC formalism is not implemented for circuits where "
            "the final measurements have not been converted to samples"
        )

    meas_wires = tape.measurements[0].wires if tape.measurements[0].wires else tape.wires

    num_qubits = len(tape.wires) + 2 + 13
    q_mgr = QubitMgr(num_qubits=num_qubits, start_idx=0)

    wire_map = {w: q_mgr.acquire_qubit() for w in tape.wires}

    with AnnotatedQueue() as q:
        for op in tape.operations:
            if isinstance(op, (X, Y, Z, Identity, GlobalPhase)):
                op.__class__(op.wires)
            elif isinstance(op, CNOT):
                (wire_map[op.wires[0]], wire_map[op.wires[1]]) = stencils[op.__class__](
                    q_mgr, wire_map[op.wires[0]], wire_map[op.wires[1]], *op.data
                )
            else:
                wire_map[op.wires[0]] = stencils[op.__class__](
                    q_mgr, wire_map[op.wires[0]], *op.data
                )

    temp_tape = QuantumScript.from_queue(q)

    new_wires = [wire_map[w] for w in meas_wires]
    new_tape = tape.copy(operations=temp_tape.operations, measurements=[sample(wires=new_wires)])

    return (new_tape,), null_postprocessing


def rot_stencil(q_mgr, in_wire, phi, theta, omega):
    """A stencil for the rotation gate RotXZX, expressed in the MBQC formalism. This
    stencil includes byproduct corrections in addition to the measurements."""

    graph_wires = q_mgr.acquire_qubits(4)
    out_wire = graph_wires[-1]

    GraphStatePrep(nx.grid_graph((4,)), wires=graph_wires)
    CZ([in_wire, graph_wires[0]])

    m1 = measure_x(in_wire, reset=True)
    m2 = cond_measure(
        m1,
        partial(measure_arbitrary_basis, angle=phi),
        partial(measure_arbitrary_basis, angle=-phi),
    )(plane="XY", wires=graph_wires[0], reset=True)
    m3 = cond_measure(
        m2,
        partial(measure_arbitrary_basis, angle=theta),
        partial(measure_arbitrary_basis, angle=-theta),
    )(plane="XY", wires=graph_wires[1], reset=True)
    m4 = cond_measure(
        (m1 + m3) % 2,
        partial(measure_arbitrary_basis, angle=omega),
        partial(measure_arbitrary_basis, angle=-omega),
    )(plane="XY", wires=graph_wires[2], reset=True)

    # corrections based on measurement outcomes
    cond((m1 + m3) % 2, Z)(out_wire)
    cond((m2 + m4) % 2, X)(out_wire)

    # release input qubit and intermediate graph qubits
    q_mgr.release_qubit(in_wire)
    q_mgr.release_qubits(graph_wires[0:-1])

    return out_wire


def rz_stencil(q_mgr, target_idx, angles):
    """A stencil for the RZ gate, expressed in the MBQC formalism. This
    stencil includes byproduct corrections in addition to the measurements."""

    graph_wires = q_mgr.acquire_qubits(4)
    output_idx = graph_wires[-1]

    # Prepare the state
    GraphStatePrep(nx.grid_graph((4,)), wires=graph_wires)

    # entangle input and graph using first qubit
    CZ([target_idx, graph_wires[0]])

    # MBQC Z rotation: X, X, +/- angle, X
    # Reset operations allow qubits to be returned to the pool
    m0 = measure_x(target_idx, reset=True)
    m1 = measure_x(graph_wires[0], reset=True)
    m2 = cond_measure(
        m1,
        partial(measure_arbitrary_basis, angle=angles, reset=True),
        partial(measure_arbitrary_basis, angle=-angles, reset=True),
    )(plane="XY", wires=graph_wires[1])
    m3 = measure_x(graph_wires[2], reset=True)

    # corrections based on measurement outcomes
    cond((m0 + m2) % 2, Z)(graph_wires[3])
    cond((m1 + m3) % 2, X)(graph_wires[3])

    # release input qubit and intermediate graph qubits
    q_mgr.release_qubit(target_idx)
    q_mgr.release_qubits(graph_wires[0:-1])

    return output_idx


def h_stencil(q_mgr, in_wire):
    """A stencil for the Hadamard gate, expressed in the MBQC formalism. This
    stencil includes byproduct corrections in addition to the measurements."""

    graph_wires = q_mgr.acquire_qubits(4)
    out_wire = graph_wires[-1]

    GraphStatePrep(nx.grid_graph((4,)), wires=graph_wires)
    CZ([in_wire, graph_wires[0]])

    m1 = measure_x(in_wire, reset=True)
    m2 = measure_y(graph_wires[0], reset=True)
    m3 = measure_y(graph_wires[1], reset=True)
    m4 = measure_y(graph_wires[2], reset=True)

    cond((m2 + m3) % 2, Z)(out_wire)
    cond((m1 + m3 + m4) % 2, X)(out_wire)

    # release input qubit and intermediate graph qubits
    q_mgr.release_qubit(in_wire)
    q_mgr.release_qubits(graph_wires[0:-1])

    return out_wire


def s_stencil(q_mgr, in_wire):
    """A stencil for the S gate, expressed in the MBQC formalism. This
    stencil includes byproduct corrections in addition to the measurements."""

    graph_wires = q_mgr.acquire_qubits(4)
    out_wire = graph_wires[-1]

    GraphStatePrep(nx.grid_graph((4,)), wires=graph_wires)
    CZ([in_wire, graph_wires[0]])

    m1 = measure_x(in_wire, reset=True)
    m2 = measure_x(graph_wires[0], reset=True)
    m3 = measure_y(graph_wires[1], reset=True)
    m4 = measure_x(graph_wires[2], reset=True)

    cond((m1 + m2 + m3 + 1) % 2, Z)(out_wire)
    cond((m2 + m4) % 2, X)(out_wire)

    # release input qubit and intermediate graph qubits
    q_mgr.release_qubit(in_wire)
    q_mgr.release_qubits(graph_wires[0:-1])

    return out_wire


def cnot_stencil(q_mgr, ctrl_idx, target_idx):
    """A stencil for the CNOT gate, expressed in the MBQC formalism. This
    stencil includes byproduct corrections in addition to the measurements."""

    graph_wires = q_mgr.acquire_qubits(13)

    # Denote the index for the final output state
    output_ctrl_idx = graph_wires[5]
    output_target_idx = graph_wires[12]

    # Prepare the state
    GraphStatePrep(_generate_cnot_graph(), wires=graph_wires)

    # entangle input and graph using first qubit
    CZ([ctrl_idx, graph_wires[0]])
    CZ([target_idx, graph_wires[7]])

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

    x_cor = m2 + m3 + m5 + m6
    z_cor = m1 + m3 + m4 + m5 + m8 + m9 + m11 + 1
    cond(z_cor % 2, Z)(output_ctrl_idx)
    cond(x_cor % 2, X)(output_ctrl_idx)

    # corrections on target
    x_cor = m2 + m3 + m8 + m10 + m12 + m14
    z_cor = m9 + m11 + m13
    cond(z_cor % 2, Z)(output_target_idx)
    cond(x_cor % 2, X)(output_target_idx)

    q_mgr.release_qubit(ctrl_idx)
    q_mgr.release_qubit(target_idx)

    # We can now free all but the last qubit, which has become the new input_idx
    q_mgr.release_qubits(graph_wires[0:5] + graph_wires[6:-1])
    return output_ctrl_idx, output_target_idx


def _generate_cnot_graph():
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


stencils = {RZ: rz_stencil, RotXZX: rot_stencil, S: s_stencil, H: h_stencil, CNOT: cnot_stencil}
