# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the ftqc.pauli_tracker module"""

import networkx as nx
import numpy as np
import pytest
from flaky import flaky

import pennylane as qml
from pennylane.ftqc import (
    GraphStatePrep,
    QubitMgr,
    diagonalize_mcms,
    generate_lattice,
    get_byproduct_corrections,
    measure_x,
    measure_y,
)
from pennylane.ftqc.pauli_tracker import apply_clifford_op, pauli_prod, pauli_to_xz, xz_to_pauli

RTOL = 2.5e-1
ATOL = 2e-2


class TestPauliTracker:
    """Test for the pauli tracker related functions."""

    @pytest.mark.parametrize(
        "op, expected",
        [(qml.I(0), (0, 0)), (qml.X(1), (1, 0)), (qml.Y(0), (1, 1)), (qml.Z(0), (0, 1))],
    )
    def test_pauli_to_xz(self, op, expected):
        xz = pauli_to_xz(op)
        assert xz == expected

    @pytest.mark.parametrize("op", [qml.S(0), qml.CNOT(wires=[0, 1]), qml.H(2)])
    def test_unsuppored_ops_pauli_to_xz(self, op):
        with pytest.raises(NotImplementedError):
            _ = pauli_to_xz(op)

    @pytest.mark.parametrize(
        "x, z, expected", [(0, 0, qml.I), (1, 0, qml.X), (1, 1, qml.Y), (0, 1, qml.Z)]
    )
    def test_xz_to_pauli(self, x, z, expected):
        op = xz_to_pauli(x, z)
        assert op == expected

    @pytest.mark.parametrize("x, z", [(0, -1), (-1, 0), (-1, -1)])
    def test_xz_decode_pauli_unsupported_error(self, x, z):
        with pytest.raises(ValueError):
            _ = xz_to_pauli(x, z)

    @pytest.mark.parametrize(
        "ops, expected",
        [
            ([qml.I(0)], qml.I(0)),
            ([qml.X(1)], qml.X(1)),
            ([qml.Y(2)], qml.Y(2)),
            ([qml.Z(3)], qml.Z(3)),
            ([qml.I(0), qml.I(0)], qml.I(0)),
            ([qml.I(1), qml.X(1)], qml.X(1)),
            ([qml.I(2), qml.Y(2)], qml.Y(2)),
            ([qml.I(3), qml.Z(3)], qml.Z(3)),
            ([qml.X(0), qml.I(0)], qml.X(0)),
            ([qml.X(1), qml.X(1)], qml.I(1)),
            ([qml.X(2), qml.Y(2)], qml.Z(2)),
            ([qml.X(3), qml.Z(3)], qml.Y(3)),
            ([qml.Y(0), qml.I(0)], qml.Y(0)),
            ([qml.Y(1), qml.X(1)], qml.Z(1)),
            ([qml.Y(2), qml.Y(2)], qml.I(2)),
            ([qml.Y(3), qml.Z(3)], qml.X(3)),
            ([qml.Z(0), qml.I(0)], qml.Z(0)),
            ([qml.Z(1), qml.X(1)], qml.Y(1)),
            ([qml.Z(2), qml.Y(2)], qml.X(2)),
            ([qml.Z(3), qml.Z(3)], qml.I(3)),
            ([qml.X(4), qml.Y(4), qml.Z(4), qml.I(4), qml.Z(4)], qml.Z(4)),
        ],
    )
    def test_pauli_prod(self, ops, expected):
        op = pauli_prod(ops)
        assert op == expected

    @pytest.mark.parametrize(
        "ops", [([]), ([qml.X(0), qml.I(1)]), ([qml.X(0), qml.Y(0), qml.Z(1)])]
    )
    def test_pauli_prod_to_xz_unsupported_error(self, ops):
        with pytest.raises(ValueError):
            _ = pauli_prod(ops)

    @pytest.mark.parametrize(
        "clifford_op, pauli, res",
        [
            (qml.S(0), [qml.I(0)], [qml.I(0)]),
            (qml.S(1), [qml.X(1)], [qml.Y(1)]),
            (qml.S(2), [qml.Y(2)], [qml.X(2)]),
            (qml.S(3), [qml.Z(3)], [qml.Z(3)]),
            (qml.H(0), [qml.I(0)], [qml.I(0)]),
            (qml.H(1), [qml.X(1)], [qml.Z(1)]),
            (qml.H(2), [qml.Y(2)], [qml.Y(2)]),
            (qml.H(3), [qml.Z(3)], [qml.X(3)]),
            (qml.CNOT(wires=[0, 1]), [qml.I(0), qml.I(1)], [qml.I(0), qml.I(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.X(0), qml.I(1)], [qml.X(0), qml.X(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.Y(0), qml.I(1)], [qml.Y(0), qml.X(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.Z(0), qml.I(1)], [qml.Z(0), qml.I(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.I(0), qml.X(1)], [qml.I(0), qml.X(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.X(0), qml.X(1)], [qml.X(0), qml.I(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.Y(0), qml.X(1)], [qml.Y(0), qml.I(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.Z(0), qml.X(1)], [qml.Z(0), qml.X(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.I(0), qml.Y(1)], [qml.Z(0), qml.Y(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.X(0), qml.Y(1)], [qml.Y(0), qml.Z(1)]),
            (qml.CNOT(wires=[0, 2]), [qml.Y(0), qml.Y(2)], [qml.X(0), qml.Z(2)]),
            (qml.CNOT(wires=[0, 2]), [qml.Z(0), qml.Y(2)], [qml.I(0), qml.Y(2)]),
            (qml.CNOT(wires=[1, 2]), [qml.I(1), qml.Z(2)], [qml.Z(1), qml.Z(2)]),
            (qml.CNOT(wires=[1, 2]), [qml.X(1), qml.Z(2)], [qml.Y(1), qml.Y(2)]),
            (qml.CNOT(wires=[1, 2]), [qml.Y(1), qml.Z(2)], [qml.X(1), qml.Y(2)]),
            (qml.CNOT(wires=[1, 2]), [qml.Z(1), qml.Z(2)], [qml.I(1), qml.Z(2)]),
            (qml.CNOT(wires=[0, 1]), [qml.I(1), qml.I(0)], [qml.I(0), qml.I(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.X(1), qml.I(0)], [qml.I(0), qml.X(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.Y(1), qml.I(0)], [qml.Z(0), qml.Y(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.Z(1), qml.I(0)], [qml.Z(0), qml.Z(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.I(1), qml.X(0)], [qml.X(0), qml.X(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.X(1), qml.X(0)], [qml.X(0), qml.I(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.Y(1), qml.X(0)], [qml.Y(0), qml.Z(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.Z(1), qml.X(0)], [qml.Y(0), qml.Y(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.I(1), qml.Y(0)], [qml.Y(0), qml.X(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.X(1), qml.Y(0)], [qml.Y(0), qml.I(1)]),
            (qml.CNOT(wires=[0, 2]), [qml.Y(2), qml.Y(0)], [qml.X(0), qml.Z(2)]),
            (qml.CNOT(wires=[0, 2]), [qml.Z(2), qml.Y(0)], [qml.X(0), qml.Y(2)]),
            (qml.CNOT(wires=[1, 2]), [qml.I(2), qml.Z(1)], [qml.Z(1), qml.I(2)]),
            (qml.CNOT(wires=[1, 2]), [qml.X(2), qml.Z(1)], [qml.Z(1), qml.X(2)]),
            (qml.CNOT(wires=[1, 2]), [qml.Y(2), qml.Z(1)], [qml.I(1), qml.Y(2)]),
            (qml.CNOT(wires=[1, 2]), [qml.Z(2), qml.Z(1)], [qml.I(1), qml.Z(2)]),
        ],
    )
    def test_apply_clifford_ops(self, clifford_op, pauli, res):
        new_pauli = apply_clifford_op(clifford_op, pauli)

        assert new_pauli == res

    @pytest.mark.parametrize(
        "clifford_op", [qml.X(0), qml.RZ(phi=0.123, wires=0), qml.RX(phi=0.123, wires=0)]
    )
    @pytest.mark.parametrize("paulis", [[qml.I(0)]])
    def test_apply_clifford_ops_not_imp(self, clifford_op, paulis):
        with pytest.raises(
            NotImplementedError, match="Only qml.H, qml.S and qml.CNOT are supported."
        ):
            _ = apply_clifford_op(clifford_op, paulis)

    @pytest.mark.parametrize(
        "clifford_op, paulis",
        [
            (qml.S(0), [qml.S(0)]),
            (qml.CNOT(wires=[0, 1]), [qml.H(0), qml.H(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.I(0), qml.H(1)]),
        ],
    )
    def test_apply_clifford_ops_val_err(self, clifford_op, paulis):
        with pytest.raises(ValueError, match="Please ensure the operator passed in are Paulis."):
            _ = apply_clifford_op(clifford_op, paulis)

    @pytest.mark.parametrize(
        "clifford_op, paulis",
        [
            (qml.CNOT(wires=[0, 1]), [qml.X(1), qml.Y(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.I(0), qml.Z(0)]),
        ],
    )
    def test_apply_clifford_ops_pauli_wire_err(self, clifford_op, paulis):
        with pytest.raises(
            ValueError, match="Please ensure each Pauli target at a different wire."
        ):
            _ = apply_clifford_op(clifford_op, paulis)

    @pytest.mark.parametrize(
        "clifford_op, paulis",
        [
            (qml.S(0), [qml.I(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.X(2), qml.X(1)]),
            (qml.CNOT(wires=[0, 1]), [qml.I(0), qml.Y(3)]),
        ],
    )
    def test_apply_clifford_ops_wire_mismatch_err(self, clifford_op, paulis):
        with pytest.raises(
            ValueError, match="Please the target wires of Clifford op match those of Paulis."
        ):
            _ = apply_clifford_op(clifford_op, paulis)


def generate_random_state(n, seed=0):
    rng = np.random.default_rng(seed=seed)
    input_state = rng.random(2**n) + 1j * rng.random(2**n)
    return input_state / np.linalg.norm(input_state)


def generate_rot_gate_graph():
    lattice = generate_lattice([4], "chain")
    return lattice.graph


def generate_cnot_graph():
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


def h_stencil(q_mgr, target_idx):
    # Acquire 4 free qubit indices
    graph_wires = q_mgr.acquire_qubits(4)

    # Denote the index for the final output state
    output_idx = graph_wires[-1]

    # Prepare the state
    qml.ftqc.GraphStatePrep(generate_rot_gate_graph(), wires=graph_wires)

    # entangle input and graph using first qubit
    qml.CZ([target_idx, graph_wires[0]])

    m = []
    # MBQC Hadamard: X, Y, Y, Y
    # Reset operations allow qubits to be returned to the pool
    m0 = measure_x(target_idx, reset=True)
    m1 = measure_y(graph_wires[0], reset=True)
    m2 = measure_y(graph_wires[1], reset=True)
    m3 = measure_y(graph_wires[2], reset=True)

    m.extend([m0, m1, m2, m3])

    # The input qubit can be freed and the output qubit becomes the next iteration's input
    q_mgr.release_qubit(target_idx)
    # We can now free all but the last qubit, which has become the new input_idx
    q_mgr.release_qubits(graph_wires[0:-1])

    return output_idx, m


def s_stencil(q_mgr, target_idx):
    # Acquire 4 free qubit indices
    graph_wires = q_mgr.acquire_qubits(4)

    # Denote the index for the final output state
    output_idx = graph_wires[-1]

    # Prepare the state
    GraphStatePrep(generate_rot_gate_graph(), wires=graph_wires)

    # entangle input and graph using first qubit
    qml.CZ([target_idx, graph_wires[0]])

    m = []
    # MBQC Z rotation: X, X, Y, X
    # Reset operations allow qubits to be returned to the pool
    m0 = measure_x(target_idx, reset=True)
    m1 = measure_x(graph_wires[0], reset=True)
    m2 = measure_y(graph_wires[1], reset=True)
    m3 = measure_x(graph_wires[2], reset=True)

    m.extend([m0, m1, m2, m3])

    # The input qubit can be freed and the output qubit becomes the next iteration's input
    q_mgr.release_qubit(target_idx)
    # We can now free all but the last qubit, which has become the new input_idx
    q_mgr.release_qubits(graph_wires[0:-1])

    return output_idx, m


def cnot_stencil(q_mgr, ctrl_idx, target_idx):
    graph_wires = q_mgr.acquire_qubits(13)

    # Denote the index for the final output state
    output_ctrl_idx = graph_wires[5]
    output_target_idx = graph_wires[12]

    # Prepare the state
    GraphStatePrep(generate_cnot_graph(), wires=graph_wires)

    # entangle input and graph using first qubit
    qml.CZ([ctrl_idx, graph_wires[0]])
    qml.CZ([target_idx, graph_wires[7]])

    m = []
    m1 = qml.ftqc.measure_x(ctrl_idx, reset=True)
    m2 = qml.ftqc.measure_y(graph_wires[0], reset=True)
    m3 = qml.ftqc.measure_y(graph_wires[1], reset=True)
    m4 = qml.ftqc.measure_y(graph_wires[2], reset=True)
    m5 = qml.ftqc.measure_y(graph_wires[3], reset=True)
    m6 = qml.ftqc.measure_y(graph_wires[4], reset=True)

    m8 = qml.ftqc.measure_y(graph_wires[6], reset=True)

    m9 = qml.ftqc.measure_x(target_idx, reset=True)
    m10 = qml.ftqc.measure_x(graph_wires[7], reset=True)
    m11 = qml.ftqc.measure_x(graph_wires[8], reset=True)
    m12 = qml.ftqc.measure_y(graph_wires[9], reset=True)
    m13 = qml.ftqc.measure_x(graph_wires[10], reset=True)
    m14 = qml.ftqc.measure_x(graph_wires[11], reset=True)

    m.extend([m1, m2, m3, m4, m5, m6, m8, m9, m10, m11, m12, m13, m14])
    q_mgr.release_qubit(ctrl_idx)
    q_mgr.release_qubit(target_idx)

    # We can now free all but the last qubit, which has become the new input_idx
    q_mgr.release_qubits(graph_wires[0:5] + graph_wires[6:-1])
    return output_ctrl_idx, output_target_idx, m


@flaky(max_runs=5)
class TestOfflineCorrection:

    @pytest.mark.parametrize("num_shots", [1000])
    @pytest.mark.parametrize("num_iter", [1, 2, 3])
    def test_cnot(self, num_shots, num_iter):
        start_state = generate_random_state(2)
        dev = qml.device("lightning.qubit", shots=num_shots)

        @diagonalize_mcms
        @qml.qnode(dev, mcm_method="one-shot")
        def circuit_mbqc(start_state, num_iter):
            q_mgr = QubitMgr(num_qubits=15, start_idx=0)
            ctrl_idx, target_idx = q_mgr.acquire_qubits(2)
            wire_map = {0: ctrl_idx, 1: target_idx}

            # prep input node
            qml.StatePrep(start_state, wires=[wire_map[0], wire_map[1]])
            mid_meas = []
            for _ in range(num_iter):
                wire_map[0], wire_map[1], m = cnot_stencil(q_mgr, wire_map[0], wire_map[1])
                mid_meas.extend(m)

            mid_meas = [qml.Z(wire_map[0]), qml.Z(wire_map[1])] + mid_meas

            return [qml.sample(op=m) for m in mid_meas]

        res = circuit_mbqc(start_state, num_iter)
        ops = []
        for _ in range(num_iter):
            ops.extend([qml.CNOT(wires=[0, 1])])

        measurements = [qml.sample(qml.Z(0)), qml.sample(qml.Z(1))]

        script = qml.tape.QuantumScript(ops, measurements, shots=num_shots)

        meas_res = res[0 : len(measurements)]

        mid_meas_res = res[len(measurements) :]

        for i in range(num_shots):
            mid_meas = [row[i] for row in mid_meas_res]
            phase_cor = get_byproduct_corrections(script, mid_meas)
            for j in range(len(measurements)):
                meas_res[j][i] = meas_res[j][i] * phase_cor[j]

        cor_res = []
        for i in range(len(measurements)):
            cor_res.append(np.sum(meas_res[i]) / num_shots)

        dev_ref = qml.device("lightning.qubit")

        @qml.qnode(dev_ref)
        def circuit_ref(start_state, num_iter):
            qml.StatePrep(start_state, wires=[0, 1])
            for _ in range(num_iter):
                qml.CNOT(wires=[0, 1])

            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        res_ref = circuit_ref(start_state, num_iter)

        assert np.allclose(cor_res, res_ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("num_shots", [1000])
    @pytest.mark.parametrize("num_iter", [1, 2, 3])
    def test_h(self, num_shots, num_iter):
        start_state = generate_random_state(2)
        dev = qml.device("lightning.qubit", shots=num_shots)

        @diagonalize_mcms
        @qml.qnode(dev, mcm_method="one-shot")
        def circuit_mbqc(start_state, num_iter):
            q_mgr = QubitMgr(num_qubits=6, start_idx=0)
            ctrl_idx, target_idx = q_mgr.acquire_qubits(2)
            wire_map = {0: ctrl_idx, 1: target_idx}

            # prep input node
            qml.StatePrep(start_state, wires=[wire_map[0], wire_map[1]])
            mid_meas = []
            for _ in range(num_iter):
                wire_map[0], m = h_stencil(q_mgr, wire_map[0])
                mid_meas.extend(m)
                wire_map[1], m = h_stencil(q_mgr, wire_map[1])
                mid_meas.extend(m)

            mid_meas = [qml.Z(wire_map[0]), qml.Z(wire_map[1])] + mid_meas

            return [qml.sample(op=m) for m in mid_meas]

        res = circuit_mbqc(start_state, num_iter)
        ops = []
        for _ in range(num_iter):
            ops.extend([qml.H(wires=[0]), qml.H(wires=[1])])

        measurements = [qml.sample(qml.Z(0)), qml.sample(qml.Z(1))]

        script = qml.tape.QuantumScript(ops, measurements, shots=num_shots)

        meas_res = res[0 : len(measurements)]

        mid_meas_res = res[len(measurements) :]

        for i in range(num_shots):
            mid_meas = [row[i] for row in mid_meas_res]
            phase_cor = get_byproduct_corrections(script, mid_meas)
            for j in range(len(measurements)):
                meas_res[j][i] = meas_res[j][i] * phase_cor[j]

        cor_res = []
        for i in range(len(measurements)):
            cor_res.append(np.sum(meas_res[i]) / num_shots)

        dev_ref = qml.device("lightning.qubit")

        @qml.qnode(dev_ref)
        def circuit_ref(start_state, num_iter):
            qml.StatePrep(start_state, wires=[0, 1])
            for _ in range(num_iter):
                qml.H(wires=[0])
                qml.H(wires=[1])

            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        res_ref = circuit_ref(start_state, num_iter)

        assert np.allclose(cor_res, res_ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("num_shots", [1000])
    @pytest.mark.parametrize("num_iter", [1, 2, 3])
    def test_s(self, num_shots, num_iter):
        start_state = generate_random_state(2)
        dev = qml.device("lightning.qubit", shots=num_shots)

        @diagonalize_mcms
        @qml.qnode(dev, mcm_method="one-shot")
        def circuit_mbqc(start_state, num_iter):
            q_mgr = QubitMgr(num_qubits=6, start_idx=0)
            ctrl_idx, target_idx = q_mgr.acquire_qubits(2)
            wire_map = {0: ctrl_idx, 1: target_idx}

            # prep input node
            qml.StatePrep(start_state, wires=[wire_map[0], wire_map[1]])
            mid_meas = []
            for _ in range(num_iter):
                wire_map[0], m = s_stencil(q_mgr, wire_map[0])
                mid_meas.extend(m)
                wire_map[1], m = s_stencil(q_mgr, wire_map[1])
                mid_meas.extend(m)

            mid_meas = [qml.Z(wire_map[0]), qml.Z(wire_map[1])] + mid_meas

            return [qml.sample(op=m) for m in mid_meas]

        res = circuit_mbqc(start_state, num_iter)
        ops = []
        for _ in range(num_iter):
            ops.extend([qml.S(wires=[0]), qml.S(wires=[1])])

        measurements = [qml.sample(qml.Z(0)), qml.sample(qml.Z(1))]

        script = qml.tape.QuantumScript(ops, measurements, shots=num_shots)

        meas_res = res[0 : len(measurements)]

        mid_meas_res = res[len(measurements) :]

        for i in range(num_shots):
            mid_meas = [row[i] for row in mid_meas_res]
            phase_cor = get_byproduct_corrections(script, mid_meas)
            for j in range(len(measurements)):
                meas_res[j][i] = meas_res[j][i] * phase_cor[j]

        cor_res = []
        for i in range(len(measurements)):
            cor_res.append(np.sum(meas_res[i]) / num_shots)

        dev_ref = qml.device("lightning.qubit")

        @qml.qnode(dev_ref)
        def circuit_ref(start_state, num_iter):
            qml.StatePrep(start_state, wires=[0, 1])
            for _ in range(num_iter):
                qml.S(wires=[0])
                qml.S(wires=[1])

            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        res_ref = circuit_ref(start_state, num_iter)

        assert np.allclose(cor_res, res_ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("num_shots", [1000])
    def test_clifford(self, num_shots):
        start_state = generate_random_state(2)
        dev = qml.device("lightning.qubit", shots=num_shots)

        @diagonalize_mcms
        @qml.qnode(dev, mcm_method="one-shot")
        def circuit_mbqc(start_state):
            q_mgr = QubitMgr(num_qubits=15, start_idx=0)
            ctrl_idx, target_idx = q_mgr.acquire_qubits(2)
            wire_map = {0: ctrl_idx, 1: target_idx}

            # prep input node
            qml.StatePrep(start_state, wires=[wire_map[0], wire_map[1]])
            mid_meas = []
            wire_map[0], wire_map[1], m = cnot_stencil(q_mgr, wire_map[0], wire_map[1])
            mid_meas.extend(m)
            wire_map[0], m = s_stencil(q_mgr, wire_map[0])
            mid_meas.extend(m)
            wire_map[1], m = h_stencil(q_mgr, wire_map[1])
            mid_meas.extend(m)

            mid_meas = [qml.Z(wire_map[0]), qml.Z(wire_map[1])] + mid_meas

            return [qml.sample(op=m) for m in mid_meas]

        res = circuit_mbqc(start_state)
        ops = []
        ops.extend([qml.CNOT(wires=[0, 1])])
        ops.extend([qml.S(wires=[0])])
        ops.extend([qml.H(wires=[1])])

        measurements = [qml.sample(qml.Z(0)), qml.sample(qml.Z(1))]

        script = qml.tape.QuantumScript(ops, measurements, shots=num_shots)

        meas_res = res[0 : len(measurements)]

        mid_meas_res = res[len(measurements) :]

        for i in range(num_shots):
            mid_meas = [row[i] for row in mid_meas_res]
            phase_cor = get_byproduct_corrections(script, mid_meas)
            for j in range(len(measurements)):
                meas_res[j][i] = meas_res[j][i] * phase_cor[j]

        cor_res = []
        for i in range(len(measurements)):
            cor_res.append(np.sum(meas_res[i]) / num_shots)

        dev_ref = qml.device("lightning.qubit")

        @qml.qnode(dev_ref)
        def circuit_ref(start_state):
            qml.StatePrep(start_state, wires=[0, 1])
            qml.CNOT(wires=[0, 1])
            qml.S(wires=[0])
            qml.H(wires=[1])

            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        res_ref = circuit_ref(start_state)

        assert np.allclose(cor_res, res_ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("num_shots", [1000])
    def test_clifford_paulis(self, num_shots):
        start_state = generate_random_state(2)
        dev = qml.device("lightning.qubit", shots=num_shots)

        @diagonalize_mcms
        @qml.qnode(dev, mcm_method="one-shot")
        def circuit_mbqc(start_state):
            q_mgr = QubitMgr(num_qubits=15, start_idx=0)
            ctrl_idx, target_idx = q_mgr.acquire_qubits(2)
            wire_map = {0: ctrl_idx, 1: target_idx}

            # prep input node
            qml.StatePrep(start_state, wires=[wire_map[0], wire_map[1]])
            mid_meas = []
            wire_map[0], wire_map[1], m = cnot_stencil(q_mgr, wire_map[0], wire_map[1])
            mid_meas.extend(m)
            wire_map[0], m = s_stencil(q_mgr, wire_map[0])
            mid_meas.extend(m)
            wire_map[1], m = h_stencil(q_mgr, wire_map[1])
            mid_meas.extend(m)

            mid_meas = [qml.Z(wire_map[0]), qml.Z(wire_map[1])] + mid_meas

            return [qml.sample(op=m) for m in mid_meas]

        res = circuit_mbqc(start_state)
        ops = []
        ops.extend([qml.X(0)])
        ops.extend([qml.CNOT(wires=[0, 1])])
        ops.extend([qml.Y(1)])
        ops.extend([qml.S(wires=[0])])
        ops.extend([qml.Z(0)])
        ops.extend([qml.H(wires=[1])])

        measurements = [qml.sample(qml.Z(0)), qml.sample(qml.Z(1))]

        script = qml.tape.QuantumScript(ops, measurements, shots=num_shots)

        meas_res = res[0 : len(measurements)]

        mid_meas_res = res[len(measurements) :]

        for i in range(num_shots):
            mid_meas = [row[i] for row in mid_meas_res]
            phase_cor = get_byproduct_corrections(script, mid_meas)
            for j in range(len(measurements)):
                meas_res[j][i] = meas_res[j][i] * phase_cor[j]

        cor_res = []
        for i in range(len(measurements)):
            cor_res.append(np.sum(meas_res[i]) / num_shots)

        dev_ref = qml.device("lightning.qubit")

        @qml.qnode(dev_ref)
        def circuit_ref(start_state):
            qml.StatePrep(start_state, wires=[0, 1])
            qml.X(0)
            qml.CNOT(wires=[0, 1])
            qml.Y(1)
            qml.S(wires=[0])
            qml.Z(0)
            qml.H(wires=[1])

            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        res_ref = circuit_ref(start_state)

        assert np.allclose(cor_res, res_ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("p0", [qml.X, qml.Y, qml.Z, qml.I])
    @pytest.mark.parametrize("p1", [qml.X, qml.Y, qml.Z, qml.I])
    @pytest.mark.parametrize("num_shots", [1000])
    def test_clifford_paulis_tensorprod(self, p0, p1, num_shots):
        start_state = generate_random_state(2)
        dev = qml.device("lightning.qubit", shots=num_shots)

        @diagonalize_mcms
        @qml.qnode(dev, mcm_method="one-shot")
        def circuit_mbqc(start_state):
            q_mgr = QubitMgr(num_qubits=15, start_idx=0)
            ctrl_idx, target_idx = q_mgr.acquire_qubits(2)
            wire_map = {0: ctrl_idx, 1: target_idx}

            # prep input node
            qml.StatePrep(start_state, wires=[wire_map[0], wire_map[1]])
            mid_meas = []
            wire_map[0], wire_map[1], m = cnot_stencil(q_mgr, wire_map[0], wire_map[1])
            mid_meas.extend(m)
            wire_map[0], m = s_stencil(q_mgr, wire_map[0])
            mid_meas.extend(m)
            wire_map[1], m = h_stencil(q_mgr, wire_map[1])
            mid_meas.extend(m)

            mid_meas = [p0(wire_map[0]) @ p1(wire_map[1])] + mid_meas

            return [qml.sample(op=m) for m in mid_meas]

        res = circuit_mbqc(start_state)
        ops = []
        ops.extend([qml.X(0)])
        ops.extend([qml.CNOT(wires=[0, 1])])
        ops.extend([qml.Y(1)])
        ops.extend([qml.S(wires=[0])])
        ops.extend([qml.Z(0)])
        ops.extend([qml.H(wires=[1])])

        measurements = [qml.sample(p0(0) @ p1(1))]

        script = qml.tape.QuantumScript(ops, measurements, shots=num_shots)

        meas_res = res[0 : len(measurements)]

        mid_meas_res = res[len(measurements) :]

        for i in range(num_shots):
            mid_meas = [row[i] for row in mid_meas_res]
            phase_cor = get_byproduct_corrections(script, mid_meas)
            for j in range(len(measurements)):
                meas_res[j][i] = meas_res[j][i] * phase_cor[j]

        cor_res = []
        for i in range(len(measurements)):
            cor_res.append(np.sum(meas_res[i]) / num_shots)

        dev_ref = qml.device("lightning.qubit")

        @qml.qnode(dev_ref)
        def circuit_ref(start_state):
            qml.StatePrep(start_state, wires=[0, 1])
            qml.X(0)
            qml.CNOT(wires=[0, 1])
            qml.Y(1)
            qml.S(wires=[0])
            qml.Z(0)
            qml.H(wires=[1])

            return qml.expval(p0(0) @ p1(1))

        res_ref = circuit_ref(start_state)

        assert np.allclose(cor_res, res_ref, rtol=RTOL, atol=ATOL)
