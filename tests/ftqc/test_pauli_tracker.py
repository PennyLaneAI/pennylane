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

import pennylane as qml
from pennylane.ftqc import (
    GraphStatePrep,
    QubitMgr,
    apply_clifford_op,
    diagonalize_mcms,
    generate_lattice,
    get_byproduct_corrections,
    measure_x,
    measure_y,
    pauli_encode_xz,
    pauli_prod_to_xz,
)

TOL = 2e-1

class TestPauliTracker:
    """Test for the pauli tracker related functions."""

    @pytest.mark.parametrize("op", [qml.I, qml.X, qml.Y, qml.Z, qml.S])
    def test_pauli_encode_xz(self, op):
        if op not in [qml.I, qml.X, qml.Y, qml.Z]:
            with pytest.raises(NotImplementedError):
                _ = pauli_encode_xz(op)
        else:
            x, z = pauli_encode_xz(op)
            assert x in [0, 1]
            assert z in [0, 1]

            x_res = 1 if op in [qml.X, qml.Y] else 0
            z_res = 1 if op in [qml.Y, qml.Z] else 0

            assert x == x_res
            assert z == z_res

    @pytest.mark.parametrize(
        "ops, res",
        [
            ([], None),
            ([qml.I], qml.I),
            ([qml.X], qml.X),
            ([qml.Y], qml.Y),
            ([qml.Z], qml.Z),
            ([qml.I, qml.I], qml.I),
            ([qml.I, qml.X], qml.X),
            ([qml.I, qml.Y], qml.Y),
            ([qml.I, qml.Z], qml.Z),
            ([qml.X, qml.I], qml.X),
            ([qml.X, qml.X], qml.I),
            ([qml.X, qml.Y], qml.Z),
            ([qml.X, qml.Z], qml.Y),
            ([qml.Y, qml.I], qml.Y),
            ([qml.Y, qml.X], qml.Z),
            ([qml.Y, qml.Y], qml.I),
            ([qml.Y, qml.Z], qml.X),
            ([qml.Z, qml.I], qml.Z),
            ([qml.Z, qml.X], qml.Y),
            ([qml.Z, qml.Y], qml.X),
            ([qml.Z, qml.Z], qml.I),
            ([qml.X, qml.Y, qml.Z, qml.I, qml.Z], qml.Z),
        ],
    )
    def test_pauli_prod_to_xz(self, ops, res):
        if len(ops) == 0:
            with pytest.raises(
                ValueError,
                match="Please ensure that a valid list of operators are passed to the method.",
            ):
                _ = pauli_prod_to_xz(ops)
        else:
            op = pauli_prod_to_xz(ops)
            assert res == op

    @pytest.mark.parametrize(
        "clifford_op, pauli, res",
        [
            (qml.S, [qml.I], [qml.I]),
            (qml.S, [qml.X], [qml.Y]),
            (qml.S, [qml.Y], [qml.X]),
            (qml.S, [qml.Z], [qml.Z]),
            (qml.H, [qml.I], [qml.I]),
            (qml.H, [qml.X], [qml.Z]),
            (qml.H, [qml.Y], [qml.Y]),
            (qml.H, [qml.Z], [qml.X]),
            (qml.CNOT, [qml.I, qml.I], [qml.I, qml.I]),
            (qml.CNOT, [qml.X, qml.I], [qml.X, qml.X]),
            (qml.CNOT, [qml.Y, qml.I], [qml.Y, qml.X]),
            (qml.CNOT, [qml.Z, qml.I], [qml.Z, qml.I]),
            (qml.CNOT, [qml.I, qml.X], [qml.I, qml.X]),
            (qml.CNOT, [qml.X, qml.X], [qml.X, qml.I]),
            (qml.CNOT, [qml.Y, qml.X], [qml.Y, qml.I]),
            (qml.CNOT, [qml.Z, qml.X], [qml.Z, qml.X]),
            (qml.CNOT, [qml.I, qml.Y], [qml.Z, qml.Y]),
            (qml.CNOT, [qml.X, qml.Y], [qml.Y, qml.Z]),
            (qml.CNOT, [qml.Y, qml.Y], [qml.X, qml.Z]),
            (qml.CNOT, [qml.Z, qml.Y], [qml.I, qml.Y]),
            (qml.CNOT, [qml.I, qml.Z], [qml.Z, qml.Z]),
            (qml.CNOT, [qml.X, qml.Z], [qml.Y, qml.Y]),
            (qml.CNOT, [qml.Y, qml.Z], [qml.X, qml.Y]),
            (qml.CNOT, [qml.Z, qml.Z], [qml.I, qml.Z]),
        ],
    )
    def test_apply_clifford_ops(self, clifford_op, pauli, res):
        new_pauli = apply_clifford_op(clifford_op, pauli)

        assert new_pauli == res

    @pytest.mark.parametrize("clifford_op", [qml.X, qml.RZ, qml.RX, qml.T])
    @pytest.mark.parametrize("paulis", [[qml.I]])
    def test_apply_clifford_ops_not_imp(self, clifford_op, paulis):
        with pytest.raises(
            NotImplementedError, match="Only qml.H, qml.S and qml.CNOT are supported."
        ):
            _ = apply_clifford_op(clifford_op, paulis)

    @pytest.mark.parametrize(
        "clifford_op, paulis",
        [(qml.S, [qml.I, qml.I]), (qml.S, [qml.RZ]), (qml.CNOT, [qml.I])],
    )
    def test_apply_clifford_ops_val_err(self, clifford_op, paulis):
        with pytest.raises(ValueError):
            _ = apply_clifford_op(clifford_op, paulis)


def generate_random_state(n, seed=42):
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


class TestOfflineCorrection:

    @pytest.mark.parametrize("num_shots", [10000])
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

        assert np.allclose(res_ref, cor_res, rtol=TOL)

    @pytest.mark.parametrize("num_shots", [10000])
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

        assert np.allclose(res_ref, cor_res, rtol=TOL)

    @pytest.mark.parametrize("num_shots", [10000])
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

        assert np.allclose(res_ref, cor_res, rtol=TOL)

    @pytest.mark.parametrize("num_shots", [10000])
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

        assert np.allclose(res_ref, cor_res, rtol=TOL)

    @pytest.mark.parametrize("num_shots", [10000])
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

        assert np.allclose(res_ref, cor_res, rtol=TOL)

    @pytest.mark.parametrize("p0", [qml.X, qml.Y, qml.Z, qml.I])
    @pytest.mark.parametrize("p1", [qml.X, qml.Y, qml.Z, qml.I])
    @pytest.mark.parametrize("num_shots", [10000])
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

        assert np.allclose(res_ref, cor_res, rtol=TOL)
