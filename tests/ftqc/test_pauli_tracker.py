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

import random

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
from pennylane.ftqc.pauli_tracker import commute_clifford_op, pauli_prod, pauli_to_xz, xz_to_pauli

_PAULIS = (qml.I, qml.X, qml.Y, qml.Z)

RTOL = 2.5e-1
ATOL = 5e-2


def generate_pauli_list(wire: int, num_ops: int):
    pauli_list = []

    for _ in range(num_ops):
        pauli_list.append(random.choice(_PAULIS)(wire))

    return pauli_list


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


class TestPauliTracker:
    """Test for the pauli tracker related functions."""

    @pytest.mark.parametrize(
        "op, expected",
        [
            (qml.I(0), (0, 0)),
            (qml.X(1), (1, 0)),
            (qml.Y(0), (1, 1)),
            (qml.Z(0), (0, 1)),
            (qml.I, (0, 0)),
            (qml.X, (1, 0)),
            (qml.Y, (1, 1)),
            (qml.Z, (0, 1)),
        ],
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

    @pytest.mark.parametrize("target_wire", [0, 10, 100, 1000])
    @pytest.mark.parametrize("num_ops", [1, 10, 100, 1000])
    def test_pauli_prod(self, target_wire, num_ops):
        pauli_list = generate_pauli_list(target_wire, num_ops)

        x, z = pauli_prod(pauli_list)

        op = xz_to_pauli(x, z)(wires=target_wire)

        op_simplify = qml.prod(*pauli_list).simplify()
        expected_op = op_simplify if isinstance(op_simplify, _PAULIS) else op_simplify.terms()[1][0]

        assert op == expected_op

    @pytest.mark.parametrize("ops", [[]])
    def test_pauli_prod_to_xz_unsupported_error(self, ops):
        with pytest.raises(ValueError):
            _ = pauli_prod(ops)

    @pytest.mark.parametrize("clifford_op", [qml.S, qml.H])
    @pytest.mark.parametrize("wires", [0, 1, 10, 100])
    @pytest.mark.parametrize("pauli", [qml.I, qml.X, qml.Y, qml.Z])
    def test_apply_clifford_ops_one_wire(self, clifford_op, wires, pauli):
        pauli = pauli(wires=wires)
        clifford_op = clifford_op(wires=wires)
        xz = [pauli_to_xz(pauli)]
        new_xz = commute_clifford_op(clifford_op, xz)
        new_x, new_z = new_xz[0]
        new_pauli = xz_to_pauli(new_x, new_z)(wires=wires)

        new_xz_clifford_op = qml.prod(new_pauli, clifford_op).matrix()
        clifford_op_xz = qml.prod(clifford_op, pauli).matrix()

        assert np.allclose(new_xz_clifford_op, clifford_op_xz) | np.allclose(
            new_xz_clifford_op, -clifford_op_xz
        )

    @pytest.mark.parametrize("clifford_op", [qml.CNOT])
    @pytest.mark.parametrize("wires", [[0, 1], [1, 2], [10, 100]])
    @pytest.mark.parametrize("pauli_control", [qml.I, qml.X, qml.Y, qml.Z])
    @pytest.mark.parametrize("pauli_target", [qml.I, qml.X, qml.Y, qml.Z])
    def test_apply_clifford_ops_two_wires(self, clifford_op, wires, pauli_control, pauli_target):
        pauli_control = pauli_control(wires=wires[0])
        pauli_target = pauli_target(wires=wires[1])
        clifford_op = clifford_op(wires=wires)

        xz = [pauli_to_xz(pauli_control), pauli_to_xz(pauli_target)]
        new_xz = commute_clifford_op(clifford_op, xz)

        _xc, _zc = new_xz[0]
        new_pauli_control = xz_to_pauli(_xc, _zc)(wires=wires[0])
        _xt, _zt = new_xz[1]
        new_pauli_target = xz_to_pauli(_xt, _zt)(wires=wires[1])

        new_xz_clifford_op = qml.prod(new_pauli_control, new_pauli_target, clifford_op).matrix()
        clifford_op_xz = qml.prod(clifford_op, pauli_control, pauli_target).matrix()

        assert np.allclose(new_xz_clifford_op, clifford_op_xz) | np.allclose(
            new_xz_clifford_op, -clifford_op_xz
        )

    @pytest.mark.parametrize(
        "clifford_op", [qml.X(0), qml.RZ(phi=0.123, wires=0), qml.RX(phi=0.123, wires=0)]
    )
    @pytest.mark.parametrize("xz", [[(0, 1)]])
    def test_apply_clifford_ops_not_imp(self, clifford_op, xz):
        with pytest.raises(
            NotImplementedError, match="Only qml.H, qml.S and qml.CNOT are supported."
        ):
            _ = commute_clifford_op(clifford_op, xz)

    @pytest.mark.parametrize(
        "clifford_op, xz",
        [
            (qml.S(0), [(0, 1), (1, 1)]),
            (qml.CNOT(wires=[0, 1]), [(0, 1)]),
            (qml.CNOT(wires=[0, 1]), [(0, 0), (0, 0), (0, 0)]),
        ],
    )
    def test_apply_clifford_ops_xz_list_size_mismatch(self, clifford_op, xz):
        with pytest.raises(
            ValueError,
            match="Please ensure that the length of xz matches the number of wires of the clifford_op.",
        ):
            _ = commute_clifford_op(clifford_op, xz)

    @pytest.mark.parametrize(
        "clifford_op, xz",
        [
            (qml.S(0), [(0, 1, 0)]),
            (qml.CNOT(wires=[0, 1]), [(0, 1), (0, 1, 1)]),
            (qml.CNOT(wires=[0, 1]), [(0, 0, 1), (0, 0)]),
            (qml.CNOT(wires=[0, 1]), [(0, 0, 1), (0, 0, 1)]),
        ],
    )
    def test_apply_clifford_ops_xz_tuple_size_mismatch(self, clifford_op, xz):
        with pytest.raises(
            ValueError,
            match="Please ensure there are 2 elements instead of in each tuple in the xz list.",
        ):
            _ = commute_clifford_op(clifford_op, xz)

    @pytest.mark.parametrize(
        "clifford_op, xz",
        [
            (qml.S(0), [(0, 2)]),
            (qml.CNOT(wires=[0, 1]), [(0, 1), (0, 3)]),
            (qml.CNOT(wires=[0, 1]), [(2, 0), (0, 0)]),
        ],
    )
    def test_apply_clifford_ops_xz_value_err(self, clifford_op, xz):
        with pytest.raises(
            ValueError,
            match="Please ensure xz are either 0 or 1.",
        ):
            _ = commute_clifford_op(clifford_op, xz)


@flaky(max_runs=5)
class TestOfflineCorrection:

    @pytest.mark.parametrize("num_shots", [500])
    @pytest.mark.parametrize("num_iter", [1, 2, 3])
    def test_cnot(self, num_shots, num_iter):
        start_state = generate_random_state(2)
        dev = qml.device("default.qubit", shots=num_shots)

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

        dev_ref = qml.device("default.qubit")

        @qml.qnode(dev_ref)
        def circuit_ref(start_state, num_iter):
            qml.StatePrep(start_state, wires=[0, 1])
            for _ in range(num_iter):
                qml.CNOT(wires=[0, 1])

            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        res_ref = circuit_ref(start_state, num_iter)

        assert np.allclose(cor_res, res_ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("num_shots", [500])
    @pytest.mark.parametrize("num_iter", [1, 2, 3])
    def test_h(self, num_shots, num_iter):
        start_state = generate_random_state(2)
        dev = qml.device("default.qubit", shots=num_shots)

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

        dev_ref = qml.device("default.qubit")

        @qml.qnode(dev_ref)
        def circuit_ref(start_state, num_iter):
            qml.StatePrep(start_state, wires=[0, 1])
            for _ in range(num_iter):
                qml.H(wires=[0])
                qml.H(wires=[1])

            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        res_ref = circuit_ref(start_state, num_iter)

        assert np.allclose(cor_res, res_ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("num_shots", [500])
    @pytest.mark.parametrize("num_iter", [1, 2, 3])
    def test_s(self, num_shots, num_iter):
        start_state = generate_random_state(2)
        dev = qml.device("default.qubit", shots=num_shots)

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

        dev_ref = qml.device("default.qubit")

        @qml.qnode(dev_ref)
        def circuit_ref(start_state, num_iter):
            qml.StatePrep(start_state, wires=[0, 1])
            for _ in range(num_iter):
                qml.S(wires=[0])
                qml.S(wires=[1])

            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        res_ref = circuit_ref(start_state, num_iter)

        assert np.allclose(cor_res, res_ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("num_shots", [500])
    def test_clifford(self, num_shots):
        start_state = generate_random_state(2)
        dev = qml.device("default.qubit", shots=num_shots)

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

        dev_ref = qml.device("default.qubit")

        @qml.qnode(dev_ref)
        def circuit_ref(start_state):
            qml.StatePrep(start_state, wires=[0, 1])
            qml.CNOT(wires=[0, 1])
            qml.S(wires=[0])
            qml.H(wires=[1])

            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        res_ref = circuit_ref(start_state)

        assert np.allclose(cor_res, res_ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("num_shots", [500])
    def test_clifford_paulis(self, num_shots):
        start_state = generate_random_state(2)
        dev = qml.device("default.qubit", shots=num_shots)

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

        dev_ref = qml.device("default.qubit")

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
    @pytest.mark.parametrize("num_shots", [500])
    def test_clifford_paulis_tensorprod(self, p0, p1, num_shots):
        start_state = generate_random_state(2)
        dev = qml.device("default.qubit", shots=num_shots)

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

        dev_ref = qml.device("default.qubit")

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
