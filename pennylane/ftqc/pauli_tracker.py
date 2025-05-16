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

r"""
This module contains Pauli Tracking functions.
"""
import copy
import itertools
from typing import List, Tuple

import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import Operator

_ENCODE_XZ_OPS = {
    qml.I: (0, 0),
    qml.X: (1, 0),
    qml.Y: (1, 1),
    qml.Z: (0, 1),
}

_DECODE_XZ = {
    (0, 0): qml.I,
    (1, 0): qml.X,
    (1, 1): qml.Y,
    (0, 1): qml.Z,
}

_CLIFFORD_TABLEAU = {
    qml.H: [[qml.Z, qml.X]],
    qml.S: [[qml.Y, qml.Z]],
    qml.CNOT: [[qml.X, qml.Z, qml.I, qml.Z], [qml.X, qml.I, qml.X, qml.Z]],
}

_PAULIS = (qml.X, qml.Y, qml.Z, qml.I)

_GATE_SET_SUPPORTED = (qml.X, qml.Y, qml.Z, qml.I, qml.H, qml.S, qml.CNOT)

_MBQC_GATES_SUPPORTED = {
    qml.H: {"meas_len": 4, "cor": [[0, 2, 3], [1, 2], [0, 0]]},
    qml.S: {"meas_len": 4, "cor": [[1, 3], [0, 1, 2], [0, 1]]},
    qml.CNOT: {
        "meas_len": 13,
        "cor": [
            [1, 2, 4, 5],
            [0, 2, 3, 4, 6, 7, 9],
            [1, 2, 6, 8, 10, 12],
            [7, 9, 11],
            [0, 1, 0, 0],
        ],
    },
}


def pauli_encode_xz(op: Operator) -> Tuple[np.uint8, np.uint8]:
    """
    Encode a `Pauli` operator to its `xz` representation up to a global phase, i.e., :math:`encode_{xz}(Pauli)=(x,z)=X^xZ^z)`, where
    :math:`x` is the exponent of the :class:`~pennylane.X` and :math:`z` is the exponent of
    the :class:`~pennylane.Z`, meaning :math:`encode_{xz}(I) = (0, 0)`, :math:`encode_{xz}(X) = (1, 0)`,
    :math:`encode_{xz}(Y) = (1, 1)` and :math:`encode_{xz}(Z) = (0, 1)`.

    Args:
        op (qml.operation.Operator): A Pauli operator.

    Return:
        A tuple of xz encoding data, :math:`x` is the exponent of the :class:`~pennylane.X`, :math:`z` is the exponent of
    the :class:`~pennylane.Z`.
    """

    if op in _ENCODE_XZ_OPS:
        return _ENCODE_XZ_OPS[op]
    print(op)
    raise NotImplementedError(f"{op.name} gate does not support xz encoding.")


def xz_decode_pauli(x: np.uint8, z: np.uint8):
    """
    Decode a x, z to a Pauli operator.

    Args:
        x (np.uint8) : Exponent of :class:`~pennylane.X` in the Pauli record.
        z (np.uint8) : Exponent of :class:`~pennylane.Z` in the Pauli record.

    Return:
        A Pauli operator.
    """
    if x in [0, 1] and z in [0, 1]:
        return _DECODE_XZ[(x, z)]
    raise ValueError("x and z should either 0 or 1.")


def pauli_prod_to_xz(ops: List[Operator]) -> Operator:
    """
    Get the result of a product of list of Pauli operators. The result is encoded with `xz` representation.

    Args:
        ops (List[qml.operation.Operator]): A list of Pauli operators with the same target wire.

    Return:
        A Pauli operator.
    """

    if len(ops) == 0:
        raise ValueError("Please ensure that a valid list of operators are passed to the method.")

    res_x, res_z = pauli_encode_xz(ops.pop())

    while len(ops) > 0:
        x, z = pauli_encode_xz(ops.pop())
        res_x ^= x
        res_z ^= z
    return xz_decode_pauli(res_x, res_z)


def apply_clifford_op(clifford_op: Operator, paulis: list[Operator]):
    """Conjugate a xz encoded ops to a new xz encoded ops with a given
    Clifford op.

        Args:
            clifford_op (qml.operation.Operator): A Clifford operator class. Supported operators are: :class:`qml.S`, :class:`qml.H`, :class:`qml.CNOT`.
            paulis (List): A list of Pauli operator

        Return:
            A list of Pauli operators that clifford_op conjugates the paulis to.
    """
    if clifford_op not in _CLIFFORD_TABLEAU:
        raise NotImplementedError("Only qml.H, qml.S and qml.CNOT are supported.")

    if not all(pauli in _PAULIS for pauli in paulis):
        raise ValueError("Please ensure the operator passed in are Paulis")

    if clifford_op.num_wires != len(paulis):
        raise ValueError(
            "Please ensure the number of Paulis matches the number of wires of the Clifford gate."
        )

    if all(pauli == qml.I for pauli in paulis):
        return paulis

    xz = [pauli_encode_xz(pauli) for pauli in paulis]
    xz = tuple(itertools.chain.from_iterable(xz))

    # A Clifford gate conjugate non-Identify Pauli ops to a new Pauli ops
    new_ops = []
    nonzero_indices = []
    for idx, element in enumerate(xz):
        if element == 1:
            nonzero_indices.append(idx)

    # Get Paulis prod for each target wire
    for table_row in _CLIFFORD_TABLEAU[clifford_op]:
        ps = []
        for idx in nonzero_indices:
            ps.append(table_row[idx])
        new_ops.append(pauli_prod_to_xz(ps))
    return new_ops


def _parse_mid_measurements(tape: qml.tape.QuantumScript, mid_meas: List) -> List:
    r"""Parse a serial of mid-measurement results of a quantum tape with only Pauli operators (:class:`~pennylane.PauliY`, :class:`~pennylane.PauliZ` and :class:`~pennylane.Identity`) and a
    set of Clifford gates (:class:`~pennylane.Hadamard`, :class:`~pennylane.S`, :class:`~pennylane.CNOT`) and the Clifford gates mentioned above are measured in the way defined in `Raussendorf et al. <https://arxiv.org/abs/quant-ph/0301052>`__.

    For :class:`~pennylane.S` operations, the measurements take on the four qubits out of a cluster(chain) state with five qubits and
    the corresponding measurements would be `X-basis`, `X-basis`, `Y-basis` and `X-basis`. The byproduct operator is $by_op = \sigma_x^{s_1+s_3}\sigma_z^{s_0+s_1+s_2+1}$
    and $s_i$ means measurement results of `i`th qubit in the cluster state. Note that the indexing here follows the `C` convention instead of
    `Fortran` convention used in the `Raussendorf et al. <https://arxiv.org/abs/quant-ph/0301052>`__.

    For :class:`~pennylane.H` operations, the measurements take on the four qubits out of a cluster(chain) state with five qubits and
    the corresponding measurements would be `X-basis`, `Y-basis`, `Y-basis` and `Y-basis`. The byproduct operator is $by_op = \sigma_x^{s_0+s_2+s_3}\sigma_z^{s_1+s_2}$.

    For :class:`~pennylane.CNOT` operations, the measurements take on the thirteen qubits out of a cluster(2D) state with fifteen qubits and
    the corresponding measurements would be `X-basis`(0), `Y-basis`(1), `Y-basis`(2), `Y-basis`(3), `Y-basis`(4), `Y-basis`(5), `Y-basis`(7),
    `X-basis`(8), `X-basis`(9), `X-basis`(10), `Y-basis`(11), `X-basis`(12) and `X-basis`(13). The byproduct operator for the control wire is
    $by_op_{ctrl} = \sigma_x^{s_1+s_2+s_4+s_5}\sigma_z^{s_0+s_2+s_3+s_4+s_7+s_8+s_10+1}$. The byproduct operator for the target wire is:
    $by_op_{tgt} = \sigma_x^{s_1+s_2+s_7+s_9+s_11+s_13}\sigma_z^{s_8+s_10+s_12}$.

    **TBD**
    - Do we want to assume that the tape is preprocessed already, which means Clifford gates are moved to the beginning of the circuit?
    - if yes, are Paulis consolidated?

    Args:
        tape (qml.tape.QuantumScript): The quantum tape in the standard circuit mode (Gates are not transformed into the MBQC formalism).
        mid_meas (list): Mid-measurements results.

    Returns:
        `byproduct` ops list and `operation` in a reversed manner. Each tuple contains the index of the corresponding Clifford gate together with the byproduct operation encoded
        with `X`, `Z` and the corresponding wire.
    """
    ops = copy.copy(tape.operations)

    by_ops = []

    mid_meas_offset = 0
    for op in ops:
        if type(op) in _CLIFFORD_TABLEAU:
            cor = []
            # There could be X and Z corrections for each wire
            for i in range(2 * op.num_wires):
                cor.append(
                    sum(
                        [
                            mid_meas[mid_meas_offset + idx]
                            for idx in _MBQC_GATES_SUPPORTED[type(op)]["cor"][i]
                        ]
                    )
                )

            # Add a const 0 or 1 and apply commutate rules
            for idx, add_cor in enumerate(_MBQC_GATES_SUPPORTED[type(op)]["cor"][-1]):
                cor[idx] += add_cor
                cor[idx] &= 1

            by_op = []
            for i in range(op.num_wires):
                by_op.append(xz_decode_pauli(cor[0 + 2 * i], cor[1 + 2 * i]))

            by_ops.append(by_op)

            # Update the mid_measurement offset
            mid_meas_offset += _MBQC_GATES_SUPPORTED[type(op)]["meas_len"]
    # To use list in a stack manner
    by_ops.reverse()
    ops.reverse()
    return by_ops, ops


def get_pauli_record(num_wires: int, by_ops: List, ops: List):
    """Track the Pauli/byproduct ops represented in the xz encoding manner.

    Args:
        num_wires (int): Number of wires of the quantum state.
        by_ops (list): List of byproduct operators for Clifford gates
        ops (list): List of Clifford/Pauli gates

    Return:
        Final tracked Paulis represented by xz.

    """
    pauli_record = [qml.Identity] * num_wires

    while len(by_ops) or len(ops):
        op = ops.pop()
        wires = list(op.wires)

        # Get the recorded pauli
        paulis = [pauli_record[wire] for wire in wires]

        # Updated xz
        new_paulis = []

        if type(op) in _CLIFFORD_TABLEAU:
            # Step 1: Conjugate recorded Paulis to new Paulis
            pauli_conjugated = apply_clifford_op(type(op), paulis)

            # Step 2: Update the x, z record with the byproduct by_op
            by_op = by_ops.pop()
            for b_op, p_conj in zip(by_op, pauli_conjugated):
                new_paulis.append(pauli_prod_to_xz([p_conj, b_op]))

        else:  # branch for Paulis
            # Conjugate step is skipped. Update the x, z record with the Pauli
            paulis.extend([type(op)])
            new_paulis.append(pauli_prod_to_xz(paulis))

        # Assign the updated the xz to the x, z record
        for idx, wire in enumerate(wires):
            pauli_record[wire] = new_paulis[idx]

    return pauli_record


def _apply_measurement_correction_rule(
    pauli: qml.operation.Operator, ob: qml.operation.Operator
) -> np.int8:
    """Get the phase correction factor based on the $X$ recorded of the target wire and the corresponding
    observable. Note that we only support corrections for `Z-basis` measurements.

        Args:
            x (np.uint8): x recorded in the xz tracking.
            z (np.unit8): z recorded in the xz tracking.
            ob (qml.operation.Observable): Observable of the measurement.

        Return:
            Phase correction factor.
    """
    if isinstance(ob, qml.Z):
        return -1 if pauli == qml.X or pauli == qml.Y else 1

    if isinstance(ob, qml.X):
        return -1 if pauli == qml.Z or pauli == qml.Y else 1

    if isinstance(ob, qml.Y):
        return -1 if pauli == qml.X or pauli == qml.Z else 1

    if isinstance(ob, qml.I):
        return 1


def get_measurements_corrections(tape: qml.tape.QuantumScript, pauli_record: List):
    """Get phase correction factor for all measurements in a tape. The phase correction factor
    is calculated based on the measurement observables with the corresponding recorded xz.
        Args:
            tape (tape: qml.tape.QuantumScript): A quantum tape.
            x (np.array): x record for each wire.
            z (np.array): z record for each wire.
        Return:
            Phase correction factor for all measurements.
    """
    phase_cor = [1] * len(tape.measurements)
    for idx, measurement in enumerate(tape.measurements):
        obs = measurement.obs
        if isinstance(obs, _PAULIS):
            # branch for NamedObs
            phase_cor[idx] *= _apply_measurement_correction_rule(
                pauli_record[obs.wires[0]], measurement.obs
            )
        elif isinstance(obs, qml.ops.Prod):
            # branch for TensorProd
            obs = measurement.obs.decomposition()
            for ob in obs:
                phase_cor[idx] *= _apply_measurement_correction_rule(pauli_record[ob.wires[0]], ob)
        else:
            raise NotImplementedError(f"{obs.name} is not supported.")
    return phase_cor


def get_byproduct_corrections(tape: qml.tape.QuantumScript, mid_meas: List):
    r"""Get measurement correction coefficients offline with a quantum script and mid-measurement results for each shot.
    The mid measurement results are first parsed with the quantum script to get the byproduct operations for each Clifford
    gates. Note that byproduct operations and ops are stored with list and used in a stack manner. The calculation iteratively
    pops out the first operation in the tape and applies conjugate rules for the first byproduct ops in the byproduct stack and
    then the results are commutated to the byproduct of the current operations in the tape if it is a Clifford gate. The calculation
    starts from applying conjugate rules for :class:`qml.I` gate or $encode\_xz(x,z)=(0,0)$ to the first gate in the tape. The
    measurement corrections are returned based on the X operations in the xz encoding record.

    Args:
        tape (tape: qml.tape.QuantumScript): A Clifford quantum tape with Paulis, qml.H, qml.S and qml.CNOT in the standard circuit formalism.
        mid_meas (list): MidMeasurement results per shot.

    **Note**
    This work is to be integrated into the MBQC transform pipeline.

    **Example:**

        .. code-block:: python3

            import pennylane as qml

            from pennylane.ftqc import diagonalize_mcms, generate_lattice, measure_x, measure_y
            from pennylane.ftqc import GraphStatePrep

            from offline_byprod_correction import get_byproduct_corrections
            import numpy as np


            def generate_random_state(n):
                seed_value = 42  # You can use any integer as the seed
                np.random.seed(seed_value)
                input_state = np.random.random(2**n) + 1j * np.random.random(2**n)
                return input_state / np.linalg.norm(input_state)


            def generate_rot_gate_graph():
                lattice = generate_lattice([4], "chain")
                return lattice.graph


            num_shots = 1000
            dev = qml.device("default.qubit", shots=num_shots)

            @diagonalize_mcms
            @qml.qnode(dev, mcm_method="one-shot")
            def circ(start_state):
                qml.StatePrep(start_state, wires=[0])
                GraphStatePrep(generate_rot_gate_graph(), wires=[1, 2, 3, 4])
                qml.CZ(wires=[0, 1])
                m0 = measure_x(0, reset=True)
                m1 = measure_y(1, reset=True)
                m2 = measure_y(2, reset=True)
                m3 = measure_y(3, reset=True)

                GraphStatePrep(generate_rot_gate_graph(), wires=[3, 2, 1, 0])
                qml.CZ(wires=[3, 4])
                m4 = measure_x(4, reset=True)
                m5 = measure_y(3, reset=True)
                m6 = measure_y(2, reset=True)
                m7 = measure_y(1, reset=True)

                GraphStatePrep(generate_rot_gate_graph(), wires=[1, 2, 3, 4])
                qml.CZ(wires=[0, 1])
                m8 = measure_x(0, reset=True)
                m9 = measure_y(1, reset=True)
                m10 = measure_y(2, reset=True)
                m11 = measure_y(3, reset=True)

                return (
                    qml.sample(qml.Z(4)),
                    qml.sample(m0),
                    qml.sample(m1),
                    qml.sample(m2),
                    qml.sample(m3),
                    qml.sample(m4),
                    qml.sample(m5),
                    qml.sample(m6),
                    qml.sample(m7),
                    qml.sample(m8), qml.sample(m9), qml.sample(m10), qml.sample(m11)
                )


            res = circ(generate_random_state(1))

            ops = [qml.H(wires=[0]), qml.H(wires=[0]), qml.H(wires=[0])]
            measurements = [qml.sample(qml.Z(0))]

            meas_res = res[0]
            mid_meas_res = res[1:]

            script = qml.tape.QuantumScript(ops, measurements, shots=num_shots)

            for i in range(num_shots):
                mid_meas = [row[i] for row in mid_meas_res]
                phase_cor = get_byproduct_corrections(script, mid_meas)
                meas_res[i] = meas_res[i] *phase_cor[0]

            res = np.sum(meas_res) / num_shots

            print(res)

    """
    if not all(isinstance(op, _GATE_SET_SUPPORTED) for op in tape.operations):
        raise NotImplementedError("Not all gate operations in the tape are supported.")

    if not all(res in [0, 1] for res in mid_meas):
        raise ValueError("The mid-measure value should be either 0 or 1.")

    by_ops, ops = _parse_mid_measurements(tape, mid_meas)

    pauli_record = get_pauli_record(tape.num_wires, by_ops, ops)

    return get_measurements_corrections(tape, pauli_record)
