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

import numpy as np

from pennylane import CNOT, H, I, S, X, Y, Z
from pennylane.operation import Operator
from pennylane.ops import Prod
from pennylane.tape import QuantumScript

_OPS_TO_XZ = {
    I: (0, 0),
    X: (1, 0),
    Y: (1, 1),
    Z: (0, 1),
}

_XZ_TO_OPS = {
    (0, 0): I,
    (1, 0): X,
    (1, 1): Y,
    (0, 1): Z,
}

_PAULIS = (X, Y, Z, I)

_CLIFFORD_TABLEAU = {
    H: [[Z, X]],
    S: [[Y, Z]],
    CNOT: [[X, Z, I, Z], [X, I, X, Z]],
}

_GATE_SET_SUPPORTED = (X, Y, Z, I, H, S, CNOT)

_MBQC_GATES_SUPPORTED = {
    H: {"meas_len": 4, "cor": [[0, 2, 3], [1, 2], [0, 0]]},
    S: {"meas_len": 4, "cor": [[1, 3], [0, 1, 2], [0, 1]]},
    CNOT: {
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


def pauli_to_xz(op: Operator) -> Tuple[np.uint8, np.uint8]:
    r"""
    Convert a `Pauli` operator to its `xz` representation up to a global phase, i.e., :math:`encode_{xz}(Pauli)=(x,z)=X^xZ^z`, where
    :math:`x` is the exponent of the :class:`~pennylane.X` and :math:`z` is the exponent of
    the :class:`~pennylane.Z`, meaning :math:`encode_{xz}(I) = (0, 0)`, :math:`encode_{xz}(X) = (1, 0)`,
    :math:`encode_{xz}(Y) = (1, 1)` and :math:`encode_{xz}(Z) = (0, 1)`.

    Args:
        op (qml.operation.Operator): A Pauli operator.

    Return:
        A tuple of xz encoding data, :math:`x` is the exponent of the :class:`~pennylane.X`, :math:`z` is the exponent of
        the :class:`~pennylane.Z`.

    **Example:**
        The following example shows how the Pauli to XZ works.

        .. code-block:: python3

            from pennylane.ftqc.pauli_tracker import pauli_to_xz
            from pennylane import I
            >>> pauli_to_xz(I(0))
            (0, 0)

        A xz tuple representation is returned for a given Pauli operator.
    """

    if isinstance(op, _PAULIS):
        return _OPS_TO_XZ[type(op)]

    if op in _PAULIS:
        return _OPS_TO_XZ[op]

    raise NotImplementedError(f"{type(op)} gate does not support xz encoding.")


def xz_to_pauli(x: np.uint8, z: np.uint8) -> Operator:
    """
    Convert x, z to a Pauli operator class.

    Args:
        x (np.uint8) : Exponent of :class:`~pennylane.X` in the Pauli record.
        z (np.uint8) : Exponent of :class:`~pennylane.Z` in the Pauli record.

    Return:
        A Pauli operator class.

    **Example:**
        The following example shows how the XZ to Pauli works.

        .. code-block:: python3

            from pennylane.ftqc.pauli_tracker import xz_to_pauli
            >>> xz_to_pauli(0, 0)(wires=0)
            I(0)

        A Pauli operator class is returned for a given xz tuple.
    """
    if x in [0, 1] and z in [0, 1]:
        return _XZ_TO_OPS[(x, z)]
    raise ValueError("x and z should either 0 or 1.")


def pauli_prod(ops: List[Operator]) -> Tuple[np.uint8, np.uint8]:
    """
    Get the result of a product of a list of Pauli operators. The result is a new Pauli operator up to a global phase.

    Args:
        ops (List[qml.operation.Operator]): A list of Pauli operators with the same target wire.

    Return:
        A xz tuple representing a new Pauli operator.

    **Example:**
        The following example shows how the `pauli_prod` works.

        .. code-block:: python3

            from pennylane.ftqc.pauli_tracker import pauli_prod
            from pennylane import I, X, Y, Z
            >>> pauli_prod([I(0),X(0),Y(0),Z(0)])
            (0, 0)

        A Pauli operator is returned for a list of Pauli operator up to a global phase.
    """

    if len(ops) == 0:
        raise ValueError("Please ensure that a valid list of operators are passed to the method.")
    res_x, res_z = pauli_to_xz(ops.pop())

    while len(ops) > 0:
        x, z = pauli_to_xz(ops.pop())
        res_x ^= x
        res_z ^= z

    return (res_x, res_z)


def apply_clifford_op(
    clifford_op: Operator, xz: List[Tuple[np.uint8, np.uint8]]
) -> List[Tuple[np.uint8, np.uint8]]:
    """Commuting a list of Pauli ops to a list of new Pauli ops with a given Clifford op.

    Args:
        clifford_op (Operator): A Clifford operator class. Supported operators are: :class:`qml.S`, :class:`qml.H`, :class:`qml.CNOT`.
        xz (list(tuple)): A list of xz tuples which map to Pauli operators

    Return:
        A list of new xz tuples that the clifford_op commute the xz to.

    **Example:**
        The following example shows how the `pauli_prod` works.

        .. code-block:: python3

            from pennylane.ftqc.pauli_tracker import apply_clifford_op
            from pennylane import I, CNOT
            >>> apply_clifford_op(CNOT(wires=[0,1]), [(0, 0), (0, 0)])
            [(0, 0), (0, 0)]

        A list of xz representation of Pauli operators is returned.
    """
    if type(clifford_op) not in _CLIFFORD_TABLEAU:
        raise NotImplementedError("Only qml.H, qml.S and qml.CNOT are supported.")

    if len(xz) != clifford_op.num_wires:
        raise ValueError(
            "Please ensure that the length of xz matches the number of wires of the clifford_op."
        )

    if not all(len(element) == 2 for element in xz):
        raise ValueError(
            "Please ensure there are 2 elements instead of in each tuple in the xz list."
        )

    xz_flatten = tuple(itertools.chain.from_iterable(xz))

    if not all(element in [0, 1] for element in xz_flatten):
        raise ValueError("Please ensure xz are either 0 or 1.")

    if all(element == (0, 0) for element in xz):
        return xz

    # A Clifford gate conjugate non-Identify Pauli ops to a new Pauli ops
    new_xz = []
    nonzero_indices = [idx for idx, element in enumerate(xz_flatten) if element == 1]

    # Get Paulis prod for each target wire
    for table_row in _CLIFFORD_TABLEAU[type(clifford_op)]:
        ps = [table_row[idx] for idx in nonzero_indices]
        new_xz.append(pauli_prod(ps))
    return new_xz


def _parse_mid_measurements(tape: QuantumScript, mid_meas: List):
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
        tape (QuantumScript): The quantum tape in the standard circuit mode (Gates are not transformed into the MBQC formalism).
        mid_meas (list): Mid-measurements results.

    Returns:
        A list of `byproduct` ops and a list of `operations` in a reversed manner.
    """
    # Copy is explicitly applied here to avoid changes made to the original tape 
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
                by_op.append((cor[0 + 2 * i], cor[1 + 2 * i]))

            by_ops.append(by_op)

            # Update the mid_measurement offset
            mid_meas_offset += _MBQC_GATES_SUPPORTED[type(op)]["meas_len"]
    # To use list in a stack manner
    by_ops.reverse()
    ops.reverse()
    return by_ops, ops


def get_xz_record(num_wires: int, by_ops: List[Tuple[np.uint8, np.uint8]], ops: List[Operator]):
    """Commutate/merge the Pauli/byproduct ops of a Clifford circuit.

    Args:
        num_wires (int): Number of wires of the quantum state.
        by_ops (list): List of byproduct operators for Clifford gates
        ops (list): List of Clifford/Pauli gates

    Return:
        The final recorded x and z for each wire.
    """
    x_record = np.zeros(num_wires, dtype=np.uint8)
    z_record = np.zeros(num_wires, dtype=np.uint8)

    while len(by_ops) or len(ops):
        op = ops.pop()
        wires = list(op.wires)

        # Get the recorded xz
        xz = [(x_record[wire], z_record[wire]) for wire in wires]

        # Updated xz
        new_xz = []

        if type(op) in _CLIFFORD_TABLEAU:
            # Step 1: Commutate the recorded xz with the Clifford gate to a new xz.
            xz_commutated = apply_clifford_op(op, xz)

            # Step 2: Merge the new xz with the byproduct by_op
            by_op = by_ops.pop()
            for _by_op, _xz_comm in zip(by_op, xz_commutated):
                new_xz.append(np.bitwise_xor(_by_op, _xz_comm))

        else:  # branch for Paulis
            # Commutate step is skipped.
            # Get the new xz by merging the recorded xz with the Pauli ops directly.
            new_xz.append(np.bitwise_xor(pauli_to_xz(op), xz[0]))
        # Assign the updated the xz to the x, z record
        for idx, wire in enumerate(wires):
            x_record[wire], z_record[wire] = new_xz[idx]

    return x_record, z_record


def _apply_measurement_correction_rule(x: np.uint8, z: np.uint8, ob: Operator):
    """Get the phase correction factor based on the recorded `x` an `z` of the target wire and the corresponding
    observable.

        Args:
            x (np.uint8): Recorded x at the target wire of ob.
            z (np.uint8): Recorded z at the target wire of ob.
            ob (Operator): Observable of the measurement.

        Return:
            Phase correction factor.
    """
    if isinstance(ob, Z):
        return -1 if x == 1 else 1

    if isinstance(ob, X):
        return -1 if z == 1 else 1

    if isinstance(ob, Y):
        return -1 if np.sum([x, z]) == 1 else 1

    if isinstance(ob, I):
        return 1

    raise NotImplementedError(f"{ob.name} is not supported.")


def get_measurements_corrections(tape: QuantumScript, x_record: np.array, z_record: np.array):
    """Get phase correction factor for all measurements in a tape. The phase correction factor
    is calculated based on the measurement observables with the corresponding recorded x an z.
        Args:
            tape (tape: qml.tape.QuantumScript): A quantum tape.
            x_record (np.array): The array of recorded x for each wire.
            z_record (np.array): The array of recorded z for each wire.
        Return:
            A list of phase correction factor for all measurements.
    """
    phase_cor = [1] * len(tape.measurements)
    for idx, measurement in enumerate(tape.measurements):
        obs = measurement.obs
        if type(obs) in _PAULIS:
            # branch for NamedObs
            phase_cor[idx] *= _apply_measurement_correction_rule(
                x_record[obs.wires[0]], z_record[obs.wires[0]], measurement.obs
            )
        elif isinstance(obs, Prod):
            # branch for TensorProd
            obs = measurement.obs.decomposition()
            for ob in obs:
                phase_cor[idx] *= _apply_measurement_correction_rule(
                    x_record[ob.wires[0]], z_record[ob.wires[0]], ob
                )
        else:
            raise NotImplementedError(f"{obs.name} is not supported.")
    return phase_cor


def get_byproduct_corrections(tape: QuantumScript, mid_meas: List):
    r"""Get measurement correction coefficients offline with a quantum script and mid-measurement results for each shot.
    The mid measurement results are first parsed with the quantum script to get the byproduct operations for each Clifford
    gates. Note that byproduct operations and ops are stored with list and used in a stack manner. The calculation iteratively
    pops out the first operation in the tape and applies commutate rules for the first byproduct ops in the byproduct stack and
    then the results are commutated to the byproduct of the current operations in the tape if it is a Clifford gate. The calculation
    starts from applying commutate rules for :class:`qml.I` gate or $encode\_xz(x,z)=(0,0)$ to the first gate in the tape. The
    measurement corrections are returned based on the observable operators and the xz recorded.

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

    x_record, z_record = get_xz_record(tape.num_wires, by_ops, ops)

    return get_measurements_corrections(tape, x_record, z_record)
