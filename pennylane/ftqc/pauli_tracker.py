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
import itertools

import numpy as np

from pennylane import math
from pennylane.operation import Operator
from pennylane.ops import CNOT, RZ, H, I, S, X, Y, Z
from pennylane.tape import QuantumScript

from .decomposition import _cnot_xz_corrections, _single_xz_corrections
from .operations import RotXZX

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

_CLIFFORD_GATES_SUPPORTED = (H, S, CNOT)

_NON_CLIFFORD_GATES_SUPPORTED = (RZ, RotXZX)


def pauli_to_xz(op: Operator) -> tuple[int, int]:
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

        >>> from pennylane.ftqc.pauli_tracker import pauli_to_xz
        >>> from pennylane import I
        >>> pauli_to_xz(I(0))
        (0, 0)

        A xz tuple representation is returned for a given Pauli operator.
    """

    if isinstance(op, _PAULIS):
        return _OPS_TO_XZ[type(op)]

    if op in _PAULIS:
        return _OPS_TO_XZ[op]

    raise NotImplementedError(f"{type(op)} gate does not support xz encoding.")


def xz_to_pauli(x: int, z: int) -> Operator:
    """
    Convert x, z to a Pauli operator class.

    Args:
        x (int) : Exponent of :class:`~pennylane.X` in the Pauli record.
        z (int) : Exponent of :class:`~pennylane.Z` in the Pauli record.

    Return:
        A Pauli operator class.

    **Example:**
        The following example shows how the XZ to Pauli works.

        >>> from pennylane.ftqc.pauli_tracker import xz_to_pauli
        >>> xz_to_pauli(0, 0)(wires=0)
        I(0)

        A Pauli operator class is returned for a given xz tuple.
    """
    if x in [0, 1] and z in [0, 1]:
        return _XZ_TO_OPS[(x, z)]
    raise ValueError("x and z should either 0 or 1.")


def pauli_prod(ops: list[Operator]) -> tuple[int, int]:
    r"""
    Get the result of a product of a list of Pauli operators. The result is a new Pauli operator up to a global phase.
    Mathematically, this function returns :math:`\prod_{i=0}^{n} ops[i]`.

    Args:
        ops (List[qml.operation.Operator]): A list of Pauli operators with the same target wire.

    Return:
        A xz tuple representing a new Pauli operator.

    **Example:**
        The following example shows how the `pauli_prod` works.

        >>> from pennylane.ftqc.pauli_tracker import pauli_prod
        >>> from pennylane import I, X, Y, Z
        >>> pauli_prod([I(0),X(0),Y(0),Z(0)])
        (0, 0)

        The result is a new Pauli operator in the xz-encoding representation.
    """
    if len(ops) == 0:
        raise ValueError("Please ensure that a valid list of operators are passed to the method.")
    res_x, res_z = pauli_to_xz(ops[0])

    for i in range(1, len(ops)):
        x, z = pauli_to_xz(ops[i])
        res_x ^= x
        res_z ^= z

    return (res_x, res_z)


def _commute_h(x: int, z: int):
    r"""
    Commute/move a Pauli represented by xz through :class:`~pennylane.H`.

    Args:
        x(int): Exponent of PauliX in the xz representation of a Pauli.
        z(int): Exponent of PauliZ in the xz representation of a Pauli.

    Return:
        A list of a tuple of xz representing a new Pauli operation that the :class:`~pennylane.H` commutes to.
    """
    return [(z, x)]


def _commute_s(x: int, z: int):
    r"""
    Commute/move a Pauli represented by xz through :class:`~pennylane.S`.

    Args:
        x(int): Exponent of PauliX in the xz representation of a Pauli.
        z(int): Exponent of PauliZ in the xz representation of a Pauli.

    Return:
        A list of a tuple of xz representing a new Pauli operation that the :class:`~pennylane.S` commutes to.
    """
    return [(x, x ^ z)]


def _commute_cnot(xc: int, zc: int, xt: int, zt: int):
    r"""
    Commute/move a Pauli represented by xz through :class:`~pennylane.CNOT`.

    Args:
        xc(int): Exponent of PauliX in the xz representation of a Pauli at the control wire.
        zc(int): Exponent of PauliZ in the xz representation of a Pauli at the control wire.
        xt(int): Exponent of PauliX in the xz representation of a Pauli at the target wire.
        zt(int): Exponent of PauliZ in the xz representation of a Pauli at the target wire.

    Return:
        A list of xz tuples representing new Paulis operation that the :class:`~pennylane.cnot` commutes to.
    """
    return [(xc, zc ^ zt), (xc ^ xt, zt)]


def commute_clifford_op(clifford_op: Operator, xz: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r"""Gets the list of xz-encoded bits representing the list of input Pauli ops after being commuted through the given Clifford op.
    Mathematically, this function applies the following equation: :math:`new\_xz \cdot clifford\_op = clifford\_op \cdot xz`
    up to a global phase to move the :math:`xz` through the :math:`clifford\_op` and returns the :math:`new\_xz`. Note that :math:`xz` and
    :math:`new\_xz` represent a list of Pauli operations.

    Args:
        clifford_op (Operator): A Clifford operator class. Supported operators are: :class:`qml.S`, :class:`qml.H`, :class:`qml.CNOT`.
        xz (list(tuple)): A list of xz tuples which map to Pauli operators

    Return:
        A list of new xz tuples that the clifford_op commute the xz to.

    **Example:**
        The following example shows how the `commute_clifford_op` works.

        >>> from pennylane.ftqc.pauli_tracker import commute_clifford_op
        >>> from pennylane import I, CNOT
        >>> commute_clifford_op(CNOT(wires=[0,1]), [(1, 1), (1, 0)])
        [(1, 1), (0, 0)]

        A list of Pauli operators in the xz representation is returned.
    """
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

    if isinstance(clifford_op, S):
        _x, _z = xz[0]
        return _commute_s(_x, _z)

    if isinstance(clifford_op, H):
        _x, _z = xz[0]
        return _commute_h(_x, _z)

    if isinstance(clifford_op, CNOT):
        _xc, _zc = xz[0]
        _xt, _zt = xz[1]
        return _commute_cnot(_xc, _zc, _xt, _zt)

    raise NotImplementedError("Only qml.H, qml.S and qml.CNOT are supported.")


def _parse_mid_measurements(tape: QuantumScript, mid_meas: list):
    r"""Parse a series of mid-measurement results of a quantum tape where
    all gates are from the set {X, Y, Z, I, H, S, CNOT, RZ, RotXZX}, and
    where only the first gate on each wire is permitted to be a non-Clifford
    gates (RZ, RotXZX). Both non-Clifford and Clifford gates mentioned above
    are measured in the way defined in Raussendorf et al. <https://arxiv.org/abs/quant-ph/0301052>`__.

    Args:
        tape (QuantumScript): The quantum tape in the standard circuit mode (Gates are not transformed into the MBQC formalism).
        mid_meas (list): Mid-measurements results.

    Returns:
        A list of `byproduct` ops in a reversed manner.
    """
    ops = tape.operations

    by_ops = []

    mid_meas_offset = 0
    _wires_used = [0] * (max(tape.wires) + 1)
    for op in ops:
        for wire in op.wires:
            _wires_used[wire] += 1

        gate_offset = 4 if op.num_wires == 1 else 13
        ms = mid_meas[mid_meas_offset : mid_meas_offset + gate_offset]
        by_op = []
        if isinstance(op, (S, H)):
            by_op = [_single_xz_corrections(op, *ms)]
        elif isinstance(op, CNOT):
            by_op = _cnot_xz_corrections(ms)
        elif isinstance(op, (RZ, RotXZX)):
            if _wires_used[op.wires[0]] > 1:
                raise NotImplementedError(
                    "The current implementation requires that only a single non-Clifford gate comes before Clifford and Pauli gates for each wire."
                )
            by_op = [_single_xz_corrections(op, *ms)]
        elif isinstance(op, _PAULIS):
            continue
        else:
            raise NotImplementedError(f"{op.name} is not supported.")

        by_ops.append(by_op)

        mid_meas_offset += gate_offset

    # To use list in a stack manner
    by_ops.reverse()
    return by_ops


def _get_xz_record(tape: QuantumScript, by_ops: list[tuple[int, int]]):
    """Commutate/merge the Pauli/byproduct ops of a Clifford circuit.

    Args:
        tape (QuantumScript): A quantum tape.
        by_ops (list): List of byproduct operators for Clifford gates

    Return:
        The final recorded x and z for each wire.
    """

    num_wires = max(tape.wires) + 1

    x_record = math.zeros(num_wires, dtype=np.uint8)
    z_record = math.zeros(num_wires, dtype=np.uint8)

    for op in tape.operations:
        wires = list(op.wires)

        # Get the recorded xz
        xz = [(x_record[wire], z_record[wire]) for wire in wires]

        # Updated xz
        new_xz = []
        if isinstance(op, _NON_CLIFFORD_GATES_SUPPORTED):
            # Branch for non-Clifford gates
            new_xz.append(by_ops.pop()[0])
        elif isinstance(op, _CLIFFORD_GATES_SUPPORTED):
            # Branch for Clifford gates
            # Step 1: Commutate the recorded xz with the Clifford gate to a new xz.
            xz_commutated = commute_clifford_op(op, xz)

            # Step 2: Merge the new xz with the byproduct by_op
            by_op = by_ops.pop()
            for _by_op, _xz_comm in zip(by_op, xz_commutated):
                new_xz.append(math.bitwise_xor(_by_op, _xz_comm))
        else:  # branch for Paulis
            # Commutate step is skipped.
            # Get the new xz by merging the recorded xz with the Pauli ops directly.
            new_xz.append(math.bitwise_xor(pauli_to_xz(op), xz[0]))
        # Assign the updated the xz to the x, z record
        for idx, wire in enumerate(wires):
            x_record[wire], z_record[wire] = new_xz[idx]

    return x_record, z_record


def _correct_samples(tape: QuantumScript, x_record: math.array, measurement_vals: list):
    """Correct sample measurements in a tape. The samples are corrected based on the `samples`
    at `wires` with the corresponding recorded x.

        Args:
            tape (tape: qml.tape.QuantumScript): A quantum tape.
            measurement_vals (List) : A list of measurement values.
            x_record (math.array): The array of recorded x for each wire.
            measurement_vals (List): Measurement values.

        Return:
            A list of corrected measurement values.
    """
    measured_wires = tape.measurements[0].wires
    x_at_target = np.array([x_record[w] for w in measured_wires])
    correct_meas = np.bitwise_xor(np.array(measurement_vals), x_at_target)

    return correct_meas


def get_byproduct_corrections(tape: QuantumScript, mid_meas: list, measurement_vals: list):
    r"""Correct sample results offline based on the executed quantum script and the mid-circuit measurement results for each shot.
    The mid measurement results are first parsed with the quantum script to get the byproduct operations for each Clifford
    and non-Clifford gates. Note that byproduct operations are stored as a list and accessed in a stack manner. The calculation iteratively
    pops out the first operation in the tape and applies commutation rules for the first byproduct ops in the byproduct stack and
    then the results are commutated to the byproduct of the current operations in the tape if it is a Clifford gate. The calculation
    starts from applying commutate rules for :class:`qml.I` gate or :math:`encode\_xz(x,z)=(0,0)` to the first gate in the tape. The
    measurement corrections are returned based on the observable operators and the xz recorded.

    Args:
        tape (tape: qml.tape.QuantumScript): A Clifford quantum tape with :class:`~pennylane.X`, :class:`~pennylane.Y`, :class:`~pennylane.Z`,
            :class:`~pennylane.I`, :class:`~pennylane.H`, :class:`~pennylane.S`, :class:`~pennylane.CNOT` and non-Clifford gates (:class:`~pennylane.RZ`
            and :class:`~pennylane.ftqc.RotXZX`) at the beginning of circuit in the standard circuit formalism. Note that one non-Clifford gate per wire
            at most is supported.
        mid_meas (list): MidMeasurement results per shot.
        measurement_vals (list): Raw measurement results.

    Return:
        A list of corrected measurement results.


    **Note**
    This work is to be integrated into the MBQC transform pipeline.

    **Example:**

        .. code-block:: python

            from pennylane.ftqc import diagonalize_mcms, generate_lattice, measure_x, measure_y
            from pennylane.ftqc import GraphStatePrep

            from pennylane.ftqc.pauli_tracker import get_byproduct_corrections

            def generate_random_state(n):
                seed_value = 42  # You can use any integer as the seed
                np.random.seed(seed_value)
                input_state = np.random.random(2**n) + 1j * np.random.random(2**n)
                return input_state / np.linalg.norm(input_state)


            def generate_rot_gate_graph():
                lattice = generate_lattice([4], "chain")
                return lattice.graph


            num_shots = 1000
            dev = qml.device("lightning.qubit")

            @qml.set_shots(num_shots)
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
                    qml.sample(wires=[4]),
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

            init_state = generate_random_state(1)
            res = circ(init_state)

            ops = [qml.H(wires=[0]), qml.H(wires=[0]), qml.H(wires=[0])]
            measurements = [qml.sample(qml.Z(0))]

            meas_res = res[0]
            mid_meas_res = res[1:]
            corrected_meas_res = []

            script = qml.tape.QuantumScript(ops, measurements, shots=num_shots)

            for i in range(num_shots):
                mid_meas = [row[i] for row in mid_meas_res]
                corrected_meas_res.extend(get_byproduct_corrections(script, mid_meas, [meas_res[i]]))

            res_corrected = 1 - 2*np.sum(corrected_meas_res) / num_shots

            dev_ref = qml.device("default.qubit")

            @diagonalize_mcms
            @qml.qnode(dev)
            def circ_ref(start_state):
                qml.StatePrep(start_state, wires=[0])
                qml.H(0)
                qml.H(0)
                qml.H(0)
                return qml.expval(qml.Z(0))

            np.allclose(res_corrected, circ_ref(init_state))

    """
    by_ops = _parse_mid_measurements(tape, mid_meas)

    x_record, _ = _get_xz_record(tape, by_ops)

    return _correct_samples(tape, x_record, measurement_vals)
