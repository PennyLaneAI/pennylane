# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
The conversion of a circuit to openqasm
"""
from collections.abc import Callable
from functools import singledispatch, wraps
from typing import Any, overload

from pennylane.devices.preprocess import decompose
from pennylane.measurements import MidMeasureMP
from pennylane.operation import Operator
from pennylane.ops import Conditional
from pennylane.tape import QuantumScript
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.wires import Wires, WiresLike
from pennylane.workflow import QNode, construct_tape

OPENQASM_GATES = {
    "CNOT": "cx",
    "CZ": "cz",
    "U3": "u3",
    "U2": "u2",
    "U1": "u1",
    "Identity": "id",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "Hadamard": "h",
    "S": "s",
    "Adjoint(S)": "sdg",
    "T": "t",
    "Adjoint(T)": "tdg",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
    "SWAP": "swap",
    "Toffoli": "ccx",
    "CSWAP": "cswap",
    "PhaseShift": "u1",
}
"""
dict[str, str]: Maps PennyLane gate names to equivalent QASM gate names.

Note that QASM has two native gates:

- ``U`` (equivalent to :class:`~.U3`)
- ``CX`` (equivalent to :class:`~.CNOT`)

All other gates are defined in the file stdgates.inc:
https://github.com/Qiskit/openqasm/blob/master/examples/stdgates.inc
"""


# pylint: disable=unused-argument
@singledispatch
def _obj_string(op: Operator, wires: Wires, bit_map: dict, precision: None | int) -> str:
    try:
        gate = OPENQASM_GATES[op.name]
    except KeyError as e:
        raise ValueError(f"Operation {op.name} not supported by the QASM serializer") from e

    wire_labels = ",".join([f"q[{wires.index(w)}]" for w in op.wires.tolist()])
    params = ""

    if op.num_params > 0:
        # If the operation takes parameters, construct a string
        # with parameter values.
        if precision is not None:
            params = "(" + ",".join([f"{p:.{precision}}" for p in op.parameters]) + ")"
        else:
            # use default precision
            params = "(" + ",".join([str(p) for p in op.parameters]) + ")"

    return f"{gate}{params} {wire_labels};"


@_obj_string.register
def _mid_measure_str(op: MidMeasureMP, wires: Wires, bit_map: dict, precision: None | int) -> str:
    if op.reset:
        raise NotImplementedError(f"Unable to translate mid circuit measurements with reset {op}.")
    if op.postselect:
        raise NotImplementedError(
            f"Unable to translate mid circuit measurement with postselection {op}"
        )
    wire = f"q[{wires.index(op.wires[0])}]"
    mcm_ind = len(bit_map)
    bit_map[op] = mcm_ind
    return f"measure {wire} -> mcms[{mcm_ind}];"


@_obj_string.register
def _conditional_str(op: Conditional, wires: Wires, bit_map: dict, precision: None | int) -> str:
    if op.meas_val.has_processing:
        raise NotImplementedError(
            "to_openqasm does not support translating Conditionals with measurement postprocessing."
        )
    mcm_name = f"mcms[{bit_map[op.meas_val.measurements[0]]}]"
    return f"if({mcm_name}==1) {_obj_string(op.base, wires, bit_map, precision)}"


def _tape_openqasm(
    tape: QuantumScript, wires: Wires, rotations: bool, measure_all: bool, precision: None | int
) -> str:
    """Helper function to serialize a tape as an OpenQASM 2.0 program."""
    wires = wires or tape.wires

    # add the QASM headers
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";']

    if tape.num_wires == 0:
        # empty circuit
        return "\n".join(lines) + "\n"

    # create the quantum and classical registers
    lines.append(f"qreg q[{len(wires)}];")
    lines.append(f"creg c[{len(wires)}];")

    num_mcms = sum(isinstance(o, MidMeasureMP) for o in tape.operations)
    if num_mcms:
        lines.append(f"creg mcms[{num_mcms}];")
    bit_map = {}

    # get the user applied circuit operations without interface information
    [transformed_tape], _ = convert_to_numpy_parameters(tape)
    operations = transformed_tape.operations

    if rotations:
        # if requested, append diagonalizing gates corresponding
        # to circuit observables
        operations += tape.diagonalizing_gates

    just_ops = QuantumScript(operations)

    def stopping_condition(op):
        return op.name in OPENQASM_GATES or isinstance(op, (MidMeasureMP, Conditional))

    [new_tape], _ = decompose(
        just_ops,
        stopping_condition=stopping_condition,
        skip_initial_state_prep=False,
        name="to_openqasm",
        error=ValueError,
    )

    # create the QASM code representing the operations
    for op in new_tape.operations:
        lines.append(_obj_string(op, wires, bit_map, precision=precision))

    # apply computational basis measurements to each quantum register
    # NOTE: This is not strictly necessary, we could inspect self.observables,
    # and then only measure wires which are requested by the user. However,
    # some devices which consume QASM require all registers to be measured, so
    # measure all wires by default to be safe.
    if measure_all:
        for wire in range(len(wires)):
            lines.append(f"measure q[{wire}] -> c[{wire}];")
    else:
        measured_wires = Wires.all_wires([m.wires for m in tape.measurements])

        for w in measured_wires:
            wire_indx = tape.wires.index(w)
            lines.append(f"measure q[{wire_indx}] -> c[{wire_indx}];")

    return "\n".join(lines) + "\n"


@overload
def to_openqasm(
    circuit: QuantumScript,
    wires: WiresLike | None = None,
    rotations: bool = True,
    measure_all: bool = True,
    precision: int | None = None,
) -> str: ...
@overload
def to_openqasm(
    circuit: QNode,
    wires: WiresLike | None = None,
    rotations: bool = True,
    measure_all: bool = True,
    precision: int | None = None,
) -> Callable[[Any], str]: ...
def to_openqasm(
    circuit,
    wires: Wires | None = None,
    rotations: bool = True,
    measure_all: bool = True,
    precision: None | int = None,
):
    """Convert a circuit to an OpenQASM 2.0 program.

    Terminal measurements are assumed to be performed on all qubits in the computational basis.
    An optional ``rotations`` argument can be provided so that the output of the OpenQASM circuit
    is diagonal in the eigenbasis of the quantum circuit's observables.
    The measurement outputs can be restricted to only those specified in the circuit by setting ``measure_all=False``.

    Args:
        circuit (QNode or QuantumScript): the quantum circuit to be serialized.
        wires (Wires or None): the wires to use when serializing the circuit.
            Default is ``None``, such that all the wires of the circuit are used for serialization.
        rotations (bool): if ``True``, add gates that rotate the quantum state into the eigenbasis
            of the circuit's observables. Default is ``True``.
        measure_all (bool): if ``True``, add a computational basis measurement on all the qubits.
            Default is ``True``.
        precision (int or None): number of decimal digits to display for the parameters.

    Returns:
        str: OpenQASM 2.0 program corresponding to the circuit.

    **Example**

    The following QNode can be serialized to an OpenQASM 2.0 program:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta, phi):
            qml.RX(theta, wires=0)
            qml.CNOT(wires=[0,1])
            qml.RZ(phi, wires=1)
            return qml.sample()

    >>> output = qml.to_openqasm(circuit)(1.2, 0.9)
    >>> print(output)
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    rx(1.2) q[0];
    cx q[0],q[1];
    rz(0.9) q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];

    Note that the terminal measurements will be re-imported as mid-circuit measurements
    when used with ``from_qasm`` or ``from_qasm3``.

    >>> print(qml.draw(qml.from_qasm(output))())
    0: ──RX(1.20)─╭●──┤↗├───────────┤
    1: ───────────╰X──RZ(0.90)──┤↗├─┤

    .. details::
        :title: Usage Details

        By default, the resulting OpenQASM code will have terminal measurements on all qubits,
        where all the measurements are performed in the computational basis.
        However, if terminal measurements in the circuit act only on a subset of the qubits
        and ``measure_all=False``, the OpenQASM code will include measurements on those
        specific qubits only.

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(0)
                qml.CNOT(wires=[0,1])
                return qml.sample(wires=1)

        >>> print(qml.to_openqasm(circuit, measure_all=False)())
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q[1] -> c[1];

        If the circuit returns an expectation value of a given observable and ``rotations=True``,
        the OpenQASM 2.0 program will also include the gates that rotate the quantum state into
        the eigenbasis of the measured observable.

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(0)
                qml.CNOT(wires=[0,1])
                return qml.expval(qml.PauliX(0) @ qml.PauliY(1))

        >>> print(qml.to_openqasm(circuit, rotations=True)())
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        h q[0];
        z q[1];
        s q[1];
        h q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    """
    if isinstance(circuit, QuantumScript):
        return _tape_openqasm(
            circuit,
            wires=wires,
            rotations=rotations,
            measure_all=measure_all,
            precision=precision,
        )

    @wraps(circuit)
    def wrapper(*args, **kwargs) -> str:
        tape = construct_tape(circuit)(*args, **kwargs)
        return _tape_openqasm(
            tape,
            wires=wires,
            rotations=rotations,
            measure_all=measure_all,
            precision=precision,
        )

    return wrapper
