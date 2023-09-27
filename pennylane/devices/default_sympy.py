# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the SymPy device"""
from typing import Callable, Optional, Sequence, Tuple, Union, Iterable

import pennylane as qml
from pennylane import DeviceError
from pennylane.tape import QuantumTape
from pennylane.devices import Device, DefaultExecutionConfig, ExecutionConfig
from pennylane.transforms.core import TransformProgram, transform
from pennylane.devices.qubit.preprocess import validate_measurements, expand_fn
from pennylane.operation import StatePrep, Operation
from pennylane.measurements import MeasurementProcess


class DefaultSympy(Device):
    """TODO."""

    name: str = "default.sympy"

    def __init__(self, *args, expand_observables: bool = True, **kwargs):
        self.expand_observables = expand_observables
        super().__init__(*args, **kwargs)

    def execute(
        self,
        circuits: Union[QuantumTape, Sequence[QuantumTape]],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        return _simulate(circuits) if isinstance(circuits, qml.tape.QuantumScript) else tuple(_simulate(c) for c in circuits)

    def preprocess(
        self,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Tuple[TransformProgram, ExecutionConfig]:
        if execution_config.gradient_method == "adjoint":
            raise DeviceError("The adjoint gradient method is not supported on the default.sympy device")

        program = TransformProgram()

        program.add_transform(_validate_shots)
        program.add_transform(validate_measurements)

        program.add_transform(expand_fn, acceptance_function=_accepted_operator)

        return program, execution_config

    def supports_derivatives(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        if execution_config.gradient_method == "backprop" and execution_config.interface == "sympy":
            return True


def _accepted_operator(op: qml.operation.Operator) -> bool:
    """Indicates whether an operation is supported on default.sympy"""
    # if isinstance(op, StatePrep) and not isinstance(op, qml.BasisState):
    #     return False
    return True


def _simulate(circuit: QuantumTape):
    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]

    state = _create_initial_state(circuit.wires, prep)
    state = _get_evolved_state(circuit.operations[bool(prep):], state, circuit.wires)

    return _measure_state(state, circuit.measurements, circuit.wires)


def _measure_state(state, measurements: Sequence[MeasurementProcess], wires: qml.wires.Wires):
    return __measure_state(state, measurements[0], wires) if len(measurements) == 1 else tuple(__measure_state(state, m, wires) for m in measurements)


from pennylane.measurements import StateMP, DensityMatrixMP, ExpectationMP
def __measure_state(state, measurement: MeasurementProcess, wires: qml.wires.Wires):
    if isinstance(measurement, DensityMatrixMP):
        raise NotImplementedError
    if isinstance(measurement, StateMP):
        return state
    if isinstance(measurement, ExpectationMP):
        if

    return 0


def _get_evolved_state(ops: Sequence[Operation], state, all_wires: qml.wires.Wires):
    from sympy.physics.quantum.gate import X, Y, Z, CNOT, H, S, SWAP, T, IdentityGate, UGate
    from sympy import Matrix

    _directly_supported_gates = {
        qml.PauliX: X,
        qml.PauliY: Y,
        qml.PauliZ: Z,
        qml.CNOT: CNOT,
        qml.Hadamard: H,
        qml.S: S,
        qml.SWAP: SWAP,
        qml.T: T,
        qml.Identity: IdentityGate,
    }

    sympy_ops = None

    for op in ops:
        wires = all_wires.indices(op.wires)
        if (t := type(op)) in _directly_supported_gates:
            sympy_op = _directly_supported_gates[t](*wires)
        else:
            sympy_mat = Matrix(qml.matrix(op))
            sympy_op = UGate(wires, sympy_mat)
        sympy_ops = _apply_op_to_ops(sympy_op, sympy_ops)

    return sympy_ops * state


def _apply_op_to_ops(op, ops):
    return op if ops is None else op * ops


def _create_initial_state(
    wires: qml.wires.Wires,
    prep_operation: qml.operation.StatePrepBase = None,
):
    from sympy.physics.quantum.qubit import Qubit, matrix_to_qubit
    if not prep_operation:
        return Qubit("0" * len(wires))

    if isinstance(prep_operation, qml.BasisState):
        prep_vals = prep_operation.parameters[0]
        return Qubit("".join(map(str, prep_vals)))

    return matrix_to_qubit(qml.math.expand_dims(prep_operation.state_vector(), 0).toarray())


@transform
def _validate_shots(tape: qml.tape.QuantumTape, _: ExecutionConfig = DefaultExecutionConfig) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Validates that no shots are present in the input tape because this is not supported on the
    device."""
    if tape.shots:
        raise DeviceError("The default.sympy device does not support finite-shot execution")
    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [tape], null_postprocessing
