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
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import pennylane as qml
from pennylane import DeviceError
from pennylane.tape import QuantumTape
from pennylane.devices import Device, DefaultExecutionConfig, ExecutionConfig
from pennylane.transforms.core import TransformProgram, transform
from pennylane.devices.qubit.preprocess import validate_measurements, expand_fn


class DefaultSympy(Device):
    """TODO."""

    name: str = "default.sympy"

    def execute(
        self,
        circuits: Union[QuantumTape, Sequence[QuantumTape]],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        return 0.0 if isinstance(circuits, qml.tape.QuantumScript) else tuple(0.0 for c in circuits)

    def preprocess(
        self,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Tuple[TransformProgram, ExecutionConfig]:
        if execution_config.gradient_method == "adjoint":
            raise DeviceError("The adjoint gradient method is not supported on the default.sympy device")

        program = TransformProgram()

        program.add_transform(_validate_shots)
        program.add_transform(validate_measurements)

        # _expand_fn = partial(expand_fn, acceptance_function=_accepted_operator)
        # program.add_transform(_expand_fn)

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
    return True

@transform
def _validate_shots(tape: qml.tape.QuantumTape, execution_config: ExecutionConfig = DefaultExecutionConfig) -> (Sequence[qml.tape.QuantumTape], Callable):
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
