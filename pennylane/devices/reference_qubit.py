# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Contains the ReferenceQubit device, a minimal device that can be used for testing
and plugin development purposes.
"""


import numpy as np

import pennylane as qml

from .device_api import Device
from .execution_config import ExecutionConfig
from .modifiers import simulator_tracking, single_tape_support
from .preprocess import decompose, validate_device_wires, validate_measurements


def sample_state(state: np.ndarray, shots: int, seed=None):
    """Generate samples from the provided state and number of shots."""

    probs = np.imag(state) ** 2 + np.real(state) ** 2
    basis_states = np.arange(len(probs))

    num_wires = int(np.log2(len(probs)))

    rng = np.random.default_rng(seed)
    probs /= np.sum(probs)  # Fix: Normalize to prevent sum ≠ 1 errors in NumPy 2.0+
    basis_samples = rng.choice(basis_states, shots, p=probs)

    # convert basis state integers to array of booleans
    bin_strings = (format(s, f"0{num_wires}b") for s in basis_samples)
    return np.array([[int(val) for val in s] for s in bin_strings])


def simulate(tape: qml.tape.QuantumTape, seed=None) -> qml.typing.Result:
    """Simulate a tape and turn it into results.

    Args:
        tape (.QuantumTape): a representation of a circuit
        seed (Any): A seed to use to control the generation of samples.

    """
    # 1) create the initial state
    state = np.zeros(2 ** len(tape.wires))
    state[0] = 1.0

    # 2) apply all the operations
    for op in tape.operations:
        op_mat = op.matrix(wire_order=tape.wires)
        if qml.math.get_interface(op_mat) != "numpy":
            raise ValueError("Reference qubit can only work with numpy data.")
        state = qml.math.matmul(op_mat, state)

    # 3) perform measurements
    # note that shots are pulled from the tape, not from the device
    if tape.shots:
        samples = sample_state(state, shots=tape.shots.total_shots, seed=seed)
        # Shot vector support
        results = []
        for lower, upper in tape.shots.bins():
            sub_samples = samples[lower:upper]
            results.append(
                tuple(mp.process_samples(sub_samples, tape.wires) for mp in tape.measurements)
            )
        if len(tape.measurements) == 1:
            results = [res[0] for res in results]
        if not tape.shots.has_partitioned_shots:
            results = results[0]
        else:
            results = tuple(results)
    else:
        results = tuple(mp.process_state(state, tape.wires) for mp in tape.measurements)
        if len(tape.measurements) == 1:
            results = results[0]

    return results


operations = frozenset(
    {"PauliX", "PauliY", "PauliZ", "Hadamard", "CNOT", "CZ", "RX", "RY", "RZ", "GlobalPhase"}
)


def supports_operation(op: qml.operation.Operator) -> bool:
    """This function used by preprocessing determines what operations
    are natively supported by the device.

    While in theory ``simulate`` can support any operation with a matrix, we limit the target
    gate set for improved testing and reference purposes.

    """
    return getattr(op, "name", None) in operations


@simulator_tracking  # update device.tracker with some relevant information
@single_tape_support  # add support for device.execute(tape) in addition to device.execute((tape,))
class ReferenceQubit(Device):
    """A slimmed down numpy-based simulator for reference and testing purposes.

    Args:
        wires (int, Iterable[Number, str]): Number of wires present on the device, or iterable that
            contains unique labels for the wires as numbers (i.e., ``[-1, 0, 2]``) or strings
            (``['aux', 'q1', 'q2']``). Default ``None`` if not specified. While this device allows
            for ``wires`` to be unspecified at construction time, other devices may make this argument
            mandatory. Devices can also implement additional restrictions on the possible wires.
        shots (int, Sequence[int], Sequence[Union[int, Sequence[int]]]): The default number of shots
            to use in executions involving this device. Note that during execution, shots
            are pulled from the circuit, not from the device.
        seed (Union[str, None, int, array_like[int], SeedSequence, BitGenerator, Generator, jax.random.PRNGKey]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``. This is an optional
            keyword argument added to follow recommend NumPy best practices. Other devices do not need
            this parameter if it does not make sense for them.

    """

    name = "reference.qubit"

    def __init__(self, wires=None, shots=None, seed=None):
        super().__init__(wires=wires, shots=shots)

        # seed and rng not necessary for a device, but part of recommended
        # numpy practices to use a local random number generator
        self._rng = np.random.default_rng(seed)

    def preprocess(self, execution_config: ExecutionConfig | None = None):
        if execution_config is None:
            execution_config = ExecutionConfig()

        # Here we convert an arbitrary tape into one natively supported by the device
        program = qml.transforms.core.TransformProgram()
        program.add_transform(validate_device_wires, wires=self.wires, name="reference.qubit")
        program.add_transform(qml.defer_measurements, allow_postselect=False)
        program.add_transform(qml.transforms.split_non_commuting)
        program.add_transform(qml.transforms.diagonalize_measurements)
        program.add_transform(qml.devices.preprocess.measurements_from_samples)
        program.add_transform(
            decompose,
            stopping_condition=supports_operation,
            skip_initial_state_prep=False,
            name="reference.qubit",
        )
        program.add_transform(validate_measurements, name="reference.qubit")
        program.add_transform(qml.transforms.broadcast_expand)

        # no need to preprocess the config as the device does not support derivatives
        return program, execution_config

    def execute(self, circuits, execution_config: ExecutionConfig | None = None):
        for tape in circuits:
            assert all(supports_operation(op) for op in tape.operations)
        return tuple(simulate(tape, seed=self._rng) for tape in circuits)
