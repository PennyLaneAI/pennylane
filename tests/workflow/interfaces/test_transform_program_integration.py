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
"""Integration tests for the transform program and the execution pipeline.

Differentiability tests are still in the ml-framework specific files.
"""
import copy
from functools import partial

import numpy as np
import pytest

import pennylane as qml
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn

device_suite = (
    qml.device("default.qubit"),
    qml.device("lightning.qubit", wires=5),
)


@pytest.mark.all_interfaces
class TestTransformProgram:
    """Non differentiability tests for the transform program keyword argument."""

    @pytest.mark.parametrize("interface", (None, "autograd", "jax", "torch"))
    def test_transform_program_none(self, interface):
        """Test that if no transform program is provided, null default behavior is used."""

        dev = qml.devices.DefaultQubit()

        tape0 = qml.tape.QuantumScript([qml.RX(1.0, 0)], [qml.expval(qml.PauliZ(0))])
        tape1 = qml.tape.QuantumScript([qml.RY(2.0, 0)], [qml.state()])

        with dev.tracker as tracker:
            results = qml.execute((tape0, tape1), dev, transform_program=None, interface=interface)

        assert qml.math.allclose(results[0], np.cos(1.0))
        assert qml.math.allclose(results[1], np.array([np.cos(1.0), np.sin(1.0)]))

        # checks on what is passed to the device. Should be exactly what we put in.
        assert tracker.totals["executions"] == 2
        assert tracker.history["resources"][0].gate_types["RX"] == 1
        assert tracker.history["resources"][1].gate_types["RY"] == 1

    @pytest.mark.parametrize("interface", (None, "autograd", "jax", "torch"))
    def test_transform_program_modifies_circuit(self, interface):
        """Integration tests for a transform program that modifies the input tapes."""

        dev = qml.devices.DefaultQubit()

        def null_postprocessing(results):
            return results[0]

        @qml.transform
        def just_pauli_x_out(
            tape: QuantumScript,
        ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            return (
                qml.tape.QuantumScript([qml.PauliX(0)], tape.measurements),
            ), null_postprocessing

        pauli_x_out_container = qml.transforms.core.TransformContainer(just_pauli_x_out)

        transform_program = qml.transforms.core.TransformProgram([pauli_x_out_container])

        tape0 = qml.tape.QuantumScript(
            [qml.Rot(1.2, 2.3, 3.4, wires=0)], [qml.expval(qml.PauliZ(0))]
        )
        tape1 = qml.tape.QuantumScript(
            [qml.Hadamard(0), qml.IsingXX(1.2, wires=(0, 1))], [qml.expval(qml.PauliX(0))]
        )

        with dev.tracker as tracker:
            results = qml.execute(
                (tape0, tape1), dev, transform_program=transform_program, interface=interface
            )

        assert qml.math.allclose(results[0], -1.0)
        assert qml.math.allclose(results[1], 0.0)

        assert tracker.totals["executions"] == 2
        assert tracker.history["resources"][0].gate_types["PauliX"] == 1
        assert tracker.history["resources"][0].num_gates == 1
        assert tracker.history["resources"][1].gate_types["PauliX"] == 1
        assert tracker.history["resources"][1].num_gates == 1

    @pytest.mark.parametrize("interface", (None, "autograd", "jax", "torch"))
    def test_shot_distributing_transform(self, interface):
        """Test a transform that creates a batch of tapes with different shots.

        Note that this only works with the new device interface.
        """
        dev = qml.devices.DefaultQubit()

        def null_postprocessing(results):
            return results

        @qml.transform
        def split_shots(tape):
            tape1 = qml.tape.QuantumScript(
                tape.operations, tape.measurements, shots=tape.shots.total_shots // 2
            )
            tape2 = qml.tape.QuantumScript(
                tape.operations, tape.measurements, shots=tape.shots.total_shots * 2
            )
            return (tape1, tape2), null_postprocessing

        scale_shots = qml.transforms.core.TransformContainer(split_shots)
        program = qml.transforms.core.TransformProgram([scale_shots])

        tape = qml.tape.QuantumScript([], [qml.counts(wires=0)], shots=100)
        results = qml.execute((tape,), dev, interface=interface, transform_program=program)[0]

        assert results[0] == {"0": 50}
        assert results[1] == {"0": 200}

    @pytest.mark.parametrize("interface", (None, "autograd", "jax", "torch"))
    @pytest.mark.parametrize("dev", device_suite)
    def test_ragged_batch_sizes(self, dev, interface):
        """Test a transform that splits input tapes up into different sizes."""

        # note this does not work for partitioned shots
        def sum_measurements(results):
            return sum(results)

        @qml.transform
        def split_sum_terms(tape):
            sum_obj = tape.measurements[0].obs
            new_tapes = tuple(
                qml.tape.QuantumScript(tape.operations, [qml.expval(o)], shots=tape.shots)
                for o in sum_obj
            )

            return new_tapes, sum_measurements

        container = qml.transforms.core.TransformContainer(split_sum_terms)
        prog = qml.transforms.core.TransformProgram((container,))

        op = qml.RX(1.2, 0)
        tape1 = qml.tape.QuantumScript([op], [qml.expval(qml.sum(qml.PauliX(0), qml.PauliZ(0)))])
        tape2 = qml.tape.QuantumScript(
            [op], [qml.expval(qml.sum(qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)))]
        )
        tape3 = qml.tape.QuantumScript(
            [op], [qml.expval(qml.sum(*(qml.PauliZ(i) for i in range(5))))]
        )
        with dev.tracker:
            results = qml.execute(
                (tape1, tape2, tape3), dev, interface=interface, transform_program=prog, cache=True
            )

        assert qml.math.allclose(results[0], np.cos(1.2))
        assert qml.math.allclose(results[1], -np.sin(1.2) + np.cos(1.2))
        assert qml.math.allclose(results[2], 4 + np.cos(1.2))

        assert dev.tracker.totals["executions"] == 7

    def test_chained_preprocessing(self):
        """Test a transform program with two transforms where their order affects the output."""

        dev = qml.device("default.qubit", wires=2)

        def null_postprocessing(results):
            return results[0]

        @qml.transform
        def just_pauli_x_out(
            tape: QuantumScript,
        ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            return (
                qml.tape.QuantumScript([qml.PauliX(0)], tape.measurements),
            ), null_postprocessing

        @qml.transform
        def repeat_operations(
            tape: QuantumScript,
        ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            new_tape = qml.tape.QuantumScript(
                tape.operations + copy.deepcopy(tape.operations), tape.measurements
            )
            return (new_tape,), null_postprocessing

        just_pauli_x_container = qml.transforms.core.TransformContainer(just_pauli_x_out)
        repeat_operations_container = qml.transforms.core.TransformContainer(repeat_operations)

        prog = qml.transforms.core.TransformProgram(
            (just_pauli_x_container, repeat_operations_container)
        )

        tape1 = qml.tape.QuantumScript([qml.RX(1.2, 0)], [qml.expval(qml.PauliZ(0))])

        with dev.tracker:
            results = qml.execute((tape1,), dev, transform_program=prog)

        assert dev.tracker.history["resources"][0].gate_types["PauliX"] == 2
        assert qml.math.allclose(results, 1.0)

        prog_reverse = qml.transforms.core.TransformProgram(
            (repeat_operations_container, just_pauli_x_container)
        )

        with dev.tracker:
            results = qml.execute((tape1,), dev, transform_program=prog_reverse)

        assert dev.tracker.history["resources"][0].gate_types["PauliX"] == 1
        assert qml.math.allclose(results, -1.0)

    @pytest.mark.parametrize("interface", (None, "autograd", "jax", "torch"))
    @pytest.mark.parametrize("dev", device_suite)
    def test_chained_postprocessing(self, dev, interface):
        def add_one(results):
            return results[0] + 1.0

        def scale_two(results):
            return results[0] * 2.0

        @qml.transform
        def transform_add(tape: QuantumScript):
            """A valid transform."""
            return (tape,), add_one

        @qml.transform
        def transform_mul(tape: QuantumScript):
            return (tape,), scale_two

        add_container = qml.transforms.core.TransformContainer(transform_add)
        mul_container = qml.transforms.core.TransformContainer(transform_mul)
        prog = qml.transforms.core.TransformProgram((add_container, mul_container))
        prog_reverse = qml.transforms.core.TransformProgram((mul_container, add_container))

        tape0 = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))])
        tape1 = qml.tape.QuantumScript([qml.PauliX(0)], [qml.expval(qml.PauliZ(0))])

        results = qml.execute((tape0, tape1), dev, interface=interface, transform_program=prog)

        # 1.0 * 2.0 + 1.0
        assert qml.math.allclose(results[0], 3.0)
        # -1.0 * 2.0 + 1.0 = -1.0
        assert qml.math.allclose(results[1], -1.0)

        results_reverse = qml.execute(
            (tape0, tape1), dev, interface=interface, transform_program=prog_reverse
        )

        # (1.0 + 1.0) * 2.0 = 4.0
        assert qml.math.allclose(results_reverse[0], 4.0)
        # (-1.0 + 1.0) * 2.0 = 0.0
        assert qml.math.allclose(results_reverse[1], 0.0)

    def test_composable_transform(self):
        """Test the composition of a gradient transform with another transform."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @partial(qml.gradients.param_shift, argnums=[0, 1])
        @qml.transforms.split_non_commuting
        @qml.qnode(device=dev, interface="jax")
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RZ(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliY(wires=0))

        x = jax.numpy.array(0.1)
        y = jax.numpy.array(0.2)

        res = circuit(x, y)

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 2

        assert isinstance(res[1], tuple)
        assert len(res[1]) == 2
