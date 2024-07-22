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
"""Tests for exeuction with default qubit 2 independent of any interface."""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.devices import DefaultQubit
from pennylane.workflow.execution import _preprocess_expand_fn


class TestPreprocessExpandFn:
    """Tests the _preprocess_expand_fn helper function."""

    def test_provided_is_callable(self):
        """Test that if the expand_fn is not "device", it is simply returned."""

        dev = DefaultQubit()

        def f(tape):
            return tape

        out = _preprocess_expand_fn(f, dev, 10)
        assert out is f

    def test_new_device_blank_expand_fn(self):
        """Test that the expand_fn is blank if is new device."""

        dev = DefaultQubit()

        out = _preprocess_expand_fn("device", dev, 10)

        x = [1]
        assert out(x) is x


class TestBatchTransformHelper:
    """Unit tests for the _batch_transform helper function."""

    def test_warns_if_requested_off(self):
        """Test that a warning is raised if the the batch transform is requested to not be used."""

        # pylint: disable=too-few-public-methods
        class CustomOp(qml.operation.Operator):
            """Dummy operator."""

            def decomposition(self):
                return [qml.PauliX(self.wires[0])]

        dev = DefaultQubit()

        qs = qml.tape.QuantumScript([CustomOp(0)], [qml.expval(qml.PauliZ(0))])

        with pytest.warns(UserWarning, match="device batch transforms cannot be turned off"):
            program, _ = dev.preprocess()
            with pytest.warns(
                qml.PennyLaneDeprecationWarning,
                match="The device_batch_transform argument is deprecated",
            ):
                qml.execute(
                    (qs, qs), device=dev, device_batch_transform=False, transform_program=program
                )

    def test_split_and_expand_performed(self):
        """Test that preprocess returns the correct tapes when splitting and expanding
        is needed."""

        class NoMatOp(qml.operation.Operation):
            """Dummy operation for expanding circuit."""

            # pylint: disable=missing-function-docstring
            num_wires = 1

            # pylint: disable=arguments-renamed, invalid-overridden-method
            @property
            def has_matrix(self):
                return False

            def decomposition(self):
                return [qml.PauliX(self.wires), qml.PauliY(self.wires)]

        ops = [qml.Hadamard(0), NoMatOp(1), qml.RX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[0]]),
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        dev = DefaultQubit()
        config = qml.devices.ExecutionConfig(gradient_method="adjoint")

        program, new_config = dev.preprocess(config)
        res_tapes, batch_fn = program(tapes)
        expected_ops = [
            [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RX(np.pi, wires=1)],
            [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RX(np.pi / 2, wires=1)],
        ]

        assert len(res_tapes) == 4
        for i, t in enumerate(res_tapes):
            for op, expected_op in zip(t.operations, expected_ops[i % 2]):
                qml.assert_equal(op, expected_op)
            assert len(t.measurements) == 1
            if i < 2:
                qml.assert_equal(t.measurements[0], measurements[0])
            else:
                qml.assert_equal(t.measurements[0], measurements[1])

        input = ([[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]])
        assert np.array_equal(batch_fn(input), np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

        assert new_config.grad_on_execution
        assert new_config.use_device_gradient


def test_warning_if_not_device_batch_transform():
    """Test that a warning is raised if the users requests to not run device batch transform."""

    # pylint: disable=too-few-public-methods
    class CustomOp(qml.operation.Operator):
        """Dummy operator."""

        def decomposition(self):
            return [qml.PauliX(self.wires[0])]

    dev = DefaultQubit()

    qs = qml.tape.QuantumScript([CustomOp(0)], [qml.expval(qml.PauliZ(0))])

    with pytest.warns(UserWarning, match="device batch transforms cannot be turned off"):
        program, _ = dev.preprocess()
        with pytest.warns(
            qml.PennyLaneDeprecationWarning,
            match="The device_batch_transform argument is deprecated",
        ):
            results = qml.execute(
                [qs], dev, device_batch_transform=False, transform_program=program
            )

    assert len(results) == 1
    assert qml.math.allclose(results[0], -1)


@pytest.mark.parametrize("gradient_fn", (None, "backprop", qml.gradients.param_shift))
def test_caching(gradient_fn):
    """Test that cache execute returns the cached result if the same script is executed
    multiple times, both in multiple times in a batch and in separate batches."""
    dev = DefaultQubit()

    qs = qml.tape.QuantumScript([qml.PauliX(0)], [qml.expval(qml.PauliZ(0))])

    cache = {}

    with qml.Tracker(dev) as tracker:
        results = qml.execute([qs, qs], dev, cache=cache, gradient_fn=gradient_fn)
        results2 = qml.execute([qs, qs], dev, cache=cache, gradient_fn=gradient_fn)

    assert len(cache) == 1
    assert cache[qs.hash] == -1.0

    assert list(results) == [-1.0, -1.0]
    assert list(results2) == [-1.0, -1.0]

    assert tracker.totals["batches"] == 1
    assert tracker.totals["executions"] == 1
    assert cache[qs.hash] == -1.0


class TestExecuteDeprecations:
    """Class to test deprecation warnings in qml.execute. Warnings should be raised even if the default value is used."""

    @pytest.mark.parametrize("expand_fn", (None, lambda qs: qs, "device"))
    def test_expand_fn_is_deprecated(self, expand_fn):
        """Test that expand_fn argument of qml.execute is deprecated."""
        dev = DefaultQubit()
        qs = qml.tape.QuantumScript([qml.PauliX(0)], [qml.expval(qml.PauliZ(0))])

        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The expand_fn argument is deprecated"
        ):
            # None is a value used for expand_fn in the QNode
            qml.execute([qs], dev, expand_fn=expand_fn)

    def test_max_expansion_is_deprecated(self):
        """Test that max_expansion argument of qml.execute is deprecated."""
        dev = DefaultQubit()
        qs = qml.tape.QuantumScript([qml.PauliX(0)], [qml.expval(qml.PauliZ(0))])

        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The max_expansion argument is deprecated"
        ):
            qml.execute([qs], dev, max_expansion=10)

    @pytest.mark.parametrize("override_shots", (False, 10))
    def test_override_shots_is_deprecated(self, override_shots):
        """Test that override_shots argument of qml.execute is deprecated."""
        dev = DefaultQubit()
        qs = qml.tape.QuantumScript([qml.PauliX(0)], [qml.expval(qml.PauliZ(0))])

        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The override_shots argument is deprecated"
        ):
            qml.execute([qs], dev, override_shots=override_shots)

    @pytest.mark.parametrize("device_batch_transform", (False, True))
    def test_device_batch_transform_is_deprecated(self, device_batch_transform):
        """Test that device_batch_transform argument of qml.execute is deprecated."""
        # Need to use legacy device, otherwise another warning would be raised due to new Device interface
        dev = qml.device("default.qubit.legacy", wires=1)

        qs = qml.tape.QuantumScript([qml.PauliX(0)], [qml.expval(qml.PauliZ(0))])

        with pytest.warns(
            qml.PennyLaneDeprecationWarning,
            match="The device_batch_transform argument is deprecated",
        ):
            qml.execute([qs], dev, device_batch_transform=device_batch_transform)
