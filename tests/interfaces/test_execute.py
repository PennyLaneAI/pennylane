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
from contextlib import nullcontext

import pytest

import pennylane as qml
from pennylane.devices import DefaultQubit


def test_warning_if_not_device_batch_transform():
    """Test that a warning is raised if the users requests to not run device batch transform."""

    # pylint: disable=too-few-public-methods
    class CustomOp(qml.operation.Operator):
        """Dummy operator."""

        def decomposition(self):
            return [qml.PauliX(self.wires[0])]

    dev = DefaultQubit()

    qs = qml.tape.QuantumScript([CustomOp(0)], [qml.expval(qml.PauliZ(0))])

    with pytest.warns(UserWarning, match="Device batch transforms cannot be turned off"):
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

        with (
            pytest.warns(UserWarning, match="Device batch transforms cannot be turned off")
            if not device_batch_transform
            else nullcontext()
        ):
            with pytest.warns(
                qml.PennyLaneDeprecationWarning,
                match="The device_batch_transform argument is deprecated",
            ):
                qml.execute([qs], dev, device_batch_transform=device_batch_transform)
