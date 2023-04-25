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

import pytest

import pennylane as qml

from pennylane.devices.experimental import DefaultQubit2


class NewDeviceUnitTests:
    """Localized tests for specific warnings, errors, and edge behaviour."""

    def test_warning_if_not_device_batch_transform(self):
        """Test that a warning is raised if the users requests to not run device batch transform."""

        class CustomOp(qml.operation.Operator):
            def decomposition(self):
                return [qml.PauliX(self.wires[0])]

        dev = DefaultQubit2()

        qs = qml.tape.QuantumScript([CustomOp(0)], [qml.expval(qml.PauliZ(0))])

        with pytest.warns(UserWarning, match="device batch transforms cannot be turned off"):
            results = qml.execute([qs], dev, device_batch_transform=False)

        assert len(results) == 1
        assert qml.math.allclose(results[0], -1)

    def test_error_if_return_types_not_enabled(self):
        """Check that an error is raised if return types is not enabled."""
        qml.disable_return()

        dev = DefaultQubit2()

        qs = qml.tape.QuantumScript([], [qml.state()])
        with pytest.raises(ValueError, "New device interface only works with return types enabled"):
            qml.execute([qs], dev)

        qml.enable_return()


@pytest.mark.parametrize("gradient_fn", (None, "backprop", qml.gradients.param_shift))
def test_caching(gradient_fn):
    dev = DefaultQubit2()

    qs = qml.tape.QuantumScript([qml.PauliX(0)], [qml.expval(qml.PauliZ(0))])

    cache = {}

    with qml.Tracker(dev) as tracker:
        results = qml.execute([qs, qs], dev, cache=cache, gradient_fn=gradient_fn)
        results2 = qml.execute([qs, qs], dev, cache=cache, gradient_fn=gradient_fn)

    assert results == [-1.0, -1.0]
    assert results2 == [-1.0, -1.0]

    assert tracker.totals["batches"] == 1
    assert tracker.totals["executions"] == 1
    assert cache[qs.hash] == -1.0
