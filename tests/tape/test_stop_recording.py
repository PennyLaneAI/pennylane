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

import pytest

import pennylane as qml


def test_tape_stop_recording():
    """Test that accessing qml.tape.stop_recording raises a deprecation warning and returns
    qml.queuing.stop_recording."""
    message = "qml.tape.stop_recording has moved to qml.QueuingManager.stop_recording"

    with qml.queuing.AnnotatedQueue() as q:
        with pytest.warns(UserWarning, match=message):
            with qml.tape.stop_recording():
                qml.PauliZ(0)

    assert len(q.queue) == 0
