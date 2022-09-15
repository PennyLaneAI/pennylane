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
This subpackage contains the quantum tape, which tracks, queues, and
validates quantum operations and measurements.
"""
from .tape import QuantumTape, get_active_tape, TapeError
from .operation_recorder import OperationRecorder
from .unwrap import Unwrap, UnwrapTape


def __getattr__(name):
    if name == "stop_recording":
        from warnings import warn  # pylint: disable=import-outside-toplevel
        from pennylane.queuing import QueuingContext  # pylint: disable=import-outside-toplevel

        warn("qml.tape.stop_recording has been moved to qml.queuing.stop_recording", UserWarning)
        return QueuingContext.stop_recording
    try:
        return globals()[name]
    except KeyError as e:
        raise AttributeError from e
