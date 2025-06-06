# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
This file contains all the custom exceptions and warnings used in PennyLane.
"""


class CaptureError(Exception):
    """Errors related to PennyLane's Program Capture execution pipeline."""


class DeviceError(Exception):  # pragma: no cover
    """Exception raised when it encounters an illegal operation in the quantum circuit."""


class QuantumFunctionError(Exception):  # pragma: no cover
    """Exception raised when an illegal operation is defined in a quantum function."""


class PennyLaneDeprecationWarning(UserWarning):  # pragma: no cover
    """Warning raised when a PennyLane feature is being deprecated."""


class ExperimentalWarning(UserWarning):  # pragma: no cover
    """Warning raised to indicate experimental/non-stable feature or support."""
