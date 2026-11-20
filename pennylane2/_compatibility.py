# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reject Classic before initializing PennyLane2 global state."""

import sys


def check_classic_import():
    """Raise ImportError if Classic PennyLane is already loaded."""
    if sys.modules.get("pennylane") is not None:
        raise ImportError(
            "PennyLane2 cannot be imported in a process that already imported Classic PennyLane "
            "(pennylane). Start a new Python process or restart the notebook kernel, then "
            "import only one generation. Both packages may remain installed."
        )
