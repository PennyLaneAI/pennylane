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
The previous location for pennylane.core.queuing.
"""

# pylint: disable=wildcard-import, unused-wildcard-import
from pennylane.core.queuing import *  # tach-ignore


def __getattr__(key):
    if key == "process_queue":
        raise AttributeError(
            "pennylane.queuing.process_queue has been moved to qp.tape.qscript.from_queue"
        )
    raise AttributeError(f"module 'pennylane.queuing' has no attribute '{key}'")
