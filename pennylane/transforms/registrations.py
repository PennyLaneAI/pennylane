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
Contains the global dictionaries for transform registrations.
"""

CONTROL_MAPS = {}
"""Dict[type, Callable[Operation, None]]:
Mapping from operation types to methods that create
concrete controlled versions of a given operation.
Functions should take the given operation as input
and return nothing. The desired controlled operation(s)
should be added to the tape context by setting ``do_queue=True``.
"""


def register_control(cls, fn):
    """Register the control transform for a custom Operation.

    TODO(chase): Documentation.
    """
    CONTROL_MAPS[cls] = fn
