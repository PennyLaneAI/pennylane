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
"""Contains the 'marker' utility for marking PennyLane objects."""

from collections.abc import Callable

from .qnode import QNode


def marker(obj: QNode | None = None, level: str | None = None) -> QNode | Callable:
    """Marks the compile pipeline of a QNode with a level label."""

    if isinstance(obj, QNode) and level is not None:
        obj.compile_pipeline.add_marker(level)
        return obj

    # NOTE: In order to use as decorator: @qml.marker(level="blah")
    if isinstance(obj, str):
        level = obj
        obj = None

    if obj is None and level is not None:

        def decorator(qnode: QNode) -> QNode:
            qnode.compile_pipeline.add_marker(level)
            return qnode

        return decorator

    raise ValueError("marker requires a 'level' argument.")
