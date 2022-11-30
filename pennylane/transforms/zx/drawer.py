# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transforms for drawing circuit in the framework for ZX calculus."""
import functools

from .converter import tape_to_graph_zx


def draw_zx(qnode):
    """Draw a qnode with PyZX"""

    @functools.wraps(qnode)
    def wrapper(*args, **kwargs_qnode):

        try:
            # pylint: disable=import-outside-toplevel
            import pyzx
        except ImportError as Error:
            raise ImportError(
                "This feature requires pyzx. It can be installed with: pip install pyzx"
            ) from Error

        qnode.construct(args, kwargs_qnode)

        graph_zx = tape_to_graph_zx(qnode.qtape)
        return pyzx.draw(graph_zx)

    return wrapper
