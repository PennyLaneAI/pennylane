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
"""
Pytest helper functions for inspecting captured programs.
"""


def loop_hints(jaxpr, skip_none=False):
    """Return the ``estimated_iterations`` of all ``for_loop`` and ``while_loop`` primitives in
    ``jaxpr`` and its nested jaxprs, in depth-first order.
    Skips hints that are None if ``skip_none=True``."""
    hints = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in {"for_loop", "while_loop"}:
            hint = eqn.params["estimated_iterations"]
            if not (skip_none and hint is None):
                hints.append(hint)
        for param in eqn.params.values():
            for sub in param if isinstance(param, (list, tuple)) else (param,):
                sub = getattr(sub, "jaxpr", sub)
                if hasattr(sub, "eqns"):
                    hints.extend(loop_hints(sub))
    return hints
