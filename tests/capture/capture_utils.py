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
Utility functions for capture-related tests.
"""

import pytest

# pylint: disable=wrong-import-position
jax = pytest.importorskip("jax")
from jax._src.core import ClosedJaxpr, Jaxpr


def extract_ops_and_meas_prims(jaxpr):
    """Extract the primitives that are operators and measurements."""
    return [
        eqn
        for eqn in jaxpr.eqns
        if getattr(eqn.primitive, "prim_type", "") in ("operator", "measurement")
        or getattr(eqn.primitive, "name", "") == "measure"
    ]


def extract_all_primitives(jaxpr):
    """Extract all primitives."""
    primitives = set()

    for eqn in jaxpr.eqns:
        primitives.add(eqn.primitive)

        # NOTE: Must go through nested jaxpr inside HOP like Qnode, cond, etc
        for val in eqn.params.values():
            if isinstance(val, Jaxpr):
                primitives.update(extract_all_primitives(val))
            elif isinstance(val, ClosedJaxpr):
                primitives.update(extract_all_primitives(val.jaxpr))
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, Jaxpr):
                        primitives.update(extract_all_primitives(item))
                    elif isinstance(val, ClosedJaxpr):
                        primitives.update(extract_all_primitives(item.jaxpr))

    return primitives
