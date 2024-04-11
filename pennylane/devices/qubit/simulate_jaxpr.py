# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simulate PLEXPR."""

from pennylane.operation import Operator

from .initialize_state import create_initial_state
from .apply_operation import apply_operation
from .measure import measure

has_jax = True
try:
    import jax
    from jax._src.util import safe_map
except ImportError:
    has_jax = False


def simulate_jaxpr(jaxpr: "jax.core.ClosedJaxpr", n_wires: int, *args):
    """Execute jaxpr using default.qubit utilities."""
    if not has_jax:
        raise ImportError

    print(jaxpr)

    if isinstance(jaxpr, jax.core.ClosedJaxpr):
        jaxpr = jaxpr.jaxpr

    state = create_initial_state(tuple(range(n_wires)))

    env = {}

    def read(var):
        return var.val if isinstance(var, jax.core.Literal) else env[var]

    def write(var, val):
        env[var] = val

    print(jaxpr.invars, args)
    args = tuple(a for a in args if a is not None)
    safe_map(write, jaxpr.invars, args)

    for eqn in jaxpr.eqns:
        invals = safe_map(read, eqn.invars)
        if eqn.primitive.name == "measure":
            shots = eqn.params["shots"]
            print(env)
            print(shots)
            outvals = [measure(mp, state) for mp in invals]
        else:
            outvals = eqn.primitive.bind(*invals, **eqn.params)

        if isinstance(outvals, Operator) and isinstance(eqn.outvars[0], jax.core.DropVar):
            state = apply_operation(outvals, state)
            continue
        if not eqn.primitive.multiple_results:
            outvals = [outvals]

        safe_map(write, eqn.outvars, outvals)

    return safe_map(read, jaxpr.outvars)
