# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
from jax import numpy as jnp
from jax.interpreters import ad

from pennylane import adjoint, generator
from pennylane.capture.primitives import AbstractOperator

from .apply_operation import apply_operation
from .initialize_state import create_initial_state


def _read(env, var):
    """Return the value and tangent for a variable."""
    return (var.val, ad.Zero(var.aval)) if isinstance(var, jax.core.Literal) else env[var]


def _operator_forward_pass(eqn, env, ket):
    """Apply an operator during the forward pass of the adjoint jvp."""
    invals, tangents = tuple(zip(*(_read(env, var) for var in eqn.invars)))
    op = eqn.primitive.bind(*invals, **eqn.params)
    env[eqn.outvars[0]] = (op, ad.Zero(AbstractOperator()))

    if isinstance(eqn.outvars[0], jax.core.DropVar):
        return apply_operation(op, ket)
    if any(not isinstance(t, ad.Zero) for t in tangents):
        # derivatives of op arithmetic. Should be possible later
        raise NotImplementedError

    return ket


def _measurement_forward_pass(eqn, env, ket):
    """Perform a measurement during the forward pass of the adjoint jvp."""
    invals, tangents = tuple(zip(*(_read(env, var) for var in eqn.invars)))

    if any(not isinstance(t, ad.Zero) for t in tangents):
        raise NotImplementedError

    if eqn.primitive.name != "expval_obs":
        raise NotImplementedError("adjoint jvp only supports expectations of observables.")

    mp = eqn.primitive.bind(*invals, **eqn.params)
    bra = apply_operation(mp.obs, ket)
    result = jnp.real(jnp.sum(jnp.conj(bra) * ket))
    env[eqn.outvars[0]] = (result, None)
    return bra


def _other_prim_forward_pass(eqn: jax.core.JaxprEqn, env: dict) -> None:
    """Handle any equation that is not an operator or measurement eqn.

    Maps outputs back to the environment
    """
    invals, tangents = tuple(zip(*(_read(env, var) for var in eqn.invars)))
    if eqn.primitive not in ad.primitive_jvps:
        raise NotImplementedError
    outvals, doutvals = ad.primitive_jvps[eqn.primitive](invals, tangents, **eqn.params)
    if not eqn.primitive.multiple_results:
        outvals = [outvals]
        doutvals = [doutvals]
    for var, v, dv in zip(eqn.outvars, outvals, doutvals):
        env[var] = (v, dv)


def _forward_pass(jaxpr: jax.core.Jaxpr, env: dict, num_wires: int):
    """Calculate the forward pass off an adjoint jvp calculation."""
    bras = []
    ket = create_initial_state(range(num_wires))

    for eqn in jaxpr.eqns:
        if getattr(eqn.primitive, "prim_type", "") == "operator":
            ket = _operator_forward_pass(eqn, env, ket)

        elif getattr(eqn.primitive, "prim_type", "") == "measurement":
            bra = _measurement_forward_pass(eqn, env, ket)
            bras.append(bra)
        else:
            _other_prim_forward_pass(eqn, env)
    return bras, ket


def _backward_pass(jaxpr, bras, ket, env):
    """Calculate the jvps during the backward pass stage of an adjoint jvp."""
    out_jvps = [jnp.array(0.0)] * len(bras)

    for eqn in reversed(jaxpr.eqns):
        if getattr(eqn.primitive, "prim_type", "") == "operator" and isinstance(
            eqn.outvars[0], jax.core.DropVar
        ):
            op = env[eqn.outvars[0]][0]

            if eqn.invars:
                tangents = [_read(env, var)[1] for var in eqn.invars]
                # assuming just one tangent for now
                t = tangents[0]
                if not isinstance(t, ad.Zero):
                    ket_temp = apply_operation(generator(op, format="observable"), ket)
                    for i, bra in enumerate(bras):
                        out_jvps[i] += -2 * t * jnp.imag(jnp.sum(jnp.conj(bra) * ket_temp))
                if any(not isinstance(t, ad.Zero) for t in tangents[1:]):
                    raise NotImplementedError(
                        "adjoint jvp only differentiable parameters in the 0 position."
                    )

            adj_op = adjoint(op)
            ket = apply_operation(adj_op, ket)
            bras = [apply_operation(adj_op, bra) for bra in bras]

    return out_jvps


def execute_and_jvp(jaxpr: jax.core.Jaxpr, args: tuple, tangents: tuple, num_wires: int):
    """Execute and calculate the jvp for a jaxpr."""
    env = {
        var: (arg, tangent)
        for var, arg, tangent in zip(jaxpr.constvars + jaxpr.invars, args, tangents, strict=True)
    }

    bras, ket = _forward_pass(jaxpr, env, num_wires)
    results = [_read(env, var)[0] for var in jaxpr.outvars]
    return results, _backward_pass(jaxpr, bras, ket, env)
