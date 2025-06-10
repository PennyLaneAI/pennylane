# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Compute the jvp of a jaxpr using the adjoint Jacobian method.
"""
import jax
from jax import numpy as jnp
from jax.interpreters import ad

from pennylane import adjoint, generator
from pennylane.capture import pause
from pennylane.capture.primitives import AbstractOperator

from .apply_operation import apply_operation
from .initialize_state import create_initial_state


def _read(env, var):
    """Return the value and tangent for a variable."""
    return (var.val, ad.Zero(var.aval)) if isinstance(var, jax.extend.core.Literal) else env[var]


def _operator_forward_pass(eqn, env, ket):
    """Apply an operator during the forward pass of the adjoint jvp."""
    invals, tangents = tuple(zip(*(_read(env, var) for var in eqn.invars)))
    op = eqn.primitive.impl(*invals, **eqn.params)
    env[eqn.outvars[0]] = (op, ad.Zero(AbstractOperator()))

    if any(not isinstance(t, ad.Zero) for t in tangents[1:]):
        raise NotImplementedError("adjoint jvp only differentiable parameters in the 0 position.")

    if isinstance(eqn.outvars[0], jax.core.DropVar):
        return apply_operation(op, ket)
    if any(not isinstance(t, ad.Zero) for t in tangents):
        # derivatives of op arithmetic. Should be possible later
        raise NotImplementedError

    return ket


def _measurement_forward_pass(eqn, env, ket):
    """Perform a measurement during the forward pass of the adjoint jvp."""
    invals, tangents = tuple(zip(*(_read(env, var) for var in eqn.invars)))

    if any(not isinstance(t, ad.Zero) for t in tangents):  # pragma: no cover
        # currently prevented by "no differentiable operator arithmetic."
        # but better safe than sorry to keep this error
        raise NotImplementedError  # pragma: no cover

    if eqn.primitive.name != "expval_obs":
        raise NotImplementedError("adjoint jvp only supports expectations of observables.")

    mp = eqn.primitive.impl(*invals, **eqn.params)
    bra = apply_operation(mp.obs, ket)
    result = jnp.real(jnp.vdot(bra, ket))
    env[eqn.outvars[0]] = (result, None)
    return bra


def _other_prim_forward_pass(eqn: jax.extend.core.JaxprEqn, env: dict) -> None:
    """Handle any equation that is not an operator or measurement eqn.

    Maps outputs back to the environment
    """
    invals, tangents = tuple(zip(*(_read(env, var) for var in eqn.invars)))
    if eqn.primitive not in ad.primitive_jvps:
        raise NotImplementedError(
            f"Primitive {eqn.primitive} does not have a jvp rule and is not supported."
        )
    outvals, doutvals = ad.primitive_jvps[eqn.primitive](invals, tangents, **eqn.params)
    if not eqn.primitive.multiple_results:
        outvals = [outvals]
        doutvals = [doutvals]
    for var, v, dv in zip(eqn.outvars, outvals, doutvals, strict=True):
        env[var] = (v, dv)


def _forward_pass(jaxpr: jax.extend.core.Jaxpr, env: dict, num_wires: int):
    """Calculate the forward pass of an adjoint jvp calculation."""
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

    results = [_read(env, var)[0] for var in jaxpr.outvars]
    return bras, ket, results


def _backward_pass(jaxpr, bras, ket, results, env):
    """Calculate the jvps during the backward pass stage of an adjoint jvp."""
    out_jvps = [jnp.zeros_like(r) for r in results]

    modified = False
    for eqn in reversed(jaxpr.eqns):
        if getattr(eqn.primitive, "prim_type", "") == "operator" and isinstance(
            eqn.outvars[0], jax.core.DropVar
        ):
            op = env[eqn.outvars[0]][0]

            if eqn.invars:
                tangent = _read(env, eqn.invars[0])[1]
                if not isinstance(tangent, ad.Zero):
                    ket_temp = apply_operation(generator(op, format="observable"), ket)
                    modified = True
                    for i, bra in enumerate(bras):
                        out_jvps[i] += -2 * tangent * jnp.imag(jnp.vdot(bra, ket_temp))

            adj_op = adjoint(op, lazy=False)
            ket = apply_operation(adj_op, ket)
            bras = [apply_operation(adj_op, bra) for bra in bras]

    if modified:
        return out_jvps
    return [ad.Zero(r.aval) for r in results]


@pause()  # need to be able to temporarily create instances, but still have it jittable
def execute_and_jvp(jaxpr: jax.extend.core.Jaxpr, args: tuple, tangents: tuple, num_wires: int):
    """Execute and calculate the jvp for a jaxpr using the adjoint method.

    Args:
        jaxpr (jax.extend.core.Jaxpr): the jaxpr to evaluate
        args : an iterable of tensorlikes.  Should include the consts followed by the inputs
        tangents: an iterable of tensorlikes and ``jax.interpreter.ad.Zero`` objects.  Should
            include the consts followed by the inputs.
        num_wires (int): the number of wires to use.

    Note that the consts for the jaxpr should be included at the beginning of both the ``args``
    and ``tangents``.
    """
    env = {
        var: (arg, tangent)
        for var, arg, tangent in zip(jaxpr.constvars + jaxpr.invars, args, tangents, strict=True)
    }

    bras, ket, results = _forward_pass(jaxpr, env, num_wires)
    return results, _backward_pass(jaxpr, bras, ket, results, env)
