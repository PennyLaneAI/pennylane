# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: This code is heavily inspired by https://github.com/google/jax/blob/main/jax/experimental/ode.py
# need to make sure copyright is not infringed
"""Fix step size ODE solver"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import linear_util as lu
from jax.flatten_util import ravel_pytree


@partial(jax.jit, static_argnums=0)
def odeint(func, y0, ts, *args):
    """Fix step size ODE solver

    Solves the initial value problem (IVP) of an ordinary differential equation (ODE)

    .. math:: \frac{dy}{dt} = f(y, t), y(t_0) = y_0

    using an `explicit Runge-Kutta method <https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods>`_.

    Args:
        func (callable): ``f(y, t, *args)`` defining the ODE
        y0 (tensor_like): initial value ``y(t0) = y0)``
        ts (tensor_like): finite time steps for the ODE solver to take.

    .. note::

        Unlike many standard implementations, this ``odeint`` solver does not adaptively choose the step
        sizes, but rather a fix list of times for evaluation have to be provided in ``ts``.

    **Example**

    We can solve the time-dependent Schrodinger equation

    .. math:: \frac{d}{dt}U = -i H(t) U

    for a time-dependent Hamiltonian :math:`H(t) = X_0 X_1 + v \sin(t) Z_0 Y_1`
    for the time window ``(t0, t1) = (0, 4)`` using ``odeint`` in the following way:

    .. code:: python3

        XX = qml.matrix(qml.PauliX(0) @ qml.PauliX(1))
        ZY = qml.matrix(qml.PauliZ(0) @ qml.PauliY(1))

        ts = jnp.linspace(0., 4., 20)
        y0 = jnp.eye(2**2, dtype=complex)
        params = jnp.ones(1, dtype=complex)

        def func(y, t, params):
            H = jnp.array([XX + params[0] * jnp.sin(t) * ZY])
            return -1j * jnp.sum(H, axis=0) @ y

        U = qml.math.odeint(func, y0, ts, params)
    """
    y0, unravel = ravel_pytree(y0)
    func = ravel_first_arg(func, unravel)
    out = _odeint(func, y0, ts, *args)
    return unravel(out)


def _odeint(func, y0, ts, *args):
    def func_(y, t):
        return func(y, t, *args)

    def scan_fun(carry, t1):
        y0, f0, t0 = carry
        dt = t1 - t0
        # not using y1_error and k atm
        y1, f1, y1_error, k = runge_kutta_step(func_, y0, f0, t0, dt)
        carry = [y1, f1, t1]
        return carry, y1
    
    # scan = fancy for-loop in jax, see https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
    f0 = func_(y0, ts[0])
    init = [y0, f0, ts[0]]
    carry, Us = jax.lax.scan(scan_fun, init, ts[1:])
    U, _, _ = carry
    # we typically only care about the final U
    # In case we want to return all Us, the output of 
    # odeint() must be vmapped, jax.vmap(unravel)(out)
    return U #jnp.concatenate((y0[None], Us))

def runge_kutta_step(func, y0, f0, t0, dt):
    # Dopri5 Butcher tableaux
    # taken from https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
    alpha = jnp.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1., 0], dtype=dt.dtype)
    beta = jnp.array(
        [[1 / 5, 0, 0, 0, 0, 0, 0], [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]],
        dtype=f0.dtype)
    c_sol = jnp.array(
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        dtype=f0.dtype)
    
    # TODO: include error check
    c_error = jnp.array([
        35 / 384 - 1951 / 21600, 0, 500 / 1113 - 22642 / 50085, 125 / 192 -
        451 / 720, -2187 / 6784 - -12231 / 42400, 11 / 84 - 649 / 6300, -1. / 60.
    ], dtype=f0.dtype)

    def body_fun(i, k):
        ti = t0 + dt * alpha[i-1]
        yi = y0 + dt.astype(f0.dtype) * jnp.dot(beta[i-1, :], k)
        ft = func(yi, ti)
        return k.at[i, :].set(ft)

    k = jnp.zeros((7, f0.shape[0]), f0.dtype).at[0, :].set(f0)
    k = jax.lax.fori_loop(1, 7, body_fun, k)

    y1 = dt.astype(f0.dtype) * jnp.dot(c_sol, k) + y0
    y1_error = dt.astype(f0.dtype) * jnp.dot(c_error, k) # see coeffs, this is y(coeff) - y(coeff*)
    f1 = k[-1]
    return y1, f1, y1_error, k


def ravel_first_arg(f, unravel):
    return ravel_first_arg_(lu.wrap_init(f), unravel).call_wrapped

@lu.transformation
def ravel_first_arg_(unravel, y_flat, *args):
    y = unravel(y_flat)
    ans = yield (y,) + args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat

