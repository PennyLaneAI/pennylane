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

import warnings
from functools import partial

import jax
import jax.numpy as jnp
from jax import linear_util as lu
from jax.flatten_util import ravel_pytree


@partial(jax.jit, static_argnums=0)
def odeint(func, y0, ts, *args):
    r"""Fix step size ODE solver with jit-compilation

    Solves the initial value problem (IVP) of an ordinary differential equation (ODE)

    .. math:: \frac{dy}{dt} = f(y, t), y(t_0) = y_0

    using an `explicit Runge-Kutta method <https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods>`_.
    In particular, it is using the 4th order term of the `Dormand Prince <https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method>`_ Butcher table.

    Args:
        func (callable): ``f(y, t, *args)`` defining the ODE. Needs to return a ``ndarray`` of the same shape as ``y``.
        y0 (ndarray): initial value ``y(t0) = y0``
        ts (ndarray): finite time steps for the ODE solver to take.

    .. note::

        Unlike many standard implementations, this ``odeint`` solver does not adaptively choose the step
        sizes, but rather a fix list of times for evaluation have to be provided in ``ts``. For an adaptive step
        size solver, we refer to `jax.experimental.ode.odeint <https://github.com/google/jax/blob/main/jax/experimental/ode.py>`_.

    .. warning::

        This function is written for ``jax`` only and will not work with other machine learning frameworks typically encountered in PennyLane.

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

    Formally, this solution can be written as the time-ordered exponential
    :math:`U(t_0, t_1) = \text{Texp}\left[-i \int_{t_0}^{t_1}d\tau H(\tau)\right]`
    (see `Dyson Series <https://en.wikipedia.org/wiki/Dyson_series>`_).
    """
    y0, unravel = ravel_pytree(y0)
    func = ravel_first_arg(func, unravel)
    out = _odeint(func, y0, ts, *args)
    return unravel(out)


def _abs2(x):
    """Computing the squared absolute value of x"""
    if jnp.iscomplexobj(x):
        return x.real**2 + x.imag**2
    return x**2


def _tolerance_warn(atol, rtol, err_ratio):
    warnings.warn(
        f"The targeted error tolerance of atol = {atol} and rtol = {rtol}"
        f"is not reached with a relative error of {err_ratio}",
        UserWarning,
    )


def _odeint(func, y0, ts, *args, atol=1e-8, rtol=1e-8):
    def func_(y, t):
        return func(y, t, *args)

    def scan_fun(carry, t1):
        y0, f0, t0 = carry
        dt = t1 - t0
        # not using y1_error and k atm
        y1, f1, y1_error = runge_kutta_step(func_, y0, f0, t0, dt)

        # check error
        # def mean_error_ratio(error_estimate, rtol, atol, y0, y1):
        err_tol = atol + rtol * jnp.maximum(jnp.abs(y0), jnp.abs(y1))
        err_ratio = y1_error / err_tol.astype(y1_error.dtype)
        mean_err_ratio = jnp.sqrt(jnp.mean(_abs2(err_ratio)))

        # warn_string = f"Warning: target tolerance atol={atol}, rtol={rtol} is not met"
        # warn_string = jnp.where(mean_err_ratio > 1.0, warn_string, None)
        # print(warn_string)

        # jax.lax.cond(
        #     mean_err_ratio > 1.0, # and isinstance(mean_err_ratio, jnp.ndarray),
        #     _tolerance_warn,
        #     lambda atol, rtol, err_ratio: None,
        #     atol,
        #     rtol,
        #     err_ratio,
        # )

        carry = [y1, f1, t1]
        return carry, y1

    # scan = fancy for-loop in jax, see https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
    f0 = func_(y0, ts[0])
    init = [y0, f0, ts[0]]
    carry, _ = jax.lax.scan(scan_fun, init, ts[1:])
    y, _, _ = carry
    # we typically only care about the final U
    # In case we want to return all ys, the output of
    # odeint() must be vmapped, jax.vmap(unravel)(out)
    return y  # jnp.concatenate((y0[None], ys))


def runge_kutta_step(func, y0, f0, t0, dt):
    """Performing the Runge Kutta 45 step

    Butcher table taken from `wikipedia.org/Dormand Prince <https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method>`_
    """
    alpha = jnp.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0, 0], dtype=dt.dtype)
    beta = jnp.array(
        [
            [1 / 5, 0, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        ],
        dtype=f0.dtype,
    )
    c_sol = jnp.array(
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=f0.dtype
    )

    # TODO: include error check
    c_error = jnp.array(
        [
            35 / 384 - 1951 / 21600,
            0,
            500 / 1113 - 22642 / 50085,
            125 / 192 - 451 / 720,
            -2187 / 6784 - -12231 / 42400,
            11 / 84 - 649 / 6300,
            -1.0 / 60.0,
        ],
        dtype=f0.dtype,
    )

    def body_fun(i, k):
        ti = t0 + dt * alpha[i - 1]
        yi = y0 + dt.astype(f0.dtype) * jnp.dot(beta[i - 1, :], k)
        ft = func(yi, ti)
        return k.at[i, :].set(ft)

    k = jnp.zeros((7, f0.shape[0]), f0.dtype).at[0, :].set(f0)
    k = jax.lax.fori_loop(1, 7, body_fun, k)

    y1 = dt.astype(f0.dtype) * jnp.dot(c_sol, k) + y0
    y1_error = dt.astype(f0.dtype) * jnp.dot(c_error, k)  # see coeffs, this is y(coeff) - y(coeff*)
    f1 = k[-1]
    return y1, f1, y1_error


def ravel_first_arg(f, unravel):
    """ "Decorate a function to work with a raveled first argument"""
    return _ravel_first_arg(lu.wrap_init(f), unravel).call_wrapped


@lu.transformation
def _ravel_first_arg(unravel, y_flat, *args):
    y = unravel(y_flat)
    ans = yield (y,) + args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat
