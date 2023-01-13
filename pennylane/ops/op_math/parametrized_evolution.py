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

# pylint: disable=too-few-public-methods,function-redefined

"""
This file contains the ``ParametrizedEvolution`` operator.
"""

import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, Operation

from .parametrized_hamiltonian import ParametrizedHamiltonian


class ParametrizedEvolution(Operation):
    r"""Parametrized evolution gate.

    For a time-dependent Hamiltonian of the form
    :math:`H(v, t) = H_\text{drift} + \sum_j f_j(v, t) H_j` it implements the corresponding
    time-evolution operator :math:`U(t_1, t_2)`, which is the solution to the time-dependent
    Schrodinger equation

    .. math:: \frac{d}{dt}U(t) = -i H(v, t) U(t).


    Under the hood, it is using a numerical ordinary differential equation solver.

    Args:
        H (ParametrizedHamiltonian): hamiltonian to evolve
        params (ndarray): trainable parameters
        t (Union[float, List[float]]): If a float, it corresponds to the duration of the evolution.
            If a list of two floats, it corresponds to the initial time and the final time of the
            evolution.
        dt (float): the time step used by the differential equation solver to evolve the
            time-dependent Hamiltonian. Defaults to XXX.
        time (str, optional): The name of the time-based parameter in the parametrized Hamiltonian.
            Defaults to "t".
        do_queue (bool): determines if the scalar product operator will be queued. Default is True.
        id (str or None): id for the scalar product operator. Default is None.
    """

    _name = "ParametrizedEvolution"
    num_wires = AnyWires
    # pylint: disable=too-many-arguments, super-init-not-called
    def __init__(
        self, H: ParametrizedHamiltonian, params: list, t, dt=None, time="t", do_queue=True, id=None
    ):
        self.H = H
        self.time = time
        self.dt = dt
        self.h_params = params
        self.t = [0, t] if qml.math.size(t) == 1 else t
        super().__init__(wires=H.wires, do_queue=do_queue, id=id)

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return all(op.has_matrix or isinstance(op, qml.Hamiltonian) for op in self.H.ops)

    # pylint: disable=import-outside-toplevel
    def matrix(self, wire_order=None):
        try:
            import jax.numpy as jnp
            from jax.experimental.ode import odeint
        except ImportError as e:
            raise ImportError(
                "Module jax is required for ``ParametrizedEvolution`` class. "
                "You can install jax via: pip install jax"
            ) from e
        y0 = jnp.eye(2 ** len(self.wires), dtype=complex)

        def fun(y, t, params):
            """dy/dt = -i H(t) y"""
            kwargs = {self.time: t}
            return -1j * qml.matrix(self.H(params, **kwargs)) @ y

        if self.dt is None:
            # TODO: Figure out what is 'p', and best values for 'rtol' and 'atol'
            self.dt = guess_dt(fun, self.t[0], y0, 2, *self.h_params)

        t = jnp.arange(self.t[0], self.t[1], self.dt, dtype=float)
        result = odeint(fun, y0, t, self.h_params)
        mat = result[-1]
        return qml.math.expand_matrix(mat, wires=self.wires, wire_order=wire_order)


def guess_dt(fun, t0, y0, p, *args, rtol=1e-4, atol=1e-4):
    """Compute a guess for the time step.

    This algorithm is described in further detail in:

    `E. Hairer, S. P. Norsett G. Wanner, Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4. <http://mezbanhabibi.ir/wp-content/uploads/2020/01/ordinary-differential-equations-vol.1.-Nonstiff-problems.pdf>`_

    Args:
        fun (Callable): function to evaluate the time derivative of the solution `y` at time
            `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
        t0 (float): initial time
        y0 (ndarray): array or pytree of arrays representing the initial value for the state.
        p (float): order (still don't know what this is)
        rtol (float): relative local error tolerance for solver
        atol (float): absolute local error tolerance for solver
        params (list): extra parameters for the function

    Raises:
        ImportError: _description_

    Returns:
        _type_: _description_
    """
    try:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError(
            "Module jax is required for ``ParametrizedEvolution`` class. "
            "You can install jax via: pip install jax"
        ) from e
    dtype = y0.dtype

    f0 = fun(y0, t0, args)

    scale = atol + np.abs(y0) * rtol
    d0 = jnp.linalg.norm(y0 / scale.astype(dtype))
    d1 = jnp.linalg.norm(f0 / scale.astype(dtype))

    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    y1 = y0 + h0.astype(dtype) * f0
    f1 = fun(y1, t0 + h0, args)
    d2 = jnp.linalg.norm((f1 - f0) / scale.astype(dtype)) / h0

    h1 = (
        jnp.maximum(1e-6, h0 * 1e-3)
        if (d1 <= 1e-15 and d2 <= 1e-15)
        else (0.01 / jnp.max(jnp.array([d1, d2]))) ** (1.0 / (p + 1.0))
    )

    return jnp.minimum(100.0 * h0, h1)
