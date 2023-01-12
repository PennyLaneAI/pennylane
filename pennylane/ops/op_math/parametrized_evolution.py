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

import pennylane as qml
from pennylane.operation import AnyWires, Operation

from .parametrized_hamiltonian import ParametrizedHamiltonian


class ParametrizedEvolution(Operation):
    r"""Parametrized evolution gate.

    For a time-dependent Hamiltonian of the form
    :math:`H(v, t) = H_\text{drift} + \sum_j f_j(v, t) H_j` it implements the corresponding
    time-evolution operator :math:`U(t_1, t_2)`, which is the solution to the time-dependent
    Schrodinger equation .. math:: \frac{d}{dt}U(t) = -i H(v, t) U(t).

    Under the hood, it is using a numerical ordinary differential equation solver.

    Args:
        base (ParametrizedHamiltonian): hamiltonian to evolve
        params (ndarray): trainable parameters
        t (Union[float, List[float]]): If a float, it corresponds to the duration of the evolution.
            If a list of two floats, it corresponds to the initial time and the final time of the
            evolution.
        dt (float): the time step used by the differential equation solver to evolve the
            time-dependent Hamiltonian. Defaults to XXX.
        do_queue (bool): determines if the scalar product operator will be queued. Default is True.
        id (str or None): id for the scalar product operator. Default is None.
    """

    _name = "ParametrizedEvolution"
    num_wires = AnyWires
    # pylint: disable=too-many-arguments, super-init-not-called
    def __init__(self, H: ParametrizedHamiltonian, params: list, t, dt=0.1, do_queue=True, id=None):
        self.H = H
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
            return -1j * qml.matrix(self.H(params, t)) @ y

        t = jnp.arange(self.t[0], self.t[1], self.dt, dtype=float)
        result = odeint(fun, y0, t, self.h_params)
        mat = result[-1]
        return qml.math.expand_matrix(mat, wires=self.wires, wire_order=wire_order)
