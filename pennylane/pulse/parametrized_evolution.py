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
from jax import numpy as jnp
from jax.experimental.ode import odeint

import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops import ParametrizedHamiltonian


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
        t1 (float): starting time
        t2 (float): end time
        time (string): The name of the time-based parameter in the parametrized Hamiltonian.
            Defaults to "t".
        dt (float): the time step used by the differential equation solver to evolve the
            time-dependent Hamiltonian. If ``None``, the value XXX is used. Defaults to ```None```.
        do_queue (bool): determines if the scalar product operator will be queued. Default is True.
        id (str or None): id for the scalar product operator. Default is None.
    """

    _name = "ParametrizedEvolution"
    num_wires = AnyWires
    # pylint: disable=too-many-arguments, super-init-not-called
    def __init__(self, H: ParametrizedHamiltonian, params: list, t, dt=0.1, do_queue=True, id=None):
        self.H = H
        self.dt = dt
        self.data = params
        self.t = [0, t] if qml.math.size(t) == 1 else t
        self._id = id
        self._inverse = False
        if do_queue:
            self.queue()

    @property
    def wires(self):
        return self.H.wires

    def matrix(self, wire_order=None):

        y0 = jnp.eye(2 ** len(self.wires), dtype=complex)

        def fun(y, t, params):
            """dy/dt = -i H(t) y"""
            return -1j * qml.matrix(self.H(params, t)) @ y

        t = jnp.arange(self.t[0], self.t[1], self.dt, dtype=float)
        result = odeint(fun, y0, t, self.data)

        return result[-1]
