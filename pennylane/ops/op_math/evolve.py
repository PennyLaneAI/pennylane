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

from typing import Union

import pennylane as qml
from pennylane.operation import AnyWires, Operation, Operator

from .exp import Evolution
from .parametrized_hamiltonian import ParametrizedHamiltonian

has_jax = True
try:
    import jax.numpy as jnp
    from jax.experimental.ode import odeint
except ImportError as e:
    has_jax = False


def evolve(op: Union[Operator, ParametrizedHamiltonian]):
    """Returns a new operator that computes the evolution of the given operator.

    Args:
        op (Operator): operator to evolve

    Returns:
        Evolution | ParametrizedEvolution: evolution operator
    """
    if isinstance(op, ParametrizedHamiltonian):

        def parametrized_evolution(params: list, t):
            """Constructor for the :class:`ParametrizedEvolution` operator.

            Args:
                params (Union[list, tuple, ndarray]): trainable parameters
                t (Union[float, List[float]]): If a float, it corresponds to the duration of the evolution.
                    If a list of two floats, it corresponds to the initial time and the final time of the
                    evolution.
                dt (float, optional): the time step used by the differential equation solver to evolve the
                    time-dependent Hamiltonian. Defaults to XXX.

            Returns:
                ParametrizedEvolution: class used to compute the parametrized evolution of the given
                    hamiltonian
            """
            return ParametrizedEvolution(H=op, params=params, t=t)

        return parametrized_evolution

    return Evolution(generator=op, param=1.0)


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
            evolution. Note that such absolute times only have meaning within an instance of
            ``ParametrizedEvolution`` and will not affect other gates.
        time (str, optional): The name of the time-based parameter in the parametrized Hamiltonian.
            Defaults to "t".
        do_queue (bool): determines if the scalar product operator will be queued. Default is True.
        id (str or None): id for the scalar product operator. Default is None.
    """

    _name = "ParametrizedEvolution"
    num_wires = AnyWires
    # pylint: disable=too-many-arguments, super-init-not-called
    def __init__(
        self, H: ParametrizedHamiltonian, params: list, t, time="t", do_queue=True, id=None
    ):
        if not has_jax:
            raise ImportError(
                "Module jax is required for the ``ParametrizedEvolution`` class. You can install jax via: pip install jax"
            )
        if not all(op.has_matrix or isinstance(op, qml.Hamiltonian) for op in H.ops):
            raise ValueError(
                "All operators inside the parametrized hamiltonian must have a matrix defined."
            )
        self.H = H
        self.time = time
        self.h_params = params
        self.t = jnp.array([0, t], dtype=float) if qml.math.size(t) == 1 else t
        super().__init__(wires=H.wires, do_queue=do_queue, id=id)

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return True

    # pylint: disable=import-outside-toplevel
    def matrix(self, wire_order=None):
        y0 = jnp.eye(2 ** len(self.wires), dtype=complex)

        def fun(y, t):
            """dy/dt = -i H(t) y"""
            return -1j * qml.matrix(self.H(self.h_params, t=t)) @ y

        result = odeint(fun, y0, self.t)
        mat = result[-1]
        return qml.math.expand_matrix(mat, wires=self.wires, wire_order=wire_order)
