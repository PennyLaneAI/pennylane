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
This file contains the ``Evolve`` pulse gate.
"""
from typing import Union

from pennylane.operation import Operator
from pennylane.ops.op_math import Evolution, ParametrizedEvolution, ParametrizedHamiltonian


def evolve(op: Union[Operator, ParametrizedHamiltonian]):
    """Returns a new operator to compute the evolution of the given operator.

    Args:
        op (Operator): operator to evolve

    Returns:
        Evolution | ParametrizedEvolution: evolution operator
    """
    if isinstance(op, ParametrizedHamiltonian):

        def parametrized_evolution(params: list, t, dt=None):
            """Constructor for the :class:`ParametrizedEvolution` operator.

            Args:
                params (list): params (ndarray): trainable parameters
                t (Union[float, List[float]]): If a float, it corresponds to the duration of the evolution.
                    If a list of two floats, it corresponds to the initial time and the final time of the
                    evolution.
                dt (float, optional): the time step used by the differential equation solver to evolve the
                    time-dependent Hamiltonian. Defaults to XXX.

            Returns:
                ParametrizedEvolution: class used to compute the parametrized evolution of the given
                    hamiltonian
            """
            return ParametrizedEvolution(H=op, params=params, t=t, dt=dt)

        return parametrized_evolution

    return Evolution(generator=op, param=1.)
