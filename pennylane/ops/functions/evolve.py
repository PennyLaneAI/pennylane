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
from functools import partial

from pennylane.operation import Operator
from pennylane.ops.op_math import Evolution, ParametrizedHamiltonian


def evolve(op: Operator, dt: float = 1e-1):
    """Returns a new operator to compute the evolution of the given operator.

    Args:
        op (Operator): operator to evolve
        time (str, optional): The name of the time-based parameter in the parametrized Hamiltonian.
            This argument is just used when ``op`` is an instance of the
            :class:`ParametrizedHamiltonian` class. Defaults to "t".
        dt (float, optional): the time step used by the differential equation solve to evolve
            the time-dependent Hamiltonian. Defaults to XXX.

    Returns:
        Evolution | ParametrizedEvolution: evolution operator
    """
    if isinstance(op, ParametrizedHamiltonian):
        # need this import here to avoid raising an error when jax is not installed
        from pennylane.pulse import ParametrizedEvolution  # pylint: disable=import-outside-toplevel

        return partial(ParametrizedEvolution, H=op, dt=dt)  # pylint: disable=no-member
    return Evolution(generator=op, param=1)
