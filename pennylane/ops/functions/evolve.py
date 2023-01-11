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
from pennylane.ops.op_math import Evolution, ParametrizedEvolution, ParametrizedHamiltonian


def evolve(op: Operator):
    """Returns a new operator to compute the evolution of the given operator.

    Args:
        op (Operator): operator to evolve

    Returns:
        Evolution | ParametrizedEvolution: evolution operator
    """
    if isinstance(op, ParametrizedHamiltonian):
        return partial(ParametrizedEvolution, H=op)  # pylint: disable=no-member
    return Evolution(generator=op, param=1)
