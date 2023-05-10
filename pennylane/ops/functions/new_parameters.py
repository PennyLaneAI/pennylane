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
"""
This module contains the qml.new_parameters function.
"""

from typing import Union, Sequence
import copy

import pennylane as qml
from pennylane.operation import Operator


def new_parameters(op: Operator, params: Sequence[Union[float, Sequence]]) -> Operator:
    """Create a new operator with updated parameters

    This function takes an :class:`~.Operator` and new parameters as input and
    returns a new :class:`~.Operator` of the same type with the new parameters.

    Args:
        op (.Operator): Operator to update
        params (Sequence[Union[float, Sequence]]): New parameters to create operator with

    Returns:
        .Operator: New operator with updated parameters

    Raises:
        ValueError: If the shape of the old and new operator parameters don't match
    """
    params = qml.math.convert_like(params, op.data)

    if qml.math.shape(params) != qml.math.shape(op.data):
        raise ValueError(
            "The shape of the new parameters does not match the expected shape; "
            f"got {qml.math.shape(params)}, expected {qml.math.shape(op.data)}."
        )

    new_op = copy.deepcopy(op)
    new_op.data = params

    return new_op
