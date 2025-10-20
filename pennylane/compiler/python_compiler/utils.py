# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General purpose utilities to use with xDSL."""

from numbers import Number

from xdsl.dialects import arith, builtin, tensor
from xdsl.ir import SSAValue

from .dialects.stablehlo import ConstantOp as hloConstantOp


def get_scalar_constant(value: SSAValue) -> Number | None:
    """Return the concrete number corresponding to an SSA value if it is a numerical constant.

    Numerical constants can be created by ``arith.constant`` and ``stablehlo.constant``. If scalar
    values are created using ``stablehlo.constant``, their owner will be a ``tensor.extract``,
    because ``stablehlo.constant`` returns a ``tensor``.

    Args:
        value (xdsl.ir.SSAValue): the SSA value to check

    Returns:
        Number or None: If the value corresponds to a scalar constant, its concrete value
        will be returned, else ``None``.
    """

    # If the value is a container, we can assume that it is not scalar. We check
    # this because constant-like operations can return container types. This includes
    # arith.constant, which may return containers, and stablehlo.constant, which
    # always returns a tensor.
    if not isinstance(value.type, builtin.ContainerType):
        owner = value.owner
        if isinstance(owner, arith.ConstantOp):
            return owner.value.data

        # If a scalar constant is created by stablehlo.constant, there will be a tensor.extract
        # to remove the scalar value from the tensor returned by stablehlo.constant.
        if isinstance(owner, tensor.ExtractOp):
            tensor_ = owner.tensor
            if len(tensor_.shape) == 0 and isinstance(tensor_.owner, hloConstantOp):
                dense_attr = tensor_.owner.value
                # We know that the tensor has shape (). Dense element attributes store
                # their data as a sequence. For a scalar, this will be a sequence with
                # a single element.
                return dense_attr.get_values()[0]

    return None
