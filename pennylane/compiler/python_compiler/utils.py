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

from xdsl.dialects.arith import ConstantOp as arithConstantOp
from xdsl.dialects.builtin import ComplexType, ShapedType
from xdsl.dialects.tensor import ExtractOp as tensorExtractOp
from xdsl.ir import SSAValue

from .dialects.stablehlo import ConstantOp as hloConstantOp


def get_constant_from_ssa(value: SSAValue) -> Number | None:
    """Return the concrete value corresponding to an SSA value if it is a numerical constant.

    .. note::

        This function currently only returns constants if they are scalar. For non-scalar
        constants, ``None`` will be returned.

    Args:
        value (xdsl.ir.SSAValue): the SSA value to check

    Returns:
        Number or None: If the value corresponds to a constant, its concrete value will
        be returned, else ``None``.
    """

    # If the value has a shape, we can assume that it is not scalar. We check
    # this because constant-like operations can return container types. This includes
    # arith.constant, which may return containers, and stablehlo.constant, which
    # always returns a container.
    if not isinstance(value.type, ShapedType):
        owner = value.owner

        if isinstance(owner, arithConstantOp):
            const_attr = owner.value
            return const_attr.value.data

        # Constant-like operations can also create scalars by returning rank 0 tensors.
        # In this case, the owner of a scalar value should be a tensor.extract, which
        # uses the aforementioned rank 0 constant tensor as input.
        if isinstance(owner, tensorExtractOp):
            tensor_ = owner.tensor
            if (
                len(owner.indices) == 0
                and len(tensor_.type.shape) == 0
                and isinstance(tensor_.owner, (arithConstantOp, hloConstantOp))
            ):
                dense_attr = tensor_.owner.value
                # We know that the tensor has shape (). Dense element attributes store
                # their data as a sequence. For a scalar, this will be a sequence with
                # a single element.
                val = dense_attr.get_values()[0]
                if isinstance(tensor_.type.element_type, ComplexType):
                    # If the dtype is complex, the value will be a 2-tuple containing
                    # the real and imaginary components of the number rather than a
                    # Python complex number
                    val = val[0] + 1j * val[1]

                return val

    return None
