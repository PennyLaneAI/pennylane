# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the base class for symbolic operators."""

from typing import override

from pennylane.core.operator import Operator2
from pennylane.core.queuing import QueuingManager


class SymbolicOp2(Operator2, is_baseclass=True):
    """Developer-facing base class for symbolic operators."""

    base: Operator2  # declares the existance of a `base` attribute for static type checking

    @property
    @override
    def arithmetic_depth(self) -> int:
        return 1 + self.base.arithmetic_depth

    @property
    @override
    def is_verified_hermitian(self) -> bool:
        return self.base.is_verified_hermitian

    @property
    @override
    def has_matrix(self) -> bool:  # pylint: disable=arguments-differ,invalid-overridden-method
        return self.base.has_matrix

    @property
    @override
    # pylint: disable=arguments-differ,invalid-overridden-method
    def has_sparse_matrix(self) -> bool:
        return self.base.has_sparse_matrix

    @override
    def queue(self, context=QueuingManager):
        context.remove(self.base)
        context.append(self)
        return self
