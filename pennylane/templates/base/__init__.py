# Copyright 2018-2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Base templates serve as building blocks for other templates. They define a fundamental structure of how to arrange
unitaries on wires. The unitary can be represented by a PennyLane :meth:`~.pennylane.operations.Operation`,
or by a user-defined template that consists of a sequence of gates.
It can also be parametrized, in which case the parameters are passed to the base template.
"""

from .single import Single

