# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pennylane as qml
from pennylane.ops.symbolic import Sum, PauliSum


def sum(*summands):

    if all(s.name in ["PauliX", "PauliY", "PauliZ", "Identity"] for s in summands):
        return PauliSum(*summands)

    return Sum(*summands)
