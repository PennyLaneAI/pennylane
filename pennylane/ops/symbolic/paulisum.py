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
from pennylane.ops.symbolic import Sum


class PauliSum(Sum):
    """Arithmetic operator subclass representing the sum of Pauli operators"""

    def __init__(self, *paulis, do_queue=True, id=None):

        # for now this class only has an additional check,
        # but it could get a lot more functionality!
        for p in paulis:
            if p.name not in ["PauliX", "PauliY", "PauliZ", "Identity"]:
                raise ValueError(f"PauliSum can only be created from Pauli operators; got {p.name}")

        super().__init__(
            *paulis, do_queue=do_queue, id=id
        )
