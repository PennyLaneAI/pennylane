# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def __getattr__(name):
    if name == "FermionOperator":
        from openfermion.ops import FermionOperator
        return FermionOperator
    if name == "bravyi_kitaev":
        from openfermion.transforms import bravyi_kitaev
        return bravyi_kitaev
    if name == "jordan_wigner":
        from openfermion.transforms import jordan_wigner
        return jordan_wigner

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
