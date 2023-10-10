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
r"""Solve the Diophantine equation :math:`t^{\dagger}t = \xi`"""


def diophantine_dyadic(g, xi):
    """
    Given some randomness and a xi value, solve the Diophantine equation or fail.

    TODO: figure out how this indicates that it failed.
    """
    return (_g + xi for _g in g)
