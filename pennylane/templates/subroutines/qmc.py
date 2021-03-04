# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Contains the ``QuantumMonteCarlo`` template.
"""
from pennylane.templates.decorator import template


def distribution_to_unitary(distribution):
    """TODO"""
    ...

def random_variable_to_unitary(random_variable):
    """TODO"""
    ...


@template
def QuantumMonteCarlo(distributions, random_variable, target_wires, estimation_wires):
    """TODO
    """
    ...