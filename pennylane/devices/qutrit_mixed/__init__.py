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
Submodule for performing mixed state simulations of qutrit-based quantum circuits.

This submodule is internal and subject to change without a deprecation cycle. Use
at your own discretion.


.. currentmodule:: pennylane.devices.qutrit_mixed
.. autosummary::
    :toctree: api

    create_initial_state
    apply_operation
    measure
    measure_with_samples
    sample_state
    simulate
"""

from .apply_operation import apply_operation
from .initialize_state import create_initial_state
from .measure import measure
from .sampling import sample_state, measure_with_samples
from .simulate import simulate, get_final_state, measure_final_state
