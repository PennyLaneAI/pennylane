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
r"""
State preperations are templates that prepare a given quantum state,
by decomposing it into elementary operations.
"""

from .mottonen import MottonenStatePreparation
from .basis import BasisStatePreparation
from .arbitrary_state_preparation import ArbitraryStatePreparation
from .basis_qutrit import QutritBasisStatePreparation
from .cosine_window import CosineWindow
