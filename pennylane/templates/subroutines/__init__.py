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
Subroutines are the most basic template, consisting of a collection of quantum operations, and not fulfilling
any of the characteristics of other templates (i.e. to prepare a specific state, to be repeated or to encode features).
"""

from .arbitrary_unitary import ArbitraryUnitary
from .commuting_evolution import CommutingEvolution
from .fermionic_double_excitation import FermionicDoubleExcitation
from .interferometer import Interferometer
from .fermionic_single_excitation import FermionicSingleExcitation
from .uccsd import UCCSD
from .approx_time_evolution import ApproxTimeEvolution
from .permute import Permute
from .qpe import QuantumPhaseEstimation
from .qmc import QuantumMonteCarlo
from .all_singles_doubles import AllSinglesDoubles
from .grover import GroverOperator
from .qft import QFT
from .kupccgsd import kUpCCGSD
from .hilbert_schmidt import HilbertSchmidt, LocalHilbertSchmidt
from .flip_sign import FlipSign
from .basis_rotation import BasisRotation
from .qsvt import QSVT, qsvt
from .select import Select
from .qdrift import QDrift
from .controlled_sequence import ControlledSequence
from .trotter import TrotterProduct
from .aqft import AQFT
from .fable import FABLE
from .reflection import Reflection
from .amplitude_amplification import AmplitudeAmplification
from .qubitization import Qubitization
