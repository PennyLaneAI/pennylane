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
from .time_evolution import (
    ApproxTimeEvolution,
    CommutingEvolution,
    QDrift,
    TrotterizedQfunc,
    TrotterProduct,
    trotterize,
)
from .interferometer import Interferometer
from .permute import Permute
from .qft import QFT
from .qpe import QuantumPhaseEstimation
from .qmc import QuantumMonteCarlo
from .grover import GroverOperator
from .hilbert_schmidt import HilbertSchmidt, LocalHilbertSchmidt
from .flip_sign import FlipSign
from .fable import FABLE
from .select import Select
from .prepselprep import PrepSelPrep
from .reflection import Reflection
from .qubitization import Qubitization
from .controlled_sequence import ControlledSequence
from .aqft import AQFT
from .amplitude_amplification import AmplitudeAmplification
from .qrom import QROM
from .gqsp import GQSP
from .select_pauli_rot import SelectPauliRot
from .qsvt import poly_to_angles, QSVT, qsvt, transform_angles

from .qchem import (
    FermionicDoubleExcitation,
    FermionicSingleExcitation,
    UCCSD,
    AllSinglesDoubles,
    kUpCCGSD,
    BasisRotation,
)
from .arithmetic import (
    PhaseAdder,
    Adder,
    Multiplier,
    OutMultiplier,
    OutAdder,
    ModExp,
    OutPoly,
    SemiAdder,
    Elbow,
    TemporaryAND,
)

__all__ = [
    "ArbitraryUnitary",
    "ApproxTimeEvolution",
    "CommutingEvolution",
    "QDrift",
    "TrotterizedQfunc",
    "TrotterProduct",
    "trotterize",
    "Interferometer",
    "Permute",
    "QFT",
    "QuantumPhaseEstimation",
    "GroverOperator",
    "HilbertSchmidt",
    "LocalHilbertSchmidt",
    "FlipSign",
    "BasisRotation",
    "QuantumMonteCarlo",
    "FABLE",
    "Select",
    "PrepSelPrep",
    "Reflection",
    "Qubitization",
    "ControlledSequence",
    "AQFT",
    "AmplitudeAmplification",
    "QROM",
    "GQSP",
    "SelectPauliRot",
    "poly_to_angles",
    "QSVT",
    "qsvt",
    "transform_angles",
    "FermionicDoubleExcitation",
    "FermionicSingleExcitation",
    "UCCSD",
    "AllSinglesDoubles",
    "kUpCCGSD",
    "PhaseAdder",
    "Adder",
    "Multiplier",
    "OutMultiplier",
    "OutAdder",
    "ModExp",
    "OutPoly",
    "SemiAdder",
    "Elbow",
    "TemporaryAND",
]
