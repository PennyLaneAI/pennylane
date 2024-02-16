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
This module contains the discrete-variable quantum operations.

The operations are divided into the following files:

* ``arithmetic_ops.py``: Operations that perform arithmetic on the states.
* ``matrix_ops.py``: Generalized operations that accept a matrix parameter,
  either unitary or hermitian depending.
* ``non_parameteric_ops.py``: All operations with no parameters.
* ``observables.py``: Qubit observables excluding the Pauli gates, which are
  located in ``non_parameteric_ops.py`` instead.
* ``parametric_ops_single_qubit.py``: Core single qubit parametric operations.
* ``parametric_ops_multi_qubit.py``: Core multi-qubit parametric operations.
* ``qchem_ops.py``: Operations for quantum chemistry applications.
* ``state_preparation.py``: Operations that initialize the state.
* ``special_unitary.py``: The ``SpecialUnitary`` operation.
"""

from .arithmetic_ops import *
from .matrix_ops import *
from .non_parametric_ops import *
from .observables import *
from .parametric_ops_single_qubit import *
from .parametric_ops_multi_qubit import *
from .qchem_ops import *
from .state_preparation import *
from .special_unitary import SpecialUnitary
from .hamiltonian import Hamiltonian
from ..identity import Identity, GlobalPhase
from ..meta import Snapshot, Barrier, WireCut

__ops__ = {
    "Identity",
    "Snapshot",
    "Hadamard",
    "PauliX",
    "X",
    "PauliY",
    "Y",
    "PauliZ",
    "Z",
    "PauliRot",
    "MultiRZ",
    "S",
    "T",
    "SX",
    "SWAP",
    "ISWAP",
    "SISWAP",
    "SQISW",
    "PSWAP",
    "ECR",
    "RX",
    "RY",
    "RZ",
    "PhaseShift",
    "PCPhase",
    "CPhaseShift00",
    "CPhaseShift01",
    "CPhaseShift10",
    "Rot",
    "U1",
    "U2",
    "U3",
    "IsingXX",
    "IsingYY",
    "IsingZZ",
    "IsingXY",
    "BasisState",
    "StatePrep",
    "QubitStateVector",
    "QubitDensityMatrix",
    "QubitUnitary",
    "BlockEncode",
    "SpecialUnitary",
    "IntegerComparator",
    "DiagonalQubitUnitary",
    "SingleExcitation",
    "SingleExcitationPlus",
    "SingleExcitationMinus",
    "DoubleExcitation",
    "DoubleExcitationPlus",
    "DoubleExcitationMinus",
    "QubitCarry",
    "QubitSum",
    "OrbitalRotation",
    "FermionicSWAP",
    "Barrier",
    "WireCut",
    "GlobalPhase",
}


__obs__ = {
    "Hadamard",
    "PauliX",
    "X",
    "PauliY",
    "Y",
    "PauliZ",
    "Z",
    "Hermitian",
    "Projector",
    "SparseHamiltonian",
    "Hamiltonian",
}


__all__ = list(__ops__ | __obs__)
