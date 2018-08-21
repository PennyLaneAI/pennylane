# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the operations"""
from .operation import Operation

from .builtins_continuous import (CatState, CoherentState, FockDensityMatrix, DisplacedSqueezedState,
                                  FockState, FockStateVector, SqueezedState, ThermalState, GaussianState)
from .builtins_continuous import (Beamsplitter, ControlledAddition, ControlledPhase, Displacement,
                                  Kerr, QuadraticPhase, Rotation, Squeezing, CubicPhase)

from .builtins_discrete import (CNOT, CZ, Hadamard, PauliX, PauliY, PauliZ, PhaseShift, Rot, RX, RY, RZ,
                                SWAP, QubitStateVector, QubitUnitary)