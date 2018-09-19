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
"""This module contains the Fock simulator device"""
import numpy as np

#import state preparations
from strawberryfields.ops import (Catstate, Coherent, DensityMatrix, DisplacedSqueezed,
                                  Fock, Ket, Squeezed, Thermal, Gaussian)
# import gates
from strawberryfields.ops import (BSgate, CKgate, CXgate, CZgate, Dgate, Fouriergate,
                                  Kgate, Pgate, Rgate, S2gate, Sgate, Vgate, Xgate, Zgate)


from .expectations import PNR, Homodyne
from .simulator import StrawberryFieldsSimulator


class StrawberryFieldsFock(StrawberryFieldsSimulator):
    """StrawberryFields Fock device for OpenQML.

    wires (int): the number of modes to initialize the device in.
    shots (int): the number of simulation runs used to calculate
        the expectaton value and variance. If 0, the exact expectation
        and variance is returned.
    cutoff_dim (int): the Fock space truncation.
    hbar (float): the convention chosen in the canonical commutation
        relation [x, p] = i hbar. The default value is hbar=2.
    """
    name = 'Strawberry Fields Fock OpenQML plugin'
    short_name = 'strawberryfields.fock'

    _operator_map = {
        'CatState:': Catstate,
        'CoherentState': Coherent,
        'FockDensityMatrix': DensityMatrix,
        'DisplacedSqueezedState': DisplacedSqueezed,
        'FockState': Fock,
        'FockStateVector': Ket,
        'SqueezedState': Squeezed,
        'ThermalState': Thermal,
        'GaussianState': Gaussian,
        'Beamsplitter': BSgate,
        'CrossKerr': CKgate,
        'ControlledAddition': CXgate,
        'ControlledPhase': CZgate,
        'Displacement': Dgate,
        'Kerr': Kgate,
        'QuadraticPhase': Pgate,
        'Rotation': Rgate,
        'TwoModeSqueezing': S2gate,
        'Squeezing': Sgate,
        'CubicPhase': Vgate
    }

    _observable_map = {
        'Fock': PNR,
        'X': Homodyne(0),
        'P': Homodyne(np.pi/2),
        'Homodyne': Homodyne()
    }

    _circuits = {}

    def __init__(self, wires, *, shots=0, cutoff_dim, hbar=2):
        self.cutoff = cutoff_dim
        super().__init__(wires, shots=shots, hbar=hbar)

    def pre_execute_expectations(self):
        self.state = self.eng.run('fock', cutoff_dim=self.cutoff)
