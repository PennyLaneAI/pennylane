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
"""
Strawberry Fields Fock plugin
=============================

**Module name:** :mod:`openqml_sf.fock`

.. currentmodule:: openqml_sf.fock

The SF Fock plugin implements all the :class:`~openqml.device.Device` methods
and provides a Fock space simulation of a continuous variable quantum circuit architecture.

Classes
-------

.. autosummary::
   StrawberryFieldsFock

----
"""

import numpy as np

#import state preparations
from strawberryfields.ops import (Catstate, Coherent, DensityMatrix, DisplacedSqueezed,
                                  Fock, Ket, Squeezed, Thermal, Gaussian)
# import gates
from strawberryfields.ops import (BSgate, CKgate, CXgate, CZgate, Dgate,
                                  Kgate, Pgate, Rgate, S2gate, Sgate, Vgate)

from .expectations import *
from .simulator import StrawberryFieldsSimulator


class StrawberryFieldsFock(StrawberryFieldsSimulator):
    r"""StrawberryFields Fock device for OpenQML.

    Args:
        wires (int): the number of modes to initialize the device in.
        shots (int): number of circuit evaluations/random samples used to estimate expectation values of observables.
            For simulator devices, 0 means the exact EV is returned.
        cutoff_dim (int): Fock space truncation dimension
        hbar (float): the convention chosen in the canonical commutation relation :math:`[x, p] = i \hbar`
    """
    name = 'Strawberry Fields Fock OpenQML plugin'
    short_name = 'strawberryfields.fock'

    _operator_map = {
        'CatState': Catstate,
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
        'PhotonNumber': PNR,
        'X': Homodyne(0),
        'P': Homodyne(np.pi/2),
        'Homodyne': Homodyne(),
        'PolyXP': Order2Poly,
    }

    _circuits = {}

    def __init__(self, wires, *, shots=0, cutoff_dim, hbar=2):
        super().__init__(wires, shots=shots, hbar=hbar)
        self.cutoff = cutoff_dim

    def pre_expectations(self):
        self.state = self.eng.run('fock', cutoff_dim=self.cutoff)
