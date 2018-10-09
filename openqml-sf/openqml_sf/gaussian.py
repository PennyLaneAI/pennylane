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
Strawberry Fields gaussian plugin
=================================

**Module name:** :mod:`openqml_sf.gaussian`

.. currentmodule:: openqml_sf.gaussian

The SF gaussian plugin implements all the :class:`~openqml.device.Device` methods
and provides a gaussian simulation of a continuous variable quantum circuit architecture.

Classes
-------

.. autosummary::
   StrawberryFieldsGaussian

----
"""

import numpy as np

#import state preparations
from strawberryfields.ops import (Coherent, DisplacedSqueezed,
                                  Squeezed, Thermal, Gaussian)
# import gates
from strawberryfields.ops import (BSgate, CXgate, CZgate, Dgate,
                                  Pgate, Rgate, S2gate, Sgate)

from .expectations import *
from .simulator import StrawberryFieldsSimulator


class StrawberryFieldsGaussian(StrawberryFieldsSimulator):
    """StrawberryFields Gaussian device for OpenQML.
    """
    name = 'Strawberry Fields Gaussian OpenQML plugin'
    short_name = 'strawberryfields.gaussian'

    _operator_map = {
        'CoherentState': Coherent,
        'DisplacedSqueezedState': DisplacedSqueezed,
        'SqueezedState': Squeezed,
        'ThermalState': Thermal,
        'GaussianState': Gaussian,
        'Beamsplitter': BSgate,
        'ControlledAddition': CXgate,
        'ControlledPhase': CZgate,
        'Displacement': Dgate,
        'QuadraticPhase': Pgate,
        'Rotation': Rgate,
        'TwoModeSqueezing': S2gate,
        'Squeezing': Sgate
    }

    _observable_map = {
        'PhotonNumber': PNR,
        'X': Homodyne(0),
        'P': Homodyne(np.pi/2),
        'Homodyne': Homodyne(),
        'PolyXP': Order2Poly,
    }

    _circuits = {}

    def pre_expectations(self):
        self.state = self.eng.run('gaussian')
