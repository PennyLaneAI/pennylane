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
"""This module contains the device class"""
import numpy as np

from openqml import Device, DeviceError

import strawberryfields as sf

#import state preparations
from strawberryfields.ops import (Catstate, Coherent, DensityMatrix, DisplacedSqueezed,
                                  Fock, Ket, Squeezed, Thermal, Gaussian)
# import decompositions
from strawberryfields.ops import (GaussianTransform, Interferometer)
# import gates
from strawberryfields.ops import (BSgate, CKgate, CXgate, CZgate, Dgate, Fouriergate,
                                  Kgate, Pgate, Rgate, S2gate, Sgate, Vgate, Xgate, Zgate)
# import measurements
from strawberryfields.ops import (MeasureFock, MeasureHeterodyne, MeasureHomodyne)


from ._version import __version__


operator_map = {
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



class StrawberryFieldsGaussian(Device):
    """StrawberryFields Gaussian device for OpenQML.

    wires (int): the number of modes to initialize the device in.
    hbar (float): the convention chosen in the canonical commutation
        relation [x, p] = i hbar. The default value is hbar=2.
    """
    name = 'Strawberry Fields Gaussian OpenQML plugin'
    short_name = 'strawberryfields.gaussian'
    api_version = '0.1.0'
    version = __version__
    author = 'Josh Izaac'
    _gates = set(operator_map.keys())
    _observables = {'Fock', 'X', 'P', 'Homodyne'}
    _circuits = {}

    def __init__(self, wires, *, shots=0, hbar=2):
        super().__init__(self.short_name, wires, shots)
        self.hbar = hbar
        self.eng = None
        self.q = None
        self.state = None

    def pre_execute_queued(self):
        self.reset()
        self.eng, self.q = sf.Engine(self.wires, hbar=self.hbar)

    def execute_queued_with(self):
        return self.eng

    def apply(self, gate_name, wires, *par):
        gate = operator_map[gate_name](*par)
        gate | [self.q[i] for i in wires] #pyling: disable=pointless-statement

    def pre_execute_expectations(self):
        self.state = self.eng.run('gaussian')

    def expectation(self, observable, wires, *par):
        # calculate expectation value
        if observable == 'Fock':
            ex = self.state.mean_photon(wires)
            var = 0
        elif observable == 'X':
            ex, var = self.state.quad_expectation(wires, 0)
        elif observable == 'P':
            ex, var = self.state.quad_expectation(wires, np.pi/2)
        elif observable == 'Homodyne':
            # note: we are explicitly intervening in `par` here because the
            # gaussian backend only works when par is numeric (not list[numeric])
            ex, var = self.state.quad_expectation(wires, par[0][0])
        else:
            raise DeviceError("Observable {} not supported by {}".format(observable.name, self.name))

        if self.shots != 0:
            # estimate the expectation value
            # use central limit theorem, sample normal distribution once, only ok
            # if shots is large (see https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem)
            ex = np.random.normal(ex, np.sqrt(var / self.shots))

        return ex

    def supported(self, gate_name):
        return gate_name in operator_map

    def reset(self):
        """Reset the device"""
        if self.eng is not None:
            self.eng.reset()
            self.eng = None
        if self.state is not None:
            self.state = None
        if self.q is not None:
            self.q = None
