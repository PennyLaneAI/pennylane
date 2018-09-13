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
"""This module contains the device class and context manager"""
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


class StrawberryFieldsFock(Device):
    """StrawberryFields Fock device for OpenQML.

    wires (int): the number of modes to initialize the device in.
    cutoff_dim (int): the Fock space truncation. Must be specified before
        applying a qfunc.
    hbar (float): the convention chosen in the canonical commutation
        relation [x, p] = i hbar. The default value is hbar=2.
    """
    name = 'Strawberry Fields Fock OpenQML plugin'
    short_name = 'strawberryfields.fock'
    api_version = '0.1.0'
    version = __version__
    author = 'Josh Izaac'
    _gates = set(operator_map.keys())
    _observables = {'Fock', 'X', 'P', 'Homodyne'}
    _circuits = {}

    def __init__(self, wires, *, shots=0, cutoff_dim, hbar=2):
        self.wires = wires
        self.cutoff = cutoff_dim
        self.hbar = hbar
        self.eng = None
        self.q = None
        self.state = None
        super().__init__(self.short_name, shots)

    def pre_execute_queued(self):
        self.reset()
        self.eng, self.q = sf.Engine(self.wires, hbar=self.hbar)

    def execute_queued_with(self):
        return self.eng

    def apply(self, gate_name, wires, *par):
        gate = operator_map[gate_name](*par)
        if isinstance(wires, int):
            gate | self.q[wires] #pylint: disable=pointless-statement
        else:
            gate | [self.q[i] for i in wires] #pylint: disable=pointless-statement

    def expectation(self, observable, wires, *par):
        self.state = self.eng.run('fock', cutoff_dim=self.cutoff)

        # calculate expectation value
        if observable == 'Fock':
            expectation_value = self.state.mean_photon(wires)
            variance = 0
        elif observable == 'X':
            expectation_value, variance = self.state.quad_expectation(wires, 0)
        elif observable == 'P':
            expectation_value, variance = self.state.quad_expectation(wires, np.pi/2)
        elif observable == 'Homodyne':
            expectation_value, variance = self.state.quad_expectation(wires, *par)
        else:
            raise DeviceError("Observable {} not supported by {}".format(observable.name, self.name))

        if self.shots != 0:
            # estimate the expectation value
            # use central limit theorem, sample normal distribution once, only ok
            # if shots is large (see https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem)
            expectation_value = np.random.normal(expectation_value, np.sqrt(var / self.shots))

        return expectation_value

    def supported(self, gate_name):
        if gate_name not in operator_map:
            raise DeviceError("Operation {} not supported by device {}".format(gate_name, self.name))

    def reset(self):
        """Reset the device"""
        if self.eng is not None:
            self.eng.reset()
            self.eng = None
        if self.state is not None:
            self.state = None
        if self.q is not None:
            self.q = None
