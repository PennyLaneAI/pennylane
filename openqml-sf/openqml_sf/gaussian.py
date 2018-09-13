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
    'CoherentState': Coherent,
    'DisplacedSqueezed': DisplacedSqueezed,
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
    'Squeeze': Sgate,
    # 'XDisplacement': Xgate,
    # 'PDisplacement': Zgate,
    # 'MeasureHomodyne': MeasureHomodyne,
    # 'MeasureHeterodyne': MeasureHeterodyne
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
    _observables = {'Fock', 'X', 'P', 'Homodyne', 'Heterodyne'}
    _circuits = {}

    def __init__(self, wires, *, shots=0, hbar=2):
        self.wires = wires
        self.hbar = hbar
        self.eng = None
        self.state = None
        super().__init__(self.short_name, shots)

    def execute(self, queue, observe):
        """Apply the queued operations to the device, and measure the expectation."""
        if self.eng:
            self.eng.reset()
            self.reset()

        self.eng, q = sf.Engine(self.wires, hbar=self.hbar)

        with self.eng:
            for operation in queue:
                if operation.name not in operator_map:
                    raise DeviceError("{} not supported by device {}".format(operation.name, self.short_name))

                p = operation.parameters()
                op = operator_map[operation.name](*p)
                if isinstance(operation.wires, int):
                    op | q[operation.wires]
                else:
                    op | [q[i] for i in operation.wires]

        self.state = self.eng.run('gaussian')

        # calculate expectation value
        ev_list = [] # list of returned expectation values
        for expectation in observe:
            reg = expectation.wires
            if expectation.name == 'Fock':
                ex = self.state.mean_photon(reg)
                var = 0
            elif expectation.name == 'X':
                ex, var = self.state.quad_expectation(reg, 0)
            elif expectation.name == 'P':
                ex, var = self.state.quad_expectation(reg, np.pi/2)
            elif expectation.name == 'Homodyne':
                ex, var = self.state.quad_expectation(reg, *expectation.params)
            elif expectation.name == 'Displacement':
                ex = self.state.displacement(modes=reg)
            else:
                raise DeviceError("Observable {} not supported by {}".format(expectation.name, expectation.name))

            if self.shots != 0:
                # estimate the expectation value
                # use central limit theorem, sample normal distribution once, only ok
                # if shots is large (see https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem)
                ex = np.random.normal(ex, np.sqrt(var / self.shots))

            ev_list.append(ex)

        return np.array(ev_list, dtype=np.float64)

    def reset(self):
        """Reset the device"""
        if self.eng is not None:
            self.eng = None
            self.state = None
