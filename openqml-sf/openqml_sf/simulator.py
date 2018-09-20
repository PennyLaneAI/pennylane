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
"""This module contains the Strawberry Fields abstract simulator device"""
import abc

import numpy as np

from openqml import Device
import strawberryfields as sf

from ._version import __version__


class StrawberryFieldsSimulator(Device):
    """Abstract StrawberryFields simulator device for OpenQML.

    Args:
      wires (int): the number of modes to initialize the device in.
      shots (int): number of circuit evaluations/random samples used to estimate expectation values of observables.
        For simulator devices, 0 means the exact EV is returned.
      hbar (float): the convention chosen in the canonical commutation relation :math:`[x, p] = i \hbar`
    """
    name = 'Strawberry Fields Simulator OpenQML plugin'
    api_version = '0.1.0'
    version = __version__
    author = 'Josh Izaac'

    short_name = 'strawberryfields'
    _operator_map = None

    def __init__(self, wires, *, shots=0, hbar=2):
        super().__init__(self.short_name, wires, shots)
        self.hbar = hbar
        self.eng = None
        self.q = None
        self.state = None

    def execution_context(self):
        """Initialize the engine"""
        self.reset()
        self.eng, self.q = sf.Engine(self.wires, hbar=self.hbar)
        return self.eng

    def apply(self, gate_name, wires, params):
        """Apply a quantum operation.

        Args:
          gate_name (str): name of the operation
          wires (Sequence[int]): subsystems the operation is applied on
          par (tuple): parameters for the operation
        """
        gate = self._operator_map[gate_name](*params)
        gate | [self.q[i] for i in wires] #pylint: disable=pointless-statement

    @abc.abstractmethod
    def pre_expectations(self):
        """Run the engine"""
        raise NotImplementedError

    def expectation(self, observable, wires, params):
        """Expectation value of an observable.

        Args:
          observable (str): name of the observable
          wires (Sequence[int]): subsystems the observable is measured on
          params (tuple): parameters for the observable
        Returns:
          float: expectation value
        """
        ex, var = self._observable_map[observable](self.state, wires, params)

        if self.shots != 0:
            # estimate the expectation value
            # use central limit theorem, sample normal distribution once, only ok
            # if shots is large (see https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem)
            ex = np.random.normal(ex, np.sqrt(var / self.shots))

        return ex

    def reset(self):
        """Reset the device"""
        if self.eng is not None:
            self.eng.reset()
            self.eng = None
        if self.state is not None:
            self.state = None
        if self.q is not None:
            self.q = None
