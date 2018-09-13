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
r"""
ProjectQ plugin
========================

**Module name:** :mod:`openqml.plugins.projectq`

.. currentmodule:: openqml.plugins.projectq

This plugin provides the interface between OpenQML and ProjecQ.
It enables OpenQML to optimize quantum circuits simulable with ProjectQ.

ProjecQ supports several different backends. Of those, the following are useful in the current context:

- projectq.backends.Simulator([gate_fusion, ...])	Simulator is a compiler engine which simulates a quantum computer using C++-based kernels.
- projectq.backends.ClassicalSimulator()	        A simple introspective simulator that only permits classical operations.
- projectq.backends.IBMBackend([use_hardware, ...])	The IBM Backend class, which stores the circuit, transforms it to JSON QASM, and sends the circuit through the IBM API.

See PluginAPI._capabilities['backend'] for a list of backend options.

Functions
---------

.. autosummary::
   init_plugin

Classes
-------

.. autosummary::
   Gate
   Observable
   PluginAPI

----
"""
import logging as log
import numpy as np
from numpy.random import (randn,)
from openqml import Device, DeviceError
from openqml import Variable

import projectq as pq

# import operations
from projectq.ops import (HGate, XGate, YGate, ZGate, SGate, TGate, SqrtXGate, SwapGate, SqrtSwapGate, Rx, Ry, Rz, R, Ph, StatePreparation, HGate, SGate, TGate, SqrtXGate, SqrtSwapGate
)
from .ops import (CNOT, CZ, Toffoli, AllZGate, Rot, QubitUnitary)

from ._version import __version__


operator_map = { #todo: make this a class property
    'PauliX': XGate,
    'PauliY': YGate,
    'PauliZ': ZGate,
    'CNOT': CNOT,
    'CZ': CZ,
    'SWAP': SwapGate,
    'RX': Rx,
    'RY': Ry,
    'RZ': Rz,
    'PhaseShift': R,
    'QubitStateVector': StatePreparation,
    'Hadamard': HGate,
    #gates not native to OpenQML
    'S': SGate,
    'T': TGate,
    'SqrtX': SqrtXGate,
    'SqrtSwap': SqrtSwapGate,
    'AllZ': AllZGate,
    #gates not implemented in ProjectQ
    'Rot': Rot,
    'QubitUnitary': QubitUnitary,
}

class ProjectQDevice(Device):
    """ProjectQ device for OpenQML.

    Args:
       wires (int): The number of qubits of the device.

    Keyword Args for Simulator backend:
      gate_fusion (bool): If True, gates are cached and only executed once a certain gate-size has been reached (only has an effect for the c++ simulator).
      rnd_seed (int): Random seed (uses random.randint(0, 4294967295) by default).

    Keyword Args for IBMBackend backend:
      use_hardware (bool): If True, the code is run on the IBM quantum chip (instead of using the IBM simulator)
      num_runs (int): Number of runs to collect statistics. (default is 1024)
      verbose (bool): If True, statistics are printed, in addition to the measurement result being registered (at the end of the circuit).
      user (string): IBM Quantum Experience user name
      password (string): IBM Quantum Experience password
      device (string): Device to use (‘ibmqx4’, or ‘ibmqx5’) if use_hardware is set to True. Default is ibmqx4.
      retrieve_execution (int): Job ID to retrieve instead of re-running the circuit (e.g., if previous run timed out).
    """
    name = 'ProjectQ OpenQML plugin'
    short_name = 'projectq'
    api_version = '0.1.0'
    plugin_version = __version__
    author = 'Christian Gogolin'
    _capabilities = {'backend': list(["Simulator", "ClassicalSimulator", "IBMBackend"])}

    def __init__(self, wires, **kwargs):
        kwargs.setdefault('shots', 0)
        super().__init__(self.short_name, kwargs['shots'])

        # translate some aguments
        for k,v in {'log':'verbose'}.items():
            if k in kwargs:
                kwargs.setdefault(v, kwargs[k])

        # clean some arguments
        if 'num_runs' in kwargs:
            if isinstance(kwargs['num_runs'], int) and kwargs['num_runs']>0:
                self.n_eval = kwargs['num_runs']
            else:
                self.n_eval = 0
                del(kwargs['num_runs'])

        self.wires = wires
        self.backend = kwargs['backend']
        del(kwargs['backend'])
        self.kwargs = kwargs
        self.eng = None
        self.reg = None
        #self.reset() #the actual initialization is done in reset(), but we don't need to call this manually as Device does it for us during __enter__()

    def reset(self):
        self.reg = self.eng.allocate_qureg(self.wires)

    def __repr__(self):
        return super().__repr__() +'Backend: ' +self.backend +'\n'

    def __str__(self):
        return super().__str__() +'Backend: ' +self.backend +'\n'

    def post_execute_queued(self):
        self._deallocate()

    def supported(self, gate_name):
        return gate_name not in operator_map

    def apply(self, gate_name, wires, *par):
        gate = operator_map[gate_name](*par)
        if isinstance(wires, int):
            gate | self.reg[wires] #pylint: disable=pointless-statement
        else:
            gate | tuple([self.reg[i] for i in wires]) #pylint: disable=pointless-statement

    def expectation(self, observable, wires):
        raise NotImplementedError("expectation() is not yet implemented for this backend")

    # def __del__(self):
    #     self._deallocate()

    def _deallocate(self):
        """Deallocate all qubits to make ProjectQ happy

        See also: https://github.com/ProjectQ-Framework/ProjectQ/issues/2

        Drawback: This is probably rather resource intensive.
        """
        if self.eng is not None and self.backend == 'Simulator' or self.backend == 'IBMBackend':
            pq.ops.All(pq.ops.Measure) | self.reg #avoid an unfriendly error message: https://github.com/ProjectQ-Framework/ProjectQ/issues/2

    def _deallocate2(self):
        """Another proposal for how to deallocate all qubits to make ProjectQ happy

        Unsuitable because: Produces a segmentation fault.
        """
        if self.eng is not None and self.backend == 'Simulator' or self.backend == 'IBMBackend':
             for qubit in self.reg:
                 self.eng.deallocate_qubit(qubit)

    def _deallocate3(self):
        """Another proposal for how to deallocate all qubits to make ProjectQ happy

        Unsuitable because: Throws an error if the probability for the given collapse is 0.
        """
        if self.eng is not None and self.backend == 'Simulator' or self.backend == 'IBMBackend':
            self.eng.flush()
            self.eng.backend.collapse_wavefunction(self.reg, [0 for i in range(len(self.reg))])

    def filter_kwargs_for_backend(self, kwargs):
        return { key:value for key,value in kwargs.items() if key in self._backend_kwargs }


class ProjectQSimulator(ProjectQDevice):
    """ProjectQ Simulator device for OpenQML.

    Args:
       wires (int): The number of qubits of the device.

    Keyword Args:
      gate_fusion (bool): If True, gates are cached and only executed once a certain gate-size has been reached (only has an effect for the c++ simulator).
      rnd_seed (int): Random seed (uses random.randint(0, 4294967295) by default).
    """

    short_name = 'projectq.simulator'
    _gates = set(operator_map.keys())
    _observables = set([ key for (key,val) in operator_map.items() if val in [XGate, YGate, ZGate, AllZGate] ])
    _circuits = {}
    _backend_kwargs = ['gate_fusion', 'rnd_seed']

    def __init__(self, wires, **kwargs):
        kwargs['backend'] = 'Simulator'
        super().__init__(wires, **kwargs)

    def reset(self):
        """Resets the engine and backend

        After the reset the Device should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        backend = pq.backends.Simulator(**self.filter_kwargs_for_backend(self.kwargs))
        self.eng = pq.MainEngine(backend)
        super().reset()


    def expectation(self, observable, wires):
        self.eng.flush(deallocate_qubits=False)

        ev_list = [] # list of returned expectation values
        for expectation in self._observe:
            if expectation.name == 'PauliX' or expectation.name == 'PauliY' or expectation.name == 'PauliZ':
                ev = self.eng.backend.get_expectation_value(pq.ops.QubitOperator(str(observable)[-1]+'0'), self.reg)
                variance = 1 - ev**2
            elif expectation.name == 'AllPauliZ':
                ev = [ self.eng.backend.get_expectation_value(pq.ops.QubitOperator("Z"+'0'), [qubit]) for qubit in self.reg]
                variance = [1 - e**2 for e in ev]
            else:
                raise DeviceError("Observable {} not supported by {}".format(self._observe.name, self.name))

            ev_list.append(ev)

        return np.array(ev_list, dtype=np.float64)


class ProjectQClassicalSimulator(ProjectQDevice):
    """ProjectQ ClassicalSimulator device for OpenQML.

    Args:
       wires (int): The number of qubits of the device.
    """

    short_name = 'projectq.classicalsimulator'
    _gates = set([ key for (key,val) in operator_map.items() if val in [XGate, CNOT] ])
    _observables = set([ key for (key,val) in operator_map.items() if val in [ZGate, AllZGate] ])
    _circuits = {}
    _backend_kwargs = []

    def __init__(self, wires, **kwargs):
        kwargs['backend'] = 'ClassicalSimulator'
        super().__init__(wires, **kwargs)

    def reset(self):
        """Resets the engine and backend

        After the reset the Device should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        backend = pq.backends.ClassicalSimulator(**self.filter_kwargs_for_backend(self.kwargs))
        self.eng = pq.MainEngine(backend)
        super().reset()

class ProjectQIBMBackend(ProjectQDevice):
    """ProjectQ IBMBackend device for OpenQML.

    Args:
       wires (int): The number of qubits of the device.

    Keyword Args:
      use_hardware (bool): If True, the code is run on the IBM quantum chip (instead of using the IBM simulator)
      num_runs (int): Number of runs to collect statistics. (default is 1024)
      verbose (bool): If True, statistics are printed, in addition to the measurement result being registered (at the end of the circuit).
      user (string): IBM Quantum Experience user name
      password (string): IBM Quantum Experience password
      device (string): Device to use (‘ibmqx4’, or ‘ibmqx5’) if use_hardware is set to True. Default is ibmqx4.
      retrieve_execution (int): Job ID to retrieve instead of re-running the circuit (e.g., if previous run timed out).
    """

    short_name = 'projectq.ibmbackend'
    _gates = set([ key for (key,val) in operator_map.items() if val in [HGate, XGate, YGate, ZGate, SGate, TGate, SqrtXGate, SwapGate, Rx, Ry, Rz, R, CNOT, CZ] ])
    _observables = set([ key for (key,val) in operator_map.items() if val in [ZGate, AllZGate] ])
    _circuits = {}
    _backend_kwargs = ['use_hardware', 'num_runs', 'verbose', 'user', 'password', 'device', 'retrieve_execution']

    def __init__(self, wires, **kwargs):
        # check that necessary arguments are given
        if 'user' not in kwargs:
            raise ValueError('An IBM Quantum Experience user name specified via the "user" keyword argument is required')
        if 'password' not in kwargs:
            raise ValueError('An IBM Quantum Experience password specified via the "password" keyword argument is required')

        import projectq.setups.ibm

        kwargs['backend'] = 'IBMBackend'
        super().__init__(wires, **kwargs)

    def reset(self):
        """Resets the engine and backend

        After the reset the Device should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        backend = pq.backends.IBMBackend(**self.filter_kwargs_for_backend(self.kwargs))
        self.eng = pq.MainEngine(backend, engine_list=pq.setups.ibm.get_engine_list())
        super().reset()

    def expectation(self, observable, wires):
        pq.ops.All(pq.ops.Measure) | self.reg
        self.eng.flush()

        ev_list = [] # list of returned expectation values
        for expectation in self._observe:
            if expectation.name == 'PauliZ':
                probabilities = self.eng.backend.get_probabilities([self.reg[wires]])
                #print("IBM probabilities="+str(probabilities))
                if '1' in probabilities:
                    ev = 2*probabilities['1']-1
                else:
                    ev = -(2*probabilities['0']-1)
                variance = 1 - ev**2
            elif expectation.name == 'AllPauliZ':
                probabilities = self.eng.backend.get_probabilities(self.reg)
                #print("IBM all probabilities="+str(probabilities))
                ev = [ ((2*sum(p for (state,p) in probabilities.items() if state[i] == '1')-1)-(2*sum(p for (state,p) in probabilities.items() if state[i] == '0')-1)) for i in range(len(self.reg)) ]
                variance = [1 - e**2 for e in ev]
            else:
                raise DeviceError("Observable {} not supported by {}".format(self._observe.name, self.name))

            ev_list.append(ex)

        return np.array(ev_list, dtype=np.float64)
