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
import projectq.setups.ibm #todo only import this if necessary

# import operations
from projectq.ops import (HGate, XGate, YGate, ZGate, SGate, TGate, SqrtXGate, SwapGate, SqrtSwapGate, Rx, Ry, Rz, R)
from .ops import (CNOT, CZ, Toffoli, AllZGate, Rot, Hermitian)

from ._version import __version__


operator_map = {
    'PauliX': XGate,
    'PauliY': YGate,
    'PauliZ': ZGate,
    'CNOT': CNOT,
    'CZ': CZ,
    'SWAP': SwapGate,
    'RX': Rx,
    'RY': Ry,
    'RZ': Rz,
    'Rot': Rot,
    #'PhaseShift': #todo: implement
    #'QubitStateVector': #todo: implement
    #'QubitUnitary': #todo: implement
    #: H, #todo: implement
    #: S, #todo: implement
    #: T, #todo: implement
    #: SqrtX, #todo: implement
    #: SqrtSwap, #todo: implement
    #: R, #todo: implement
    #'AllPauliZ': AllZGate, #todo: implement
    #'Hermitian': #todo: implement
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

    # def __del__(self):
    #     self._deallocate()

    def execute(self):
        """ """
        #todo: I hope this function will become superfluous, see https://github.com/XanaduAI/openqml/issues/18
        self._out = self.execute_queued()

    def execute_queued(self):
        """Apply the queued operations to the device, and measure the expectation."""
        #expectation_values = {}
        for operation in self._queue:
            if operation.name not in operator_map:
                raise DeviceError("{} not supported by device {}".format(operation.name, self.short_name))

            par = [x.val if isinstance(x, Variable) else x for x in operation.params]
            #expectation_values[tuple(operation.wires)] = self.apply(operator_map[operation.name](*p), self.reg, operation.wires)
            self.apply(operation.name, operation.wires, *par)

        result = self.expectation(self._observe.name, self._observe.wires)
        self._deallocate()
        return result

        # if self._observe.wires is not None:
        #     if isinstance(self._observe.wires, int):
        #         return expectation_values[tuple([self._observe.wires])]
        #     else:
        #         return np.array([expectation_values[tuple([idx])] for idx in self._observe.wires if tuple([idx]) in expectation_values])

    def apply(self, gate_name, wires, *par):
        if gate_name not in self._gates:
            raise ValueError('Gate {} not supported on this backend'.format(gate))

        gate = operator_map[gate_name](*par)
        if isinstance(wires, int):
            gate | self.reg[wires]
        else:
            gate | tuple([self.reg[i] for i in wires])

    def expectation(self, observable, wires):
        raise NotImplementedError("expectation() is not yet implemented for this backend")

    def shutdown(self):
        """Shutdown.

        """
        pass

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


    # def requires_credentials(self):
    #     """Check whether this plugin requires credentials
    #     """
    #     if self.backend == 'IBMBackend':
    #         return True
    #     else:
    #         return False


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
    _observables = set([ key for (key,val) in operator_map.items() if val in [XGate, YGate, ZGate, AllZGate, Hermitian] ])
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
        if observable == 'PauliX' or observable == 'PauliY' or observable == 'PauliZ':
            expectation_value = self.eng.backend.get_expectation_value(pq.ops.QubitOperator(str(observable)[-1]+'0'), self.reg)
            variance = 1 - expectation_value**2
        elif observable == 'AllPauliZ':
            expectation_value = [ self.eng.backend.get_expectation_value(pq.ops.QubitOperator("Z"+'0'), [qubit]) for qubit in self.reg]
            variance = [1 - e**2 for e in expectation_value]
        else:
            raise NotImplementedError("Estimation of expectation values not yet implemented for the observable {} in backend {}.".format(observable, self.backend))

        return expectation_value#, variance


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

        kwargs['backend'] = 'IBMBackend'
        #kwargs['verbose'] = True #todo: remove when done testing
        #kwargs['log'] = True #todo: remove when done testing
        #kwargs['use_hardware'] = False #todo: remove when done testing
        #kwargs['num_runs'] = 3 #todo: remove when done testing
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
        pq.ops.R(0) | self.reg[0]# todo:remove this once https://github.com/ProjectQ-Framework/ProjectQ/issues/259 is resolved

        pq.ops.All(pq.ops.Measure) | self.reg
        self.eng.flush()

        if observable == 'PauliZ':
            probabilities = self.eng.backend.get_probabilities([self.reg[wires]])
            #print("IBM probabilities="+str(probabilities))
            if '1' in probabilities:
                expectation_value = 2*probabilities['1']-1
            else:
                expectation_value = -(2*probabilities['0']-1)
            variance = 1 - expectation_value**2
        elif observable == 'AllPauliZ':
            probabilities = self.eng.backend.get_probabilities(self.reg)
            #print("IBM all probabilities="+str(probabilities))
            expectation_value = [ ((2*sum(p for (state,p) in probabilities.items() if state[i] == '1')-1)-(2*sum(p for (state,p) in probabilities.items() if state[i] == '0')-1)) for i in range(len(self.reg)) ]
            variance = [1 - e**2 for e in expectation_value]
        else:
            raise NotImplementedError("Estimation of expectation values not yet implemented for the observable {} in backend {}.".format(observable, self.backend))

        return expectation_value#, variance
