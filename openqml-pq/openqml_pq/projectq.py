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
    #:H, #todo: implement
    #:S, #todo: implement
    #:T, #todo: implement
    #:SqrtX, #todo: implement
    #:SqrtSwap, #todo: implement
    #:R, #todo: implement
    #:AllZGate, #todo: implement
    #'Hermitian': #todo: implement
}

class ProjectQDevice(Device):
    """ProjectQ device for OpenQML.

    Keyword Args:
      backend (str): backend name

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

    def __init__(self, wires, *, shots=0, **kwargs):

        if 'backend' in kwargs:
            del(kwargs['backend'])
        super().__init__(self.short_name, shots)

        # sensible defaults
        kwargs.setdefault('backend', 'Simulator')

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
        self.init_kwargs = kwargs
        self.eng = None
        self.reg = None

    def __repr__(self):
        return super().__repr__() +'Backend: ' +self.backend +'\n'

    def __str__(self):
        return super().__str__() +'Backend: ' +self.backend +'\n'

    def __del__(self):
        self.reset()

    def reset(self):
        """Resets the engine and backend"""
        if self.eng is not None:
            self._deallocate()
            self.eng = None

    def measure(self, observable, reg, par=[], n_eval=0):
        """ """
        return self.measurement_statistics(observable, reg, par, n_eval)

    def measurement_statistics(self, observable, reg, par=[], n_eval=0):
        """Compute the expection value.

        Returns the expectation value of the given observable in the given qubits.

        This method is only used during testing of the plugin.

        Args:
          observable (Observable): observable to compute the expectatoin value for
          reg (Sequence[int]): subsystems for which to do the computation
          par (Sequence[float]): parameters of the observable
          n_eval (int): number of samples from which to compute the expectation value
        """
        if n_eval != 0:
            log.warning("Non-zero value of n_eval ignored, as the IBMBackend does not support setting n_eval on the fly and all other backends yield exact expectation values.")

        if isinstance(reg, int):
            reg = [reg]

        temp = self.n_eval  # store the original
        self.n_eval = n_eval

        expectation_value, variance = observable.execute(par, [self.reg[i] for i in reg], self)
        self.n_eval = temp  # restore it
        return expectation_value, variance






    def execute_circuit(self, circuit, params=[], *, reset=True, **kwargs):
        super().execute_circuit(circuit, params, reset=reset, **kwargs)
        circuit = self.circuit

        def parmap(p):
            "Mapping function for gate parameters. Replaces ParRefs with the corresponding parameter values."
            if isinstance(p, ParRef):
                return params[p.idx]
            return p

        # set the required number of subsystems
        if self.eng is None or self.reg is None or self.circuit.n_sys != len(self.reg):
            self.reset()
            if self.backend == 'Simulator':
                backend = pq.backends.Simulator(**kwargs)
                self.eng = pq.MainEngine(backend)
            elif self.backend == 'ClassicalSimulator':
                backend = pq.backends.ClassicalSimulator()
                self.eng = pq.MainEngine(backend)
            elif self.backend == 'IBMBackend':
                backend = pq.backends.IBMBackend(**self.ibm_backend_kwargs)
                self.eng = pq.MainEngine(backend, engine_list=pq.setups.ibm.get_engine_list())

            self.reg = None

        # input the program
        if self.reg is None:
            self.reg = self.eng.allocate_qureg(circuit.n_sys)
        expectation_values = {}
        for cmd in circuit.seq:
            # prepare the parameters
            par = map(parmap, cmd.par)
            if cmd.gate.name not in self._gates and cmd.gate.name not in self._observables:
                log.warning("The circuit {} contains the gate {}, which is not supported by the {} backend. Abortig execution of this circuit.".format(circuit, cmd.gate.name, self.backend))
                break
            # execute the gate
            expectation_values[tuple(cmd.reg)] = cmd.gate.execute(par, [self.reg[i] for i in cmd.reg], self)

        #print('expectation_values='+str(expectation_values))
        if circuit.out is not None:
            # return the estimated expectation values for the requested modes
            return np.array([expectation_values[tuple([idx])] for idx in circuit.out if tuple([idx]) in expectation_values])







    def execute(self):
        """ """
        #todo: I hope this function will become superflous, see https://github.com/XanaduAI/openqml/issues/18
        self._out = self.execute_queued()

    def execute_queued(self):
        """Apply the queued operations to the device, and measure the expectation."""
        self.reg = self.eng.allocate_qureg(self.wires)

        expectation_values = {}



        return 1 #todo: handling of output should be better encapsulated from the plugin developers


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


    def requires_credentials(self):
        """Check whether this plugin requires credentials
        """
        if self.backend == 'IBMBackend':
            return True
        else:
            return False


class ProjectQSimulator(ProjectQDevice):
    """ProjectQ Simulator device for OpenQML.

    Keyword Args:
      backend (str): backend name

    Keyword Args:
      gate_fusion (bool): If True, gates are cached and only executed once a certain gate-size has been reached (only has an effect for the c++ simulator).
      rnd_seed (int): Random seed (uses random.randint(0, 4294967295) by default).
    """
    short_name = 'projectq.simulator'
    _gates = set(operator_map.keys())
    _observables = set([ key for (key,val) in operator_map.items() if val in [XGate, YGate, ZGate, AllZGate, Hermitian] ])
    _circuits = {}
    def __init__(self, **kwargs):
        kwargs['backend'] = 'Simulator'
        super().__init__(**kwargs)

        backend = pq.backends.Simulator(**kwargs)
        self.eng = pq.MainEngine(backend)


class ProjectQClassicalSimulator(ProjectQDevice):
    """ProjectQ ClassicalSimulator device for OpenQML.
    """
    short_name = 'projectq.classicalsimulator'
    _gates = set([ key for (key,val) in operator_map.items() if val in [XGate, CNOT] ])
    _observables = set([ key for (key,val) in operator_map.items() if val in [ZGate, AllZGate] ])
    _circuits = {}
    def __init__(self, **kwargs):
        kwargs.set('backend', 'ClassicalSimulator')
        super().__init__(**kwargs)

        backend = pq.backends.ClassicalSimulator()
        self.eng = pq.MainEngine(backend)


class ProjectQIBMBackend(ProjectQDevice):
    """ProjectQ IBMBackend device for OpenQML.

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
    def __init__(self, **kwargs):
        kwargs.set('backend', 'IBMBackend')
        super().__init__(**kwargs)

        backend = pq.backends.IBMBackend(**self.ibm_backend_kwargs)
        self.eng = pq.MainEngine(backend, engine_list=pq.setups.ibm.get_engine_list())
