# Copyright 2018 Xanadu Quantum Technologies Inc.
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
from openqml import Device, DeviceError
from openqml import Variable

from numpy.random import (randn,)

# import openqml.plugins
# from openqml.circuit import (GateSpec, Command, ParRef, Circuit)

import projectq as pq
import projectq.setups.ibm

from ._version import __version__


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
    _circuits = {c.name: c for c in _circuit_list}
    _capabilities = {'backend': list(["Simulator", "ClassicalSimulator", "IBMBackend"])}

    def __init__(self, name='default', **kwargs):
        super().__init__(name, **kwargs)

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

        # backend-specific capabilities
        self.backend = kwargs['backend']

        # gate and observable sets depend on the backend, so they (currently still) have to be instance properties
        gates = [H, X, Y, Z, S, T, SqrtX, Swap, SqrtSwap, Rx, Ry, Rz, R, CRz, CNOT, CZ]
        observables = [MeasureX, MeasureY, MeasureZ, MeasureAllZ]

        if self.backend == 'Simulator':
            pass
        elif self.backend == 'ClassicalSimulator':
            classical_backend = pq.backends.ClassicalSimulator()
            eng = pq.MainEngine(classical_backend)
            reg = eng.allocate_qureg(max([gate.n_sys for gate in gates]))
            gates = [gate for gate in gates if classical_backend.is_available(pq.ops.Command(eng, gate.cls(*randn(gate.n_par)), [[reg[i]] for i in range(0,gate.n_sys)]))]
            observables = [MeasureZ]
        elif self.backend == 'IBMBackend':
            import inspect
            self.ibm_backend_kwargs = {param:kwargs[param] for param in inspect.signature(pq.backends.IBMBackend).parameters if param in kwargs}
            # ibm_backend = pq.backends.IBMBackend(**self.ibm_backend_kwargs)
            # eng = pq.MainEngine(ibm_backend, engine_list=pq.setups.ibm.get_engine_list())
            # if True:
            #     reg = eng.allocate_qureg(max([gate.n_sys for gate in gates]))
            #     #gates = [gate for gate in gates if (ibm_backend.is_available(pq.ops.Command(eng, gate.cls(*randn(gate.n_par)), ([reg[i]] for i in range(0,gate.n_sys)))) or gate == CNOT)] #todo: do not treat CNOT as a special case one it is understood why ibm_backend.is_available() returns false for CNOT, see also here: https://github.com/ProjectQ-Framework/ProjectQ/issues/257

            #     # print('IBM supports the following gates: '+str([gate.name for gate in gates]))
            #     # print('ibm_backend.is_available(CNOTClass)='+str(ibm_backend.is_available(pq.ops.Command(eng, CNOTClass(), ([reg[0]], [reg[1]]) ))) )
            #     # print('ibm_backend.is_available(CNOT)='+str(ibm_backend.is_available(pq.ops.Command(eng, pq.ops.CNOT, ([reg[0]], [reg[1]]) ))) )
            #     # print('ibm_backend.is_available(X+control)='+str(ibm_backend.is_available(   pq.ops.Command(engine=eng, gate=pq.ops.X, qubits=([reg[0]],), controls=[reg[1]])    )) )
            #     # print('CNOTClass()==NOT: '+str(CNOTClass()==pq.ops.NOT))
            #     # print('C(NOT)==NOT: '+str(pq.ops.C(pq.ops.NOT)==pq.ops.NOT))
            #     # print('CNOT==NOT: '+str(pq.ops.CNOT==pq.ops.NOT))
            gates = [H, X, Y, Z, S, T, SqrtX, Swap, Rx, Ry, Rz, R, CRz, CNOT, CZ]
            observables = [MeasureZ,MeasureAllZ]
        else:
            raise ValueError("Unknown backend '{}'.".format(self.backend))

        self._gates = {g.name: g for g in gates}
        self._observables = {g.name: g for g in observables}

        self.init_kwargs = kwargs  #: dict: initialization arguments
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
                log.warning("The cirquit {} contains the gate {}, which is not supported by the {} backend. Abortig execution of this circuit.".format(circuit, cmd.gate.name, self.backend))
                break
            # execute the gate
            expectation_values[tuple(cmd.reg)] = cmd.gate.execute(par, [self.reg[i] for i in cmd.reg], self)

        #print('expectation_values='+str(expectation_values))
        if circuit.out is not None:
            # return the estimated expectation values for the requested modes
            return np.array([expectation_values[tuple([idx])] for idx in circuit.out if tuple([idx]) in expectation_values])

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
    def __init__(self, name='default', **kwargs):
        kwargs.set('backend', 'Simulator')
        super().__init__(name, **kwargs)


class ProjectQClassicalSimulator(ProjectQDevice):
    """ProjectQ ClassicalSimulator device for OpenQML.
    """
    def __init__(self, name='default', **kwargs):
        kwargs.set('backend', 'ClassicalSimulator')
        super().__init__(name, **kwargs)

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
    def __init__(self, name='default', **kwargs):
        kwargs.set('backend', 'IBMBackend')
        super().__init__(name, **kwargs)
