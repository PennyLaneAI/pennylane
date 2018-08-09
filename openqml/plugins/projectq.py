# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
ProjectQ plugin
========================

**Module name:** :mod:`openqml.plugins.projectq`

.. currentmodule:: openqml.plugins.projectq

This plugin provides the interface between OpenQML and ProjecQ.
It enables OpenQML to optimize quantum circuits simulable with ProjectQ.

ProjecQ supports several different backends. Of those the following are useful in the current context:

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
import warnings

import numpy as np

import openqml.plugin
from openqml.circuit import (GateSpec, Command, ParRef, Circuit)

import projectq as pq

# tolerance for numerical errors
tolerance = 1e-10

#========================================================
#  define the gate set
#========================================================

class Gate(GateSpec):
    """Implements the quantum gates and observables.
    """
    def __init__(self, name, n_sys, n_par, cls=None, par_domain='R'):
        super().__init__(name, n_sys, n_par, grad_method='F', par_domain=par_domain)
        self.cls = cls  #: class: pq subclass corresponding to the gate

    def execute(self, par, reg, sim):
        """Applies a single gate or measurement on the current system state.

        Args:
          par (Sequence[float]): gate parameters
          reg   (Sequence[int]): subsystems to which the gate is applied
          sim (~openqml.plugin.PluginAPI): simulator instance keeping track of the system state and measurement results
        """
        # construct the Operation instance
        G = self.cls(*par)
        # apply it
        G | reg


class Observable(Gate):
    """Implements hermitian observables.

    We assume that all the observables in the circuit are consequtive, and commute.
    Since we are only interested in the expectation values, there is no need to project the state after the measurement.
    See :ref:`measurements`.
    """
    #todo: Do we assume that all the observables in the circuit are consequtive, and commute?
    def execute(self, par, reg, sim):
        """Estimates the expectation value of the observable in the current system state.

        The arguments are the same as for :meth:`Gate.execute`.
        """
        if self.n_sys != 1:
            raise ValueError('This plugin supports only one-qubit observables.')

        state = sim.eng.run(**sim.init_kwargs)
        n_eval = sim.n_eval

        if self.cls == sfo.MeasureHomodyne:
            ev, var = state.quad_expectation(reg[0], *par)
        elif self.cls == sfo.MeasureFock:
            ev = state.mean_photon(reg[0])  # FIXME should return var too!
            var = 0
        else:
            warnings.warn('No expectation value method defined for {}.'.format(self.cls))
            ev = 0
            var = 0
        log.info('observable: ev: {}, var: {}'.format(ev, var))

        if n_eval != 0:
            # estimate the ev
            # TODO implement sampling in SF
            # use central limit theorem, sample normal distribution once, only ok if n_eval is large (see https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem)
            ev = np.random.normal(ev, np.sqrt(var / n_eval))

        sim.eng.register[reg[0]].val = ev  # TODO HACK: store the result (there should be a SF method for computing and storing the expectation value!)


# gates (and state preparations)
H = Gate('H', 1, 0, pq.ops.HGate)
X = Gate('X', 1, 0, pq.ops.XGate)
Y = Gate('Y', 1, 0, pq.ops.YGate)
Z = Gate('Z', 1, 0, pq.ops.ZGate)
S = Gate('S', 1, 0, pq.ops.SGate)
T = Gate('T', 1, 0, pq.ops.TGate)
SqrtX = Gate('SqrtX', 1, 0, pq.ops.SqrtXGate)
Swap = Gate('Swap', 2, 0, pq.ops.SwapGate)
SqrtSwap = Gate('SqrtSwap', 2, 0, pq.ops.SqrtSwapGate)
#Entangle = Gate('Entangle', n, 0, pq.ops.EntangleGate
Ph = Gate('Ph', 0, 1, pq.ops.Ph) #(angle) Phase gate (global phase)
Rx = Gate('Rx', 1, 1, pq.ops.Rx) #(angle) RotationX gate class
Ry = Gate('Ry', 1, 1, pq.ops.Ry) #(angle) RotationY gate class
Rz = Gate('Rz', 1, 1, pq.ops.Rz) #(angle) RotationZ gate class
R = Gate('R', 1, 1, pq.ops.R) #(angle) Phase-shift gate (equivalent to Rz up to a global phase)
#pq.ops.DaggeredGate) #(gate) Wrapper class allowing to execute the inverse of a gate, even when it does not define one.
#pq.ops.ControlledGate) #(gate[, n]) Controlled version of a gate.
#pq.ops.C) #(gate[, n]) Return n-controlled version of the provided gate.
#n, 0, pq.ops.AllGate #(instance of) pq.ops.Tensor
#n, 0, pq.ops.Tensor #(gate) Wrapper class allowing to apply a (single-qubit) gate to every qubit in a quantum register.
#pq.ops.QFTGate #(instance of) pq.ops.QFTGate
#pq.ops.QubitOperator) #([term, coefficient]) A sum of terms acting on qubits, e.g., 0.5 * ‘X0 X5’ + 0.3 * ‘Z1 Z2’.
CRz = Gate('CRz', 1, 1, pq.ops.CRz) #(angle) Shortcut for C(Rz(angle), n=1).
CNOT = Gate('CNOT', 2, 0, pq.ops.C(pq.ops.NOT)) #Controlled version of a gate.
CZ = Gate('CZ', 2, 0, pq.ops.C(pq.ops.ZGate)) #Controlled version of a gate.
#pq.ops.Toffoli) #Controlled version of a gate.
#n, 1, pq.ops.TimeEvolution) #(time, hamiltonian) Gate for time evolution under a Hamiltonian (QubitOperator object).


# measurements
Measure = Observable('MFock', 1, 0, pq.ops.Measure)


demo = [
    Command(Rx,  [0], [ParRef(0)]),
#    Command(CNOT, [0, 1], []),
]

# circuit templates
_circuit_list = [
  Circuit(demo, 'demo'),
  Circuit(demo +[Command(Measure, [0], [])], 'demo_ev', out=[1]),
]



class PluginAPI(openqml.plugin.PluginAPI):
    """ProjectQ OpenQML plugin API class.

    Keyword Args:
      backend (str): backend name
    """
    plugin_name = 'ProjectQ OpenQML plugin'
    plugin_api_version = '0.1.0'
    plugin_version = '0.1.0'
    author = 'Xanadu Inc.'
    _circuits = {c.name: c for c in _circuit_list}
    _capabilities = {'backend': list(["Simulator", "ClassicalSimulator", "IBMBackend"])}

    def __init__(self, name='default', **kwargs):
        super().__init__(name, **kwargs)

        # sensible defaults
        kwargs.setdefault('backend', 'Simulator')

        # backend-specific capabilities
        self.backend = kwargs['backend']
        # gate and observable sets depend on the backend, so they have to be instance properties
        #gates = [H, X, Y, Z, S, T, SqrtX, Swap, SqrtSwap, Ph, Rx, Ry, Rz, R, CRz, CNOT]
        gates = [H, X, Y, Z, S, T]
        observables = [Measure]
        if self.backend == 'Simulator':
            pass
        elif self.backend == 'ClassicalSimulator':
            gates = [X, Z, CNOT]
        elif self.backend == 'IBMBackend':
            ibm_backend = projectq.backends.IBMBackend()
            gates = [gate for gate in gates if ibm_backend.is_available(gate.cls)]
        else:
            raise ValueError("Unknown backend '{}'.".format(self.backend))

        self._gates = {g.name: g for g in gates}
        self._observables = {g.name: g for g in observables}

        self.init_kwargs = kwargs  #: dict: initialization arguments
        self.eng = None

    def __str__(self):
        return super().__str__() +'ProjecQ with Backend: ' +self.backend +'\n'

    def reset(self):
        """Resets the engine and backend"""
        if self.eng is not None:
            self.eng = None  #todo: this is wasteful, now we construct a new Engine and backend after each reset (because the next circuit may have a different num_subsystems)
            #self.eng.reset()

    def measure(self, A, reg, par=[], n_eval=0):
        temp = self.n_eval  # store the original
        self.n_eval = n_eval
        with self.eng:
            A.execute(par, [reg], self)  # compute the expectation value
        self.n_eval = temp  # restore it
        return self.eng.register[reg].val

    def execute_circuit(self, circuit, params=[], *, reset=True, **kwargs):
        super().execute_circuit(circuit, params, reset=reset, **kwargs)
        circuit = self.circuit

        # set the required number of subsystems
        if self.eng is None:
            if self.backend == 'Simulator':
                backend = pq.backends.Simulator()
            elif self.backend == 'ClassicalSimulator':
                backend = pq.backends.ClassicalSimulator()
            elif self.backend == 'IBMBackend':
                backend = pq.backends.IBMBackend()
        self.eng = pq.MainEngine(backend)

        def parmap(p):
            "Mapping function for gate parameters. Replaces ParRefs with the corresponding parameter values."
            if isinstance(p, ParRef):
                return params[p.idx]
            return p

        # input the program
        reg = self.eng.allocate_qureg(circuit.n_sys)
        for cmd in circuit.seq:
            # prepare the parameters
            par = map(parmap, cmd.par)
            # execute the gate
            print("Gate="+cmd.gate.name)
            print("cmd.reg="+str(cmd.reg))
            print("reg="+str(reg))
            cmd.gate.execute(par, [reg[i] for i in cmd.reg], self)  #MUST construct a projctQ register here instead form cmd.reg

        pq.ops.Measure | reg # avoid am unfriendly error message by ProjectQ: https://github.com/ProjectQ-Framework/ProjectQ/issues/2
        self.eng.flush()

        if circuit.out is not None:
            # return the estimated expectation values for the requested modes
            return np.array([reg[idx].val for idx in circuit.out])



def init_plugin():
    """Initialize the plugin.

    Every plugin must define this function.
    It should perform whatever initializations are necessary, and then return an API class.

    Returns:
      class: plugin API class
    """

    return PluginAPI
