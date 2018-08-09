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
        reg = tuple(reg) #I get an index out of bounds exceptoin if reg is a list of qbits and don't understand why. Therefore I convert to tuple.
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

        if type(sim.eng.backend).__name__ == 'Simulator':
            sim.eng.flush()
            if self.cls == pq.ops.X or self.cls == pq.ops.Y or self.cls == pq.ops.Z:
                ev = sim.eng.backend.get_expectation_value(pq.ops.QubitOperator(str(str(self.cls)+'0')), reg)
            else:
                raise NotImplementedError("Estimation of expectation values not yet implemented the observable {}.".format(self.cls))
        elif type(sim.eng.backend).__name__ == 'ClassicalSimulator':
            ev = sim.eng.backend.read_bit(reg[0])
        else:
            raise NotImplementedError("Estimation of expectation values not yet implemented for the {} backend.".format(self.backend))

        var = 0

        log.info('observable: ev: {}, var: {}'.format(ev, var))

        #sim.eng.register[reg[0]].val = ev  # TODO HACK: In the SF plugin the expecation value is stored in the register of the simulator and it is argued there that SF should have a method to store expectation values. I am not sure I agree with that. I think storage of the expectation values should be done in the openqml plugin.
        return ev

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
#Ph = Gate('Ph', 0, 1, pq.ops.Ph) #(angle) Phase gate (global phase)
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
CNOT = Gate('CNOT', 2, 0, pq.ops.CNOT) #Controlled version of a gate.
CZ = Gate('CZ', 2, 0, pq.ops.C(pq.ops.ZGate)) #Controlled version of a gate.
#pq.ops.Toffoli) #Controlled version of a gate.
#n, 1, pq.ops.TimeEvolution) #(time, hamiltonian) Gate for time evolution under a Hamiltonian (QubitOperator object).


# measurements
MeasureX = Observable('X', 1, 0, pq.ops.X)
MeasureY = Observable('Y', 1, 0, pq.ops.Y)
MeasureZ = Observable('Z', 1, 0, pq.ops.Z)


demo = [
    Command(Rx,  [0], [ParRef(0)]),
    Command(Rx,  [1], [ParRef(1)]),
#    Command(CNOT, [0, 1], []),
]

# circuit templates
_circuit_list = [
  Circuit(demo, 'demo'),
  Circuit(demo +[Command(MeasureZ, [0])], 'demo_ev', out=[0]),
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
    #_capabilities = {'backend': list(["Simulator", "ClassicalSimulator", "IBMBackend"])}
    _capabilities = {'backend': list(["Simulator"])} #todo: the IBMBackend needs account data and can thus not be used during during unit tests - disabled for now, ClassicalSimulator produces a "maximum recursion depth exceeded" exceeded error - disabled for now as couldn't get it to work quickly and of limited use

    def __init__(self, name='default', **kwargs):
        super().__init__(name, **kwargs)

        # sensible defaults
        kwargs.setdefault('backend', 'Simulator')

        # backend-specific capabilities
        self.backend = kwargs['backend']
        # gate and observable sets depend on the backend, so they have to be instance properties
        #gates = [H, X, Y, Z, S, T, SqrtX, Swap, SqrtSwap, Ph, Rx, Ry, Rz, R, CRz, CNOT]
        gates = [H, X, Y, Z, S, T, SqrtX, Swap, SqrtSwap, Rx, Ry, Rz]
        observables = [MeasureX, MeasureY, MeasureZ]
        if self.backend == 'Simulator':
            pass
        elif self.backend == 'ClassicalSimulator':
            classical_backend = pq.backends.ClassicalSimulator()
            eng = pq.MainEngine(classical_backend)
            reg = eng.allocate_qureg(1)
            gates = [gate for gate in gates if classical_backend.is_available(pq.ops.Command(eng, gate.cls(), [reg]))]
            observables = [MeasureZ]
        elif self.backend == 'IBMBackend':
            ibm_backend = pq.backends.IBMBackend()
            eng = pq.MainEngine(ibm_backend)
            reg = eng.allocate_qureg(1)
            gates = [gate for gate in gates if ibm_backend.is_available(pq.ops.Command(eng, gate.cls(), [reg]))]
        else:
            raise ValueError("Unknown backend '{}'.".format(self.backend))

        self._gates = {g.name: g for g in gates}
        self._observables = {g.name: g for g in observables}

        self.init_kwargs = kwargs  #: dict: initialization arguments
        self.eng = None
        self.reg = None

    def __str__(self):
        return super().__str__() +'ProjecQ with Backend: ' +self.backend +'\n'

    def reset(self):
        """Resets the engine and backend"""
        if self.eng is not None:
            self.eng = None  #todo: this is wasteful, now we construct a new Engine and backend after each reset (because the next circuit may have a different num_subsystems)
            #self.eng.reset()

    def measure(self, A, reg, par=[], n_eval=0):
        if isinstance(reg, int):
            reg = [reg]

        temp = self.n_eval  # store the original
        self.n_eval = n_eval

        ev = A.execute(par, [self.reg[i] for i in reg], self)  # compute the expectation value
        self.n_eval = temp  # restore it
        return ev

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
        with pq.meta.Compute(self.eng):
            self.reg = self.eng.allocate_qureg(circuit.n_sys)
            expectation_values = {}
            for cmd in circuit.seq:
                # prepare the parameters
                par = map(parmap, cmd.par)
                # execute the gate
                expectation_values[tuple(cmd.reg)] = cmd.gate.execute(par, [self.reg[i] for i in cmd.reg], self)

        if self.backend == 'Simulator':
            pq.ops.All(pq.ops.Measure) | self.reg #avoid an unfriendly error message: https://github.com/ProjectQ-Framework/ProjectQ/issues/2

        if circuit.out is not None:
            # return the estimated expectation values for the requested modes
            return np.array([expectation_values[tuple([idx])] for idx in circuit.out])


def init_plugin():
    """Initialize the plugin.

    Every plugin must define this function.
    It should perform whatever initializations are necessary, and then return an API class.

    Returns:
      class: plugin API class
    """

    return PluginAPI
