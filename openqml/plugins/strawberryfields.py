# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Strawberry Fields plugin for OpenQML
====================================

**Module name:** :mod:`openqml.plugins.strawberryfields`

.. currentmodule:: openqml.plugins.strawberryfields

This plugin provides the interface between OpenQML and Strawberry Fields.
It enables OpenQML to optimize continuous variable quantum circuits.

Strawberry Fields supports several different backends for executing quantum circuits.
The default is a NumPy-based Fock basis simulator, but also a TensorFlow-based Fock basis simulator and a Gaussian simulator are available.
See PluginAPI._capabilities['backend'] for a list of backend options.

"""

import warnings

import numpy as np

import openqml.plugin
from openqml.circuit import (GateSpec, Command, ParRef, Circuit)

import strawberryfields as sf
import strawberryfields.ops as sfo
import strawberryfields.engine as sfe


# tolerance for numerical errors
tolerance = 1e-10

#========================================================
#  define the gate set
#========================================================

class Gate(GateSpec):
    """Implements the quantum gates and measurements.
    """
    def __init__(self, name, n_sys, n_par, cls=None):
        super().__init__(name, n_sys, n_par)
        self.cls = cls  #: class: sf.ops.Operation subclass corresponding to the gate

# gates (and state preparations)
Vac  = Gate('Vac', 1, 0, sfo.Vacuum)
Coh  = Gate('Coh', 1, 2, sfo.Coherent)
Squ  = Gate('Squ', 1, 2, sfo.Squeezed)
The  = Gate('The', 1, 1, sfo.Thermal)
Fock = Gate('Fock', 1, 1, sfo.Fock)
D = Gate('D', 1, 2, sfo.Dgate)
S = Gate('S', 1, 2, sfo.Sgate)
X = Gate('X', 1, 1, sfo.Xgate)
Z = Gate('Z', 1, 1, sfo.Zgate)
R = Gate('R', 1, 1, sfo.Rgate)
F = Gate('Fourier', 1, 0, sfo.Fouriergate)
P = Gate('P', 1, 1, sfo.Pgate)
V = Gate('V', 1, 1, sfo.Vgate)
K = Gate('K', 1, 1, sfo.Kgate)
BS = Gate('BS', 2, 2, sfo.BSgate)
S2 = Gate('S2', 2, 2, sfo.S2gate)
CX = Gate('CX', 2, 1, sfo.CXgate)
CZ = Gate('CZ', 2, 1, sfo.CZgate)

# measurements
MFock = Gate('MFock', 1, 0, sfo.MeasureFock)
MHo   = Gate('MHomodyne', 1, 1, sfo.MeasureHomodyne)
#MX    = Gate('MX', 1, 0, sfo.MeasureX)
#MP    = Gate('MP', 1, 0, sfo.MeasureP)
MHe   = Gate('MHeterodyne', 1, 0, sfo.MeasureHeterodyne)


demo = [
    Command(S,  [0], [ParRef(0), 0]),
    Command(BS, [0, 1], [np.pi/4, 0]),
    Command(R,  [0], [np.pi/3]),
    Command(D,  [1], [ParRef(1), np.pi/3]),
    Command(BS, [0, 1], [-np.pi/4, 0]),
    Command(R,  [0], [ParRef(1)]),
    Command(BS, [0, 1], [np.pi/4, 0]),
]

# circuit templates
_circuit_list = [
  Circuit(demo, 'demo'),
  Circuit(demo +[Command(MHo, [0], [0])], 'demo_ev', out=[0]),
]



class PluginAPI(openqml.plugin.PluginAPI):
    """Strawberry Fields OpenQML plugin API class.

    Keyword Args:
      backend (str): backend name
      cutoff_dim (int): Hilbert space truncation dimension for Fock basis backends
    """
    plugin_name = 'Strawberry Fields OpenQML plugin'
    plugin_api_version = '0.1.0'
    plugin_version = sf.version()
    author = 'Xanadu Inc.'
    _circuits = {c.name: c for c in _circuit_list}

    def __init__(self, name='default', **kwargs):
        super().__init__(name, **kwargs)

        # sensible defaults
        kwargs.setdefault('backend', 'fock')

        # backend-specific capabilities
        temp = kwargs['backend']
        self.backend = temp  #: str: backend name
        # gate and observable sets depend on the backend, so they have to be instance properties
        gates = [Vac, Coh, Squ, The, D, S, X, Z, R, F, P, BS, S2, CX, CZ]
        observables = [MHo]
        if temp in ('fock', 'tf'):
            kwargs.setdefault('cutoff_dim', 5)  # Fock space truncation dimension
            observables.append(MFock)
            gates.extend([Fock, V, K])  # nongaussian gates: Fock state prep, cubic phase and Kerr
        elif temp == 'gaussian':
            observables.append(MHe)  # TODO move to observables when the Fock basis backends support heterodyning
        else:
            raise ValueError("Unknown backend '{}'.".format(temp))
        self._gates = {g.name: g for g in gates}
        self._observables = {g.name: g for g in observables}

        self.init_kwargs = kwargs  #: dict: initialization arguments
        self.eng = None  #: strawberryfields.engine.Engine: engine for executing SF programs

    def __str__(self):
        return super().__str__() +'Backend: ' +self.backend +'\n'

    def reset(self):
        # reset the engine and backend
        if self.eng is not None:
            self.eng = None  # FIXME this is wasteful, now we construct a new Engine and backend after each reset (because the next circuit may have a different num_subsystems)
            #self.eng.reset()

    def measure(self, A, reg, par=[], n_eval=0):
        rr = self.eng.register
        G = A.cls(*par)  # construct the measurement operation
        with self.eng:
            # apply it
            G | rr[reg]
        self.eng.run(n_eval=n_eval)
        # measured value FIXME is not the EV!!!
        return rr[reg].val

    def execute_circuit(self, circuit, params=[], *, reset=True, **kwargs):
        super().execute_circuit(circuit, params, reset=reset, **kwargs)
        circuit = self.circuit

        # set the required number of subsystems
        n = circuit.n_sys
        if self.eng is None:
            self.eng = sfe.Engine(num_subsystems=n, hbar=2)  # FIXME add **self.init_kwargs here when SF is updated, remove hbar=2
        elif self.eng.num_subsystems != n:  # FIXME change to init_num_subsystems when SF is updated to next version
            raise ValueError("Trying to execute a {}-mode circuit '{}' on a {}-mode state.".format(n, circuit.name, self.eng.num_subsystems))

        def parmap(p):
            "Mapping function for gate parameters. Replaces ParRefs with the corresponding parameter values."
            if isinstance(p, ParRef):
                return params[p.idx]
            return p

        # input the program
        reg = self.eng.register
        with self.eng:
            for cmd in circuit.seq:
                # prepare the parameters
                par = map(parmap, cmd.par)
                # construct the gate
                G = cmd.gate.cls(*par)
                # apply it
                G | cmd.reg  #reg[cmd.reg]  # use numeric subsystem references for simplicity
        self.eng.run(**self.init_kwargs)  # FIXME remove **kwargs here when SF is updated

        if circuit.out is not None:
            # return the measurement results for the requested modes
            # measured value FIXME is not the EV!!!
            return np.array([reg[idx].val for idx in circuit.out])



def init_plugin():
    """Every plugin must define this function.

    It should perform whatever initializations are necessary, and then return an API class.

    Returns:
      class: plugin API class
    """
    # find out which SF backends are available
    temp = list(sf.backends.supported_backends.keys())
    temp.remove('base')  # HACK
    PluginAPI._capabilities['backend'] = temp

    return PluginAPI
