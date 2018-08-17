# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Strawberry Fields plugin
========================

**Module name:** :mod:`openqml.plugins.strawberryfields`

.. currentmodule:: openqml.plugins.strawberryfields

This plugin provides the interface between OpenQML and Strawberry Fields.
It enables OpenQML to optimize continuous variable quantum circuits.

Strawberry Fields supports several different backends for executing quantum circuits.
The default is a NumPy-based Fock basis simulator, but also a TensorFlow-based Fock basis simulator and a Gaussian simulator are available.
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

import strawberryfields as sf
import strawberryfields.ops as sfo
import strawberryfields.engine as sfe


# tolerance for numerical errors
tolerance = 1e-10

#========================================================
#  define the gate set
#========================================================

class Gate(GateSpec):
    """Implements the quantum gates and observables.
    """
    def __init__(self, name, n_sys, n_par, cls=None, grad_recipe=None, *, par_domain='R'):
        # if a recipe is provided, use the angular method
        grad_method = 'F' if grad_recipe is None else 'A'
        super().__init__(name, n_sys, n_par, par_domain=par_domain, grad_method=grad_method, grad_recipe=grad_recipe)
        self.cls = cls  #: class: sf.ops.Operation subclass corresponding to the gate

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
    def execute(self, par, reg, sim):
        """Estimates the expectation value of the observable in the current system state.

        The arguments are the same as for :meth:`Gate.execute`.
        """
        if self.n_sys != 1:
            raise ValueError('This plugin supports only one-mode observables.')

        #A = self.cls(*par)  # Operation instance
        # run the queued program so that we obtain the state before the measurement
        # FIXME this only works for simulator backends, not hardware!
        state = sim.eng.run(**sim.init_kwargs)  # FIXME remove **kwargs here when SF is updated
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
Vac  = Gate('Vac', 1, 0, sfo.Vacuum)
Coh  = Gate('Coh', 1, 2, sfo.Coherent)
Squ  = Gate('Squ', 1, 2, sfo.Squeezed)
The  = Gate('The', 1, 1, sfo.Thermal)
Fock = Gate('Fock', 1, 1, sfo.Fock, par_domain='N')

# TODO More gradient formulas! At least all gaussian gates should have one.

Ds = 1.0  # gradient computation shift for displacements
D = Gate('D', 1, 2, sfo.Dgate, [(0.5/Ds, Ds), None])  # TODO d\tilde{D}(r, phi)/dr does not depend on r! The gradient formula can be simplified further, we can make do with smaller displacements!
X = Gate('X', 1, 1, sfo.Xgate, [(0.5/Ds, Ds)])
Z = Gate('Z', 1, 1, sfo.Zgate, [(0.5/Ds, Ds)])
Ss = 1.0  # gradient computation shift for squeezing
S = Gate('S', 1, 2, sfo.Sgate, [(0.5/np.sinh(Ss), Ss), None])
R = Gate('R', 1, 1, sfo.Rgate, [None])
F = Gate('Fourier', 1, 0, sfo.Fouriergate)
P = Gate('P', 1, 1, sfo.Pgate)
V = Gate('V', 1, 1, sfo.Vgate)
K = Gate('K', 1, 1, sfo.Kgate)
BS = Gate('BS', 2, 2, sfo.BSgate, [None, None])  # both parameters are rotation-like
S2 = Gate('S2', 2, 2, sfo.S2gate)
CX = Gate('CX', 2, 1, sfo.CXgate)
CZ = Gate('CZ', 2, 1, sfo.CZgate)

# measurements
MFock = Observable('MFock', 1, 0, sfo.MeasureFock)
MHo   = Observable('MHomodyne', 1, 1, sfo.MeasureHomodyne)
#MX    = Observable('MX', 1, 0, sfo.MeasureX)
#MP    = Observable('MP', 1, 0, sfo.MeasureP)
MHe   = Observable('MHeterodyne', 1, 0, sfo.MeasureHeterodyne)


demo = [
    Command(D,  [0], [ParRef(0), ParRef(1)]),
    Command(S,  [0], [ParRef(2), ParRef(3)]),
    Command(BS, [0, 1], [np.pi/4, 0]),
    Command(X,  [0], [ParRef(4)]),
    Command(Z,  [1], [ParRef(5)]),
    Command(D,  [0], [0.8, 0.7]),
    Command(D,  [1], [0.5, -1.2]),
    Command(BS, [0, 1], [ParRef(6), ParRef(7)]),
    Command(R,  [0], [ParRef(8)]),
    Command(BS, [0, 1], [-np.pi/4, 0]),
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
        n = circuit.n_sys
        if self.eng is None:
            self.eng = sfe.Engine(num_subsystems=n, hbar=2)  # FIXME add **self.init_kwargs here when SF is updated, remove hbar=2
        elif self.eng.num_subsystems != n:  # FIXME change to init_num_subsystems when SF is updated to next version
            raise ValueError("Trying to execute a {}-mode circuit '{}' on a {}-mode state.".format(n, circuit.name, self.eng.num_subsystems))

        # input the program
        reg = self.eng.register
        with self.eng:
            for cmd in circuit.seq:
                # prepare the parameters
                par = ParRef.map(cmd.par, params)
                # execute the gate
                cmd.gate.execute(par, cmd.reg, self)

        self.eng.run(**self.init_kwargs)  # FIXME remove **kwargs here when SF is updated

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
    # find out which SF backends are available
    temp = list(sf.backends.supported_backends.keys())
    temp.remove('base')  # HACK
    PluginAPI._capabilities['backend'] = sorted(temp)

    return PluginAPI
