# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Quantum circuits
================

**Module name:** :mod:`openqml.circuit`

.. currentmodule:: openqml.circuit


Classes
-------

.. autosummary::
   GateSpec
   Command
   ParRef
   Circuit
"""

import numpy as np

import logging as log
import warnings

#import numpy as np


class GateSpec:
    """Defines a single type of quantum gate supported by a backend, and its properies.

    Args:
      name  (str): name of the gate
      n_sys (int): number of subsystems it acts on
      n_par (int): number of real parameters it takes
      grad  (str): gradient computation method (generator, numeric?)
    """
    def __init__(self, name, n_sys=1, n_par=1, grad=None):
        self.name  = name   #: str: name of the gate
        self.n_sys = n_sys  #: int: number of subsystems it acts on
        self.n_par = n_par  #: int: number of real parameters it takes
        self.grad  = grad   #: str: gradient computation method (generator, numeric?)

    def __str__(self):
        return self.name +': {} params, {} subsystems'.format(self.n_par, self.n_sys)


class Command:
    """Gate closure.

    Applying a given gate with given parameters on given subsystems.
    A quantum circuit can be described as a list of Commands.

    Args:
      gate (GateSpec): quantum operation to apply
      par (Sequence[float, ParRef]): parameter values
      reg (Sequence[int]): Subsystems to which the operation is applied. Note that the order matters here.
        TODO collections.OrderedDict to automatically avoid duplicate indices?
    """
    def __init__(self, gate, reg, par=[]):
        #if not isinstance(reg, Sequence):
        #    reg = [reg]
        if len(par) != gate.n_par:
            raise ValueError('Wrong number of parameters.')
        if len(reg) != gate.n_sys:
            raise ValueError('Wrong number of subsystems.')

        self.gate = gate  #: GateSpec: quantum operation to apply
        self.par  = par   #: Sequence[float, ParRef]: parameter values
        self.reg  = reg   #: Sequence[int]: subsystems to which the operation is applied

    def __str__(self):
        return self.gate.name +'({}) | \t[{}]'.format(", ".join(map(str, self.par)), ", ".join(map(str, self.reg)))


class ParRef:
    """Parameter reference.

    Represents a circuit parameter with a non-fixed value.
    Each time the circuit is executed, it is given a vector of parameter values. ParRef is essentially an index into that vector.

    Args:
      idx (int): parameter index >= 0
    """
    def __init__(self, idx):
        self.idx = idx  #: int: parameter index

    def __str__(self):
        return 'ParRef: {}'.format(self.idx)


class Circuit:
    """Quantum circuit.

    The quantum circuit is described in terms of a list of :class:`Command` s.
    The Commands must not be used elsewhere, as they are mutable and are sometimes written into.

    Args:
      seq (Sequence[Command]): sequence of quantum operations to apply to the state
      name (str): circuit name
    """
    def __init__(self, seq, name='', obs=None):
        self.seq  = list(seq)  #: list[Command]:
        self.name = name  #: str: circuit name
        self.pars = {}    #: dict[int->list[Command]]: map from non-fixed parameter index to the list of Commands (in this circuit!) that depend on it
        self.obs = obs    #: Command: observable HACK FIXME

        # TODO check the validity of the circuit?
        # count the subsystems and parameter references used
        subsys = set()
        for cmd in self.seq:
            subsys.update(cmd.reg)
            for p in cmd.par:
                if isinstance(p, ParRef):
                    self.pars.setdefault(p.idx, []).append(cmd)
        self.n_sys = len(subsys)  #: int: number of subsystems

        msg = "Circuit '{}': ".format(self.name)
        # remap the subsystem indices to a continuous range 0..n_sys-1
        if not self.check_indices(subsys, msg+'subsystems: '):
            # we treat the subsystem indices as abstract labels, but preserve their relative order nevertheless in case the architecture benefits from it
            m = dict(zip(sorted(subsys), range(len(subsys))))
            for cmd in self.seq:
                cmd.reg = [m[s] for s in cmd.reg]
            log.info(msg +'subsystem indices remapped.')

        # parameter indices must not contain gaps
        if not self.check_indices(self.pars.keys(), msg+'params: '):
            raise ValueError(msg +'parameter indices ambiguous.')


    @property
    def n_par(self):
        """Number of non-fixed parameters used in the circuit.

        Returns:
          int: number of non-fixed parameters
        """
        return len(self.pars)

    @staticmethod
    def check_indices(inds, msg):
        """Check if the given indices form a gapless range starting from zero.

        Args:
          inds (set[int]): set of indices

        Returns:
          bool: True if the indices are ok
        """
        if len(inds) == 0:
            return True
        ok = True
        if min(inds) < 0:
            warnings.warn(msg + 'negative indices')
            ok = False
        n_ind = max(inds) +1
        if n_ind > len(inds)+10:
            warnings.warn(msg + '> 10 unused indices')
            return False
        temp = set(range(n_ind))
        temp -= inds
        if len(temp) != 0:
            warnings.warn(msg + 'unused indices: {}'.format(temp))
            return False
        return ok

    def __str__(self):
        return "Quantum circuit '{}': len={}, n_sys={}, n_par={}".format(self.name, len(self), self.n_sys, self.n_par)

    def __len__(self):
        return len(self.seq)


class QNode:
    """Quantum node in the computational graph.

    Each quantum node is defined by a :class:`Circuit` instance representing the quantum program, and
    a :class:`PluginAPI` instance representing the backend to execute it on.
    """
    def __init__(self, circuit, backend):
        self.circuit = circuit  #: Circuit: quantum circuit representing the program
        self.backend = backend  #: PluginAPI: backend for executing the program

    def evaluate(self, params, **kwargs):
        """Evaluate the node
        """
        return self.backend.execute_circuit(self.circuit, params, **kwargs)


    def gradient_finite_diff(self, params, h=1e-7, **kwargs):
        """Compute the gradient of the node using finite differences.

        Given an n-parameter quantum circuit, this function computes its gradient with respect to the parameters
        using the finite difference method. The current implementation evaluates the circuit at n+1 points of the parameter space.

        Args:
          params (Sequence[float]): point in parameter space at which to evaluate the gradient
          h (float): step size
        Returns:
          array: gradient vector
        """
        params = np.asarray(params)
        grad = np.zeros(params.shape)
        # value at the evaluation point
        x0 = self.backend.execute_circuit(self.circuit, params, **kwargs)
        for k in range(len(params)):
            # shift the k:th parameter by h
            temp = params.copy()
            temp[k] += h
            x = self.backend.execute_circuit(self.circuit, temp, **kwargs)
            grad[k] = (x-x0) / h
        return grad


    def gradient_angle(self, params, **kwargs):
        """Compute the gradient of the node using the angle method.

        Given an n-parameter quantum circuit, this function computes its gradient with respect to the parameters
        using the angle method. The method only works for one-parameter gates where the parameter is the rotation angle,
        and the generator eigenvalues are all :math:`\pm 1/2`. TODO r?
        The circuit is evaluated twice for each incidence of each parameter in the circuit.

        Args:
          params (Sequence[float]): point in parameter space at which to evaluate the gradient
        Returns:
          array: gradient vector
        """
        params = np.asarray(params)
        grad = np.zeros(params.shape)
        n = self.circuit.n_par
        for k in range(n):
            # find the Commands in which the parameter appears, use the product rule
            for cmd in self.circuit.pars[k]:
                if cmd.gate.n_par != 1:
                    raise ValueError('For now we can only differentiate one-parameter gates.')
                # we temporarily edit the Command so that parameter k is replaced by a new one,
                # which we can modify without affecting other Commands depending on the original.
                orig = cmd.par[0]
                assert(orig.idx == k)
                cmd.par[0] = ParRef(n)  # reference to a new, temporary parameter
                self.circuit.pars[n] = None  # we just need to add something to the map, it's not actually used
                # shift it by pi/2 and -pi/2
                temp = np.r_[params, params[k]+np.pi/2]
                x2 = self.backend.execute_circuit(self.circuit, temp, **kwargs)
                temp[-1] = params[k] -np.pi/2
                x1 = self.backend.execute_circuit(self.circuit, temp, **kwargs)
                # restore the original parameter
                cmd.par[0] = orig
                del self.circuit.pars[n]
                grad[k] += (x2-x1) / 2
        return grad
