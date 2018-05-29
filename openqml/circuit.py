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
        return self.gate.name +'({}) | \t({})'.format(", ".join(self.par), ", ".join(self.reg))


class ParRef:
    """Parameter reference.

    Represents a circuit parameter with a non-fixed value.

    Args:
      idx (int): parameter index >= 0
    """
    def __init__(self, idx):
        self.idx = idx  #: int: parameter index

    def __str__(self):
        return 'ParRef: {}'.format(self.idx)


class Circuit:
    """Quantum circuit.

    Represents a list of Commands.

    Args:
      seq (Sequence[Command]): sequence of quantum operations to apply to the state
      name (str): circuit name
    """
    def __init__(self, seq, name=''):
        self.seq  = seq   #: Sequence[Command]:
        self.name = name  #: str: circuit name
        self.pars = {}    #: dict[int->list[Command]]: map from non-fixed parameter index to the list of Commands (in this circuit!) that depend on it

        # TODO check the validity of the circuit?
        # count the subsystems and parameter references used
        subsys = set()
        for c in seq:
            subsys.update(c.reg)
            for p in c.par:
                if isinstance(p, ParRef):
                    self.pars.setdefault(p.idx, []).append(c)
        self.check_indices(subsys)
        # TODO remap the subsystem indices to a continuous range 0..n_sys-1
        self.check_indices(self.pars.keys())
        self.n_sys = len(subsys)     #: int: number of subsystems

    @property
    def n_par(self):
        """Number of non-fixed parameters used in the circuit.

        Returns:
          int: number of non-fixed parameters
        """
        return len(self.pars)

    @staticmethod
    def check_indices(inds):
        """Check if the given indices form a continuous range.

        Args:
          inds (set[int]): set of indices
        """
        if min(inds) < 0:
            warnings.warn('Negative indices!')
        temp = set(range(max(inds)+1))
        temp -= inds
        if len(temp) != 0:
            warnings.warn('Unused indices: ', temp)

    def __str__(self):
        return "Quantum circuit '{}': len={}, n_sys={}, n_par={}".format(self.name, len(self), self.n_sys, self.n_par)

    def __len__(self):
        return len(self.seq)
