# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Quantum circuits
================

**Module name:** :mod:`openqml.circuit`

.. currentmodule:: openqml.circuit

Quantum circuits, implemented by the :class:`Circuit` class, are abstract representations of the programs that quantum computers and simulators can execute.
In OpenQML they are typically encapsulated inside :class:`QNode` instances in the computational graph.
Each OpenQML plugin typically :meth:`provides <openqml.plugin.PluginAPI.templates>` a few ready-made parametrized circuit templates (variational quantum circuits)
that can be used in quantum machine learning tasks, but the users can also build their own circuits
out of the :class:`GateSpec` instances the plugin :meth:`supports <openqml.plugin.PluginAPI.gates>`.


Classes
-------

.. autosummary::
   GateSpec
   Command
   ParRef
   Circuit
   QNode


QNode methods
-------------

.. currentmodule:: openqml.circuit.QNode

.. autosummary::
   evaluate
   gradient

.. currentmodule:: openqml.circuit

----
"""

import autograd.numpy as np
import autograd.extend

import copy
import logging as log
import numbers



__all__ = ['GateSpec', 'Command', 'ParRef', 'Circuit', 'QNode']


class GateSpec:
    r"""A type of quantum operation supported by a backend, and its properies.

    GateSpec is used to describe both unitary quantum gates and measurements/observables.
    GateSpec instances are immutable, and normally there is just one for each type of gate.

    Args:
      name  (str): name of the operation
      n_sys (int): number of subsystems it acts on. Zero means the operation is not specific to any subset of subsystems.
      n_par (int): number of real parameters it takes
      par_domain  (str): domain of the gate parameters: 'N': natural numbers (incl. zero), 'R': floats. Parameters outside the domain are truncated into it.
      grad_method (str): gradient computation method: 'A': angular, 'F': finite differences
      grad_recipe (list[tuple[float]]): gradient recipe for the 'A' method. One tuple for each parameter: (multiplier c_k, parameter shift s_k). None means (0.5, \pi/2) (the most common case).

    .. math:: \frac{\partial Q(\ldots, \theta_k, \ldots)}{\partial \theta_k}} = c_k (Q(\ldots, \theta_k+s_k, \ldots) -Q(\ldots, \theta_k-s_k, \ldots))

    To find out in detail how the circuit gradients are computed, see :ref:`circuit_gradients`.
    """
    def __init__(self, name, n_sys=1, n_par=1, *, par_domain='R', grad_method='A', grad_recipe=None):
        self.name  = name   #: str: name of the gate
        self.n_sys = n_sys  #: int: number of subsystems it acts on
        self.n_par = n_par  #: int: number of real parameters it takes
        self.par_domain  = par_domain   #: str: domain of the gate parameters
        self.grad_method = grad_method  #: str: gradient computation method
        if grad_method == 'A' and grad_recipe is None:
            grad_recipe = [None] * n_par  # default recipe for every parameter
        if grad_recipe is not None and len(grad_recipe) != n_par:
            raise ValueError('Gradient recipe must have one entry for each parameter!')
        self.grad_recipe = grad_recipe

    def __str__(self):
        return self.name +': {} params, {} subsystems'.format(self.n_par, self.n_sys)


class Command:
    """Gate closure.

    Applying a given gate with given parameters on given subsystems.
    A quantum circuit can be described as a list of Commands.

    Args:
      gate (GateSpec): quantum operation to apply
      reg (Sequence[int]): Subsystems to which the operation is applied. Note that the order matters here.
      par (Sequence[float, int, ParRef]): parameter values, each either a fixed immediate value or a reference to a free parameter

    .. todo:: Use collections.OrderedDict to automatically avoid duplicate indices?
    """
    def __init__(self, gate, reg, par=[]):
        #if not isinstance(reg, Sequence):
        #    reg = [reg]
        if len(par) != gate.n_par:
            raise ValueError('Wrong number of parameters.')
        if len(reg) != gate.n_sys:
            raise ValueError('Wrong number of subsystems.')

        # convert fixed parameters into nonnegative integers if necessary,
        # it's up to the user to make sure the free parameters (ParRefs) have integer values when evaluating the circuit
        if gate.par_domain == 'N':
            def convert_par_to_N(p):
                if isinstance(p, ParRef):
                    return p
                if not isinstance(p, numbers.Integral):
                    p = int(p)
                    log.warning('Real parameter value truncated to int.')
                if p < 0:
                    p = 0
                    log.warning('Negative parameter value set to zero.')
                return p
            par = list(map(convert_par_to_N, par))

        self.gate = gate  #: GateSpec: quantum operation to apply
        self.par  = par   #: Sequence[float, int, ParRef]: parameter values
        self.reg  = reg   #: Sequence[int]: subsystems to which the operation is applied

    def __str__(self):
        return self.gate.name +'({}) | \t[{}]'.format(", ".join(map(str, self.par)), ", ".join(map(str, self.reg)))


class ParRef:
    """Parameter reference.

    Represents a free circuit parameter (with a non-fixed value).
    Each time the circuit is executed, it is given a vector of parameter values as input.
    ParRef is essentially an index into that vector, with a possible scalar multiplier.

    Args:
      idx (int): parameter index >= 0
    """
    def __init__(self, idx):
        self.idx  = idx  #: int: parameter index
        self.mult = 1.0  #: float: parameter scalar multiplier

    def __str__(self):
        temp = ' * {}'.format(self.mult) if self.mult != 1.0 else ''
        return 'ParRef: p{}'.format(self.idx) + temp

    def __neg__(self):
        """Unary negation."""
        temp = copy.copy(self)
        temp.mult = -temp.mult
        return temp

    def __mul__(self, scalar):
        """Right multiplication by scalars."""
        temp = copy.copy(self)
        temp.mult *= scalar
        return temp

    __rmul__ = __mul__ # """Left multiplication by scalars."""

    @staticmethod
    def map(par, par_free):
        """Mapping function for gate parameters. Replaces ParRefs with their actual values.

        Args:
          par (Sequence[float, int, ParRef]): parameter values to map, each either a fixed immediate value or a reference to a free parameter
          par_free    (Sequence[float, int]): values for the free parameters
        Returns:
          list[float, int]: mapped parameters
        """
        return [par_free[p.idx] * p.mult if isinstance(p, ParRef) else p for p in par]


class Circuit:
    """Quantum circuit.

    The quantum circuit is described in terms of a list of :class:`Command` instances.
    The Commands must not be used elsewhere, as they are mutable and are sometimes written into.

    .. note::

       The `out` argument reflects the way Strawberry Fields currently stores measurement results
       in a classical variable associated with the mode being measured. This approach does not work if one wishes to measure
       the same subsystem several times during the circuit and retain all the results.

    Args:
      seq (Sequence[Command]): sequence of quantum operations to apply to the state
      name (str): circuit name
      out (None, Sequence[int]): Subsystem indices from which the circuit output array is constructed.
        The command sequence should contain a measurement for each subsystem listed here.
        None means the circuit returns no value.
    """
    def __init__(self, seq, name='', out=None):
        self.seq  = list(seq)  #: list[Command]:
        self.name = name  #: str: circuit name
        self.pars = {}    #: dict[int->list[(Command, int)]]: map from free parameter index to the list of Commands (in this circuit!) that depend on it. The second element in the tuple is the index of the parameter within the Command.
        self.out = out    #: Sequence[int]: subsystem indices for circuit output

        # TODO check the validity of the circuit?
        # count the subsystems and parameter references used
        subsys = set()
        for cmd in self.seq:
            subsys.update(cmd.reg)
            for k, p in enumerate(cmd.par):
                if isinstance(p, ParRef):
                    self.pars.setdefault(p.idx, []).append((cmd, k))  # free parameter p.idx appears in cmd as gate parameter number k
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

        # determine the correct gradient computation method for each free parameter
        def best_method(par_dep_list):
            # use the angle method iff every gate that depends on the parameter supports it
            temp = [cmd.gate.grad_method == 'A' for cmd, _ in par_dep_list]
            return 'A' if all(temp) else 'F'
        self.grad_method = {k: best_method(v) for k, v in self.pars.items()}  #: dict[int->str]: map from free parameter index to the gradient method to be used with that parameter

    @property
    def n_par(self):
        """Number of free parameters in the circuit.

        Returns:
          int: number of free parameters
        """
        return len(self.pars)

    @property
    def n_out(self):
        """Circuit output array dimension.

        Returns:
          int: output array dimension
        """
        return len(self.out)

    @staticmethod
    def check_indices(inds, msg):
        """Check if the given indices form a gapless range starting from zero.

        Args:
          inds (set[int]): set of indices
          msg       (str): head of the possible error message

        Returns:
          bool: True if the indices are ok
        """
        if len(inds) == 0:
            return True
        ok = True
        if min(inds) < 0:
            log.warning(msg + 'negative indices')
            ok = False
        n_ind = max(inds) +1
        if n_ind > len(inds)+10:
            log.warning(msg + '> 10 unused indices')
            return False
        temp = set(range(n_ind))
        temp -= inds
        if len(temp) != 0:
            log.warning(msg + 'unused indices: {}'.format(temp))
            return False
        return ok

    def __str__(self):
        return "Quantum circuit '{}': len={}, n_sys={}, n_par={}".format(self.name, len(self), self.n_sys, self.n_par)

    def __len__(self):
        return len(self.seq)


class QNode:
    """Quantum node in the computational graph, encapsulating a circuit and a backend for executing it.

    Args:
      circuit (Circuit): quantum circuit representing the program
      backend (~openqml.plugin.PluginAPI): backend for executing the program
    """
    def __init__(self, circuit, backend):
        self.circuit = circuit  #: Circuit: quantum circuit representing the program
        self.backend = backend  #: PluginAPI: backend for executing the program

    @autograd.extend.primitive
    def evaluate(self, params, **kwargs):
        """Evaluate the node.

        .. todo:: rename to __call__?

        .. todo:: Should we delete the backend state after the call to save memory?

        Args:
          params (Sequence[float]): circuit parameters

        Returns:
          vector[float]: (approximate) expectation value(s) of the measured observable(s)

        The keyword arguments are passed on to :meth:`openqml.plugin.PluginAPI.execute_circuit`.
        """
        return self.backend.execute_circuit(self.circuit, params, **kwargs)


    def gradient(self, params, which=None, method='B', *, h=1e-7, order=1, **kwargs):
        """Compute the gradient (or Jacobian) of the node.

        Returns the gradient of the parametrized quantum circuit encapsulated in the QNode.

        The gradient can be computed using several methods:

        * finite differences ('F'). The first order method evaluates the circuit at n+1 points of the parameter space,
          the second order method at 2n points, where n = len(which).
        * angular method ('A'). Works for all one-parameter gates where the generator only has two unique eigenvalues, e.g. all one-qubit gates.
          Additionally can be used in CV systems for gaussian circuits containing only first-order observables.
          The circuit is evaluated twice for each incidence of each parameter in the circuit.
        * best known method for each parameter ('B'): uses the angular method if possible, otherwise finite differences.

        .. note::

           The finite difference method cannot tolerate any statistical noise in the circuit output, since it compares
           the output at two points infinitesimally close to each other. Hence the 'F' method requires exact expectation values,
           i.e. `n_eval=0`.

        Args:
          params (Sequence[float]): point in parameter space at which to evaluate the gradient
          which  (Sequence[int], None): return the gradient with respect to these parameters. None means all.
          method (str): gradient computation method, see above

        Keyword Args:
          h (float): finite difference method step size
          order (int): finite difference method order, 1 or 2
          n_eval (int): How many times should the circuit be evaluated (or sampled) to estimate the expectation values?
            For simulator backends, zero yields the exact result.

        Returns:
          array[float]: gradient vector/Jacobian matrix, shape == (n_out, len(which))
        """
        if which is None:
            which = range(len(params))
        elif len(which) != len(set(which)):  # set removes duplicates
            raise ValueError('Parameter indices must be unique.')
        params = np.asarray(params)

        if method in ('A', 'F'):
            method = {k: method for k in which}
        elif method == 'B':
            method = self.circuit.grad_method

        if 'F' in method.values():
            if order == 1:
                # the value of the circuit at params, computed only once here
                y0 = self.backend.execute_circuit(self.circuit, params, **kwargs)
            else:
                y0 = None

        # compute the partial derivative w.r.t. each parameter using the proper method
        #grad = np.zeros((self.circuit.n_out, len(which)), dtype=float)
        # FIXME autograd.grad does not play well with a Jacobian, see test_qnode.py
        grad = np.zeros((len(which),), dtype=float)
        for i, k in enumerate(which):
            temp = method[k]
            if temp == 'A':
                grad[i] = self._pd_angle(params, k, **kwargs)
            elif temp == 'F':
                grad[i] = self._pd_finite_diff(params, k, h, order, y0, **kwargs)
            else:
                raise ValueError('Unknown gradient method.')
        return grad


    def _pd_finite_diff(self, params, idx, h=1e-7, order=1, y0=None, **kwargs):
        """Partial derivative of the node using the finite difference method.

        Args:
          params (array[float]): point in parameter space at which to evaluate the partial derivative
          idx    (int): return the partial derivative with respect to this parameter
          h    (float): step size
          order  (int): finite difference method order, 1 or 2
          y0   (float): Value of the circuit at params. Should only be computed once.

        Returns:
          float: partial derivative of the node
        """
        if order == 1:
            # shift one parameter by h
            temp = params.copy()
            temp[idx] += h
            y = self.backend.execute_circuit(self.circuit, temp, **kwargs)
            return (y-y0) / h
        elif order == 2:
            # symmetric difference
            # shift one parameter by +-h/2
            temp = params.copy()
            temp[idx] += 0.5*h
            y2 = self.backend.execute_circuit(self.circuit, temp, **kwargs)
            temp[idx] = params[idx] -0.5*h
            y1 = self.backend.execute_circuit(self.circuit, temp, **kwargs)
            return (y2-y1) / h
        else:
            raise ValueError('Order must be 1 or 2.')


    def _pd_angle(self, params, idx, **kwargs):
        """Partial derivative of the node using the angle method.

        Args:
          params (array[float]): point in parameter space at which to evaluate the partial derivative
          idx    (int): return the partial derivative with respect to this parameter

        Returns:
          float: partial derivative of the node
        """
        n = self.circuit.n_par
        pd = 0.0
        # find the Commands in which the parameter appears, use the product rule
        for cmd, parnum in self.circuit.pars[idx]:
            if cmd.gate.grad_method != 'A':
                raise ValueError('Attempted to use the angular method on a gate that does not support it.')
            # we temporarily edit the Command such that parameter idx is replaced by a new one,
            # which we can modify without affecting other Commands that depend on the same parameter.
            orig = cmd.par[parnum]
            assert(orig.idx == idx)
            # reference to a new, temporary parameter with index n, otherwise identical with orig (this method only supports 1-parameter gates!)
            temp_par = copy.copy(orig)
            temp_par.idx = n
            cmd.par[parnum] = temp_par
            self.circuit.pars[n] = None  # we just need to add something to the map, it's not actually used

            # get the gradient recipe for this parameter
            recipe = cmd.gate.grad_recipe[parnum]
            multiplier = 0.5 if recipe is None else recipe[0]
            multiplier *= orig.mult
            # shift the temp parameter value by +- this amount
            shift = np.pi / 2 if recipe is None else recipe[1]
            shift /= orig.mult

            temp = np.r_[params, params[idx] +shift]
            y2 = self.backend.execute_circuit(self.circuit, temp, **kwargs)
            temp[-1] = params[idx] -shift
            y1 = self.backend.execute_circuit(self.circuit, temp, **kwargs)
            # restore the original parameter
            cmd.par[parnum] = orig
            del self.circuit.pars[n]  # remove the temporary entry
            pd += (y2-y1) * multiplier
        return pd


# define the vector-Jacobian product function for QNode.evaluate
autograd.extend.defvjp(QNode.evaluate, lambda ans, self, params: lambda g: g * self.gradient(params), argnums=[1])
