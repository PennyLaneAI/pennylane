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
"""
Symbolic quantum operations
===========================

**Module name:** :mod:`openqml.operation`

.. currentmodule:: openqml.operation


Classes
-------

.. autosummary::
   Operation
   Expectation
   CV
   CVOperation
   CVExpectation

----
"""
import abc
import numbers
import logging as log
log.getLogger()

from pkg_resources import iter_entry_points

import autograd.numpy as np

from .qnode import _flatten, _unflatten, QNode, QuantumFunctionError
from .variable import Variable


class Operation(abc.ABC):
    r"""Base class for quantum operations supported by a device.

    * Each Operation subclass represents a type of quantum operation,
      for example a unitary quantum gate.

    * Each instance of these subclasses represents an application of the
      operation with given parameter values to a given sequence of wires (subsystems).

    In general, an operation is differentiable (at least using the finite difference method 'F') iff

    * it has parameters, i.e. ``num_params > 0``, and
    * its parameter domain is continuous (i.e. ``par_domain`` is not ``'N'``).

    For an operation to be differentiable using the analytic method 'A', additionally:

    * its `par_domain` must be 'R'.

    .. note::

        These conditions are *not* sufficient for analytic differentiation. For example,
    This is not sufficient though, the 'A' method does not work on non-Gaussian CV gates for example.

    The gradient recipe (multiplier :math:`c_k`, parameter shift :math:`s_k`)
    works as follows:

    .. math::
       \frac{\partial Q(\ldots, \theta_k, \ldots)}{\partial \theta_k}
       = c_k \left(Q(\ldots, \theta_k+s_k, \ldots) -Q(\ldots, \theta_k-s_k, \ldots)\right)

    Args:
        args (tuple[float, int, array, Variable]): operation parameters.

    Keyword Args:
        wires (Sequence[int]): subsystems it acts on. If not given, args[-1]
            is interpreted as wires.
        do_queue (bool): Indicates whether the operation should be
            immediately pushed into a :class:`QNode` circuit queue.
            This flag is useful if there is some reason to run an Operation
            outside of a QNode context.
    """
    _grad_recipe = None

    @abc.abstractproperty
    def num_params(self):
        """Number of parameters the operation takes."""
        raise NotImplementedError

    @abc.abstractproperty
    def num_wires(self):
        """Number of wires the operation acts on.

        The value 0 allows the operation to act on any number of wires.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def num_params(self):
        """Number of parameters the operation takes."""
        raise NotImplementedError

    @abc.abstractproperty
    def par_domain(self):
        """Domain of the gate parameters.

        * ``'N'``: natural numbers (including zero).
        * ``'R'``: floats.
        * ``'A'``: arrays of real or complex values.
        * ``None``: if there are no parameters.
        """
        raise NotImplementedError

    @property
    def grad_method(self):
        """Gradient computation method.

        * ``'A'``: analytic differentiation.
        * ``'F'``: finite differences numerical differentiation.
        * ``None``: the operation may not be differentiated.

        Default is 'F'.
        """
        return 'F'

    @property
    def grad_recipe(self):
        r"""Gradient recipe for the analytic differentiation method.

        This is a list with one tuple per operation parameter. For parameter
        :math:`k`, the tuple is of the form :math:`(c_k, s_k)`, resulting in
        a gradient recipe of

        .. math:: \frac{d}{d\phi_k}O = c_k\left[O(\phi_k+s_k)-O(\phi_k-s_k)\right].

        If this property returns ``None``, the default gradient recipe
        :math:`(c_k, s_k)=(1/2, \pi/2)` is assumed for every parameter.
        """
        return self._grad_recipe

    @grad_recipe.setter
    def grad_recipe(self, value):
        """Setter for the grad_recipe property"""
        self._grad_recipe = value

    def __init__(self, *args, wires=None, do_queue=True):
        self.name  = self.__class__.__name__   #: str: name of the operation

        # extract the arguments
        if wires is not None:
            params = args
        else:
            params = args[:-1]
            wires = args[-1]

        if len(params) != self.num_params:
            raise ValueError("{}: wrong number of parameters. "
                             "{} parameters passed, {} expected.".format(self.name, params, self.num_params))

        # check the validity of the params
        for p in params:
            self.check_domain(p)
        self.params = list(params)

        # check the grad_method validity
        if self.par_domain == 'N':
            assert self.grad_method is None, 'An operation may only be differentiated with respect to real scalar parameters.'
        elif self.par_domain == 'A':
            assert self.grad_method in (None, 'F'), 'Operations that depend on arrays containing free variables may only be differentiated using the F method.'

        # check the grad_recipe validity
        if self.grad_method == 'A':
            if self.grad_recipe is None:
                # default recipe for every parameter
                self.grad_recipe = [None] * self.num_params
            else:
                assert len(self.grad_recipe) == self.num_params, 'Gradient recipe must have one entry for each parameter!'
        else:
            assert self.grad_recipe is None, 'Gradient recipe is only used by the A method!'

        # apply the operation on the given wires
        if isinstance(wires, int):
            wires = [wires]

        if self.num_wires != 0 and len(wires) != self.num_wires:
            raise ValueError("{}: wrong number of wires. "
                             "{} wires given, {} expected.".format(self.name, len(wires), self.num_wires))

        if len(set(wires)) != len(wires):
            raise ValueError('{}: wires must be unique, got {}.'.format(self.name, wires))

        self.wires = wires  #: Sequence[int]: subsystems the operation acts on
        if do_queue:
            self.queue()

    def __str__(self):
        """Print the operation name and some information."""
        return self.name +': {} params, wires {}'.format(len(self.params), self.wires)

    def check_domain(self, p, flattened=False):
        """Check the validity of a parameter.

        Args:
            p (Number, array, Variable): parameter to check
            flattened (bool): True means p is an element of a flattened parameter
                sequence (affects the handling of 'A' parameters)
        Raises:
            TypeError: parameter is not an element of the expected domain
        Returns:
            Number, array, Variable: p
        """
        try:
            if isinstance(p, Variable):
                if self.par_domain == 'A':
                    # NOTE: for now Variables can only represent real scalars.
                    raise TypeError('Free parameters must represent scalars, I need an array.')
                return p

            # p is not a Variable
            if self.par_domain == 'A':
                if flattened:
                    if isinstance(p, np.ndarray):
                        raise TypeError('Flattened array parameter expected, got {}.'.format(type(p)))
                else:
                    if not isinstance(p, np.ndarray):
                        raise TypeError('Array parameter expected, got {}.'.format(type(p)))
            elif self.par_domain in ('R', 'N'):
                if not isinstance(p, numbers.Real):
                    raise TypeError('Real scalar parameter expected, got {}.'.format(type(p)))

                if self.par_domain == 'N':
                    if not isinstance(p, numbers.Integral):
                        raise TypeError('Natural number parameter expected, got {}.'.format(type(p)))
                    if p < 0:
                        raise TypeError('Natural number parameter expected, got {}.'.format(p))
            else:
                raise TypeError('Unknown parameter domain \'{}\'.'.format(self.par_domain))
            return p
        except TypeError as exc:
            # add the name of the operation to the error message
            raise TypeError(self.name + ': ' +str(exc)) from None

    @property
    def parameters(self):
        """Current parameter values.

        Fixed parameters are returned as is, free parameters represented by :class:`~.variable.Variable` instances are replaced by their current numerical value.

        Returns:
          list[float]: parameter values
        """
        temp = list(_flatten(self.params))
        temp_val = [self.check_domain(x.val, True) if isinstance(x, Variable) else x for x in temp]
        return _unflatten(temp_val, self.params)[0]

    def queue(self):
        """Append the operation to a QNode queue."""
        if QNode._current_context is None:
            raise QuantumFunctionError("Quantum operations can only be used inside a qfunc.")
        else:
            QNode._current_context._append_op(self)
        return self  # so pre-constructed Expectation instances can be queued and returned in a single statement


class Expectation(Operation):
    """Base class for expectation value measurements supported by a device.

    :class:`Expectation` is used to describe Hermitian quantum observables.

    Args:
        args (tuple[float, int, array, Variable]): Expectation parameters

    Keyword Args:
        wires (Sequence[int]): subsystems it acts on.
            Currently, only one subsystem is supported.
        do_queue (bool): Indicates whether the operation should be immediately
            pushed into a :class:`QNode` circuit queue. This flag is useful if
            there is some reason to run an Expectation outside of a QNode context.
    """
    pass


class CV:
    """A mixin base class denoting a continuous-variable operation."""

    def heisenberg_expand(self, U, num_wires):
        """Expand the given local Heisenberg-picture array into a full-system one.

        Args:
            U (array[float]): array to expand (expected to be of the dimension 1+2*self.num_wires)
            num_wires  (int): total number of wires in the quantum circuit. If zero, return U as is.
        Returns:
            array[float]: expanded array, dimension 1+2*num_wires
        """
        U_dim = len(U)
        nw = len(self.wires)
        if U_dim != 1+2*nw:
            raise ValueError('{}: Heisenberg matrix is the wrong size {}.'.format(self.name, U_dim))

        if num_wires == 0 or list(self.wires) == list(range(num_wires)):
            # no expansion necessary (U is a full-system matrix in the correct order)
            return U

        if num_wires < len(self.wires):
            raise ValueError('{}: Number of wires is too small to fit Heisenberg matrix'.format(self.name, num_wires))

        # expand U into the I, x_0, p_0, x_1, p_1, ... basis
        dim = 1 + num_wires*2
        def loc(w):
            "Returns the slice denoting the location of (x_w, p_w) in the basis."
            ind = 2*w+1
            return slice(ind, ind+2)

        if U.ndim == 1:
            W = np.zeros(dim)
            W[0] = U[0]
            for k, w in enumerate(self.wires):
                W[loc(w)] = U[loc(k)]
        elif U.ndim == 2:
            if isinstance(self, Expectation):
                W = np.zeros((dim, dim))
            else:
                W = np.eye(dim)

            W[0, 0] = U[0, 0]

            for k1, w1 in enumerate(self.wires):
                s1 = loc(k1)
                d1 = loc(w1)

                # first column
                W[d1, 0] = U[s1, 0]
                # first row (for gates, the first row is always (1, 0, 0, ...), but not for observables!)
                W[0, d1] = U[0, s1]

                for k2, w2 in enumerate(self.wires):
                    W[d1, loc(w2)] = U[s1, loc(k2)]  # block k1, k2 in U goes to w1, w2 in W.
        else:
            raise ValueError('Only order-1 and order-2 arrays supported.')
        return W

    @staticmethod
    def _heisenberg_rep(p):
        r"""Heisenberg picture representation of the operation.

        * For Gaussian CV gates, this method returns the matrix of the linear
          transformation carried out by the gate for the given parameter values.
          The method is not defined for non-Gaussian gates.

          .. note:: The existence of this method is equivalent to ``grad_method = 'A'``.

        * For observables, returns a real vector (first-order observables) or
          symmetric matrix (second-order observables) of expansion coefficients
          of the observable.

        For single-mode Operations we use the basis :math:`\mathbf{r} = (\I, \x, \p)`.
        For multi-mode Operations we use the basis :math:`\mathbf{r} = (\I, \x_0, \p_0, \x_1, \p_1, \ldots)`.

        .. note::

            For gates, we assume that the inverse transformation is obtained
            by negating the first parameter.

        Args:
            p (Sequence[float]): parameter values for the transformation

        Returns:
            array[float]: :math:`\tilde{U}` or :math:`q`
        """
        # TODO: check the note in the docstring.
        pass

    _heisenberg_rep = None  # disable the method, we just want the docstring here


class CVOperation(CV, Operation):
    """Base class for continuous-variable quantum operations."""

    def heisenberg_pd(self, idx):
        """Partial derivative of the Heisenberg picture transform matrix.

        Computed using grad_recipe.

        Args:
          idx (int): index of the parameter wrt. which the partial derivative is computed
        Returns:
          array[float]: partial derivative
        """
        # get the gradient recipe for this parameter
        recipe = self.grad_recipe[idx]
        multiplier = 0.5 if recipe is None else recipe[0]
        shift = np.pi / 2 if recipe is None else recipe[1]

        p = self.parameters
        # evaluate the transform at the shifted parameter values
        p[idx] += shift
        U2 = self._heisenberg_rep(p)
        p[idx] -= 2*shift
        U1 = self._heisenberg_rep(p)
        return (U2-U1) * multiplier  # partial derivative of the transformation


    def heisenberg_tr(self, num_wires, inverse=False):
        r"""Heisenberg picture representation of the linear transformation carried out by the gate at current parameter values.

        Given a unitary quantum gate :math:`U`, we may consider its linear transformation in the Heisenberg picture,
        :math:`U^\dagger(\cdot) U`. If the gate is Gaussian, this linear transformation preserves the polynomial order
        of any observables that are polynomials in :math:`\vec{E} = (\I, x_0, p_0, x_1, p_1, \ldots)`.
        This also means it maps :math:`\text{span} \: \vec{E}` into itself:

        .. math:: U^\dagger E_i U = \sum_j \tilde{U}_{ij} E_j

        For Gaussian CV gates, this method returns the transformation matrix for the current parameter values
        of the Operation. The method is not defined for non-Gaussian (and non-CV) gates.

        Args:
          num_wires (int): total number of wires in the quantum circuit
          inverse  (bool): if True, return the inverse transformation instead

        Returns:
          array[float]: :math:`\tilde{U}`
        """
        # not defined?
        if self._heisenberg_rep is None:
            raise RuntimeError('{} is not a Gaussian operation, or is missing the _heisenberg_rep method.'.format(self.name))

        p = self.parameters
        if inverse:
            p[0] = -p[0]  # negate first parameter
        U = self._heisenberg_rep(p)

        return self.heisenberg_expand(U, num_wires)


class CVExpectation(CV, Expectation):
    """Base class for continuous-variable expectation value measurements.

    :meth:`_heisenberg_rep` is defined iff `ev_order` is not None, and it returns an array of the corresponding ndim.
    """
    ev_order = None  #: None, int: if not None, the observable is a polynomial of the given order in `(x, p)`.

    def heisenberg_obs(self, num_wires):
        r"""Representation of the observable in the position/momentum operator basis.

        Returns the expansion :math:`q` of the observable, :math:`Q`, in the basis :math:`\vec{E} = (\I, x_0, p_0, x_1, p_1, \ldots)`.
        For first-order observables returns a real vector such that :math:`Q = \sum_i q_i E_i`.
        For second-order observables returns a real symmetric matrix such that :math:`Q = \sum_{ij} q_{ij} E_i E_j`.

        Args:
          num_wires (int): total number of wires in the quantum circuit
        Returns:
          array[float]: :math:`q`
        """
        p = self.parameters
        U = self._heisenberg_rep(p)
        return self.heisenberg_expand(U, num_wires)
