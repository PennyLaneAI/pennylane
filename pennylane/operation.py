# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=protected-access
r"""
This module contains the abstract base classes for defining PennyLane
operations and observables.

Description
-----------

Qubit Operations
~~~~~~~~~~~~~~~~
The :class:`Operator` class serves as a base class for operators,
and is inherited by both the :class:`Observable` class and the
:class:`Operation` class. These classes are subclassed to implement quantum operations
and measure observables in PennyLane.

* Each :class:`~.Operator` subclass represents a general type of
  map between physical states. Each instance of these subclasses
  represents either

  - an application of the operator or
  - an instruction to measure and return the respective result.

  Operators act on a sequence of wires (subsystems) using given parameter values.

* Each :class:`~.Operation` subclass represents a type of quantum operation,
  for example a unitary quantum gate. Each instance of these subclasses
  represents an application of the operation with given parameter values to
  a given sequence of wires (subsystems).

* Each  :class:`~.Observable` subclass represents a type of physical observable.
  Each instance of these subclasses represents an instruction to measure and
  return the respective result for the given parameter values on a
  sequence of wires (subsystems).

Differentiation
^^^^^^^^^^^^^^^

In general, an :class:`Operation` is differentiable (at least using the finite-difference
method) with respect to a parameter iff

* the domain of that parameter is continuous.

For an :class:`Operation` to be differentiable with respect to a parameter using the
analytic method of differentiation, it must satisfy an additional constraint:

* the parameter domain must be real.

.. note::

    These conditions are *not* sufficient for analytic differentiation. For example,
    CV gates must also define a matrix representing their Heisenberg linear
    transformation on the quadrature operators.

For gates that *are* supported via the analytic method, the gradient recipe
(with multiplier :math:`c_k`, parameter shift :math:`s_k` for parameter :math:`\phi_k`)
works as follows:

.. math:: \frac{\partial}{\partial\phi_k}O = c_k\left[O(\phi_k+s_k)-O(\phi_k-s_k)\right].

CV Operation base classes
~~~~~~~~~~~~~~~~~~~~~~~~~

Due to additional requirements, continuous-variable (CV) operations must subclass the
:class:`~.CVOperation` or :class:`~.CVObservable` classes instead of :class:`~.Operation`
and :class:`~.Observable`.

Differentiation
^^^^^^^^^^^^^^^

To enable gradient computation using the analytic method for Gaussian CV operations, in addition, you need to
provide the static class method :meth:`~.CV._heisenberg_rep` that returns the Heisenberg representation of
the operation given its list of parameters, namely:

* For Gaussian CV Operations this method should return the matrix of the linear transformation carried out by the
  operation on the vector of quadrature operators :math:`\mathbf{r}` for the given parameter
  values.

* For Gaussian CV Observables this method should return a real vector (first-order observables)
  or symmetric matrix (second-order observables) of coefficients of the quadrature
  operators :math:`\x` and :math:`\p`.

PennyLane uses the convention :math:`\mathbf{r} = (\I, \x, \p)` for single-mode operations and observables
and :math:`\mathbf{r} = (\I, \x_0, \p_0, \x_1, \p_1, \ldots)` for multi-mode operations and observables.

.. note::
    Non-Gaussian CV operations and observables are currently only supported via
    the finite-difference method of gradient computation.
"""
import abc
import itertools
import functools
import numbers
from collections.abc import Sequence
from enum import Enum, IntEnum

import numpy as np
from numpy.linalg import multi_dot

import pennylane as qml

from .utils import _flatten, _unflatten, pauli_eigs
from .variable import Variable

#=============================================================================
# Wire types
#=============================================================================

class Wires(IntEnum):
    """Integer enumeration class
    to represent the number of wires
    an operation acts on"""
    Any = -1
    All = 0


All = Wires.All
"""IntEnum: An enumeration which represents all wires in the
subsystem. It is equivalent to an integer with value 0."""

Any = Wires.Any
"""IntEnum: An enumeration which represents any wires in the
subsystem. It is equivalent to an integer with value -1."""


#=============================================================================
# ObservableReturnTypes types
#=============================================================================

class ObservableReturnTypes(Enum):
    """Enumeration class to
    represent the type of
    return types of an observable."""
    Sample = 1
    Variance = 2
    Expectation = 3
    Probability = 4


Sample = ObservableReturnTypes.Sample
"""Enum: An enumeration which represents sampling an observable."""

Variance = ObservableReturnTypes.Variance
"""Enum: An enumeration which represents returning the variance of
an observable on specified wires."""

Expectation = ObservableReturnTypes.Expectation
"""Enum: An enumeration which represents returning the expectation
value of an observable on specified wires."""

Probability = ObservableReturnTypes.Probability
"""Enum: An enumeration which represents returning probabilities
of all computational basis states."""

#=============================================================================
# Class property
#=============================================================================


class ClassPropertyDescriptor: # pragma: no cover
    """Allows a class property to be defined"""
    # pylint: disable=too-few-public-methods
    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        """Set the function as a class method, and store as an attribute."""
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self

def classproperty(func):
    """The class property decorator"""
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


#=============================================================================
# Base Operator class
#=============================================================================

class Operator(abc.ABC):
    r"""Base class for quantum operators supported by a device.

    The following class attributes must be defined for all Operators:

    * :attr:`~.Operator.num_params`
    * :attr:`~.Operator.num_wires`
    * :attr:`~.Operator.par_domain`

    Args:
        params (tuple[float, int, array, Variable]): operator parameters

    Keyword Args:
        wires (Sequence[int]): Subsystems it acts on. If not given, args[-1]
            is interpreted as wires.
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into a :class:`BaseQNode` circuit queue.
            The circuit queue is determined by the presence of an
            applicable `qml._current_context`. If no context is
            available, this argument is ignored.
    """
    @staticmethod
    def _matrix(*params):
        """Matrix representation of the operator
        in the computational basis.

        This is a *static method* that should be defined for all
        new operations and observables, that returns the matrix representing
        the operator in the computational basis.

        This private method allows matrices to be computed
        directly without instantiating the operators first.

        To return the matrices of *instantiated* operators,
        please use the :attr:`~.Operator.matrix` property instead.

        **Example:**

        >>> qml.RY._matrix(0.5)
        >>> array([[ 0.96891242+0.j, -0.24740396+0.j],
                   [ 0.24740396+0.j,  0.96891242+0.j]])

        Returns:
            array: matrix representation
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_params(self):
        """Number of parameters the operator takes."""

    @property
    @abc.abstractmethod
    def num_wires(self):
        """Number of wires the operator acts on."""

    @property
    @abc.abstractmethod
    def par_domain(self):
        """Domain of the gate parameters.

        * ``'N'``: natural numbers (including zero).
        * ``'R'``: floats.
        * ``'A'``: arrays of real or complex values.
        * ``None``: if there are no parameters.
        """

    @property
    def name(self):
        """String for the name of the operator.
        """
        return self._name

    @property
    def matrix(self):
        r"""Matrix representation of an instantiated operator
        in the computational basis.

        **Example:**

        >>> U = qml.RY(0.5, wires=1)
        >>> U.matrix
        >>> array([[ 0.96891242+0.j, -0.24740396+0.j],
                   [ 0.24740396+0.j,  0.96891242+0.j]])

        Returns:
            array: matrix representation
        """
        return self._matrix(*self.parameters)

    @name.setter
    def name(self, value):
        self._name = value

    def __init__(self, *params, wires=None, do_queue=True):
        # pylint: disable=too-many-branches
        self._name = self.__class__.__name__   #: str: name of the operator
        self.queue_idx = None  #: int, None: index of the Operator in the circuit queue, or None if not in a queue

        if wires is None:
            raise ValueError("Must specify the wires that {} acts on".format(self.name))

        if len(params) != self.num_params:
            raise ValueError("{}: wrong number of parameters. "
                             "{} parameters passed, {} expected.".format(self.name, params, self.num_params))

        # check the validity of the params
        for p in params:
            self.check_domain(p)
        self.params = list(params)  #: list[Any]: parameters of the operator

        # apply the operator on the given wires
        if not isinstance(wires, Sequence):
            wires = [wires]
        self._check_wires(wires)
        self._wires = wires  #: tuple[int]: wires on which the operator acts

        if do_queue and (qml._current_context is not None):
            self.queue()

    def __str__(self):
        """Print the operator name and some information."""
        return  "{}: {} params, wires {}".format(self.name, len(self.params), self.wires)

    def _check_wires(self, wires):
        """Check the validity of the operator wires.

        Args:
            wires (Sequence[Any]): wires to check
        Raises:
            TypeError, ValueError: list of wires is invalid
        Returns:
            tuple[int]: wires converted to integers
        """
        for w in wires:
            if not isinstance(w, numbers.Integral):
                raise TypeError('{}: Wires must be integers, or integer-valued nondifferentiable parameters in mutable circuits.'.format(self.name))

        if self.num_wires != All and self.num_wires != Any and len(wires) != self.num_wires:
            raise ValueError("{}: wrong number of wires. "
                             "{} wires given, {} expected.".format(self.name, len(wires), self.num_wires))

        if len(set(wires)) != len(wires):
            raise ValueError('{}: wires must be unique, got {}.'.format(self.name, wires))

        return tuple(int(w) for w in wires)

    def check_domain(self, p, flattened=False):
        """Check the validity of a parameter.

        :class:`Variable` instances can represent any real scalars (but not arrays).

        Args:
            p (Number, array, Variable): parameter to check
            flattened (bool): True means p is an element of a flattened parameter
                sequence (affects the handling of 'A' parameters)
        Raises:
            TypeError: parameter is not an element of the expected domain
            ValueError: parameter is an element of an unknown domain
        Returns:
            Number, array, Variable: p
        """
        if isinstance(p, Variable):
            if self.par_domain == 'A':
                raise TypeError('{}: Array parameter expected, got a Variable,'
                                'which can only represent real scalars.'.format(self.name))
            return p

        # p is not a Variable
        if self.par_domain == 'A':
            if flattened:
                if isinstance(p, np.ndarray):
                    raise TypeError('{}: Flattened array parameter expected, got {}.'.format(self.name, type(p)))
            else:
                if not isinstance(p, np.ndarray):
                    raise TypeError('{}: Array parameter expected, got {}.'.format(self.name, type(p)))
        elif self.par_domain in ('R', 'N'):
            if not isinstance(p, numbers.Real):
                raise TypeError('{}: Real scalar parameter expected, got {}.'.format(self.name, type(p)))

            if self.par_domain == 'N':
                if not isinstance(p, numbers.Integral):
                    raise TypeError('{}: Natural number parameter expected, got {}.'.format(self.name, type(p)))
                if p < 0:
                    raise TypeError('{}: Natural number parameter expected, got {}.'.format(self.name, p))
        else:
            raise ValueError('{}: Unknown parameter domain \'{}\'.'.format(self.name, self.par_domain))
        return p

    @property
    def wires(self):
        """Wire values.

        Returns:
            tuple[int]: wire values
        """
        return self._wires

    @property
    def parameters(self):
        """Current parameter values.

        Fixed parameters are returned as is, free parameters represented by
        :class:`~.variable.Variable` instances are replaced by their
        current numerical value.

        Returns:
            list[float]: parameter values
        """
        temp = list(_flatten(self.params))
        temp_val = [self.check_domain(x.val, True) if isinstance(x, Variable) else x for x in temp]
        return _unflatten(temp_val, self.params)[0]

    def queue(self):
        """Append the operator to a BaseQNode queue."""

        qml._current_context._append_op(self)
        return self  # so pre-constructed Observable instances can be queued and returned in a single statement

#=============================================================================
# Base Operation class
#=============================================================================

class Operation(Operator):
    r"""Base class for quantum operations supported by a device.

    As with :class:`~.Operator`, the following class attributes must be
    defined for all operations:

    * :attr:`~.Operator.num_params`
    * :attr:`~.Operator.num_wires`
    * :attr:`~.Operator.par_domain`

    The following two class attributes are optional, but in most cases
    should be clearly defined to avoid unexpected behavior during
    differentiation.

    * :attr:`~.Operation.grad_method`
    * :attr:`~.Operation.grad_recipe`

    Finally, there are some additional optional class attributes
    that may be set, and used by certain quantum optimizers:

    * :attr:`~.Operation.generator`

    Args:
        params (tuple[float, int, array, Variable]): operation parameters

    Keyword Args:
        wires (Sequence[int]): Subsystems it acts on. If not given, args[-1]
            is interpreted as wires.
        do_queue (bool): Indicates whether the operation should be
            immediately pushed into a :class:`BaseQNode` circuit queue.
            This flag is useful if there is some reason to run an Operation
            outside of a BaseQNode context.
    """
    # pylint: disable=abstract-method
    string_for_inverse = ".inv"

    @property
    def grad_method(self):
        """Gradient computation method.

        * ``'A'``: analytic differentiation using the parameter-shift method.
        * ``'F'``: finite difference numerical differentiation.
        * ``None``: the operation may not be differentiated.

        Default is ``'F'``, or ``None`` if the Operation has zero parameters.
        """
        return None if self.num_params == 0 else 'F'

    grad_recipe = None
    r"""list[tuple[float]] or None: Gradient recipe for the parameter-shift method.

        This is a list with one tuple per operation parameter. For parameter
        :math:`k`, the tuple is of the form :math:`(c_k, s_k)`, resulting in
        a gradient recipe of

        .. math:: \frac{\partial}{\partial\phi_k}O = c_k\left[O(\phi_k+s_k)-O(\phi_k-s_k)\right].

        If ``None``, the default gradient recipe
        :math:`(c_k, s_k)=(1/2, \pi/2)` is assumed for every parameter.
    """

    def get_parameter_shift(self, idx):
        """Multiplier and shift for the given parameter, based on its gradient recipe.

        Args:
            idx (int): parameter index

        Returns:
            float, float: multiplier, shift
        """
        # get the gradient recipe for this parameter
        recipe = self.grad_recipe[idx]
        # internal multiplier in the Variable
        var_mult = self.params[idx].mult

        multiplier = 0.5 if recipe is None else recipe[0]
        multiplier *= var_mult
        shift = np.pi / 2 if recipe is None else recipe[1]
        shift /= var_mult
        return multiplier, shift

    @property
    def generator(self):
        r"""Generator of the operation.

        A length-2 list ``[generator, scaling_factor]``, where

        * ``generator`` is an existing PennyLane
          operation class or :math:`2\times 2` Hermitian array
          that acts as the generator of the current operation

        * ``scaling_factor`` represents a scaling factor applied
          to the generator operation

        For example, if :math:`U(\theta)=e^{i0.7\theta \sigma_x}`, then
        :math:`\sigma_x`, with scaling factor :math:`s`, is the generator
        of operator :math:`U(\theta)`:

        .. code-block:: python

            generator = [PauliX, 0.7]

        Default is ``[None, 1]``, indicating the operation has no generator.
        """
        return [None, 1]

    @property
    def inverse(self):
        """Boolean determining if the inverse of the operation was requested.
        """
        return self._inverse

    @inverse.setter
    def inverse(self, boolean):
        self._inverse = boolean

    @staticmethod
    def decomposition(*params, wires):
        """Returns a template decomposing the operation into other
        quantum operations."""
        raise NotImplementedError

    def inv(self):
        """Inverts the operation, such that the inverse will
        be used for the computations by the specific device.

        This method concatenates a string to the name of the operation,
        to indicate that the inverse will be used for computations.

        Any subsequent call of this method will toggle between the original
        operation and the inverse of the operation.

        Returns:
            :class:`Operator`: operation to be inverted
        """
        self.inverse = not self._inverse
        return self

    @property
    def matrix(self):
        if self.inverse:
            return np.linalg.inv(self._matrix(*self.parameters))

        return self._matrix(*self.parameters)

    @property
    def base_name(self):
        """Get base name of the operator.
        """
        return self.__class__.__name__

    @property
    def name(self):
        """Get and set the name of the operator.
        """
        return self._name + Operation.string_for_inverse if self.inverse else self._name

    def __init__(self, *params, wires=None, do_queue=True):

        self._inverse = False

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

        super().__init__(*params, wires=wires, do_queue=do_queue)


#=============================================================================
# Base Observable class
#=============================================================================


class Observable(Operator):
    """Base class for observables supported by a device.

    :class:`Observable` is used to describe Hermitian quantum observables.

    As with :class:`~.Operator`, the following class attributes must be
    defined for all observables:

    * :attr:`~.Operator.num_params`
    * :attr:`~.Operator.num_wires`
    * :attr:`~.Operator.par_domain`

    Args:
        params (tuple[float, int, array, Variable]): observable parameters

    Keyword Args:
        wires (Sequence[int]): subsystems it acts on.
            Currently, only one subsystem is supported.
        do_queue (bool): Indicates whether the operation should be
            immediately pushed into a :class:`BaseQNode` observable queue.
            The observable queue is determined by the presence of an
            applicable `qml._current_context`. If no context is
            available, this argument is ignored.
    """
    # pylint: disable=abstract-method
    return_type = None

    def __init__(self, *params, wires=None, do_queue=True):
        # extract the arguments
        if wires is None:
            wires = params[-1]
            params = params[:-1]

        super().__init__(*params, wires=wires, do_queue=do_queue)

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            return other.__rmatmul__(self)

        if isinstance(other, Observable):
            return Tensor(self, other)

        raise ValueError("Can only perform tensor products between observables.")

    @property
    def eigvals(self):
        r"""Returns the eigenvalues of the observable"""
        raise NotImplementedError

    def diagonalizing_gates(self):
        r"""Returns the list of operations such that they
        diagonalize the observable in the computational basis.

        Returns:
            list(qml.Operation): A list of gates that diagonalize
            the observable in the computational basis.
        """
        raise NotImplementedError


class Tensor(Observable):
    """Container class representing tensor products of observables.

    To create a tensor, simply initiate it like so:

    >>> T = Tensor(qml.PauliX(0), qml.Hermitian(A, [1, 2]))

    You can also create a tensor from other Tensors:

    >>> T = Tensor(T, qml.PauliZ(4))

    The ``@`` symbol can be used as a tensor product operation:

    >>> T = qml.PauliX(0) @ qml.Hadamard(2)
    """
    # pylint: disable=abstract-method
    return_type = None
    tensor = True
    par_domain = None

    def __init__(self, *args): #pylint: disable=super-init-not-called

        self._eigvals = None
        self.obs = []

        for o in args:
            if isinstance(o, Tensor):
                self.obs.extend(o.obs)
            elif isinstance(o, Observable):
                self.obs.append(o)
            else:
                raise ValueError("Can only perform tensor products between observables.")

    def __str__(self):
        """Print the tensor product and some information."""
        return 'Tensor product {}: {} params, wires {}'.format([i.name for i in self.obs], len(self.params), self.wires)

    @property
    def name(self):
        """All constituent observable names making up the tensor product.

        Returns:
            list[str]: list containing all observable names
        """
        return [o.name for o in self.obs]

    @property
    def num_wires(self):
        """Number of wires the tensor product acts on.

        Returns:
            int: number of wires
        """
        return len(list(_flatten(self.wires)))

    @property
    def wires(self):
        """All wires in the system the tensor product acts on.

        Returns:
            list[list[Any]]: nested list containing the wires per observable
            in the tensor product
        """
        return [o.wires for o in self.obs]

    @property
    def params(self):
        """Raw parameters of all constituent observables in the tensor product.

        Returns:
            list[Any]: flattened list containing all dependent parameters
        """
        return [p for sublist in [o.params for o in self.obs] for p in sublist]

    @property
    def num_params(self):
        """Raw parameters of all constituent observables in the tensor product.

        Returns:
            list[Any]: flattened list containing all dependent parameters
        """
        return len(self.params)

    @property
    def parameters(self):
        """Evaluated parameter values of all constituent observables in the tensor product.

        Returns:
            list[list[Any]]: nested list containing the parameters per observable
            in the tensor product
        """
        return [o.parameters for o in self.obs]

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            self.obs.extend(other.obs)
            return self

        if isinstance(other, Observable):
            self.obs.append(other)
            return self

        raise ValueError("Can only perform tensor products between observables.")

    def __rmatmul__(self, other):
        if isinstance(other, Observable):
            self.obs[:0] = [other]
            return self

        raise ValueError("Can only perform tensor products between observables.")

    __imatmul__ = __matmul__

    @property
    def eigvals(self):
        """Return the eigenvalues of the specified tensor product observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible.

        Returns:
            array[float]: array containing the eigenvalues of the tensor product
            observable
        """
        if self._eigvals is not None:
            return self._eigvals

        standard_observables = {"PauliX", "PauliY", "PauliZ", "Hadamard"}

        # observable should be Z^{\otimes n}
        self._eigvals = pauli_eigs(len(self.wires))

        # TODO: check for edge cases of the sorting, e.g. Tensor(Hermitian(obs, wires=[0, 2]),
        # Hermitian(obs, wires=[1, 3, 4])
        # Sorting the observables based on wires, so that the order of
        # the eigenvalues is correct
        obs_sorted = sorted(self.obs, key=lambda x: x.wires)

        # check if there are any non-standard observables (such as Identity)
        if set(self.name) - standard_observables:
            # Tensor product of observables contains a mixture
            # of standard and non-standard observables
            self._eigvals = np.array([1])
            for k, g in itertools.groupby(obs_sorted, lambda x: x.name in standard_observables):
                if k:
                    # Subgroup g contains only standard observables.
                    self._eigvals = np.kron(self._eigvals, pauli_eigs(len(list(g))))
                else:
                    # Subgroup g contains only non-standard observables.
                    for ns_ob in g:
                        # loop through all non-standard observables
                        self._eigvals = np.kron(self._eigvals, ns_ob.eigvals)

        wire_ordering = np.argsort(np.argsort(list(_flatten(self.wires))))
        tuples = np.array(list(itertools.product([0, 1], repeat=self.num_wires)))
        perm = np.ravel_multi_index(tuples[:, wire_ordering].T, [2] * self.num_wires)

        return self._eigvals[perm]

    def diagonalizing_gates(self):
        """Return the gate set that diagonalizes a circuit according to the
        specified tensor observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        Returns:
            list: list containing the gates diagonalizing the tensor observable
        """
        diag_gates = []
        for o in self.obs:
            diag_gates.extend(o.diagonalizing_gates())

        return diag_gates

    @property
    def matrix(self):
        r"""Matrix representation of the tensor operator
        in the computational basis.

        **Example:**

        Note that the returned matrix *only includes explicitly
        declared observables* making up the tensor product;
        that is, it only returns the matrix for the specified
        subsystem it is defined for.

        >>> O = qml.PauliZ(0) @ qml.PauliZ(2)
        >>> O.matrix
        array([[ 1,  0,  0,  0],
               [ 0, -1,  0,  0],
               [ 0,  0, -1,  0],
               [ 0,  0,  0,  1]])

        To get the full :math:`2^3\times 2^3` Hermitian matrix
        acting on the 3-qubit system, the identity on wire 1
        must be explicitly included:

        >>> O = qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2)
        >>> O.matrix
        array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0., -1.,  0., -0.,  0., -0.,  0., -0.],
               [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
               [ 0., -0.,  0., -1.,  0., -0.,  0., -0.],
               [ 0.,  0.,  0.,  0., -1., -0., -0., -0.],
               [ 0., -0.,  0., -0., -0.,  1., -0.,  0.],
               [ 0.,  0.,  0.,  0., -0., -0., -1., -0.],
               [ 0., -0.,  0., -0., -0.,  0., -0.,  1.]])

        Returns:
            array: matrix representation
        """
        # group the observables based on what wires they act on
        U_list = []
        for _, g in itertools.groupby(self.obs, lambda x: x.wires):
            # extract the matrices of each diagonalizing gate
            mats = [i.matrix for i in g]

            if len(mats) > 1:
                # multiply all unitaries together before appending
                mats = [multi_dot(mats)]

            # append diagonalizing unitary for specific wire to U_list
            U_list.append(mats[0])

        # Return the Hermitian matrix representing the observable
        # over the defined wires.
        return functools.reduce(np.kron, U_list)

#=============================================================================
# CV Operations and observables
#=============================================================================


class CV:
    """A mixin base class denoting a continuous-variable operation."""
    # pylint: disable=no-member

    def heisenberg_expand(self, U, num_wires):
        """Expand the given local Heisenberg-picture array into a full-system one.

        Args:
            U (array[float]): array to expand (expected to be of the dimension ``1+2*self.num_wires``)
            num_wires (int): total number of wires in the quantum circuit. If zero, return ``U`` as is.

        Raises:
            ValueError: if the size of the input matrix is invalid or `num_wires` is incorrect

        Returns:
            array[float]: expanded array, dimension ``1+2*num_wires``
        """
        U_dim = len(U)
        nw = len(self.wires)


        if U.ndim > 2:
            raise ValueError('Only order-1 and order-2 arrays supported.')

        if U_dim != 1+2*nw:
            raise ValueError('{}: Heisenberg matrix is the wrong size {}.'.format(self.name, U_dim))

        if num_wires == 0 or list(self.wires) == list(range(num_wires)):
            # no expansion necessary (U is a full-system matrix in the correct order)
            return U

        if num_wires < len(self.wires):
            raise ValueError('{}: Number of wires {} is too small to fit Heisenberg matrix'.format(self.name, num_wires))

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
            if isinstance(self, Observable):
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
        return W

    @staticmethod
    def _heisenberg_rep(p):
        r"""Heisenberg picture representation of the operation.

        * For Gaussian CV gates, this method returns the matrix of the linear
          transformation carried out by the gate for the given parameter values.
          The method is not defined for non-Gaussian gates.

          **The existence of this method is equivalent to setting** ``grad_method = 'A'``.

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
        # pylint: disable=unused-argument
        return None

    @classproperty
    def supports_heisenberg(self):
        """Returns True iff the CV Operation has overridden the :meth:`~.CV._heisenberg_rep`
        static method, thereby indicating that it is Gaussian and does not block the use
        of the parameter-shift differentiation method if found between the differentiated gate
        and an observable.
        """
        return CV._heisenberg_rep != self._heisenberg_rep


class CVOperation(CV, Operation):
    """Base class for continuous-variable quantum operations."""
    # pylint: disable=abstract-method

    @classproperty
    def supports_parameter_shift(self):
        """Returns True iff the CV Operation supports the parameter-shift differentiation method.
        This means that it has ``grad_method='A'`` and
        has overridden the :meth:`~.CV._heisenberg_rep` static method.
        """
        return self.grad_method == 'A' and self.supports_heisenberg

    def heisenberg_pd(self, idx):
        """Partial derivative of the Heisenberg picture transform matrix.

        Computed using grad_recipe.

        Args:
            idx (int): index of the parameter with respect to which the
                partial derivative is computed.
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
        U2 = self._heisenberg_rep(p) # pylint: disable=assignment-from-none
        p[idx] -= 2*shift
        U1 = self._heisenberg_rep(p) # pylint: disable=assignment-from-none
        return (U2-U1) * multiplier  # partial derivative of the transformation

    def heisenberg_tr(self, num_wires, inverse=False):
        r"""Heisenberg picture representation of the linear transformation carried
        out by the gate at current parameter values.

        Given a unitary quantum gate :math:`U`, we may consider its linear
        transformation in the Heisenberg picture, :math:`U^\dagger(\cdot) U`.

        If the gate is Gaussian, this linear transformation preserves the polynomial order
        of any observables that are polynomials in :math:`\mathbf{r} = (\I, \x_0, \p_0, \x_1, \p_1, \ldots)`.
        This also means it maps :math:`\text{span}(\mathbf{r})` into itself:

        .. math:: U^\dagger \mathbf{r}_i U = \sum_j \tilde{U}_{ij} \mathbf{r}_j

        For Gaussian CV gates, this method returns the transformation matrix for
        the current parameter values of the Operation. The method is not defined
        for non-Gaussian (and non-CV) gates.

        Args:
            num_wires (int): total number of wires in the quantum circuit
            inverse  (bool): if True, return the inverse transformation instead

        Raises:
            RuntimeError: if the specified operation is not Gaussian or is missing the `_heisenberg_rep` method

        Returns:
            array[float]: :math:`\tilde{U}`, the Heisenberg picture representation of the linear transformation
        """
        p = self.parameters
        if inverse:
            if self.par_domain == 'A':
                # TODO: expand this for the new par domain class, for non-unitary matrices.
                p[0] = np.linalg.inv(p[0])
            else:
                p[0] = -p[0]  # negate first parameter
        U = self._heisenberg_rep(p) # pylint: disable=assignment-from-none

        # not defined?
        if U is None:
            raise RuntimeError('{} is not a Gaussian operation, or is missing the _heisenberg_rep method.'.format(self.name))

        return self.heisenberg_expand(U, num_wires)


class CVObservable(CV, Observable):
    r"""Base class for continuous-variable observables.

    The class attribute :attr:`~.ev_order` can be defined to indicate
    to PennyLane whether the corresponding CV observable is a polynomial in the
    quadrature operators. If so,

    * ``ev_order = 1`` indicates a first order polynomial in quadrature
      operators :math:`(\x, \p)`.

    * ``ev_order = 2`` indicates a second order polynomial in quadrature
      operators :math:`(\x, \p)`.

    If :attr:`~.ev_order` is not ``None``, then the Heisenberg representation
    of the observable should be defined in the static method :meth:`~.CV._heisenberg_rep`,
    returning an array of the correct dimension.
    """
    # pylint: disable=abstract-method
    ev_order = None  #: None, int: if not None, the observable is a polynomial of the given order in `(x, p)`.

    def heisenberg_obs(self, num_wires):
        r"""Representation of the observable in the position/momentum operator basis.

        Returns the expansion :math:`q` of the observable, :math:`Q`, in the
        basis :math:`\mathbf{r} = (\I, \x_0, \p_0, \x_1, \p_1, \ldots)`.

        * For first-order observables returns a real vector such
          that :math:`Q = \sum_i q_i \mathbf{r}_i`.

        * For second-order observables returns a real symmetric matrix
          such that :math:`Q = \sum_{ij} q_{ij} \mathbf{r}_i \mathbf{r}_j`.

        Args:
            num_wires (int): total number of wires in the quantum circuit
        Returns:
            array[float]: :math:`q`
        """
        p = self.parameters
        U = self._heisenberg_rep(p) # pylint: disable=assignment-from-none
        return self.heisenberg_expand(U, num_wires)
