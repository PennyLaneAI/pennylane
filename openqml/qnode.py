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

import collections
from inspect import signature

import logging as log
log.getLogger()

import numbers

import autograd.numpy as np
from autograd.extend import primitive, defvjp

from .variable import Variable


def _flatten(x):
    """Flatten an irregular iterator"""
    for item in x:
        if isinstance(item, collections.Iterable) and not isinstance(item, (str, bytes)):
            yield from _flatten(item)
        else:
            yield item


class QNode:
    """A quantum node in the hybrid computational graph, encapsulating
    both the quantum function and the computational device.

    The computational device is an instance of the :class:`~.openqml.Device`
    class, and can interface with either a simulator or hardware device.
    Additional devices are available to install as plugins.

    A quantum function (or qfunc) must be of the following form:

    .. code-block:: python

        def my_quantum_function(x):
            qm.Zrotation(x, 0)
            qm.CNOT(0,1)
            qm.Yrotation(x**2, 1)
            qm.expectation.Z(0)

    To produce a valid QNode, the qfunc must consist of
    only OpenQML operators and expectation values, one per line, and must
    end with the measurement of an expectation value.

    Once the device and qfunc are defined, the QNode can then be used
    to evaluate the quantum function on the particular device.
    and processed classically using NumPy. For example,

    .. code-block:: python

        device1 = qm.device('strawberryfields.fock', cutoff=5)
        qnode1 = QNode(my_quantum_function, device1)
        result = qnode1(np.pi/4)

    .. note::

        The :func:`~.openqml.qfunc` decorator is provided as a convenience
        to automate the process of creating quantum nodes. Using this decorator,
        the above example becomes:

        .. code-block:: python

            @qfunc(device1)
            def my_quantum_function(x):
                qm.Zrotation(x, 0)
                qm.CNOT(0,1)
                qm.Yrotation(x**2, 1)
                qm.expectation.Z(0)

            result = my_quantum_function(np.pi/4)

    Args:
        func (qfunc): a Python function containing OpenQML quantum operations,
            one per line, ending with an expectation value.
        device (openqml.Device): an OpenQML-compatible device.
    """
    def __init__(self, func, device):
        self.func = func
        self.device = device
        self.variables = []
        self.variable_ops = {}

        # determine number of variables
        sig = signature(func).parameters.items()
        self.num_variables = len([key for key, p in sig])# if p.kind == p.POSITIONAL_OR_KEYWORD])

    @primitive
    def __call__(self, *args, **kwargs):
        """Evaluates the quantum function on the specified device, and returns
        the output expectation value."""

        self.variables = []
        self.device._queue = []

        # wrap each argument with a Variable class
        p = []
        for idx, val in enumerate(_flatten(args)):
            p.append(Variable(idx, val=val))

        self.variables = p

        # evaluate device
        with self.device:
            self.func(*p, **kwargs)

        if not self.variable_ops:
            # map each variable to the operations which depend on it
            for op in self.device._queue:
                for p in op.params:
                    if isinstance(p, Variable):
                        self.variable_ops.setdefault(p.idx, []).append(op)

            # map from free parameter index to the gradient method to be used with that parameter
            self.grad_method = {k: self.best_method(v) for k, v in self.variable_ops.items()}

        return self.expectation

    @property
    def expectation(self):
        """Expectation value of the QNode"""
        return self.device._out

    def best_method(self, ops):
        """determine the correct gradient computation method for each free parameter"""
        # use the angle method iff every gate that depends on the parameter supports it
        list_of_methods = [op.grad_method == 'A' and len(op.params) == 1 for op in ops]

        if all(list_of_methods):
            return 'A'

        return 'F'

    def gradient(self, params, which=None, method='B', *, h=1e-7, order=1, **kwargs):
        """Compute the gradient of the node.

        Returns the gradient of the parametrized quantum circuit encapsulated in the QNode.

        The gradient can be computed using several methods:

        * Finite differences (``'F'``). The first order method evaluates the circuit at
          n+1 points of the parameter space, the second order method at 2n points,
          where n = len(which).
        * Angular method (``'A'``). Only works for one-parameter gates where the generator
          only has two unique eigenvalues. The circuit is evaluated twice for each incidence
          of each parameter in the circuit.
        * Best known method for each parameter (``'B'``): uses the angular method if
          possible, otherwise finite differences.

        Args:
            params (Sequence[float]): point in parameter space at which to evaluate the gradient
            which  (Sequence[int], None): return the gradient with respect to these parameters.
                None means all.
            method (str): gradient computation method, see above

        Keyword Args:
            h (float): finite difference method step size
            order (int): finite difference method order, 1 or 2

        Returns:
            vector[float]: gradient vector
        """
        if isinstance(params, numbers.Number):
            params = [params]

        if which is None:
            which = range(len(params))
        elif len(which) != len(set(which)):  # set removes duplicates
            raise ValueError('Parameter indices must be unique.')
        params = np.asarray(params)

        if method in ('A', 'F'):
            method = {k: method for k in which}
        elif method == 'B':
            method = self.grad_method

        if 'F' in method.values():
            if order == 1:
                # the value of the circuit at params, computed only once here
                y0 = self.__call__(*params, **kwargs)
            else:
                y0 = None

        # compute the partial derivative w.r.t. each parameter using the proper method
        grad = np.zeros(len(which), dtype=float)
        for i, k in enumerate(which):
            par_method = method[k]
            if par_method == 'A':
                grad[i] = self._pd_angle(params, k, **kwargs)
            elif par_method == 'F':
                grad[i] = self._pd_finite_diff(params, k, h, order, y0, **kwargs)
            else:
                raise ValueError('Unknown gradient method.')
        return grad


    def _pd_finite_diff(self, params, idx, h=1e-7, order=1, y0=None, **kwargs):
        """Partial derivative of the node using the finite difference method.

        Args:
            params (array[float]): point in parameter space at which to evaluate
                the partial derivative.
            idx (int): return the partial derivative with respect to this parameter
            h (float): step size.
            order (int): finite difference method order, 1 or 2
            y0 (float): Value of the circuit at params. Should only be computed once.

        Returns:
            float: partial derivative of the node.
        """
        if order == 1:
            # shift one parameter by h
            shift_params = params.copy()
            shift_params[idx] += h
            y = self.__call__(*shift_params, **kwargs)
            return (y-y0) / h
        elif order == 2:
            # symmetric difference
            # shift one parameter by +-h/2
            shift_params = params.copy()
            shift_params[idx] += 0.5*h
            y2 = self.__call__(*shift_params, **kwargs)
            shift_params[idx] = params[idx] -0.5*h
            y1 = self.__call__(*shift_params, **kwargs)
            return (y2-y1) / h
        else:
            raise ValueError('Order must be 1 or 2.')


    def _pd_angle(self, params, idx, **kwargs):
        """Partial derivative of the node using the angle method.

        Args:
          params (array[float]): point in parameter space at which to evaluate
            the partial derivative.
          idx (int): return the partial derivative with respect to this parameter.

        Returns:
          float: partial derivative of the node.
        """
        n = len(self.variables)
        pd = 0.0
        # find the Commands in which the parameter appears, use the product rule
        for op in self.variable_ops[idx]:
            if len(op.params) != 1:
                raise ValueError('For now angular method can only differentiate one-parameter gates.')
            if op.grad_method != 'A':
                raise ValueError('Attempted to use the angular method on a gate that does not support it.')

            # # we temporarily edit the Command so that parameter idx is replaced by a new one,
            # # which we can modify without affecting other Commands depending on the original.
            # orig = op.params[0]
            # assert(orig.idx == idx)
            # op.params[0] = Variable(n, orig)  # reference to a new, temporary parameter (this method only supports 1-parameter gates!)
            # self.variable_ops[n] = None  # we just need to add something to the map, it's not actually used
            # # shift it by pi/2 and -pi/2
            # shift_params = np.r_[params, params[idx] + np.pi/2]
            # print('here', *shift_params)
            # y2 = self.__call__(*shift_params, **kwargs)
            # shift_params[-1] = params[idx] - np.pi/2
            # y1 = self.__call__(*shift_params, **kwargs)
            # # restore the original parameter
            # op.params[0] = orig
            # del self.variable_ops[n]  # remove the temporary entry
            # pd += (y2-y1) / 2

            # shift it by pi/2 and -pi/2
            shift_params = params.copy()
            shift_params[idx] = params[idx] + np.pi/2
            y2 = self.__call__(*shift_params, **kwargs)
            shift_params[idx] = params[idx] - np.pi/2
            y1 = self.__call__(*shift_params, **kwargs)
            pd += (y2-y1) / 2
        return pd


def QNode_vjp(ans, self, params, *args, **kwargs):
    """Returns the vector Jacobian product for a QNode, as a function
    of the QNode evaluation at the specified parameter values.
    """
    p = list(params) + list(args)
    return lambda g: g * self.gradient(p)


# define the vector-Jacobian product function for QNode.__call__()
defvjp(QNode.__call__, QNode_vjp, argnums=[1])
