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
import copy
import collections

import logging as log
log.getLogger()

import numbers

import autograd.numpy as np
import autograd.extend as ae
import autograd.builtins

from .device import QuantumFunctionError
import openqml.operation
from .variable import Variable


def _flatten(x):
    """Iterate through an arbitrarily nested structure, flattening it in depth-first order.

    See also :func:`_unflatten`.

    Args:
      x (array, Iterable, other): each element of the Iterable may be of the same types as x
    Yieds:
      other: elements of x in depth-first order
    """
    if isinstance(x, np.ndarray):
        yield from _flatten(x.flat)  # should we allow object arrays? or just "yield from x.flat"?
    elif isinstance(x, collections.Iterable) and not isinstance(x, (str, bytes)):
        for item in x:
            yield from _flatten(item)
    else:
        yield x


def _unflatten(flat, model):
    """Restores an arbitrary nested structure to a flattened iterable.

    See also :func:`_flatten`.

    Args:
      flat (array): 1D array of items
      model (array, Iterable, Number): model nested structure
    Returns:
      (other, array): first elements of flat arranged into the nested structure of model, unused elements of flat
    """
    if isinstance(model, np.ndarray):
        idx = model.size
        res = flat[:idx].reshape(model.shape)
        return res, flat[idx:]
    elif isinstance(model, collections.Iterable):
        res = []
        for x in model:
            val, flat = _unflatten(flat, x)
            res.append(val)
        return res, flat
    elif isinstance(model, numbers.Number):
        return flat[0], flat[1:]
    else:
        raise TypeError('Unsupported type in the model: {}'.format(type(model)))


def unflatten(flat, model):
    """Wrapper for :func:`_unflatten`.
    """
    res, tail = _unflatten(np.asarray(flat), model)
    if len(tail) != 0:
        raise ValueError('Flattened iterable has more elements than the model.')
    return res



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
            return qm.expectation.Z(0)

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
                return qm.expectation.Z(0)

            result = my_quantum_function(np.pi/4)

    Args:
        func (qfunc): a Python function containing OpenQML quantum operations,
            one per line, ending with an expectation value.
        device (openqml.Device): an OpenQML-compatible device.
    """
    _current_context = None  #: QNode: for building Operation sequences by executing qfuncs

    def __init__(self, func, device):
        self.func = func
        self.device = device
        self.variable_ops = {}  #: dict[int->list[(Operation, int)]]: Mapping from free parameter index to the list of Operations (in this circuit) that depend on it. The second element of the tuple is the index of the parameter within the Operation.


    def construct(self, args, **kwargs):
        """Constructs the quantum circuit, and creates the variable mapping.

        args represent the free parameters passed to the circuit.
        Here we are not concerned with their values, but their structure.
        Each free param is replaced with a Variable instance.

        .. note:: kwargs are assumed to not be variables by default; should we change this?

        Returns:
        """
        self.variable_ops = {}
        self._queue   = []
        self._observe = []  # TODO remove, combine with _queue

        # flatten the args, replace each with a Variable instance with a unique index
        temp = [Variable(idx) for idx, val in enumerate(_flatten(args))]
        self.num_variables = len(temp)

        # arrange the newly created Variables in the nested structure of args
        variables = unflatten(temp, args)

        # set up the context for Operation entry
        if QNode._current_context is None:
            QNode._current_context = self
        else:
            raise QuantumFunctionError('Should not happen.')
        # generate the program queue by executing the qfunc
        try:
            res = self.func(*variables, **kwargs)
        finally:
            # remove the context
            QNode._current_context = None

        # qfunc return validation
        if isinstance(res, openqml.operation.Expectation):
            self.output_type = float
            self.output_dim = 1
            res = [res]
        elif isinstance(res, collections.Sequence):
            # for multiple expectation values, we only support tuples or lists.
            self.output_dim = len(res)
            if isinstance(res, tuple):
                self.output_type = tuple
            elif isinstance(res, list):  # FIXME
                self.output_type = list
        else:
            raise QuantumFunctionError("A quantum function must return either a single expectation value or a list/tuple of expectation values.")

        self.out_wires = [ex.wires[0] for ex in res]  # FIXME for now only 1-wire ev:s


        # check that no wires are measured twice
        measured_wires = [ex.wires[0] for ex in self._observe]
        if len(measured_wires) != len(set(measured_wires)):
            raise QuantumFunctionError('Each wire in the quantum circuit can only be measured once.')

        # TODO remove when _observe is removed
        self._queue.extend(self._observe)
        del self._observe

        # map each free variable to the operations which depend on it
        for op in self._queue:
            for idx, p in enumerate(op.params):
                if isinstance(p, Variable):
                    self.variable_ops.setdefault(p.idx, []).append((op, idx))

        # map from free parameter index to the gradient method to be used with that parameter
        self.grad_method = {k: self.best_method(v) for k, v in self.variable_ops.items()}


    def __call__(self, *args, **kwargs):
        """Wrapper."""
        args = autograd.builtins.tuple(args)  # prevents autograd boxed arguments from going through to evaluate
        return self.evaluate(args, **kwargs)  # args as one tuple

    @ae.primitive
    def evaluate(self, args, **kwargs):
        """Evaluates the quantum function on the specified device, and returns
        the output expectation value."""

        #if len(args) == 1 and isinstance(args[0], np.ndarray):
            # HACK: args are being passed as a list, try unpacking arguments
            # Should be a better way to deal with autograd passing the arguments
            # in the form (np.array([...]), )
        #    args = args[0]

        if not self.variable_ops:
            # construct the circuit
            self.construct(args, **kwargs)

        # store the new free parameter values in the Variable class
        Variable.free_param_values = np.array(list(_flatten(args)))

        self.device.reset()
        ret = self.device.execute(self._queue)
        return ret[self.out_wires]


    #@property
    #def expectation(self):
    #    """Expectation value of the QNode"""
    #    if isinstance(self.output_type(), float):
    #        return self.device._out[0]
    #    return self.output_type(self.device._out)

    @staticmethod
    def best_method(ops):
        """Determine the correct gradient computation method for each free parameter.

        Args:
          ops (list[(Operation, int)]): Operations that depend on this free parameter
        Returns:
          str: gradient method to be used
        """
        # use the angle method iff every gate that depends on the parameter supports it
        if all(op.grad_method == 'A' for op, _ in ops):
            return 'A'
        return 'F'


    def gradient(self, params, which=None, *, method='B', h=1e-7, order=1, **kwargs):
        """Compute the gradient (or Jacobian) of the node.

        Returns the gradient of the parametrized quantum circuit encapsulated in the QNode.

        The gradient can be computed using several methods:

        * Finite differences (``'F'``). The first order method evaluates the circuit at
          n+1 points of the parameter space, the second order method at 2n points,
          where n = len(which).
        * Angular method (``'A'``). Works for all one-parameter gates where the generator
          only has two unique eigenvalues. Additionally can be used in CV systems for gaussian
          circuits containing only first-order observables.The circuit is evaluated twice for each incidence
          of each parameter in the circuit.
        * Best known method for each parameter (``'B'``): uses the angular method if
          possible, otherwise finite differences.

        .. note::
           The finite difference method cannot tolerate any statistical noise in the circuit output,
           since it compares the output at two points infinitesimally close to each other. Hence the
           'F' method requires exact expectation values, i.e. `shots=0`.

        Args:
            params (nested): point in parameter space at which to evaluate the gradient
            which  (Sequence[int], None): return the gradient with respect to these parameters.
                None means all.
            method (str): gradient computation method, see above

        Keyword Args:
            h (float): finite difference method step size
            order (int): finite difference method order, 1 or 2
            shots (int): How many times should the circuit be evaluated (or sampled) to estimate
                the expectation values. For simulator backends, zero yields the exact result.

        Returns:
            array[float]: gradient vector/Jacobian matrix, shape == (n_out, len(which))
        """
        flat_params = np.array(list(_flatten(params)))
        #print('grad params:', flat_params)

        if which is None:
            which = range(len(flat_params))
        elif len(which) != len(set(which)):  # set removes duplicates
            raise ValueError('Parameter indices must be unique.')

        if not self.variable_ops:
            # construct the circuit
            self.construct(params, **kwargs)

        if method in ('A', 'F'):
            method = {k: method for k in which}
        elif method == 'B':
            method = self.grad_method

        if 'F' in method.values():
            if order == 1:
                # the value of the circuit at params, computed only once here
                y0 = np.asarray(self.evaluate(flat_params, **kwargs))
            else:
                y0 = None

        # compute the partial derivative w.r.t. each parameter using the proper method
        grad = np.zeros((len(which), self.output_dim), dtype=float)

        for i, k in enumerate(which):
            if k not in self.variable_ops:
                # unused parameter
                continue
            par_method = method[k]
            if par_method == 'A':
                grad[i, :] = self._pd_angle(flat_params, k, **kwargs)
            elif par_method == 'F':
                grad[i, :] = self._pd_finite_diff(flat_params, k, h, order, y0, **kwargs)
            else:
                raise ValueError('Unknown gradient method.')

        return grad  #grad.flatten() # FIXME


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
        shift_params = params.copy()
        if order == 1:
            # shift one parameter by h
            shift_params[idx] += h
            y = np.asarray(self.evaluate(shift_params, **kwargs))
            return (y-y0) / h
        elif order == 2:
            # symmetric difference
            # shift one parameter by +-h/2
            shift_params[idx] += 0.5*h
            y2 = np.asarray(self.evaluate(shift_params, **kwargs))
            shift_params[idx] = params[idx] -0.5*h
            y1 = np.asarray(self.evaluate(shift_params, **kwargs))
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
        n = self.num_variables
        pd = 0.0
        # find the Commands in which the parameter appears, use the product rule
        for op, p_idx in self.variable_ops[idx]:
            if op.grad_method != 'A':
                raise ValueError('Attempted to use the angular method on a gate that does not support it.')

            # we temporarily edit the Command such that parameter p_idx is replaced by a new one,
            # which we can modify without affecting other Commands depending on the original.
            orig = op.params[p_idx]
            assert(orig.idx == idx)

            # reference to a new, temporary parameter with index n, otherwise identical with orig
            temp_var = copy.copy(orig)
            temp_var.idx = n
            op.params[p_idx] = temp_var

            # we just need to add something to the map, it's not actually used
            self.variable_ops[n] = [[op, p_idx]]
            #self.vars.append(temp_var)

            # get the gradient recipe for this parameter
            recipe = op.grad_recipe[p_idx]
            multiplier = 0.5 if recipe is None else recipe[0]
            multiplier *= orig.mult

            # shift the temp parameter value by +- this amount
            shift = np.pi / 2 if recipe is None else recipe[1]
            shift /= orig.mult

            shift_params = np.r_[params, params[idx] + shift]
            y2 = np.asarray(self.evaluate(shift_params, **kwargs))
            shift_params[-1] = params[idx] - shift
            y1 = np.asarray(self.evaluate(shift_params, **kwargs))

            # restore the original parameter
            op.params[p_idx] = orig
            # remove the temporary entry
            del self.variable_ops[n]
            #del self.vars[n]
            pd += (y2-y1) * multiplier

        return pd


#def QNode_vjp(ans, self, params, *args, **kwargs):
def QNode_vjp(ans, self, args, **kwargs):
    """Returns the vector Jacobian product for a QNode, as a function
    of the QNode evaluation at the specified parameter values.
    """
    #if isinstance(params, numbers.Number):
    #    params = [params]
    #p = list(params) + list(args)
    return lambda g: g * self.gradient(args, **kwargs)


# define the vector-Jacobian product function for QNode.__call__()
ae.defvjp(QNode.evaluate, QNode_vjp, argnums=[1])
