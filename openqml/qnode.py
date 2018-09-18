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
Quantum nodes
=============

**Module name:** :mod:`openqml.qnode`

.. currentmodule:: openqml

The :class:`~qnode.QNode` class encapsulates a quantum circuit and the computational device it is executed on.
The computational device is an instance of the :class:`~device.Device`
class, and can represent either a simulator or hardware device.
Additional devices can be installed as plugins.

The quantum circuit is described using a quantum function (qfunc), which must be of the following form:

.. code-block:: python

    def my_quantum_function(x, y):
        qm.Zrotation(x, 0)
        qm.CNOT(0,1)
        qm.Yrotation(y**2, 1)
        return qm.expectation.Z(0)

The body of the qfunc must consist of only :class:`~operation.Operation` constructor calls, and must return
a tuple of :class:`~operation.Expectation` instances (or just a single instance).

.. note:: The Operation instances must be constructed in the qfunc, in the correct order, because Operation.__init__ does the queueing!

.. note:: Expectation values must come after the other operations at the end of the function.

Once the device and qfunc are defined, the QNode can then be used
to evaluate the quantum function on the particular device.
For example,

.. code-block:: python

    device = qm.device('strawberryfields.fock', cutoff=5)
    qnode1 = QNode(my_quantum_function, device)
    result = qnode1(np.pi/4)

.. note::

        The :func:`~openqml.qfunc` decorator is provided as a convenience
        to automate the process of creating quantum nodes. Using this decorator,
        the above example becomes:

        .. code-block:: python

            @qfunc(device)
            def my_quantum_function(x, y):
                qm.Zrotation(x, 0)
                qm.CNOT(0,1)
                qm.Yrotation(y**2, 1)
                return qm.expectation.Z(0)

            result = my_quantum_function(np.pi/4)


.. currentmodule:: openqml.qnode

Functions
---------

.. autosummary::
   _flatten
   _unflatten
   unflatten


QNode methods
-------------

.. currentmodule:: openqml.qnode.QNode

.. autosummary::
   construct
   __call__
   evaluate
   best_method
   gradient
   _pd_finite_diff
   _pd_analytic

.. currentmodule:: openqml.qnode

----
"""

import copy
import collections

import logging as log
log.getLogger()

import numbers

import autograd.numpy as np
import autograd.extend as ae
import autograd.builtins

import openqml.operation

from .device import QuantumFunctionError
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
    """Quantum node in the hybrid computational graph.


    Args:
        func (callable): a Python function containing :class:`~.operation.Operation` constructor calls,
          returning a tuple of :class:`~.operation.Expectation` instances.
        device (~.device.Device): device to execute the function on
    """
    _current_context = None  #: QNode: for building Operation sequences by executing qfuncs

    def __init__(self, func, device):
        self.func = func
        self.device = device
        self.variable_ops = {}  #: dict[int->list[(int, int)]]: Mapping from free parameter index to the list of Operations (in this circuit) that depend on it. The first element of the tuple is the index of Operation in the program queue, the second the index of the parameter within the Operation.


    def construct(self, args, **kwargs):
        """Constructs a representation of the quantum circuit.

        The user should never have to call this method.
        Called automatically the first time :meth:`QNode.__call__`, :meth:`QNode.evaluate` or :meth:`QNode.gradient` is called.
        Executes the quantum function, stores the resulting sequence of :class:`~.operation.Operation` instances, and creates the variable mapping.

        Args:
          args (tuple): Represent the free parameters passed to the circuit.
            Here we are not concerned with their values, but with their structure.
            Each free param is replaced with a :class:`~.variable.Variable` instance.

        .. note:: kwargs are assumed to not be variables by default; should we change this?
        """
        self.variable_ops = {}
        self._queue   = []
        self._observe = []

        # flatten the args, replace each with a Variable instance with a unique index
        temp = [Variable(idx) for idx, val in enumerate(_flatten(args))]
        self.num_variables = len(temp)

        # arrange the newly created Variables in the nested structure of args
        variables = unflatten(temp, args)

        # set up the context for Operation entry
        if QNode._current_context is None:
            QNode._current_context = self
        else:
            raise QuantumFunctionError('Should not happen, QNode._current_context must not be modified outside this method.')
        # generate the program queue by executing the qfunc
        try:
            res = self.func(*variables, **kwargs)
        finally:
            # remove the context
            QNode._current_context = None

        #----------------------------------------------------------
        # check the validity of the circuit

        # qfunc return validation
        if isinstance(res, openqml.operation.Expectation):
            self.output_type = float
            self.output_dim = 1
            res = [res]
        elif isinstance(res, tuple):
            # for multiple expectation values, we only support tuples.
            self.output_dim = len(res)
            self.output_type = np.asarray
        else:
            raise QuantumFunctionError("A quantum function must return either a single expectation value or a tuple of expectation values.")

        def check_op(op):
            # make sure only existing wires are referenced
            for w in op.wires:
                if w < 0 or w >= self.device.wires:
                    raise QuantumFunctionError('Operation {} applied to wire {}, device only has {}.'.format(op.name, w, self.device.wires))

        # check every gate/preparation and ev measurement
        for op in self._queue + res:
            check_op(op)

        # check that all ev:s are returned
        if set(res) != set(self._observe):
            raise QuantumFunctionError('All measured expectation values must be returned.')

        # check that no wires are measured more than once
        m_wires = list(w for ex in res for w in ex.wires)
        if len(m_wires) != len(set(m_wires)):
            raise QuantumFunctionError('Each wire in the quantum circuit can only be measured once.')

        # TODO ensure that the gates precede every measurement

        #----------------------------------------------------------

        self.ex = res  #: tuple[Expectation]: returned expectation values

        # map each free variable to the operations which depend on it
        for k, op in enumerate(self._queue):
            for idx, p in enumerate(op.params):
                if isinstance(p, Variable):
                    self.variable_ops.setdefault(p.idx, []).append((k, idx))

        # map from free parameter index to the gradient method to be used with that parameter
        self.grad_method = {k: self.best_method(v) for k, v in self.variable_ops.items()}


    def __call__(self, *args, **kwargs):
        """Wrapper for :meth:`QNode.evaluate`.
        """
        args = autograd.builtins.tuple(args)  # prevents autograd boxed arguments from going through to evaluate
        return self.evaluate(args, **kwargs)  # args as one tuple


    @ae.primitive
    def evaluate(self, args, **kwargs):
        """Evaluates the quantum function on the specified device.

        Args:
          args (tuple): input parameters to the quantum function

        Returns:
          float, array[float]: output expectation value(s)
        """
        if not self.variable_ops:
            # construct the circuit
            self.construct(args, **kwargs)

        # temporarily store the free parameter values in the Variable class
        Variable.free_param_values = np.array(list(_flatten(args)))

        self.device.reset()
        ret = self.device.execute(self._queue, self._observe)
        return self.output_type(ret)


    def best_method(self, ops):
        """Determine the correct gradient computation method for each free parameter.

        Args:
          ops (list[(int, int)]): Operations that depend on this free parameter
        Returns:
          str: gradient method to be used
        """
        # use the analytic method iff every gate that depends on the parameter supports it
        # TODO FIXME not enough in CV case if nongaussian gates follow them.
        if all(self._queue[k].grad_method == 'A' for k, _ in ops):
            return 'A'
        return 'F'


    def gradient(self, params, which=None, *, method='B', h=1e-7, order=1, **kwargs):
        """Compute the gradient (or Jacobian) of the node.

        Returns the gradient of the parametrized quantum circuit encapsulated in the QNode.

        The gradient can be computed using several methods:

        * Finite differences (``'F'``). The first order method evaluates the circuit at
          n+1 points of the parameter space, the second order method at 2n points,
          where n = len(which).
        * Analytic method (``'A'``). Works for all one-parameter gates where the generator
          only has two unique eigenvalues. Additionally can be used in CV systems for gaussian
          circuits containing only first-order observables.The circuit is evaluated twice for each incidence
          of each parameter in the circuit.
        * Best known method for each parameter (``'B'``): uses the analytic method if
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
                the expectation values. For simulator backends, 0 yields the exact result.

        Returns:
            array[float]: gradient vector/Jacobian matrix, shape == (n_out, len(which))
        """
        flat_params = np.array(list(_flatten(params)))

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
                grad[i, :] = self._pd_analytic(flat_params, k, order, **kwargs)
            elif par_method == 'F':
                grad[i, :] = self._pd_finite_diff(flat_params, k, h, order, y0, **kwargs)
            else:
                raise ValueError('Unknown gradient method.')

        if self.output_dim == 1:
            return grad[:, 0]

        return grad.T

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


    def _pd_analytic(self, params, idx, order=1, **kwargs):
        """Partial derivative of the node using the analytic method.

        .. todo:: Detect non-gaussian gates in CV circuits and raise an exception if they are differentiated or succeed a differentiated gate (since in these cases the formula is invalid).

        The 2nd order method can handle also first order observables, but 1st order method may be more efficient unless it's really easy to experimentally measure arbitrary 2nd order observables.

        Args:
          params (array[float]): point in parameter space at which to evaluate
            the partial derivative.
          idx (int): return the partial derivative with respect to this free parameter.

        Returns:
          float: partial derivative of the node.
        """
        n = self.num_variables
        pd = 0.0
        # find the Commands in which the free parameter appears, use the product rule
        for op_idx, p_idx in self.variable_ops[idx]:
            op = self._queue[op_idx]
            if op.grad_method != 'A':
                raise ValueError('Attempted to use the analytic method on a gate that does not support it.')

            # we temporarily edit the Operation such that parameter p_idx is replaced by a new one,
            # which we can modify without affecting other Operations depending on the original.
            orig = op.params[p_idx]
            assert orig.idx == idx

            # reference to a new, temporary parameter with index n, otherwise identical with orig
            temp_var = copy.copy(orig)
            temp_var.idx = n
            op.params[p_idx] = temp_var

            # get the gradient recipe for this parameter
            recipe = op.grad_recipe[p_idx]
            multiplier = 0.5 if recipe is None else recipe[0]
            multiplier *= orig.mult

            # shift the temp parameter value by +- this amount
            shift = np.pi / 2 if recipe is None else recipe[1]
            shift /= orig.mult

            if order == 1:
                # evaluate the circuit in two points with shifted parameter values
                shift_params = np.r_[params, params[idx] + shift]
                y2 = np.asarray(self.evaluate(shift_params, **kwargs))
                shift_params[-1] = params[idx] - shift
                y1 = np.asarray(self.evaluate(shift_params, **kwargs))
                pd += (y2-y1) * multiplier
            elif order == 2:
                # first build the Z transformation matrix
                Variable.free_param_values = np.r_[params, params[idx] + shift]
                Z2 = op.heisenberg_transform()
                Variable.free_param_values[-1] = params[idx] - shift
                Z1 = op.heisenberg_transform()
                unshifted_params = np.r_[params, params[idx]]
                Variable.free_param_values[-1] = unshifted_params
                Z0 = op.heisenberg_transform()
                Z = (Z2-Z1) * multiplier  # derivative of the operation
                Z = Z @ np.linalg.inv(Z0)  # FIXME maybe replace inv with h-transform evaluated at -par[0] ?
                # conjugate with all the following operations
                B = np.eye(3)
                for BB in self._queue[op_idx+1:]:
                    temp = BB.heisenberg_transform()
                    B = temp @ B
                Z = B @ Z @ np.linalg.inv(B)  # conjugation

                def tr_obs(ex):
                    "Transform the observable, compute its expectation value."
                    q = ex.heisenberg_expand()
                    temp = q @ Z
                    if q.ndim == 2:
                        # 2nd order observable
                        temp = temp +temp.T.conj()
                    return self.evaluate_obs(temp, unshifted_params, **kwargs)  # TODO needs to be implemented

                # measure transformed observables
                temp = np.asarray(list(map(tr_obs, self.ex)))
                pd += temp
            else:
                raise ValueError('Order must be 1 or 2.')

            # restore the original parameter
            op.params[p_idx] = orig

        return pd




#def QNode_vjp(ans, self, params, *args, **kwargs):
def QNode_vjp(ans, self, args, **kwargs):
    """Returns the vector Jacobian product operator for a QNode, as a function
    of the QNode evaluation for specific argnums at the specified parameter values.
    """
    def gradient_product(g):
        """Vector Jacobian product operator"""
        if len(g.shape) == 0:
            if len(args) == 1 and isinstance(args[0], np.ndarray):
                return [self.gradient(args, **kwargs)]

            return g * self.gradient(args, **kwargs)

        if len(args) == 1 and isinstance(args[0], np.ndarray):
            # This feels hacky, but is required if the argument
            # is a single np.ndarray
            return [g] @ self.gradient(args, **kwargs)

        return g @ self.gradient(args, **kwargs)

    return gradient_product


# define the vector-Jacobian product function for QNode.__call__()
ae.defvjp(QNode.evaluate, QNode_vjp, argnums=[1])
