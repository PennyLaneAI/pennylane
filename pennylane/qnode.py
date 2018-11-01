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
# pylint: disable=cell-var-from-loop,attribute-defined-outside-init,too-many-branches,too-many-arguments
"""
The quantum node object
=======================

**Module name:** :mod:`pennylane.qnode`

.. currentmodule:: pennylane

The :class:`~qnode.QNode` class is used to construct quantum nodes,
encapsulating a :ref:`quantum function <qfuncs>` and the computational
device it is executed on.

The computational device is an instance of the :class:`~_device.Device`
class, and can represent either a simulator or hardware device. They can
instantiated using the :func:`~device` loader. PennyLane comes included with
some basic devices; additional devices can be installed as plugins
(see :ref:`plugins` for more details).

Qfuncs
------

The quantum function (qfunc) encapsulated by the QNode must be of the following form:

.. code-block:: python

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(-2*y, wires=1)
        return qml.expval.PauliZ(0)

Qfuncs are a restricted subset of Python functions, adhering to the following
constraints:

* The body of the qfunc must consist of only supported PennyLane
  :mod:`operations <pennylane.ops>`, one per line.

* The qfunc must always return either a single or a tuple of
  :mod:`expectation values <pennylane.expval>`.

* Classical processing of function arguments, either by arithmetic operations
  or external functions, is not allowed. One current exception is simple scalar
  multiplication.

.. note::

    The quantum operations cannot be used outside of a qfunc, as the
    :class:`~.Operation` requires a QNode to perform the queuing on initialization.

.. note::

    Expectation values **must** come after all other operations at the end
    of the qfunc as part of the return statement, and cannot appear in the middle.

Once the device and qfunc are defined, the QNode can then be used
to evaluate the quantum function on the particular device.

For example,

.. code-block:: python

    device = qml.device('default.qubit', wires=1)
    qnode1 = QNode(my_quantum_function, device)
    result = qnode1(np.pi/4, 0.7)

.. note::

        The :func:`~pennylane.decorator.qnode` decorator is provided as a convenience
        to automate the process of creating quantum nodes. Using this decorator,
        the above example becomes:

        .. code-block:: python

            @qml.qnode(device)
            def my_quantum_function(x, y):
                qml.RZ(x, wires=0)
                qml.CNOT(wires=[0,1])
                qml.RY(-2*y, wires=1)
                return qml.expval.PauliZ(0)

            result = my_quantum_function(np.pi/4)


.. currentmodule:: pennylane.qnode

Auxiliary functions
-------------------

.. autosummary::
   _inv_dict
   _get_default_args


QNode methods
-------------

.. currentmodule:: pennylane.qnode.QNode

.. autosummary::
   __call__
   evaluate
   evaluate_obs
   jacobian
   var

QNode internal methods
----------------------

.. autosummary::
   construct
   _best_method
   _append_op
   _op_successors
   _pd_finite_diff
   _pd_analytic

.. currentmodule:: pennylane.qnode

Code details
~~~~~~~~~~~~
"""
import inspect
import copy

import logging as log

import numbers

import autograd.numpy as np
import autograd.extend as ae
import autograd.builtins

import pennylane.operation
from .variable  import Variable
from .utils import _flatten, unflatten


log.getLogger()


class QuantumFunctionError(Exception):
    """Exception raised when an illegal operation is defined in a quantum function."""
    pass


def _inv_dict(d):
    """Reverse a dictionary mapping.

    Returns multimap where the keys are the former values,
    and values are sets of the former keys.

    Args:
        d (dict[a->b]): mapping to reverse

    Returns:
        dict[b->set[a]]: reversed mapping
    """
    ret = {}
    for k, v in d.items():
        ret.setdefault(v, set()).add(k)
    return ret


def _get_default_args(func):
    """Get the default arguments of a function.

    Args:
        func (function): a valid Python function

    Returns:
        dict: dictionary containing the argument name and tuple
        (positional idx, default value)
    """
    signature = inspect.signature(func)
    return {
        k: (idx, v.default)
        for idx, (k, v) in enumerate(signature.parameters.items())
        if v.default is not inspect.Parameter.empty
    }


class QNode:
    """Quantum node in the hybrid computational graph.

    Args:
        func (callable): a Python function containing :class:`~.operation.Operation`
            constructor calls, returning a tuple of :class:`~.operation.Expectation` instances.
        device (~pennylane._device.Device): device to execute the function on
    """
    # pylint: disable=too-many-instance-attributes
    _current_context = None  #: QNode: for building Operation sequences by executing qfuncs

    def __init__(self, func, device):
        self.func = func
        self.device = device
        self.num_wires = device.num_wires
        self.ops = []

        self.variable_ops = {}
        """ dict[int->list[(int, int)]]: Mapping from free parameter index to the list of
        Operations (in this circuit) that depend on it.

        The first element of the tuple is the index of Operation in the program queue,
        the second the index of the parameter within the Operation.
        """

    def _append_op(self, op):
        """Appends a quantum operation into the circuit queue.

        Args:
            op (Operation): quantum operation to be added to the circuit
        """
        # EVs go to their own, temporary queue
        if isinstance(op, pennylane.operation.Expectation):
            self.ev.append(op)
        else:
            if self.ev:
                raise QuantumFunctionError('State preparations and gates must precede expectation values.')
            self.queue.append(op)

    def construct(self, args, **kwargs):
        """Constructs a representation of the quantum circuit.

        The user should never have to call this method.

        This method is called automatically the first time :meth:`QNode.evaluate`
        or :meth:`QNode.jacobian` is called. It executes the quantum function,
        stores the resulting sequence of :class:`~.operation.Operation` instances,
        and creates the variable mapping.

        Args:
            args (tuple): Represent the free parameters passed to the circuit.
                Here we are not concerned with their values, but with their structure.
                Each free param is replaced with a :class:`~.variable.Variable` instance.

        .. note::

            Additional keyword arguments may be passed to the qfunc, however PennyLane
            does not support differentiating with respect to keyword arguments. Instead,
            keyword arguments are useful for providing data or 'placeholders' to the qfunc.
        """
        # pylint: disable=too-many-branches
        self.queue = []
        self.ev = []  # temporary queue for EVs

        # flatten the args, replace each with a Variable instance with a unique index
        temp = [Variable(idx) for idx, val in enumerate(_flatten(args))]
        self.num_variables = len(temp)

        # arrange the newly created Variables in the nested structure of args
        variables = unflatten(temp, args)

        # get default kwargs that weren't passed
        keyword_sig = _get_default_args(self.func)
        self.keyword_defaults = {k: v[1] for k, v in keyword_sig.items()}
        self.keyword_positions = {v[0]: k for k, v in keyword_sig.items()}

        keyword_values = {}
        keyword_values.update(self.keyword_defaults)
        keyword_values.update(kwargs)

        # warp each keyword argument as a Variable
        kwarg_variables = {}
        for key, val in keyword_values.items():
            temp = [Variable(idx, name=key) for idx, _ in enumerate(_flatten(val))]
            kwarg_variables[key] = unflatten(temp, val)

        # set up the context for Operation entry
        if QNode._current_context is None:
            QNode._current_context = self
        else:
            raise QuantumFunctionError('QNode._current_context must not be modified outside this method.')
        # generate the program queue by executing the qfunc
        try:
            res = self.func(*variables, **kwarg_variables)
        finally:
            # remove the context
            QNode._current_context = None

        #----------------------------------------------------------
        # check the validity of the circuit

        # qfunc return validation
        if isinstance(res, pennylane.operation.Expectation):
            self.output_type = float
            self.output_dim = 1
            res = (res,)
        elif isinstance(res, tuple) and res and all(isinstance(x, pennylane.operation.Expectation) for x in res):
            # for multiple expectation values, we only support tuples.
            self.output_dim = len(res)
            self.output_type = np.asarray
        else:
            raise QuantumFunctionError("A quantum function must return either a single expectation "
                                       "value or a nonempty tuple of expectation values.")

        # check that all ev:s are returned, in the correct order
        if res != tuple(self.ev):
            raise QuantumFunctionError("All measured expectation values must be returned in the "
                                       "order they are measured.")

        self.ev = res  #: tuple[Expectation]: returned expectation values

        # check that no wires are measured more than once
        m_wires = list(w for ex in self.ev for w in ex.wires)
        if len(m_wires) != len(set(m_wires)):
            raise QuantumFunctionError('Each wire in the quantum circuit can only be measured once.')

        self.ops = self.queue + list(self.ev)  #: list[Operation]: combined list of circuit operations

        def check_op(op):
            """Make sure only existing wires are referenced."""
            for w in op.wires:
                if w < 0 or w >= self.num_wires:
                    raise QuantumFunctionError("Operation {} applied to wire {}, "
                                               "device only has {}.".format(op.name, w, self.num_wires))

        # check every gate/preparation and ev measurement
        for op in self.ops:
            check_op(op)

        # classify the circuit contents
        temp = [isinstance(op, pennylane.operation.CV) for op in self.ops]
        if all(temp):
            self.type = 'CV'
        elif not True in temp:
            self.type = 'discrete'
        else:
            raise QuantumFunctionError("Continuous and discrete operations are not "
                                       "allowed in the same quantum circuit.")

        #----------------------------------------------------------

        # map each free variable to the operations which depend on it
        self.variable_ops = {}
        for k, op in enumerate(self.ops):
            for idx, p in enumerate(_flatten(op.params)):
                if isinstance(p, Variable):
                    if p.name is None: # ignore keyword arguments
                        self.variable_ops.setdefault(p.idx, []).append((k, idx))

        #: dict[int->str]: map from free parameter index to the gradient method to be used with that parameter
        self.grad_method_for_par = {k: self._best_method(k) for k in self.variable_ops}

    def _op_successors(self, o_idx, only='G'):
        """Successors of the given Operation in the quantum circuit.

        Args:
            o_idx (int): index of the operation in the operation queue
            only (str): the type of successors to return.

                - ``'G'``: only return non-Expectations (default)
                - ``'E'``: only return Expectations
                - ``None``: return all successors

        Returns:
            list[Operation]: successors in a topological order
        """
        succ = self.ops[o_idx+1:]
        # TODO at some point we may wish to upgrade to a DAG description of the circuit instead
        # of a simple queue, in which case,
        #
        #   succ = nx.dag.topological_sort(self.DAG.subgraph(nx.dag.descendants(self.DAG, op)).copy())
        #
        # or maybe just
        #
        #   succ = nx.dfs_preorder_nodes(self.DAG, op)
        #
        # if it is in a topological order? the docs aren't clear.
        if only == 'E':
            return list(filter(lambda x: isinstance(x, pennylane.operation.Expectation), succ))
        elif only == 'G':
            return list(filter(lambda x: not isinstance(x, pennylane.operation.Expectation), succ))
        return succ

    def _best_method(self, idx):
        """Determine the correct gradient computation method for a free parameter.

        Use the analytic method iff every gate that depends on the parameter supports it.
        If not, use the finite difference method.

        Note that If even one gate does not support differentiation, we cannot differentiate
        with respect to this parameter at all.

        Args:
            idx (int): free parameter index
        Returns:
            str: gradient method to be used
        """
        # TODO: For CV circuits, when the circuit DAG is implemented, determining which gradient
        # method to use for should work like this:
        #
        # 1. To check whether we can use the 'A' or 'A2' method, we need first to check for the
        #    presence of non-Gaussian ops and order-2 observables.
        #
        # 2. Starting from the expectation values (all leaf nodes under current limitations on
        #    observables, see :ref:`measurements`), walk through the DAG against the edges
        #    (upstream) in arbitrary order.
        #
        # 3. If the starting leaf is an order-2 EV, mark every Gaussian operation you hit with
        #    op.grad_method='A2' (instance variable, does not mess up the class variable!).
        #
        # 4. If you hit a non-Gaussian gate (grad_method != 'A'), from that gate upstream mark every
        #    gaussian operation with op.grad_method='F'.
        #
        # 5. Then run the standard discrete-case algorithm for determining the best gradient method
        # for every free parameter.
        def best_for_op(o_idx):
            "Returns the best gradient method for the operation op."
            op = self.ops[o_idx]
            # for discrete operations, other ops do not affect the choice
            if not isinstance(op, pennylane.operation.CV):
                return op.grad_method

            # for CV ops it is more complicated
            if op.grad_method == 'A':
                # op is Gaussian and has the heisenberg_* methods
                # check that all successor ops are also Gaussian
                # TODO when we upgrade to a DAG: a non-Gaussian successor is OK if it
                # isn't succeeded by any observables?
                successors = self._op_successors(o_idx, 'G')

                if all(x.grad_method == 'A' for x in successors):
                    # check successor EVs, if any order-2 observables are found return 'A2', else return 'A'
                    ev_successors = self._op_successors(o_idx, 'E')
                    for x in ev_successors:
                        if x.ev_order is None:
                            return 'F'
                        if x.ev_order == 2:
                            op.grad_method = 'A2'  # bit of a hack
                    return 'A'

                return 'F'

            return op.grad_method  # 'F' or None

        # indices of operations that depend on the free parameter idx
        ops = self.variable_ops[idx]
        temp = [best_for_op(k) for k, _ in ops]
        if all(k == 'A' for k in temp):
            return 'A'
        elif None in temp:
            return None

        return 'F'

    def __call__(self, *args, **kwargs):
        """Wrapper for :meth:`QNode.evaluate`."""
        # pylint: disable=no-member
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
        if not self.ops:
            # construct the circuit
            self.construct(args, **kwargs)

        # temporarily store keyword arguments
        keyword_values = {}
        keyword_values.update({k: np.array(list(_flatten(v))) for k, v in self.keyword_defaults.items()})
        keyword_values.update({k: np.array(list(_flatten(v))) for k, v in kwargs.items()})

        # Try and insert kwargs-as-positional back into the kwargs dictionary.
        # NOTE: this works, but the creation of new, temporary arguments
        # by pd_analytic breaks this.
        # positional = []
        # kwargs_as_position = {}
        # for idx, v in enumerate(args):
        #     if idx not in self.keyword_positions:
        #     positional.append(v)
        #     else:
        #         kwargs_as_position[self.keyword_positions[idx]] = np.array(list(_flatten(v)))
        # keyword_values.update(kwargs_as_position)

        # temporarily store the free parameter values in the Variable class
        Variable.free_param_values = np.array(list(_flatten(args)))
        Variable.kwarg_values = keyword_values

        self.device.reset()
        ret = self.device.execute(self.queue, self.ev)
        return self.output_type(ret)

    def var(self, *args, **kwargs):
        """Evaluates the quantum function on the specified device, and
        returns the variance of the expectation values.

        Args:
            args (tuple): input parameters to the quantum function

        Returns:
            float, array[float]: output variance value(s)
        """
        # pylint: disable=no-member
        args = autograd.builtins.tuple(args)

        if not self.ops:
            # construct the circuit
            self.construct(args, **kwargs)

        # temporarily store keyword arguments
        keyword_values = {}
        keyword_values.update({k: np.array(list(_flatten(v))) for k, v in self.keyword_defaults.items()})
        keyword_values.update({k: np.array(list(_flatten(v))) for k, v in kwargs.items()})

        # temporarily store the free parameter values in the Variable class
        Variable.free_param_values = np.array(list(_flatten(args)))
        Variable.kwarg_values = keyword_values

        self.device.reset()
        ret = self.device.execute(self.queue, self.ev, variance=True)
        return self.output_type(ret)

    def evaluate_obs(self, obs, args, **kwargs):
        """Evaluate the expectation values of the given observables.

        Assumes :meth:`construct` has already been called.

        Args:
            obs  (Iterable[Expectation]): observables to measure
            args (array[float]): circuit input parameters

        Returns:
            array[float]: expectation values
        """
        # temporarily store the free parameter values in the Variable class
        Variable.free_param_values = args

        self.device.reset()
        ret = self.device.execute(self.queue, obs)
        return ret

    def jacobian(self, params, which=None, *, method='B', h=1e-7, order=1, **kwargs):
        """Compute the Jacobian of the QNode.

        Returns the Jacobian of the parametrized quantum circuit encapsulated in the QNode.

        The Jacobian can be computed using several methods:

        * Finite differences (``'F'``). The first order method evaluates the circuit at
          :math:`n+1` points of the parameter space, the second order method at :math:`2n` points,
          where ``n = len(which)``.

        * Analytic method (``'A'``). Works for all one-parameter gates where the generator
          only has two unique eigenvalues; this includes one-parameter qubit gates, as well as
          Gaussian circuits of order one or two. Additionally, can be used in CV systems for Gaussian
          circuits containing first- and second-order observables.

          The circuit is evaluated twice for each incidence of each parameter in the circuit.

        * Best known method for each parameter (``'B'``): uses the analytic method if
          possible, otherwise finite difference.

        .. note::
           The finite difference method is sensitive to statistical noise in the circuit output,
           since it compares the output at two points infinitesimally close to each other. Hence the
           'F' method requires exact expectation values, i.e. `shots=0`.

        Args:
            params (nested Sequence[Number], Number): point in parameter space at which
                to evaluate the gradient
            which  (Sequence[int], None): return the Jacobian with respect to these parameters.
                None (the default) means with respect to all parameters. Note that keyword
                arguments to the QNode are *always* treated as fixed values and not included
                in the Jacobian calculation.
            method (str): Jacobian computation method, see above.

        Keyword Args:
            h (float): finite difference method step size
            order (int): finite difference method order, 1 or 2
            shots (int): How many times the circuit should be evaluated (or sampled) to estimate
                the expectation values. For simulator backends, 0 yields the exact result.

        Returns:
            array[float]: Jacobian matrix, with shape ``(n_out, len(which))``, where ``len(which)`` is the
            number of free parameters, and ``n_out`` is the number of expectation values returned
            by the QNode.
        """
        # in QNode.construct we need to be able to (essentially) apply the unpacking operator to params
        if isinstance(params, numbers.Number):
            params = (params,)

        if not self.ops:
            # construct the circuit
            self.construct(params, **kwargs)

        flat_params = np.array(list(_flatten(params)))

        if which is None:
            which = range(len(flat_params))
        else:
            if min(which) < 0 or max(which) >= self.num_variables:
                raise ValueError("Tried to compute the gradient wrt. free parameters {} "
                                 "(this node has {} free parameters).".format(which, self.num_variables))
            if len(which) != len(set(which)):  # set removes duplicates
                raise ValueError("Parameter indices must be unique.")

        # check if the method can be used on the requested parameters
        mmap = _inv_dict(self.grad_method_for_par)
        def check_method(m):
            """Intersection of ``which`` with free params whose best grad method is m."""
            return mmap.get(m, set()).intersection(which)

        bad = check_method(None)
        if bad:
            raise ValueError('Cannot differentiate wrt parameter(s) {}.'.format(bad))

        if method in ('A', 'F'):
            if method == 'A':
                bad = check_method('F')
                if bad:
                    raise ValueError("The analytic gradient method cannot be "
                                     "used with the parameter(s) {}.".format(bad))
            method = {k: method for k in which}
        elif method == 'B':
            method = self.grad_method_for_par
        else:
            raise ValueError('Unknown gradient method.')

        if 'F' in method.values():
            if order == 1:
                # the value of the circuit at params, computed only once here
                y0 = np.asarray(self.evaluate(flat_params, **kwargs))
            else:
                y0 = None

        # compute the partial derivative w.r.t. each parameter using the proper method
        grad = np.zeros((self.output_dim, len(which)), dtype=float)

        for i, k in enumerate(which):
            if k not in self.variable_ops:
                # unused parameter
                continue

            par_method = method[k]
            if par_method == 'A':
                grad[:, i] = self._pd_analytic(flat_params, k, **kwargs)
            elif par_method == 'F':
                grad[:, i] = self._pd_finite_diff(flat_params, k, h, order, y0, **kwargs)
            else:
                raise ValueError('Unknown gradient method.')

        return grad

    def _pd_finite_diff(self, params, idx, h=1e-7, order=1, y0=None, **kwargs):
        """Partial derivative of the node using the finite difference method.

        Args:
            params (array[float]): point in parameter space at which to evaluate
                the partial derivative
            idx (int): return the partial derivative with respect to this parameter
            h (float): step size
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


    def _pd_analytic(self, params, idx, force_order2=False, **kwargs):
        """Partial derivative of the node using the analytic method.

        The 2nd order method can handle also first order observables, but
        1st order method may be more efficient unless it's really easy to
        experimentally measure arbitrary 2nd order observables.

        Args:
            params (array[float]): point in free parameter space at which
                to evaluate the partial derivative
            idx (int): return the partial derivative with respect to this
                free parameter

        Returns:
            float: partial derivative of the node.
        """
        n = self.num_variables
        w = self.num_wires
        pd = 0.0
        # find the Commands in which the free parameter appears, use the product rule
        for o_idx, p_idx in self.variable_ops[idx]:
            op = self.ops[o_idx]

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

            # shifted parameter values
            shift_p1 = np.r_[params, params[idx] +shift]
            shift_p2 = np.r_[params, params[idx] -shift]

            if not force_order2 and op.grad_method != 'A2':
                # basic analytic method, for discrete gates and gaussian CV gates succeeded by order-1 observables
                # evaluate the circuit in two points with shifted parameter values
                y2 = np.asarray(self.evaluate(shift_p1, **kwargs))
                y1 = np.asarray(self.evaluate(shift_p2, **kwargs))
                pd += (y2-y1) * multiplier
            else:
                # order-2 method, for gaussian CV gates succeeded by order-2 observables
                # evaluate transformed observables at the original parameter point
                # first build the Z transformation matrix
                Variable.free_param_values = shift_p1
                Z2 = op.heisenberg_tr(w)
                Variable.free_param_values = shift_p2
                Z1 = op.heisenberg_tr(w)
                Z = (Z2-Z1) * multiplier  # derivative of the operation

                unshifted_params = np.r_[params, params[idx]]
                Variable.free_param_values = unshifted_params
                Z0 = op.heisenberg_tr(w, inverse=True)
                Z = Z @ Z0

                # conjugate Z with all the following operations
                B = np.eye(1 +2*w)
                B_inv = B.copy()
                for BB in self._op_successors(o_idx, 'G'):
                    temp = BB.heisenberg_tr(w)
                    B = temp @ B
                    temp = BB.heisenberg_tr(w, inverse=True)
                    B_inv = B_inv @ temp
                Z = B @ Z @ B_inv  # conjugation

                ev_successors = self._op_successors(o_idx, 'E')

                def tr_obs(ex):
                    """Transform the observable"""
                    # TODO: At initial release, since we use a queue to represent circuit, all expectations values
                    # are successors to all gates in the same circuit.
                    # When library uses a DAG representation for circuits, uncomment following if statement

                    ## if ex is not a successor of op, multiplying by Z should do nothing.
                    #if ex not in ev_successors:
                    #    return ex
                    q = ex.heisenberg_obs(w)
                    qp = q @ Z
                    if q.ndim == 2:
                        # 2nd order observable
                        qp = qp +qp.T
                    return pennylane.expval.PolyXP(qp, wires=range(w), do_queue=False)

                # transform the observables
                obs = list(map(tr_obs, self.ev))
                # measure transformed observables
                temp = self.evaluate_obs(obs, unshifted_params, **kwargs)
                pd += temp

            # restore the original parameter
            op.params[p_idx] = orig

        return pd


#def QNode_vjp(ans, self, params, *args, **kwargs):
def QNode_vjp(ans, self, args, **kwargs):
    """Returns the vector Jacobian product operator for a QNode, as a function
    of the QNode evaluation for specific argnums at the specified parameter values.

    This function is required for integration with Autograd.
    """
    # pylint: disable=unused-argument
    def gradient_product(g):
        """Vector Jacobian product operator.

        Args:
            g (array): scalar or vector multiplying the Jacobian
                from the left (output side).

        Returns:
            nested Sequence[float]: vector-Jacobian product, arranged
            into the nested structure of the QNode input arguments.
        """
        # Jacobian matrix of the circuit
        jac = self.jacobian(args, **kwargs)
        if not g.shape:
            temp = g * jac  # numpy treats 0d arrays as scalars, hence @ cannot be used
        else:
            temp = g @ jac

        # restore the nested structure of the input args
        temp = unflatten(temp.flat, args)
        return temp

    return gradient_product


# define the vector-Jacobian product function for QNode.__call__()
ae.defvjp(QNode.evaluate, QNode_vjp, argnums=[1])
