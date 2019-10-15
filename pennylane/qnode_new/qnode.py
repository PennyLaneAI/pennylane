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

**Module name:** :mod:`pennylane.qnode`

.. currentmodule:: pennylane

The :class:`~qnode.QNode` class represents *quantum nodes*,
encapsulating a *quantum function* or :ref:`variational circuit <varcirc>`
and the computational device it is executed on.

The computational device is an instance of the :class:`~_device.Device`
class, and can represent either a simulator or hardware device. They can be
instantiated using the :func:`~device` loader. PennyLane comes included with
some basic devices; additional devices can be installed as plugins
(see :ref:`plugins` for more details).


Quantum functions
-----------------

.. _quantumfunc:

The quantum circuit function encapsulated by the QNode must be of the following form:

.. code-block:: python

    def my_quantum_function(x, y, *, w):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(y, wires=0)
        qml.RY(w, wires=1)
        return qml.expval(qml.PauliZ(0))

Quantum circuit functions are a restricted subset of Python functions, adhering to the following
constraints:

* The body of the function must consist of only supported PennyLane
  :mod:`operations <pennylane.ops>`, one per line.

* The function must always return either a single or a tuple of
  *measured observable values*, by applying a :mod:`measurement function <pennylane.measure>`
  to an :mod:`observable <pennylane.ops>`.

* Classical processing of function arguments, either by arithmetic operations
  or external functions, is not allowed. One current exception is simple scalar
  multiplication.

* The function can only be differentiated with respect to its positional parameters.
  Additionally it may have keyword-only parameters.

.. note::

    The quantum operations cannot be used outside of a quantum circuit function, as all
    :class:`Operations <pennylane.operation.Operation>` require a QNode in order to perform queuing on initialization.

.. note::

    Measured observables **must** come after all other operations at the end
    of the circuit function as part of the return statement, and cannot appear in the middle.

The quantum function cannot be executed on its own. Instead, a :class:`~.QNode` object must be
created, which wraps the function and binds it to a device. The QNode can then be used to evaluate
the quantum circuit function on the particular device.

For example:

.. code-block:: python

    device = qml.device('default.qubit', wires=2)
    qnode1 = qml.QNode(my_quantum_function, device)
    result = qnode1(np.pi/4, 0.7)

.. note::

        The :func:`~pennylane.decorator.qnode` decorator is provided as a convenience
        to automate the process of creating quantum nodes. The decorator is used as follows:

        .. code-block:: python

            @qml.qnode(device)
            def my_quantum_function(x, y):
                qml.RZ(x, wires=0)
                qml.CNOT(wires=[0,1])
                qml.RY(y, wires=1)
                return qml.expval(qml.PauliZ(0))

            result = my_quantum_function(np.pi/4, 0.7)


.. currentmodule:: pennylane.qnode


Classes
-------

.. autosummary::
   QNode
   JacobianQNode
   ParSig
   ParDep
   AutogradMixin


QNode methods
-------------

.. currentmodule:: pennylane.qnode_new.qnode.QNode

.. autosummary::
   __call__
   evaluate
   evaluate_obs
   _append_op
   _construct
   _check_circuit
   _sort_args
   _check_args
   _set_variables
   _op_descendants


JacobianQNode methods
---------------------

.. currentmodule:: pennylane.qnode_new.jacobian.JacobianQNode

.. autosummary::
   jacobian
   _best_method
   _pd_finite_diff
   _pd_parameter_shift
   _pd_analytic_var
   _transform_observable

.. currentmodule:: pennylane.qnode_new.qnode

Exceptions
----------

.. autosummary::
   QuantumFunctionError

Code details
~~~~~~~~~~~~
"""

# DISABLEpylint: disable=cell-var-from-loop,attribute-defined-outside-init,too-many-branches,too-many-arguments

from collections.abc import Sequence
from collections import namedtuple, OrderedDict
import copy
import inspect
import itertools
import numbers

import autograd.extend as ae
import autograd.builtins
import numpy as np

import pennylane.operation as plo
import pennylane.ops as plops
from pennylane.utils import _flatten, unflatten, _inv_dict, expand
from pennylane.circuit_graph import CircuitGraph, _is_observable
from pennylane.variable import Variable
from pennylane.qnode import QNode as QNode_old, QuantumFunctionError


_MARKER = inspect.Parameter.empty  # singleton marker, could be any singleton class


ParDep = namedtuple("ParDep", ["op", "par_idx"])
"""Represents the dependence of an Operation on a free parameter.

Args:
    op (Operation): operation depending on the free parameter in question
    par_idx (int): operation parameter index of the corresponding :class:`~.Variable` instance
"""


ParSig = namedtuple("ParSig", ["idx", "par"])
"""Describes a parameter in a function signature.

Args:
    idx (int): positional index of the parameter in the function signature
    par (inspect.Parameter): parameter description
"""

def _get_signature(func):
    """Introspects the parameter signature of a function.

    Adds the following attributes to func:
        :attr:`func.sig`: OrderedDict[str, ParSig]: mapping from parameters' names to their descriptions
        :attr:`func.n_pos`: int: number of required positional arguments
        :attr:`func.var_pos`: bool: can take a variable number of positional arguments (*args)
        :attr:`func.var_keyword`: bool: can take a variable number of keyword arguments (**kwargs)

    Args:
        func (callable): a function
    """
    sig = inspect.signature(func)
    # count positional args, see if VAR_ args are present
    n_pos = 0
    func.var_pos = False
    func.var_keyword = False
    for p in sig.parameters.values():
        if p.kind <= inspect.Parameter.POSITIONAL_OR_KEYWORD:
            n_pos += 1
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            func.var_pos = True
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            func.var_keyword = True

    func.sig = OrderedDict([(p.name, ParSig(idx, p)) for idx, p in enumerate(sig.parameters.values())])
    func.n_pos = n_pos


class QNode:
    """Base class for quantum nodes in the hybrid computational graph.

    A QNode encapsulates a :ref:`quantum function <quantumfunc>`
    (corresponding to a :ref:`variational circuit <varcirc>`)
    and the computational device it is executed on.

    The QNode calls the quantum function to construct a :class:`.CircuitGraph` instance represeting
    the quantum circuit. The circuit can be either

    * *mutable*, which means the quantum function is called each time the QNode is evaluated, or
    * *immutable*, which means the quantum function is called only once, on first evaluation,
       to construct the circuit representation.

    If the circuit is mutable, the quantum function can contain classical flow control structures
    that depend on its keyword-only parameters, potentially resulting in a different circuit
    on each call. The keyword-only parameters may also determine the wires on which operations act.

    For immutable circuits the quantum function must build the same circuit graph consisting of the same
    :class:`.Operation` instances regardless of its arguments; they can only appear as the
    arguments of the Operations in the circuit. Immutable circuits are slightly faster to execute, and
    can be optimized, but can be used only if the layout of the circuit is always fixed.

    Args:
        func (callable): The *quantum function* of the QNode.
            A Python function containing :class:`~.operation.Operation` constructor calls,
            and returning a tuple of measured :class:`~.operation.Observable` instances.
        device (~pennylane._device.Device): computational device to execute the function on
        mutable (bool): whether the circuit is mutable, see above
    """
    def __init__(self, func, device, mutable=True):
        self.func = func  #: callable: quantum function
        self.device = device  #: Device: device that executes the circuit
        self.num_wires = device.num_wires  #: int: number of subsystems (wires) in the circuit
        self.num_variables = None  #: int: number of differentiable parameters in the circuit
        self.ops = []  #: List[Operation]: quantum circuit, in the order the quantum function defines it
        self.circuit = None  #: CircuitGraph: DAG representation of the quantum circuit

        self.mutable = mutable  #: bool: is the circuit mutable?

        self.variable_deps = {}
        """dict[int, list[ParDep]]: Mapping from free parameter index to the list of
        :class:`~pennylane.operation.Operation` instances (in this circuit) that depend on it.
        """

        self._metric_tensor_subcircuits = {}  #: TODO define
        # introspect the quantum function signature
        _get_signature(self.func)


    def __repr__(self):
        """String representation."""
        detail = "<QNode: device='{}', func={}, wires={}, interface=NumPy/Autograd>"
        return detail.format(self.device.short_name, self.func.__name__, self.num_wires)


    @staticmethod
    def _set_variables(args, kwargs):
        """Temporarily store the values of the free parameters in the Variable class.

        Args:
            args (tuple[Any]): values for the positional, differentiable parameters
            kwargs (dict[str, Any]): values for the keyword-only parameters
        """
        Variable.free_param_values = np.array(list(_flatten(args)))
        Variable.kwarg_values = {k: np.array(list(_flatten(v))) for k, v in kwargs.items()}


    def _op_descendants(self, op, only):
        """Descendants of the given operation in the quantum circuit.

        Args:
            op (Operation): operation in the quantum circuit
            only (str, None): the type of descendants to return.

                - ``'G'``: only return non-observables (default)
                - ``'E'``: only return observables
                - ``None``: return all descendants

        Returns:
            list[Operation]: descendants in a topological order
        """
        succ = self.circuit.descendants_in_order((op,))
        if only == 'E':
            return list(filter(_is_observable, succ))
        if only == 'G':
            return list(itertools.filterfalse(_is_observable, succ))
        return succ


    def _append_op(self, op):
        """Appends a quantum operation into the circuit queue.

        Args:
            op (~.operation.Operation): quantum operation to be added to the circuit
        """
        if op.num_wires == plo.Wires.All:
            if set(op.wires) != set(range(self.num_wires)):
                raise QuantumFunctionError("Operation {} must act on all wires".format(op.name))

        # Make sure only existing wires are used.
        for w in op.wires:
            if w < 0 or w >= self.num_wires:
                raise QuantumFunctionError("Operation {} applied to invalid wire {} "
                                           "on device with {} wires.".format(op.name, w, self.num_wires))

        # EVs go to their own, temporary queue
        if isinstance(op, plo.Observable):
            if op.return_type is None:
                self.queue.append(op)
            else:
                self.obs_queue.append(op)
        else:
            if self.obs_queue:
                raise QuantumFunctionError('State preparations and gates must precede measured observables.')
            self.queue.append(op)


    def _construct(self, args, kwargs):
        """Constructs the quantum circuit graph by calling the quantum function.

        .. note:: The user should never have to call this method directly.

        This method is called automatically the first time :meth:`QNode.evaluate`
        or :meth:`QNode.jacobian` is called. It executes the quantum function,
        stores the resulting sequence of :class:`.Operation` instances,
        converts it into a circuit graph, and creates the Variable mapping.

        .. note::
           The Variables are only required for analytic differentiation,
           for evaluation we could simply reconstruct the circuit each time.

        Args:
            args (tuple): Positional arguments passed to the quantum function.
                During the construction we are not concerned with the numerical values, but with
                the nesting structure.
                Each parameter is replaced with a :class:`~.variable.Variable` instance.
            kwargs (dict): Additional keyword-only arguments may be passed to the quantum function,
                however PennyLane does not support differentiation with respect to them.
                Instead, keyword arguments are useful for providing data or 'placeholders'
                to the quantum function.
        """
        self.args_model = args  #: nested Sequence[Number]: nested shape of the arguments for later unflattening

        # flatten the args, replace each argument with a Variable instance carrying a unique index
        arg_vars = [Variable(idx) for idx, _ in enumerate(_flatten(args))]
        self.num_variables = len(arg_vars)
        # arrange the newly created Variables in the nested structure of args
        arg_vars = unflatten(arg_vars, args)

        # temporary queues for operations and observables
        self.queue = []   #: list[Operation]: applied operations
        self.obs_queue = []  #: list[Observable]: applied observables

        # set up the context for Operation entry
        if QNode_old._current_context is None:
            QNode_old._current_context = self
        else:
            raise QuantumFunctionError('QNode._current_context must not be modified outside this method.')
        try:
            # generate the program queue by executing the quantum circuit function
            if self.mutable:
                # no caching, it's ok to directly pass kwarg values
                # (positional args must be replaced because parameter-shift differentiation requires Variables)
                res = self.func(*arg_vars, **kwargs)
            else:
                # caching mode, must use variables for kwargs so they can be updated without re-constructing
                # replace each keyword argument with a list of named Variables
                kwarg_vars = {}
                for key, val in kwargs.items():
                    temp = [Variable(idx, name=key) for idx, _ in enumerate(_flatten(val))]
                    kwarg_vars[key] = unflatten(temp, val)

                res = self.func(*arg_vars, **kwarg_vars)
        finally:
            QNode_old._current_context = None

        # check the validity of the circuit
        self._check_circuit(res)

        # map each free variable to the operations which depend on it
        self.variable_deps = {}
        for k, op in enumerate(self.ops):
            for j, p in enumerate(_flatten(op.params)):
                if isinstance(p, Variable):
                    if p.name is None: # ignore keyword arguments
                        self.variable_deps.setdefault(p.idx, []).append(ParDep(op, j))

        # generate the DAG
        self.circuit = CircuitGraph(self.ops, self.variable_deps)

        # check for operations that cannot affect the output
        visible = self.circuit.ancestors(self.circuit.observables)
        invisible = set(self.circuit.operations) - visible
        if invisible:
            raise QuantumFunctionError("The operations {} cannot affect the output of the circuit.".format(invisible))


    def _check_circuit(self, res):
        """Checks that the generated operation queue corresponds to a valid quantum circuit.

        .. note:: The validity of individual Operations is checked already in :meth:`_append_op`.

        Args:
            res (Sequence[Observable], Observable): output returned by the quantum function

        Raises:
            QuantumFunctionError: an error was discovered in the circuit
        """
        # check the return value
        if isinstance(res, plo.Observable):
            if res.return_type is plo.Sample:
                # Squeezing ensures that there is only one array of values returned
                # when only a single-mode sample is requested
                self.output_conversion = np.squeeze
            else:
                self.output_conversion = float
            self.output_dim = 1
            res = (res,)
        elif isinstance(res, Sequence) and res and all(isinstance(x, plo.Observable) for x in res):
            # for multiple observables values, any valid Python sequence of observables
            # (i.e., lists, tuples, etc) are supported in the QNode return statement.

            # Device already returns the correct numpy array, so no further conversion is required
            self.output_conversion = np.asarray
            self.output_dim = len(res)
            res = tuple(res)
        else:
            raise QuantumFunctionError("A quantum function must return either a single measured observable "
                                       "or a nonempty sequence of measured observables.")

        # check that all returned observables have a return_type specified
        for x in res:
            if x.return_type is None:
                raise QuantumFunctionError(
                    "Observable '{}' does not have the measurement type specified.".format(x))

        # check that all ev's are returned, in the correct order
        if res != tuple(self.obs_queue):
            raise QuantumFunctionError(
                "All measured observables must be returned in the order they are measured.")

        # check that no wires are measured more than once
        m_wires = list(w for ob in res for w in ob.wires)
        if len(m_wires) != len(set(m_wires)):
            raise QuantumFunctionError('Each wire in the quantum circuit can only be measured once.')

        self.ops = self.queue + list(res)
        del self.queue
        del self.obs_queue

        # True if op is a CV, False if it is a discrete variable (Identity could be either)
        are_cvs = [isinstance(op, plo.CV) for op in self.ops if not isinstance(op, plops.Identity)]
        if not all(are_cvs) and any(are_cvs):
            raise QuantumFunctionError(
                "Continuous and discrete operations are not allowed in the same quantum circuit.")

        # TODO: we should enforce plugins using the Device.capabilities dictionary to specify
        # whether they are qubit or CV devices, and remove this logic here.
        self.type = 'cv' if all(are_cvs) else 'qubit'  #: str: circuit type, in {'cv', 'qubit'}


    def _sort_args(self, args, kwargs):
        """Sort the quantum function arguments to positional and keyword-only.

        When the QNode is evaluated via :meth:`~QNode.__call__`, ``kwargs`` may also contain
        differentiable POSITIONAL_OR_KEYWORD arguments given using the keyword syntax,
        in addition to KEYWORD_ONLY arguments.

        This method moves all positional (differentiable) arguments from ``kwargs`` to ``args``,
        and substitute missing positional arguments with special singleton markers.

        Args:
            args (tuple[Any]): positionally given arguments
            kwargs (dict[str, Any]): keyword-given arguments

        Returns:
            tuple[Any], dict[str, Any]: positional arguments, keyword-only arguments
        """
        n_args = len(args)  # number of positionally-given args
        pos = list(args)
        for name, s in self.func.sig.items():
            # is it positional?
            if s.par.kind <= inspect.Parameter.POSITIONAL_OR_KEYWORD:
                if s.idx < n_args:
                    # this parameter is already given in args
                    if name in kwargs:
                        raise QuantumFunctionError("Quantum function parameter '{}' given twice.".format(name))
                else:
                    # not in args, maybe we find it in kwargs?
                    p = kwargs.pop(name, _MARKER)  # None may be a valid parameter value, hence _MARKER
                    pos.append(p)
        return tuple(pos), kwargs


    def _check_args(self, args, kwargs):
        """Validate the quantum function arguments, apply defaults.

        Here we apply all the default values for the parameters found in the signature of
        :attr:`QNode.func`, and check if any arguments are invalid or missing.

        Args:
            args (tuple[Any]): positional (differentiable) arguments
            kwargs (dict[str, Any]): keyword-only arguments

        Returns:
            tuple[Any], dict[str, Any]: positional arguments, keyword-only arguments
        """
        # TODO maybe disallow *args in qfunc signature entirely? it complicates things and may be confusing

        forbidden_kinds = (inspect.Parameter.POSITIONAL_ONLY,
                           inspect.Parameter.VAR_POSITIONAL,
                           inspect.Parameter.VAR_KEYWORD)

        # check the validity of kwargs items
        for name, val in kwargs.items():
            s = self.func.sig.get(name)
            if s is None:
                if self.func.var_keyword:
                    continue  # unknown parameter, but **kwargs will take it
                else:
                    raise QuantumFunctionError("Unknown quantum function parameter '{}'.".format(name))
            if s.par.kind in forbidden_kinds:
                raise QuantumFunctionError("Quantum function parameter '{}' cannot be given using the keyword syntax.".format(name))

        n_args = len(args)  # number of positional args given
        if n_args < self.func.n_pos or (n_args > self.func.n_pos and not self.func.var_pos):
            raise QuantumFunctionError("Quantum function takes {} positional parameters, {} given.".format(self.func.n_pos, n_args))

        # apply defaults, move positional arguments to args
        pos = list(args)
        for name, s in self.func.sig.items():
            default = s.par.default
            if s.idx < n_args:
                # positional
                if pos[s.idx] == _MARKER:
                    # missing
                    if default != inspect.Parameter.empty:
                        pos[s.idx] = default
                    else:
                        raise QuantumFunctionError("Quantum function positional parameter '{}' missing.".format(name))
            else:
                # keyword-only
                # can we find the parameter in kwargs?
                p = kwargs.get(name, _MARKER)  # None may be a valid parameter value, hence _MARKER
                if p is _MARKER and s.par.kind not in forbidden_kinds:
                    # missing
                    if default != inspect.Parameter.empty:
                        kwargs[name] = default
                    else:
                        raise QuantumFunctionError("Quantum function keyword-only parameter '{}' missing.".format(name))

        return tuple(pos), kwargs


    def __call__(self, *args, **kwargs):
        """Wrapper for :meth:`~.QNode.evaluate`.
        """
        args, kwargs = self._sort_args(args, kwargs)
        args = autograd.builtins.tuple(args)  # prevents autograd boxed arguments from going through to evaluate
        return self.evaluate(args, kwargs)


    def evaluate(self, args, kwargs):
        """Evaluates the quantum function on the specified device.

        Args:
            args (tuple[Any]): positional arguments to the quantum function (differentiable)
            kwargs (dict[str, Any]): keyword-only arguments (not differentiable)

        Returns:
            float or array[float]: output measured value(s)
        """
        args, kwargs = self._check_args(args, kwargs)

        if self.circuit is None or self.mutable:
            if self.circuit is not None:
                # circuit construction has previously been called
                flat_args = list(_flatten(args))
                if len(flat_args) == self.num_variables:
                    # only construct the circuit if the number
                    # of arguments matches the allowed number
                    # of variables.
                    # This avoids construction happening
                    # via self._pd_analytic, where temporary
                    # variables are appended to the positional argument list.

                    # FIXME maybe we require that if extra args are given, the caller must _construct()

                    # unflatten arguments,  why??? (because _pd_parameter_shift passes a flat array!)
                    shaped_args = unflatten(flat_args, self.args_model)

                    # construct the circuit
                    self._construct(shaped_args, kwargs)
            else:
                # circuit has not yet been constructed
                self._construct(args, kwargs)

        # temporarily store the parameter values in the Variable class
        self._set_variables(args, kwargs)

        self.device.reset()
        ret = self.device.execute(self.circuit.operations, self.circuit.observables, self.variable_deps)
        return self.output_conversion(ret)


    def evaluate_obs(self, obs, args, kwargs):
        """Evaluate the value of the given observables.

        Assumes :meth:`construct` has already been called.

        Args:
            obs  (Iterable[Observable]): observables to measure
            args (array[float]): positional arguments to the quantum function (differentiable)
            kwargs (dict[str, Any]): keyword-only arguments (not differentiable)

        Returns:
            array[float]: measured values
        """
        args, kwargs = self._check_args(args, kwargs)

        # temporarily store the parameter values in the Variable class
        self._set_variables(args, kwargs)

        self.device.reset()
        ret = self.device.execute(self.circuit.operations, obs, self.circuit.variable_deps)
        return ret
