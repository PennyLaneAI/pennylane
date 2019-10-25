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
r"""
Quantum nodes
=============

**Module name:** :mod:`pennylane.qnode`

.. currentmodule:: pennylane

The :class:`~pennylane.qnode_new.qnode.QNode` class represents a *quantum node*,
encapsulating a *quantum function* (aka :ref:`variational circuit <varcirc>`)
and the computational device it is executed on.

The computational device is an instance of the :class:`~_device.Device`
class, and can represent either a simulator or hardware device. They can be
instantiated using the :func:`~device` loader. PennyLane comes included with
some basic devices; additional devices can be installed as plugins
(see :ref:`plugin_overview` for more details).


Quantum functions
-----------------

.. _quantumfunc:

We use the term *quantum function* to refer both to the abstract mathematical
:math:`\mathbb{R}^m \to \mathbb{R}^n` function represented by the variational quantum circuit,
and the concrete Python function that defines it. The latter must be of the following form:

.. code-block:: python

    def my_quantum_function(x, y, *, w=None):
        qml.RX(x, wires=0)
        qml.RY(2 * y, wires=1)
        qml.CNOT(wires=[0,1])
        for k in range(2):
            qml.RY(w, wires=k)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

Quantum functions are a restricted subset of Python functions, adhering to the following
constraints:

* The body of the function should consist of only supported PennyLane
  :mod:`operations <pennylane.ops>`, one per line for clarity.
  The function can contain classical flow control structures such as ``for`` loops,
  but in general they must not depend on the parameters of the function.

.. note::

    The quantum operations cannot be used outside of a quantum circuit function, as all
    :class:`Operators <pennylane.operation.Operator>` require a QNode to work.

* The function must always return either a single or a tuple of
  *measured observable values*, by applying a :mod:`measurement function <pennylane.measure>`
  to an :mod:`observable <pennylane.ops>`.

.. note::

    Measured observables **must** come after all other operations at the end
    of the circuit function as part of the return statement, and cannot appear in the middle.

* The quantum function can take two kinds of parameters: *positional* and *auxiliary*.

  * The function can *only* be differentiated with respect to its positional parameters.
    The positional parameters should be only used as the parameters of the :class:`.Operator`
    constructors in the function, and they must take the values of nested sequences of real numbers.
    Classical processing of positional parameters, either by arithmetic operations
    or external functions, is not allowed. One current exception is simple scalar multiplication.

  * The auxiliary parameters can *not* be differentiated with respect to.
    They are useful for providing data or 'placeholders' to the quantum function.

    * In *mutable* nodes the auxiliary parameters can undergo any kind of classical
      processing, appear as the ``wires`` argument of Operators, and even affect the
      classical flow control structures of the quantum function.

    * In *immutable* nodes they are subject to the same restrictions as the positional parameters.

    Parameters that have default values are interpreted as auxiliary parameters. They *must* be
    given using the keyword syntax.

The quantum function cannot be executed on its own. Instead, a :class:`~.QNode` object must be
created, which wraps the function and binds it to a device. The QNode can then be used to evaluate
the variational quantum circuit defined by the function on the particular device.

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
                return qml.expval(qml.PauliZ(1))

            result = my_quantum_function(np.pi/4, 0.7)


.. currentmodule:: pennylane.qnode_new.qnode

Classes
-------

.. autosummary::
   QNode
   SignatureParameter
   ParDep

.. currentmodule:: pennylane.qnode_new.jacobian

.. autosummary::
   AutogradMixin
   _JacobianQNode


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
   _default_args
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

from functools import wraps, lru_cache
from collections.abc import Sequence
from collections import namedtuple, OrderedDict
import inspect
import itertools

import numpy as np
from scipy import linalg

from pennylane.operation import Observable, CV, Wires, ObservableReturnTypes
import pennylane.ops as qml
import pennylane.measure as pm
from pennylane.utils import _flatten, unflatten, expand
from pennylane.circuit_graph import CircuitGraph, _is_observable
from pennylane.variable import Variable
from pennylane.qnode import QNode as QNode_old, QuantumFunctionError



def qnode(device, *, mutable=True, properties=None):
    """Decorator for creating QNodes.

    When applied to a quantum function, this decorator converts it into
    a (wrapped) :class:`QNode` instance.

    Args:
        device (~.Device): a PennyLane-compatible device
        mutable (bool): whether the node is mutable
        properties (dict[str->Any]): additional keyword properties passed to the QNode
    """
    @lru_cache()
    def qfunc_decorator(func):
        """The actual decorator"""

        node = QNode(func, device, mutable=mutable, properties=properties)

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function"""
            return node(*args, **kwargs)

        # bind the jacobian method to the wrapped function
        #wrapper.jacobian = node.jacobian
        wrapper.metric_tensor = node.metric_tensor

        # bind the node attributes to the wrapped function
        wrapper.__dict__.update(node.__dict__)

        return wrapper
    return qfunc_decorator



_MARKER = inspect.Parameter.empty  # singleton marker, could be any singleton class


ParDep = namedtuple("ParDep", ["op", "par_idx"])
"""Represents the dependence of an Operator on a free parameter.

Args:
    op (Operator): operator depending on the free parameter in question
    par_idx (int): operator parameter index of the corresponding :class:`~pennylane.variable.Variable` instance
"""


SignatureParameter = namedtuple("SignatureParameter", ["idx", "par"])
"""Describes a single parameter in a function signature.

Args:
    idx (int): positional index of the parameter in the function signature
    par (inspect.Parameter): parameter description
"""


def _get_signature(func):
    """Introspect the parameter signature of a function.

    Adds the following attributes to func:

    * :attr:`func.sig`: OrderedDict[str, SignatureParameter]: mapping from parameters' names to their descriptions
    * :attr:`func.n_pos`: int: number of required positional arguments
    * :attr:`func.var_pos`: bool: can take a variable number of positional arguments (``*args``)
    * :attr:`func.var_keyword`: bool: can take a variable number of keyword arguments (``**kwargs``)

    Args:
        func (callable): function to introspect
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

    func.sig = OrderedDict([(p.name, SignatureParameter(idx, p)) for idx, p in enumerate(sig.parameters.values())])
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
    that depend on its auxiliary parameters, potentially resulting in a different circuit
    on each call. The auxiliary parameters may also determine the wires on which operators act.

    For immutable circuits the quantum function must build the same circuit graph consisting of the same
    :class:`.Operator` instances regardless of its arguments; they can only appear as the
    arguments of the Operators in the circuit. Immutable circuits are slightly faster to execute, and
    can be optimized, but require that the layout of the circuit is fixed.

    Args:
        func (callable): The *quantum function* of the QNode.
            A Python function containing :class:`~.operation.Operation` constructor calls,
            and returning a tuple of measured :class:`~.operation.Observable` instances.
        device (~pennylane._device.Device): computational device to execute the function on
        mutable (bool): whether the circuit is mutable, see above
        properties (dict[str, Any] or None): additional keyword properties for adjusting the QNode behavior
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, func, device, *, mutable=True, properties=None):
        self.func = func  #: callable: quantum function
        self.device = device  #: Device: device that executes the circuit
        self.num_wires = device.num_wires  #: int: number of subsystems (wires) in the circuit
        self.num_variables = None  #: int: number of differentiable parameters in the circuit
        self.ops = []  #: List[Operator]: quantum circuit, in the order the quantum function defines it
        self.circuit = None  #: CircuitGraph: DAG representation of the quantum circuit

        self.mutable = mutable  #: bool: is the circuit mutable?
        self.properties = properties or {}  #: dict[str, Any]: additional keyword properties for adjusting the QNode behavior

        self.variable_deps = {}
        """dict[int, list[ParDep]]: Mapping from free parameter index to the list of
        :class:`~pennylane.operation.Operator` instances (in this circuit) that depend on it.
        """

        self._metric_tensor_subcircuits = None  #: dict[tuple[int], dict[str, Any]]: circuit descriptions for computing the metric tensor
        # introspect the quantum function signature
        _get_signature(self.func)

        self.output_conversion = None  #: callable: for transforming the output of :meth:`.Device.execute` to QNode output
        self.output_dim = None  #: int: dimension of the QNode output vector


    def __repr__(self):
        """String representation."""
        detail = "<QNode: device='{}', func={}, wires={}>"
        return detail.format(self.device.short_name, self.func.__name__, self.num_wires)


    def _set_variables(self, args, kwargs):
        """Store the current values of the free parameters in the Variable class
        so the Operators may access them.

        Args:
            args (tuple[Any]): positional (differentiable) arguments
            kwargs (dict[str, Any]): auxiliary arguments
        """
        Variable.free_param_values = np.array(list(_flatten(args)))
        if not self.mutable:
            # only immutable circuits access auxiliary arguments through Variables
            Variable.kwarg_values = {k: np.array(list(_flatten(v))) for k, v in kwargs.items()}


    def _op_descendants(self, op, only):
        """Descendants of the given operator in the quantum circuit.

        Args:
            op (Operator): operator in the quantum circuit
            only (str, None): the type of descendants to return.

                - ``'G'``: only return non-observables (default)
                - ``'E'``: only return observables
                - ``None``: return all descendants

        Returns:
            list[Operator]: descendants in a topological order
        """
        succ = self.circuit.descendants_in_order((op,))
        if only == 'E':
            return list(filter(_is_observable, succ))
        if only == 'G':
            return list(itertools.filterfalse(_is_observable, succ))
        return succ


    def _append_op(self, op):
        """Append a quantum operation into the circuit queue.

        Args:
            op (~.operation.Operator): quantum operation to be added to the circuit
        """
        if op.num_wires == Wires.All:
            if set(op.wires) != set(range(self.num_wires)):
                raise ValueError("Operator {} must act on all wires".format(op.name))

        # Make sure only existing wires are used.
        for w in op.wires:
            if w < 0 or w >= self.num_wires:
                raise QuantumFunctionError("Operation {} applied to invalid wire {} "
                                           "on device with {} wires.".format(op.name, w, self.num_wires))

        # observables go to their own, temporary queue
        if isinstance(op, Observable):
            if op.return_type is None:
                self.queue.append(op)
            else:
                self.obs_queue.append(op)
        else:
            if self.obs_queue:
                raise QuantumFunctionError('State preparations and gates must precede measured observables.')
            self.queue.append(op)


    def _construct(self, args, kwargs):
        """Construct the quantum circuit graph by calling the quantum function.

        For immutable nodes this method is called the first time :meth:`QNode.evaluate`
        or :meth:`QNode.jacobian` is called, and for mutable nodes *each time*
        they are called. It executes the quantum function,
        stores the resulting sequence of :class:`.Operator` instances,
        converts it into a circuit graph, and creates the Variable mapping.

        .. note::
           The Variables are only required for analytic differentiation,
           for evaluation we could simply reconstruct the circuit each time.

        Args:
            args (tuple[Any]): Positional arguments passed to the quantum function.
                During the construction we are not concerned with the numerical values, but with
                the nesting structure.
                Each positional argument is replaced with a :class:`~.variable.Variable` instance.
            kwargs (dict[str, Any]): Auxiliary arguments passed to the quantum function.
        """
        # pylint: disable=protected-access  # remove when QNode_old is gone
        # pylint: disable=attribute-defined-outside-init, too-many-branches
        self.args_model = args  #: nested Sequence[Number]: nested shape of the arguments for later unflattening

        # flatten the args, replace each argument with a Variable instance carrying a unique index
        arg_vars = [Variable(idx) for idx, _ in enumerate(_flatten(args))]
        self.num_variables = len(arg_vars)
        # arrange the newly created Variables in the nested structure of args
        arg_vars = unflatten(arg_vars, args)

        # temporary queues for operations and observables
        self.queue = []   #: list[Operation]: applied operations
        self.obs_queue = []  #: list[Observable]: applied observables

        # set up the context for Operator entry
        if QNode_old._current_context is None:
            QNode_old._current_context = self
        else:
            raise QuantumFunctionError('QNode._current_context must not be modified outside this method.')
        try:
            # generate the program queue by executing the quantum circuit function
            if self.mutable:
                # it's ok to directly pass auxiliary arguments since the circuit is re-constructed each time
                # (positional args must be replaced because parameter-shift differentiation requires Variables)
                res = self.func(*arg_vars, **kwargs)
            else:
                # must convert auxiliary arguments to named Variables so they can be updated without re-constructing the circuit
                kwarg_vars = {}
                for key, val in kwargs.items():
                    temp = [Variable(idx, name=key) for idx, _ in enumerate(_flatten(val))]
                    kwarg_vars[key] = unflatten(temp, val)

                res = self.func(*arg_vars, **kwarg_vars)
        finally:
            QNode_old._current_context = None

        # check the validity of the circuit
        self._check_circuit(res)

        # map each free variable to the operators which depend on it
        self.variable_deps = {}
        for k, op in enumerate(self.ops):
            for j, p in enumerate(_flatten(op.params)):
                if isinstance(p, Variable):
                    if p.name is None: # ignore auxiliary arguments
                        self.variable_deps.setdefault(p.idx, []).append(ParDep(op, j))

        # generate the DAG
        self.circuit = CircuitGraph(self.ops, self.variable_deps)

        # check for operations that cannot affect the output
        if self.properties.get('vis_check', False):
            visible = self.circuit.ancestors(self.circuit.observables)
            invisible = set(self.circuit.operations) - visible
            if invisible:
                raise QuantumFunctionError("The operations {} cannot affect the output of the circuit.".format(invisible))


    def _check_circuit(self, res):
        """Check that the generated Operator queue corresponds to a valid quantum circuit.

        .. note:: The validity of individual Operators is checked already in :meth:`_append_op`.

        Args:
            res (Sequence[Observable], Observable): output returned by the quantum function

        Raises:
            QuantumFunctionError: an error was discovered in the circuit
        """
        # check the return value
        if isinstance(res, Observable):
            if res.return_type is ObservableReturnTypes.Sample:
                # Squeezing ensures that there is only one array of values returned
                # when only a single-mode sample is requested
                self.output_conversion = np.squeeze
            else:
                self.output_conversion = float
            self.output_dim = 1
            res = (res,)
        elif isinstance(res, Sequence) and res and all(isinstance(x, Observable) for x in res):
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
        are_cvs = [isinstance(op, CV) for op in self.ops if not isinstance(op, qml.Identity)]
        if not all(are_cvs) and any(are_cvs):
            raise QuantumFunctionError(
                "Continuous and discrete operations are not allowed in the same quantum circuit.")

        # TODO: we should enforce plugins using the Device.capabilities dictionary to specify
        # whether they are qubit or CV devices, and remove this logic here.
        self.type = 'cv' if all(are_cvs) else 'qubit'  #: str: circuit type, in {'cv', 'qubit'}


    def _default_args(self, kwargs):
        """Validate the quantum function arguments, apply defaults.

        Here we apply default values for the auxiliary parameters of :attr:`QNode.func`.

        Args:
            kwargs (dict[str, Any]): auxiliary arguments (given using the keyword syntax)

        Returns:
            dict[str, Any]: all auxiliary arguments (with defaults)
        """
        forbidden_kinds = (inspect.Parameter.POSITIONAL_ONLY,
                           inspect.Parameter.VAR_POSITIONAL,
                           inspect.Parameter.VAR_KEYWORD)

        # check the validity of kwargs items
        for name in kwargs:
            s = self.func.sig.get(name)
            if s is None:
                if self.func.var_keyword:
                    continue  # unknown parameter, but **kwargs will take it TODO should it?
                raise QuantumFunctionError("Unknown quantum function parameter '{}'.".format(name))
            if s.par.kind in forbidden_kinds or s.par.default == inspect.Parameter.empty:
                raise QuantumFunctionError("Quantum function parameter '{}' cannot be given using the keyword syntax.".format(name))

        # apply defaults
        for name, s in self.func.sig.items():
            default = s.par.default
            if default != inspect.Parameter.empty:
                # meant to be given using keyword syntax
                kwargs.setdefault(name, default)

        return kwargs


    def __call__(self, *args, **kwargs):
        """Wrapper for :meth:`~.QNode.evaluate`.
        """
        return self.evaluate(args, kwargs)


    def evaluate(self, args, kwargs):
        """Evaluate the quantum function on the specified device.

        Args:
            args (tuple[Any]): positional arguments to the quantum function (differentiable)
            kwargs (dict[str, Any]): auxiliary arguments (not differentiable)

        Returns:
            float or array[float]: output measured value(s)
        """
        kwargs = self._default_args(kwargs)

        if self.circuit is None or self.mutable:
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
            kwargs (dict[str, Any]): auxiliary arguments (not differentiable)

        Returns:
            array[float]: measured values
        """
        # temporarily store the parameter values in the Variable class
        self._set_variables(args, kwargs)

        self.device.reset()
        ret = self.device.execute(self.circuit.operations, obs, self.circuit.variable_deps)
        return ret


    def _construct_metric_tensor(self, *, diag_approx=False):
        """Construct metric tensor subcircuits for qubit circuits.

        Constructs a set of quantum circuits for computing a block-diagonal approximation of the
        Fubini-Study metric tensor on the parameter space of the variational circuit represented
        by the QNode, using the Quantum Geometric Tensor.

        If the parameter appears in a gate :math:`G`, the subcircuit contains
        all gates which precede :math:`G`, and :math:`G` is replaced by the variance
        value of its generator.

        Args:
            diag_approx (bool): iff True, use the diagonal approximation
        """
        # pylint: disable=too-many-statements, too-many-branches

        self._metric_tensor_subcircuits = {}
        for queue, curr_ops, param_idx, _ in self.circuit.iterate_layers():
            obs = []
            scale = []

            Ki_matrices = []
            KiKj_matrices = []
            Ki_ev = []
            KiKj_ev = []
            V = None

            # for each operation in the layer, get the generator and convert it to a variance
            for n, op in enumerate(curr_ops):
                gen, s = op.generator
                w = op.wires

                if gen is None:
                    raise QuantumFunctionError("Can't generate metric tensor, operation {}"
                                               "has no defined generator".format(op))

                # get the observable corresponding to the generator of the current operation
                if isinstance(gen, np.ndarray):
                    # generator is a Hermitian matrix
                    variance = pm.var(qml.Hermitian(gen, w, do_queue=False))

                    if not diag_approx:
                        Ki_matrices.append((n, expand(gen, w, self.num_wires)))

                elif issubclass(gen, Observable):
                    # generator is an existing PennyLane operation
                    variance = pm.var(gen(w, do_queue=False))

                    if not diag_approx:
                        if issubclass(gen, qml.PauliX):
                            mat = np.array([[0, 1], [1, 0]])
                        elif issubclass(gen, qml.PauliY):
                            mat = np.array([[0, -1j], [1j, 0]])
                        elif issubclass(gen, qml.PauliZ):
                            mat = np.array([[1, 0], [0, -1]])

                        Ki_matrices.append((n, expand(mat, w, self.num_wires)))

                else:
                    raise QuantumFunctionError(
                        "Can't generate metric tensor, generator {}"
                        "has no corresponding observable".format(gen)
                    )

                obs.append(variance)
                scale.append(s)

            if not diag_approx:
                # In order to compute the block diagonal portion of the metric tensor,
                # we need to compute 'second order' <psi|K_i K_j|psi> terms.

                for i, j in itertools.product(range(len(Ki_matrices)), repeat=2):
                    # compute the matrices representing all K_i K_j terms
                    obs1 = Ki_matrices[i]
                    obs2 = Ki_matrices[j]
                    KiKj_matrices.append(((obs1[0], obs2[0]), obs1[1] @ obs2[1]))

                V = np.identity(2**self.num_wires, dtype=np.complex128)

                # generate the unitary operation to rotate to
                # the shared eigenbasis of all observables
                for _, term in Ki_matrices:
                    _, S = linalg.eigh(V.conj().T @ term @ V)
                    V = np.round(V @ S, 15)

                V = V.conj().T

                # calculate the eigenvalues for
                # each observable in the shared eigenbasis
                for idx, term in Ki_matrices:
                    eigs = np.diag(V @ term @ V.conj().T).real
                    Ki_ev.append((idx, eigs))

                for idx, term in KiKj_matrices:
                    eigs = np.diag(V @ term @ V.conj().T).real
                    KiKj_ev.append((idx, eigs))

            self._metric_tensor_subcircuits[param_idx] = {
                "queue": queue,
                "observable": obs,
                "Ki_expectations": Ki_ev,
                "KiKj_expectations": KiKj_ev,
                "eigenbasis_matrix": V,
                "result": None,
                "scale": scale,
            }


    def metric_tensor(self, args, kwargs=None, *, diag_approx=False, only_construct=False):
        """Evaluate the value of the metric tensor.

        Args:
            args (tuple[Any]): positional (differentiable) arguments
            kwargs (dict[str, Any]): auxiliary arguments
            diag_approx (bool): iff True, use the diagonal approximation
            only_construct (bool): Iff True, construct the circuits used for computing
                the metric tensor but do not execute them, and return None.

        Returns:
            array[float]: metric tensor
        """
        kwargs = kwargs or {}
        kwargs = self._default_args(kwargs)

        if self.circuit is None or self.mutable:
            # construct the circuit
            self._construct(args, kwargs)

        if self._metric_tensor_subcircuits is None:
            self._construct_metric_tensor(diag_approx=diag_approx)

        if only_construct:
            return None

        # temporarily store the parameter values in the Variable class
        self._set_variables(args, kwargs)

        tensor = np.zeros([self.num_variables, self.num_variables])

        # execute constructed metric tensor subcircuits
        for params, circuit in self._metric_tensor_subcircuits.items():
            self.device.reset()

            s = np.array(circuit['scale'])
            V = circuit['eigenbasis_matrix']

            if not diag_approx:
                # block diagonal approximation

                unitary_op = qml.QubitUnitary(V, wires=list(range(self.num_wires)), do_queue=False)
                self.device.execute(circuit['queue'] + [unitary_op], circuit['observable'])
                probs = list(self.device.probability().values())

                first_order_ev = np.zeros([len(params)])
                second_order_ev = np.zeros([len(params), len(params)])

                for idx, ev in circuit['Ki_expectations']:
                    first_order_ev[idx] = ev @ probs

                for idx, ev in circuit['KiKj_expectations']:
                    # idx is a 2-tuple (i, j), representing
                    # generators K_i, K_j
                    second_order_ev[idx] = ev @ probs

                    # since K_i and K_j are assumed to commute,
                    # <psi|K_j K_i|psi> = <psi|K_i K_j|psi>,
                    # and thus the matrix of second-order expectations
                    # is symmetric
                    second_order_ev[idx[1], idx[0]] = second_order_ev[idx]

                g = np.zeros([len(params), len(params)])

                for i, j in itertools.product(range(len(params)), repeat=2):
                    g[i, j] = s[i] * s[j] * (second_order_ev[i, j] - first_order_ev[i] * first_order_ev[j])

                row = np.array(params).reshape(-1, 1)
                col = np.array(params).reshape(1, -1)
                circuit['result'] = np.diag(g)
                tensor[row, col] = g

            else:
                # diagonal approximation
                circuit['result'] = s**2 * self.device.execute(circuit['queue'], circuit['observable'])
                tensor[np.array(params), np.array(params)] = circuit['result']

        return tensor
