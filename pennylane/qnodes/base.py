# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Base QNode class and utilities
"""
from collections.abc import Sequence
from collections import namedtuple, OrderedDict
import inspect
import itertools

import numpy as np

import pennylane as qml
from pennylane.operation import Observable, CV, Wires, ObservableReturnTypes
from pennylane.utils import _flatten, unflatten
from pennylane.circuit_graph import CircuitGraph, _is_observable
from pennylane.variable import Variable



ParameterDependency = namedtuple("ParameterDependency", ["op", "par_idx"])
"""Represents the dependence of an Operator on a positional parameter of the quantum function.

Args:
    op (Operator): operator depending on the positional parameter in question
    par_idx (int): flattened operator parameter index of the corresponding
        :class:`~pennylane.variable.Variable` instance
"""


SignatureParameter = namedtuple("SignatureParameter", ["idx", "par"])
"""Describes a single parameter in a function signature.

Args:
    idx (int): positional index of the parameter in the function signature
    par (inspect.Parameter): parameter description
"""


class QuantumFunctionError(Exception):
    """Exception raised when an illegal operation is defined in a quantum function."""


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

    func.sig = OrderedDict(
        [(p.name, SignatureParameter(idx, p)) for idx, p in enumerate(sig.parameters.values())]
    )
    func.n_pos = n_pos


def _decompose_queue(ops, device):
    """Recursively loop through a queue and decompose
    operations that are not supported by a device.

    Args:
        ops (List[~.Operation]): operation queue
        device (~.Device): a PennyLane device
    """
    new_ops = []

    for op in ops:
        if device.supports_operation(op.name):
            new_ops.append(op)
        else:
            decomposed_ops = op.decomposition(*op.params, wires=op.wires)
            decomposition = _decompose_queue(decomposed_ops, device)
            new_ops.extend(decomposition)

    return new_ops


def decompose_queue(ops, device):
    """Decompose operations in a queue that are not supported by a device.

    This is a wrapper function for :func:`~._decompose_queue`,
    which raises an error if an operation or its decomposition
    is not supported by the device.

    Args:
        ops (List[~.Operation]): operation queue
        device (~.Device): a PennyLane device
    """
    new_ops = []

    for op in ops:
        try:
            new_ops.extend(_decompose_queue([op], device))
        except NotImplementedError:
            raise qml.DeviceError("Gate {} not supported on device {}".format(op.name, device.short_name))

    return new_ops


class BaseQNode:
    """Base class for quantum nodes in the hybrid computational graph.

    A *quantum node* encapsulates a :ref:`quantum function <intro_vcirc_qfunc>`
    (corresponding to a :ref:`variational circuit <varcirc>`)
    and the computational device it is executed on.

    The QNode calls the quantum function to construct a :class:`.CircuitGraph` instance represeting
    the quantum circuit. The circuit can be either

    * *mutable*, which means the quantum function is called each time the QNode is evaluated, or
    * *immutable*, which means the quantum function is called only once, on first evaluation,
      to construct the circuit representation.

    If the circuit is mutable, its **auxiliary** parameters can undergo any kind of classical
    processing inside the quantum function. It can also contain classical flow control structures
    that depend on the auxiliary parameters, potentially resulting in a different circuit
    on each call. The auxiliary parameters may also determine the wires on which operators act.

    For immutable circuits the quantum function must build the same circuit graph consisting of the same
    :class:`.Operator` instances regardless of its parameters; they can only appear as the
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
        self.num_variables = (
            None  #: int: number of flattened differentiable parameters in the circuit
        )
        self.ops = (
            []
        )  #: List[Operator]: quantum circuit, in the order the quantum function defines it
        self.circuit = None  #: CircuitGraph: DAG representation of the quantum circuit

        self.mutable = mutable  #: bool: whether the circuit is mutable
        self.properties = (
            properties or {}
        )  #: dict[str, Any]: additional keyword properties for adjusting the QNode behavior

        self.variable_deps = {}
        """dict[int, list[ParameterDependency]]: Mapping from flattened qfunc positional parameter
        index to the list of :class:`~pennylane.operation.Operator` instances (in this circuit)
        that depend on it.
        """

        self._metric_tensor_subcircuits = None
        """dict[tuple[int], dict[str, Any]]: circuit descriptions for computing the metric tensor"""

        # introspect the quantum function signature
        _get_signature(self.func)

        self.output_conversion = None  #: callable: for transforming the output of :meth:`.Device.execute` to QNode output
        self.output_dim = None  #: int: dimension of the QNode output vector
        self.model = self.device.capabilities()["model"]  #: str: circuit type, in {'cv', 'qubit'}

    def __repr__(self):
        """String representation."""
        detail = "<QNode: device='{}', func={}, wires={}>"
        return detail.format(self.device.short_name, self.func.__name__, self.num_wires)

    def print_applied(self):
        """Prints the most recently applied operations from the QNode.

        FIXME what's the purpose of this? Could be a CircuitGraph method.
        """
        if self.circuit is None:
            print("QNode has not yet been executed.")
            return

        print("Operations")
        print("==========")
        for op in self.circuit.operations:
            if op.parameters:
                params = ", ".join([str(p) for p in op.parameters])
                print("{}({}, wires={})".format(op.name, params, op.wires))
            else:
                print("{}(wires={})".format(op.name, op.wires))

        return_map = {
            qml.operation.Expectation: "expval",
            qml.operation.Variance: "var",
            qml.operation.Sample: "sample",
        }

        print("\nObservables")
        print("===========")
        for op in self.circuit.observables:
            return_type = return_map[op.return_type]
            if op.parameters:
                params = "".join([str(p) for p in op.parameters])
                print("{}({}({}, wires={}))".format(return_type, op.name, params, op.wires))
            else:
                print("{}({}(wires={}))".format(return_type, op.name, op.wires))

    def _set_variables(self, args, kwargs):
        """Store the current values of the quantum function parameters in the Variable class
        so the Operators may access them.

        Args:
            args (tuple[Any]): positional (differentiable) arguments
            kwargs (dict[str, Any]): auxiliary arguments
        """
        Variable.positional_arg_values = np.array(list(_flatten(args)))
        if not self.mutable:
            # only immutable circuits access auxiliary arguments through Variables
            Variable.kwarg_values = {k: np.array(list(_flatten(v))) for k, v in kwargs.items()}

    def _op_descendants(self, op, only):
        """Descendants of the given operator in the quantum circuit.

        Args:
            op (Operator): operator in the quantum circuit
            only (str, None): the type of descendants to return.

                - ``'G'``: only return non-observables (default)
                - ``'O'``: only return observables
                - ``None``: return all descendants

        Returns:
            list[Operator]: descendants in a topological order
        """
        succ = self.circuit.descendants_in_order((op,))
        if only == "O":
            return list(filter(_is_observable, succ))
        if only == "G":
            return list(itertools.filterfalse(_is_observable, succ))
        return succ

    def _remove_op(self, op):
        """Remove a quantum operation from the circuit queue.

        Args:
            op (:class:`~.operation.Operation`): quantum operation to be removed from the circuit
        """
        self.queue.remove(op)

    def _append_op(self, op):
        """Append a quantum operation into the circuit queue.

        Args:
            op (:class:`~.operation.Operation`): quantum operation to be added to the circuit

        Raises:
            ValueError: if `op` does not act on all wires
            QuantumFunctionError: if state preparations and gates do not precede measured observables
        """
        if op.num_wires == Wires.All:
            if set(op.wires) != set(range(self.num_wires)):
                raise QuantumFunctionError("Operator {} must act on all wires".format(op.name))

        # Make sure only existing wires are used.
        for w in _flatten(op.wires):
            if w < 0 or w >= self.num_wires:
                raise QuantumFunctionError(
                    "Operation {} applied to invalid wire {} "
                    "on device with {} wires.".format(op.name, w, self.num_wires)
                )

        # observables go to their own, temporary queue
        if isinstance(op, Observable):
            if op.return_type is None:
                self.queue.append(op)
            else:
                self.obs_queue.append(op)
        else:
            if self.obs_queue:
                raise QuantumFunctionError(
                    "State preparations and gates must precede measured observables."
                )
            self.queue.append(op)  # TODO rename self.queue to self.op_queue

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

        Raises:
            QuantumFunctionError: if the :class:`pennylane.QNode`'s _current_context is attempted to be modified
                inside of this method, the quantum function returns incorrect values or if
                both continuous and discrete operations are specified in the same quantum circuit
        """
        # pylint: disable=attribute-defined-outside-init, too-many-branches

        # flatten the args, replace each argument with a Variable instance carrying a unique index
        arg_vars = [Variable(idx) for idx, _ in enumerate(_flatten(args))]
        self.num_variables = len(arg_vars)
        # arrange the newly created Variables in the nested structure of args
        arg_vars = unflatten(arg_vars, args)

        # temporary queues for operations and observables
        self.queue = []  #: list[Operation]: applied operations
        self.obs_queue = []  #: list[Observable]: applied observables

        # set up the context for Operator entry
        if qml._current_context is None:
            qml._current_context = self
        else:
            raise QuantumFunctionError(
                "qml._current_context must not be modified outside this method."
            )
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
            qml._current_context = None

        # check the validity of the circuit
        self._check_circuit(res)

        # map each free variable to the operators which depend on it
        self.variable_deps = {k: [] for k in range(self.num_variables)}
        for k, op in enumerate(self.ops):
            for j, p in enumerate(_flatten(op.params)):
                if isinstance(p, Variable):
                    if p.name is None:  # ignore auxiliary arguments
                        self.variable_deps[p.idx].append(ParameterDependency(op, j))

        # generate the DAG
        self.circuit = CircuitGraph(self.ops, self.variable_deps)

        # check for unused positional params
        if self.properties.get("par_check", False):
            unused = [k for k, v in self.variable_deps.items() if not v]
            if unused:
                raise QuantumFunctionError(
                    "The positional parameters {} are unused.".format(unused)
                )

        # check for operations that cannot affect the output
        if self.properties.get("vis_check", False):
            visible = self.circuit.ancestors(self.circuit.observables)
            invisible = set(self.circuit.operations) - visible
            if invisible:
                raise QuantumFunctionError(
                    "The operations {} cannot affect the output of the circuit.".format(invisible)
                )

    def _check_circuit(self, res):
        """Check that the generated Operator queue corresponds to a valid quantum circuit.

        .. note:: The validity of individual Operators is checked already in :meth:`_append_op`.

        Args:
            res (Sequence[Observable], Observable): output returned by the quantum function

        Raises:
            QuantumFunctionError: an error was discovered in the circuit
        """
        # pylint: disable=too-many-branches

        # check the return value
        if isinstance(res, Observable):
            self.output_dim = 1

            if res.return_type is ObservableReturnTypes.Sample:
                # Squeezing ensures that there is only one array of values returned
                # when only a single-mode sample is requested
                self.output_conversion = np.squeeze
            elif res.return_type is ObservableReturnTypes.Probability:
                self.output_conversion = np.squeeze
                self.output_dim = 2**len(res.wires)
            else:
                self.output_conversion = float

            res = (res,)

        elif isinstance(res, Sequence) and res and all(isinstance(x, Observable) for x in res):
            # for multiple observables values, any valid Python sequence of observables
            # (i.e., lists, tuples, etc) are supported in the QNode return statement.

            # Device already returns the correct numpy array, so no further conversion is required
            self.output_conversion = np.asarray
            self.output_dim = len(res)
            res = tuple(res)
        else:
            raise QuantumFunctionError(
                "A quantum function must return either a single measured observable "
                "or a nonempty sequence of measured observables."
            )

        # check that all returned observables have a return_type specified
        for x in res:
            if x.return_type is None:
                raise QuantumFunctionError(
                    "Observable '{}' does not have the measurement type specified.".format(x)
                )

        # check that all ev's are returned, in the correct order
        if res != tuple(self.obs_queue):
            raise QuantumFunctionError(
                "All measured observables must be returned in the order they are measured."
            )

        # check that no wires are measured more than once
        m_wires = list(w for ob in res for w in _flatten(ob.wires))
        if len(m_wires) != len(set(m_wires)):
            raise QuantumFunctionError(
                "Each wire in the quantum circuit can only be measured once."
            )

        # True if op is a CV, False if it is a discrete variable (Identity could be either)
        are_cvs = [
            isinstance(op, CV) for op in self.queue + list(res) if not isinstance(op, qml.Identity)
        ]

        if not all(are_cvs) and any(are_cvs):
            raise QuantumFunctionError(
                "Continuous and discrete operations are not allowed in the same quantum circuit."
            )

        if any(are_cvs) and self.model == "qubit":
            raise QuantumFunctionError(
                "Device {} is a qubit device; CV operations are not allowed.".format(
                    self.device.short_name
                )
            )

        if not all(are_cvs) and self.model == "cv":
            raise QuantumFunctionError(
                "Device {} is a CV device; qubit operations are not allowed.".format(
                    self.device.short_name
                )
            )

        queue = self.queue
        if self.device.operations:
            # replace operations in the queue with any decompositions if required
            queue = decompose_queue(self.queue, self.device)

        self.ops = queue + list(res)
        del self.queue
        del self.obs_queue

    def _default_args(self, kwargs):
        """Validate the quantum function arguments, apply defaults.

        Here we apply default values for the auxiliary parameters of :attr:`QNode.func`.

        Args:
            kwargs (dict[str, Any]): auxiliary arguments (given using the keyword syntax)

        Raises:
            QuantumFunctionError: some of the arguments are invalid

        Returns:
            dict[str, Any]: all auxiliary arguments (with defaults)
        """
        forbidden_kinds = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )

        # check the validity of kwargs items
        for name in kwargs:
            s = self.func.sig.get(name)
            if s is None:
                if self.func.var_keyword:
                    continue  # unknown parameter, but **kwargs will take it TODO should it?
                raise QuantumFunctionError("Unknown quantum function parameter '{}'.".format(name))

            default_parameter = s.par.default

            # The following is a check of the default parameter which works for numpy
            # arrays as well (if it is a numpy array, each element is checked separately).
            # FIXME why are numpy array default values not good automatically?
            correct_default_parameter = any(d == inspect.Parameter.empty for d in default_parameter)\
                                        if isinstance(default_parameter, np.ndarray)\
                                        else default_parameter == inspect.Parameter.empty
            if s.par.kind in forbidden_kinds or correct_default_parameter:
                raise QuantumFunctionError(
                    "Quantum function parameter '{}' cannot be given using the keyword syntax.".format(
                        name
                    )
                )

        # apply defaults
        for name, s in self.func.sig.items():
            default = s.par.default
            correct_default = all(d != inspect.Parameter.empty for d in default)\
                              if isinstance(default, np.ndarray)\
                              else default != inspect.Parameter.empty

            if correct_default:
                # meant to be given using keyword syntax
                kwargs.setdefault(name, default)

        return kwargs

    def __call__(self, *args, **kwargs):
        """Wrapper for :meth:`.BaseQNode.evaluate`.
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

        # temporarily store the parameter values in the Variable class
        self._set_variables(args, kwargs)

        if self.circuit is None or self.mutable:
            self._construct(args, kwargs)

        self.device.reset()

        if isinstance(self.device, qml.QubitDevice):
            ret = self.device.execute(self.circuit)
        else:
            ret = self.device.execute(
                self.circuit.operations, self.circuit.observables, self.variable_deps
            )
        return self.output_conversion(ret)

    def evaluate_obs(self, obs, args, kwargs):
        """Evaluate the value of the given observables.

        Assumes :meth:`construct` has already been called.

        Args:
            obs  (Iterable[Observable]): observables to measure
            args (tuple[Any]): positional arguments to the quantum function (differentiable)
            kwargs (dict[str, Any]): auxiliary arguments (not differentiable)

        Returns:
            array[float]: measured values
        """
        kwargs = self._default_args(kwargs)

        # temporarily store the parameter values in the Variable class
        self._set_variables(args, kwargs)

        self.device.reset()

        if isinstance(self.device, qml.QubitDevice):
            # create a circuit graph containing the existing operations, and the
            # observables to be evaluated.
            circuit_graph = CircuitGraph(self.circuit.operations + list(obs),
                                         self.circuit.variable_deps)
            ret = self.device.execute(circuit_graph)
        else:
            ret = self.device.execute(self.circuit.operations, obs, self.circuit.variable_deps)
        return ret
