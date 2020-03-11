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
from collections.abc import Sequence, Iterable
from collections import namedtuple, OrderedDict
import copy
import inspect
import itertools

import numpy as np

import pennylane as qml
from pennylane.operation import Observable, CV, Wires, ObservableReturnTypes
from pennylane.utils import _flatten, unflatten, equal_nested
from pennylane.circuit_graph import CircuitGraph, _is_observable
from pennylane.variable import Variable


ParameterDependency = namedtuple("ParameterDependency", ["op", "par_idx"])
"""Represents the dependence of an Operator on a positional parameter of the quantum function.

Args:
    op (Operator): operator depending on the positional parameter in question
    par_idx (int): flattened operator parameter index of the corresponding
        :class:`~.Variable` instance
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
    * :attr:`func.n_pos`: int: number of positional parameters
    * :attr:`func.n_pos_nd`: int: number of positional parameters without default value
    * :attr:`func.var_pos`: None, str: name of the var-positional parameter (``*args``), or None if there isn't one
    * :attr:`func.var_keyword`: None, str: name of the var-keyword parameter (``**kwargs``), or None if there isn't one
    * :attr:`func.aux_defaults`: dict[str, Any]: default values for the auxiliary parameters

    Args:
        func (callable): function to introspect
    """
    sig = inspect.signature(func)
    # count positional args, see if VAR_ args are present
    func.n_pos = 0
    func.n_pos_nd = 0
    func.var_pos = None
    func.var_keyword = None
    func.aux_defaults = {}
    for p in sig.parameters.values():
        if p.kind <= inspect.Parameter.POSITIONAL_OR_KEYWORD:
            func.n_pos += 1
            if p.default is inspect.Parameter.empty:
                # positional parameter with no default value
                func.n_pos_nd += 1
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            func.var_pos = p.name
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            func.var_keyword = p.name

        # record the default values for auxiliary parameters in a dict
        if p.default is not inspect.Parameter.empty:
            # default value => auxiliary parameter
            func.aux_defaults[p.name] = p.default

    func.sig = OrderedDict(
        [(p.name, SignatureParameter(idx, p)) for idx, p in enumerate(sig.parameters.values())]
    )


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
            raise qml.DeviceError(
                "Gate {} not supported on device {}".format(op.name, device.short_name)
            )

    return new_ops


class BaseQNode:
    """Base class for quantum nodes in the hybrid computational graph.

    A *quantum node* encapsulates a :ref:`quantum function <intro_vcirc_qfunc>`
    (corresponding to a :ref:`variational circuit <glossary_variational_circuit>`)
    and the computational device it is executed on.

    The quantum function can take two kinds of classical input parameters:
    *positional* and *auxiliary*.
    Parameters that have default values in the quantum function signature are interpreted as
    auxiliary parameters. All other parameters are positional.

    * The quantum function can *only* be differentiated with respect to its positional parameters.
      The positional parameters should be only used as the parameters of the
      :class:`quantum operations <.Operator>` in the function,
      and they must be real numbers, or nested ``Iterables`` of real numbers
      (note that this includes NumPy arrays).
      Classical processing of positional parameters, either by arithmetic operations
      or external functions, is not allowed. One current exception is simple scalar multiplication or division.
      Positional parameters *must* be given using the positional syntax.

    * The quantum function can *not* be differentiated with respect to auxiliary parameters.
      They are useful for providing data or 'placeholders' to the quantum function.
      Auxiliary parameters *must* be given using the keyword syntax.

    The QNode calls the quantum function to construct a :class:`.CircuitGraph` instance represeting
    the quantum circuit. The circuit can be either

    * *mutable*, which means the quantum function is called each time the QNode is evaluated, or
    * *immutable*, which means the quantum function is called only once, on first evaluation,
      to construct the circuit representation.

    Iff the circuit is mutable, its **auxiliary** parameters can

    * undergo any kind of classical processing inside the quantum function,
    * be used in classical flow control structures (such as ``if`` statements and ``for`` loops),
    * determine the wires on which operators act.

    The last two uses can potentially result in a different circuit structure on each call, hence
    the term mutable. On the other hand,
    for immutable circuits the quantum function must build the same circuit graph consisting of the same
    :class:`.Operator` instances regardless of its parameters; they can only appear as the
    arguments of the Operators in the circuit. Immutable circuits are slightly faster to execute, and
    can be optimized, but require that the layout of the circuit is fixed.

    Args:
        func (callable): The *quantum function* of the QNode.
            A Python function containing :class:`~.operation.Operation` constructor calls,
            and returning a tuple of measured :class:`~.operation.Observable` instances.
        device (~pennylane._device.Device): computational device to execute the function on
        mutable (bool): whether the circuit is mutable, see above

    Keyword Args:
        vis_check (bool): whether to check for operations that cannot affect the output
        par_check (bool): whether to check for unused positional params
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, func, device, *, mutable=True, **kwargs):
        self.func = func  #: callable: quantum function
        self.device = device  #: Device: device that executes the circuit
        self.num_wires = device.num_wires  #: int: number of subsystems (wires) in the circuit
        #: int: number of (flattened) differentiable parameters in the circuit
        self.num_primary_parameters = None
        #: dict[str, Any]: deepcopy of the auxiliary arguments the qnode was passed on last call
        self.last_auxiliary_args = None

        #: List[Operator]: quantum circuit, in the order the quantum function defines it
        self.ops = []

        self.circuit = None  #: CircuitGraph: DAG representation of the quantum circuit

        self.mutable = mutable  #: bool: whether the circuit is mutable
        #: dict[str, Any]: additional keyword kwargs for adjusting the QNode behavior
        self.kwargs = kwargs or {}

        self.variable_deps = {}
        """dict[int, list[ParameterDependency]]: Mapping from flattened qfunc positional parameter
        index to the list of :class:`~pennylane.operation.Operator` instances (in this circuit)
        that depend on it.
        """

        self._metric_tensor_subcircuits = None
        """dict[tuple[int], dict[str, Any]]: circuit descriptions for computing the metric tensor"""

        # introspect the quantum function signature
        _get_signature(self.func)

        #: callable: for transforming the output of :meth:`.Device.execute` to QNode output
        self.output_conversion = None
        self.output_dim = None  #: int: dimension of the QNode output vector
        self.model = self.device.capabilities()["model"]  #: str: circuit type, in {'cv', 'qubit'}

    def __repr__(self):
        """String representation."""
        detail = "<QNode: device='{}', func={}, wires={}>"
        return detail.format(self.device.short_name, self.func.__name__, self.num_wires)

    def __enter__(self):
        """Make this node the current execution context for quantum functions.
        """
        if qml._current_context is None:
            qml._current_context = self
        else:
            raise QuantumFunctionError(
                "qml._current_context must not be modified outside this method."
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Reset the quantum function execution context to None.
        """
        qml._current_context = None

    def print_applied(self):
        """Prints the most recently applied operations from the QNode.
        """
        if self.circuit is None:
            print("QNode has not yet been executed.")
            return
        self.circuit.print_contents()

    def draw(self, charset="unicode", show_variable_names=False):
        """Draw the QNode as a circuit diagram.

        Consider the following circuit as an example:

        .. code-block:: python3

            @qml.qnode(dev)
            def qfunc(a, w):
                qml.Hadamard(0)
                qml.CRX(a, wires=[0, 1])
                qml.Rot(w[0], w[1], w[2], wires=[1])
                qml.CRX(-a, wires=[0, 1])

                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        We can draw the circuit after it has been executed:

        .. code-block:: python

            >>> result = qfunc(2.3, [1.2, 3.2, 0.7])
            >>> print(qfunc.draw())
            0: ──H──╭C────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩
            1: ─────╰RX(2.3)──Rot(1.2, 3.2, 0.7)──╰RX(-2.3)──╰┤ ⟨Z ⊗ Z⟩
            >>> print(qfunc.draw(charset="ascii"))
            0: --H--+C----------------------------+C---------+| <Z @ Z>
            1: -----+RX(2.3)--Rot(1.2, 3.2, 0.7)--+RX(-2.3)--+| <Z @ Z>
            >>> print(qfunc.draw(show_variable_names=True))
            0: ──H──╭C─────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩
            1: ─────╰RX(a)──Rot(w[0], w[1], w[2])──╰RX(-1*a)──╰┤ ⟨Z ⊗ Z⟩

        Args:
            charset (str, optional): The charset that should be used. Currently, "unicode" and "ascii" are supported.
            show_variable_names (bool, optional): Show variable names instead of values.

        Raises:
            ValueError: If the given charset is not supported
            pennylane.QuantumFunctionError: Drawing is impossible because the underlying CircuitGraph has not yet been constructed

        Returns:
            str: The circuit representation of the QNode
        """
        if self.circuit:
            return self.circuit.draw(charset=charset, show_variable_names=show_variable_names)

        raise RuntimeError(
            "The QNode can only be drawn after its CircuitGraph has been constructed."
        )

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

    def _determine_structured_variable_name(self, arg, prefix):
        """Determine the variable names corresponding to a parameter.

        This method unrolls the parameter value if it has an array
        or list structure.

        Args:
            arg (Union[Number, Sequence[Any], array[Any]]): The value of the parameter.
                This will be used as a blueprint for the returned variable name(s).
            prefix (str): Prefix that will be added to the variable name(s), usually the parameter name

        Returns:
            Union[str, Sequence[str], array[str]]: variable name(s) in the same nested structure
                as the parameter value
        """
        if isinstance(arg, np.ndarray):
            variable_names = np.empty_like(arg, dtype=object)

            for index in np.ndindex(*variable_names.shape):
                variable_names[index] = "{}[{}]".format(prefix, ",".join([str(i) for i in index]))
        elif isinstance(arg, Iterable) and not isinstance(arg, (str, bytes)):
            # recurse into the nested Iterable
            variable_names = []

            for idx, val in enumerate(arg):
                variable_names.append(
                    self._determine_structured_variable_name(val, "{}[{}]".format(prefix, idx))
                )
        else:
            variable_names = prefix

        return variable_names

    def _make_variables(self, args, kwargs):
        """Create the :class:`~.variable.Variable` instances representing the QNode's arguments.

        The created :class:`~.variable.Variable` instances are given in the same nested structure
        as the original arguments, and they are named according
        to the parameter names in the qfunc signature. Consider the following example:

        .. code-block:: python3

            @qml.qnode(dev)
            def qfunc(a, w):
                qml.Hadamard(0)
                qml.CRX(a, wires=[0, 1])
                qml.Rot(w[0], w[1], w[2], wires=[1])
                qml.CRX(-a, wires=[0, 1])

                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        In this example, ``_make_variables`` will return the following :class:`~.variable.Variable` instances

        .. code-block:: python3

            >>> qfunc(3.4, [1.2, 3.4, 5.6])
            -0.031664133410566786
            >>> qfunc._make_variables([3.4, [1.2, 3.4, 5.6]], {})
            ["a", ["w[0]", "w[1]", "w[2]"]], {}

        where the Variable instances are replaced with their name for readability.

        Args:
            args (tuple[Any]): Positional arguments passed to the quantum function.
            kwargs (dict[str, Any]): Auxiliary arguments passed to the quantum function.
                They are only represented with Variables if the circuit is immutable.

        Returns:
            nested list[Variable], Union[nested list[Variable], None]: references for args, references for kwargs
        """
        # positional args
        variable_names = []
        # the positional args must come first in the signature
        for prefix, a in zip(self.func.sig, args[: self.func.n_pos_nd]):
            variable_names.append(self._determine_structured_variable_name(a, prefix))

        # var-positional args come next
        n_var_pos = len(args) - self.func.n_pos_nd
        if n_var_pos > 0:
            variable_names.append(
                self._determine_structured_variable_name(args[-n_var_pos:], self.func.var_pos)
            )

        # create the Variables
        arg_vars = [Variable(idx, name) for idx, name in enumerate(_flatten(variable_names))]
        self.num_primary_parameters = len(arg_vars)

        # arrange the Variables in the nested structure of args
        arg_vars = unflatten(arg_vars, args)

        if self.mutable:
            # only immutable circuits use Variables for auxiliary args
            return arg_vars, None

        # auxiliary args
        kwarg_vars = {}
        for prefix, a in kwargs.items():
            variable_names = self._determine_structured_variable_name(a, prefix)
            temp = [
                Variable(idx, name, basename=prefix)
                for idx, name in enumerate(_flatten(variable_names))
            ]
            kwarg_vars[prefix] = unflatten(temp, a)

        return arg_vars, kwarg_vars

    def _construct(self, args, kwargs):
        """Construct the quantum circuit graph by calling the quantum function.

        For immutable nodes this method is called the first time :meth:`BaseQNode.evaluate`
        or :meth:`.JacobianQNode.jacobian` is called, and for mutable nodes *each time*
        they are called. It executes the quantum function,
        stores the resulting sequence of :class:`.Operator` instances,
        converts it into a circuit graph, and creates the Variable mapping.

        .. note::
           In mutable circuits the Variables are only used to represent the nested elements of
           the positional args. We reconstruct the circuit each time the auxiliary args change.

        Args:
            args (tuple[Any]): Positional arguments passed to the quantum function.
                During the construction we are not concerned with the numerical values, but with
                the nesting structure.
                Each atomic element ("scalar") within a nested positional argument is replaced with
                a :class:`~.Variable` instance.
            kwargs (dict[str, Any]): Auxiliary arguments passed to the quantum function.

        Raises:
            QuantumFunctionError: if :data:`pennylane._current_context` is attempted to be modified
                inside of this method, the quantum function returns incorrect values or if
                both continuous and discrete operations are specified in the same quantum circuit
        """
        # pylint: disable=attribute-defined-outside-init, too-many-branches, too-many-statements

        def equal_dicts(a, b):
            """Return True iff a and b are equal dicts."""
            if b is None:
                return False

            # Defaults have been applied, a and b have the same keys.
            # Check that they have same nested values.
            return all(equal_nested(val, b[key]) for key, val in a.items())

        if equal_dicts(kwargs, self.last_auxiliary_args):
            return

        # If the auxiliary args have changed (or this is the first call),
        # we must (re)construct the circuit and the Variables.

        # make the Variables
        arg_vars, kwarg_vars = self._make_variables(args, kwargs)

        # temporary queues for operations and observables
        self.queue = []  #: list[Operation]: applied operations
        self.obs_queue = []  #: list[Observable]: applied observables

        # set up the context for Operator entry
        with self:
            try:
                # generate the program queue by executing the quantum circuit function
                if self.mutable:
                    # It's ok to directly pass auxiliary args since the circuit is re-constructed
                    # each time they change. Positional args must be replaced because parameter-shift
                    # differentiation requires Variables.
                    res = self.func(*arg_vars, **kwargs)
                else:
                    # immutable circuits are only constructed once, hence we must use Variables
                    res = self.func(*arg_vars, **kwarg_vars)
            except:
                # The qfunc call may have failed because the user supplied bad parameters, which is why we must wipe the created Variables.
                self.arg_vars = None
                self.kwarg_vars = None
                raise

        # check the validity of the circuit
        self._check_circuit(res)
        del self.queue
        del self.obs_queue

        # Prune all the Tensor objects that have been used in the circuit
        self.ops = self._prune_tensors(self.ops)

        # map each differentiable parameter to the operators which depend on it
        self.variable_deps = {k: [] for k in range(self.num_primary_parameters)}
        for op in self.ops:
            for j, p in enumerate(_flatten(op.params)):
                if isinstance(p, Variable):
                    if not p.is_kwarg:  # ignore auxiliary arguments
                        self.variable_deps[p.idx].append(ParameterDependency(op, j))

        # generate the DAG
        self.circuit = CircuitGraph(self.ops, self.variable_deps)

        # The qfunc call may fail for various reasons.
        # We only update the aux args here to ensure that they represent the current circuit.
        self.last_auxiliary_args = copy.deepcopy(kwargs)

        # check for unused positional params
        if self.kwargs.get("par_check", False):
            unused = [k for k, v in self.variable_deps.items() if not v]
            if unused:
                raise QuantumFunctionError(
                    "The positional parameters {} are unused.".format(unused)
                )

        # check for operations that cannot affect the output
        if self.kwargs.get("vis_check", False):
            invisible = self.circuit.invisible_operations()
            if invisible:
                raise QuantumFunctionError(
                    "The operations {} cannot affect the circuit output.".format(invisible)
                )

    @staticmethod
    def _prune_tensors(res):
        """Prune the tensors that have been passed by the quantum function.

        .. seealso:: :meth:`~.Tensor.prune`

        Args:
            res (Sequence[Observable], Observable): output returned by the quantum function

        Returns:
            res (Sequence[Observable], Observable): pruned output returned by the quantum function
        """
        if isinstance(res, qml.operation.Tensor):
            return res.prune()

        if isinstance(res, Sequence):
            ops = []
            for o in res:
                if isinstance(o, qml.operation.Tensor):
                    ops.append(o.prune())
                else:
                    ops.append(o)
            return ops

        return res

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
                self.output_dim = 2 ** len(res.wires)
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

    def _default_args(self, kwargs):
        """Validate the quantum function arguments, apply defaults.

        Here we apply default values for the auxiliary parameters of :attr:`BaseQNode.func`.

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

            if s.par.kind in forbidden_kinds or s.par.default is inspect.Parameter.empty:
                raise QuantumFunctionError(
                    "Quantum function parameter '{}' cannot be given using the keyword syntax.".format(
                        name
                    )
                )

        # apply default values
        return {**self.func.aux_defaults, **kwargs}

    def __call__(self, *args, **kwargs):
        """Wrapper for :meth:`BaseQNode.evaluate`.
        """
        return self.evaluate(args, kwargs)

    def evaluate(self, args, kwargs):
        """Evaluate the quantum function on the specified device.

        Args:
            args (tuple[Any]): positional arguments to the quantum function (differentiable)
            kwargs (dict[str, Any]): auxiliary arguments (not differentiable)

        Keyword Args:
            use_native_type (bool): If True, return the result in whatever type the device uses
                internally, otherwise convert it into array[float]. Default: False.

        Returns:
            float or array[float]: output measured value(s)
        """
        kwargs = self._default_args(kwargs)
        self._set_variables(args, kwargs)

        if self.circuit is None or self.mutable:
            self._construct(args, kwargs)

        self.device.reset()

        temp = self.kwargs.get("use_native_type", False)
        if isinstance(self.device, qml.QubitDevice):
            # TODO: remove this if statement once all devices are ported to the QubitDevice API
            ret = self.device.execute(self.circuit, return_native_type=temp)
        else:
            ret = self.device.execute(
                self.circuit.operations,
                self.circuit.observables,
                self.variable_deps,
                return_native_type=temp,
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
        self._set_variables(args, kwargs)

        self.device.reset()

        if isinstance(self.device, qml.QubitDevice):
            # create a circuit graph containing the existing operations, and the
            # observables to be evaluated.
            circuit_graph = CircuitGraph(
                self.circuit.operations + list(obs), self.circuit.variable_deps
            )
            ret = self.device.execute(circuit_graph)
        else:
            ret = self.device.execute(self.circuit.operations, obs, self.circuit.variable_deps)
        return ret
