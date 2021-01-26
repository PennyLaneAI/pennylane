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
from pennylane.operation import Observable, CV, WiresEnum, ObservableReturnTypes
from pennylane.utils import _flatten, unflatten
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
            decomposed_ops = op.decomposition(*op.data, wires=op.wires)
            if op.inverse:
                decomposed_ops = qml.inv(decomposed_ops)

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
        except NotImplementedError as e:
            raise qml.DeviceError(
                "Gate {} not supported on device {}".format(op.name, device.short_name)
            ) from e

    return new_ops


class BaseQNode(qml.QueuingContext):
    """Base class for quantum nodes in the hybrid computational graph.

    A *quantum node* encapsulates a :ref:`quantum function <intro_vcirc_qfunc>`
    (corresponding to a :ref:`variational circuit <glossary_variational_circuit>`)
    and the computational device it is executed on.

    The QNode calls the quantum function to construct a :class:`.CircuitGraph` instance representing
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

    Keyword Args:
        vis_check (bool): whether to check for operations that cannot affect the output
        par_check (bool): whether to check for unused positional params
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, func, device, *, mutable=True, **kwargs):
        self.func = func  #: callable: quantum function
        self.device = device  #: Device: device that executes the circuit
        self.num_wires = device.num_wires  #: int: number of subsystems (wires) in the circuit
        #: int: number of flattened differentiable parameters in the circuit
        self.num_variables = None
        self.arg_vars = None
        self.kwarg_vars = None

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

        self._trainable_args = None

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
        """Prints the most recently applied operations from the QNode."""
        if self.circuit is None:
            print("QNode has not yet been executed.")
            return
        self.circuit.print_contents()

    def draw(
        self, charset="unicode", show_variable_names=False, wire_order=None, show_all_wires=False
    ):
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
            wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
            show_all_wires (bool): If True, all wires, including empty wires, are printed.

        Raises:
            ValueError: If the given charset is not supported
            pennylane.QuantumFunctionError: Drawing is impossible because the underlying CircuitGraph has not yet been constructed

        Returns:
            str: The circuit representation of the QNode
        """
        if self.circuit:
            return self.circuit.draw(
                charset=charset,
                show_variable_names=show_variable_names,
                wire_order=wire_order,
                show_all_wires=show_all_wires,
            )

        raise RuntimeError(
            "The QNode can only be drawn after its CircuitGraph has been constructed."
        )

    def set_trainable_args(self, arg_indices):
        """Store the indices of quantum function positional arguments
        that support differentiability.

        Args:
            args (None or Set[int]): Differentiable positional argument indices. A
                value of ``None`` means that all argument indices are differentiable.
        """
        if not self.mutable and self.circuit is not None:

            if self.get_trainable_args() == arg_indices:
                return

            raise QuantumFunctionError(
                "The trainability of arguments on immutable QNodes cannot be modified after the first evaluation."
            )

        if arg_indices is None:
            # all arguments are differentiable
            self._trainable_args = None
            return

        if not arg_indices:
            # The provided arg_indices are an empty set;
            # no arguments are differentiable.
            self._trainable_args = set()
            return

        # Perform validation
        if not self.func.var_pos and max(arg_indices) > self.func.n_pos:
            # QNode does not allow variable positional arguments (*args), and
            # the provided index set contains a value larger than the number
            # of positional arguments.
            raise ValueError(
                f"Argument index not available. QNode has at most {self.func.n_pos} arguments."
            )

        if any(not isinstance(i, int) or i < 0 for i in arg_indices):
            raise ValueError("Argument indices must be positive integers.")

        self._trainable_args = arg_indices

    def get_trainable_args(self):
        """Returns the indices of quantum function positional arguments
        that support differentiability.

        Returns:
            None or Set[int]: Differentiable positional argument indices. A
                value of ``None`` means that all argument indices are differentiable.
        """
        return self._trainable_args

    def _set_variables(self, args, kwargs):
        """Store the current values of the quantum function parameters in the Variable class
        so the Operators may access them.

        Args:
            args (tuple[Any]): positional (differentiable) arguments
            kwargs (dict[str, Any]): auxiliary arguments
        """
        Variable.positional_arg_values = list(_flatten(args))
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

    def _remove(self, obj):
        if isinstance(obj, Observable) and obj.return_type is not None:
            self.obs_queue.remove(obj)
        else:
            self.queue.remove(obj)

    def _append(self, obj):
        if obj.num_wires == WiresEnum.AllWires:
            # check here only if enough wires
            if len(obj.wires) != self.num_wires:
                raise QuantumFunctionError("Operator {} must act on all wires".format(obj.name))

        # Make sure only existing wires are used.
        for w in obj.wires:
            if w not in self.device.wires:
                raise QuantumFunctionError(
                    "Operation {} applied to invalid wire {} "
                    "on device with wires {}.".format(obj.name, w, self.device.wires.labels)
                )

        # observables go to their own, temporary queue
        if isinstance(obj, Observable):
            if obj.return_type is None:
                self.queue.append(obj)
            else:
                self.obs_queue.append(obj)
        else:
            if self.obs_queue:
                raise QuantumFunctionError(
                    "State preparations and gates must precede measured observables."
                )
            self.queue.append(obj)

    def _determine_structured_variable_name(self, parameter_value, prefix):
        """Determine the variable names corresponding to a parameter.

        This method unrolls the parameter value if it has an array
        or list structure.

        Args:
            parameter_value (Union[Number, Sequence[Any], array[Any]]): The value of the parameter. This will be used
                as a blueprint for the returned variable name(s).
            prefix (str): Prefix that will be added to the variable name(s), usually the parameter name

        Returns:
            Union[str,Sequence[str],array[str]]: The variable name(s) in the same structure as the parameter value
        """
        if isinstance(parameter_value, np.ndarray):
            variable_name_string = np.empty_like(parameter_value, dtype=object)

            for index in np.ndindex(*variable_name_string.shape):
                variable_name_string[index] = "{}[{}]".format(
                    prefix, ",".join([str(i) for i in index])
                )
        elif isinstance(parameter_value, Sequence):
            variable_name_string = []

            for idx, val in enumerate(parameter_value):
                variable_name_string.append(
                    self._determine_structured_variable_name(val, "{}[{}]".format(prefix, idx))
                )
        else:
            variable_name_string = prefix

        return variable_name_string

    def _make_variables(self, args, kwargs):
        """Create the :class:`~.variable.Variable` instances representing the QNode's arguments.

        The created :class:`~.variable.Variable` instances are given in the same nested structure
        as the original arguments. The :class:`~.variable.Variable` instances are named according
        to the argument names given in the QNode definition. Consider the following example:

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
                During the construction we are not concerned with the numerical values, but with
                the nesting structure.
                Each positional argument is replaced with a :class:`~.variable.Variable` instance.
            kwargs (dict[str, Any]): Auxiliary arguments passed to the quantum function.
        """
        kwargs = self.unwrap_tensor_kwargs(kwargs)

        # Get the name of the qfunc's arguments
        full_argspec = inspect.getfullargspec(self.func)

        # args
        variable_name_strings = []
        for variable_name, variable_value in zip(full_argspec.args, args):
            variable_name_strings.append(
                self._determine_structured_variable_name(variable_value, variable_name)
            )

        # varargs
        len_diff = len(args) - len(full_argspec.args)
        if len_diff > 0:
            for idx, variable_value in enumerate(args[-len_diff:]):
                variable_name = "{}[{}]".format(full_argspec.varargs, idx)

                variable_name_strings.append(
                    self._determine_structured_variable_name(variable_value, variable_name)
                )

        arg_vars = [Variable(idx, name) for idx, name in enumerate(_flatten(variable_name_strings))]
        self.num_variables = len(arg_vars)

        # Arrange the newly created Variables in the nested structure of args.
        # Make sure that NumPy scalars are converted into Python scalars.
        arg_vars = [
            i.item() if isinstance(i, np.ndarray) and i.ndim == 0 else i
            for i in unflatten(arg_vars, args)
        ]

        if self._trainable_args is not None and len(self._trainable_args) != len(args):
            # If some of the input arguments are marked as non-differentiable,
            # then replace the variable instances in arg_vars back with the
            # original objects.
            for a, _ in enumerate(args):
                if a not in self._trainable_args:
                    arg_vars[a] = args[a]

        # kwargs
        # if not mutable: must convert auxiliary arguments to named Variables so they can be updated without re-constructing the circuit
        # kwarg_vars = {}
        # for key, val in kwargs.items():
        #    temp = [Variable(idx, name=key) for idx, _ in enumerate(_flatten(val))]
        #    kwarg_vars[key] = unflatten(temp, val)

        variable_name_strings = {}
        kwarg_vars = {}
        for variable_name in full_argspec.kwonlyargs:
            if variable_name in kwargs:
                variable_value = kwargs[variable_name]
            else:
                variable_value = full_argspec.kwonlydefaults[variable_name]

            if isinstance(variable_value, np.ndarray):
                variable_name_string = np.empty_like(variable_value, dtype=object)

                for index in np.ndindex(*variable_name_string.shape):
                    variable_name_string[index] = "{}[{}]".format(
                        variable_name, ",".join([str(i) for i in index])
                    )

                kwarg_variable = [
                    Variable(idx, name=name, is_kwarg=True)
                    for idx, name in enumerate(_flatten(variable_name_string))
                ]
            else:
                kwarg_variable = Variable(0, name=variable_name, is_kwarg=True)

            kwarg_vars[variable_name] = kwarg_variable

        return arg_vars, kwarg_vars

    @staticmethod
    def unwrap_tensor_kwargs(kwargs):
        """Unwraps the pennylane.numpy.tensor objects that were passed as
        keyword arguments so that they can be handled as gate parameters by
        arbitrary devices.

        Args:
            kwargs (dict[str, Any]): Auxiliary arguments passed to the quantum function.

        Returns:
            dict[str, Any]: Auxiliary arguments passed to the quantum function
            in an unwrapped form (if applicable).
        """
        for k, v in kwargs.items():
            if isinstance(v, qml.numpy.tensor):
                kwargs[k] = v.unwrap()

        return kwargs

    def _construct(self, args, kwargs):
        """Construct the quantum circuit graph by calling the quantum function.

        For immutable nodes this method is called the first time :meth:`BaseQNode.evaluate`
        or :meth:`.JacobianQNode.jacobian` is called, and for mutable nodes *each time*
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
                Each positional argument is replaced with a :class:`~.Variable` instance.
            kwargs (dict[str, Any]): Auxiliary arguments passed to the quantum function.
        """
        # TODO: Update the docstring to reflect the kwargs and the raising conditions
        # pylint: disable=attribute-defined-outside-init, too-many-branches, too-many-statements

        self.arg_vars, self.kwarg_vars = self._make_variables(args, kwargs)

        # temporary queues for operations and observables
        # TODO rename self.queue to self.op_queue
        self.queue = []  #: list[Operation]: applied operations
        self.obs_queue = []  #: list[Observable]: applied observables

        tape_mode = qml.tape_mode_active()
        if tape_mode:
            qml.disable_tape()

        try:
            # set up the context for Operator entry
            with self:
                try:
                    # generate the program queue by executing the quantum circuit function
                    if self.mutable:
                        # it's ok to directly pass auxiliary arguments since the circuit is
                        # re-constructed each time (positional args must be replaced because
                        # parameter-shift differentiation requires Variables)
                        res = self.func(*self.arg_vars, **kwargs)
                    else:
                        # TODO: Maybe we should only convert the kwarg_vars that were actually given
                        res = self.func(*self.arg_vars, **self.kwarg_vars)
                except:
                    # The qfunc call may have failed because the user supplied bad parameters,
                    # which is why we must wipe the created Variables.
                    self.arg_vars = None
                    self.kwarg_vars = None
                    raise
        finally:
            if tape_mode:
                qml.enable_tape()

        # check the validity of the circuit
        self._check_circuit(res)
        del self.queue
        del self.obs_queue

        # Prune all the Tensor objects that have been used in the circuit
        self.ops = self._prune_tensors(self.ops)

        # map each free variable to the operators which depend on it
        self.variable_deps = {k: [] for k in range(self.num_variables)}
        for op in self.ops:
            for j, p in enumerate(_flatten(op.data)):
                if isinstance(p, Variable):
                    if not p.is_kwarg:  # ignore auxiliary arguments
                        self.variable_deps[p.idx].append(ParameterDependency(op, j))

        # generate the DAG
        self.circuit = CircuitGraph(self.ops, self.variable_deps, self.device.wires)

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

                if self.model == "qubit":
                    num_basis_states = 2
                elif self.model == "cv":
                    num_basis_states = getattr(self.device, "cutoff", 10)

                self.output_dim = num_basis_states ** len(res.wires)

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
        m_wires = list(w for ob in res for w in ob.wires)
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

            default_parameter = s.par.default

            # The following is a check of the default parameter which works for numpy
            # arrays as well (if it is a numpy array, each element is checked separately).
            # FIXME why are numpy array default values not good automatically?
            correct_default_parameter = (
                any(d == inspect.Parameter.empty for d in default_parameter)
                if isinstance(default_parameter, np.ndarray)
                else default_parameter == inspect.Parameter.empty
            )
            if s.par.kind in forbidden_kinds or correct_default_parameter:
                raise QuantumFunctionError(
                    "Quantum function parameter '{}' cannot be given using the keyword syntax.".format(
                        name
                    )
                )

        # apply defaults
        for name, s in self.func.sig.items():
            default = s.par.default
            correct_default = (
                all(d != inspect.Parameter.empty for d in default)
                if isinstance(default, np.ndarray)
                else default != inspect.Parameter.empty
            )

            if correct_default:
                # meant to be given using keyword syntax
                kwargs.setdefault(name, default)

        return kwargs

    def __call__(self, *args, **kwargs):
        """Wrapper for :meth:`BaseQNode.evaluate`."""
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
                self.circuit.operations + list(obs),
                self.circuit.variable_deps,
                self.device.wires,
            )
            ret = self.device.execute(circuit_graph)
        else:
            ret = self.device.execute(self.circuit.operations, obs, self.circuit.variable_deps)
        return ret
