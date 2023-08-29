# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This submodule defines the symbolic operation that indicates the control of an operator.
"""
import warnings
from copy import copy
from functools import wraps
from inspect import signature
from typing import List

import numpy as np
from scipy import sparse

import pennylane as qml
from pennylane import operation
from pennylane import math as qmlmath
from pennylane.operation import Operator
from pennylane.wires import Wires

from .symbolicop import SymbolicOp


def ctrl(op, control, control_values=None, work_wires=None):
    """Create a method that applies a controlled version of the provided op.

    Args:
        op (function or :class:`~.operation.Operator`): A single operator or a function that applies pennylane operators.
        control (Wires): The control wire(s).
        control_values (bool or list[bool]): The value(s) the control wire(s) should take.
            Integers other than 0 or 1 will be treated as ``int(bool(x))``.
        work_wires (Any): Any auxiliary wires that can be used in the decomposition

    Returns:
        (function or :class:`~.operation.Operator`): If an Operator is provided, returns a Controlled version of the Operator.
        If a function is provided, returns a function with the same call signature that creates a controlled version of the
        provided function.

    .. seealso:: :class:`~.Controlled`.

    **Example**

    .. code-block:: python3

        @qml.qnode(qml.device('default.qubit', wires=range(4)))
        def circuit(x):
            qml.PauliX(2)
            qml.ctrl(qml.RX, (1,2,3), control_values=(0,1,0))(x, wires=0)
            return qml.expval(qml.PauliZ(0))

    >>> print(qml.draw(circuit)("x"))
    0: ────╭RX(x)─┤  <Z>
    1: ────├○─────┤
    2: ──X─├●─────┤
    3: ────╰○─────┤
    >>> x = np.array(1.2)
    >>> circuit(x)
    tensor(0.36235775, requires_grad=True)
    >>> qml.grad(circuit)(x)
    -0.9320390859672264

    :func:`~.ctrl` works on both callables like ``qml.RX`` or a quantum function
    and individual :class:`~.operation.Operator`'s.

    >>> qml.ctrl(qml.Hadamard(0), (1,2))
    Controlled(Hadamard(wires=[0]), control_wires=[1, 2])

    Controlled operations work with all other forms of operator math and simplification:

    >>> op = qml.ctrl(qml.RX(1.2, wires=0) ** 2 @ qml.RY(0.1, wires=0), control=1)
    >>> qml.simplify(qml.adjoint(op))
    Controlled(RY(12.466370614359173, wires=[0]) @ RX(10.166370614359172, wires=[0]), control_wires=[1])

    """
    custom_controlled_ops = {
        (qml.PauliZ, 1): qml.CZ,
        (qml.PauliY, 1): qml.CY,
        (qml.PauliX, 1): qml.CNOT,
        (qml.PauliX, 2): qml.Toffoli,
    }
    control_values = [control_values] if isinstance(control_values, (int, bool)) else control_values
    control = qml.wires.Wires(control)
    custom_key = (type(op), len(control))

    if custom_key in custom_controlled_ops and (control_values is None or all(control_values)):
        qml.QueuingManager.remove(op)
        return custom_controlled_ops[custom_key](control + op.wires)
    if isinstance(op, qml.PauliX):
        qml.QueuingManager.remove(op)
        control_string = (
            None if control_values is None else "".join([str(int(v)) for v in control_values])
        )
        return qml.MultiControlledX(
            wires=control + op.wires, control_values=control_string, work_wires=work_wires
        )
    if isinstance(op, Operator):
        return Controlled(
            op, control_wires=control, control_values=control_values, work_wires=work_wires
        )
    if not callable(op):
        raise ValueError(
            f"The object {op} of type {type(op)} is not an Operator or callable. "
            "This error might occur if you apply ctrl to a list "
            "of operations instead of a function or Operator."
        )

    @wraps(op)
    def wrapper(*args, **kwargs):
        qscript = qml.tape.make_qscript(op)(*args, **kwargs)

        # flip control_values == 0 wires here, so we don't have to do it for each individual op.
        flip_control_on_zero = (len(qscript) > 1) and (control_values is not None)
        op_control_values = None if flip_control_on_zero else control_values
        if flip_control_on_zero:
            _ = [qml.PauliX(w) for w, val in zip(control, control_values) if not val]

        _ = [
            ctrl(op, control=control, control_values=op_control_values, work_wires=work_wires)
            for op in qscript.operations
        ]

        if flip_control_on_zero:
            _ = [qml.PauliX(w) for w, val in zip(control, control_values) if not val]

        if qml.QueuingManager.recording():
            _ = [qml.apply(m) for m in qscript.measurements]

        return qscript.measurements

    return wrapper

def ctrl_evolution(op, control):

    """Create a method that applies a controlled version of the provided op.

    Args:
        op (function or :class:`~.operation.Operator`): A single operator or a function that applies pennylane operators.
        control (Wires): The control wire(s).
        work_wires (Any): Any auxiliary wires that can be used in the decomposition

    Returns:
        (function or :class:`~.operation.Operator`): If an Operator is provided, returns a Controlled version of the Operator.
        If a function is provided, returns a function with the same call signature that creates a controlled version of the
        provided function.

    .. seealso:: :class:`~.Controlled`.

    """

    control = [control] if isinstance(control, (int, bool)) else control
    ops = []

    if isinstance(op, Operator):
        for ind, c in enumerate(control):
                ops.append(qml.ctrl(qml.pow(op, z = 2 ** (len(control) - ind - 1)), control = c))
        return ops

    if not callable(op):
        raise ValueError(
            f"The object {op} of type {type(op)} is not an Operator or callable. "
            "This error might occur if you apply ctrl to a list "
            "of operations instead of a function or Operator."
        )

    #TODO: Ajustar esto al operador que queremos
    @wraps(op)
    def wrapper(*args, **kwargs):
        qscript = qml.tape.make_qscript(op)(*args, **kwargs)

        _ = [
            ctrl_evolution(op, control=control)
            for op in qscript.operations
        ]

        if qml.QueuingManager.recording():
            _ = [qml.apply(m) for m in qscript.measurements]

        return qscript.measurements

    return wrapper




# pylint: disable=too-many-arguments, too-many-public-methods
class Controlled(SymbolicOp):
    """Symbolic operator denoting a controlled operator.

    Args:
        base (~.operation.Operator): the operator that is controlled
        control_wires (Any): The wires to control on.

    Keyword Args:
        control_values (Iterable[Bool]): The values to control on. Must be the same
            length as ``control_wires``. Defaults to ``True`` for all control wires.
            Provided values are converted to `Bool` internally.
        work_wires (Any): Any auxiliary wires that can be used in the decomposition

    .. note::
        This class, ``Controlled``, denotes a controlled version of any individual operation.
        :class:`~.ControlledOp` adds :class:`~.Operation` specific methods and properties to the
        more general ``Controlled`` class.

    .. seealso:: :class:`~.ControlledOp`, and :func:`~.ctrl`

    **Example**

    >>> base = qml.RX(1.234, 1)
    >>> Controlled(base, (0, 2, 3), control_values=[True, False, True])
    Controlled(RX(1.234, wires=[1]), control_wires=[0, 2, 3], control_values=[True, False, True])
    >>> op = Controlled(base, 0, control_values=[0])
    >>> op
    Controlled(RX(1.234, wires=[1]), control_wires=[0], control_values=[0])

    The operation has both standard :class:`~.operation.Operator` properties
    and ``Controlled`` specific properties:

    >>> op.base
    RX(1.234, wires=[1])
    >>> op.data
    (1.234,)
    >>> op.wires
    <Wires = [0, 1]>
    >>> op.control_wires
    <Wires = [0]>
    >>> op.target_wires
    <Wires = [1]>

    Control values are lists of booleans, indicating whether or not to control on the
    ``0==False`` value or the ``1==True`` wire.

    >>> op.control_values
    [0]

    Provided control values are converted to booleans internally, so
    any "truthy" or "falsy" objects work.

    >>> Controlled(base, ("a", "b", "c"), control_values=["", None, 5]).control_values
    [False, False, True]

    Representations for an operator are available if the base class defines them.
    Sparse matrices are available if the base class defines either a sparse matrix
    or only a dense matrix.

    >>> np.set_printoptions(precision=4) # easier to read the matrix
    >>> qml.matrix(op)
    array([[0.8156+0.j    , 0.    -0.5786j, 0.    +0.j    , 0.    +0.j    ],
           [0.    -0.5786j, 0.8156+0.j    , 0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 1.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 1.    +0.j    ]])
    >>> qml.eigvals(op)
    array([1.    +0.j    , 1.    +0.j    , 0.8156+0.5786j, 0.8156-0.5786j])
    >>> print(qml.generator(op, format='observable'))
    (-0.5) [Projector0 X1]
    >>> op.sparse_matrix()
    <4x4 sparse matrix of type '<class 'numpy.complex128'>'
                with 6 stored elements in Compressed Sparse Row format>

    If the provided base matrix is an :class:`~.operation.Operation`, then the created
    object will be of type :class:`~.ops.op_math.ControlledOp`. This class adds some additional
    methods and properties to the basic :class:`~.ops.op_math.Controlled` class.

    >>> type(op)
    <class 'pennylane.ops.op_math.controlled_class.ControlledOp'>
    >>> op.parameter_frequencies
    [(0.5, 1.0)]

    """

    def _flatten(self):
        return (self.base,), (self.control_wires, tuple(self.control_values), self.work_wires)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(
            data[0], control_wires=metadata[0], control_values=metadata[1], work_wires=metadata[2]
        )

    # pylint: disable=no-self-argument
    @operation.classproperty
    def __signature__(cls):  # pragma: no cover
        # this method is defined so inspect.signature returns __init__ signature
        # instead of __new__ signature
        # See PEP 362

        # use __init__ signature instead of __new__ signature
        sig = signature(cls.__init__)
        # get rid of self from signature
        new_parameters = tuple(sig.parameters.values())[1:]
        new_sig = sig.replace(parameters=new_parameters)
        return new_sig

    # pylint: disable=unused-argument
    def __new__(cls, base, *_, **__):
        """If base is an ``Operation``, then a ``ControlledOp`` should be used instead."""
        if isinstance(base, operation.Operation):
            return object.__new__(ControlledOp)
        return object.__new__(Controlled)

    # pylint: disable=too-many-function-args
    def __init__(self, base, control_wires, control_values=None, work_wires=None, id=None):
        control_wires = Wires(control_wires)
        work_wires = Wires([]) if work_wires is None else Wires(work_wires)

        if control_values is None:
            control_values = [True] * len(control_wires)
        else:
            if isinstance(control_values, str):
                warnings.warn(
                    "Specifying control values as a string is deprecated. Please use Sequence[Bool]",
                    UserWarning,
                )
                # All values not 0 are cast as true. Assumes a string of 1s and 0s.
                control_values = [(x != "0") for x in control_values]

            control_values = (
                [bool(control_values)]
                if isinstance(control_values, int)
                else [bool(control_value) for control_value in control_values]
            )

            if len(control_values) != len(control_wires):
                raise ValueError("control_values should be the same length as control_wires")

        if len(Wires.shared_wires([base.wires, control_wires])) != 0:
            raise ValueError("The control wires must be different from the base operation wires.")

        if len(Wires.shared_wires([work_wires, base.wires + control_wires])) != 0:
            raise ValueError(
                "Work wires must be different the control_wires and base operation wires."
            )

        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["control_values"] = control_values
        self.hyperparameters["work_wires"] = work_wires

        self._name = f"C({base.name})"

        super().__init__(base, id)

    @property
    def hash(self):
        # these gates do not consider global phases in their hash
        if self.base.name in ("RX", "RY", "RZ", "Rot"):
            base_params = str(
                [qml.math.round(qml.math.real(d) % (4 * np.pi), 10) for d in self.base.data]
            )
            base_hash = hash(
                (
                    str(self.base.name),
                    tuple(self.base.wires.tolist()),
                    base_params,
                )
            )
        else:
            base_hash = self.base.hash
        return hash(
            (
                "Controlled",
                base_hash,
                tuple(self.control_wires.tolist()),
                tuple(self.control_values),
                tuple(self.work_wires.tolist()),
            )
        )

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return self.base.has_matrix

    # pylint: disable=protected-access
    def _check_batching(self, params):
        self.base._check_batching(params)

    @property
    def batch_size(self):
        return self.base.batch_size

    @property
    def ndim_params(self):
        return self.base.ndim_params

    # Properties on the control values ######################
    @property
    def control_values(self):
        """Iterable[Bool]. For each control wire, denotes whether to control on ``True`` or
        ``False``."""
        return self.hyperparameters["control_values"]

    @property
    def _control_int(self):
        """Int. Conversion of ``control_values`` to an integer."""
        return sum(2**i for i, val in enumerate(reversed(self.control_values)) if val)

    # Properties on the wires ##########################

    @property
    def control_wires(self):
        """The control wires."""
        return self.hyperparameters["control_wires"]

    @property
    def target_wires(self):
        """The wires of the target operator."""
        return self.base.wires

    @property
    def work_wires(self):
        """Additional wires that can be used in the decomposition. Not modified by the operation."""
        return self.hyperparameters["work_wires"]

    @property
    def active_wires(self):
        """Wires modified by the operator. This is the control wires followed by the target wires."""
        return self.control_wires + self.target_wires

    @property
    def wires(self):
        return self.control_wires + self.target_wires + self.work_wires

    def map_wires(self, wire_map: dict):
        new_base = self.base.map_wires(wire_map=wire_map)
        new_control_wires = Wires([wire_map.get(wire, wire) for wire in self.control_wires])
        new_work_wires = Wires([wire_map.get(wire, wire) for wire in self.work_wires])

        return ctrl(
            op=new_base,
            control=new_control_wires,
            control_values=self.control_values,
            work_wires=new_work_wires,
        )

    # Methods ##########################################

    def __repr__(self):
        params = [f"control_wires={self.control_wires.tolist()}"]
        if self.work_wires:
            params.append(f"work_wires={self.work_wires.tolist()}")
        if self.control_values and not all(self.control_values):
            params.append(f"control_values={self.control_values}")
        return f"Controlled({self.base}, {', '.join(params)})"

    def label(self, decimals=None, base_label=None, cache=None):
        return self.base.label(decimals=decimals, base_label=base_label, cache=cache)

    def matrix(self, wire_order=None):
        base_matrix = self.base.matrix()
        interface = qmlmath.get_interface(base_matrix)

        num_target_states = 2 ** len(self.target_wires)
        num_control_states = 2 ** len(self.control_wires)
        total_matrix_size = num_control_states * num_target_states

        padding_left = self._control_int * num_target_states
        padding_right = total_matrix_size - padding_left - num_target_states

        left_pad = qmlmath.convert_like(
            qmlmath.cast_like(qmlmath.eye(padding_left, like=interface), 1j), base_matrix
        )
        right_pad = qmlmath.convert_like(
            qmlmath.cast_like(qmlmath.eye(padding_right, like=interface), 1j), base_matrix
        )

        shape = qml.math.shape(base_matrix)
        if len(shape) == 3:  # stack if batching
            canonical_matrix = qml.math.stack(
                [qml.math.block_diag([left_pad, _U, right_pad]) for _U in base_matrix]
            )
        else:
            canonical_matrix = qmlmath.block_diag([left_pad, base_matrix, right_pad])

        if wire_order is None or self.wires == Wires(wire_order):
            return qml.math.expand_matrix(
                canonical_matrix, wires=self.active_wires, wire_order=self.wires
            )

        return qml.math.expand_matrix(
            canonical_matrix, wires=self.active_wires, wire_order=wire_order
        )

    # pylint: disable=arguments-differ
    def sparse_matrix(self, wire_order=None, format="csr"):
        if wire_order is not None:
            raise NotImplementedError("wire_order argument is not yet implemented.")

        try:
            target_mat = self.base.sparse_matrix()
        except operation.SparseMatrixUndefinedError as e:
            if self.base.has_matrix:
                target_mat = sparse.lil_matrix(self.base.matrix())
            else:
                raise operation.SparseMatrixUndefinedError from e

        num_target_states = 2 ** len(self.target_wires)
        num_control_states = 2 ** len(self.control_wires)
        total_states = num_target_states * num_control_states

        start_ind = self._control_int * num_target_states
        end_ind = start_ind + num_target_states

        m = sparse.eye(total_states, format="lil", dtype=target_mat.dtype)

        m[start_ind:end_ind, start_ind:end_ind] = target_mat

        return m.asformat(format=format)

    def eigvals(self):
        base_eigvals = self.base.eigvals()
        num_target_wires = len(self.target_wires)
        num_control_wires = len(self.control_wires)

        total = 2 ** (num_target_wires + num_control_wires)
        ones = np.ones(total - len(base_eigvals))

        return qmlmath.concatenate([ones, base_eigvals])

    @property
    def has_diagonalizing_gates(self):
        return self.base.has_diagonalizing_gates

    def diagonalizing_gates(self):
        return self.base.diagonalizing_gates()

    @property
    def has_decomposition(self):
        if not all(self.control_values):
            return True
        if len(self.control_wires) == 1 and hasattr(self.base, "_controlled"):
            return True
        if isinstance(self.base, qml.PauliX):
            return True
        # if len(self.base.wires) == 1 and getattr(self.base, "has_matrix", False):
        #    return True
        if self.base.has_decomposition:
            return True

        return False

    def decomposition(self):
        if all(self.control_values):
            decomp = _decompose_no_control_values(self)
            if decomp is None:
                raise qml.operation.DecompositionUndefinedError
            return decomp

        # We need to add paulis to flip some control wires
        d = [qml.PauliX(w) for w, val in zip(self.control_wires, self.control_values) if not val]

        decomp = _decompose_no_control_values(self)
        if decomp is None:
            no_control_values = copy(self).queue()
            no_control_values.hyperparameters["control_values"] = [1] * len(self.control_wires)
            d.append(no_control_values)
        else:
            d += decomp

        d += [qml.PauliX(w) for w, val in zip(self.control_wires, self.control_values) if not val]
        return d

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_generator(self):
        return self.base.has_generator

    def generator(self):
        sub_gen = self.base.generator()
        proj_tensor = operation.Tensor(*(qml.Projector([1], wires=w) for w in self.control_wires))
        return 1.0 * proj_tensor @ sub_gen

    @property
    def has_adjoint(self):
        return self.base.has_adjoint

    def adjoint(self):
        return ctrl(
            self.base.adjoint(),
            self.control_wires,
            control_values=self.control_values,
            work_wires=self.work_wires,
        )

    def pow(self, z):
        base_pow = self.base.pow(z)
        return [
            ctrl(
                op,
                self.control_wires,
                control_values=self.control_values,
                work_wires=self.work_wires,
            )
            for op in base_pow
        ]

    def simplify(self) -> "Controlled":
        if isinstance(self.base, Controlled):
            base = self.base.base.simplify()
            return ctrl(
                base,
                control=self.control_wires + self.base.control_wires,
                control_values=self.control_values + self.base.control_values,
                work_wires=self.work_wires + self.base.work_wires,
            )

        return ctrl(
            op=self.base.simplify(),
            control=self.control_wires,
            control_values=self.control_values,
            work_wires=self.work_wires,
        )


# pylint: disable=protected-access
def _decompose_no_control_values(op: "operation.Operator") -> List["operation.Operator"]:
    """Provides a decomposition without considering control values. Returns None if
    no decomposition.
    """
    if len(op.control_wires) == 1 and hasattr(op.base, "_controlled"):
        result = op.base._controlled(op.control_wires[0])
        # disallow decomposing to itself
        # pylint: disable=unidiomatic-typecheck
        if type(result) != type(op):
            return [result]
        qml.QueuingManager.remove(result)
    if isinstance(op.base, qml.PauliX):
        # has some special case handling of its own for further decomposition
        return [qml.MultiControlledX(wires=op.active_wires, work_wires=op.work_wires)]
    # if (
    #    len(op.base.wires) == 1
    #    and len(op.control_wires) >= 2
    #    and getattr(op.base, "has_matrix", False)
    #    and qmlmath.get_interface(*op.data) == "numpy"  # as implemented, not differentiable
    # ):
    # Bisect algorithms use CNOTs and single qubit unitary
    #    return ctrl_decomp_bisect(op.base, op.control_wires)
    # if len(op.base.wires) == 1 and getattr(op.base, "has_matrix", False):
    #    return ctrl_decomp_zyz(op.base, op.control_wires)

    if not op.base.has_decomposition:
        return None

    base_decomp = op.base.decomposition()

    return [Controlled(newop, op.control_wires, work_wires=op.work_wires) for newop in base_decomp]


class ControlledOp(Controlled, operation.Operation):
    """Operation-specific methods and properties for the :class:`~.ops.op_math.Controlled` class.

    When an :class:`~.operation.Operation` is provided to the :class:`~.ops.op_math.Controlled`
    class, this type is constructed instead. It adds some additional :class:`~.operation.Operation`
    specific methods and properties.

    When we no longer rely on certain functionality through ``Operation``, we can get rid of this
    class.

    .. seealso:: :class:`~.Controlled`
    """

    def __new__(cls, *_, **__):
        # overrides dispatch behavior of ``Controlled``
        return object.__new__(cls)

    # pylint: disable=too-many-function-args
    def __init__(self, base, control_wires, control_values=None, work_wires=None, id=None):
        super().__init__(base, control_wires, control_values, work_wires, id)
        # check the grad_recipe validity
        if self.grad_recipe is None:
            # Make sure grad_recipe is an iterable of correct length instead of None
            self.grad_recipe = [None] * self.num_params

    @property
    def name(self):
        return self._name

    @property
    def grad_method(self):
        return self.base.grad_method

    # pylint: disable=missing-function-docstring
    @property
    def basis(self):
        return self.base.basis

    @property
    def parameter_frequencies(self):
        if self.base.num_params == 1:
            try:
                base_gen = qml.generator(self.base, format="observable")
            except operation.GeneratorUndefinedError as e:
                raise operation.ParameterFrequenciesUndefinedError(
                    f"Operation {self.base.name} does not have parameter frequencies defined."
                ) from e

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message=r".+ eigenvalues will be computed numerically\."
                )
                base_gen_eigvals = qml.eigvals(base_gen)

            # The projectors in the full generator add a eigenvalue of `0` to
            # the eigenvalues of the base generator.
            gen_eigvals = np.append(base_gen_eigvals, 0)

            processed_gen_eigvals = tuple(np.round(gen_eigvals, 8))
            return [qml.gradients.eigvals_to_frequencies(processed_gen_eigvals)]
        raise operation.ParameterFrequenciesUndefinedError(
            f"Operation {self.name} does not have parameter frequencies defined, "
            "and parameter frequencies can not be computed via generator for more than one "
            "parameter."
        )
