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

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Sequence
from copy import copy
from functools import wraps
from inspect import signature
from typing import Any, Optional, overload

import numpy as np
from scipy import sparse

import pennylane as qml
from pennylane import math
from pennylane.compiler import compiler
from pennylane.decomposition.resources import resolve_work_wire_type
from pennylane.operation import (
    GeneratorUndefinedError,
    Operation,
    Operator,
    ParameterFrequenciesUndefinedError,
    SparseMatrixUndefinedError,
    classproperty,
)
from pennylane.wires import Wires, WiresLike

from .decompositions.controlled_decompositions import ctrl_decomp_bisect, ctrl_decomp_zyz
from .symbolicop import SymbolicOp


@overload
def ctrl(
    op: Operator,
    control: Any,
    control_values: Optional[Sequence[bool | int]] = None,
    work_wires: Optional[Any] = None,
    work_wire_type: Optional[str] = "dirty",
) -> Operator: ...
@overload
def ctrl(
    op: Callable,
    control: Any,
    control_values: Optional[Sequence[bool | int]] = None,
    work_wires: Optional[Any] = None,
    work_wire_type: Optional[str] = "dirty",
) -> Callable: ...
def ctrl(op, control: Any, control_values=None, work_wires=None, work_wire_type="dirty"):
    r"""Create a method that applies a controlled version of the provided op.
    :func:`~.qjit` compatible.

    .. note::

        When used with :func:`~.qjit`, this function only supports the Catalyst compiler.
        See :func:`catalyst.ctrl` for more details.

        Please see the Catalyst :doc:`quickstart guide <catalyst:dev/quick_start>`,
        as well as the :doc:`sharp bits and debugging tips <catalyst:dev/sharp_bits>`
        page for an overview of the differences between Catalyst and PennyLane.

    Args:
        op (function or :class:`~.operation.Operator`): A single operator or a function that applies pennylane operators.
        control (Wires): The control wire(s).
        control_values (bool or int or list[bool or int]): The value(s) the control wire(s)
            should take. Integers other than 0 or 1 will be treated as ``int(bool(x))``.
        work_wires (Any): Any auxiliary wires that can be used in the decomposition
        work_wire_type: The type of work wire(s), can be ``"clean"`` or ``"dirty"``. ``"clean"``
            indicates that the work wires are in the :math:`|0\rangle` state, whereas ``"dirty"``
            work wires can be in any arbitrary state. Defaults to ``"dirty"``.

    Returns:
        function or :class:`~.operation.Operator`: If an Operator is provided, returns a Controlled version of the Operator.
        If a function is provided, returns a function with the same call signature that creates a controlled version of the
        provided function.

    .. seealso:: :class:`~.Controlled`.

    **Example**

    .. code-block:: python3

        @qml.qnode(qml.device('default.qubit', wires=range(4)))
        def circuit(x):
            qml.X(2)
            qml.ctrl(qml.RX, (1,2,3), control_values=(0,1,0))(x, wires=0)
            return qml.expval(qml.Z(0))

    >>> print(qml.draw(circuit)("x"))
    0: ────╭RX(x)─┤  <Z>
    1: ────├○─────┤
    2: ──X─├●─────┤
    3: ────╰○─────┤
    >>> x = np.array(1.2)
    >>> circuit(x)
    tensor(0.36235775, requires_grad=True)
    >>> qml.grad(circuit)(x)
    tensor(-0.93203909, requires_grad=True)

    :func:`~.ctrl` works on both callables like ``qml.RX`` or a quantum function
    and individual :class:`~.operation.Operator`'s.

    >>> qml.ctrl(qml.Hadamard(0), (1,2))
    Controlled(H(0), control_wires=[1, 2])

    Controlled operations work with all other forms of operator math and simplification:

    >>> op = qml.ctrl(qml.RX(1.2, wires=0) ** 2 @ qml.RY(0.1, wires=0), control=1)
    >>> qml.simplify(qml.adjoint(op))
    Controlled(RY(12.466370614359173, wires=[0]) @ RX(10.166370614359172, wires=[0]), control_wires=[1])

    **Example with compiler**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit
        @qml.qnode(dev)
        def workflow(theta, w, cw):
            qml.Hadamard(wires=[0])
            qml.Hadamard(wires=[1])

            def func(arg):
                qml.RX(theta, wires=arg)

            def cond_fn():
                qml.RY(theta, wires=w)

            qml.ctrl(func, control=[cw])(w)
            qml.ctrl(qml.cond(theta > 0.0, cond_fn), control=[cw])()
            qml.ctrl(qml.RZ, control=[cw])(theta, wires=w)
            qml.ctrl(qml.RY(theta, wires=w), control=[cw])
            return qml.probs()

    >>> workflow(jnp.pi/4, 1, 0)
    array([0.25, 0.25, 0.03661165, 0.46338835])
    """

    if active_jit := compiler.active_compiler():
        available_eps = compiler.AvailableCompilers.names_entrypoints
        ops_loader = available_eps[active_jit]["ops"].load()
        return ops_loader.ctrl(op, control, control_values=control_values, work_wires=work_wires)
    if math.is_abstract(op):
        return Controlled(
            op,
            control,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
    return create_controlled_op(
        op,
        control,
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


def create_controlled_op(op, control, control_values=None, work_wires=None, work_wire_type="dirty"):
    """Default ``qml.ctrl`` implementation, allowing other implementations to call it when needed."""

    control = qml.wires.Wires(control)
    if isinstance(control_values, (int, bool)):
        control_values = [control_values]
    elif control_values is None:
        control_values = [True] * len(control)
    elif isinstance(control_values, tuple):
        control_values = list(control_values)

    ctrl_op = _try_wrap_in_custom_ctrl_op(
        op,
        control=control,
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    if ctrl_op is not None:
        return ctrl_op

    pauli_x_based_ctrl_ops = _get_pauli_x_based_ops()

    # Special handling for PauliX-based controlled operations
    if isinstance(op, pauli_x_based_ctrl_ops):
        qml.QueuingManager.remove(op)
        return _handle_pauli_x_based_controlled_ops(
            op,
            control=control,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )

    # Flatten nested controlled operations to a multi-controlled operation for better
    # decomposition algorithms. This includes special cases like CRX, CRot, etc.
    if isinstance(op, Controlled):
        work_wires = Wires(() if work_wires is None else work_wires)
        work_wire_type = resolve_work_wire_type(
            op.work_wires, op.work_wire_type, work_wires, work_wire_type
        )
        qml.QueuingManager.remove(op)
        return ctrl(
            op.base,
            control=control + op.control_wires,
            control_values=control_values + op.control_values,
            work_wires=work_wires + op.work_wires,
            work_wire_type=work_wire_type,
        )

    if isinstance(op, Operator):
        return Controlled(
            op,
            control_wires=control,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )

    if not callable(op):
        raise ValueError(
            f"The object {op} of type {type(op)} is not an Operator or callable. "
            "This error might occur if you apply ctrl to a list "
            "of operations instead of a function or Operator."
        )
    if qml.capture.enabled():
        return _capture_ctrl_transform(op, control, control_values, work_wires)
    return _ctrl_transform(op, control, control_values, work_wires)


def _ctrl_transform(op, control, control_values, work_wires):
    @wraps(op)
    def wrapper(*args, **kwargs):
        qscript = qml.tape.make_qscript(op)(*args, **kwargs)

        leaves, _ = qml.pytrees.flatten((args, kwargs), lambda obj: isinstance(obj, Operator))
        _ = [qml.QueuingManager.remove(l) for l in leaves if isinstance(l, Operator)]

        # flip control_values == 0 wires here, so we don't have to do it for each individual op.
        flip_control_on_zero = (len(qscript) > 1) and (control_values is not None)
        op_control_values = None if flip_control_on_zero else control_values
        if flip_control_on_zero:
            _ = [qml.X(w) for w, val in zip(control, control_values) if not val]

        _ = [
            ctrl(op, control=control, control_values=op_control_values, work_wires=work_wires)
            for op in qscript.operations
        ]

        if flip_control_on_zero:
            _ = [qml.X(w) for w, val in zip(control, control_values) if not val]

        if qml.QueuingManager.recording():
            _ = [qml.apply(m) for m in qscript.measurements]

        return qscript.measurements

    return wrapper


@functools.lru_cache  # only create the first time requested
def _get_ctrl_qfunc_prim():
    """See capture/explanations.md : Higher Order primitives for more information on this code."""
    # if capture is enabled, jax should be installed

    # pylint: disable=import-outside-toplevel
    from pennylane.capture.custom_primitives import QmlPrimitive

    ctrl_prim = QmlPrimitive("ctrl_transform")
    ctrl_prim.multiple_results = True
    ctrl_prim.prim_type = "higher_order"

    @ctrl_prim.def_impl
    def _(*args, n_control, jaxpr, control_values, work_wires, n_consts):
        from pennylane.tape.plxpr_conversion import CollectOpsandMeas

        consts = args[:n_consts]
        control_wires = args[-n_control:]
        args = args[n_consts:-n_control]

        collector = CollectOpsandMeas()
        with qml.QueuingManager.stop_recording():
            collector.eval(jaxpr, consts, *args)

        for op in collector.state["ops"]:
            ctrl(op, control_wires, control_values, work_wires)
        return []

    @ctrl_prim.def_abstract_eval
    def _(*_, **__):
        return []

    return ctrl_prim


def _capture_ctrl_transform(qfunc: Callable, control, control_values, work_wires) -> Callable:
    """Capture compatible way of performing an ctrl transform."""
    # note that this logic is tested in `tests/capture/test_nested_plxpr.py`
    import jax  # pylint: disable=import-outside-toplevel

    ctrl_prim = _get_ctrl_qfunc_prim()

    @wraps(qfunc)
    def new_qfunc(*args, **kwargs):
        abstracted_axes, abstract_shapes = qml.capture.determine_abstracted_axes(args)
        jaxpr = jax.make_jaxpr(functools.partial(qfunc, **kwargs), abstracted_axes=abstracted_axes)(
            *args
        )
        flat_args = jax.tree_util.tree_leaves(args)
        control_wires = qml.wires.Wires(control)  # make sure is iterable
        ctrl_prim.bind(
            *jaxpr.consts,
            *abstract_shapes,
            *flat_args,
            *control_wires,
            jaxpr=jaxpr.jaxpr,
            n_control=len(control_wires),
            control_values=control_values,
            work_wires=work_wires,
            n_consts=len(jaxpr.consts),
        )

    return new_qfunc


@functools.lru_cache(maxsize=1)
def _get_pauli_x_based_ops():
    """Gets a list of pauli-x based operations

    This is placed inside a function to avoid circular imports.

    """
    return qml.X, qml.CNOT, qml.Toffoli, qml.MultiControlledX


def _try_wrap_in_custom_ctrl_op(
    op, control, control_values=None, work_wires=None, work_wire_type="dirty"
):
    """Wraps a controlled operation in custom ControlledOp, returns None if not applicable."""

    ops_with_custom_ctrl_ops = base_to_custom_ctrl_op()
    custom_key = (type(op), len(control))

    if custom_key in ops_with_custom_ctrl_ops and all(control_values):
        qml.QueuingManager.remove(op)
        return ops_with_custom_ctrl_ops[custom_key](*op.data, control + op.wires)

    if isinstance(op, qml.QubitUnitary):
        qml.QueuingManager.remove(op)
        return qml.ControlledQubitUnitary(
            op.matrix() if op.has_matrix else op.sparse_matrix(),
            wires=control + op.wires,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )

    return None


def _handle_pauli_x_based_controlled_ops(op, control, control_values, work_wires, work_wire_type):
    """Handles PauliX-based controlled operations."""

    op_map = {
        (qml.PauliX, 1): qml.CNOT,
        (qml.PauliX, 2): qml.Toffoli,
        (qml.CNOT, 1): qml.Toffoli,
    }

    custom_key = (type(op), len(control))
    if custom_key in op_map and all(control_values):
        qml.QueuingManager.remove(op)
        return op_map[custom_key](wires=control + op.wires)

    if isinstance(op, qml.PauliX):
        return qml.MultiControlledX(
            wires=control + op.wires,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )

    work_wires = Wires([] if work_wires is None else work_wires)
    work_wire_type = resolve_work_wire_type(
        op.work_wires, op.work_wire_type, work_wires, work_wire_type
    )
    return qml.MultiControlledX(
        wires=control + op.wires,
        control_values=control_values + op.control_values,
        work_wires=work_wires + op.work_wires,
        work_wire_type=work_wire_type,
    )


# pylint: disable=too-many-arguments, too-many-public-methods
class Controlled(SymbolicOp):
    r"""Symbolic operator denoting a controlled operator.

    Args:
        base (~.operation.Operator): the operator that is controlled
        control_wires (Any): The wires to control on.

    Keyword Args:
        control_values (Iterable[Bool]): The values to control on. Must be the same
            length as ``control_wires``. Defaults to ``True`` for all control wires.
            Provided values are converted to `Bool` internally.
        work_wires (Any): Any auxiliary wires that can be used in the decomposition
        work_wire_type: The type of work wire(s), can be ``"clean"`` or ``"dirty"``. ``"clean"``
            indicates that the work wires are in the :math:`|0\rangle` state, whereas ``"dirty"``
            work wires can be in any arbitrary state. Defaults to ``"dirty"``.

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
    Wires([0, 1])
    >>> op.control_wires
    Wires([0])
    >>> op.target_wires
    Wires([1])

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

    resource_keys = {
        "base_class",
        "base_params",
        "num_control_wires",
        "num_zero_control_values",
        "num_work_wires",
        "work_wire_type",
    }

    def _flatten(self):
        return (self.base,), (
            self.control_wires,
            tuple(self.control_values),
            self.work_wires,
            self.work_wire_type,
        )

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(
            data[0],
            control_wires=metadata[0],
            control_values=metadata[1],
            work_wires=metadata[2],
            work_wire_type=metadata[3],
        )

    # pylint: disable=no-self-argument
    @classproperty
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
        if isinstance(base, Operation):
            return object.__new__(ControlledOp)
        return object.__new__(Controlled)

    # pylint: disable=arguments-differ, too-many-positional-arguments
    @classmethod
    def _primitive_bind_call(
        cls,
        base,
        control_wires,
        control_values=None,
        work_wires=None,
        work_wire_type="dirty",
        id=None,
    ):
        control_wires = Wires(control_wires)
        return cls._primitive.bind(
            base,
            *control_wires,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        base,
        control_wires: WiresLike,
        control_values=None,
        work_wires: WiresLike = None,
        work_wire_type: Optional[str] = "dirty",
        id=None,
    ):
        control_wires = Wires(control_wires)
        work_wires = Wires(() if work_wires is None else work_wires)

        if control_values is None:
            control_values = [True] * len(control_wires)
        else:
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

        if work_wire_type not in {"clean", "dirty"}:
            raise ValueError(
                f"work_wire_type must be either 'clean' or 'dirty'. Got '{work_wire_type}'."
            )

        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["control_values"] = control_values
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["work_wire_type"] = work_wire_type
        self._name = f"C({base.name})"

        super().__init__(base, id)

    @property
    def hash(self):
        # these gates do not consider global phases in their hash
        if self.base.name in ("RX", "RY", "RZ", "Rot"):
            base_params = str(
                [
                    (id(d) if math.is_abstract(d) else math.round(math.real(d) % (4 * np.pi), 10))
                    for d in self.base.data
                ]
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
    def work_wire_type(self):
        """The type of work wires provided, can be ``"clean"`` or ``"dirty"``."""
        return self.hyperparameters["work_wire_type"]

    @property
    def wires(self):
        return self.control_wires + self.target_wires

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

    # Properties for resource estimation ###############

    @property
    def resource_params(self):
        return {
            "base_class": type(self.base),
            "base_params": self.base.resource_params,
            "num_control_wires": len(self.control_wires),
            "num_zero_control_values": len([val for val in self.control_values if not val]),
            "num_work_wires": len(self.work_wires),
            "work_wire_type": self.work_wire_type,
        }

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

    def _compute_matrix_from_base(self):
        base_matrix = self.base.matrix()
        interface = math.get_interface(base_matrix)

        num_target_states = 2 ** len(self.target_wires)
        num_control_states = 2 ** len(self.control_wires)
        total_matrix_size = num_control_states * num_target_states

        padding_left = self._control_int * num_target_states
        padding_right = total_matrix_size - padding_left - num_target_states

        left_pad = math.convert_like(
            math.cast_like(math.eye(padding_left, like=interface), 1j), base_matrix
        )
        right_pad = math.convert_like(
            math.cast_like(math.eye(padding_right, like=interface), 1j), base_matrix
        )

        shape = math.shape(base_matrix)
        if len(shape) == 3:  # stack if batching
            return math.stack([math.block_diag([left_pad, _U, right_pad]) for _U in base_matrix])

        return math.block_diag([left_pad, base_matrix, right_pad])

    def matrix(self, wire_order=None):
        if self.compute_matrix is not Operator.compute_matrix:
            canonical_matrix = self.compute_matrix(*self.data)
        else:
            canonical_matrix = self._compute_matrix_from_base()

        wire_order = wire_order or self.wires
        return math.expand_matrix(canonical_matrix, wires=self.wires, wire_order=wire_order)

    @property
    def has_sparse_matrix(self):
        return self.base.has_sparse_matrix or self.base.has_matrix

    def sparse_matrix(self, wire_order=None, format="csr"):
        try:
            target_mat = self.base.sparse_matrix()
        except SparseMatrixUndefinedError as e:
            if self.base.has_matrix:
                target_mat = sparse.lil_matrix(self.base.matrix())
            else:
                raise SparseMatrixUndefinedError from e

        num_target_states = 2 ** len(self.target_wires)
        num_control_states = 2 ** len(self.control_wires)
        total_states = num_target_states * num_control_states

        start_ind = self._control_int * num_target_states
        end_ind = start_ind + num_target_states

        m = sparse.eye(total_states, format="lil", dtype=target_mat.dtype)

        m[start_ind:end_ind, start_ind:end_ind] = target_mat

        wire_order = wire_order or self.wires
        m = math.expand_matrix(m, wires=self.wires, wire_order=wire_order)

        return m.asformat(format=format)

    def eigvals(self):
        base_eigvals = self.base.eigvals()
        num_target_wires = len(self.target_wires)
        num_control_wires = len(self.control_wires)

        total = 2 ** (num_target_wires + num_control_wires)
        ones = np.ones(total - len(base_eigvals))

        return math.concatenate([ones, base_eigvals])

    @property
    def has_diagonalizing_gates(self):
        return self.base.has_diagonalizing_gates

    def diagonalizing_gates(self):
        return self.base.diagonalizing_gates()

    @property
    def has_decomposition(self):
        if self.compute_decomposition is not Operator.compute_decomposition:
            return True
        if not all(self.control_values):
            return True
        # not already the simplified version
        if (
            len(self.control_wires) == 1
            and hasattr(self.base, "_controlled")
            and type(self) in {Controlled, ControlledOp}
        ):
            return True
        is_su2 = _is_single_qubit_special_unitary(self.base)
        if not math.is_abstract(is_su2) and is_su2:
            return True
        if self.base.has_decomposition:
            return True

        return False

    def decomposition(self):

        if self.compute_decomposition is not Operator.compute_decomposition:
            return self.compute_decomposition(*self.data, self.wires)

        if all(self.control_values):
            decomp = _decompose_no_control_values(self)
            if decomp is None:
                raise qml.operation.DecompositionUndefinedError
            return decomp

        # We need to add paulis to flip some control wires
        d = [qml.X(w) for w, val in zip(self.control_wires, self.control_values) if not val]

        decomp = _decompose_no_control_values(self)
        if decomp is None:
            no_control_values = copy(self).queue()
            no_control_values.hyperparameters["control_values"] = [1] * len(self.control_wires)
            d.append(no_control_values)
        else:
            d += decomp

        d += [qml.X(w) for w, val in zip(self.control_wires, self.control_values) if not val]
        return d

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_generator(self):
        return self.base.has_generator

    def generator(self):
        sub_gen = self.base.generator()
        projectors = (
            qml.Projector([val], wires=w) for val, w in zip(self.control_values, self.control_wires)
        )
        # needs to return a new_opmath instance regardless of whether new_opmath is enabled, because
        # it otherwise can't handle ControlledGlobalPhase, see PR #5194
        return qml.prod(*projectors, sub_gen)

    @property
    def has_adjoint(self):
        return self.base.has_adjoint

    def adjoint(self):
        return ctrl(
            self.base.adjoint(),
            self.control_wires,
            control_values=self.control_values,
            work_wires=self.work_wires,
            work_wire_type=self.work_wire_type,
        )

    def pow(self, z):
        base_pow = self.base.pow(z)
        return [
            ctrl(
                op,
                self.control_wires,
                control_values=self.control_values,
                work_wires=self.work_wires,
                work_wire_type=self.work_wire_type,
            )
            for op in base_pow
        ]

    def simplify(self) -> "Operator":
        if isinstance(self.base, Controlled):
            base = self.base.base.simplify()
            return ctrl(
                base,
                control=self.control_wires + self.base.control_wires,
                control_values=self.control_values + self.base.control_values,
                work_wires=self.work_wires + self.base.work_wires,
                work_wire_type=resolve_work_wire_type(
                    self.base.work_wires,
                    self.base.work_wire_type,
                    self.work_wires,
                    self.work_wire_type,
                ),
            )

        simplified_base = self.base.simplify()
        if isinstance(simplified_base, qml.Identity):
            return simplified_base

        return ctrl(
            op=simplified_base,
            control=self.control_wires,
            control_values=self.control_values,
            work_wires=self.work_wires,
            work_wire_type=self.work_wire_type,
        )


def _is_single_qubit_special_unitary(op):
    if not op.has_matrix or len(op.wires) != 1:
        return False
    mat = op.matrix()
    det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    return math.allclose(det, 1)


def _decompose_pauli_x_based_no_control_values(op: Controlled):
    """Decomposes a PauliX-based operation"""

    if isinstance(op.base, qml.PauliX) and len(op.control_wires) == 1:
        return [qml.CNOT(wires=op.wires)]

    if isinstance(op.base, qml.PauliX) and len(op.control_wires) == 2:
        return qml.Toffoli.compute_decomposition(wires=op.wires)

    if isinstance(op.base, qml.CNOT) and len(op.control_wires) == 1:
        return qml.Toffoli.compute_decomposition(wires=op.wires)

    return qml.MultiControlledX.compute_decomposition(
        wires=op.wires,
        work_wires=op.work_wires,
        work_wire_type=op.work_wire_type,
    )


def _decompose_custom_ops(op: Controlled) -> Optional[list[Operator]]:
    """Custom handling for decomposing a controlled operation"""

    pauli_x_based_ctrl_ops = _get_pauli_x_based_ops()
    ops_with_custom_ctrl_ops = base_to_custom_ctrl_op()

    custom_key = (type(op.base), len(op.control_wires))
    if custom_key in ops_with_custom_ctrl_ops:
        custom_op_cls = ops_with_custom_ctrl_ops[custom_key]
        return custom_op_cls.compute_decomposition(*op.data, op.wires)
    if isinstance(op.base, pauli_x_based_ctrl_ops):
        # has some special case handling of its own for further decomposition
        return _decompose_pauli_x_based_no_control_values(op)

    if isinstance(op.base, qml.GlobalPhase):
        # A singly-controlled global phase is the same as a phase shift on the control wire
        # (Lemma 5.2 from https://arxiv.org/pdf/quant-ph/9503016)
        # Mathematically, this is the equation (with Id_2 being the 2-dim. identity matrix)
        # |0><0|⊗ Id_2 + |1><1|⊗ e^{i\phi} = [|0><0| + |1><1| e^{i\phi}] ⊗ Id_2
        phase_shift = qml.PhaseShift(phi=-op.data[0], wires=op.control_wires[-1])
        if len(op.control_wires) == 1:
            return [phase_shift]
        # For N>1 control wires, we simply add N-1 control wires to the phase shift
        # Mathematically, this is the equation (proven by inserting an identity)
        # (Id_{2^N} - |1><1|^N)⊗ Id_2 + |1><1|^N ⊗ e^{i\phi}
        # = (Id_{2^{N-1}} - |1><1|^{N-1}) ⊗ Id_4 + |1><1|^{N-1} ⊗ [|0><0|+|1><1|e^{i\phi}]⊗ Id_2
        return [ctrl(phase_shift, control=op.control_wires[:-1])]

    # TODO: will be removed in the second part of the controlled rework [sc-37951]
    if len(op.control_wires) == 1 and hasattr(op.base, "_controlled"):
        result = op.base._controlled(op.control_wires[0])  # pylint: disable=protected-access
        # disallow decomposing to itself
        # pylint: disable=unidiomatic-typecheck
        if type(result) != type(op):
            return [result]
        qml.QueuingManager.remove(result)

    return None


def _decompose_no_control_values(op: Controlled) -> Optional[list[Operator]]:
    """Decompose without considering control values. Returns None if no decomposition."""

    decomp = _decompose_custom_ops(op)
    if decomp is not None:
        return decomp

    is_su2 = _is_single_qubit_special_unitary(op.base)
    if not math.is_abstract(is_su2) and is_su2:
        if len(op.control_wires) >= 2 and math.get_interface(*op.data) == "numpy":
            return ctrl_decomp_bisect(op.base, op.control_wires)
        return ctrl_decomp_zyz(
            op.base,
            control_wires=op.control_wires,
            work_wires=op.work_wires,
            work_wire_type=op.work_wire_type,
        )

    if not op.base.has_decomposition:
        return None

    base_decomp = op.base.decomposition()
    return [
        ctrl(newop, op.control_wires, work_wires=op.work_wires, work_wire_type=op.work_wire_type)
        for newop in base_decomp
    ]


class ControlledOp(Controlled, Operation):
    """Operation-specific methods and properties for the :class:`~.ops.op_math.Controlled` class.

    When an :class:`~.operation.Operation` is provided to the :class:`~.ops.op_math.Controlled`
    class, this type is constructed instead. It adds some additional :class:`~.operation.Operation`
    specific methods and properties.

    When we no longer rely on certain functionality through ``Operation``, we can get rid of this
    class.

    .. seealso:: :class:`~.Controlled`
    """

    def __new__(cls, *_, **__):
        # overrides dispatch behaviour of ``Controlled``
        return object.__new__(cls)

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        base,
        control_wires,
        control_values=None,
        work_wires=None,
        work_wire_type="dirty",
        id=None,
    ):
        super().__init__(base, control_wires, control_values, work_wires, work_wire_type, id)
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

    @property
    def parameter_frequencies(self):
        if self.base.num_params == 1:
            try:
                base_gen = qml.generator(self.base, format="observable")
            except GeneratorUndefinedError as e:
                raise ParameterFrequenciesUndefinedError(
                    f"Operation {self.base.name} does not have parameter frequencies defined."
                ) from e

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message=r".+ eigenvalues will be computed numerically\."
                )
                base_gen_eigvals = qml.eigvals(base_gen, k=2**self.base.num_wires)

            # The projectors in the full generator add a eigenvalue of `0` to
            # the eigenvalues of the base generator.
            gen_eigvals = np.append(base_gen_eigvals, 0)

            processed_gen_eigvals = tuple(np.round(gen_eigvals, 8))
            return [qml.gradients.eigvals_to_frequencies(processed_gen_eigvals)]
        raise ParameterFrequenciesUndefinedError(
            f"Operation {self.name} does not have parameter frequencies defined, "
            "and parameter frequencies can not be computed via generator for more than one "
            "parameter."
        )


# Program capture with controlled ops needs to unpack and re-pack the control wires to support dynamic wires
# See capture module for more information on primitives
# If None, jax isn't installed so the class never got a primitive.
if Controlled._primitive is not None:  # pylint: disable=protected-access

    @Controlled._primitive.def_impl  # pylint: disable=protected-access
    def _(
        base, *control_wires, control_values=None, work_wires=None, work_wire_type="dirty", id=None
    ):
        return type.__call__(
            Controlled,
            base,
            control_wires,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
            id=id,
        )


# easier to just keep the same primitive for both versions
# dispatch between the two types happens inside instance creation anyway
ControlledOp._primitive = Controlled._primitive  # pylint: disable=protected-access


@functools.lru_cache(maxsize=1)
def base_to_custom_ctrl_op():
    """A dictionary mapping base op types to their custom controlled versions.
    This dictionary is used under the assumption that all custom controlled operations do not
    have resource params (which is why `ControlledQubitUnitary` is not included here).
    """

    ops_with_custom_ctrl_ops = {
        (qml.PauliZ, 1): qml.CZ,
        (qml.PauliZ, 2): qml.CCZ,
        (qml.PauliY, 1): qml.CY,
        (qml.CZ, 1): qml.CCZ,
        (qml.SWAP, 1): qml.CSWAP,
        (qml.Hadamard, 1): qml.CH,
        (qml.RX, 1): qml.CRX,
        (qml.RY, 1): qml.CRY,
        (qml.RZ, 1): qml.CRZ,
        (qml.Rot, 1): qml.CRot,
        (qml.PhaseShift, 1): qml.ControlledPhaseShift,
    }
    return ops_with_custom_ctrl_ops
