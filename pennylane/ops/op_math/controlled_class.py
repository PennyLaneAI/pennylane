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

from inspect import signature

import numpy as np
from scipy import sparse

import pennylane as qml
from pennylane import math as qmlmath
from pennylane import operation
from pennylane.wires import Wires

from .symbolicop import SymbolicOp


# pylint: disable=too-many-arguments, too-many-public-methods
class Controlled(SymbolicOp):
    """Symbolic operator denoting a controlled operator.

    Args:
        base (~.operation.Operator): the operator that is controlled
        control_wires (Any): The wires to control on.

    Keyword Args:
        control_values (Iterable[Bool]): The values to control on. Must be the same
            length as ``control_wires``. Defaults to ``True`` for all control wires.
        work_wires (Any): Any auxiliary wires that can be used in the decomposition

    .. note::
        This class, ``Controlled``, denotes a controlled version of any inidividual operation.
        :class:`~.ControlledOp` adds :class:`~.Operation` specific methods and properties to the
        more general ``Controlled`` class.

        The :class:`~.ControlledOperation` currently constructed by the :func:`~.ctrl` transform wraps
        an entire tape and does not provide as many representations and attributes as ``Controlled``,
        but :class:`~.ControlledOperation` does decompose.

    .. seealso:: :class:`~.ControlledOp` and ::class:`~.ControlledOperation`

    **Example**

    >>> base = qml.RX(1.234, 1)
    >>> Controlled(base, (0, 2, 3), control_values=[True, False, True])
    CRX(1.234, wires=[0, 2, 3, 1])
    >>> op = Controlled(base, 0, control_values=[0])
    >>> op
    CRX(1.234, wires=[0, 1])

    The operation has both standard :class:`~.operation.Operator` properties
    and ``Controlled`` specific properties:

    >>> op.base
    RX(1.234, wires=[1])
    >>> op.data
    [1.234]
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
    >>> qml.generator(op, format='observable')
    (-0.5) [Projector0 X1]
    >>> op.sparse_matrix()
    <4x4 sparse matrix of type '<class 'numpy.complex128'>'
                with 6 stored elements in Compressed Sparse Row format>

    If the provided base matrix is an :class:`~.operation.Operation`, then the created
    object will be of type :class:`~.ops.op_math.ControlledOp`. This class adds some additional
    methods and properties to the basic :class:`~.ops.op_math.Controlled` class.

    >>> type(op)
    pennylane.ops.op_math.controlled_class.ControlledOp
    >>> op.parameter_frequencies
    [(0.5, 1.0)]

    """

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
        """If base is an ``Operation``, then the a ``ControlledOp`` should be used instead."""
        if isinstance(base, operation.Operation):
            return object.__new__(ControlledOp)
        return object.__new__(Controlled)

    # pylint: disable=too-many-function-args
    def __init__(
        self, base, control_wires, control_values=None, work_wires=None, do_queue=True, id=None
    ):
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
                control_values = [(x == "1") for x in control_values]

            if len(control_values) != len(control_wires):
                raise ValueError("control_values should be the same length as control_wires")
            if not set(control_values).issubset({False, True}):
                raise ValueError("control_values can only take on True or False")

        if len(Wires.shared_wires([base.wires, control_wires])) != 0:
            raise ValueError("The control wires must be different from the base operation wires.")

        if len(Wires.shared_wires([work_wires, base.wires + control_wires])) != 0:
            raise ValueError(
                "Work wires must be different the control_wires and base operation wires."
            )

        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["control_values"] = control_values
        self.hyperparameters["work_wires"] = work_wires

        self._name = f"C{base.name}"

        super().__init__(base, do_queue, id)

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
    def wires(self):
        return self.control_wires + self.base.wires + self.work_wires

    # pylint: disable=protected-access
    @property
    def _wires(self):
        return self.wires

    # pylint: disable=protected-access
    @_wires.setter
    def _wires(self, new_wires):
        new_wires = new_wires if isinstance(new_wires, Wires) else Wires(new_wires)

        num_control = len(self.control_wires)
        num_base = len(self.base.wires)
        num_control_and_base = num_control + num_base

        assert num_control_and_base <= len(new_wires), (
            f"{self.name} needs at least {num_control_and_base} wires."
            f" {len(new_wires)} provided."
        )

        self.hyperparameters["control_wires"] = new_wires[:num_control]

        self.base._wires = new_wires[num_control:num_control_and_base]

        if len(new_wires) > num_control_and_base:
            self.hyperparameters["work_wires"] = new_wires[num_control_and_base:]
        else:
            self.hyperparameters["work_wires"] = Wires([])

    # Methods ##########################################

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

        left_pad = qmlmath.cast_like(qmlmath.eye(padding_left, like=interface), 1j)
        right_pad = qmlmath.cast_like(qmlmath.eye(padding_right, like=interface), 1j)

        canonical_matrix = qmlmath.block_diag([left_pad, base_matrix, right_pad])

        active_wires = self.control_wires + self.target_wires

        if wire_order is None or self.wires == Wires(wire_order):
            return operation.expand_matrix(
                canonical_matrix, wires=active_wires, wire_order=self.wires
            )

        return operation.expand_matrix(canonical_matrix, wires=active_wires, wire_order=wire_order)

    # pylint: disable=arguments-differ
    def sparse_matrix(self, wire_order=None, format="csr"):
        if wire_order is not None:
            raise NotImplementedError("wire_order argument is not yet implemented.")

        try:
            target_mat = self.base.sparse_matrix()
        except operation.SparseMatrixUndefinedError:
            try:
                target_mat = sparse.lil_matrix(self.base.matrix())
            except operation.MatrixUndefinedError as e:
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

    def diagonalizing_gates(self):
        return self.base.diagonalizing_gates()

    def decomposition(self):
        if not all(self.control_values):
            d = [
                qml.PauliX(w) for w, val in zip(self.control_wires, self.control_values) if not val
            ]
            d += [Controlled(self.base, self.control_wires, work_wires=self.work_wires)]
            d += [
                qml.PauliX(w) for w, val in zip(self.control_wires, self.control_values) if not val
            ]

            return d
        # More to come.  This will be an extensive PR in and of itself.
        return super().decomposition()

    def generator(self):
        sub_gen = self.base.generator()
        proj_tensor = operation.Tensor(*(qml.Projector([1], wires=w) for w in self.control_wires))
        return 1.0 * proj_tensor @ sub_gen

    def adjoint(self):
        return Controlled(
            self.base.adjoint(), self.control_wires, self.control_values, self.work_wires
        )

    def pow(self, z):
        base_pow = self.base.pow(z)
        return [
            Controlled(op, self.control_wires, self.control_values, self.work_wires)
            for op in base_pow
        ]


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

    @property
    def _inverse(self):
        return False

    @_inverse.setter
    def _inverse(self, boolean):
        self.base._inverse = boolean  # pylint: disable=protected-access
        # refresh name as base_name got updated.
        self._name = f"C{self.base.name}"

    def inv(self):
        self.base.inv()
        # refresh name as base_name got updated.
        self._name = f"C{self.base.name}"
        return self

    @property
    def base_name(self):
        return f"C{self.base.base_name}"

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
            "and parameter frequencies can not be computed via generator for more than one"
            "parameter."
        )
