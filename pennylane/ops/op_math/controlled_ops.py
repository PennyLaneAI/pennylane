# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
This submodule contains controlled operators based on the ControlledOp class.
"""

# pylint: disable=arguments-differ,arguments-renamed

from collections.abc import Iterable
from functools import lru_cache, partial
from typing import Literal

import numpy as np
from scipy.linalg import block_diag

import pennylane as qml
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.decomposition.symbolic_decomposition import (
    adjoint_rotation,
    flip_zero_control,
    pow_involutory,
    pow_rotation,
    self_adjoint,
)
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

from .controlled import ControlledOp
from .controlled_decompositions import decompose_mcx
from .decompositions.controlled_decompositions import (
    controlled_two_qubit_unitary_rule,
    ctrl_decomp_bisect_rule,
    decompose_mcx_many_borrowed_workers,
    decompose_mcx_many_workers_explicit,
    decompose_mcx_many_zeroed_workers,
    decompose_mcx_one_borrowed_worker,
    decompose_mcx_one_worker_explicit,
    decompose_mcx_one_zeroed_worker,
    decompose_mcx_two_borrowed_workers,
    decompose_mcx_two_workers_explicit,
    decompose_mcx_two_zeroed_workers,
    decompose_mcx_with_no_worker,
    multi_control_decomp_zyz_rule,
    single_ctrl_decomp_zyz_rule,
)

INV_SQRT2 = 1 / qml.math.sqrt(2)

stack_last = partial(qml.math.stack, axis=-1)


class ControlledQubitUnitary(ControlledOp):
    r"""ControlledQubitUnitary(U, wires)
    Apply an arbitrary fixed unitary matrix ``U`` to ``wires``. If ``n = len(wires) `` and ``U`` has ``k`` wires, then the first ``n - k`` from ``wires`` serve as control, and ``U`` lives on the last ``k`` wires.

    In addition to default ``Operation`` instance attributes, the following are
    available for ``ControlledQubitUnitary``:

    * ``wires``: wires of the final controlled unitary, consisting of control wires following by target wires
    * ``control_values``: the state on which to apply the controlled operation (see below)
    * ``work_wires``: wires made use of during the decomposition of the operation into native operations

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (2,)
    * Gradient recipe: None

    Args:
        base (array[complex]): a square unitary matrix that will be used to construct a QubitUnitary
            operator, used as the base operator.
        wires (Union[Wires, Sequence[int], or int]): the wires the full
            controlled unitary acts on, composed of the controlled wires followed
            by the target wire.
        control_values (List[int or bool]): a list providing the state of the control qubits to
            control on (default is the all 1s state).
        unitary_check (bool): whether to check whether an array U is unitary when creating the
            operator (default False).
        work_wires (Union[Wires, Sequence[int], or int]): auxiliary wire(s) that may be utilized during
            the decomposition of the operator into native operations.

    **Example**

    The following shows how a single-qubit unitary can be applied to wire ``2`` with control on
    both wires ``0`` and ``1``:

    >>> U = np.array([[ 0.94877869,  0.31594146], [-0.31594146,  0.94877869]])
    >>> qml.ControlledQubitUnitary(U, wires=[0, 1, 2])
    Controlled(QubitUnitary(array([[ 0.948...,  0.3159...],
        [-0.3159...,  0.948...]]), wires=[2]), control_wires=[0, 1])

    Typically, controlled operations apply a desired gate if the control qubits
    are all in the state :math:`\vert 1\rangle`. However, there are some situations where
    it is necessary to apply a gate conditioned on all qubits being in the
    :math:`\vert 0\rangle` state, or a mix of the two.

    The state on which to control can be changed by passing a string of bits to
    `control_values`. For example, if we want to apply a single-qubit unitary to
    wire ``3`` conditioned on three wires where the first is in state ``0``, the
    second is in state ``1``, and the third in state ``1``, we can write:

    >>> qml.ControlledQubitUnitary(U, wires=[0, 1, 2, 3], control_values=[0, 1, 1])
    Controlled(QubitUnitary(array([[ 0.948...,  0.3159...],
           [-0.3159...,  0.948...]]), wires=[3]), control_wires=[0, 1, 2], control_values=[False, True, True])

    or

    >>> qml.ControlledQubitUnitary(U, wires=[0, 1, 2, 3], control_values=[False, True, True])
    Controlled(QubitUnitary(array([[ 0.948...,  0.3159...],
           [-0.3159...,  0.948...]]), wires=[3]), control_wires=[0, 1, 2], control_values=[False, True, True])
    """

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (2,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = {
        "num_target_wires",
        "num_control_wires",
        "num_zero_control_values",
        "num_work_wires",
        "work_wire_type",
    }

    grad_method = None
    """Gradient computation method."""

    def _flatten(self):
        return (self.base.data[0],), (
            self.wires,
            tuple(self.control_values),
            self.work_wires,
            self.work_wire_type,
        )

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(
            data[0],
            wires=metadata[0],
            control_values=metadata[1],
            work_wires=metadata[2],
            work_wire_type=metadata[3],
        )

    # pylint: disable=too-many-arguments,unused-argument,too-many-positional-arguments
    @classmethod
    def _primitive_bind_call(
        cls,
        base,
        wires: WiresLike,
        control_values=None,
        unitary_check=False,
        work_wires: WiresLike = (),
        work_wire_type="borrowed",
    ):

        work_wires = Wires(() if work_wires is None else work_wires)
        return cls._primitive.bind(
            base,
            wires=wires,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        base,
        wires: WiresLike,
        control_values=None,
        unitary_check=False,
        work_wires: WiresLike = (),
        work_wire_type: str | None = "borrowed",
    ):

        if wires is None:
            raise TypeError("Must specify a set of wires. None is not a valid `wires` label.")

        if not isinstance(base, Iterable):
            raise ValueError("Base must be a matrix.")

        work_wires = Wires(() if work_wires is None else work_wires)

        num_base_wires = int(qml.math.log2(qml.math.shape(base)[-1]))
        target_wires = wires[-num_base_wires:]
        control_wires = wires[:-num_base_wires]

        # We use type.__call__ instead of calling the class directly so that we don't bind the
        # operator primitive when new program capture is enabled
        base = type.__call__(
            qml.QubitUnitary, base, wires=target_wires, unitary_check=unitary_check
        )

        super().__init__(
            base,
            control_wires,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )

        self._name = "ControlledQubitUnitary"

    @property
    def resource_params(self) -> dict:
        return {
            "num_target_wires": len(self.base.wires),
            "num_control_wires": len(self.control_wires),
            "num_zero_control_values": len([val for val in self.control_values if not val]),
            "num_work_wires": len(self.work_wires),
            "work_wire_type": self.work_wire_type,
        }

    def _controlled(self, wire):
        ctrl_wires = wire + self.control_wires
        values = None if self.control_values is None else [True] + self.control_values
        base = self.base
        if isinstance(self.base, qml.QubitUnitary):
            base = self.base.matrix()

        return ControlledQubitUnitary(
            base,
            wires=ctrl_wires + self.wires,
            control_values=values,
            work_wires=self.work_wires,
            work_wire_type=self.work_wire_type,
        )


add_decomps(
    ControlledQubitUnitary,
    flip_zero_control(ctrl_decomp_bisect_rule),
    flip_zero_control(single_ctrl_decomp_zyz_rule),
    flip_zero_control(multi_control_decomp_zyz_rule),
    controlled_two_qubit_unitary_rule,
)


class CH(ControlledOp):
    r"""CH(wires)
    The controlled-Hadamard operator

    .. math:: CH = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
            0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """

    num_wires = 2
    """int: Number of wires that the operation acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = ()
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    name = "CH"

    def _flatten(self):
        return tuple(), (self.wires,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(metadata[0])

    @classmethod
    def _primitive_bind_call(cls, wires, id=None):
        return cls._primitive.bind(*wires, n_wires=2)

    def __init__(self, wires, id=None):
        control_wires = wires[:1]
        target_wires = wires[1:]

        # We use type.__call__ instead of calling the class directly so that we don't bind the
        # operator primitive when new program capture is enabled
        base = type.__call__(qml.Hadamard, wires=target_wires)
        super().__init__(base, control_wires, id=id)

    def __repr__(self):
        return f"CH(wires={self.wires.tolist()})"

    def adjoint(self):
        return CH(self.wires)

    @property
    def resource_params(self) -> dict:
        return {}

    @staticmethod
    @lru_cache
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CH.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.CH.compute_matrix())
        [[ 1.          0.          0.          0.        ]
         [ 0.          1.          0.          0.        ]
         [ 0.          0.          0.707...  0.707...]
         [ 0.          0.          0.707... -0.707...]]
        """
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, INV_SQRT2, INV_SQRT2],
                [0, 0, INV_SQRT2, -INV_SQRT2],
            ]
        )

    @staticmethod
    def compute_decomposition(wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators (static method).


        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.CH.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.CH.compute_decomposition([0, 1]))
        [RY(-0.7853981633974483, wires=[1]), CZ(wires=[0, 1]), RY(0.7853981633974483, wires=[1])]

        """
        return [
            qml.RY(-np.pi / 4, wires=wires[1]),
            qml.CZ(wires=wires),
            qml.RY(+np.pi / 4, wires=wires[1]),
        ]


def _ch_to_ry_cz_ry_resources():
    return {qml.RY: 2, qml.CZ: 1}


@register_resources(_ch_to_ry_cz_ry_resources)
def _ch_to_ry_cz_ry(wires: WiresLike, **__):
    qml.RY(-np.pi / 4, wires=wires[1])
    qml.CZ(wires=wires)
    qml.RY(+np.pi / 4, wires=wires[1])


add_decomps(CH, _ch_to_ry_cz_ry)
add_decomps("Adjoint(CH)", self_adjoint)
add_decomps("Pow(CH)", pow_involutory)


class CY(ControlledOp):
    r"""CY(wires)
    The controlled-Y operator

    .. math:: CY = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & 0 & -i\\
            0 & 0 & i & 0
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.
    """

    num_wires = 2
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = ()
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    name = "CY"

    def _flatten(self):
        return tuple(), (self.wires,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(metadata[0])

    @classmethod
    def _primitive_bind_call(cls, wires, id=None):
        return cls._primitive.bind(*wires, n_wires=2)

    def __init__(self, wires, id=None):
        # We use type.__call__ instead of calling the class directly so that we don't bind the
        # operator primitive when new program capture is enabled
        base = type.__call__(qml.Y, wires=wires[1:])
        super().__init__(base, wires[:1], id=id)

    def __repr__(self):
        return f"CY(wires={self.wires.tolist()})"

    @property
    def resource_params(self) -> dict:
        return {}

    def adjoint(self):
        return CY(self.wires)

    @staticmethod
    @lru_cache
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CY.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.CY.compute_matrix())
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j -0.-1.j]
         [ 0.+0.j  0.+0.j  0.+1.j  0.+0.j]]
        """
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, -1j],
                [0, 0, 1j, 0],
            ]
        )

    @staticmethod
    def compute_decomposition(wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators (static method).


        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.CY.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.CY.compute_decomposition([0, 1]))
        [CRY(3.141592653589793, wires=[0, 1])), S(0)]

        """
        return [qml.CRY(np.pi, wires=wires), qml.S(wires=wires[0])]


def _cy_to_cry_s_resources():
    return {qml.CRY: 1, qml.S: 1}


@register_resources(_cy_to_cry_s_resources)
def _cy(wires: WiresLike, **__):
    qml.CRY(np.pi, wires=wires)
    qml.S(wires=wires[0])


add_decomps(CY, _cy)
add_decomps("Adjoint(CY)", self_adjoint)
add_decomps("Pow(CY)", pow_involutory)


class CZ(ControlledOp):
    r"""CZ(wires)
    The controlled-Z operator

    .. math:: CZ = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & 1 & 0\\
            0 & 0 & 0 & -1
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """

    num_wires = 2
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = ()
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    name = "CZ"

    def _flatten(self):
        return tuple(), (self.wires,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(metadata[0])

    @classmethod
    def _primitive_bind_call(cls, wires, id=None):
        return cls._primitive.bind(*wires, n_wires=2)

    def __init__(self, wires, id=None):
        # We use type.__call__ instead of calling the class directly so that we don't bind the
        # operator primitive when new program capture is enabled
        base = type.__call__(qml.Z, wires=wires[1:])
        super().__init__(base, wires[:1], id=id)

    def __repr__(self):
        return f"CZ(wires={self.wires.tolist()})"

    @property
    def resource_params(self) -> dict:
        return {}

    def adjoint(self):
        return CZ(self.wires)

    @staticmethod
    @lru_cache
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CZ.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.CZ.compute_matrix())
        [[ 1  0  0  0]
         [ 0  1  0  0]
         [ 0  0  1  0]
         [ 0  0  0 -1]]
        """
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

    def _controlled(self, wire):
        return qml.CCZ(wires=wire + self.wires)

    @staticmethod
    def compute_decomposition(wires):  # pylint: disable=arguments-differ
        return [qml.ControlledPhaseShift(np.pi, wires=wires)]


def _cz_to_cps_resources():
    return {qml.ControlledPhaseShift: 1}


@register_resources(_cz_to_cps_resources)
def _cz_to_cps(wires: WiresLike, **__):
    qml.ControlledPhaseShift(np.pi, wires=wires)


def _cz_to_cnot_resources():
    return {qml.H: 2, qml.CNOT: 1}


@register_resources(_cz_to_cnot_resources)
def _cz_to_cnot(wires: WiresLike, **__):
    qml.H(wires=wires[1])
    qml.CNOT(wires=wires)
    qml.H(wires=wires[1])


add_decomps(CZ, _cz_to_cps, _cz_to_cnot)
add_decomps("Adjoint(CZ)", self_adjoint)
add_decomps("Pow(CZ)", pow_involutory)


class CSWAP(ControlledOp):
    r"""CSWAP(wires)
    The controlled-swap operator

    .. math:: CSWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """

    num_wires = 3
    """int : Number of wires that the operation acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = ()
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    name = "CSWAP"

    def _flatten(self):
        return tuple(), (self.wires,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(metadata[0])

    @classmethod
    def _primitive_bind_call(cls, wires, id=None):
        return cls._primitive.bind(*wires, n_wires=3)

    def __init__(self, wires, id=None):
        control_wires = wires[:1]
        target_wires = wires[1:]

        # We use type.__call__ instead of calling the class directly so that we don't bind the
        # operator primitive when new program capture is enabled
        base = type.__call__(qml.SWAP, wires=target_wires)
        super().__init__(base, control_wires, id=id)

    def __repr__(self):
        return f"CSWAP(wires={self.wires.tolist()})"

    @property
    def resource_params(self) -> dict:
        return {}

    def adjoint(self):
        return CSWAP(self.wires)

    @staticmethod
    @lru_cache
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CSWAP.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.CSWAP.compute_matrix())
        [[1 0 0 0 0 0 0 0]
         [0 1 0 0 0 0 0 0]
         [0 0 1 0 0 0 0 0]
         [0 0 0 1 0 0 0 0]
         [0 0 0 0 1 0 0 0]
         [0 0 0 0 0 0 1 0]
         [0 0 0 0 0 1 0 0]
         [0 0 0 0 0 0 0 1]]
        """
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

    @staticmethod
    def compute_decomposition(wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.CSWAP.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.CSWAP.compute_decomposition((0,1,2)))
        [Toffoli(wires=[0, 2, 1]), Toffoli(wires=[0, 1, 2]), Toffoli(wires=[0, 2, 1])]

        """
        decomp_ops = [
            qml.Toffoli(wires=[wires[0], wires[2], wires[1]]),
            qml.Toffoli(wires=[wires[0], wires[1], wires[2]]),
            qml.Toffoli(wires=[wires[0], wires[2], wires[1]]),
        ]
        return decomp_ops


def _cswap_to_toffoli_resources():
    return {qml.Toffoli: 3}


@register_resources(_cswap_to_toffoli_resources)
def _cswap(wires: WiresLike, **__):
    qml.Toffoli(wires=[wires[0], wires[2], wires[1]])
    qml.Toffoli(wires=[wires[0], wires[1], wires[2]])
    qml.Toffoli(wires=[wires[0], wires[2], wires[1]])


add_decomps(CSWAP, _cswap)
add_decomps("Adjoint(CSWAP)", self_adjoint)
add_decomps("Pow(CSWAP)", pow_involutory)


class CCZ(ControlledOp):
    r"""CCZ(wires)
    CCZ (controlled-controlled-Z) gate.

    .. math::

        CCZ =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
        \end{pmatrix}

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the subsystem the gate acts on
    """

    @classmethod
    def _primitive_bind_call(cls, wires, id=None):
        return cls._primitive.bind(*wires, n_wires=3)

    def _flatten(self):
        return tuple(), (self.wires,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(metadata[0])

    num_wires = 3
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = ()
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    name = "CCZ"

    def __init__(self, wires, id=None):
        control_wires = wires[:2]
        target_wires = wires[2:]

        # We use type.__call__ instead of calling the class directly so that we don't bind the
        # operator primitive when new program capture is enabled
        base = type.__call__(qml.Z, wires=target_wires)
        super().__init__(base, control_wires, id=id)

    def __repr__(self):
        return f"CCZ(wires={self.wires.tolist()})"

    @property
    def resource_params(self) -> dict:
        return {}

    def adjoint(self):
        return CCZ(self.wires)

    @staticmethod
    @lru_cache
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CCZ.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.CCZ.compute_matrix())
        [[ 1  0  0  0  0  0  0  0]
        [ 0  1  0  0  0  0  0  0]
        [ 0  0  1  0  0  0  0  0]
        [ 0  0  0  1  0  0  0  0]
        [ 0  0  0  0  1  0  0  0]
        [ 0  0  0  0  0  1  0  0]
        [ 0  0  0  0  0  0  1  0]
        [ 0  0  0  0  0  0  0 -1]]
        """
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, -1],
            ]
        )

    @staticmethod
    def compute_decomposition(
        wires: WiresLike,
    ) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.Toffoli.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CCZ.compute_decomposition((0,1,2))
        [CNOT(wires=[1, 2]),
         Adjoint(T(2)),
         CNOT(wires=[0, 2]),
         T(2),
         CNOT(wires=[1, 2]),
         Adjoint(T(2)),
         CNOT(wires=[0, 2]),
         T(2),
         T(1),
         CNOT(wires=[0, 1]),
         H(2),
         T(0),
         Adjoint(T(1)),
         CNOT(wires=[0, 1]),
         H(2)]

        """
        return [
            qml.CNOT(wires=[wires[1], wires[2]]),
            qml.adjoint(qml.T(wires=wires[2])),
            qml.CNOT(wires=[wires[0], wires[2]]),
            qml.T(wires=wires[2]),
            qml.CNOT(wires=[wires[1], wires[2]]),
            qml.adjoint(qml.T(wires=wires[2])),
            qml.CNOT(wires=[wires[0], wires[2]]),
            qml.T(wires=wires[2]),
            qml.T(wires=wires[1]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.Hadamard(wires=wires[2]),
            qml.T(wires=wires[0]),
            qml.adjoint(qml.T(wires=wires[1])),
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.Hadamard(wires=wires[2]),
        ]


def _ccz_resources():
    return {
        qml.CNOT: 6,
        qml.decomposition.adjoint_resource_rep(qml.T, {}): 3,
        qml.T: 4,
        qml.Hadamard: 2,
    }


@register_resources(_ccz_resources)
def _ccz(wires: WiresLike, **__):
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.adjoint(qml.T(wires=wires[2]))
    qml.CNOT(wires=[wires[0], wires[2]])
    qml.T(wires=wires[2])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.adjoint(qml.T(wires=wires[2]))
    qml.CNOT(wires=[wires[0], wires[2]])
    qml.T(wires=wires[2])
    qml.T(wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.Hadamard(wires=wires[2])
    qml.T(wires=wires[0])
    qml.adjoint(qml.T(wires=wires[1]))
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.Hadamard(wires=wires[2])


add_decomps(CCZ, _ccz)
add_decomps("Adjoint(CCZ)", self_adjoint)
add_decomps("Pow(CCZ)", pow_involutory)


class CNOT(ControlledOp):
    r"""CNOT(wires)
    The controlled-NOT operator

    .. math:: CNOT = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0\\
        0 & 0 & 0 & 1\\
        0 & 0 & 1 & 0
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """

    num_wires = 2
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = ()
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    name = "CNOT"

    def _flatten(self):
        return tuple(), (self.wires,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(metadata[0])

    @classmethod
    def _primitive_bind_call(cls, wires, id=None):
        return cls._primitive.bind(*wires, n_wires=2)

    def __init__(self, wires, id=None):
        # We use type.__call__ instead of calling the class directly so that we don't bind the
        # operator primitive when new program capture is enabled
        base = type.__call__(qml.X, wires=wires[1:])
        super().__init__(base, wires[:1], id=id)

    def adjoint(self):
        return CNOT(self.wires)

    @property
    def has_decomposition(self):
        return False

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):  # -> List["Operator"]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. note::
            Operations making up the decomposition should be queued within the
            ``compute_decomposition`` method.

        .. seealso:: :meth:`~.Operator.decomposition`.

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            wires (Iterable[Any], Wires): wires that the operator acts on
            **hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

        Raises:
            qml.DecompositionUndefinedError
        """
        raise qml.operation.DecompositionUndefinedError

    @property
    def resource_params(self) -> dict:
        return {}

    def __repr__(self):
        return f"CNOT(wires={self.wires.tolist()})"

    @staticmethod
    @lru_cache
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CNOT.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.CNOT.compute_matrix())
        [[1 0 0 0]
         [0 1 0 0]
         [0 0 0 1]
         [0 0 1 0]]
        """
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    def _controlled(self, wire):
        return qml.Toffoli(wires=wire + self.wires)


def _cnot_cz_h_resources():
    return {qml.H: 2, qml.CZ: 1}


@register_resources(_cnot_cz_h_resources)
def _cnot_to_cz_h(wires: WiresLike, **__):
    qml.H(wires[1])
    qml.CZ(wires=wires)
    qml.H(wires[1])


add_decomps(CNOT, _cnot_to_cz_h)
add_decomps("Adjoint(CNOT)", self_adjoint)
add_decomps("Pow(CNOT)", pow_involutory)


class Toffoli(ControlledOp):
    r"""Toffoli(wires)
    Toffoli (controlled-controlled-X) gate.

    .. math::

        Toffoli =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
        \end{pmatrix}

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the subsystem the gate acts on
    """

    num_wires = 3
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = ()
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    name = "Toffoli"

    def _flatten(self):
        return tuple(), (self.wires,)

    @classmethod
    def _unflatten(cls, _, metadata):
        return cls(metadata[0])

    @classmethod
    def _primitive_bind_call(cls, wires, id=None):
        return cls._primitive.bind(*wires, n_wires=3)

    def __init__(self, wires, id=None):
        control_wires = wires[:2]
        target_wires = wires[2:]
        # We use type.__call__ instead of calling the class directly so that we don't bind the
        # operator primitive when new program capture is enabled
        base = type.__call__(qml.X, wires=target_wires)
        super().__init__(base, control_wires, id=id)

    def __repr__(self):
        return f"Toffoli(wires={self.wires.tolist()})"

    @property
    def resource_params(self) -> dict:
        return {}

    def adjoint(self):
        return Toffoli(self.wires)

    @staticmethod
    @lru_cache
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Toffoli.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.Toffoli.compute_matrix())
        [[1 0 0 0 0 0 0 0]
         [0 1 0 0 0 0 0 0]
         [0 0 1 0 0 0 0 0]
         [0 0 0 1 0 0 0 0]
         [0 0 0 0 1 0 0 0]
         [0 0 0 0 0 1 0 0]
         [0 0 0 0 0 0 0 1]
         [0 0 0 0 0 0 1 0]]
        """
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )

    @staticmethod
    def compute_decomposition(
        wires: WiresLike,
    ) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.Toffoli.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.Toffoli.compute_decomposition((0,1,2))
        [H(2),
         CNOT(wires=[1, 2]),
         Adjoint(T(2)),
         CNOT(wires=[0, 2]),
         T(2),
         CNOT(wires=[1, 2]),
         Adjoint(T(2)),
         CNOT(wires=[0, 2]),
         T(2),
         T(1),
         CNOT(wires=[0, 1]),
         H(2),
         T(0),
         Adjoint(T(1)),
         CNOT(wires=[0, 1])]

        """
        return [
            qml.Hadamard(wires=wires[2]),
            CNOT(wires=[wires[1], wires[2]]),
            qml.adjoint(qml.T(wires=wires[2])),
            CNOT(wires=[wires[0], wires[2]]),
            qml.T(wires=wires[2]),
            CNOT(wires=[wires[1], wires[2]]),
            qml.adjoint(qml.T(wires=wires[2])),
            CNOT(wires=[wires[0], wires[2]]),
            qml.T(wires=wires[2]),
            qml.T(wires=wires[1]),
            CNOT(wires=[wires[0], wires[1]]),
            qml.Hadamard(wires=wires[2]),
            qml.T(wires=wires[0]),
            qml.adjoint(qml.T(wires=wires[1])),
            CNOT(wires=[wires[0], wires[1]]),
        ]


def _check_and_convert_control_values(control_values, control_wires):
    if isinstance(control_values, str):
        # Make sure all values are either 0 or 1
        if not set(control_values).issubset({"1", "0"}):
            raise ValueError("String of control values can contain only '0' or '1'.")

        control_values = [int(x) for x in control_values]

    if control_values is None:
        return [1] * len(control_wires)

    if len(control_values) != len(control_wires):
        raise ValueError("Length of control values must equal number of control wires.")

    return control_values


def _toffoli_resources():
    return {
        qml.Hadamard: 2,
        qml.CNOT: 6,
        qml.T: 4,
        qml.decomposition.adjoint_resource_rep(qml.T, {}): 3,
    }


@register_resources(_toffoli_resources)
def _toffoli(wires: WiresLike, **__):
    qml.Hadamard(wires=wires[2])
    CNOT(wires=[wires[1], wires[2]])
    qml.adjoint(qml.T(wires=wires[2]))
    CNOT(wires=[wires[0], wires[2]])
    qml.T(wires=wires[2])
    CNOT(wires=[wires[1], wires[2]])
    qml.adjoint(qml.T(wires=wires[2]))
    CNOT(wires=[wires[0], wires[2]])
    qml.T(wires=wires[2])
    qml.T(wires=wires[1])
    CNOT(wires=[wires[0], wires[1]])
    qml.Hadamard(wires=wires[2])
    qml.T(wires=wires[0])
    qml.adjoint(qml.T(wires=wires[1]))
    CNOT(wires=[wires[0], wires[1]])


add_decomps(Toffoli, _toffoli)
add_decomps("Adjoint(Toffoli)", self_adjoint)
add_decomps("Pow(Toffoli)", pow_involutory)


class MultiControlledX(ControlledOp):
    r"""Apply a :class:`~.PauliX` gate controlled on an arbitrary computational basis state.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 0
    * Gradient recipe: None

    Args:
        wires (Union[Wires, Sequence[int], or int]): control wire(s) followed by a single target wire (the last entry of ``wires``) where
            the operation acts on
        control_values (Union[bool, list[bool], int, list[int]]): The value(s) the control wire(s)
            should take. Integers other than 0 or 1 will be treated as :code:`int(bool(x))`.
        work_wires (Union[Wires, Sequence[int], or int]): optional work wires used to decompose
            the operation into a series of :class:`~.Toffoli` gates
        work_wire_type (str): whether the work wires are ``"zeroed"`` or ``"borrowed"``. ``"zeroed"`` indicates that
            the work wires are in the state :math:`|0\rangle`, while ``"borrowed"`` indicates that the
            work wires are in an arbitrary state. Defaults to ``"borrowed"``.

    .. note::

        If :class:`~.MultiControlledX` is not supported on the targeted device, PennyLane will decompose
        the operation into :class:`~.Toffoli` and/or :class:`~.CNOT` gates. When controlling on
        three or more wires, the Toffoli-based decompositions described in Lemmas 7.2 of
        `Barenco et al. <https://arxiv.org/abs/quant-ph/9503016>`__ and Sec 5 of `Khattar and Gidney
        <https://arxiv.org/abs/2407.17966>`__  will be used. These methods require at least one
        work wire.

        The number of work wires provided determines the decomposition method used and the resulting
        number of Toffoli gates required. When :class:`~.MultiControlledX` is controlling on :math:`n`
        wires:

        #. If at least :math:`n - 2` work wires are provided, the decomposition in Lemma 7.2 will be
           applied using the first :math:`n - 2` work wires.
        #. If at least :math:`2` work wires are provided, Sec. 5.2 and 5.4 of Khattar and Gidney
           will be used depending on whether the ``work_wire_type`` is ``"zeroed"`` or ``"borrowed"``.
        #. If at least :math:`1` work wire is provided, Sec. 5.1 and 5.3 of Khattar and Gidney
           will be used depending on whether the ``work_wire_type`` is ``"zeroed"`` or ``"borrowed"``.

        These methods present a tradeoff between qubit number and depth. The method in point 1
        requires fewer Toffoli gates but a greater number of qubits.

        Note that the state of the work wires before and after the decomposition takes place is
        unchanged.

    """

    is_self_inverse = True
    """bool: Whether or not the operator is self-inverse."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = ()
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = {
        "num_control_wires",
        "num_zero_control_values",
        "num_work_wires",
        "work_wire_type",
    }

    name = "MultiControlledX"

    def _flatten(self):
        return (), (self.wires, tuple(self.control_values), self.work_wires, self.work_wire_type)

    @classmethod
    def _unflatten(cls, _, metadata):
        return cls(
            wires=metadata[0],
            control_values=metadata[1],
            work_wires=metadata[2],
            work_wire_type=metadata[3],
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments

    @classmethod
    def _primitive_bind_call(
        cls, wires, control_values=None, work_wires=None, work_wire_type="borrowed", id=None
    ):
        return cls._primitive.bind(
            *wires,
            n_wires=len(wires),
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )

    @staticmethod
    def _validate_control_values(control_values):
        if control_values is not None:
            if not (
                isinstance(control_values, (bool, int))
                or (
                    isinstance(control_values, (list, tuple))
                    and all(isinstance(val, (bool, int)) for val in control_values)
                )
            ):
                raise ValueError(f"control_values must be boolean or int. Got: {control_values}")

    def __init__(
        self,
        wires: WiresLike,
        control_values: None | bool | list[bool] | int | list[int] = None,
        work_wires: WiresLike = (),
        work_wire_type: Literal["zeroed", "borrowed"] = "borrowed",
    ):
        wires = Wires(() if wires is None else wires)
        work_wires = Wires(() if work_wires is None else work_wires)
        self._validate_control_values(control_values)

        if len(wires) == 0:
            raise ValueError("Must specify the wires where the operation acts on")

        if len(wires) < 2:
            raise ValueError(
                f"MultiControlledX: wrong number of wires. {len(wires)} wire(s) given. "
                f"Need at least 2."
            )
        control_wires = wires[:-1]
        wires = wires[-1:]

        control_values = _check_and_convert_control_values(control_values, control_wires)

        # We use type.__call__ instead of calling the class directly so that we don't bind the
        # operator primitive when new program capture is enabled
        base = type.__call__(qml.X, wires=wires)
        super().__init__(
            base,
            control_wires=control_wires,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )

    def __repr__(self):
        return (
            f"MultiControlledX(wires={self.wires.tolist()}, control_values={self.control_values})"
        )

    @property
    def wires(self):
        return self.control_wires + self.target_wires

    @property
    def resource_params(self) -> dict:
        return {
            "num_control_wires": len(self.control_wires),
            "num_zero_control_values": len([val for val in self.control_values if not val]),
            "num_work_wires": len(self.work_wires),
            "work_wire_type": self.work_wire_type,
        }

    def adjoint(self):
        return MultiControlledX(
            wires=self.wires,
            control_values=self.control_values,
            work_wires=self.work_wires,
            work_wire_type=self.work_wire_type,
        )

    # pylint: disable=unused-argument, arguments-differ
    @staticmethod
    def compute_matrix(control_wires: WiresLike, control_values=None, **kwargs):
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.MultiControlledX.matrix`

        Args:
            control_wires (Any or Iterable[Any]): wires to place controls on
            control_values (Union[bool, list[bool], int, list[int]]): The value(s) the control wire(s)
                should take. Integers other than 0 or 1 will be treated as ``int(bool(x))``.

        Returns:
            tensor_like: matrix representation

        **Example**

        >>> print(qml.MultiControlledX.compute_matrix([0], [1]))
        [[1. 0. 0. 0.]
         [0. 1. 0. 0.]
         [0. 0. 0. 1.]
         [0. 0. 1. 0.]]
        >>> print(qml.MultiControlledX.compute_matrix([1], [0]))
        [[0. 1. 0. 0.]
         [1. 0. 0. 0.]
         [0. 0. 1. 0.]
         [0. 0. 0. 1.]]

        """

        control_values = _check_and_convert_control_values(control_values, control_wires)
        padding_left = sum(2**i * int(val) for i, val in enumerate(reversed(control_values))) * 2
        padding_right = 2 ** (len(control_wires) + 1) - 2 - padding_left
        return block_diag(np.eye(padding_left), qml.X.compute_matrix(), np.eye(padding_right))

    def matrix(self, wire_order=None):
        canonical_matrix = self.compute_matrix(self.control_wires, self.control_values)
        wire_order = wire_order or self.wires
        return qml.math.expand_matrix(canonical_matrix, wires=self.wires, wire_order=wire_order)

    # pylint: disable=unused-argument, arguments-differ
    @staticmethod
    def compute_decomposition(
        wires: WiresLike = None,
        work_wires: WiresLike = None,
        control_values=None,
        work_wire_type: Literal["zeroed", "borrowed"] = "borrowed",
        **kwargs,
    ):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.MultiControlledX.decomposition`.

        Args:
            wires (Iterable[Any] or Wires): wires that the operation acts on
            work_wires (Wires): optional work wires used to decompose
                the operation into a series of Toffoli gates.
            control_values (Union[bool, list[bool], int, list[int]]): The value(s) the control wire(s)
                should take. Integers other than 0 or 1 will be treated as ``int(bool(x))``.
            work_wire_type (str): whether the work wires are zeroed or borrowed.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        .. code-block:: python

            decomp = qml.MultiControlledX.compute_decomposition(wires=[0,1,2,3], control_values=[1,1,1], work_wires=qml.wires.Wires("aux"))

        >>> print(decomp)
        [Toffoli(wires=[0, 'aux', 3]), Toffoli(wires=[2, 1, 'aux']), Toffoli(wires=[0, 'aux', 3]), Toffoli(wires=[2, 1, 'aux'])]

        """
        wires = Wires(() if wires is None else wires)

        if len(wires) < 2:
            raise ValueError(f"Wrong number of wires. {len(wires)} given. Need at least 2.")

        target_wire = wires[-1]
        control_wires = wires[:-1]

        if control_values is None:
            control_values = [True] * len(control_wires)

        work_wires = work_wires or []

        flips1 = [qml.X(w) for w, val in zip(control_wires, control_values) if not val]

        if work_wire_type not in {"zeroed", "borrowed"}:
            raise ValueError(
                f"work_wire_type must be either 'zeroed' or 'borrowed'. Got '{work_wire_type}'."
            )

        decomp = decompose_mcx(control_wires, target_wire, work_wires, work_wire_type)

        flips2 = [qml.X(w) for w, val in zip(control_wires, control_values) if not val]

        return flips1 + decomp + flips2

    def decomposition(self):
        return self.compute_decomposition(
            self.wires, self.work_wires, self.control_values, self.work_wire_type
        )


def _mcx_to_cnot_or_toffoli_resource(num_control_wires, num_zero_control_values, **__):
    if num_control_wires == 1:
        return {qml.CNOT: 1, qml.X: num_zero_control_values}
    return {qml.Toffoli: 1, qml.X: num_zero_control_values * 2}


@register_condition(lambda num_control_wires, **_: num_control_wires < 3)
@register_resources(_mcx_to_cnot_or_toffoli_resource)
def _mcx_to_cnot_or_toffoli(wires, control_wires, control_values, **__):
    if len(wires) == 2 and not control_values[0]:
        qml.CNOT(wires=wires)
        qml.X(wires[1])
    elif len(wires) == 2:
        qml.CNOT(wires=wires)
    elif len(wires) == 3:
        zero_control_wires = [w for w, val in zip(control_wires, control_values) if not val]
        for w in zero_control_wires:
            qml.PauliX(w)
        qml.Toffoli(wires=wires)
        for w in zero_control_wires:
            qml.PauliX(w)


add_decomps(
    MultiControlledX,
    _mcx_to_cnot_or_toffoli,
    decompose_mcx_many_workers_explicit,
    decompose_mcx_many_borrowed_workers,
    decompose_mcx_many_zeroed_workers,
    decompose_mcx_two_workers_explicit,
    decompose_mcx_two_borrowed_workers,
    decompose_mcx_two_zeroed_workers,
    decompose_mcx_one_worker_explicit,
    decompose_mcx_one_borrowed_worker,
    decompose_mcx_one_zeroed_worker,
    decompose_mcx_with_no_worker,
)
add_decomps("Adjoint(MultiControlledX)", self_adjoint)
add_decomps("Pow(MultiControlledX)", pow_involutory)


class CRX(ControlledOp):
    r"""The controlled-RX operator

    .. math::

        \begin{align}
            CR_x(\phi) &=
            \begin{bmatrix}
            & 1 & 0 & 0 & 0 \\
            & 0 & 1 & 0 & 0\\
            & 0 & 0 & \cos(\phi/2) & -i\sin(\phi/2)\\
            & 0 & 0 & -i\sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.
        \end{align}

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: The controlled-RX operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://doi.org/10.1088/1367-2630/ac2cb3):

      .. math::

          \frac{d}{d\phi}f(CR_x(\phi)) = c_+ \left[f(CR_x(\phi+a)) - f(CR_x(\phi-a))\right] - c_- \left[f(CR_x(\phi+b)) - f(CR_x(\phi-b))\right]

      where :math:`f` is an expectation value depending on :math:`CR_x(\phi)`, and

      - :math:`a = \pi/2`
      - :math:`b = 3\pi/2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 2
    """int: Number of wires that the operation acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    name = "CRX"
    parameter_frequencies = [(0.5, 1.0)]

    def __init__(self, phi, wires: WiresLike, id=None):
        # We use type.__call__ instead of calling the class directly so that we don't bind the
        # operator primitive when new program capture is enabled
        base = type.__call__(qml.RX, phi, wires=wires[1:])
        super().__init__(base, control_wires=wires[:1], id=id)

    def __repr__(self):
        return f"CRX({self.data[0]}, wires={self.wires.tolist()})"

    def _flatten(self):
        return self.data, (self.wires,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data, wires=metadata[0])

    @classmethod
    def _primitive_bind_call(cls, phi, wires: WiresLike, id=None):
        return cls._primitive.bind(phi, *wires, n_wires=len(wires))

    @property
    def resource_params(self) -> dict:
        return {}

    def adjoint(self):
        return CRX(-self.data[0], wires=self.wires)

    @staticmethod
    def compute_matrix(theta):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CRX.matrix`

        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.CRX.compute_matrix(torch.tensor(0.5))
        tensor([[1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.9689+0.0000j, 0.0000-0.2474j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.0000-0.2474j, 0.9689+0.0000j]])
        """

        interface = qml.math.get_interface(theta)

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        if (
            interface == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)

        # The following avoids casting an imaginary quantity to reals when back propagating
        c = (1 + 0j) * c
        js = -1j * s
        ones = qml.math.ones_like(js)
        zeros = qml.math.zeros_like(js)
        matrix = [
            [ones, zeros, zeros, zeros],
            [zeros, ones, zeros, zeros],
            [zeros, zeros, c, js],
            [zeros, zeros, js, c],
        ]

        return qml.math.stack([stack_last(row) for row in matrix], axis=-2)

    @staticmethod
    def compute_decomposition(phi: TensorLike, wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.CRot.decomposition`.

        Args:
            phi (TensorLike): rotation angle :math:`\phi`
            wires (Iterable, Wires): the wires the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CRX.compute_decomposition(1.2, wires=(0,1))
        [RZ(np.float64(1.5707963267948966), wires=[1]), RY(0.6, wires=[1]), CNOT(wires=[0, 1]), RY(-0.6, wires=[1]), CNOT(wires=[0, 1]), RZ(np.float64(-1.5707963267948966), wires=[1])]

        """
        pi_half = qml.math.ones_like(phi) * (np.pi / 2)
        return [
            qml.RZ(pi_half, wires=wires[1]),
            qml.RY(phi / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            qml.RY(-phi / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            qml.RZ(-pi_half, wires=wires[1]),
        ]


def _crx_to_rz_ry_resources():
    return {qml.RZ: 2, qml.RY: 2, qml.CNOT: 2}


@register_resources(_crx_to_rz_ry_resources)
def _crx_to_rz_ry(phi: TensorLike, wires: WiresLike, **__):
    qml.RZ(np.pi / 2, wires=wires[1])
    qml.RY(phi / 2, wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RY(-phi / 2, wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RZ(-np.pi / 2, wires=wires[1])


def _crx_to_rx_cz_resources():
    return {qml.RX: 2, qml.CZ: 2}


@register_resources(_crx_to_rx_cz_resources)
def _crx_to_rx_cz(phi: TensorLike, wires: WiresLike, **__):
    qml.RX(phi / 2, wires=wires[1])
    qml.CZ(wires=wires)
    qml.RX(-phi / 2, wires=wires[1])
    qml.CZ(wires=wires)


def _crx_to_h_crz_resources():
    return {qml.Hadamard: 2, qml.CRZ: 1}


@register_resources(_crx_to_h_crz_resources)
def _crx_to_h_crz(phi: TensorLike, wires: WiresLike, **__):
    qml.Hadamard(wires=wires[1])
    qml.CRZ(phi, wires=wires)
    qml.Hadamard(wires=wires[1])


add_decomps(CRX, _crx_to_rx_cz, _crx_to_rz_ry, _crx_to_h_crz)
add_decomps("Adjoint(CRX)", adjoint_rotation)
add_decomps("Pow(CRX)", pow_rotation)


class CRY(ControlledOp):
    r"""The controlled-RY operator

    .. math::

        \begin{align}
            CR_y(\phi) &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0\\
                0 & 0 & \cos(\phi/2) & -\sin(\phi/2)\\
                0 & 0 & \sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.
        \end{align}

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: The controlled-RY operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://doi.org/10.1088/1367-2630/ac2cb3):

      .. math::

          \frac{d}{d\phi}f(CR_y(\phi)) = c_+ \left[f(CR_y(\phi+a)) - f(CR_y(\phi-a))\right] - c_- \left[f(CR_y(\phi+b)) - f(CR_y(\phi-b))\right]

      where :math:`f` is an expectation value depending on :math:`CR_y(\phi)`, and

      - :math:`a = \pi/2`
      - :math:`b = 3\pi/2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 2
    """int: Number of wires that the operation acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    name = "CRY"
    parameter_frequencies = [(0.5, 1.0)]

    def __init__(self, phi, wires, id=None):
        # We use type.__call__ instead of calling the class directly so that we don't bind the
        # operator primitive when new program capture is enabled
        base = type.__call__(qml.RY, phi, wires=wires[1:])
        super().__init__(base, control_wires=wires[:1], id=id)

    def __repr__(self):
        return f"CRY({self.data[0]}, wires={self.wires.tolist()}))"

    def _flatten(self):
        return self.data, (self.wires,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data, wires=metadata[0])

    @classmethod
    def _primitive_bind_call(cls, phi, wires, id=None):
        return cls._primitive.bind(phi, *wires, n_wires=len(wires))

    @property
    def resource_params(self) -> dict:
        return {}

    def adjoint(self):
        return CRY(-self.data[0], wires=self.wires)

    @staticmethod
    def compute_matrix(theta):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CRY.matrix`


        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.CRY.compute_matrix(torch.tensor(0.5))
        tensor([[ 1.0000+0.j,  0.0000+0.j,  0.0000+0.j,  0.0000+0.j],
                [ 0.0000+0.j,  1.0000+0.j,  0.0000+0.j,  0.0000+0.j],
                [ 0.0000+0.j,  0.0000+0.j,  0.9689+0.j, -0.2474-0.j],
                [ 0.0000+0.j,  0.0000+0.j,  0.2474+0.j,  0.9689+0.j]])
        """
        interface = qml.math.get_interface(theta)

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        if (
            interface == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)

        # The following avoids casting an imaginary quantity to reals when back propagating
        c = (1 + 0j) * c
        s = (1 + 0j) * s
        ones = qml.math.ones_like(s)
        zeros = qml.math.zeros_like(s)
        matrix = [
            [ones, zeros, zeros, zeros],
            [zeros, ones, zeros, zeros],
            [zeros, zeros, c, -s],
            [zeros, zeros, s, c],
        ]

        return qml.math.stack([stack_last(row) for row in matrix], axis=-2)

    @staticmethod
    def compute_decomposition(phi: TensorLike, wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.CRY.decomposition`.

        Args:
            phi (TensorLike): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CRY.compute_decomposition(1.2, wires=(0,1))
        [RY(0.6, wires=[1]),
        CNOT(wires=[0, 1]),
        RY(-0.6, wires=[1]),
        CNOT(wires=[0, 1])]

        """
        return [
            qml.RY(phi / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            qml.RY(-phi / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
        ]


def _cry_resources():
    return {qml.RY: 2, qml.CNOT: 2}


@register_resources(_cry_resources)
def _cry(phi: TensorLike, wires: WiresLike, **__):
    qml.RY(phi / 2, wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RY(-phi / 2, wires=wires[1])
    qml.CNOT(wires=wires)


add_decomps(CRY, _cry)
add_decomps("Adjoint(CRY)", adjoint_rotation)
add_decomps("Pow(CRY)", pow_rotation)


class CRZ(ControlledOp):
    r"""The controlled-RZ operator

    .. math::

        \begin{align}
             CR_z(\phi) &=
             \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0\\
                0 & 0 & e^{-i\phi/2} & 0\\
                0 & 0 & 0 & e^{i\phi/2}
            \end{bmatrix}.
        \end{align}


    .. note:: The subscripts of the operations in the formula refer to the wires they act on, e.g. 1 corresponds
        to the first element in ``wires`` that is the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: The controlled-RZ operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://doi.org/10.1088/1367-2630/ac2cb3):

      .. math::

          \frac{d}{d\phi}f(CR_z(\phi)) = c_+ \left[f(CR_z(\phi+a)) - f(CR_z(\phi-a))\right] - c_- \left[f(CR_z(\phi+b)) - f(CR_z(\phi-b))\right]

      where :math:`f` is an expectation value depending on :math:`CR_z(\phi)`, and

      - :math:`a = \pi/2`
      - :math:`b = 3\pi/2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    """

    num_wires = 2
    """int: Number of wires that the operation acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    name = "CRZ"
    parameter_frequencies = [(0.5, 1.0)]

    def __init__(self, phi, wires, id=None):
        # We use type.__call__ instead of calling the class directly so that we don't bind the
        # operator primitive when new program capture is enabled
        base = type.__call__(qml.RZ, phi, wires=wires[1:])
        super().__init__(base, control_wires=wires[:1], id=id)

    def __repr__(self):
        return f"CRZ({self.data[0]}, wires={self.wires})"

    def _flatten(self):
        return self.data, (self.wires,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data, wires=metadata[0])

    @classmethod
    def _primitive_bind_call(cls, phi, wires, id=None):
        return cls._primitive.bind(phi, *wires, n_wires=len(wires))

    @property
    def resource_params(self) -> dict:
        return {}

    def adjoint(self):
        return CRZ(-self.data[0], wires=self.wires)

    @staticmethod
    def compute_matrix(theta):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CRZ.matrix`

        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.CRZ.compute_matrix(torch.tensor(0.5))
        tensor([[1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.9689-0.2474j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9689+0.2474j]])
        """
        if (
            qml.math.get_interface(theta) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            p = qml.math.exp(-0.5j * qml.math.cast_like(theta, 1j))
            if qml.math.ndim(p) == 0:
                return qml.math.diag([1, 1, p, qml.math.conj(p)])

            ones = qml.math.ones_like(p)
            diags = stack_last([ones, ones, p, qml.math.conj(p)])
            return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)

        signs = qml.math.array([0, 0, 1, -1], like=theta)
        arg = -0.5j * theta

        if qml.math.ndim(arg) == 0:
            return qml.math.diag(qml.math.exp(arg * signs))

        diags = qml.math.exp(qml.math.outer(arg, signs))
        return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)

    @staticmethod
    def compute_eigvals(theta, **_):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.CRZ.eigvals`


        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.CRZ.compute_eigvals(torch.tensor(0.5))
        tensor([1.0000+0.0000j, 1.0000+0.0000j, 0.9689-0.2474j, 0.9689+0.2474j])
        """
        if (
            qml.math.get_interface(theta) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phase = qml.math.exp(-0.5j * qml.math.cast_like(theta, 1j))
            ones = qml.math.ones_like(phase)
            return stack_last([ones, ones, phase, qml.math.conj(phase)])

        prefactors = qml.math.array([0, 0, -0.5j, 0.5j], like=theta)
        if qml.math.ndim(theta) == 0:
            product = theta * prefactors
        else:
            product = qml.math.outer(theta, prefactors)
        return qml.math.exp(product)

    def eigvals(self):
        return self.compute_eigvals(*self.parameters)

    @staticmethod
    def compute_decomposition(phi: TensorLike, wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.CRZ.decomposition`.

        Args:
            phi (TensorLike): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CRZ.compute_decomposition(1.2, wires=(0,1))
        [PhaseShift(0.6, wires=[1]),
        CNOT(wires=[0, 1]),
        PhaseShift(-0.6, wires=[1]),
        CNOT(wires=[0, 1])]

        """
        return [
            qml.PhaseShift(phi / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            qml.PhaseShift(-phi / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
        ]


def _crz_resources():
    return {qml.RZ: 2, qml.CNOT: 2}


@register_resources(_crz_resources)
def _crz(phi: TensorLike, wires: WiresLike, **__):
    qml.RZ(phi / 2, wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RZ(-phi / 2, wires=wires[1])
    qml.CNOT(wires=wires)


add_decomps(CRZ, _crz)
add_decomps("Adjoint(CRZ)", adjoint_rotation)
add_decomps("Pow(CRZ)", pow_rotation)


class CRot(ControlledOp):
    r"""The controlled-Rot operator

    .. math:: CR(\phi, \theta, \omega) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
            0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 3
    * Number of dimensions per parameter: (0, 0, 0)
    * Gradient recipe: The controlled-Rot operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://doi.org/10.1088/1367-2630/ac2cb3):

      .. math::

          \frac{d}{d\mathbf{x}_i}f(CR(\mathbf{x}_i)) = c_+ \left[f(CR(\mathbf{x}_i+a)) - f(CR(\mathbf{x}_i-a))\right] - c_- \left[f(CR(\mathbf{x}_i+b)) - f(CR(\mathbf{x}_i-b))\right]

      where :math:`f` is an expectation value depending on :math:`CR(\mathbf{x}_i)`, and

      - :math:`\mathbf{x} = (\phi, \theta, \omega)` and `i` is an index to :math:`\mathbf{x}`
      - :math:`a = \pi/2`
      - :math:`b = 3\pi/2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        omega (float): rotation angle :math:`\omega`
        wires (Sequence[int]): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    """

    num_wires = 2
    """int: Number of wires this operator acts on."""

    num_params = 3
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0, 0, 0)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    name = "CRot"
    parameter_frequencies = [(0.5, 1.0), (0.5, 1.0), (0.5, 1.0)]

    def __init__(
        self, phi, theta, omega, wires, id=None
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        # We use type.__call__ instead of calling the class directly so that we don't bind the
        # operator primitive when new program capture is enabled
        base = type.__call__(qml.Rot, phi, theta, omega, wires=wires[1:])
        super().__init__(base, control_wires=wires[:1], id=id)

    def __repr__(self):
        params = ", ".join([repr(p) for p in self.parameters])
        return f"CRot({params}, wires={self.wires})"

    def _flatten(self):
        return self.data, (self.wires,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data, wires=metadata[0])

    # pylint: disable=too-many-arguments
    @classmethod
    def _primitive_bind_call(
        cls, phi, theta, omega, wires, id=None
    ):  # pylint: disable=too-many-positional-arguments
        return cls._primitive.bind(phi, theta, omega, *wires, n_wires=len(wires))

    @property
    def resource_params(self) -> dict:
        return {}

    def adjoint(self):
        phi, theta, omega = self.parameters
        return CRot(-omega, -theta, -phi, wires=self.wires)

    @staticmethod
    def compute_matrix(phi, theta, omega):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CRot.matrix`


        Args:
            phi(tensor_like or float): first rotation angle
            theta (tensor_like or float): second rotation angle
            omega (tensor_like or float): third rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.CRot.compute_matrix(torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3))
        tensor([[ 1.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
                [ 0.0000+0.0000j,  1.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
                [ 0.0000+0.0000j,  0.0000+0.0000j,  0.9752-0.1977j, -0.0993+0.0100j],
                [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0993+0.0100j,  0.9752+0.1977j]])
        """
        # It might be that they are in different interfaces, e.g.,
        # CRot(0.2, 0.3, tf.Variable(0.5), wires=[0, 1])
        # So we need to make sure the matrix comes out having the right type
        interface = qml.math.get_interface(phi, theta, omega)

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        # If anything is not tensorflow, it has to be casted
        if (
            interface == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phi = qml.math.cast_like(qml.math.asarray(phi, like=interface), 1j)
            omega = qml.math.cast_like(qml.math.asarray(omega, like=interface), 1j)
            c = qml.math.cast_like(qml.math.asarray(c, like=interface), 1j)
            s = qml.math.cast_like(qml.math.asarray(s, like=interface), 1j)

        # The following variable is used to assert the all terms to be stacked have same shape
        one = qml.math.ones_like(phi) * qml.math.ones_like(omega)
        c = c * one
        s = s * one

        o = qml.math.ones_like(c)
        z = qml.math.zeros_like(c)
        mat = [
            [o, z, z, z],
            [z, o, z, z],
            [
                z,
                z,
                qml.math.exp(-0.5j * (phi + omega)) * c,
                -qml.math.exp(0.5j * (phi - omega)) * s,
            ],
            [
                z,
                z,
                qml.math.exp(-0.5j * (phi - omega)) * s,
                qml.math.exp(0.5j * (phi + omega)) * c,
            ],
        ]

        return qml.math.stack([stack_last(row) for row in mat], axis=-2)

    @staticmethod
    def compute_decomposition(
        phi: TensorLike, theta: TensorLike, omega: TensorLike, wires: WiresLike
    ) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.CRot.decomposition`.

        Args:
            phi (TensorLike): rotation angle :math:`\phi`
            theta (TensorLike): rotation angle :math:`\theta`
            omega (TensorLike): rotation angle :math:`\omega`
            wires (Iterable, Wires): the wires the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CRot.compute_decomposition(1.234, 2.34, 3.45, wires=[0, 1])
        [RZ(-1.108, wires=[1]),
         CNOT(wires=[0, 1]),
         RZ(-2.342, wires=[1]),
         RY(-1.17, wires=[1]),
         CNOT(wires=[0, 1]),
         RY(1.17, wires=[1]),
         RZ(3.45, wires=[1])]

        """
        return [
            qml.RZ((phi - omega) / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            qml.RZ(-(phi + omega) / 2, wires=wires[1]),
            qml.RY(-theta / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            qml.RY(theta / 2, wires=wires[1]),
            qml.RZ(omega, wires=wires[1]),
        ]


def _crot_resources():
    return {qml.RZ: 3, qml.CNOT: 2, qml.RY: 2}


@register_resources(_crot_resources)
def _crot(phi: TensorLike, theta: TensorLike, omega: TensorLike, wires: WiresLike, **__):
    qml.RZ((phi - omega) / 2, wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RZ(-(phi + omega) / 2, wires=wires[1])
    qml.RY(-theta / 2, wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RY(theta / 2, wires=wires[1])
    qml.RZ(omega, wires=wires[1])


add_decomps(CRot, _crot)


@register_resources({CRot: 1})
def _adjoint_crot(phi, theta, omega, wires, **_):
    CRot(-omega, -theta, -phi, wires=wires)


add_decomps("Adjoint(CRot)", _adjoint_crot)


class ControlledPhaseShift(ControlledOp):
    r"""A qubit controlled phase shift.

    .. math:: CR_\phi(\phi) = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i\phi}
            \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(CR_\phi(\phi)) = \frac{1}{2}\left[f(CR_\phi(\phi+\pi/2)) - f(CR_\phi(\phi-\pi/2))\right]`
        where :math:`f` is an expectation value depending on :math:`CR_{\phi}(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    """

    num_wires = 2
    """int: Number of wires the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    name = "ControlledPhaseShift"
    parameter_frequencies = [(1,)]

    def __init__(self, phi, wires, id=None):
        # We use type.__call__ instead of calling the class directly so that we don't bind the
        # operator primitive when new program capture is enabled
        base = type.__call__(qml.PhaseShift, phi, wires=wires[1:])
        super().__init__(base, control_wires=wires[:1], id=id)

    def __repr__(self):
        return f"ControlledPhaseShift({self.data[0]}, wires={self.wires})"

    def _flatten(self):
        return self.data, (self.wires,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data, wires=metadata[0])

    @classmethod
    def _primitive_bind_call(cls, phi, wires, id=None):
        return cls._primitive.bind(phi, *wires, n_wires=len(wires))

    @property
    def resource_params(self) -> dict:
        return {}

    def adjoint(self):
        return ControlledPhaseShift(-self.data[0], wires=self.wires)

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.ControlledPhaseShift.matrix`

        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.ControlledPhaseShift.compute_matrix(torch.tensor(0.5))
        tensor([[1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 1.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.8776+0.4794j]])
        """
        if (
            qml.math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            p = qml.math.exp(1j * qml.math.cast_like(phi, 1j))
            if qml.math.ndim(p) == 0:
                return qml.math.diag([1, 1, 1, p])

            ones = qml.math.ones_like(p)
            diags = stack_last([ones, ones, ones, p])
            return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)

        signs = qml.math.array([0, 0, 0, 1], like=phi)
        arg = 1j * phi

        if qml.math.ndim(arg) == 0:
            return qml.math.diag(qml.math.exp(arg * signs))

        diags = qml.math.exp(qml.math.outer(arg, signs))
        return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)

    @staticmethod
    def compute_eigvals(phi, **_):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.ControlledPhaseShift.eigvals`


        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.ControlledPhaseShift.compute_eigvals(torch.tensor(0.5))
        tensor([1.0000+0.0000j, 1.0000+0.0000j, 1.0000+0.0000j, 0.8776+0.4794j])
        """
        if (
            qml.math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phase = qml.math.exp(1j * qml.math.cast_like(phi, 1j))
            ones = qml.math.ones_like(phase)
            return stack_last([ones, ones, ones, phase])

        prefactors = qml.math.array([0, 0, 0, 1j], like=phi)
        if qml.math.ndim(phi) == 0:
            product = phi * prefactors
        else:
            product = qml.math.outer(phi, prefactors)
        return qml.math.exp(product)

    def eigvals(self):
        return self.compute_eigvals(*self.parameters)

    @staticmethod
    def compute_decomposition(phi, wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.ControlledPhaseShift.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.ControlledPhaseShift.compute_decomposition(1.234, wires=(0,1))
        [PhaseShift(0.617, wires=[0]),
         CNOT(wires=[0, 1]),
         PhaseShift(-0.617, wires=[1]),
         CNOT(wires=[0, 1]),
         PhaseShift(0.617, wires=[1])]

        """
        return [
            qml.PhaseShift(phi / 2, wires=wires[0]),
            qml.CNOT(wires=wires),
            qml.PhaseShift(-phi / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            qml.PhaseShift(phi / 2, wires=wires[1]),
        ]


def _cphase_rz_resource():
    return {qml.RZ: 3, qml.CNOT: 2, qml.GlobalPhase: 1}


@register_resources(_cphase_rz_resource)
def _cphase_to_rz_cnot(phi: TensorLike, wires: WiresLike, **__):
    qml.RZ(phi / 2, wires=wires[0])
    qml.CNOT(wires=wires)
    qml.RZ(-phi / 2, wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RZ(phi / 2, wires=wires[1])
    qml.GlobalPhase(-phi / 4)


add_decomps(ControlledPhaseShift, _cphase_to_rz_cnot)
add_decomps("Adjoint(ControlledPhaseShift)", adjoint_rotation)
add_decomps("Pow(ControlledPhaseShift)", pow_rotation)

CPhase = ControlledPhaseShift
