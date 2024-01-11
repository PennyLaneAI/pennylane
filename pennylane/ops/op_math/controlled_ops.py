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
This submodule contains controlled operators based on the Controlled and ControlledOp class.
"""

import warnings
from typing import Iterable
from functools import lru_cache

import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires
from pennylane.ops.qubit.non_parametric_ops import PauliY, S, PauliZ, CCZ
from pennylane.ops.qubit.matrix_ops import QubitUnitary
from .controlled import ControlledOp


# pylint: disable=too-few-public-methods
class ControlledQubitUnitary(ControlledOp):
    r"""ControlledQubitUnitary(U, control_wires, wires, control_values)
    Apply an arbitrary fixed unitary to ``wires`` with control from the ``control_wires``.

    In addition to default ``Operation`` instance attributes, the following are
    available for ``ControlledQubitUnitary``:

    * ``control_wires``: wires that act as control for the operation
    * ``control_values``: the state on which to apply the controlled operation (see below)
    * ``target_wires``: the wires the unitary matrix will be applied to
    * ``active_wires``: Wires modified by the operator. This is the control wires followed by the target wires.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (2,)
    * Gradient recipe: None

    Args:
        base (Union[array[complex], QubitUnitary]): square unitary matrix or a QubitUnitary operation. If passing a matrix,
          this will be used to construct a QubitUnitary operator that will be used as the base operator. If providing
          a ``qml.QubitUnitary``, this will be used as the base directly.
        control_wires (Union[Wires, Sequence[int], or int]): the control wire(s)
        wires (Union[Wires, Sequence[int], or int]): the wire(s) the unitary acts on (optional if U is provided as a QubitUnitary)
        control_values (List[int, bool]): a list providing the state of the control qubits to control on (default is the all 1s state)
        unitary_check (bool): whether to check whether an array U is unitary when creating the operator (default False)

    **Example**

    The following shows how a single-qubit unitary can be applied to wire ``2`` with control on
    both wires ``0`` and ``1``:

    >>> U = np.array([[ 0.94877869,  0.31594146], [-0.31594146,  0.94877869]])
    >>> qml.ControlledQubitUnitary(U, control_wires=[0, 1], wires=2)
    Controlled(QubitUnitary(array([[ 0.94877869,  0.31594146],
       [-0.31594146,  0.94877869]]), wires=[2]), control_wires=[0, 1])

    Alternatively, the same operator can be constructed with a QubitUnitary:

    >>> base = qml.QubitUnitary(U, wires=2)
    >>> qml.ControlledQubitUnitary(base, control_wires=[0, 1])
    Controlled(QubitUnitary(array([[ 0.94877869,  0.31594146],
       [-0.31594146,  0.94877869]]), wires=[2]), control_wires=[0, 1])

    Typically controlled operations apply a desired gate if the control qubits
    are all in the state :math:`\vert 1\rangle`. However, there are some situations where
    it is necessary to apply a gate conditioned on all qubits being in the
    :math:`\vert 0\rangle` state, or a mix of the two.

    The state on which to control can be changed by passing a string of bits to
    `control_values`. For example, if we want to apply a single-qubit unitary to
    wire ``3`` conditioned on three wires where the first is in state ``0``, the
    second is in state ``1``, and the third in state ``1``, we can write:

    >>> qml.ControlledQubitUnitary(U, control_wires=[0, 1, 2], wires=3, control_values=[0, 1, 1])

    or

    >>> qml.ControlledQubitUnitary(U, control_wires=[0, 1, 2], wires=3, control_values=[False, True, True])
    """
    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (2,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(
            data[0], control_wires=metadata[0], control_values=metadata[1], work_wires=metadata[2]
        )

    # pylint: disable= too-many-arguments
    def __init__(
        self,
        base,
        control_wires,
        wires=None,
        control_values=None,
        unitary_check=False,
        work_wires=None,
    ):
        if getattr(base, "wires", False) and wires is not None:
            warnings.warn(
                "base operator already has wires; values specified through wires kwarg will be ignored."
            )

        if isinstance(base, Iterable):
            base = QubitUnitary(base, wires=wires, unitary_check=unitary_check)

        super().__init__(
            base,
            control_wires,
            control_values=control_values,
            work_wires=work_wires,
        )
        self._name = "ControlledQubitUnitary"

    def _controlled(self, wire):
        ctrl_wires = self.control_wires + wire
        values = None if self.control_values is None else self.control_values + [True]
        return ControlledQubitUnitary(
            self.base,
            control_wires=ctrl_wires,
            control_values=values,
            work_wires=self.work_wires,
        )

    @property
    def has_decomposition(self):
        if not super().has_decomposition:
            return False
        with qml.QueuingManager.stop_recording():
            # we know this is using try-except as logical control, but are favouring
            # certainty in it being correct over explicitness in an edge case.
            try:
                self.decomposition()
            except qml.operation.DecompositionUndefinedError:
                return False
        return True


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

    grad_method = None
    """Gradient computation method."""

    def _flatten(self):
        return tuple(), (self.wires,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(metadata[0])

    def __init__(self, wires, id=None):
        control_wire, wire = wires
        base = PauliY(wire)

        super().__init__(base, control_wire, id=id)
        self._name = "CY"

    @staticmethod
    @lru_cache()
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
        [CRY(3.141592653589793, wires=[0, 1]), S(wires=[0])]

        """
        return [qml.CRY(np.pi, wires=wires), S(wires=wires[0])]

    def decomposition(self):
        return self.compute_decomposition(self.wires)


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

    def _flatten(self):
        return tuple(), (self.wires,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(metadata[0])

    def __init__(self, wires):
        control_wire, wire = wires
        base = PauliZ(wires=wire)
        super().__init__(base, control_wire)
        self._name = "CZ"

    @staticmethod
    @lru_cache()
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
        return CCZ(wires=wire + self.wires)
