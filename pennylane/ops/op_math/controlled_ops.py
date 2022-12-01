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


from typing import Iterable

import pennylane as qml

from pennylane.wires import Wires
from pennylane.operation import AnyWires
from pennylane.ops.qubit.matrix_ops import QubitUnitary
from .controlled_class import ControlledOp


class ControlledQubitUnitary(ControlledOp):
    r"""ControlledQubitUnitary(U, control_wires, wires, control_values)
    Apply an arbitrary fixed unitary to ``wires`` with control from the ``control_wires``.

    In addition to default ``Operation`` instance attributes, the following are
    available for ``ControlledQubitUnitary``:

    * ``control_wires``: wires that act as control for the operation
    * ``U``: unitary applied to the target wires

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (2,)
    * Gradient recipe: None

    Args:
        U (array[complex]): square unitary matrix
        control_wires (Union[Wires, Sequence[int], or int]): the control wire(s)
        wires (Union[Wires, Sequence[int], or int]): the wire(s) the unitary acts on
        control_values (str): a string of bits representing the state of the control
            qubits to control on (default is the all 1s state)

    **Example**

    The following shows how a single-qubit unitary can be applied to wire ``2`` with control on
    both wires ``0`` and ``1``:

    >>> U = np.array([[ 0.94877869,  0.31594146], [-0.31594146,  0.94877869]])
    >>> qml.ControlledQubitUnitary(U, control_wires=[0, 1], wires=2)

    Typically controlled operations apply a desired gate if the control qubits
    are all in the state :math:`\vert 1\rangle`. However, there are some situations where
    it is necessary to apply a gate conditioned on all qubits being in the
    :math:`\vert 0\rangle` state, or a mix of the two.

    The state on which to control can be changed by passing a string of bits to
    `control_values`. For example, if we want to apply a single-qubit unitary to
    wire ``3`` conditioned on three wires where the first is in state ``0``, the
    second is in state ``1``, and the third in state ``1``, we can write:

    >>> qml.ControlledQubitUnitary(U, control_wires=[0, 1, 2], wires=3, control_values='011')

    """
    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (2,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(
        self,
        base,
        control_wires=None,
        wires=None,
        control_values=None,
        do_queue=True,
        **kwargs,  # pylint: disable=unused-argument, too-many-arguments
    ):
        if control_wires is None:
            raise ValueError("Must specify control wires")

        if isinstance(base, Iterable):
            base = QubitUnitary(base, wires=wires)

        wires = Wires(base.wires)
        control_wires = Wires(control_wires)

        if Wires.shared_wires([wires, control_wires]):
            raise ValueError(
                "The control wires must be different from the wires specified to apply the unitary on."
            )

        self._hyperparameters = {
            "u_wires": wires,
            "control_wires": control_wires,
            "control_values": control_values,
        }

        super().__init__(base, control_wires, control_values=control_values, do_queue=do_queue)

        self._name = "ControlledQubitUnitary"

    @staticmethod
    def compute_matrix(
        U, control_wires, u_wires, control_values=None
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.ControlledQubitUnitary.matrix`

        Args:
            U (tensor_like): unitary matrix
            control_wires (Iterable): the control wire(s)
            u_wires (Iterable): the wire(s) the unitary acts on
            control_values (str or None): a string of bits representing the state of the control
                qubits to control on (default is the all 1s state)

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> U = np.array([[ 0.94877869,  0.31594146], [-0.31594146,  0.94877869]])
        >>> qml.ControlledQubitUnitary.compute_matrix(U, control_wires=[1], u_wires=[0], control_values="1")
        [[ 1.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j]
         [ 0.        +0.j  1.        +0.j  0.        +0.j  0.        +0.j]
         [ 0.        +0.j  0.        +0.j  0.94877869+0.j  0.31594146+0.j]
         [ 0.        +0.j  0.        +0.j -0.31594146+0.j  0.94877869+0.j]]
        """
        target_dim = 2 ** len(u_wires)
        shape = qml.math.shape(U)
        if not (len(shape) in {2, 3} and shape[-2:] == (target_dim, target_dim)):
            raise ValueError(
                f"Input unitary must be of shape {(target_dim, target_dim)} or "
                f"(batch_size, {target_dim}, {target_dim})."
            )

        # A multi-controlled operation is a block-diagonal matrix partitioned into
        # blocks where the operation being applied sits in the block positioned at
        # the integer value of the control string. For example, controlling a
        # unitary U with 2 qubits will produce matrices with block structure
        # (U, I, I, I) if the control is on bits '00', (I, U, I, I) if on bits '01',
        # etc. The positioning of the block is controlled by padding the block diagonal
        # to the left and right with the correct amount of identity blocks.

        total_wires = qml.wires.Wires(control_wires) + qml.wires.Wires(u_wires)

        # if control values unspecified, we control on the all-ones string
        if not control_values:
            control_values = "1" * len(control_wires)

        if isinstance(control_values, str):
            if len(control_values) != len(control_wires):
                raise ValueError("Length of control bit string must equal number of control wires.")

            # Make sure all values are either 0 or 1
            if not set(control_values).issubset({"0", "1"}):
                raise ValueError("String of control values can contain only '0' or '1'.")

            control_int = int(control_values, 2)
        else:
            raise ValueError("Alternative control values must be passed as a binary string.")

        padding_left = control_int * target_dim
        padding_right = 2 ** len(total_wires) - target_dim - padding_left

        interface = qml.math.get_interface(U)
        left_pad = qml.math.cast_like(qml.math.eye(padding_left, like=interface), 1j)
        right_pad = qml.math.cast_like(qml.math.eye(padding_right, like=interface), 1j)
        if len(qml.math.shape(U)) == 3:
            return qml.math.stack([qml.math.block_diag([left_pad, _U, right_pad]) for _U in U])
        return qml.math.block_diag([left_pad, U, right_pad])

    def pow(self, z):
        base_pow = self.base.pow(z)
        return [
            ControlledQubitUnitary(
                op, control_wires=self.control_wires, control_values=self.control_values
            )
            for op in base_pow
        ]

    def _controlled(self, wire):
        ctrl_wires = self.control_wires + wire
        values = None if self.control_values is None else self.control_values + [True]
        new_op = ControlledQubitUnitary(
            *self.parameters,
            control_wires=ctrl_wires,
            wires=self.hyperparameters["u_wires"],
            control_values=values,
        )
        return new_op.inv() if self.inverse else new_op
