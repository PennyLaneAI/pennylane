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
        if getattr(base, "wires", False) and wires is not None:
            warnings.warn(
                "base operator already has wires; values specified through wires kwarg will be ignored."
            )

        if isinstance(base, Iterable):
            base = QubitUnitary(base, wires=wires)

        super().__init__(base, control_wires, control_values=control_values, do_queue=do_queue)
        self.hyperparameters["u_wires"] = Wires(base.wires)

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
