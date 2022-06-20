# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Contains the FlipSign template.
"""

import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import BasisState

class FlipSign(Operation):
    """Single sentence that summarizes the class.

    Multi-line description of the class (optional, if required).

    Args:
        arg1 (type): Description.
            Continuation line is indented if needed.
        arg2 (type): description

    Keyword Args:
        kwarg1 (type): description

    Attributes:
        attr1 (type): description

    Raises:
        ExceptionType: description

    .. seealso:: :func:`~.relevant_func`, :class:`~.RelevantClass` (optional)

    **Example**

    Minimal example with 1 or 2 code blocks (required).

    .. UsageDetails::

        More complicated use cases, options, and larger code blocks (optional).

    **Related tutorials**

    Links to any relevant PennyLane demos (optional).
    """

    num_wires = AnyWires

    def __init__(self, bin_arr, wires, do_queue=True, id=None):

        if not isinstance(bin_arr, list):
            raise ValueError(
                f"expected at integer binary array "
            )

        if np.array(bin_arr).dtype != np.dtype("int"):
            raise ValueError(
                f"expected at integer binary array "
            )

        if not isinstance(wires, int):
            raise ValueError(
                f"expected at integer value for wires "
            )

        if len(bin_arr) == 0:
            raise ValueError(
                f"expected at integer binary array not empty"
            )

        if wires <= 0:
            raise ValueError(
                f"expected at least one wire representing the qubit; "
                f"got {len(wires)}"
            )

        self._hyperparameters = {"bin_arr": bin_arr}
        super().__init__(wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def find_occur(value, arr):
        r"""Find index of occurrences for value in binary array.

        ..



        ..

        Args:

        Returns:
            list[.Operator]: decomposition of the operator
        """

        narray = np.array(arr)
        res_occur = list(np.where(narray == value)[0])
        return res_occur

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(wires, bin_arr):
        r"""



        .. note::



        .. seealso:: :meth:`~.Operator.decomposition`.

        Args:

        Returns:
            list[Operator]: decomposition of the operator
        """

        zeros_idx = find_occur(0, bin_arr)

        op_list = []

        for wire in zeros_idx[:-1]:
            op_list.append(PauliX(wire))
            op_list.append(ctrl(PauliZ, control = wires[:-1], control_values = bin_arr[:-1])(wires = wire))
            op_list.append(PauliX(wire))

        raise op_list
