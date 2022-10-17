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
This module contains the meta operations i.e. they use Operation abstraction
but aren't actually operations in the theoretical sense
"""
from copy import copy

import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.wires import Wires


# pylint: disable=unused-argument
class Snapshot(Operation):
    r"""
    The Snapshot operation saves the internal simulator state at specific
    execution steps of a quantum function. As such, it is a pseudo operation
    with no effect on the quantum state.

    **Details:**

    * Number of wires: AllWires
    * Number of parameters: 0

    Args:
        tag (str or None): An optional custom tag for the snapshot, used to index it
                           in the snapshots dictionary.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=None)
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

    >>> qml.snapshots(circuit)()
    {0: array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]),
    'very_important_state': array([0.70710678+0.j, 0.+0.j, 0.70710678+0.j, 0.+0.j]),
    2: array([0.70710678+0.j, 0.+0.j, 0.+0.j, 0.70710678+0.j]),
    'execution_results': array(0.)}

    .. seealso:: :func:`~.snapshots`
    """
    num_wires = AnyWires
    num_params = 0
    grad_method = None

    def __init__(self, tag=None, do_queue=True):
        self.tag = tag
        super().__init__(wires=[], do_queue=do_queue)

    def label(self, decimals=None, base_label=None, cache=None):
        return "|S|"

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        return []

    def _controlled(self, _):
        return Snapshot(tag=self.tag)

    def adjoint(self):
        return Snapshot(tag=self.tag)


class Barrier(Operation):
    r"""Barrier(wires)
    The Barrier operator, used to separate the compilation process into blocks or as a visual tool.

    **Details:**

    * Number of wires: AnyWires
    * Number of parameters: 0

    Args:
        only_visual (bool): True if we do not want it to have an impact on the compilation process. Default is False.
        wires (Sequence[int] or int): the wires the operation acts on
    """
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    num_wires = AnyWires
    par_domain = None

    def __init__(self, wires=Wires([]), only_visual=False, do_queue=True, id=None):
        self.only_visual = only_visual
        self.hyperparameters["only_visual"] = only_visual
        super().__init__(wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(wires, only_visual=False):  # pylint: disable=unused-argument
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.Barrier.decomposition`.

        ``Barrier`` decomposes into an empty list for all arguments.

        Args:
            wires (Iterable, Wires): wires that the operator acts on
            only_visual (Bool): True if we do not want it to have an impact on the compilation process. Default is False.

        Returns:
            list: decomposition of the operator

        **Example:**

        >>> print(qml.Barrier.compute_decomposition(0))
        []

        """
        return []

    def label(self, decimals=None, base_label=None, cache=None):
        return "||"

    def _controlled(self, _):
        return copy(self).queue()

    def adjoint(self):
        return copy(self)

    def pow(self, z):
        return [copy(self)]

    def simplify(self):
        if self.only_visual:
            if len(self.wires) == 1:
                return qml.Identity(self.wires[0])
            return qml.prod(*(qml.Identity(w) for w in self.wires))
        return self


class WireCut(Operation):
    r"""WireCut(wires)
    The wire cut operation, used to manually mark locations for wire cuts.

    .. note::

        This operation is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    **Details:**

    * Number of wires: AnyWires
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wires the operation acts on
    """
    num_params = 0
    num_wires = AnyWires
    grad_method = None

    def __init__(self, *params, wires=None, do_queue=True, id=None):
        if wires == []:
            raise ValueError(
                f"{self.__class__.__name__}: wrong number of wires. "
                f"At least one wire has to be given."
            )
        super().__init__(*params, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(wires):  # pylint: disable=unused-argument
        r"""Representation of the operator as a product of other operators (static method).

        Since this operator is a placeholder inside a circuit, it decomposes into an empty list.

        Args:
            wires (Any, Wires): Wire that the operator acts on.

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> print(qml.WireCut.compute_decomposition(0))
        []

        """
        return []

    def label(self, decimals=None, base_label=None, cache=None):
        return "//"

    def adjoint(self):
        return WireCut(wires=self.wires)

    def pow(self, z):
        return [copy(self)]
