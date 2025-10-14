# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains the SemiAdder template for performing the semi-out-place addition."""

from pennylane.decomposition import add_decomps, adjoint_resource_rep, register_resources
from pennylane.operation import Operation
from pennylane.ops import CNOT, adjoint
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.wires import Wires, WiresLike

from .temporary_and import TemporaryAND


def _left_operator(wires, ik_is_zero=False):
    """Implement the left block in figure 2, https://arxiv.org/pdf/1709.06648"""

    if not ik_is_zero:
        ck, ik, tk, aux = wires
        return [
            CNOT([ck, ik]),
            CNOT([ck, tk]),
            TemporaryAND([ik, tk, aux]),
            CNOT([ck, aux]),
        ]

    ck, tk, aux = wires

    return [CNOT([ck, tk]), TemporaryAND([ck, tk, aux]), CNOT([ck, aux])]


def _right_operator(wires, ik_is_zero=False):
    """Implement the right block in figure 2, https://arxiv.org/pdf/1709.06648"""

    if not ik_is_zero:
        ck, ik, tk, aux = wires
        return [
            CNOT([ck, aux]),
            adjoint(TemporaryAND([ik, tk, aux])),
            CNOT([ck, ik]),
            CNOT([ik, tk]),
        ]

    ck, tk, aux = wires

    return [CNOT([ck, aux]), adjoint(TemporaryAND([ck, tk, aux]))]


class SemiAdder(Operation):
    r"""This operator performs the plain addition of two integers :math:`x` and :math:`y` in the computational basis:

    .. math::

        \text{SemiAdder} |x \rangle | y \rangle = |x \rangle | x + y  \rangle,

    This operation is also referred to as semi-out-place addition or quantum-quantum in-place addition in the literature.

    The implementation is based on `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_.

    Args:
        x_wires (Sequence[int]): The wires that store the integer :math:`x`. The number of wires must be sufficient to
            represent :math:`x` in binary.
        y_wires (Sequence[int]): The wires that store the integer :math:`y`. The number of wires must be sufficient to
            represent :math:`y` in binary. These wires are also used
            to encode the integer :math:`x+y` which is computed modulo :math:`2^{\text{len(y_wires)}}` in the computational basis.
        work_wires (Optional(Sequence[int])): The auxiliary wires to use for the addition. At least, ``len(y_wires) - 1`` work
            wires should be provided.

    **Example**

    This example computes the sum of two integers :math:`x=3` and :math:`y=4`.

    .. code-block:: python

        x = 3
        y = 4

        wires = qml.registers({"x":3, "y":6, "work":5})

        dev = qml.device("default.qubit")

        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires=wires["x"])
            qml.BasisEmbedding(y, wires=wires["y"])
            qml.SemiAdder(wires["x"], wires["y"], wires["work"])
            return qml.sample(wires=wires["y"])

    .. code-block:: pycon

        >>> print(circuit())
        [[0 0 0 1 1 1]]

    The result :math:`[[0 0 0 1 1 1]]`, is the binary representation of :math:`3 + 4 = 7`.

    Note that the result is computed modulo :math:`2^{\text{len(y_wires)}}` which makes the computed value dependent on the size of the ``y_wires`` register. This behavior is demonstrated in the following example.

    .. code-block:: python

        x = 3
        y = 1

        wires = qml.registers({"x":3, "y":2, "work":1})

        dev = qml.device("default.qubit")

        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires=wires["x"])
            qml.BasisEmbedding(y, wires=wires["y"])
            qml.SemiAdder(wires["x"], wires["y"], wires["work"])
            return qml.sample(wires=wires["y"])

    >>> print(circuit())
    [[0 0]]

    The result :math:`[0\ 0]` is the binary representation of :math:`3 + 1 = 4` where :math:`4 \mod 2^2 = 0`.
    """

    grad_method = None

    resource_keys = {"num_y_wires"}

    def __init__(
        self, x_wires: WiresLike, y_wires: WiresLike, work_wires: WiresLike | None, id=None
    ):

        x_wires = Wires(x_wires)
        y_wires = Wires(y_wires)
        work_wires = Wires(work_wires if work_wires is not None else [])

        if work_wires:
            if len(work_wires) < len(y_wires) - 1:
                raise ValueError(f"At least {len(y_wires)-1} work_wires should be provided.")
            if work_wires.intersection(x_wires):
                raise ValueError("None of the wires in work_wires should be included in x_wires.")
            if work_wires.intersection(y_wires):
                raise ValueError("None of the wires in work_wires should be included in y_wires.")
        if x_wires.intersection(y_wires):
            raise ValueError("None of the wires in y_wires should be included in x_wires.")

        self.hyperparameters["x_wires"] = x_wires
        self.hyperparameters["y_wires"] = y_wires
        self.hyperparameters["work_wires"] = work_wires

        if work_wires:
            all_wires = Wires.all_wires([x_wires, y_wires, work_wires])
        else:
            all_wires = Wires.all_wires([x_wires, y_wires])

        super().__init__(wires=all_wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "num_y_wires": len(self.hyperparameters["y_wires"]),
        }

    @property
    def num_params(self):
        return 0

    def _flatten(self):
        metadata = tuple((key, value) for key, value in self.hyperparameters.items())
        return tuple(), metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(**hyperparams_dict)

    def map_wires(self, wire_map: dict) -> "SemiAdder":
        new_dict = {
            key: [wire_map.get(w, w) for w in self.hyperparameters[key]]
            for key in ["x_wires", "y_wires", "work_wires"]
        }

        return SemiAdder(
            new_dict["x_wires"],
            new_dict["y_wires"],
            new_dict["work_wires"],
        )

    def decomposition(self):
        r"""Representation of the operator as a product of other operators."""
        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(x_wires, y_wires, work_wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.
        The implementation is based on `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_.

        Args:

            x_wires (Sequence[int]): The wires that store the integer :math:`x`. The number of wires must be sufficient to
                represent :math:`x` in binary.
            y_wires (Sequence[int]): The wires that store the integer :math:`y`. The number of wires must be sufficient to
                represent :math:`y` in binary. These wires are also used
                to encode the integer :math:`x+y` which is computed modulo :math:`2^{\text{len(y_wires)}}` in the computational basis.
            work_wires (Sequence[int]): The auxiliary wires to use for the addition. At least, ``len(y_wires) - 1`` work
                wires should be provided.

        Returns:
            list[.Operator]: Decomposition of the operator
        """

        with AnnotatedQueue() as q:
            _semiadder(x_wires, y_wires, work_wires)

        if QueuingManager.recording():
            for op in q.queue:
                apply(op)

        return q.queue


def _semiadder_resources(num_y_wires):
    # Resources extracted from `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_.
    # In the case where len(x_wires) < len(y_wires), this is an upper bound.
    return {
        TemporaryAND: num_y_wires - 1,
        adjoint_resource_rep(TemporaryAND, {}): num_y_wires - 1,
        CNOT: 6 * (num_y_wires - 2) + 3,
    }


@register_resources(_semiadder_resources, exact=False)
def _semiadder(x_wires, y_wires, work_wires, **_):

    num_y_wires = len(y_wires)
    num_x_wires = len(x_wires)

    if num_y_wires == 1:
        CNOT([x_wires[-1], y_wires[0]])
        return

    x_wires_pl = x_wires[::-1][:num_y_wires]
    y_wires_pl = y_wires[::-1]
    work_wires_pl = work_wires[::-1]
    TemporaryAND([x_wires_pl[0], y_wires_pl[0], work_wires_pl[0]])

    for i in range(1, num_y_wires - 1):
        if i < num_x_wires:
            _left_operator([work_wires_pl[i - 1], x_wires_pl[i], y_wires_pl[i], work_wires_pl[i]])
        else:
            _left_operator([work_wires_pl[i - 1], y_wires_pl[i], work_wires_pl[i]], ik_is_zero=True)

    CNOT([work_wires_pl[-1], y_wires_pl[-1]])

    if num_x_wires >= num_y_wires:
        CNOT([x_wires_pl[-1], y_wires_pl[-1]])

    for i in range(len(y_wires_pl) - 2, 0, -1):
        if i < num_x_wires:
            _right_operator([work_wires_pl[i - 1], x_wires_pl[i], y_wires_pl[i], work_wires_pl[i]])
        else:
            _right_operator(
                [work_wires_pl[i - 1], y_wires_pl[i], work_wires_pl[i]], ik_is_zero=True
            )

    adjoint(TemporaryAND([x_wires_pl[0], y_wires_pl[0], work_wires_pl[0]]))
    CNOT([x_wires_pl[0], y_wires_pl[0]])


add_decomps(SemiAdder, _semiadder)
