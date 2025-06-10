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

import pennylane as qml
from pennylane.decomposition import add_decomps, register_resources
from pennylane.operation import Operation
from pennylane.wires import WiresLike


def _left_operator(ck, ik, tk, aux):
    """Implement the left block in figure 2, https://arxiv.org/pdf/1709.06648"""
    op_list = []
    op_list.append(qml.CNOT([ck, ik]))
    op_list.append(qml.CNOT([ck, tk]))
    op_list.append(qml.TemporaryAND([ik, tk, aux]))
    op_list.append(qml.CNOT([ck, aux]))
    return op_list


def _right_operator(ck, ik, tk, aux):
    """Implement the right block in figure 2, https://arxiv.org/pdf/1709.06648"""
    op_list = []
    op_list.append(qml.CNOT([ck, aux]))
    op_list.append(qml.adjoint(qml.TemporaryAND([ik, tk, aux])))
    op_list.append(qml.CNOT([ck, ik]))
    op_list.append(qml.CNOT([ik, tk]))
    return op_list


class SemiAdder(Operation):
    r"""Performs the semi-out-place addition operation.
    More specifically, the operation is an in-place quantum-quantum addition.

    This operator performs the plain addition of two integers :math:`x` and :math:`y` in the computational basis:

    .. math::

        \text{SemiAdder} |x \rangle | y \rangle = |x \rangle | x + y  \rangle,

    The implementation is based on `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_.

    Args:
        x_wires (Sequence[int]): the wires that store the integer :math:`x`
        y_wires (Sequence[int]): the wires that store the integer :math:`y`
        work_wires (Sequence[int]): the auxiliary wires to use for the addition. ``len(y_wires) - 1`` work
            wires should be provided.

    **Example**

    This example computes the sum of two integers :math:`x=5` and :math:`y=6`.

    .. code-block::

        x = 3
        y = 4

        wires = qml.registers({"x":3, "y":6, "work":5})

        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires=wires["x"])
            qml.BasisEmbedding(y, wires=wires["y"])
            qml.SemiAdder(wires["x"], wires["y"], wires["work"])
            return qml.sample(wires=wires["y"])

    .. code-block:: pycon

        >>> print(circuit())
        [0 0 0 1 1 1]

    The result :math:`[0 0 0 1 1 1]`, is the binary representation of :math:`3 + 4 = 7`.

    .. details::
        :title: Usage Details

        This template takes three different sets of wires as input:

        The first one is ``x_wires`` which is used
        to encode the integer :math:`x` in the computational basis. Therefore, ``x_wires`` must contain
        at least :math:`\lceil \log_2(x)\rceil` to represent :math:`x`.

        The second one is ``y_wires`` which is used
        to encode the integer :math:`y` in the computational basis. Therefore, ``y_wires`` must contain
        at least :math:`\lceil \log_2(y)\rceil` wires to represent :math:`y`. ``y_wires`` is also used
        to encode the integer :math:`x+y` in the computational basis.

        The fourth set of wires is ``work_wires`` which consists of :math:`m-1` auxiliary qubits used to perform the addition operation,
        where :math:`m` is the number of ``y_wires``.
    """

    grad_method = None

    resource_keys = {"num_y_wires"}

    def __init__(
        self,
        x_wires: WiresLike,
        y_wires: WiresLike,
        work_wires,
        id=None,
    ):  # pylint: disable=too-many-arguments

        x_wires = qml.wires.Wires(x_wires)
        y_wires = qml.wires.Wires(y_wires)
        work_wires = qml.wires.Wires(work_wires)

        if len(work_wires) < len(y_wires) - 1:
            raise ValueError(f"At least {len(y_wires)-1} work_wires should be provided.")
        if not set(work_wires).isdisjoint(x_wires):
            raise ValueError("None of the wires in work_wires should be included in x_wires.")
        if not set(work_wires).isdisjoint(y_wires):
            raise ValueError("None of the wires in work_wires should be included in y_wires.")
        if not set(x_wires).isdisjoint(y_wires):
            raise ValueError("None of the wires in y_wires should be included in x_wires.")

        for key in ["x_wires", "y_wires", "work_wires"]:
            self.hyperparameters[key] = qml.wires.Wires(locals()[key])

        # pylint: disable=consider-using-generator
        all_wires = sum(
            [self.hyperparameters[key] for key in ["x_wires", "y_wires", "work_wires"]], start=[]
        )

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

    def map_wires(self, wire_map: dict):
        new_dict = {
            key: [wire_map.get(w, w) for w in self.hyperparameters[key]]
            for key in ["x_wires", "y_wires", "work_wires"]
        }

        return SemiAdder(
            new_dict["x_wires"],
            new_dict["y_wires"],
            new_dict["work_wires"],
        )

    def decomposition(self):  # pylint: disable=arguments-differ

        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(x_wires, y_wires, work_wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        Args:
            x_wires (Sequence[int]): the wires that store the integer :math:`x`
            y_wires (Sequence[int]): the wires that store the integer :math:`y`
            work_wires (Sequence[int]): the auxiliary wires to use for the addition. ``len(y_wires) - 1`` work
                wires should be provided.

        Returns:
            list[.Operator]: Decomposition of the operator
        """
        op_list = []

        # revert wires to follow PennyLane convention

        x_wires_pl = x_wires[::-1][: len(y_wires)]
        y_wires_pl = y_wires[::-1]
        work_wires_pl = work_wires[::-1]
        op_list.append(qml.TemporaryAND([x_wires_pl[0], y_wires_pl[0], work_wires_pl[0]]))

        for i in range(1, len(y_wires_pl) - 1):
            if i < len(x_wires_pl):
                op_list += _left_operator(
                    work_wires_pl[i - 1], x_wires_pl[i], y_wires_pl[i], work_wires_pl[i]
                )
            else:
                # If the number of qubits in |x> is smaller than |y>, we can conceptually complete the |x> state
                # with |0> on the left making sure they are of the same size. Assuming this, we can simplify the left operator.
                op_list.append(qml.CNOT([work_wires_pl[i - 1], y_wires_pl[i]]))
                op_list.append(
                    qml.TemporaryAND([work_wires_pl[i - 1], y_wires_pl[i], work_wires_pl[i]])
                )
                op_list.append(qml.CNOT([work_wires_pl[i - 1], work_wires_pl[i]]))

        op_list.append(qml.CNOT([work_wires_pl[-1], y_wires_pl[-1]]))

        if len(x_wires_pl) >= len(y_wires_pl):
            op_list.append(qml.CNOT([x_wires_pl[-1], y_wires_pl[-1]]))

        for i in range(len(y_wires_pl) - 2, 0, -1):
            if i < len(x_wires_pl):
                op_list += _right_operator(
                    work_wires_pl[i - 1], x_wires_pl[i], y_wires_pl[i], work_wires_pl[i]
                )
            else:
                # If the number of qubits in |x> is smaller than |y>, we can conceptually complete the |x> state
                # with |0> on the left making sure they are of the same size. Assuming this, we can simplify the right operator.
                op_list.append(qml.CNOT([work_wires_pl[i - 1], work_wires_pl[i]]))
                op_list.append(
                    qml.adjoint(
                        qml.TemporaryAND([work_wires_pl[i - 1], y_wires_pl[i], work_wires_pl[i]])
                    )
                )

        op_list.append(
            qml.adjoint(qml.TemporaryAND([x_wires_pl[0], y_wires_pl[0], work_wires_pl[0]]))
        )

        op_list.append(qml.CNOT([x_wires_pl[0], y_wires_pl[0]]))

        return op_list


def _semiadder_resources(num_y_wires):
    # In the case where len(x_wires) < len(y_wires), this is an upper bound.
    return {
        qml.TemporaryAND: num_y_wires - 1,
        qml.decomposition.adjoint_resource_rep(qml.TemporaryAND, {}): num_y_wires - 1,
        qml.CNOT: 6 * (num_y_wires - 2) + 3,
    }


@register_resources(_semiadder_resources)
def _semiadder(x_wires, y_wires, work_wires, **_):
    x_wires_pl = x_wires[::-1][: len(y_wires)]
    y_wires_pl = y_wires[::-1]
    work_wires_pl = work_wires[::-1]
    qml.TemporaryAND([x_wires_pl[0], y_wires_pl[0], work_wires_pl[0]])

    for i in range(1, len(y_wires_pl) - 1):
        if i < len(x_wires_pl):
            _left_operator(work_wires_pl[i - 1], x_wires_pl[i], y_wires_pl[i], work_wires_pl[i])
        else:
            qml.CNOT([work_wires_pl[i - 1], y_wires_pl[i]])
            qml.TemporaryAND([work_wires_pl[i - 1], y_wires_pl[i], work_wires_pl[i]])
            qml.CNOT([work_wires_pl[i - 1], work_wires_pl[i]])

    qml.CNOT([work_wires_pl[-1], y_wires_pl[-1]])

    if len(x_wires_pl) >= len(y_wires_pl):
        qml.CNOT([x_wires_pl[-1], y_wires_pl[-1]])

    for i in range(len(y_wires_pl) - 2, 0, -1):
        if i < len(x_wires_pl):
            _right_operator(work_wires_pl[i - 1], x_wires_pl[i], y_wires_pl[i], work_wires_pl[i])
        else:
            qml.CNOT([work_wires_pl[i - 1], work_wires_pl[i]])
            qml.adjoint(qml.TemporaryAND([work_wires_pl[i - 1], y_wires_pl[i], work_wires_pl[i]]))

    qml.adjoint(qml.TemporaryAND([x_wires_pl[0], y_wires_pl[0], work_wires_pl[0]]))

    qml.CNOT([x_wires_pl[0], y_wires_pl[0]])


add_decomps(SemiAdder, _semiadder)
