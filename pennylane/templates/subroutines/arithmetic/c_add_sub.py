# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains the CAddSub template for performing out of place addition or subtraction,
controlled by a control qubit."""

from pennylane.decomposition import (
    add_decomps,
    controlled_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation
from pennylane.ops import BasisState, ctrl
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.wires import Wires, WiresLike

from .semi_adder import SemiAdder


class CAddSub(Operation):
    r"""This operator performs modular addition or subtraction of two integers :math:`x` and
    :math:`y`, with the decision controlled by a control qubit:

    .. math::

        \text{CAddSub} |0\rangle |x \rangle | y \rangle = |0\rangle |x \rangle | y - x \!\mod\! N \rangle,\\
        \text{CAddSub} |1\rangle |x \rangle | y \rangle = |1\rangle |x \rangle | y + x \!\mod\! N \rangle.

    Here, :math:`N` is the modulus of the arithmetic operation, given by the size of the
    input register that holds :math:`y`.

    Args:
        control_wire (WiresLike): The wire controlling between addition (:math:`|1\rangle`) and subtraction (:math:`|0\rangle`).
        x_wires (WiresLike): The wires that store the integer :math:`x`.
        y_wires (WiresLike): The wires that store the integer :math:`y` as well as the
            output of the operation, which is computed modulo :math:`N=2^{n}` where :math:`n`
            is the length of ``y_wires``.
        work_wires (WiresLike): The auxiliary wires to use for the operation.
            At least ``len(y_wires) - 1`` work wires should be provided.

    **Example**

    This example computes the sum and difference of two integers :math:`x=5` and :math:`y=13` in
    superposition:

    .. code-block:: python

        import pennylane as qml
        x = 5
        y = 13

        wires = qml.registers({"control": 1, "x": 3, "y": 4, "work": 3})

        dev = qml.device("default.qubit", seed=195)

        @qml.set_shots(100)
        @qml.qnode(dev)
        def circuit():
            qml.H(wires["control"])
            qml.BasisEmbedding(x, wires=wires["x"])
            qml.BasisEmbedding(y, wires=wires["y"])
            qml.CAddSub(wires["control"], wires["x"], wires["y"], wires["work"])
            return qml.counts(wires=wires["y"])

    .. code-block:: pycon

        >>> output = circuit()
        >>> print({int(key, 2): count for key, count in output.items()})
        {2: np.int64(49), 8: np.int64(51)}

    As we can see, we compute :math:`(x+y)\mod 2^4=2` and :math:`(y-x)\mod 2^4=8` about half of
    the time each.
    """

    grad_method = None

    resource_keys = {"num_y_wires"}

    def __init__(
        self,
        control_wire: WiresLike,
        x_wires: WiresLike,
        y_wires: WiresLike,
        work_wires: WiresLike | None,
    ):

        control_wire = Wires(control_wire)
        x_wires = Wires(x_wires)
        y_wires = Wires(y_wires)
        work_wires = Wires(work_wires if work_wires is not None else [])

        if work_wires:
            if len(work_wires) < len(y_wires) - 1:
                raise ValueError(f"At least {len(y_wires)-1} work_wires should be provided.")
            if work_wires.intersection(control_wire):
                raise ValueError("None of the wires in work_wires should be the control wire.")
            if work_wires.intersection(x_wires):
                raise ValueError("None of the wires in work_wires should be included in x_wires.")
            if work_wires.intersection(y_wires):
                raise ValueError("None of the wires in work_wires should be included in y_wires.")
        if x_wires.intersection(control_wire):
            raise ValueError("None of the wires in x_wires should be the control wire.")
        if x_wires.intersection(y_wires):
            raise ValueError("None of the wires in y_wires should be included in x_wires.")
        if y_wires.intersection(control_wire):
            raise ValueError("None of the wires in y_wires should be the control wire.")

        self.hyperparameters["control_wire"] = control_wire
        self.hyperparameters["x_wires"] = x_wires
        self.hyperparameters["y_wires"] = y_wires
        self.hyperparameters["work_wires"] = work_wires

        all_wires = [control_wire, x_wires, y_wires]
        if work_wires:
            all_wires.append(work_wires)
        all_wires = Wires.all_wires(all_wires)
        super().__init__(wires=all_wires)

    @property
    def resource_params(self) -> dict:
        return {"num_y_wires": len(self.hyperparameters["y_wires"])}

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

    def map_wires(self, wire_map: dict) -> "CAddSub":
        new_dict = {
            key: [wire_map.get(w, w) for w in self.hyperparameters[key]]
            for key in ["control_wire", "x_wires", "y_wires", "work_wires"]
        }

        return CAddSub(**new_dict)

    def decomposition(self):
        r"""Representation of the operator as a product of other operators."""
        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        control_wire, x_wires, y_wires, work_wires
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        Args:
            control_wire (WiresLike): The wire controlling between addition (:math:`|1\rangle`)
                and subtraction (:math:`|0\rangle`).
            x_wires (WiresLike): The wires that store the integer :math:`x`.
            y_wires (WiresLike): The wires that store the integer :math:`y` and the resulting
                integer :math:`x+y` or :math:`y-x` after the computation, which is computed modulo
                :math:`2^{\text{len(y_wires)}}`.
            work_wires (WiresLike): The auxiliary wires to use for the addition.
                At least ``len(y_wires) - 1`` work wires should be provided.

        Returns:
            list[.Operator]: Decomposition of the operator
        """

        with AnnotatedQueue() as q:
            _c_add_sub(control_wire, x_wires, y_wires, work_wires)

        if QueuingManager.recording():
            for op in q.queue:
                apply(op)

        return q.queue


def _c_add_sub_resources(num_y_wires):
    return {
        controlled_resource_rep(
            BasisState,
            base_params={"num_wires": num_y_wires},
            num_control_wires=1,
            num_zero_control_values=1,
        ): 2,
        resource_rep(SemiAdder, num_y_wires=num_y_wires): 1,
    }


@register_resources(_c_add_sub_resources, exact=True)
def _c_add_sub(control_wire, x_wires, y_wires, work_wires, **_):
    ctrl(BasisState([1] * len(y_wires), y_wires), control=control_wire, control_values=[0])
    SemiAdder(x_wires, y_wires, work_wires)
    ctrl(BasisState([1] * len(y_wires), y_wires), control=control_wire, control_values=[0])


add_decomps(CAddSub, _c_add_sub)
