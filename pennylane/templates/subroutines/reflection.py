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
This submodule contains the template for the Reflection operation.
"""
import copy

import numpy as np

from pennylane import ops
from pennylane.decomposition import (
    add_decomps,
    adjoint_resource_rep,
    controlled_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires


class Reflection(Operation):
    r"""Apply a reflection about a state :math:`|\Psi\rangle`.

    This operator works by providing an operation, :math:`U`, that prepares the desired state, :math:`\vert \Psi \rangle`,
    that we want to reflect about. We can also provide a reflection angle :math:`\alpha`
    to define the operation in a more generic form:

    .. math::

       R(U, \alpha) = -I + (1 - e^{i\alpha}) |\Psi\rangle \langle \Psi|

    This operator is an important component of quantum algorithms such as amplitude amplification [`arXiv:quant-ph/0005055 <https://arxiv.org/abs/quant-ph/0005055>`__]
    and oblivious amplitude amplification [`arXiv:1312.1414 <https://arxiv.org/abs/1312.1414>`__].

    Args:
        U (Operator): the operator that prepares the state :math:`|\Psi\rangle`
        alpha (float): the angle of the operator, default is :math:`\pi`
        reflection_wires (Any or Iterable[Any]): subsystem of wires on which to reflect, the
            default is ``None`` and the reflection will be applied on the ``U`` wires.

    **Example**

    This example shows how to apply the reflection :math:`-I + 2|+\rangle \langle +|` to the state :math:`|1\rangle`.

    .. code-block:: python

        U = qml.Hadamard(wires=0)
        dev = qml.device('default.qubit')

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            qml.Reflection(U)
            return qml.state()

    >>> circuit() # doctest: +SKIP
    array([1.+6.123234e-17j, 0.-6.123234e-17j])

    For cases when :math:`U` comprises many operations, you can create a quantum
    function containing each operation, one per line, then decorate the quantum
    function with ``@qml.prod``:

    .. code-block:: python

        @qml.prod
        def U(wires):
            qml.Hadamard(wires=wires[0])
            qml.RY(0.1, wires=wires[1])

        @qml.qnode(dev)
        def circuit():
            qml.Reflection(U([0, 1]))
            return qml.state()

    >>> circuit() # doctest: +SKIP
    array([-0.0025-6.1385e-17j,  0.0499+3.0565e-18j,  0.9975+6.1079e-17j,
            0.0499+3.0565e-18j])

    .. details::
        :title: Theory

        The operator is built as follows:

        .. math::

            \text{R}(U, \alpha) = -I + (1 - e^{i\alpha}) |\Psi\rangle \langle \Psi| = U(-I + (1 - e^{i\alpha}) |0\rangle \langle 0|)U^{\dagger}.

        The central block is obtained through a :class:`~.PhaseShift` controlled operator.

        In the case of specifying the reflection wires, the operator would have the following expression.

        .. math::

            U(-I + (1 - e^{i\alpha}) |0\rangle^{\otimes m} \langle 0|^{\otimes m}\otimes I^{n-m})U^{\dagger},

        where :math:`m` is the number of reflection wires and :math:`n` is the total number of wires.

    """

    grad_method = None

    resource_keys = {"base_class", "base_params", "num_wires", "num_reflection_wires"}

    def _flatten(self):
        data = (self.hyperparameters["base"], self.parameters[0])
        return data, (self.hyperparameters["reflection_wires"],)

    # pylint: disable=arguments-differ
    @classmethod
    def _primitive_bind_call(cls, U, alpha, reflection_wires, **kwargs):
        return super()._primitive_bind_call(U, alpha, wires=reflection_wires, **kwargs)

    @classmethod
    def _unflatten(cls, data, metadata):
        U, alpha = data
        return cls(U, alpha=alpha, reflection_wires=metadata[0])

    def __init__(self, U, alpha=np.pi, reflection_wires=None, id=None):
        self._name = "Reflection"
        wires = U.wires

        if reflection_wires is None:
            reflection_wires = U.wires

        if not set(reflection_wires).issubset(set(U.wires)):
            raise ValueError("The reflection wires must be a subset of the operation wires.")

        self._hyperparameters = {
            "base": U,
            "reflection_wires": tuple(reflection_wires),
        }

        super().__init__(alpha, *U.data, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "base_class": self.hyperparameters["base"].__class__,
            "base_params": self.hyperparameters["base"].resource_params,
            "num_wires": len(self.wires),
            "num_reflection_wires": len(self.hyperparameters["reflection_wires"]),
        }

    def map_wires(self, wire_map: dict):
        # pylint: disable=protected-access
        new_op = copy.deepcopy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        new_op._hyperparameters["base"] = new_op._hyperparameters["base"].map_wires(wire_map)
        new_op._hyperparameters["reflection_wires"] = tuple(
            wire_map.get(w, w) for w in new_op._hyperparameters["reflection_wires"]
        )

        return new_op

    @property
    def alpha(self):
        """The alpha angle for the operation."""
        return self.parameters[0]

    @property
    def reflection_wires(self):
        """The reflection wires for the operation."""
        return self.hyperparameters["reflection_wires"]

    def queue(self, context=QueuingManager):
        context.remove(self.hyperparameters["base"])
        context.append(self)
        return self

    @staticmethod
    def compute_decomposition(*parameters, wires=None, **hyperparameters):
        alpha = parameters[0]
        U = hyperparameters["base"]
        reflection_wires = hyperparameters["reflection_wires"]

        wires = Wires(reflection_wires) if reflection_wires is not None else wires

        decomp_ops = []

        decomp_ops.append(ops.GlobalPhase(np.pi))
        decomp_ops.append(ops.adjoint(U))

        if len(wires) > 1:
            decomp_ops.append(ops.X(wires=wires[-1]))
            decomp_ops.append(
                ops.ctrl(
                    ops.PhaseShift(alpha, wires=wires[-1]),
                    control=wires[:-1],
                    control_values=[0] * (len(wires) - 1),
                )
            )
            decomp_ops.append(ops.X(wires=wires[-1]))

        else:
            decomp_ops.append(ops.X(wires=wires))
            decomp_ops.append(ops.PhaseShift(alpha, wires=wires))
            decomp_ops.append(ops.X(wires=wires))

        decomp_ops.append(U)

        return decomp_ops


def _reflection_decomposition_resources(
    base_class, base_params, num_wires, num_reflection_wires=None
) -> dict:

    num_wires = num_reflection_wires if num_reflection_wires is not None else num_wires

    resources = {
        ops.GlobalPhase: 1,
        adjoint_resource_rep(base_class, base_params): 1,
        ops.PauliX: 2,
    }

    if num_wires > 1:
        resources[
            controlled_resource_rep(
                ops.PhaseShift,
                {},
                num_control_wires=num_wires - 1,
                num_zero_control_values=num_wires - 1,
            )
        ] = 1
    else:
        resources[resource_rep(ops.PhaseShift)] = 1

    resources[resource_rep(base_class, **base_params)] = 1

    return resources


@register_resources(_reflection_decomposition_resources)
def _reflection_decomposition(*parameters, wires=None, **hyperparameters):
    alpha = parameters[0]
    U = hyperparameters["base"]
    reflection_wires = hyperparameters["reflection_wires"]

    wires = Wires(reflection_wires) if reflection_wires is not None else wires

    ops.GlobalPhase(np.pi)
    ops.adjoint(U)

    if len(wires) > 1:
        ops.PauliX(wires=wires[-1])

        ops.ctrl(
            ops.PhaseShift(alpha, wires=wires[-1]),
            control=wires[:-1],
            control_values=[0] * (len(wires) - 1),
        )

        ops.PauliX(wires=wires[-1])

    else:
        ops.PauliX(wires=wires)
        ops.PhaseShift(alpha, wires=wires)
        ops.PauliX(wires=wires)

    U._unflatten(*U._flatten())  # pylint: disable=protected-access


add_decomps(Reflection, _reflection_decomposition)

# pylint: disable=protected-access
if Reflection._primitive is not None:

    @Reflection._primitive.def_impl
    def _(*args, n_wires, **kwargs):
        (U, alpha), reflection_wires = args[:-n_wires], args[-n_wires:]
        return type.__call__(Reflection, U, alpha, reflection_wires=reflection_wires, **kwargs)
