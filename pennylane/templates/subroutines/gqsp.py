# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Contains the GQSP template.
"""

import copy

from pennylane import capture, ops
from pennylane.decomposition import add_decomps, controlled_resource_rep, register_resources
from pennylane.operation import Operation
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

has_jax = True
try:
    from jax import numpy as jnp
except (ModuleNotFoundError, ImportError) as import_error:  # pragma: no cover
    has_jax = False  # pragma: no cover


class GQSP(Operation):
    r"""
    Implements the generalized quantum signal processing (GQSP) circuit.

    This operation encodes a polynomial transformation of an input unitary operator following the algorithm
    described in `arXiv:2308.01501 <https://arxiv.org/abs/2308.01501>`__ as:

    .. math::
         U
         \xrightarrow{GQSP}
         \begin{pmatrix}
         \text{poly}(U) & * \\
         * & * \\
         \end{pmatrix}

    The implementation requires one control qubit.

    Args:

        unitary (Operator): the operator to be encoded by the GQSP circuit
        angles (tensor[float]): array of angles defining the polynomial transformation. The shape of the array must be `(3, d+1)`, where `d` is the degree of the polynomial.
        control (Union[Wires, int, str]): control qubit used to encode the polynomial transformation

    .. note::

       The  :func:`~.poly_to_angles` function can be used to calculate the angles for a given polynomial.

    Example:

    .. code-block:: python

        # P(x) = 0.1 + 0.2j x + 0.3 x^2
        poly = [0.1, 0.2j, 0.3]

        angles = qml.poly_to_angles(poly, "GQSP")

        @qml.prod # transforms the qfunc into an Operator
        def unitary(wires):
            qml.RX(0.3, wires)

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(angles):
            qml.GQSP(unitary(wires = 1), angles, control = 0)
            return qml.state()

        matrix = qml.matrix(circuit, wire_order=[0, 1])(angles)

    .. code-block:: pycon

        >>> print(np.round(matrix,3)[:2, :2])
        [[0.387+0.198j 0.03 -0.089j]
        [0.03 -0.089j 0.387+0.198j]]
    """

    grad_method = None

    resource_keys = {"unitary", "num_iters"}

    def __init__(self, unitary, angles, control, id=None):
        total_wires = Wires(control) + unitary.wires

        self._hyperparameters = {"unitary": unitary, "control": control}

        super().__init__(angles, *unitary.data, wires=total_wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "unitary": self.hyperparameters["unitary"],
            "num_iters": min(len(self.data[0][0]), len(self.data[0][1]), len(self.data[0][2])),
        }

    def _flatten(self):
        return (*self.data, self.hyperparameters["unitary"]), (self.hyperparameters["control"],)

    @classmethod
    def _unflatten(cls, data, metadata):
        # Data contains (angles, derived_data_from_unitary, unitary)
        return cls(unitary=data[-1], angles=data[0], control=metadata[0])

    # pylint: disable=arguments-differ
    @classmethod
    def _primitive_bind_call(cls, unitary, angles, control, id=None):
        return super()._primitive_bind_call(unitary, angles, wires=control, id=id)

    def map_wires(self, wire_map: dict):
        # pylint: disable=protected-access
        new_op = copy.deepcopy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        new_op._hyperparameters["unitary"] = ops.functions.map_wires(
            new_op._hyperparameters["unitary"], wire_map
        )
        new_op._hyperparameters["control"] = tuple(
            wire_map.get(w, w) for w in Wires(new_op._hyperparameters["control"])
        )

        return new_op

    @staticmethod
    def compute_decomposition(*parameters, **hyperparameters):
        r"""
        Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.Operator.decomposition`.

        Args:
            *parameters (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            wires (Iterable[Any], Wires): wires that the operator acts on
            **hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            list[Operator]: decomposition of the operator
        """

        unitary = hyperparameters["unitary"]
        control = hyperparameters["control"]

        angles = parameters[0]

        thetas, phis, lambds = angles[0], angles[1], angles[2]

        op_list = []

        # These four gates adapt PennyLane's qml.U3 to the chosen U3 format in the GQSP paper.
        op_list.append(ops.X(control))
        op_list.append(ops.U3(2 * thetas[0], phis[0], lambds[0], wires=control))
        op_list.append(ops.X(control))
        op_list.append(ops.Z(control))

        for theta, phi, lamb in zip(thetas[1:], phis[1:], lambds[1:]):

            op_list.append(ops.ctrl(unitary, control=control, control_values=0))

            op_list.append(ops.X(control))
            op_list.append(ops.U3(2 * theta, phi, lamb, wires=control))
            op_list.append(ops.X(control))
            op_list.append(ops.Z(control))

        return op_list

    def queue(self, context=QueuingManager):
        context.remove(self.hyperparameters["unitary"])
        context.append(self)
        return self


def _GQSP_resources(unitary, num_iters):
    resources = {
        ops.X: 2 + 2 * (num_iters - 1),
        ops.U3: num_iters,
        ops.Z: num_iters,
        controlled_resource_rep(
            base_class=unitary.__class__,
            base_params=unitary.resource_params,
            num_control_wires=1,
            num_zero_control_values=1,
        ): num_iters
        - 1,
    }

    return resources


@register_resources(_GQSP_resources)
def _GQSP_decomposition(*parameters, **hyperparameters):
    unitary = hyperparameters["unitary"]
    control = hyperparameters["control"]

    angles = parameters[0]

    thetas, phis, lambds = angles[0], angles[1], angles[2]

    if has_jax and capture.enabled():
        thetas, phis, lambds = jnp.array(thetas), jnp.array(phis), jnp.array(lambds)

    # These four gates adapt PennyLane's ops.U3 to the chosen U3 format in the GQSP paper.
    ops.X(control)
    ops.U3(2 * thetas[0], phis[0], lambds[0], wires=control)
    ops.X(control)
    ops.Z(control)

    for theta, phi, lamb in zip(thetas[1:], phis[1:], lambds[1:]):

        ops.ctrl(unitary, control=control, control_values=[0])

        ops.X(control)
        ops.U3(2 * theta, phi, lamb, wires=control)
        ops.X(control)
        ops.Z(control)


add_decomps(GQSP, _GQSP_decomposition)

# pylint: disable=protected-access
if GQSP._primitive is not None:

    @GQSP._primitive.def_impl
    def _(*args, n_wires, **kwargs):
        (unitary, angles), control = args[:-n_wires], args[-n_wires:]
        return type.__call__(GQSP, unitary, angles, control=control, **kwargs)
