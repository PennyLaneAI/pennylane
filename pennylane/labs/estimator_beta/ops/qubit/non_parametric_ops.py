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

r"""Resource operators for non parametric single qubit operations."""

import pennylane.labs.estimator_beta as qre
from pennylane.estimator.resource_operator import GateCount, resource_rep

# pylint: disable=arguments-differ, unused-argument


def hadamard_controlled_resource_decomp(
    num_ctrl_wires: int,
    num_zero_ctrl: int,
    target_resource_params: dict | None = None,
) -> list[GateCount | qre.Allocate | qre.Deallocate]:
    r"""Returns a list representing the resources for a controlled version of the :class:`~.pennylane.estimator.ops.qubit.non_parametric_ops.Hadamard` operator.

    Args:
        num_ctrl_wires (int): the number of qubits the
            operation is controlled on
        num_zero_ctrl (int): the number of control qubits, that are
            controlled when in the :math:`|0\rangle` state
        target_resource_params (dict | None): A dictionary containing the resource parameters
            of the target operator.

    Resources:
        For a single control wire, the cost is a single instance of ``CH``.
        Two additional ``X`` gates are used to flip the control qubit if it is zero-controlled.
        In the case where multiple controlled wires are provided, the resources are derived from
        the following identities:

        .. math::

            \begin{align}
                \hat{H} &= \hat{R}_{y}(\frac{\pi}{4}) \cdot \hat{Z}  \cdot \hat{R}_{y}(\frac{-\pi}{4}), \\
                \hat{Z} &= \hat{H} \cdot \hat{X}  \cdot \hat{H}.
            \end{align}

        Specifically, the resources are given by two ``RY`` gates, two
        ``Hadamard`` gates and a ``X`` gate. By replacing the
        ``X`` gate with ``MultiControlledX`` gate, we obtain a
        controlled-version of this identity.

        Decomposing the :math:`\hat{R}_y(\pm\frac{\pi}{4})` rotations into the Clifford+T basis and substituting yields:

        .. math::

            \begin{align}
                \hat{H} &= (S H T H S^\dagger) \cdot \hat{Z} \cdot (S H T^\dagger H S^\dagger) \\
                        &= S H T \cdot (\hat{H} \hat{Z} \hat{H}) \cdot T^\dagger H S^\dagger \\
                        &= S H T \cdot \hat{X} \cdot T^\dagger H S^\dagger
            \end{align}

        The final resources are: 2 ``Hadamard``, 1 ``T``, 1 ``Adjoint(T)``,
        1 ``S``, 1 ``Adjoint(S)``, and 1 ``MultiControlledX`` controlled on ``num_ctrl_wires``.


    Returns:
        list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
        represents a specific quantum gate and the number of times it appears
        in the decomposition.
    """
    if num_ctrl_wires == 1:
        gate_lst = [GateCount(resource_rep(qre.CH))]

        if num_zero_ctrl:
            gate_lst.append(GateCount(resource_rep(qre.X), 2))

        return gate_lst

    gate_lst = []

    h = resource_rep(qre.Hadamard)
    mcx = resource_rep(
        qre.MultiControlledX,
        {
            "num_ctrl_wires": num_ctrl_wires,
            "num_zero_ctrl": num_zero_ctrl,
        },
    )
    gate_lst.append(qre.Allocate(1))
    gate_lst.append(GateCount(h, 2))
    gate_lst.append(GateCount(resource_rep(qre.T), 1))
    gate_lst.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.T)}), 1))
    gate_lst.append(GateCount(resource_rep(qre.S), 1))
    gate_lst.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1))
    gate_lst.append(GateCount(mcx, 2))
    gate_lst.append(qre.Deallocate(1))
    return gate_lst


def hadamard_toffoli_based_controlled_decomp(
    num_ctrl_wires: int,
    num_zero_ctrl: int,
    target_resource_params: dict | None = None,
) -> list[GateCount | qre.Allocate | qre.Deallocate]:
    r"""Returns a list representing the resources for a controlled version of the :class:`~.pennylane.estimator.ops.qubit.non_parametric_ops.Hadamard` operator.

    .. note::

        This operation assumes a `catalytic T state <https://pennylane.ai/qml/demos/tutorial_magic_state_distillation>`_ is available.
        Users should ensure the cost of constructing such a state has been accounted for.

    Args:
        num_ctrl_wires (int): the number of qubits the
            operation is controlled on
        num_zero_ctrl (int): the number of control qubits, that are
            controlled when in the :math:`|0\rangle` state
        target_resource_params (dict | None): A dictionary containing the resource parameters
            of the target operator.

    Resources:
        The resources are derived from Figure 17 in `arXiv:2011.03494 <https://arxiv.org/pdf/2011.03494>`_.

    Returns:
        list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
        represents a specific quantum gate and the number of times it appears
        in the decomposition.
    """

    gate_lst = []

    if num_ctrl_wires > 1:
        gate_lst.append(qre.Allocate(1))

    gate_lst.append(qre.Allocate(1))
    h = resource_rep(qre.Hadamard)
    mcx = resource_rep(
        qre.MultiControlledX,
        {
            "num_ctrl_wires": num_ctrl_wires,
            "num_zero_ctrl": num_zero_ctrl,
        },
    )

    gate_lst.append(GateCount(h, 5))
    gate_lst.append(GateCount(resource_rep(qre.S), 2))
    gate_lst.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1))
    gate_lst.append(GateCount(resource_rep(qre.Toffoli), 1))
    gate_lst.append(GateCount(resource_rep(qre.CNOT), 5))
    gate_lst.append(GateCount(resource_rep(qre.CZ), 1))
    gate_lst.append(GateCount(resource_rep(qre.X), 4))

    if num_ctrl_wires > 1:
        gate_lst.append(GateCount(mcx, 2))
        gate_lst.append(qre.Deallocate(1))

    gate_lst.append(qre.Deallocate(1))

    return gate_lst
