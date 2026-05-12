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
"""Resource operators for controlled operations"""

import pennylane.labs.estimator_beta as qre
from pennylane.estimator import GateCount, resource_rep
from pennylane.labs.estimator_beta.wires_manager.base_classes import Allocate, Deallocate

# pylint: disable = unused-argument


def ch_resource_decomp() -> list[GateCount | Allocate | Deallocate]:
    r"""Returns a list of :class:`~.pennylane.estimator.resource_operator.GateCount` objects representing the resources of the :class:`~.pennylane.estimator.ops.op_math.controlled_ops.CH` operator.

    Resources:
        The resources are derived from the following identities:

        .. math::

            \begin{align}
                \hat{H} &= \hat{R}_{y}(\frac{\pi}{4}) \cdot \hat{Z}  \cdot \hat{R}_{y}(\frac{-\pi}{4}), \\
                \hat{Z} &= \hat{H} \cdot \hat{X}  \cdot \hat{H}.
            \end{align}

        Specifically, the resources are defined as two ``RY``, two ``Hadamard`` and one ``CNOT`` gates.

        Decomposing the :math:`\hat{R}_y(\pm\frac{\pi}{4})` rotations into the Clifford+T basis and substituting yields:

        .. math::

            \begin{align}
                \hat{H} &= (S H T H S^\dagger) \cdot \hat{Z} \cdot (S H T^\dagger H S^\dagger) \\
                        &= S H T \cdot (\hat{H} \hat{Z} \hat{H}) \cdot T^\dagger H S^\dagger \\
                        &= S H T \cdot \hat{X} \cdot T^\dagger H S^\dagger
            \end{align}

        The final resources are: 2 ``Hadamard``, 1 ``T``, 1 ``Adjoint(T)``,
        1 ``S``, 1 ``Adjoint(S)``, and 1 ``CNOT``.

    Returns:
        list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
        where each object represents a specific quantum gate and the number of times it appears
        in the decomposition.
    """

    s = resource_rep(qre.S)
    s_dag = resource_rep(qre.Adjoint, {"base_cmpr_op": s})
    t = resource_rep(qre.T)
    t_dag = resource_rep(qre.Adjoint, {"base_cmpr_op": t})
    h = resource_rep(qre.Hadamard)
    cnot = resource_rep(qre.CNOT)
    return [
        GateCount(h, 2),
        GateCount(t, 1),
        GateCount(t_dag, 1),
        GateCount(s, 1),
        GateCount(s_dag, 1),
        GateCount(cnot, 1),
    ]


def ch_toffoli_based_resource_decomp() -> list[GateCount | Allocate | Deallocate]:
    r"""Returns a list representing the resources of the :class:`~.estimator.ops.op_math.controlled_ops.CH` operator.

    .. note::

        This operation assumes a :doc:`catalytic T state <demo:demos/tutorial_magic_state_distillation>` is available.
        Users should ensure the cost of constructing such a state has been accounted for.

    Resources:
        The resources are derived from Figure 17 in `arXiv:2011.03494 <https://arxiv.org/pdf/2011.03494>`_.

    Returns:
        list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
        represents a specific quantum gate and the number of times it appears
        in the decomposition.
    """

    gate_lst = []

    gate_lst.append(Allocate(1))
    h = resource_rep(qre.Hadamard)

    gate_lst.append(GateCount(h, 5))
    gate_lst.append(GateCount(resource_rep(qre.S), 2))
    gate_lst.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1))
    gate_lst.append(GateCount(resource_rep(qre.Toffoli), 1))
    gate_lst.append(GateCount(resource_rep(qre.CNOT), 5))
    gate_lst.append(GateCount(resource_rep(qre.CZ), 1))
    gate_lst.append(GateCount(resource_rep(qre.X), 4))
    gate_lst.append(Deallocate(1))

    return gate_lst


def mcx_many_clean_aux_resource_decomp(
    num_ctrl_wires: int, num_zero_ctrl: int
) -> list[GateCount | Allocate | Deallocate]:
    r"""Returns a list representing the resources of the operator.

    Args:
        num_ctrl_wires (int): the number of qubits the operation is controlled on
        num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

    Resources:
        The resources are obtained based on the unary iteration technique described in
        `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_. Specifically, the
        resources are defined as the following rules:

        * If there are no control qubits, treat the operation as a :class:`~.pennylane.estimator.ops.X` gate.

        * If there is only one control qubit, treat the resources as a :class:`~.pennylane.estimator.ops.CNOT` gate.

        * If there are two control qubits, treat the resources as a :class:`~.pennylane.estimator.ops.Toffoli` gate.

        * If there are three or more control qubits (:math:`n`), the resources are obtained based on the unary
          iteration technique described in `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_.
          Specifically, it requires :math:`n - 2` clean qubits, and produces :math:`n - 2` pairs of elbow gates and
          a single :class:`~.pennylane.estimator.ops.Toffoli`.

    Returns:
        list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
        where each object represents a specific quantum gate and the number of times it appears
        in the decomposition.
    """
    gate_lst = []

    x = resource_rep(qre.X)
    if num_ctrl_wires == 0:
        if num_zero_ctrl:
            return []

        return [GateCount(x)]

    if num_zero_ctrl:
        gate_lst.append(GateCount(x, num_zero_ctrl * 2))

    cnot = resource_rep(qre.CNOT)
    if num_ctrl_wires == 1:
        gate_lst.append(GateCount(cnot))
        return gate_lst

    toffoli = resource_rep(qre.Toffoli)
    if num_ctrl_wires == 2:
        gate_lst.append(GateCount(toffoli))
        return gate_lst

    l_elbow = resource_rep(qre.TemporaryAND)
    r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})

    allocated_register = Allocate(num_ctrl_wires - 2, restored=True)
    res = [
        allocated_register,
        GateCount(l_elbow, num_ctrl_wires - 2),
        GateCount(r_elbow, num_ctrl_wires - 2),
        GateCount(toffoli, 1),
        Deallocate(allocated_register=allocated_register),
    ]
    gate_lst.extend(res)
    return gate_lst


def mcx_one_clean_aux_resource_decomp(
    num_ctrl_wires: int, num_zero_ctrl: int
) -> list[GateCount | Allocate | Deallocate]:
    r"""Returns a list representing the resources of the operator.

    Args:
        num_ctrl_wires (int): the number of qubits the operation is controlled on
        num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

    Resources:
        The resources are obtained based on the unary iteration technique described in
        `Khattar and Gidney, (2024) <https://arxiv.org/abs/2407.17966>`_. Specifically, the
        resources are defined as the following rules:

        * If there are no control qubits, treat the operation as a :class:`~.pennylane.estimator.ops.X` gate.

        * If there is only one control qubit, treat the resources as a :class:`~.pennylane.estimator.ops.CNOT` gate.

        * If there are two control qubits, treat the resources as a :class:`~.pennylane.estimator.ops.Toffoli` gate.

        * If there are three or more control qubits (:math:`n`), the resources are obtained based on the conditionally clean technique described in `Khattar and Gidney, (2024) <https://arxiv.org/abs/2407.17966>`_. Specifically, it requires :math:`1` clean qubit, and produces :math:`2n - 3` :class:`~.pennylane.estimator.ops.Toffoli` gates.

    Returns:
        list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
        where each object represents a specific quantum gate and the number of times it appears
        in the decomposition.
    """
    gate_lst = []

    x = resource_rep(qre.X)
    if num_ctrl_wires == 0:
        if num_zero_ctrl:
            return []

        return [GateCount(x)]

    if num_zero_ctrl:
        gate_lst.append(GateCount(x, num_zero_ctrl * 2))

    cnot = resource_rep(qre.CNOT)
    if num_ctrl_wires == 1:
        gate_lst.append(GateCount(cnot))
        return gate_lst

    toffoli = resource_rep(qre.Toffoli)
    if num_ctrl_wires == 2:
        gate_lst.append(GateCount(toffoli))
        return gate_lst

    aux_reg = Allocate(1, state="zero", restored=True)
    res = [
        aux_reg,
        GateCount(toffoli, 2 * num_ctrl_wires - 3),
        Deallocate(allocated_register=aux_reg),
    ]
    gate_lst.extend(res)
    return gate_lst


def mcx_one_dirty_aux_resource_decomp(
    num_ctrl_wires: int, num_zero_ctrl: int
) -> list[GateCount | Allocate | Deallocate]:
    r"""Returns a list representing the resources of the operator.

    Args:
        num_ctrl_wires (int): the number of qubits the operation is controlled on
        num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

    Resources:
        The resources are obtained based on the unary iteration technique described in
        `Khattar and Gidney, (2024) <https://arxiv.org/abs/2407.17966>`_. Specifically, the
        resources are defined as the following rules:

        * If there are no control qubits, treat the operation as a :class:`~.pennylane.estimator.ops.X` gate.

        * If there is only one control qubit, treat the resources as a :class:`~.pennylane.estimator.ops.CNOT` gate.

        * If there are two control qubits, treat the resources as a :class:`~.pennylane.estimator.ops.Toffoli` gate.

        * If there are three or more control qubits (:math:`n`), the resources are obtained based on the conditionally clean technique described in `Khattar and Gidney, (2024) <https://arxiv.org/abs/2407.17966>`_. Specifically, it requires :math:`1` dirty qubit, and produces :math:`4n - 8` :class:`~.pennylane.estimator.ops.Toffoli` gates.

    Returns:
        list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
        where each object represents a specific quantum gate and the number of times it appears
        in the decomposition.
    """
    gate_lst = []

    x = resource_rep(qre.X)
    if num_ctrl_wires == 0:
        if num_zero_ctrl:
            return []

        return [GateCount(x)]

    if num_zero_ctrl:
        gate_lst.append(GateCount(x, num_zero_ctrl * 2))

    cnot = resource_rep(qre.CNOT)
    if num_ctrl_wires == 1:
        gate_lst.append(GateCount(cnot))
        return gate_lst

    toffoli = resource_rep(qre.Toffoli)
    if num_ctrl_wires == 2:
        gate_lst.append(GateCount(toffoli))
        return gate_lst

    aux_reg = Allocate(1, state="any", restored=True)
    res = [
        aux_reg,
        GateCount(toffoli, 4 * num_ctrl_wires - 8),
        Deallocate(allocated_register=aux_reg),
    ]
    gate_lst.extend(res)
    return gate_lst
