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

# pylint: disable = unused-argument


def ch_resource_decomp() -> list[GateCount | qre.Allocate | qre.Deallocate]:
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


def ch_toffoli_based_resource_decomp() -> list[GateCount | qre.Allocate | qre.Deallocate]:
    r"""Returns a list representing the resources of the :class:`~.pennylane.estimator.ops.op_math.controlled_ops.CH` operator.

    .. note::

        This operation assumes a catalytic T state is available.
        Users should ensure the cost of constructing such a state has been accounted for.

    Resources:
        The resources are derived from Figure: 17 in `arXiv:2011.03494 <https://arxiv.org/pdf/2011.03494>`_.

    Returns:
        list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
        represents a specific quantum gate and the number of times it appears
        in the decomposition.
    """

    gate_lst = []

    gate_lst.append(qre.Allocate(1))
    h = resource_rep(qre.Hadamard)

    gate_lst.append(GateCount(h, 5))
    gate_lst.append(GateCount(resource_rep(qre.S), 2))
    gate_lst.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1))
    gate_lst.append(GateCount(resource_rep(qre.Toffoli), 1))
    gate_lst.append(GateCount(resource_rep(qre.CNOT), 5))
    gate_lst.append(GateCount(resource_rep(qre.CZ), 1))
    gate_lst.append(GateCount(resource_rep(qre.X), 4))
    gate_lst.append(qre.Deallocate(1))

    return gate_lst
