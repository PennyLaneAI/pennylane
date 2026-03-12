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
"""Resource operators for controlled operations"""

import pennylane.labs.estimator_beta as qre
from pennylane.estimator import CompressedResourceOp, GateCount, ResourceOperator, resource_rep
from pennylane.wires import Wires, WiresLike

# pylint: disable=arguments-differ


class CH(ResourceOperator):
    r"""Resource class for the CH gate.

    Args:
        wires (Sequence[int] | None): the wires the operation acts on

    Resources:
        The resources are derived from the following identities:

        .. math::

            \begin{align}
                \hat{H} &= \hat{R}_{y}(\frac{\pi}{4}) \cdot \hat{Z}  \cdot \hat{R}_{y}(\frac{-\pi}{4}), \\
                \hat{Z} &= \hat{H} \cdot \hat{X}  \cdot \hat{H}.
            \end{align}

        Specifically, the resources are defined as two ``RY``, two ``Hadamard`` and one ``CNOT`` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.CH`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CH.resource_decomp()
    [(6 x Hadamard), (2 x S), (1 x CNOT)]

    """

    num_wires = 2

    def __init__(self, wires: WiresLike = None) -> None:
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: A compressed representation of the operator.
        """
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list of ``GateCount`` objects representing the resources of the operator..

        Resources:
            The resources are derived from the following identities:

            .. math::

                \begin{align}
                    \hat{H} &= \hat{R}_{y}(\frac{\pi}{4}) \cdot \hat{Z}  \cdot \hat{R}_{y}(\frac{-\pi}{4}), \\
                    \hat{Z} &= \hat{H} \cdot \hat{X}  \cdot \hat{H}.
                \end{align}

            Specifically, the resources are defined as two ``RY``, two ``Hadamard`` and one ``CNOT`` gates.

            The ``RY(:math:`\frac{\pi}{4}`)`` and ``RY(:math:`\frac{-\pi}{4}`)`` gates are further decomposed as
            :math:`e^{-i\frac{\pi}{8}}SHTHS^{\dagger}` and :math:`e^{i\frac{\pi}{8}}}SHT^{\dagger}HS^{\dagger}`, respectively.


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
            GateCount(h, 4),
            GateCount(t, 1),
            GateCount(t_dag, 1),
            GateCount(s, 2),
            GateCount(s_dag, 2),
            GateCount(cnot, 1),
        ]

    @classmethod
    def toffoli_based_resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator.

        .. note::

        This operation assumes a catalytic T state is available.
        Users should ensure the cost of constructing such a state has been accounted for.

        Resources:
            The resources are derived from Figure: 17 in `arXiv:2011.03494<https://arxiv.org/pdf/2011.03494>`_.

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
        gate_lst.append(
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1)
        )
        gate_lst.append(GateCount(resource_rep(qre.Toffoli), 1))
        gate_lst.append(GateCount(resource_rep(qre.CNOT), 5))
        gate_lst.append(GateCount(resource_rep(qre.CZ), 1))
        gate_lst.append(qre.Deallocate(1))

        return gate_lst

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            are same as the originial operation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are expressed using the symbolic :class:`~.pennylane.estimator.ops.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.pennylane.estimator.ops.Hadamard` class.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        ctrl_h = resource_rep(
            qre.Controlled,
            {
                "base_cmpr_op": resource_rep(qre.Hadamard),
                "num_ctrl_wires": num_ctrl_wires + 1,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [GateCount(ctrl_h)]

    @classmethod
    def pow_resource_decomp(
        cls, pow_z: int, target_resource_params: dict | None = None
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return (
            [GateCount(resource_rep(qre.Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep())]
        )
