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
r"""Resource operators for PennyLane subroutine templates."""

import math
from collections import defaultdict

import pennylane.estimator as qre
from pennylane import numpy as qnp
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    _dequeue,
    resource_rep,
)
from pennylane.estimator.wires_manager import Allocate, Deallocate
from pennylane.exceptions import ResourcesUndefinedError
from pennylane.wires import Wires, WiresLike

# pylint: disable=arguments-differ,too-many-arguments,unused-argument,super-init-not-called, signature-differs


class OutOfPlaceSquare(ResourceOperator):
    r"""Resource class for the OutofPlaceSquare gate.

    Args:
        register_size (int): the size of the input register
        wires (Sequence[int], None): the wires the operation acts on

    Resources:
        The resources are obtained from appendix G, lemma 7 in `PRX Quantum, 2, 040332 (2021)
        <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332>`_. Specifically,
        the resources are given as :math:`(n - 1)^2` Toffoli gates, and :math:`n` CNOT gates, where
        :math:`n` is the size of the input register.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> out_square = qre.OutOfPlaceSquare(register_size=3)
    >>> print(qre.estimate(out_square))
    --- Resources: ---
    Total wires: 9
        algorithmic wires: 9
        allocated wires: 0
        zero state: 0
        any state: 0
    Total gates : 7
    'Toffoli': 4,
    'CNOT': 3
    """

    resource_keys = {"register_size"}

    def __init__(self, register_size: int, wires: WiresLike = None):
        self.register_size = register_size
        self.num_wires = 3 * register_size
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * register_size (int): the size of the input register
        """
        return {"register_size": self.register_size}

    @classmethod
    def resource_rep(cls, register_size: int):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            register_size (int): the size of the input register

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = 3 * register_size
        return CompressedResourceOp(cls, num_wires, {"register_size": register_size})

    @classmethod
    def resource_decomp(cls, register_size):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            register_size (int): the size of the input register

        Resources:
            The resources are obtained from appendix G, lemma 7 in `PRX Quantum, 2, 040332 (2021)
            <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332>`_. Specifically,
            the resources are given as :math:`(n - 1)^2` Toffoli gates, and :math:`n` CNOT gates.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_lst = []

        gate_lst.append(GateCount(resource_rep(qre.Toffoli), (register_size - 1) ** 2))
        gate_lst.append(GateCount(resource_rep(qre.CNOT), register_size))

        return gate_lst


class PhaseGradient(ResourceOperator):
    r"""Resource class for the PhaseGradient gate.

    This operation prepares the phase gradient state
    :math:`\frac{1}{\sqrt{2^b}} \cdot \sum_{k=0}^{2^b - 1} e^{-i2\pi \frac{k}{2^b}}\ket{k}`, where
    :math:`b` is the number of qubits. The equation is taken from page 4 of
    `C. Gidney, Quantum 2, 74, (2018) <https://quantum-journal.org/papers/q-2018-06-18-74/>`_.

    Args:
        num_wires (int | None): the number of wires to prepare in the phase gradient state
        wires (Sequence[int], None): the wires the operation acts on

    Resources:
        The phase gradient state is defined as an equal superposition of phase shifts where each shift
        is progressively more precise. This is achieved by applying Hadamard gates to each qubit and
        then applying Z-rotations to each qubit with progressively smaller rotation angle. The first
        three rotations can be compiled to a Z-gate, S-gate and a T-gate.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> phase_grad = qre.PhaseGradient(num_wires=5)
    >>> gate_set={"Z", "S", "T", "RZ", "Hadamard"}
    >>> print(qre.estimate(phase_grad, gate_set))
    --- Resources: ---
    Total wires: 5
        algorithmic wires: 5
        allocated wires: 0
        zero state: 0
        any state: 0
    Total gates : 10
    'RZ': 2,
    'T': 1,
    'Z': 1,
    'S': 1,
    'Hadamard': 5
    """

    resource_keys = {"num_wires"}

    def __init__(self, num_wires: int | None = None, wires: WiresLike = None):
        if num_wires is None:
            if wires is None:
                raise ValueError("Must provide atleast one of `num_wires` and `wires`.")
            num_wires = len(wires)
        self.num_wires = num_wires
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int): the number of qubits to prepare in the phase gradient state
        """
        return {"num_wires": self.num_wires}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            num_wires (int): the number of qubits to prepare in the phase gradient state

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, num_wires, {"num_wires": num_wires})

    @classmethod
    def resource_decomp(cls, num_wires: int):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_wires (int): the number of qubits to prepare in the phase gradient state

        Resources:
            The resources are obtained by construction. The phase gradient state is defined as an
            equal superposition of phase shifts where each shift is progressively more precise. This
            is achieved by applying Hadamard gates to each qubit and then applying Z-rotations to each
            qubit with progressively smaller rotation angle. The first three rotations can be compiled to
            a Z-gate, S-gate and a T-gate.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_counts = [GateCount(resource_rep(qre.Hadamard), num_wires)]
        if num_wires > 0:
            gate_counts.append(GateCount(resource_rep(qre.Z)))

        if num_wires > 1:
            gate_counts.append(GateCount(resource_rep(qre.S)))

        if num_wires > 2:
            gate_counts.append(GateCount(resource_rep(qre.T)))

        if num_wires > 3:
            gate_counts.append(GateCount(resource_rep(qre.RZ), num_wires - 3))

        return gate_counts


class OutMultiplier(ResourceOperator):
    r"""Resource class for the OutMultiplier gate.

    Args:
        a_num_wires (int): the size of the first input register
        b_num_wires (int): the size of the second input register
        wires (Sequence[int], None): the wires the operation acts on

    Resources:
        The resources are obtained from appendix G, lemma 10 in `PRX Quantum, 2, 040332 (2021)
        <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332>`_.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.OutMultiplier`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> out_mul = qre.OutMultiplier(4, 4)
    >>> print(qre.estimate(out_mul))
    --- Resources: ---
    Total wires: 16
        algorithmic wires: 16
        allocated wires: 0
        zero state: 0
        any state: 0
    Total gates : 70
    'Toffoli': 14,
    'CNOT': 14,
    'Hadamard': 42
    """

    resource_keys = {"a_num_wires", "b_num_wires"}

    def __init__(self, a_num_wires: int, b_num_wires: int, wires: WiresLike = None) -> None:
        self.num_wires = a_num_wires + b_num_wires + 2 * max((a_num_wires, b_num_wires))
        self.a_num_wires = a_num_wires
        self.b_num_wires = b_num_wires
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * a_num_wires (int): the size of the first input register
                * b_num_wires (int): the size of the second input register
        """
        return {"a_num_wires": self.a_num_wires, "b_num_wires": self.b_num_wires}

    @classmethod
    def resource_rep(cls, a_num_wires, b_num_wires) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            a_num_wires (int): the size of the first input register
            b_num_wires (int): the size of the second input register

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = a_num_wires + b_num_wires + 2 * max((a_num_wires, b_num_wires))
        return CompressedResourceOp(
            cls, num_wires, {"a_num_wires": a_num_wires, "b_num_wires": b_num_wires}
        )

    @classmethod
    def resource_decomp(cls, a_num_wires, b_num_wires) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            a_num_wires (int): the size of the first input register
            b_num_wires (int): the size of the second input register

        Resources:
            The resources are obtained from appendix G, lemma 10 in `PRX Quantum, 2, 040332 (2021)
            <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        l = max(a_num_wires, b_num_wires)

        toff = resource_rep(qre.Toffoli)
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})

        toff_count = 2 * a_num_wires * b_num_wires - l
        elbow_count = toff_count // 2
        toff_count = toff_count - (elbow_count * 2)

        gate_lst = [
            GateCount(l_elbow, elbow_count),
            GateCount(r_elbow, elbow_count),
        ]

        if toff_count:
            gate_lst.append(GateCount(toff))
        return gate_lst


class SemiAdder(ResourceOperator):
    r"""Resource class for the SemiAdder gate.

    Args:
        max_register_size (int): the size of the larger of the two registers being added together
        wires (Sequence[int], None): the wires the operation acts on

    Resources:
        The resources are obtained from figures 1 and 2 in `Gidney (2018)
        <https://quantum-journal.org/papers/q-2018-06-18-74/pdf/>`_.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.SemiAdder`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> semi_add = qre.SemiAdder(max_register_size=4)
    >>> print(qre.estimate(semi_add))
    --- Resources: ---
    Total wires: 11
        algorithmic wires: 8
        allocated wires: 3
        zero state: 3
        any state: 0
    Total gates : 30
    'Toffoli': 3,
    'CNOT': 18,
    'Hadamard': 9
    """

    resource_keys = {"max_register_size"}

    def __init__(self, max_register_size: int, wires: WiresLike = None):
        self.max_register_size = max_register_size
        self.num_wires = 2 * max_register_size
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * max_register_size (int): the size of the larger of the two registers being added together

        """
        return {"max_register_size": self.max_register_size}

    @classmethod
    def resource_rep(cls, max_register_size):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            max_register_size (int): the size of the larger of the two registers being added together

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = 2 * max_register_size
        return CompressedResourceOp(cls, num_wires, {"max_register_size": max_register_size})

    @classmethod
    def resource_decomp(cls, max_register_size: int):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            max_register_size (int): the size of the larger of the two registers being added together

        Resources:
            The resources are obtained from figures 1 and 2 in `Gidney (2018)
            <https://quantum-journal.org/papers/q-2018-06-18-74/pdf/>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        cnot = resource_rep(qre.CNOT)
        if max_register_size == 1:
            return [GateCount(cnot)]

        x = resource_rep(qre.X)
        toff = resource_rep(qre.Toffoli)
        if max_register_size == 2:
            return [GateCount(cnot, 2), GateCount(x, 2), GateCount(toff)]

        cnot_count = (6 * (max_register_size - 2)) + 3
        elbow_count = max_register_size - 1

        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})
        return [
            Allocate(max_register_size - 1),
            GateCount(cnot, cnot_count),
            GateCount(l_elbow, elbow_count),
            GateCount(r_elbow, elbow_count),
            Deallocate(max_register_size - 1),
        ]  # Obtained resource from Fig1 and Fig2 https://quantum-journal.org/papers/q-2018-06-18-74/pdf/

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires: int, num_zero_ctrl: int, target_resource_params: dict
    ):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): dictionary containing the size of the larger of the two registers being added together

        Resources:
            The resources are obtained from figure 4a in `Gidney (2018)
            <https://quantum-journal.org/papers/q-2018-06-18-74/pdf/>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        max_register_size = target_resource_params["max_register_size"]
        if max_register_size <= 2:
            raise ResourcesUndefinedError
        gate_lst = []

        if num_ctrl_wires > 1:
            mcx = resource_rep(
                qre.MultiControlledX,
                {
                    "num_ctrl_wires": num_ctrl_wires,
                    "num_zero_ctrl": num_zero_ctrl,
                },
            )
            gate_lst.append(Allocate(1))
            gate_lst.append(GateCount(mcx, 2))

        cnot_count = (7 * (max_register_size - 2)) + 3
        elbow_count = 2 * (max_register_size - 1)

        x = resource_rep(qre.X)
        cnot = resource_rep(qre.CNOT)
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})
        gate_lst.extend(
            [
                Allocate(max_register_size - 1),
                GateCount(cnot, cnot_count),
                GateCount(l_elbow, elbow_count),
                GateCount(r_elbow, elbow_count),
                Deallocate(max_register_size - 1),
            ],
        )

        if num_ctrl_wires > 1:
            gate_lst.append(Deallocate(1))
        elif num_zero_ctrl > 0:
            gate_lst.append(GateCount(x, 2 * num_zero_ctrl))

        return gate_lst


class ControlledSequence(ResourceOperator):
    r"""Resource class for the ControlledSequence gate.

    This operator represents a sequence of controlled gates, one for each control wire, with the
    base operator raised to decreasing powers of 2.

    Args:
        base (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): The operator to repeatedly
            apply in a controlled fashion.
        num_control_wires (int): the number of controlled wires to run the sequence over
        wires (Sequence[int], None): the wires the operation acts on

    Resources:
        The resources are obtained as a direct result of the definition of the operator:

        .. code-block:: bash

            0: ──╭●───────────────┤
            1: ──│────╭●──────────┤
            2: ──│────│────╭●─────┤
            t: ──╰U⁴──╰U²──╰U¹────┤

    .. seealso:: The associated PennyLane operation :class:`~.pennylane.ControlledSequence`

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> ctrl_seq = qre.ControlledSequence(
    ...     base = qre.RX(),
    ...     num_control_wires = 3,
    ... )
    >>> gate_set={"CRX"}
    >>> print(qre.estimate(ctrl_seq, gate_set))
    --- Resources: ---
    Total wires: 4
        algorithmic wires: 4
        allocated wires: 0
        zero state: 0
        any state: 0
    Total gates : 3
    'CRX': 3
    """

    resource_keys = {"base_cmpr_op", "num_ctrl_wires"}

    def __init__(
        self, base: ResourceOperator, num_control_wires: int, wires: WiresLike = None
    ) -> None:
        _dequeue(op_to_remove=base)
        self.queue()
        base_cmpr_op = base.resource_rep_from_op()

        self.base_cmpr_op = base_cmpr_op
        self.num_ctrl_wires = num_control_wires

        self.num_wires = num_control_wires + base_cmpr_op.num_wires
        if wires:
            self.wires = Wires(wires)
            if base_wires := base.wires:
                self.wires = Wires.all_wires([self.wires, base_wires])
            if len(self.wires) != self.num_wires:
                raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}.")
        else:
            self.wires = None

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * base_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed resource operator, corresponding
                  to the operator that we will be applying controlled powers of.
                * num_ctrl_wires (int): the number of controlled wires to run the sequence over
        """
        return {"base_cmpr_op": self.base_cmpr_op, "num_ctrl_wires": self.num_ctrl_wires}

    @classmethod
    def resource_rep(
        cls, base_cmpr_op: CompressedResourceOp, num_ctrl_wires: int
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            base_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed resource operator, corresponding
                to the operator that we will be applying controlled powers of.
            num_ctrl_wires (int): the number of controlled wires to run the sequence over

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        params = {"base_cmpr_op": base_cmpr_op, "num_ctrl_wires": num_ctrl_wires}
        num_wires = num_ctrl_wires + base_cmpr_op.num_wires
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, base_cmpr_op, num_ctrl_wires):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            base_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed resource operator, corresponding
                to the operator that we will be applying controlled powers of.
            num_ctrl_wires (int): the number of controlled wires to run the sequence over

        Resources:
            The resources are obtained as a direct result of the definition of the operator:

            .. code-block:: bash

                0: ──╭●───────────────┤
                1: ──│────╭●──────────┤
                2: ──│────│────╭●─────┤
                t: ──╰U⁴──╰U²──╰U¹────┤

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_counts = []
        base_op = base_cmpr_op

        if base_cmpr_op.op_type == qre.ChangeOpBasis:
            base_op = base_cmpr_op.params["cmpr_target_op"]
            compute_op = base_cmpr_op.params["cmpr_compute_op"]
            uncompute_op = base_cmpr_op.params["cmpr_uncompute_op"]

            gate_counts.append(GateCount(compute_op))

        for z in range(num_ctrl_wires):
            ctrl_pow_u = qre.Controlled.resource_rep(
                qre.Pow.resource_rep(base_op, 2**z),
                num_ctrl_wires=1,
                num_zero_ctrl=0,
            )
            gate_counts.append(GateCount(ctrl_pow_u))

        if base_cmpr_op.op_type == qre.ChangeOpBasis:
            gate_counts.append(GateCount(uncompute_op))

        return gate_counts


class QPE(ResourceOperator):
    r"""Resource class for QuantumPhaseEstimation (QPE).

    Args:
        base (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): the phase estimation operator
        num_estimation_wires (int): the number of wires used for measuring out the phase
        adj_qft_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator` | None): An optional
            argument to set the subroutine used to perform the adjoint QFT operation.
        wires (Sequence[int], None): the wires the operation acts on

    Resources:
        The resources are obtained from the standard decomposition of QPE as presented
        in (Section 5.2) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum
        Information <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.QuantumPhaseEstimation`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> gate_set = {"Hadamard", "Adjoint(QFT(5))", "CRX"}
    >>> qpe = qre.QPE(qre.RX(precision=1e-3), 5)
    >>> print(qre.estimate(qpe, gate_set))
    --- Resources: ---
     Total wires: 6
        algorithmic wires: 6
        allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 11
      'CRX': 5,
      'Adjoint(QFT(5))': 1,
      'Hadamard': 5

    .. details::
        :title: Usage Details

        Additionally, we can customize the implementation of the QFT operator we wish to use within
        the textbook QPE algorithm. This allows users to optimize the implementation of QPE by using
        more efficient implementations of the QFT.

        For example, consider the cost using the default :class:`~.pennylane.estimator.templates.QFT` implementation below:

        >>> import pennylane.estimator as qre
        >>> qpe = qre.QPE(qre.RX(precision=1e-3), 5, adj_qft_op=None)
        >>> print(qre.estimate(qpe))
        --- Resources: ---
         Total wires: 6
            algorithmic wires: 6
            allocated wires: 0
                 zero state: 0
                 any state: 0
         Total gates : 1.586E+3
          'T': 1.530E+3,
          'CNOT': 36,
          'Hadamard': 20

        Now we use the :class:`~.pennylane.estimator.templates.AQFT`:

        >>> aqft = qre.AQFT(order=3, num_wires=5)
        >>> adj_aqft = qre.Adjoint(aqft)
        >>> qpe = qre.QPE(qre.RX(precision=1e-3), 5, adj_qft_op=adj_aqft)
        >>> print(qre.estimate(qpe))
        --- Resources: ---
         Total wires: 8
            algorithmic wires: 6
             allocated wires: 2
             zero state: 2
            any state: 0
        Total gates : 321
         'Toffoli': 7,
         'T': 222,
         'CNOT': 34,
         'X': 4,
         'Z': 8,
         'S': 8,
         'Hadamard': 38
    """

    resource_keys = {"base_cmpr_op", "num_estimation_wires", "adj_qft_cmpr_op"}

    def __init__(
        self,
        base: ResourceOperator,
        num_estimation_wires: int,
        adj_qft_op: ResourceOperator | None = None,
        wires: WiresLike | None = None,
    ):
        remove_ops = [base, adj_qft_op] if adj_qft_op is not None else [base]
        _dequeue(remove_ops)
        self.queue()

        base_cmpr_op = base.resource_rep_from_op()
        adj_qft_cmpr_op = None if adj_qft_op is None else adj_qft_op.resource_rep_from_op()

        self.base_cmpr_op = base_cmpr_op
        self.adj_qft_cmpr_op = adj_qft_cmpr_op
        self.num_estimation_wires = num_estimation_wires

        self.num_wires = self.num_estimation_wires + base_cmpr_op.num_wires
        if wires:
            self.wires = Wires(wires)
            if base_wires := base.wires:
                self.wires = Wires.all_wires([self.wires, base_wires])
            if len(self.wires) != self.num_wires:
                raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}.")
        else:
            self.wires = None

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * base_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed resource operator, corresponding
                  to the phase estimation operator.
                * num_estimation_wires (int): the number of wires used for measuring out the phase
                * adj_qft_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp` | None]): An optional compressed
                  resource operator, corresponding to the adjoint QFT routine. If :code:`None`, the
                  default :class:`~.pennylane.estimator.templates.subroutines.QFT` will be used.
        """

        return {
            "base_cmpr_op": self.base_cmpr_op,
            "num_estimation_wires": self.num_estimation_wires,
            "adj_qft_cmpr_op": self.adj_qft_cmpr_op,
        }

    @classmethod
    def resource_rep(
        cls,
        base_cmpr_op: CompressedResourceOp,
        num_estimation_wires: int,
        adj_qft_cmpr_op: CompressedResourceOp = None,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            base_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed resource operator, corresponding
                to the phase estimation operator.
            num_estimation_wires (int): the number of wires used for measuring out the phase
            adj_qft_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp` | None): An optional compressed
                resource operator, corresponding to the adjoint QFT routine. If :code:`None`, the
                default :class:`~.pennylane.estimator.templates.subroutines.QFT` will be used.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        params = {
            "base_cmpr_op": base_cmpr_op,
            "num_estimation_wires": num_estimation_wires,
            "adj_qft_cmpr_op": adj_qft_cmpr_op,
        }
        num_wires = num_estimation_wires + base_cmpr_op.num_wires
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        base_cmpr_op: CompressedResourceOp,
        num_estimation_wires: int,
        adj_qft_cmpr_op: CompressedResourceOp | None = None,
    ):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            base_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed resource operator, corresponding
                to the phase estimation operator.
            num_estimation_wires (int): the number of wires used for measuring out the phase
            adj_qft_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp` | None): An optional compressed
                resource operator, corresponding to the adjoint QFT routine. If :code:`None`, the
                default :class:`~.pennylane.estimator.templates.subroutines.QFT` will be used.

        Resources:
            The resources are obtained from the standard decomposition of QPE as presented
            in (section 5.2) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum
            Information <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.
        """
        hadamard = resource_rep(qre.Hadamard)
        ctrl_op = ControlledSequence.resource_rep(base_cmpr_op, num_estimation_wires)
        if adj_qft_cmpr_op is None:
            adj_qft_cmpr_op = resource_rep(
                qre.Adjoint,
                {
                    "base_cmpr_op": resource_rep(QFT, {"num_wires": num_estimation_wires}),
                },
            )

        return [
            GateCount(hadamard, num_estimation_wires),
            GateCount(ctrl_op),
            GateCount(adj_qft_cmpr_op),
        ]

    @staticmethod
    def tracking_name(
        base_cmpr_op: CompressedResourceOp,
        num_estimation_wires: int,
        adj_qft_cmpr_op: CompressedResourceOp | None = None,
    ) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_cmpr_op.name
        adj_qft_name = None if adj_qft_cmpr_op is None else adj_qft_cmpr_op.name
        return f"QPE({base_name}, {num_estimation_wires}, adj_qft={adj_qft_name})"


class IterativeQPE(ResourceOperator):
    r"""Resource class for Iterative Quantum Phase Estimation (IQPE).

    Args:
        base (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): the phase estimation operator
        num_iter (int): the number of mid-circuit measurements performed to read out the phase

    Resources:
        The resources are obtained following the construction from `arXiv:0610214v3 <https://arxiv.org/abs/quant-ph/0610214v3>`_.

    .. seealso:: :func:`~.pennylane.iterative_qpe`

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> gate_set = {"Hadamard", "CRX", "PhaseShift"}
    >>> iqpe = qre.IterativeQPE(qre.RX(), 5)
    >>> print(qre.estimate(iqpe, gate_set))
    --- Resources: ---
    Total wires: 2
        algorithmic wires: 1
        allocated wires: 1
        zero state: 1
        any state: 0
    Total gates : 25
    'CRX': 5,
    'PhaseShift': 10,
    'Hadamard': 10
    """

    resource_keys = {"base_cmpr_op", "num_iter"}

    def __init__(self, base: ResourceOperator, num_iter: int):
        _dequeue(base)
        self.queue()

        self.base_cmpr_op = base.resource_rep_from_op()
        self.num_iter = num_iter

        self.wires = base.wires
        self.num_wires = self.base_cmpr_op.num_wires

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * base_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed resource operator, corresponding
                  to the phase estimation operator.
                * num_iter (int): the number of mid-circuit measurements made to read out the phase
        """
        return {"base_cmpr_op": self.base_cmpr_op, "num_iter": self.num_iter}

    @classmethod
    def resource_rep(
        cls, base_cmpr_op: CompressedResourceOp, num_iter: int
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            base_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed resource operator, corresponding
                to the phase estimation operator.
            num_iter (int): the number of mid-circuit measurements made to read out the phase

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = base_cmpr_op.num_wires
        return CompressedResourceOp(
            cls, num_wires, {"base_cmpr_op": base_cmpr_op, "num_iter": num_iter}
        )

    @classmethod
    def resource_decomp(cls, base_cmpr_op, num_iter):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            base_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed resource operator, corresponding
                to the phase estimation operator.
            num_iter (int): the number of mid-circuit measurements made to read out the phase

        Resources:
            The resources are obtained following the construction from `arXiv:0610214v3
            <https://arxiv.org/abs/quant-ph/0610214v3>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_counts = [
            GateCount(resource_rep(qre.Hadamard), 2 * num_iter),
            Allocate(1),
        ]

        # Here we want to use this particular decomposition, not any random one the user might override
        gate_counts += ControlledSequence.resource_decomp(base_cmpr_op, num_iter)

        num_phase_gates = num_iter * (num_iter - 1) // 2
        gate_counts.append(
            GateCount(qre.PhaseShift.resource_rep(), num_phase_gates)
        )  # Classically controlled PS

        gate_counts.append(Deallocate(1))
        return gate_counts


class QFT(ResourceOperator):
    r"""Resource class for QFT.

    Args:
        num_wires (int | None): the number of qubits the operation acts upon
        wires (Sequence[int], None): the wires the operation acts on

    Resources:
        The resources are obtained from the standard decomposition of QFT as presented
        in (chapter 5) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum Information
        <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.QFT`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> qft = qre.QFT(3)
    >>> gate_set = {"SWAP", "Hadamard", "ControlledPhaseShift"}
    >>> print(qre.estimate(qft, gate_set))
    --- Resources: ---
    Total wires: 3
        algorithmic wires: 3
        allocated wires: 0
        zero state: 0
        any state: 0
    Total gates : 7
    'SWAP': 1,
    'ControlledPhaseShift': 3,
    'Hadamard': 3

    .. details::
        :title: Usage Details

        This operation provides an alternative decomposition method when an appropriately sized
        phase gradient state is available. This decomposition can be used as a custom decomposition
        using the operation's ``phase_grad_resource_decomp`` method and the
        :class:`~.pennylane.estimator.resource_config.ResourceConfig` class. See the
        following example for more details.

        >>> import pennylane.estimator as qre
        >>> config = qre.ResourceConfig()
        >>> config.set_decomp(qre.QFT, qre.QFT.phase_grad_resource_decomp)
        >>> print(qre.estimate(qre.QFT(3), config=config))
        --- Resources: ---
        Total wires: 4
            algorithmic wires: 3
            allocated wires: 1
            zero state: 1
            any state: 0
        Total gates : 17
        'Toffoli': 5,
        'CNOT': 6,
        'Hadamard': 6
    """

    resource_keys = {"num_wires"}

    def __init__(self, num_wires: int | None = None, wires: WiresLike = None) -> None:
        if num_wires is None:
            if wires is None:
                raise ValueError("Must provide atleast one of `num_wires` and `wires`.")
            num_wires = len(wires)
        self.num_wires = num_wires
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int): the number of qubits the operation acts upon
        """
        return {"num_wires": self.num_wires}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            num_wires (int): the number of qubits the operation acts upon

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        params = {"num_wires": num_wires}
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, num_wires) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_wires (int): the number of qubits the operation acts upon

        Resources:
            The resources are obtained from the standard decomposition of QFT as presented
            in (Chapter 5) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum Information
            <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        hadamard = resource_rep(qre.Hadamard)
        swap = resource_rep(qre.SWAP)
        ctrl_phase_shift = resource_rep(qre.ControlledPhaseShift)

        if num_wires == 1:
            return [
                GateCount(hadamard),
            ]

        return [
            GateCount(hadamard, num_wires),
            GateCount(swap, num_wires // 2),
            GateCount(ctrl_phase_shift, num_wires * (num_wires - 1) // 2),
        ]

    @classmethod
    def phase_grad_resource_decomp(cls, num_wires) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        .. note::

            This decomposition assumes an appropriately sized phase gradient state is available.
            Users should ensure the cost of constructing such a state has been accounted for.
            See also :class:`~.pennylane.estimator.templates.PhaseGradient`.

        Args:
            num_wires (int): the number of qubits the operation acts upon

        Resources:
            The resources are obtained as presented in the article
            `Turning Gradients into Additions into QFTs <https://algassert.com/post/1620>`_.
            Specifically, following the figure titled "8 qubit Quantum Fourier Transform with gradient shifts"

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        hadamard = resource_rep(qre.Hadamard)
        swap = resource_rep(qre.SWAP)

        if num_wires == 1:
            return [GateCount(hadamard)]

        gate_types = [
            GateCount(hadamard, num_wires),
            GateCount(swap, num_wires // 2),
        ]

        for size_reg in range(1, num_wires):
            ctrl_add = qre.Controlled.resource_rep(
                qre.SemiAdder.resource_rep(max_register_size=size_reg),
                num_ctrl_wires=1,
                num_zero_ctrl=0,
            )
            gate_types.append(GateCount(ctrl_add))

        return gate_types

    @staticmethod
    def tracking_name(num_wires) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return f"QFT({num_wires})"


class AQFT(ResourceOperator):
    r"""Resource class for the Approximate QFT.

    .. note::

        This operation assumes an appropriately sized phase gradient state is available.
        Users should ensure the cost of constructing such a state has been accounted for.
        See also :class:`~.pennylane.estimator.templates.PhaseGradient`.

    Args:
        order (int): the maximum number of controlled phase shifts per qubit to which the operation is truncated
        num_wires (int | None): the number of qubits the operation acts upon
        wires (Sequence[int], None): the wires the operation acts on

    Resources:
        The resources are obtained from (Fig. 4) of `arXiv:1803.04933, <https://arxiv.org/abs/1803.04933>`_
        excluding the allocation and instantiation of the phase gradient state. The phased :code:`Toffoli`
        gates and the classical measure-and-reset (Fig. 2) are accounted for as :code:`TemporaryAND`
        operations.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.AQFT`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> aqft = qre.AQFT(order=2, num_wires=3)
    >>> gate_set = {"SWAP", "Hadamard", "T", "CNOT"}
    >>> print(qre.estimate(aqft, gate_set))
    --- Resources: ---
    Total wires: 4
        algorithmic wires: 3
        allocated wires: 1
        zero state: 1
        any state: 0
    Total gates : 57
    'SWAP': 1,
    'T': 40,
    'CNOT': 9,
    'Hadamard': 7
    """

    resource_keys = {"order, num_wires"}

    def __init__(self, order: int, num_wires: int | None = None, wires: WiresLike = None) -> None:
        if num_wires is None:
            if wires is None:
                raise ValueError("Must provide atleast one of `num_wires` and `wires`.")
            num_wires = len(wires)
        self.order = order
        self.num_wires = num_wires

        if order < 1:
            raise ValueError("Order must be a positive integer greater than 0.")

        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * order (int): the maximum number of controlled phase shifts to which the operation is truncated
                * num_wires (int): the number of qubits the operation acts upon
        """
        return {"order": self.order, "num_wires": self.num_wires}

    @classmethod
    def resource_rep(cls, order, num_wires) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            order (int): the maximum number of controlled phase shifts to truncate
            num_wires (int): the number of qubits the operation acts upon

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        params = {"order": order, "num_wires": num_wires}
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, order, num_wires) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            order (int): the maximum number of controlled phase shifts to which the operation is truncated
            num_wires (int): the number of qubits the operation acts upon

        Resources:
            The resources are obtained from (Fig. 4) `arXiv:1803.04933 <https://arxiv.org/abs/1803.04933>`_
            excluding the allocation and instantiation of the phase gradient state. The phased Toffoli
            gates and the classical measure-and-reset (Fig. 2) are accounted for as :code:`TemporaryAND`
            operations.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        hadamard = resource_rep(qre.Hadamard)
        swap = resource_rep(qre.SWAP)
        cs = qre.Controlled.resource_rep(
            base_cmpr_op=resource_rep(qre.S),
            num_ctrl_wires=1,
            num_zero_ctrl=0,
        )

        if order >= num_wires:
            order = num_wires - 1

        gate_types = [
            GateCount(hadamard, num_wires),
        ]

        if order > 1 and num_wires > 1:
            gate_types.append(GateCount(cs, num_wires - 1))

            for index in range(2, order):
                addition_reg_size = index - 1

                temp_and = resource_rep(qre.TemporaryAND)
                temp_and_dag = qre.Adjoint.resource_rep(temp_and)
                in_place_add = qre.SemiAdder.resource_rep(addition_reg_size)

                cost_iter = [
                    Allocate(addition_reg_size),
                    GateCount(temp_and, addition_reg_size),
                    GateCount(in_place_add),
                    GateCount(hadamard),
                    GateCount(temp_and_dag, addition_reg_size),
                    Deallocate(addition_reg_size),
                ]
                gate_types.extend(cost_iter)

            addition_reg_size = order - 1
            repetitions = num_wires - order

            temp_and = resource_rep(qre.TemporaryAND)
            temp_and_dag = qre.Adjoint.resource_rep(temp_and)
            in_place_add = qre.SemiAdder.resource_rep(addition_reg_size)

            cost_iter = [
                Allocate(addition_reg_size),
                GateCount(temp_and, addition_reg_size * repetitions),
                GateCount(in_place_add, repetitions),
                GateCount(hadamard, repetitions),
                GateCount(temp_and_dag, addition_reg_size * repetitions),
                Deallocate(addition_reg_size),
            ]
            gate_types.extend(cost_iter)

            gate_types.append(GateCount(swap, num_wires // 2))

        return gate_types

    @staticmethod
    def tracking_name(order, num_wires) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return f"AQFT({order}, {num_wires})"


class BasisRotation(ResourceOperator):
    r"""Resource class for the BasisRotation gate.

    Args:
        dim (int | None): The dimensions of the input matrix specifying the basis transformation.
            This is equivalent to the number of rows or columns of the matrix.
        wires (Sequence[int], None): the wires the operation acts on, should be equal to the dimension

    Resources:
        The resources are obtained from the construction scheme given in `Optica, 3, 1460 (2016)
        <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_. Specifically,
        the resources are given as :math:`N \times (N - 1) / 2` instances of the
        ``SingleExcitation`` gate, and :math:`N \times (1 + (N - 1) / 2)`
        instances of the ``PhaseShift`` gate, where :math:`N` is the dimensions of the input matrix.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.BasisRotation`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> basis_rot = qre.BasisRotation(dim = 5)
    >>> print(qre.estimate(basis_rot))
    --- Resources: ---
    Total wires: 5
        algorithmic wires: 5
        allocated wires: 0
        zero state: 0
        any state: 0
    Total gates : 1.740E+3
    'T': 1.580E+3,
    'CNOT': 20,
    'Z': 40,
    'S': 60,
    'Hadamard': 40
    """

    resource_keys = {"dim"}

    def __init__(self, dim: int | None = None, wires: WiresLike = None):
        if dim is None:
            if wires is None:
                raise ValueError("Must provide atleast one of `dim` and `wires`.")
            dim = len(wires)
        self.num_wires = dim
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls, dim) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            dim (int): The dimensions of the input :code:`unitary_matrix`. This is computed
                as the number of columns of the matrix.

        Resources:
            The resources are obtained from the construction scheme given in `Optica, 3, 1460 (2016)
            <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_. Specifically,
            the resources are given as :math:`N * (N - 1) / 2` instances of the
            ``SingleExcitation`` gate, and :math:`N * (1 + (N - 1) / 2)` instances
            of the ``PhaseShift`` gate, where :math:`N` is the dimensions of the input matrix.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        phase_shift = resource_rep(qre.PhaseShift)
        single_excitation = resource_rep(qre.SingleExcitation)

        se_count = dim * (dim - 1) // 2
        ps_count = dim + se_count

        return [GateCount(phase_shift, ps_count), GateCount(single_excitation, se_count)]

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * dim (int): The dimensions of the input :code:`unitary_matrix`. This is computed as the number of columns of the matrix.

        """
        return {"dim": self.num_wires}

    @classmethod
    def resource_rep(cls, dim) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            dim (int): The dimensions of the input :code:`unitary_matrix`. This is computed
                as the number of columns of the matrix.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        params = {"dim": dim}
        num_wires = dim
        return CompressedResourceOp(cls, num_wires, params)

    @staticmethod
    def tracking_name(dim) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return f"BasisRotation({dim})"


class Select(ResourceOperator):
    r"""Resource class for the Select gate.

    Args:
        ops (list[:class:`~.pennylane.estimator.resource_operator.ResourceOperator`]): the set of operations to select over
        wires (Sequence[int], None): The wires the operation acts on. If :code:`ops`
            provide wire labels, then this is just the set of control wire labels. Otherwise, it
            also includes the target wire labels of the selected operators.

    Resources:
        The resources are based on the analysis in `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_ section III.A,
        'Unary Iteration and Indexed Operations'. See Figures 4, 6, and 7.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.Select`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> ops = [qre.X(), qre.Y(), qre.Z()]
    >>> select_op = qre.Select(ops=ops)
    >>> print(qre.estimate(select_op))
    --- Resources: ---
    Total wires: 4
        algorithmic wires: 3
        allocated wires: 1
        zero state: 1
        any state: 0
    Total gates : 24
    'Toffoli': 2,
    'CNOT': 7,
    'X': 4,
    'Z': 1,
    'S': 2,
    'Hadamard': 8
    """

    resource_keys = {"num_wires", "cmpr_ops"}

    def __init__(self, ops: list, wires: WiresLike = None) -> None:
        _dequeue(op_to_remove=ops)
        self.queue()
        num_select_ops = len(ops)
        num_ctrl_wires = math.ceil(math.log2(num_select_ops))

        try:
            cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)
            self.cmpr_ops = cmpr_ops
        except AttributeError as error:
            raise ValueError(
                "All factors of the Select must be instances of `ResourceOperator` in order to obtain resources."
            ) from error

        ops_wires = Wires.all_wires([op.wires for op in ops if op.wires is not None])
        fewest_unique_wires = max(op.num_wires for op in cmpr_ops)
        minimum_num_wires = max(fewest_unique_wires, len(ops_wires)) + num_ctrl_wires

        if wires:
            self.wires = Wires.all_wires([Wires(wires), ops_wires])
            if len(self.wires) < minimum_num_wires:
                raise ValueError(
                    f"Expected at least {minimum_num_wires} wires ({num_ctrl_wires} control + {fewest_unique_wires} target), got {len(Wires(wires))}."
                )
            self.num_wires = len(self.wires)
        else:
            self.wires = None
            self.num_wires = minimum_num_wires

    @classmethod
    def resource_decomp(cls, cmpr_ops, num_wires):  # pylint: disable=unused-argument
        r"""The resources for a select implementation taking advantage of the unary iterator trick.

        Args:
            cmpr_ops (list[:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`]): The list of operators, in the compressed
                representation, to be applied according to the selected qubits.
            num_wires (int): The number of wires the operation acts on. This is a sum of the
                control wires (:math:`\lceil(log_{2}(N))\rceil`) required and the number wires
                targeted by the :code:`ops`.

        Resources:
            The resources are based on the analysis in `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_ section III.A,
            'Unary Iteration and Indexed Operations'. See Figures 4, 6, and 7.

            Note: This implementation assumes we have access to :math:`n - 1` additional work qubits,
            where :math:`n = \left\lceil log_{2}(N) \right\rceil` and :math:`N` is the number of batches of unitaries
            to select.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_types = []
        x = qre.X.resource_rep()
        cnot = qre.CNOT.resource_rep()
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})

        num_ops = len(cmpr_ops)
        work_qubits = math.ceil(math.log2(num_ops)) - 1

        gate_types.append(Allocate(work_qubits))
        for cmp_rep in cmpr_ops:
            ctrl_op = qre.Controlled.resource_rep(cmp_rep, 1, 0)
            gate_types.append(GateCount(ctrl_op))

        gate_types.append(GateCount(x, 2 * (num_ops - 1)))  # conjugate 0 controlled toffolis
        gate_types.append(GateCount(cnot, num_ops - 1))
        gate_types.append(GateCount(l_elbow, num_ops - 1))
        gate_types.append(GateCount(r_elbow, num_ops - 1))

        gate_types.append(Deallocate(work_qubits))
        return gate_types

    @staticmethod
    def textbook_resources(cmpr_ops) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            cmpr_ops (list[:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`]): The list of operators, in the compressed
                representation, to be applied according to the selected qubits.
            num_wires (int): The number of wires the operation acts on. This is a sum of the
                control wires (:math:`\lceil(log_{2}(N))\rceil`) required and the number wires
                targeted by the :code:`ops`.

        Resources:
            The resources correspond directly to the definition of the operation. Specifically,
            for each operator in :code:`cmpr_ops`, the cost is given as a controlled version of the operator
            controlled on the associated bitstring.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_types = defaultdict(int)
        x = qre.X.resource_rep()

        num_ops = len(cmpr_ops)
        num_ctrl_wires = int(qnp.ceil(qnp.log2(num_ops)))
        num_total_ctrl_possibilities = 2**num_ctrl_wires  # 2^n

        num_zero_controls = num_total_ctrl_possibilities // 2
        gate_types[x] = num_zero_controls * 2  # conjugate 0 controls

        for cmp_rep in cmpr_ops:
            ctrl_op = qre.Controlled.resource_rep(
                cmp_rep,
                num_ctrl_wires,
                0,
            )
            gate_types[ctrl_op] += 1

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * cmpr_ops (list[:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`]): The list of operators, in the compressed representation, to be applied according to the selected qubits.
                * num_wires (int): The number of wires the operation acts on. This is a sum of the
                  control wires (:math:`\lceil(log_{2}(N))\rceil`) required and the number wires
                  targeted by the :code:`ops`.

        """
        return {"cmpr_ops": self.cmpr_ops, "num_wires": self.num_wires}

    @classmethod
    def resource_rep(cls, cmpr_ops, num_wires: WiresLike = None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            cmpr_ops (list[:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`]): The list of operators, in the compressed
                representation, to be applied according to the selected qubits.
            num_wires (int): An optional parameter representing the number of wires the operation
                acts on. This is a sum of the control wires (:math:`\lceil(log_{2}(N))\rceil`)
                required and the number of wires targeted by the :code:`ops`.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_ctrl_wires = math.ceil(math.log2(len(cmpr_ops)))
        fewest_unique_wires = max(op.num_wires for op in cmpr_ops)

        num_wires = num_wires or fewest_unique_wires + num_ctrl_wires
        params = {"cmpr_ops": cmpr_ops, "num_wires": num_wires}
        return CompressedResourceOp(cls, num_wires, params)


class QROM(ResourceOperator):
    r"""Resource class for the QROM template.

    Args:
        num_bitstrings (int): the number of bitstrings that are to be encoded
        size_bitstring (int): the length of each bitstring
        num_bit_flips (int, optional): The total number of :math:`1`'s in the dataset. Defaults to
            :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
        restored (bool, optional): Determine if allocated qubits should be reset after the computation
            (at the cost of higher gate counts). Defaults to :code:`True`.
        select_swap_depth (int | None): A parameter :math:`\lambda` that determines
            if data will be loaded in parallel by adding more rows following Figure 1.C of
            `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be :code:`None`,
            :code:`1` or a positive integer power of two. Defaults to :code:`None`, which internally
            determines the optimal depth.
        wires (Sequence[int], None): The wires the operation acts on (control and target).
            Excluding any additional qubits allocated during the decomposition (e.g select-swap wires).

    Resources:
        The resources for QROM are taken from the following two papers:
        `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_ (Figure 1.C) for
        :code:`restored = False` and `Berry et al. (2019) <https://arxiv.org/pdf/1902.02134>`_
        (Figure 4) for :code:`restored = True`.

    .. seealso:: The associated PennyLane operation :class:`~.pennylane.QROM`

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> qrom = qre.QROM(
    ...     num_bitstrings=10,
    ...     size_bitstring=4,
    ... )
    >>> print(qre.estimate(qrom))
    --- Resources: ---
    Total wires: 11
        algorithmic wires: 8
        allocated wires: 3
        zero state: 3
        any state: 0
    Total gates : 178
    'Toffoli': 16,
    'CNOT': 72,
    'X': 34,
    'Hadamard': 56
    """

    resource_keys = {
        "num_bitstrings",
        "size_bitstring",
        "num_bit_flips",
        "select_swap_depth",
        "restored",
    }

    @staticmethod
    def _t_optimized_select_swap_width(num_bitstrings, size_bitstring):
        opt_width_continuous = math.sqrt((2 / 3) * (num_bitstrings / size_bitstring))
        w1 = 2 ** math.floor(math.log2(opt_width_continuous))
        w2 = 2 ** math.ceil(math.log2(opt_width_continuous))

        w1 = 1 if w1 < 1 else w1
        w2 = 1 if w2 < 1 else w2  # The continuous solution could be non-physical

        def t_cost_func(w):
            return 4 * (math.ceil(num_bitstrings / w) - 2) + 6 * (w - 1) * size_bitstring

        if t_cost_func(w2) < t_cost_func(w1):
            return w2
        return w1

    def __init__(
        self,
        num_bitstrings: int,
        size_bitstring: int,
        num_bit_flips: int = None,
        restored: bool = True,
        select_swap_depth=None,
        wires: WiresLike = None,
    ) -> None:
        self.restored = restored
        self.num_bitstrings = num_bitstrings
        self.size_bitstring = size_bitstring
        self.num_bit_flips = num_bit_flips or (num_bitstrings * size_bitstring // 2)

        self.num_control_wires = math.ceil(math.log2(num_bitstrings))
        self.num_wires = size_bitstring + self.num_control_wires

        if select_swap_depth is not None:
            if not isinstance(select_swap_depth, int):
                raise ValueError(
                    f"`select_swap_depth` must be None or an integer. Got {type(select_swap_depth)}"
                )

            exponent = int(math.log2(select_swap_depth))
            if 2**exponent != select_swap_depth:
                raise ValueError(
                    f"`select_swap_depth` must be 1 or a positive integer power of 2. Got {select_swap_depth}"
                )

        self.select_swap_depth = select_swap_depth
        super().__init__(wires=wires)

    # pylint: disable=protected-access
    @classmethod
    def resource_decomp(
        cls,
        num_bitstrings,
        size_bitstring,
        num_bit_flips,
        select_swap_depth=None,
        restored=True,
    ) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the operator's resources.

        Args:
            num_bitstrings (int): the number of bitstrings that are to be encoded
            size_bitstring (int): the length of each bitstring
            num_bit_flips (int, optional): The total number of :math:`1`'s in the dataset. Defaults to
                :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
            select_swap_depth (int | None): A parameter :math:`\lambda` that determines
                if data will be loaded in parallel by adding more rows following Figure 1.C of
                `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be :code:`None`,
                :code:`1` or a positive integer power of two. Defaults to :code:`None`, which internally
                determines the optimal depth.
            restored (bool, optional): Determine if allocated qubits should be reset after the computation
                (at the cost of higher gate counts). Defaults to :code`True`.

        Resources:
            The resources for QROM are taken from the following two papers:
            `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_ (Figure 1.C) for
            :code:`restored = False` and `Berry et al. (2019) <https://arxiv.org/pdf/1902.02134>`_
            (Figure 4) for :code:`restored = True`.

            Note: we use the unary iterator trick to implement the Select. This
            implementation assumes we have access to :math:`n - 1` additional
            work qubits, where :math:`n = \left\lceil log_{2}(N) \right\rceil` and :math:`N` is
            the number of batches of unitaries to select.
        """

        if select_swap_depth:
            max_depth = 2 ** math.ceil(math.log2(num_bitstrings))
            select_swap_depth = min(max_depth, select_swap_depth)  # truncate depth beyond max depth

        W_opt = select_swap_depth or cls._t_optimized_select_swap_width(
            num_bitstrings, size_bitstring
        )
        L_opt = math.ceil(num_bitstrings / W_opt)
        l = math.ceil(math.log2(L_opt))

        gate_cost = []
        num_alloc_wires = (W_opt - 1) * size_bitstring  # Swap registers
        if L_opt > 1:
            num_alloc_wires += l - 1  # + work_wires for UI trick

        gate_cost.append(Allocate(num_alloc_wires))

        x = resource_rep(qre.X)
        cnot = resource_rep(qre.CNOT)
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})
        hadamard = resource_rep(qre.Hadamard)

        swap_restored_prefactor = 1
        select_restored_prefactor = 1

        if restored:
            gate_cost.append(GateCount(hadamard, 2 * size_bitstring))
            swap_restored_prefactor = 4
            select_restored_prefactor = 2

        # SELECT cost:
        if L_opt > 1:
            gate_cost.append(
                GateCount(x, select_restored_prefactor * (2 * (L_opt - 2) + 1))
            )  # conjugate 0 controlled toffolis + 1 extra X gate from un-controlled unary iterator decomp
            gate_cost.append(
                GateCount(
                    cnot,
                    select_restored_prefactor * (L_opt - 2)
                    + select_restored_prefactor * num_bit_flips,
                )  # num CNOTs in unary iterator trick   +   each unitary in the select is just a CNOT
            )
            gate_cost.append(GateCount(l_elbow, select_restored_prefactor * (L_opt - 2)))
            gate_cost.append(GateCount(r_elbow, select_restored_prefactor * (L_opt - 2)))

            gate_cost.append(Deallocate(l - 1))  # release UI trick work wires

        else:
            gate_cost.append(
                GateCount(
                    x, select_restored_prefactor * num_bit_flips
                )  # each unitary in the select is just an X gate to load the data
            )

        # SWAP cost:
        ctrl_swap = resource_rep(qre.CSWAP)
        gate_cost.append(
            GateCount(ctrl_swap, swap_restored_prefactor * (W_opt - 1) * size_bitstring)
        )

        if restored:
            gate_cost.append(Deallocate((W_opt - 1) * size_bitstring))  # release Swap registers

        return gate_cost

    @classmethod
    def single_controlled_res_decomp(
        cls,
        num_bitstrings,
        size_bitstring,
        num_bit_flips,
        select_swap_depth,
        restored,
    ):
        r"""The resource decomposition for QROM controlled on a single wire."""
        if select_swap_depth:
            max_depth = 2 ** math.ceil(math.log2(num_bitstrings))
            select_swap_depth = min(max_depth, select_swap_depth)  # truncate depth beyond max depth

        W_opt = select_swap_depth or qre.QROM._t_optimized_select_swap_width(
            num_bitstrings, size_bitstring
        )
        L_opt = math.ceil(num_bitstrings / W_opt)
        l = math.ceil(math.log2(L_opt))

        gate_cost = []
        num_alloc_wires = (W_opt - 1) * size_bitstring  # Swap registers
        if L_opt > 1:
            num_alloc_wires += l  # + work_wires for UI trick

        gate_cost.append(Allocate(num_alloc_wires))

        x = resource_rep(qre.X)
        cnot = resource_rep(qre.CNOT)
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})
        hadamard = resource_rep(qre.Hadamard)

        swap_restored_prefactor = 1
        select_restored_prefactor = 1

        if restored:
            gate_cost.append(GateCount(hadamard, 2 * size_bitstring))
            swap_restored_prefactor = 4
            select_restored_prefactor = 2

        # SELECT cost:
        if L_opt > 1:
            gate_cost.append(
                GateCount(x, select_restored_prefactor * (2 * (L_opt - 1)))
            )  # conjugate 0 controlled toffolis
            gate_cost.append(
                GateCount(
                    cnot,
                    select_restored_prefactor * (L_opt - 1)
                    + select_restored_prefactor * num_bit_flips,
                )  # num CNOTs in unary iterator trick   +   each unitary in the select is just a CNOT
            )
            gate_cost.append(GateCount(l_elbow, select_restored_prefactor * (L_opt - 1)))
            gate_cost.append(GateCount(r_elbow, select_restored_prefactor * (L_opt - 1)))

            gate_cost.append(Deallocate(l))  # release UI trick work wires
        else:
            gate_cost.append(
                GateCount(
                    x,
                    select_restored_prefactor * num_bit_flips,
                )  #  each unitary in the select is just an X
            )

        # SWAP cost:
        w = math.ceil(math.log2(W_opt))
        ctrl_swap = qre.CSWAP.resource_rep()
        gate_cost.append(Allocate(1))  # need one temporary qubit for l/r-elbow to control SWAP

        gate_cost.append(GateCount(l_elbow, w))
        gate_cost.append(
            GateCount(ctrl_swap, swap_restored_prefactor * (W_opt - 1) * size_bitstring)
        )
        gate_cost.append(GateCount(r_elbow, w))

        gate_cost.append(Deallocate(1))  # temp wires
        if restored:
            gate_cost.append(
                Deallocate((W_opt - 1) * size_bitstring)
            )  # release Swap registers + temp wires
        return gate_cost

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires: int, num_zero_ctrl: int, target_resource_params: dict
    ):
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            The resources for QROM are taken from the following two papers:
            `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_ (Figure 1.C) for
            :code:`restored = False` and `Berry et al. (2019) <https://arxiv.org/pdf/1902.02134>`_
            (Figure 4) for :code:`restored = True`.

            Note: we use the single-controlled unary iterator trick to implement the Select. This
            implementation assumes we have access to :math:`n - 1` additional work qubits,
            where :math:`n = \ceil{log_{2}(N)}` and :math:`N` is the number of batches of
            unitaries to select.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        num_bitstrings = target_resource_params["num_bitstrings"]
        size_bitstring = target_resource_params["size_bitstring"]
        num_bit_flips = target_resource_params.get("num_bit_flips", None)
        select_swap_depth = target_resource_params.get("select_swap_depth", None)
        restored = target_resource_params.get("restored", True)
        gate_cost = []
        if num_zero_ctrl:
            x = qre.X.resource_rep()
            gate_cost.append(GateCount(x, 2 * num_zero_ctrl))

        if num_bit_flips is None:
            num_bit_flips = (num_bitstrings * size_bitstring) // 2

        single_ctrl_cost = cls.single_controlled_res_decomp(
            num_bitstrings,
            size_bitstring,
            num_bit_flips,
            select_swap_depth,
            restored,
        )

        if num_ctrl_wires == 1:
            gate_cost.extend(single_ctrl_cost)
            return gate_cost

        gate_cost.append(Allocate(1))
        gate_cost.append(GateCount(qre.MultiControlledX.resource_rep(num_ctrl_wires, 0)))
        gate_cost.extend(single_ctrl_cost)
        gate_cost.append(GateCount(qre.MultiControlledX.resource_rep(num_ctrl_wires, 0)))
        gate_cost.append(Deallocate(1))
        return gate_cost

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_bitstrings (int): the number of bitstrings that are to be encoded
                * size_bitstring (int): the length of each bitstring
                * num_bit_flips (int, optional): The total number of :math:`1`'s in the dataset.
                  Defaults to :code:`(num_bitstrings * size_bitstring) // 2`, which is half the
                  dataset.
                * restored (bool, optional): Determine if allocated qubits should be reset after the
                  computation (at the cost of higher gate counts). Defaults to :code`True`.
                * select_swap_depth (int | None): A parameter :math:`\lambda` that
                  determines if data will be loaded in parallel by adding more rows following
                  Figure 1.C of `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be
                  :code:`None`, :code:`1` or a positive integer power of two. Defaults to :code:`None`,
                  which internally determines the optimal depth.

        """

        return {
            "num_bitstrings": self.num_bitstrings,
            "size_bitstring": self.size_bitstring,
            "num_bit_flips": self.num_bit_flips,
            "select_swap_depth": self.select_swap_depth,
            "restored": self.restored,
        }

    @classmethod
    def resource_rep(
        cls,
        num_bitstrings,
        size_bitstring,
        num_bit_flips=None,
        restored=True,
        select_swap_depth=None,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_bitstrings (int): the number of bitstrings that are to be encoded
            size_bitstring (int): the length of each bitstring
            num_bit_flips (int, optional): The total number of :math:`1`'s in the dataset. Defaults to
                :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
            restored (bool, optional): Determine if allocated qubits should be reset after the computation
                (at the cost of higher gate counts). Defaults to :code`True`.
            select_swap_depth (int | None): A parameter :math:`\lambda` that determines
                if data will be loaded in parallel by adding more rows following Figure 1.C of
                `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be :code:`None`,
                :code:`1` or a positive integer power of two. Defaults to :code:`None`, which internally
                determines the optimal depth.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        if num_bit_flips is None:
            num_bit_flips = num_bitstrings * size_bitstring // 2

        if select_swap_depth is not None:
            if not isinstance(select_swap_depth, int):
                raise ValueError(
                    f"`select_swap_depth` must be None or an integer. Got {type(select_swap_depth)}"
                )

            exponent = int(math.log2(select_swap_depth))
            if 2**exponent != select_swap_depth:
                raise ValueError(
                    f"`select_swap_depth` must be 1 or a positive integer power of 2. Got f{select_swap_depth}"
                )

        params = {
            "num_bitstrings": num_bitstrings,
            "num_bit_flips": num_bit_flips,
            "size_bitstring": size_bitstring,
            "select_swap_depth": select_swap_depth,
            "restored": restored,
        }
        num_wires = size_bitstring + math.ceil(math.log2(num_bitstrings))
        return CompressedResourceOp(cls, num_wires, params)


class SelectPauliRot(ResourceOperator):
    r"""Resource class for the SelectPauliRot gate.

    Args:
        rot_axis (str): the rotation axis used in the multiplexer
        num_ctrl_wires (int): the number of control wires of the multiplexer
        precision (float | None): the precision used in the single qubit rotations
        wires (Sequence[int], None): the wires the operation acts on

    Resources:
        The resources are obtained from the construction scheme given in `Möttönen and Vartiainen
        (2005), Fig 7a <https://arxiv.org/abs/quant-ph/0504100>`_. Specifically, the resources
        for an :math:`n` qubit unitary are given as :math:`2^{n}` instances of the :code:`CNOT`
        gate and :math:`2^{n}` instances of the single qubit rotation gate (:code:`RX`,
        :code:`RY` or :code:`RZ`) depending on the :code:`rot_axis`.

    .. seealso:: The associated PennyLane operation :class:`~.pennylane.SelectPauliRot`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> mltplxr = qre.SelectPauliRot(
    ...     rot_axis = "Y",
    ...     num_ctrl_wires = 4,
    ...     precision = 1e-3,
    ... )
    >>> print(qre.estimate(mltplxr, gate_set=['RY','CNOT']))
    --- Resources: ---
    Total wires: 5
        algorithmic wires: 5
        allocated wires: 0
        zero state: 0
        any state: 0
    Total gates : 32
    'RY': 16,
    'CNOT': 16
    """

    resource_keys = {"num_ctrl_wires", "rot_axis", "precision"}

    def __init__(
        self,
        rot_axis: str,
        num_ctrl_wires: int,
        precision: float | None = None,
        wires: WiresLike = None,
    ) -> None:
        if rot_axis not in ("X", "Y", "Z"):
            raise ValueError("The `rot_axis` argument must be one of ('X', 'Y', 'Z')")

        self.num_ctrl_wires = num_ctrl_wires
        self.rot_axis = rot_axis
        self.precision = precision

        self.num_wires = num_ctrl_wires + 1
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * rot_axis (str): the rotation axis used in the multiplexer
                * num_ctrl_wires (int): the number of control wires of the multiplexer
                * precision (float): the precision used in the single qubit rotations
        """
        return {
            "num_ctrl_wires": self.num_ctrl_wires,
            "rot_axis": self.rot_axis,
            "precision": self.precision,
        }

    @classmethod
    def resource_rep(cls, num_ctrl_wires, rot_axis, precision=None):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            rot_axis (str): the rotation axis used in the multiplexer
            num_ctrl_wires (int): the number of control wires of the multiplexer
            precision (float | None): the precision used in the single qubit rotations

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = num_ctrl_wires + 1
        return CompressedResourceOp(
            cls,
            num_wires,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "rot_axis": rot_axis,
                "precision": precision,
            },
        )

    @classmethod
    def resource_decomp(cls, num_ctrl_wires, rot_axis, precision):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            rot_axis (str): the rotation axis used in the multiplexer
            num_ctrl_wires (int): the number of control wires of the multiplexer
            precision (float): the precision used in the single qubit rotations

        Resources:
            The resources are obtained from the construction scheme given in `Möttönen and Vartiainen
            (2005), Fig 7a <https://arxiv.org/abs/quant-ph/0504100>`_. Specifically, the resources
            for an :math:`n` qubit unitary are given as :math:`2^{n}` instances of the :code:`CNOT`
            gate and :math:`2^{n}` instances of the single qubit rotation gate (:code:`RX`,
            :code:`RY` or :code:`RZ`) depending on the :code:`rot_axis`.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        rotation_gate_map = {
            "X": qre.RX,
            "Y": qre.RY,
            "Z": qre.RZ,
        }

        gate = resource_rep(rotation_gate_map[rot_axis], {"precision": precision})
        cnot = resource_rep(qre.CNOT)

        gate_lst = [
            GateCount(gate, 2**num_ctrl_wires),
            GateCount(cnot, 2**num_ctrl_wires),
        ]

        return gate_lst

    @classmethod
    def phase_grad_resource_decomp(cls, num_ctrl_wires, rot_axis, precision):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            rot_axis (str): the rotation axis used in the multiplexer
            num_ctrl_wires (int): the number of control wires of the multiplexer
            precision (float): the precision used in the single qubit rotations

        Resources:
            The resources are obtained from the construction scheme given in `O'Brien and Sünderhauf
            (2025), Fig 4 <https://arxiv.org/pdf/2409.07332>`_. Specifically, the resources
            use two :class:`~.pennylane.estimator.templates.subroutines.QROM`s to digitally load and unload
            the phase angles up to some precision. These are then applied using a single controlled
            :class:`~.pennylane.estimator.templates.subroutines.SemiAdder`.

            .. note::

                This method assumes a phase gradient state is prepared on an auxiliary register.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        num_prec_wires = math.ceil(math.log2(math.pi / precision)) + 1
        gate_lst = []

        qrom = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": 2**num_ctrl_wires,
                "num_bit_flips": 2**num_ctrl_wires * num_prec_wires // 2,
                "size_bitstring": num_prec_wires,
                "restored": False,
            },
        )

        gate_lst.append(Allocate(num_prec_wires))
        gate_lst.append(GateCount(qrom))
        gate_lst.append(
            GateCount(
                resource_rep(
                    qre.Controlled,
                    {
                        "base_cmpr_op": resource_rep(
                            qre.SemiAdder,
                            {"max_register_size": num_prec_wires},
                        ),
                        "num_ctrl_wires": 1,
                        "num_zero_ctrl": 0,
                    },
                )
            )
        )
        gate_lst.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": qrom})))
        gate_lst.append(Deallocate(num_prec_wires))

        h = resource_rep(qre.Hadamard)
        s = resource_rep(qre.S)
        s_dagg = resource_rep(qre.Adjoint, {"base_cmpr_op": s})

        if rot_axis == "X":
            gate_lst.append(GateCount(h, 2))
        if rot_axis == "Y":
            gate_lst.append(GateCount(h, 2))
            gate_lst.append(GateCount(s))
            gate_lst.append(GateCount(s_dagg))

        return gate_lst
