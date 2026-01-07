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

from pennylane import numpy as qnp
from pennylane.labs import resource_estimation as re
from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.wires import Wires

# pylint: disable=arguments-differ,too-many-arguments,unused-argument,super-init-not-called


class ResourceOutOfPlaceSquare(ResourceOperator):
    r"""Resource class for the OutofPlaceSquare gate.

    This operation takes two quantum registers. The input register is of size :code:`register_size`
    and the output register of size :code:`2 * register_size`. The number encoded in the input is
    squared and returned in the output register.

    Args:
        register_size (int): the size of the input register
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from appendix G, lemma 7 in `PRX Quantum, 2, 040332 (2021)
        <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332>`_. Specifically,
        the resources are given as :math:`(n - 1)^2` Toffoli gates, and :math:`n` CNOT gates.

    **Example**

    The resources for this operation are computed using:

    >>> out_square = plre.ResourceOutOfPlaceSquare(register_size=3)
    >>> print(plre.estimate(out_square))
    --- Resources: ---
    Total qubits: 9
    Total gates : 7
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 9
    Gate breakdown:
     {'Toffoli': 4, 'CNOT': 3}
    """

    resource_keys = {"register_size"}

    def __init__(self, register_size: int, wires=None):
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
    def resource_rep(cls, register_size):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            register_size (int): the size of the input register

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        num_wires = 3 * register_size
        return CompressedResourceOp(cls, num_wires, {"register_size": register_size})

    @classmethod
    def resource_decomp(cls, register_size, **kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            register_size (int): the size of the input register

        Resources:
            The resources are obtained from appendix G, lemma 7 in `PRX Quantum, 2, 040332 (2021)
            <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332>`_. Specifically,
            the resources are given as :math:`(n - 1)^2` Toffoli gates, and :math:`n` CNOT gates.

        Returns:
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_lst = []

        gate_lst.append(GateCount(resource_rep(re.ResourceToffoli), (register_size - 1) ** 2))
        gate_lst.append(GateCount(resource_rep(re.ResourceCNOT), register_size))

        return gate_lst


class ResourcePhaseGradient(ResourceOperator):
    r"""Resource class for the PhaseGradient gate.

    This operation prepares the phase gradient state
    :math:`\frac{1}{\sqrt{2^b}} \cdot \sum_{k=0}^{2^b - 1} e^{-i2\pi \frac{k}{2^b}}\ket{k}`.

    Args:
        num_wires (int): the number of qubits to prepare in the phase gradient state
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The phase gradient state is defined as an equal superposition of phaseshifts where each shift
        is progressively more precise. This is achieved by applying Hadamard gates to each qubit and
        then applying RZ-rotations to each qubit with progressively smaller rotation angle. The first
        three rotations can be compiled to a Z-gate, S-gate and a T-gate.

    **Example**

    The resources for this operation are computed using:

    >>> phase_grad = plre.ResourcePhaseGradient(num_wires=5)
    >>> gate_set={"Z", "S", "T", "RZ", "Hadamard"}
    >>> print(plre.estimate(phase_grad, gate_set))
    --- Resources: ---
    Total qubits: 5
    Total gates : 10
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 5
    Gate breakdown:
     {'Hadamard': 5, 'Z': 1, 'S': 1, 'T': 1, 'RZ': 2}
    """

    resource_keys = {"num_wires"}

    def __init__(self, num_wires, wires=None):
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
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, num_wires, {"num_wires": num_wires})

    @classmethod
    def resource_decomp(cls, num_wires, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            num_wires (int): the number of qubits to prepare in the phase gradient state

        Resources:
            The resources are obtained by construction. The phase gradient state is defined as an
            equal superposition of phaseshifts where each shift is progressively more precise. This
            is achieved by applying Hadamard gates to each qubit and then applying RZ-rotations to each
            qubit with progressively smaller rotation angle. The first three rotations can be compiled to
            a Z-gate, S-gate and a T-gate.

        Returns:
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_counts = [GateCount(resource_rep(re.ResourceHadamard), num_wires)]
        if num_wires > 0:
            gate_counts.append(GateCount(resource_rep(re.ResourceZ)))

        if num_wires > 1:
            gate_counts.append(GateCount(resource_rep(re.ResourceS)))

        if num_wires > 2:
            gate_counts.append(GateCount(resource_rep(re.ResourceT)))

        if num_wires > 3:
            gate_counts.append(GateCount(resource_rep(re.ResourceRZ), num_wires - 3))

        return gate_counts


class ResourceOutMultiplier(ResourceOperator):
    r"""Resource class for the OutMultiplier gate.

    Args:
        a_num_qubits (int): the size of the first input register
        b_num_qubits (int): the size of the second input register
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from appendix G, lemma 10 in `PRX Quantum, 2, 040332 (2021)
        <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332>`_.

    .. seealso:: :class:`~.OutMultiplier`

    **Example**

    The resources for this operation are computed using:

    >>> out_mul = plre.ResourceOutMultiplier(4, 4)
    >>> print(plre.estimate(out_mul))
    --- Resources: ---
    Total qubits: 16
    Total gates : 70
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 16
    Gate breakdown:
     {'Toffoli': 14, 'Hadamard': 42, 'CNOT': 14}
    """

    resource_keys = {"a_num_qubits", "b_num_qubits"}

    def __init__(self, a_num_qubits, b_num_qubits, wires=None) -> None:
        self.num_wires = a_num_qubits + b_num_qubits + 2 * max((a_num_qubits, b_num_qubits))
        self.a_num_qubits = a_num_qubits
        self.b_num_qubits = b_num_qubits
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * a_num_qubits (int): the size of the first input register
                * b_num_qubits (int): the size of the second input register
        """
        return {"a_num_qubits": self.a_num_qubits, "b_num_qubits": self.b_num_qubits}

    @classmethod
    def resource_rep(cls, a_num_qubits, b_num_qubits) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            a_num_qubits (int): the size of the first input register
            b_num_qubits (int): the size of the second input register

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        num_wires = a_num_qubits + b_num_qubits + 2 * max((a_num_qubits, b_num_qubits))
        return CompressedResourceOp(
            cls, num_wires, {"a_num_qubits": a_num_qubits, "b_num_qubits": b_num_qubits}
        )

    @classmethod
    def resource_decomp(cls, a_num_qubits, b_num_qubits, **kwargs) -> list[GateCount]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            a_num_qubits (int): the size of the first input register
            b_num_qubits (int): the size of the second input register
            wires (Sequence[int], optional): the wires the operation acts on

        Resources:
            The resources are obtained from appendix G, lemma 10 in `PRX Quantum, 2, 040332 (2021)
            <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332>`_.

        Returns:
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        l = max(a_num_qubits, b_num_qubits)

        toff = resource_rep(re.ResourceToffoli)
        l_elbow = resource_rep(re.ResourceTempAND)
        r_elbow = resource_rep(re.ResourceAdjoint, {"base_cmpr_op": l_elbow})

        toff_count = 2 * a_num_qubits * b_num_qubits - l
        elbow_count = toff_count // 2
        toff_count = toff_count - (elbow_count * 2)

        gate_lst = [
            GateCount(l_elbow, elbow_count),
            GateCount(r_elbow, elbow_count),
        ]

        if toff_count:
            gate_lst.append(GateCount(toff))
        return gate_lst


class ResourceSemiAdder(ResourceOperator):
    r"""Resource class for the SemiOutAdder gate.

    Args:
        max_register_size (int): the size of the larger of the two registers being added together
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from figures 1 and 2 in `Gidney (2018)
        <https://quantum-journal.org/papers/q-2018-06-18-74/pdf/>`_.

    .. seealso:: :class:`~.SemiAdder`

    **Example**

    The resources for this operation are computed using:

    >>> semi_add = plre.ResourceSemiAdder(max_register_size=4)
    >>> print(plre.estimate(semi_add))
    --- Resources: ---
    Total qubits: 11
    Total gates : 30
    Qubit breakdown:
     clean qubits: 3, dirty qubits: 0, algorithmic qubits: 8
    Gate breakdown:
     {'CNOT': 18, 'Toffoli': 3, 'Hadamard': 9}
    """

    resource_keys = {"max_register_size"}

    def __init__(self, max_register_size, wires=None):
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
            CompressedResourceOp: the operator in a compressed representation
        """
        num_wires = 2 * max_register_size
        return CompressedResourceOp(cls, num_wires, {"max_register_size": max_register_size})

    @classmethod
    def resource_decomp(cls, max_register_size, **kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            max_register_size (int): the size of the larger of the two registers being added together

        Resources:
            The resources are obtained from figures 1 and 2 in `Gidney (2018)
            <https://quantum-journal.org/papers/q-2018-06-18-74/pdf/>`_.

        Returns:
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        cnot = resource_rep(re.ResourceCNOT)
        if max_register_size == 1:
            return [GateCount(cnot)]

        x = resource_rep(re.ResourceX)
        toff = resource_rep(re.ResourceToffoli)
        if max_register_size == 2:
            return [GateCount(cnot, 2), GateCount(x, 2), GateCount(toff)]

        cnot_count = (6 * (max_register_size - 2)) + 3
        elbow_count = max_register_size - 1

        l_elbow = resource_rep(re.ResourceTempAND)
        r_elbow = resource_rep(re.ResourceAdjoint, {"base_cmpr_op": l_elbow})
        return [
            AllocWires(max_register_size - 1),
            GateCount(cnot, cnot_count),
            GateCount(l_elbow, elbow_count),
            GateCount(r_elbow, elbow_count),
            FreeWires(max_register_size - 1),
        ]  # Obtained resource from Fig1 and Fig2 https://quantum-journal.org/papers/q-2018-06-18-74/pdf/

    @classmethod
    def controlled_resource_decomp(
        cls, ctrl_num_ctrl_wires, ctrl_num_ctrl_values, max_register_size, **kwargs
    ):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            max_register_size (int): the size of the larger of the two registers being added together

        Resources:
            The resources are obtained from figure 4a in `Gidney (2018)
            <https://quantum-journal.org/papers/q-2018-06-18-74/pdf/>`_.

        Returns:
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if max_register_size > 2:
            gate_lst = []

            if ctrl_num_ctrl_wires > 1:
                mcx = resource_rep(
                    re.ResourceMultiControlledX,
                    {
                        "num_ctrl_wires": ctrl_num_ctrl_wires,
                        "num_ctrl_values": ctrl_num_ctrl_values,
                    },
                )
                gate_lst.append(AllocWires(1))
                gate_lst.append(GateCount(mcx, 2))

            cnot_count = (7 * (max_register_size - 2)) + 3
            elbow_count = 2 * (max_register_size - 1)

            x = resource_rep(re.ResourceX)
            cnot = resource_rep(re.ResourceCNOT)
            l_elbow = resource_rep(re.ResourceTempAND)
            r_elbow = resource_rep(re.ResourceAdjoint, {"base_cmpr_op": l_elbow})
            gate_lst.extend(
                [
                    AllocWires(max_register_size - 1),
                    GateCount(cnot, cnot_count),
                    GateCount(l_elbow, elbow_count),
                    GateCount(r_elbow, elbow_count),
                    FreeWires(max_register_size - 1),
                ],
            )

            if ctrl_num_ctrl_wires > 1:
                gate_lst.append(FreeWires(1))
            elif ctrl_num_ctrl_values > 0:
                gate_lst.append(GateCount(x, 2 * ctrl_num_ctrl_values))

            return gate_lst  # Obtained resource from Fig 4a https://quantum-journal.org/papers/q-2018-06-18-74/pdf/

        raise re.ResourcesNotDefined


class ResourceControlledSequence(ResourceOperator):
    r"""Resource class for the ControlledSequence gate.

    This operator represents a sequence of controlled gates, one for each control wire, with the
    base operator (:code:`base`) raised to decreasing powers of 2.

    Args:
        base (~.pennylane.labs.resource_estimation.ResourceOperator): The operator that we
            will be applying controlled powers of.
        num_control_wires (int): the number of controlled wires to run the sequence over
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained as a direct result of the definition of the operator:

        .. code-block:: bash

            0: ──╭●───────────────┤
            1: ──│────╭●──────────┤
            2: ──│────│────╭●─────┤
            t: ──╰U⁴──╰U²──╰U¹────┤

    .. seealso:: :class:`~.ControlledSequence`

    **Example**

    The resources for this operation are computed using:

    >>> ctrl_seq = plre.ResourceControlledSequence(
    ...     base = plre.ResourceRX(),
    ...     num_control_wires = 3,
    ... )
    >>> gate_set={"CRX"}
    >>> print(plre.estimate(ctrl_seq, gate_set))
    --- Resources: ---
     Total qubits: 4
     Total gates : 3
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 4
     Gate breakdown:
      {'CRX': 3}
    """

    resource_keys = {"base_cmpr_op", "num_ctrl_wires"}

    def __init__(self, base: ResourceOperator, num_control_wires, wires=None) -> None:
        self.dequeue(op_to_remove=base)
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
                raise ValueError(f"Expected {self.num_wires} wires, got {wires}.")
        else:
            self.wires = None

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * base_cmpr_op (CompressedResourceOp): A compressed resource operator, corresponding
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
            base_cmpr_op (CompressedResourceOp): A compressed resource operator, corresponding
                to the operator that we will be applying controlled powers of.
            num_ctrl_wires (int): the number of controlled wires to run the sequence over

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"base_cmpr_op": base_cmpr_op, "num_ctrl_wires": num_ctrl_wires}
        num_wires = num_ctrl_wires + base_cmpr_op.num_wires
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, base_cmpr_op, num_ctrl_wires, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            base_cmpr_op (CompressedResourceOp): A compressed resource operator, corresponding
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
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_counts = []
        base_op = base_cmpr_op

        if base_cmpr_op.op_type == re.ResourceChangeBasisOp:
            base_op = base_cmpr_op.params["cmpr_base_op"]
            compute_op = base_cmpr_op.params["cmpr_compute_op"]
            uncompute_op = base_cmpr_op.params["cmpr_uncompute_op"]

            gate_counts.append(GateCount(compute_op))

        for z in range(num_ctrl_wires):
            ctrl_pow_u = re.ResourceControlled.resource_rep(
                re.ResourcePow.resource_rep(base_op, 2**z),
                num_ctrl_wires=1,
                num_ctrl_values=0,
            )
            gate_counts.append(GateCount(ctrl_pow_u))

        if base_cmpr_op.op_type == re.ResourceChangeBasisOp:
            gate_counts.append(GateCount(uncompute_op))

        return gate_counts


class ResourceQPE(ResourceOperator):
    r"""Resource class for QuantumPhaseEstimation (QPE).

    Args:
        base (~.pennylane.labs.resource_estimation.ResourceOperator): the phase estimation operator
        num_estimation_wires (int): the number of wires used for measuring out the phase
        adj_qft_op (Union[~.pennylane.labs.resource_estimation.ResourceOperator, None]): An optional
            argument to set the subroutine used to perform the adjoint QFT operation.
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from the standard decomposition of QPE as presented
        in (Section 5.2) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum
        Information <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.

    .. seealso:: :class:`~.QuantumPhaseEstimation`

    **Example**

    The resources for this operation are computed using:

    >>> gate_set = {"Hadamard", "Adjoint(QFT(5))", "CRX"}
    >>> qpe = plre.ResourceQPE(plre.ResourceRX(precision=1e-3), 5)
    >>> print(plre.estimate(qpe, gate_set))
    --- Resources: ---
     Total qubits: 6
     Total gates : 11
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 6
     Gate breakdown:
      {'Hadamard': 5, 'CRX': 5, 'Adjoint(QFT(5))': 1}

    .. details::
        :title: Usage Details

        Additionally, we can customize the implementation of the QFT operator we wish to use within
        the textbook QPE algorithm. This allows users to optimize the implementation of QPE by using
        more efficient implementations of the QFT.

        For example, consider the cost using the default QFT implmentation below:

        >>> qpe = plre.ResourceQPE(plre.ResourceRX(precision=1e-3), 5, adj_qft_op=None)
        >>> print(plre.estimate(qpe))
        --- Resources: ---
         Total qubits: 6
         Total gates : 1.586E+3
         Qubit breakdown:
          clean qubits: 0, dirty qubits: 0, algorithmic qubits: 6
         Gate breakdown:
          {'Hadamard': 20, 'CNOT': 36, 'T': 1.530E+3}

        Now we use the :class:`~.pennylane.labs.resource_estimation.ResourceAQFT` class:

        >>> aqft = plre.ResourceAQFT(order=3, num_wires=5)
        >>> adj_aqft = plre.ResourceAdjoint(aqft)
        >>> qpe = plre.ResourceQPE(plre.ResourceRX(precision=1e-3), 5, adj_qft_op=adj_aqft)
        >>> print(plre.estimate(qpe))
        --- Resources: ---
         Total qubits: 8
         Total gates : 321
         Qubit breakdown:
          clean qubits: 2, dirty qubits: 0, algorithmic qubits: 6
         Gate breakdown:
          {'Hadamard': 38, 'CNOT': 34, 'T': 222, 'Toffoli': 7, 'X': 4, 'S': 8, 'Z': 8}
    """

    resource_keys = {"base_cmpr_op", "num_estimation_wires", "adj_qft_cmpr_op"}

    def __init__(
        self,
        base: ResourceOperator,
        num_estimation_wires: int,
        adj_qft_op: ResourceOperator = None,
        wires=None,
    ):
        remove_ops = [base, adj_qft_op] if adj_qft_op is not None else [base]
        self.dequeue(remove_ops)
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
                raise ValueError(f"Expected {self.num_wires} wires, got {wires}.")
        else:
            self.wires = None

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * base_cmpr_op (CompressedResourceOp): A compressed resource operator, corresponding
                  to the phase estimation operator.
                * num_estimation_wires (int): the number of wires used for measuring out the phase
                * adj_qft_cmpr_op (Union[CompressedResourceOp, None]): An optional compressed
                  resource operator, corresponding to the adjoint QFT routine. If :code:`None`, the
                  default :class:`~.pennylane.labs.resource_estimation.ResourceQFT` will be used.
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
        adj_qft_cmpr_op: CompressedResourceOp,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            base_cmpr_op (CompressedResourceOp): A compressed resource operator, corresponding
                to the phase estimation operator.
            num_estimation_wires (int): the number of wires used for measuring out the phase
            adj_qft_cmpr_op (Union[CompressedResourceOp, None]): An optional compressed
                resource operator, corresponding to the adjoint QFT routine. If :code:`None`, the
                default :class:`~.pennylane.labs.resource_estimation.ResourceQFT` will be used.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "base_cmpr_op": base_cmpr_op,
            "num_estimation_wires": num_estimation_wires,
            "adj_qft_cmpr_op": adj_qft_cmpr_op,
        }
        num_wires = num_estimation_wires + base_cmpr_op.num_wires
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, base_cmpr_op, num_estimation_wires, adj_qft_cmpr_op, **kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            base_cmpr_op (CompressedResourceOp): A compressed resource operator, corresponding
                to the phase estimation operator.
            num_estimation_wires (int): the number of wires used for measuring out the phase
            adj_qft_cmpr_op (Union[CompressedResourceOp, None]): An optional compressed
                resource operator, corresponding to the adjoint QFT routine. If :code:`None`, the
                default :class:`~.pennylane.labs.resource_estimation.ResourceQFT` will be used.

        Resources:
            The resources are obtained from the standard decomposition of QPE as presented
            in (section 5.2) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum
            Information <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.
        """
        hadamard = resource_rep(re.ResourceHadamard)
        ctrl_op = ResourceControlledSequence.resource_rep(base_cmpr_op, num_estimation_wires)
        if adj_qft_cmpr_op is None:
            adj_qft_cmpr_op = resource_rep(
                re.ResourceAdjoint,
                {
                    "base_cmpr_op": resource_rep(ResourceQFT, {"num_wires": num_estimation_wires}),
                },
            )

        return [
            GateCount(hadamard, num_estimation_wires),
            GateCount(ctrl_op),
            GateCount(adj_qft_cmpr_op),
        ]

    @staticmethod
    def tracking_name(base_cmpr_op, num_estimation_wires, adj_qft_cmpr_op) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_cmpr_op.name
        adj_qft_name = None if adj_qft_cmpr_op is None else adj_qft_cmpr_op.name
        return f"QPE({base_name}, {num_estimation_wires}, adj_qft={adj_qft_name})"


class ResourceIterativeQPE(ResourceOperator):
    r"""Resource class for Iterative Quantum Phase Estimation (IQPE).

    Args:
        base (~.pennylane.labs.resource_estimation.ResourceOperator): the phase estimation operator
        num_iter (int): the number of mid-circuit measurements made to read out the phase

    Resources:
        The resources are obtained following the construction from `arXiv:0610214v3 <https://arxiv.org/abs/quant-ph/0610214v3>`_.

    .. seealso:: :func:`~.iterative_qpe`

    **Example**

    The resources for this operation are computed using:

    >>> gate_set = {"Hadamard", "CRX", "PhaseShift"}
    >>> iqpe = plre.ResourceIterativeQPE(plre.ResourceRX(), 5)
    >>> print(plre.estimate(iqpe, gate_set))
    --- Resources: ---
     Total qubits: 2
     Total gates : 25
     Qubit breakdown:
      clean qubits: 1, dirty qubits: 0, algorithmic qubits: 1
     Gate breakdown:
      {'Hadamard': 10, 'CRX': 5, 'PhaseShift': 10}
    """

    resource_keys = {"base_cmpr_op", "num_iter"}

    def __init__(self, base: ResourceOperator, num_iter: int):
        self.dequeue(base)
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
                * base_cmpr_op (CompressedResourceOp): A compressed resource operator, corresponding
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
            base_cmpr_op (CompressedResourceOp): A compressed resource operator, corresponding
                to the phase estimation operator.
            num_iter (int): the number of mid-circuit measurements made to read out the phase

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        num_wires = base_cmpr_op.num_wires
        return CompressedResourceOp(
            cls, num_wires, {"base_cmpr_op": base_cmpr_op, "num_iter": num_iter}
        )

    @classmethod
    def resource_decomp(cls, base_cmpr_op, num_iter, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            base_cmpr_op (CompressedResourceOp): A compressed resource operator, corresponding
                to the phase estimation operator.
            num_iter (int): the number of mid-circuit measurements made to read out the phase

        Resources:
            The resources are obtained following the construction from `arXiv:0610214v3
            <https://arxiv.org/abs/quant-ph/0610214v3>`_.

        Returns:
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_counts = [
            GateCount(resource_rep(re.ResourceHadamard), 2 * num_iter),
            AllocWires(1),
        ]

        # Here we want to use this particular decomposition, not any random one the user might override
        gate_counts += ResourceControlledSequence.resource_decomp(base_cmpr_op, num_iter, **kwargs)

        num_phase_gates = num_iter * (num_iter - 1) // 2
        gate_counts.append(
            GateCount(re.ResourcePhaseShift.resource_rep(), num_phase_gates)
        )  # Classically controlled PS

        gate_counts.append(FreeWires(1))
        return gate_counts


class ResourceQFT(ResourceOperator):
    r"""Resource class for QFT.

    Args:
        num_wires (int): the number of qubits the operation acts upon
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from the standard decomposition of QFT as presented
        in (chapter 5) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum Information
        <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.

    .. seealso:: :class:`~.QFT`

    **Example**

    The resources for this operation are computed using:

    >>> qft = plre.ResourceQFT(3)
    >>> gate_set = {"SWAP", "Hadamard", "ControlledPhaseShift"}
    >>> print(plre.estimate(qft, gate_set))
    --- Resources: ---
     Total qubits: 3
     Total gates : 7
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
     Gate breakdown:
      {'Hadamard': 3, 'SWAP': 1, 'ControlledPhaseShift': 3}
    """

    resource_keys = {"num_wires"}

    def __init__(self, num_wires, wires=None) -> None:
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
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"num_wires": num_wires}
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, num_wires, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_wires (int): the number of qubits the operation acts upon

        Resources:
            The resources are obtained from the standard decomposition of QFT as presented
            in (Chapter 5) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum Information
            <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.

        Returns:
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        hadamard = resource_rep(re.ResourceHadamard)
        swap = resource_rep(re.ResourceSWAP)
        ctrl_phase_shift = resource_rep(re.ResourceControlledPhaseShift)

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
    def phase_grad_resource_decomp(cls, num_wires, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        .. note::

            This decomposition assumes an appropriately sized phase gradient state is available.
            Users should ensure the cost of constructing such a state has been accounted for.
            See also :class:`~.pennylane.labs.resource_estimation.ResourcePhaseGradient`.

        Args:
            num_wires (int): the number of qubits the operation acts upon

        Resources:
            The resources are obtained as presented in the article
            `Turning Gradients into Additions into QFTs <https://algassert.com/post/1620>`_.
            Specifically, following the figure titled "8 qubit Quantum Fourier Transform with gradient shifts"

        Returns:
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        hadamard = resource_rep(re.ResourceHadamard)
        swap = resource_rep(re.ResourceSWAP)

        if num_wires == 1:
            return [GateCount(hadamard)]

        gate_types = [
            GateCount(hadamard, num_wires),
            GateCount(swap, num_wires // 2),
        ]

        for size_reg in range(1, num_wires):
            ctrl_add = re.ResourceControlled.resource_rep(
                re.ResourceSemiAdder.resource_rep(max_register_size=size_reg),
                num_ctrl_wires=1,
                num_ctrl_values=0,
            )
            gate_types.append(GateCount(ctrl_add))

        return gate_types

    @staticmethod
    def tracking_name(num_wires) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return f"QFT({num_wires})"


class ResourceAQFT(ResourceOperator):
    r"""Resource class for the Approximate QFT.

    .. note::

        This operation assumes an appropriately sized phase gradient state is available.
        Users should ensure the cost of constructing such a state has been accounted for.
        See also :class:`~.pennylane.labs.resource_estimation.ResourcePhaseGradient`.

    Args:
        order (int): the maximum number of controlled phaseshifts to which the operation is truncated
        num_wires (int): the number of qubits the operation acts upon
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from (Fig. 4) of `arXiv:1803.04933, <https://arxiv.org/abs/1803.04933>`_
        excluding the allocation and instantiation of the phase gradient state. The phased :code:`Toffoli`
        gates and the classical measure-and-reset (Fig. 2) are accounted for as :code:`TempAND`
        operations.

    .. seealso:: :class:`~.AQFT`

    **Example**

    The resources for this operation are computed using:

    >>> aqft = plre.ResourceAQFT(order=2, num_wires=3)
    >>> gate_set = {"SWAP", "Hadamard", "T", "CNOT"}
    >>> print(plre.estimate(aqft, gate_set))
    --- Resources: ---
     Total qubits: 4
     Total gates : 57
     Qubit breakdown:
      clean qubits: 1, dirty qubits: 0, algorithmic qubits: 3
     Gate breakdown:
      {'Hadamard': 7, 'CNOT': 9, 'T': 40, 'SWAP': 1}
    """

    resource_keys = {"order, num_wires"}

    def __init__(self, order, num_wires, wires=None) -> None:
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
                * order (int): the maximum number of controlled phaseshifts to which the operation is truncated
                * num_wires (int): the number of qubits the operation acts upon
        """
        return {"order": self.order, "num_wires": self.num_wires}

    @classmethod
    def resource_rep(cls, order, num_wires) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            order (int): the maximum number of controlled phaseshifts to truncate
            num_wires (int): the number of qubits the operation acts upon

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"order": order, "num_wires": num_wires}
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, order, num_wires, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            order (int): the maximum number of controlled phaseshifts to which the operation is truncated
            num_wires (int): the number of qubits the operation acts upon

        Resources:
            The resources are obtained from (Fig. 4) `arXiv:1803.04933 <https://arxiv.org/abs/1803.04933>`_
            excluding the allocation and instantiation of the phase gradient state. The phased Toffoli
            gates and the classical measure-and-reset (Fig. 2) are accounted for as `TempAND`
            operations.

        Returns:
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        hadamard = resource_rep(re.ResourceHadamard)
        swap = resource_rep(re.ResourceSWAP)
        cs = re.ResourceControlled.resource_rep(
            base_cmpr_op=resource_rep(re.ResourceS),
            num_ctrl_wires=1,
            num_ctrl_values=0,
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

                temp_and = resource_rep(re.ResourceTempAND)
                temp_and_dag = re.ResourceAdjoint.resource_rep(temp_and)
                in_place_add = re.ResourceSemiAdder.resource_rep(addition_reg_size)

                cost_iter = [
                    AllocWires(addition_reg_size),
                    GateCount(temp_and, addition_reg_size),
                    GateCount(in_place_add),
                    GateCount(hadamard),
                    GateCount(temp_and_dag, addition_reg_size),
                    FreeWires(addition_reg_size),
                ]
                gate_types.extend(cost_iter)

            addition_reg_size = order - 1
            repetitions = num_wires - order

            temp_and = resource_rep(re.ResourceTempAND)
            temp_and_dag = re.ResourceAdjoint.resource_rep(temp_and)
            in_place_add = re.ResourceSemiAdder.resource_rep(addition_reg_size)

            cost_iter = [
                AllocWires(addition_reg_size),
                GateCount(temp_and, addition_reg_size * repetitions),
                GateCount(in_place_add, repetitions),
                GateCount(hadamard, repetitions),
                GateCount(temp_and_dag, addition_reg_size * repetitions),
                FreeWires(addition_reg_size),
            ]
            gate_types.extend(cost_iter)

            gate_types.append(GateCount(swap, num_wires // 2))

        return gate_types

    @staticmethod
    def tracking_name(order, num_wires) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return f"AQFT({order}, {num_wires})"


class ResourceBasisRotation(ResourceOperator):
    r"""Resource class for the BasisRotation gate.

    Args:
        dim_N (int): The dimensions of the input :code:`unitary_matrix`. This is computed as the
            number of columns of the matrix.
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from the construction scheme given in `Optica, 3, 1460 (2016)
        <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_. Specifically,
        the resources are given as :math:`dim_N \times (dim_N - 1) / 2` instances of the
        :class:`~.ResourceSingleExcitation` gate, and :math:`dim_N \times (1 + (dim_N - 1) / 2)` instances
        of the :class:`~.ResourcePhaseShift` gate.

    .. seealso:: :class:`~.BasisRotation`

    **Example**

    The resources for this operation are computed using:

    >>> basis_rot = plre.ResourceBasisRotation(dim_N = 5)
    >>> print(plre.estimate(basis_rot))
    --- Resources: ---
    Total qubits: 5
    Total gates : 1.740E+3
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 5
    Gate breakdown:
     {'T': 1.580E+3, 'S': 60, 'Z': 40, 'Hadamard': 40, 'CNOT': 20}
    """

    resource_keys = {"dim_N"}

    def __init__(self, dim_N, wires=None):
        self.num_wires = dim_N
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls, dim_N, **kwargs) -> list[GateCount]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            dim_N (int): The dimensions of the input :code:`unitary_matrix`. This is computed
                as the number of columns of the matrix.

        Resources:
            The resources are obtained from the construction scheme given in `Optica, 3, 1460 (2016)
            <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_. Specifically,
            the resources are given as :math:`dim_N * (dim_N - 1) / 2` instances of the
            :class:`~.ResourceSingleExcitation` gate, and :math:`dim_N * (1 + (dim_N - 1) / 2)` instances
            of the :class:`~.ResourcePhaseShift` gate.

        Returns:
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        phase_shift = resource_rep(re.ResourcePhaseShift)
        single_excitation = resource_rep(re.ResourceSingleExcitation)

        se_count = dim_N * (dim_N - 1) // 2
        ps_count = dim_N + se_count

        return [GateCount(phase_shift, ps_count), GateCount(single_excitation, se_count)]

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * dim_N (int): The dimensions of the input :code:`unitary_matrix`. This is computed as the number of columns of the matrix.

        """
        return {"dim_N": self.num_wires}

    @classmethod
    def resource_rep(cls, dim_N) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            dim_N (int): The dimensions of the input :code:`unitary_matrix`. This is computed
                as the number of columns of the matrix.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"dim_N": dim_N}
        num_wires = dim_N
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def tracking_name(cls, dim_N) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return f"BasisRotation({dim_N})"


class ResourceSelect(ResourceOperator):
    r"""Resource class for the Select gate.

    Args:
        select_ops (list[~.ResourceOperator]): the set of operations to select over
        wires (Sequence[int], optional): The wires the operation acts on. If :code:`select_ops`
            provide wire labels, then this is just the set of control wire labels. Otherwise, it
            also includes the target wire labels of the selected operators.

    Resources:
        The resources are based on the analysis in `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_ section III.A,
        'Unary Iteration and Indexed Operations'. See Figures 4, 6, and 7.

        Note: This implementation assumes we have access to :math:`n - 1` additional work qubits,
        where :math:`n = \left\lceil log_{2}(N) \right\rceil` and :math:`N` is the number of batches of unitaries
        to select.

    .. seealso:: :class:`~.Select`

    **Example**

    The resources for this operation are computed using:

    >>> ops = [plre.ResourceX(), plre.ResourceY(), plre.ResourceZ()]
    >>> select_op = plre.ResourceSelect(select_ops=ops)
    >>> print(plre.estimate(select_op))
    --- Resources: ---
    Total qubits: 4
    Total gates : 24
    Qubit breakdown:
     clean qubits: 1, dirty qubits: 0, algorithmic qubits: 3
    Gate breakdown:
     {'CNOT': 7, 'S': 2, 'Z': 1, 'Hadamard': 8, 'X': 4, 'Toffoli': 2}
    """

    resource_keys = {"num_wires", "cmpr_ops"}

    def __init__(self, select_ops, wires=None) -> None:
        self.dequeue(op_to_remove=select_ops)
        self.queue()
        num_select_ops = len(select_ops)
        num_ctrl_wires = math.ceil(math.log2(num_select_ops))

        try:
            cmpr_ops = tuple(op.resource_rep_from_op() for op in select_ops)
            self.cmpr_ops = cmpr_ops
        except AttributeError as error:
            raise ValueError(
                "All factors of the Select must be instances of `ResourceOperator` in order to obtain resources."
            ) from error

        ops_wires = Wires.all_wires([op.wires for op in select_ops if op.wires is not None])
        fewest_unique_wires = max(op.num_wires for op in cmpr_ops)
        minimum_num_wires = max(fewest_unique_wires, len(ops_wires)) + num_ctrl_wires

        if wires:
            self.wires = Wires.all_wires([Wires(wires), ops_wires])
            if len(self.wires) < minimum_num_wires:
                raise ValueError(
                    f"Expected atleast {minimum_num_wires} wires ({num_ctrl_wires} control + {fewest_unique_wires} target). Got {self.wires}."
                )
            self.num_wires = len(self.wires)
        else:
            self.wires = None
            self.num_wires = minimum_num_wires

    @classmethod
    def resource_decomp(cls, cmpr_ops, num_wires, **kwargs):  # pylint: disable=unused-argument
        r"""The resources for a select implementation taking advantage of the unary iterator trick.

        Args:
            cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed
                representation, to be applied according to the selected qubits.
            num_wires (int): The number of wires the operation acts on. This is a sum of the
                control wires (:math:`\lceil(log_{2}(N))\rceil`) required and the number wires
                targeted by the :code:`select_ops`.

        Resources:
            The resources are based on the analysis in `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_ section III.A,
            'Unary Iteration and Indexed Operations'. See Figures 4, 6, and 7.

            Note: This implementation assumes we have access to :math:`n - 1` additional work qubits,
            where :math:`n = \left\lceil log_{2}(N) \right\rceil` and :math:`N` is the number of batches of unitaries
            to select.

        Returns:
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_types = []
        x = re.ResourceX.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()
        l_elbow = resource_rep(re.ResourceTempAND)
        r_elbow = resource_rep(re.ResourceAdjoint, {"base_cmpr_op": l_elbow})

        num_ops = len(cmpr_ops)
        work_qubits = math.ceil(math.log2(num_ops)) - 1

        gate_types.append(AllocWires(work_qubits))
        for cmp_rep in cmpr_ops:
            ctrl_op = re.ResourceControlled.resource_rep(cmp_rep, 1, 0)
            gate_types.append(GateCount(ctrl_op))

        gate_types.append(GateCount(x, 2 * (num_ops - 1)))  # conjugate 0 controlled toffolis
        gate_types.append(GateCount(cnot, num_ops - 1))
        gate_types.append(GateCount(l_elbow, num_ops - 1))
        gate_types.append(GateCount(r_elbow, num_ops - 1))

        gate_types.append(FreeWires(work_qubits))
        return gate_types

    @staticmethod
    def textbook_resources(cmpr_ops, num_wires, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed
                representation, to be applied according to the selected qubits.
            num_wires (int): The number of wires the operation acts on. This is a sum of the
                control wires (:math:`\lceil(log_{2}(N))\rceil`) required and the number wires
                targeted by the :code:`select_ops`.

        Resources:
            The resources correspond directly to the definition of the operation. Specifically,
            for each operator in :code:`cmpr_ops`, the cost is given as a controlled version of the operator
            controlled on the associated bitstring.

        Returns:
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_types = defaultdict(int)
        x = re.ResourceX.resource_rep()

        num_ops = len(cmpr_ops)
        num_ctrl_wires = int(qnp.ceil(qnp.log2(num_ops)))
        num_total_ctrl_possibilities = 2**num_ctrl_wires  # 2^n

        num_zero_controls = num_total_ctrl_possibilities // 2
        gate_types[x] = num_zero_controls * 2  # conjugate 0 controls

        for cmp_rep in cmpr_ops:
            ctrl_op = re.ResourceControlled.resource_rep(
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
                * cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed representation, to be applied according to the selected qubits.
                * num_wires (int): The number of wires the operation acts on. This is a sum of the
                  control wires (:math:`\lceil(log_{2}(N))\rceil`) required and the number wires
                  targeted by the :code:`select_ops`.

        """
        return {"cmpr_ops": self.cmpr_ops, "num_wires": self.num_wires}

    @classmethod
    def resource_rep(cls, cmpr_ops, num_wires=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed
                representation, to be applied according to the selected qubits.
            num_wires (int): An optional parameter representing the number of wires the operation
                acts on. This is a sum of the control wires (:math:`\lceil(log_{2}(N))\rceil`)
                required and the number of wires targeted by the :code:`select_ops`.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        num_ctrl_wires = math.ceil(math.log2(len(cmpr_ops)))
        fewest_unique_wires = max(op.num_wires for op in cmpr_ops)

        num_wires = num_wires or fewest_unique_wires + num_ctrl_wires
        params = {"cmpr_ops": cmpr_ops, "num_wires": num_wires}
        return CompressedResourceOp(cls, num_wires, params)


class ResourceQROM(ResourceOperator):
    r"""Resource class for the QROM template.

    Args:
        num_bitstrings (int): the number of bitstrings that are to be encoded
        size_bitstring (int): the length of each bitstring
        num_bit_flips (int, optional): The total number of :math:`1`'s in the dataset. Defaults to
            :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
        clean (bool, optional): Determine if allocated qubits should be reset after the computation
            (at the cost of higher gate counts). Defaults to :code`True`.
        select_swap_depth (Union[int, None], optional): A parameter :math:`\lambda` that determines
            if data will be loaded in parallel by adding more rows following Figure 1.C of
            `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be :code:`None`,
            :code:`1` or a positive integer power of two. Defaults to :code:`None`, which internally
            determines the optimal depth.
        wires (Sequence[int], optional): The wires the operation acts on (control and target).
            Excluding any additional qubits allocated during the decomposition (e.g select-swap wires).

    Resources:
        The resources for QROM are taken from the following two papers:
        `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_ (Figure 1.C) for
        :code:`clean = False` and `Berry et al. (2019) <https://arxiv.org/pdf/1902.02134>`_
        (Figure 4) for :code:`clean = True`.

    .. seealso:: :class:`~.QROM`

    **Example**

    The resources for this operation are computed using:

    >>> qrom = plre.ResourceQROM(
    ...     num_bitstrings=10,
    ...     size_bitstring=4,
    ... )
    >>> print(plre.estimate(qrom))
    --- Resources: ---
    Total qubits: 11
    Total gates : 178
    Qubit breakdown:
     clean qubits: 3, dirty qubits: 0, algorithmic qubits: 8
    Gate breakdown:
     {'Hadamard': 56, 'X': 34, 'CNOT': 72, 'Toffoli': 16}

    """

    resource_keys = {
        "num_bitstrings",
        "size_bitstring",
        "num_bit_flips",
        "select_swap_depth",
        "clean",
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
        num_bitstrings,
        size_bitstring,
        num_bit_flips=None,
        clean=True,
        select_swap_depth=None,
        wires=None,
    ) -> None:
        self.clean = clean
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

    @classmethod
    def resource_decomp(
        cls,
        num_bitstrings,
        size_bitstring,
        num_bit_flips,
        select_swap_depth=None,
        clean=True,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the operator's resources.

        Args:
            num_bitstrings (int): the number of bitstrings that are to be encoded
            size_bitstring (int): the length of each bitstring
            num_bit_flips (int, optional): The total number of :math:`1`'s in the dataset. Defaults to
                :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
            clean (bool, optional): Determine if allocated qubits should be reset after the computation
                (at the cost of higher gate counts). Defaults to :code`True`.
            select_swap_depth (Union[int, None], optional): A parameter :math:`\lambda` that determines
                if data will be loaded in parallel by adding more rows following Figure 1.C of
                `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be :code:`None`,
                :code:`1` or a positive integer power of two. Defaults to :code:`None`, which internally
                determines the optimal depth.
            wires (Sequence[int], optional): the wires the operation acts on

        Resources:
            The resources for QROM are taken from the following two papers:
            `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_ (Figure 1.C) for
            :code:`clean = False` and `Berry et al. (2019) <https://arxiv.org/pdf/1902.02134>`_
            (Figure 4) for :code:`clean = True`.

            Note: we use the unary iterator trick to implement the Select. This
            implementation assumes we have access to :math:`n - 1` additional
            work qubits, where :math:`n = \left\lceil log_{2}(N) \right\rceil` and :math:`N` is
            the number of batches of unitaries to select.
        """

        if select_swap_depth:
            max_depth = 2 ** math.ceil(math.log2(num_bitstrings))
            select_swap_depth = min(max_depth, select_swap_depth)  # truncate depth beyond max depth

        W_opt = select_swap_depth or ResourceQROM._t_optimized_select_swap_width(
            num_bitstrings, size_bitstring
        )
        L_opt = math.ceil(num_bitstrings / W_opt)
        l = math.ceil(math.log2(L_opt))

        gate_cost = []
        num_alloc_wires = (W_opt - 1) * size_bitstring  # Swap registers
        if L_opt > 1:
            num_alloc_wires += l - 1  # + work_wires for UI trick

        gate_cost.append(AllocWires(num_alloc_wires))

        x = resource_rep(re.ResourceX)
        cnot = resource_rep(re.ResourceCNOT)
        l_elbow = resource_rep(re.ResourceTempAND)
        r_elbow = resource_rep(re.ResourceAdjoint, {"base_cmpr_op": l_elbow})
        hadamard = resource_rep(re.ResourceHadamard)

        swap_clean_prefactor = 1
        select_clean_prefactor = 1

        if clean:
            gate_cost.append(GateCount(hadamard, 2 * size_bitstring))
            swap_clean_prefactor = 4
            select_clean_prefactor = 2

        # SELECT cost:
        if L_opt > 1:
            gate_cost.append(
                GateCount(x, select_clean_prefactor * (2 * (L_opt - 2) + 1))
            )  # conjugate 0 controlled toffolis + 1 extra X gate from un-controlled unary iterator decomp
            gate_cost.append(
                GateCount(
                    cnot,
                    select_clean_prefactor * (L_opt - 2) + select_clean_prefactor * num_bit_flips,
                )  # num CNOTs in unary iterator trick   +   each unitary in the select is just a CNOT
            )
            gate_cost.append(GateCount(l_elbow, select_clean_prefactor * (L_opt - 2)))
            gate_cost.append(GateCount(r_elbow, select_clean_prefactor * (L_opt - 2)))

            gate_cost.append(FreeWires(l - 1))  # release UI trick work wires

        else:
            gate_cost.append(
                GateCount(
                    x, select_clean_prefactor * num_bit_flips
                )  # each unitary in the select is just an X gate to load the data
            )

        # SWAP cost:
        ctrl_swap = resource_rep(re.ResourceCSWAP)
        gate_cost.append(GateCount(ctrl_swap, swap_clean_prefactor * (W_opt - 1) * size_bitstring))

        if clean:
            gate_cost.append(FreeWires((W_opt - 1) * size_bitstring))  # release Swap registers

        return gate_cost

    @classmethod
    def single_controlled_res_decomp(
        cls,
        num_bitstrings,
        size_bitstring,
        num_bit_flips,
        select_swap_depth,
        clean,
    ):
        r"""The resource decomposition for QROM controlled on a single wire."""
        if select_swap_depth:
            max_depth = 2 ** math.ceil(math.log2(num_bitstrings))
            select_swap_depth = min(max_depth, select_swap_depth)  # truncate depth beyond max depth

        W_opt = select_swap_depth or ResourceQROM._t_optimized_select_swap_width(
            num_bitstrings, size_bitstring
        )
        L_opt = math.ceil(num_bitstrings / W_opt)
        l = math.ceil(math.log2(L_opt))

        gate_cost = []
        num_alloc_wires = (W_opt - 1) * size_bitstring  # Swap registers
        if L_opt > 1:
            num_alloc_wires += l  # + work_wires for UI trick

        gate_cost.append(AllocWires(num_alloc_wires))

        x = resource_rep(re.ResourceX)
        cnot = resource_rep(re.ResourceCNOT)
        l_elbow = resource_rep(re.ResourceTempAND)
        r_elbow = resource_rep(re.ResourceAdjoint, {"base_cmpr_op": l_elbow})
        hadamard = resource_rep(re.ResourceHadamard)

        swap_clean_prefactor = 1
        select_clean_prefactor = 1

        if clean:
            gate_cost.append(GateCount(hadamard, 2 * size_bitstring))
            swap_clean_prefactor = 4
            select_clean_prefactor = 2

        # SELECT cost:
        if L_opt > 1:
            gate_cost.append(
                GateCount(x, select_clean_prefactor * (2 * (L_opt - 1)))
            )  # conjugate 0 controlled toffolis
            gate_cost.append(
                GateCount(
                    cnot,
                    select_clean_prefactor * (L_opt - 1) + select_clean_prefactor * num_bit_flips,
                )  # num CNOTs in unary iterator trick   +   each unitary in the select is just a CNOT
            )
            gate_cost.append(GateCount(l_elbow, select_clean_prefactor * (L_opt - 1)))
            gate_cost.append(GateCount(r_elbow, select_clean_prefactor * (L_opt - 1)))

            gate_cost.append(FreeWires(l))  # release UI trick work wires
        else:
            gate_cost.append(
                GateCount(
                    x,
                    select_clean_prefactor * num_bit_flips,
                )  #  each unitary in the select is just an X
            )

        # SWAP cost:
        w = math.ceil(math.log2(W_opt))
        ctrl_swap = re.ResourceCSWAP.resource_rep()
        gate_cost.append(AllocWires(1))  # need one temporary qubit for l/r-elbow to control SWAP

        gate_cost.append(GateCount(l_elbow, w))
        gate_cost.append(GateCount(ctrl_swap, swap_clean_prefactor * (W_opt - 1) * size_bitstring))
        gate_cost.append(GateCount(r_elbow, w))

        gate_cost.append(FreeWires(1))  # temp wires
        if clean:
            gate_cost.append(
                FreeWires((W_opt - 1) * size_bitstring)
            )  # release Swap registers + temp wires
        return gate_cost

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires: int,
        ctrl_num_ctrl_values: int,
        num_bitstrings,
        size_bitstring,
        num_bit_flips=None,
        select_swap_depth=None,
        clean=True,
        **kwargs,
    ):
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_bitstrings (int): the number of bitstrings that are to be encoded
            size_bitstring (int): the length of each bitstring
            num_bit_flips (int, optional): The total number of :math:`1`'s in the dataset. Defaults to
                :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
            clean (bool, optional): Determine if allocated qubits should be reset after the computation
                (at the cost of higher gate counts). Defaults to :code`True`.
            select_swap_depth (Union[int, None], optional): A parameter :math:`\lambda` that determines
                if data will be loaded in parallel by adding more rows following Figure 1.C of
                `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be :code:`None`,
                :code:`1` or a positive integer power of two. Defaults to :code:`None`, which internally
                determines the optimal depth.

        Resources:
            The resources for QROM are taken from the following two papers:
            `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_ (Figure 1.C) for
            :code:`clean = False` and `Berry et al. (2019) <https://arxiv.org/pdf/1902.02134>`_
            (Figure 4) for :code:`clean = True`.

            Note: we use the single-controlled unary iterator trick to implement the Select. This
            implementation assumes we have access to :math:`n - 1` additional work qubits,
            where :math:`n = \ceil{log_{2}(N)}` and :math:`N` is the number of batches of
            unitaries to select.

        Returns:
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_cost = []
        if ctrl_num_ctrl_values:
            x = re.ResourceX.resource_rep()
            gate_cost.append(GateCount(x, 2 * ctrl_num_ctrl_values))

        if num_bit_flips is None:
            num_bit_flips = (num_bitstrings * size_bitstring) // 2

        single_ctrl_cost = cls.single_controlled_res_decomp(
            num_bitstrings,
            size_bitstring,
            num_bit_flips,
            select_swap_depth,
            clean,
        )

        if ctrl_num_ctrl_wires == 1:
            gate_cost.extend(single_ctrl_cost)
            return gate_cost

        gate_cost.append(AllocWires(1))
        gate_cost.append(
            GateCount(re.ResourceMultiControlledX.resource_rep(ctrl_num_ctrl_wires, 0))
        )
        gate_cost.extend(single_ctrl_cost)
        gate_cost.append(
            GateCount(re.ResourceMultiControlledX.resource_rep(ctrl_num_ctrl_wires, 0))
        )
        gate_cost.append(FreeWires(1))
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
                * clean (bool, optional): Determine if allocated qubits should be reset after the
                  computation (at the cost of higher gate counts). Defaults to :code`True`.
                * select_swap_depth (Union[int, None], optional): A parameter :math:`\lambda` that
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
            "clean": self.clean,
        }

    @classmethod
    def resource_rep(
        cls,
        num_bitstrings,
        size_bitstring,
        num_bit_flips=None,
        clean=True,
        select_swap_depth=None,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_bitstrings (int): the number of bitstrings that are to be encoded
            size_bitstring (int): the length of each bitstring
            num_bit_flips (int, optional): The total number of :math:`1`'s in the dataset. Defaults to
                :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
            clean (bool, optional): Determine if allocated qubits should be reset after the computation
                (at the cost of higher gate counts). Defaults to :code`True`.
            select_swap_depth (Union[int, None], optional): A parameter :math:`\lambda` that determines
                if data will be loaded in parallel by adding more rows following Figure 1.C of
                `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be :code:`None`,
                :code:`1` or a positive integer power of two. Defaults to :code:`None`, which internally
                determines the optimal depth.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
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
            "clean": clean,
        }
        num_wires = size_bitstring + math.ceil(math.log2(num_bitstrings))
        return CompressedResourceOp(cls, num_wires, params)


class ResourceQubitUnitary(ResourceOperator):
    r"""Resource class for the QubitUnitary template.

    Args:
        num_wires (int): the number of qubits the operation acts upon
        precision (Union[float, None], optional): The precision used when preparing the single qubit
            rotations used to synthesize the n-qubit unitary.
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are defined by combining the two equalities in `Möttönen and Vartiainen
        (2005), Fig 14 <https://arxiv.org/pdf/quant-ph/0504100>`_ , we can express an :math:`n`
        qubit unitary as four :math:`n - 1` qubit unitaries and three multiplexed rotations
        via ( :class:`~.labs.resource_estimation.ResourceSelectPauliRot` ). Specifically, the cost
        is given by:

        * 1-qubit unitary, the cost is approximated as a single :code:`RZ` rotation.

        * 2-qubit unitary, the cost is approximated as four single qubit rotations and three :code:`CNOT` gates.

        * 3-qubit unitary or more, the cost is given according to the reference above, recursively.

    .. seealso:: :class:`~.QubitUnitary`

    **Example**

    The resources for this operation are computed using:

    >>> qu = plre.ResourceQubitUnitary(num_wires=3)
    >>> print(plre.estimate(qu, gate_set))
    --- Resources: ---
     Total qubits: 3
     Total gates : 52
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
     Gate breakdown:
      {'RZ': 24, 'CNOT': 24, 'RY': 4}
    """

    resource_keys = {"num_wires", "precision"}

    def __init__(self, num_wires, precision=None, wires=None):
        self.num_wires = num_wires
        self.precision = precision
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int): the number of qubits the operation acts upon
                * precision (Union[float, None], optional): The precision used when preparing the
                  single qubit rotations used to synthesize the n-qubit unitary.
        """
        return {"num_wires": self.num_wires, "precision": self.precision}

    @classmethod
    def resource_rep(cls, num_wires, precision=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            num_wires (int): the number of qubits the operation acts upon
            precision (Union[float, None], optional): The precision used when preparing the single
                qubit rotations used to synthesize the n-qubit unitary.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"num_wires": num_wires, "precision": precision}
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, num_wires, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_wires (int): the number of qubits the operation acts upon
            precision (Union[float, None], optional): The precision used when preparing the single
                qubit rotations used to synthesize the n-qubit unitary.

        Resources:
            The resources are defined by combining the two equalities in `Möttönen and Vartiainen
            (2005), Fig 14 <https://arxiv.org/pdf/quant-ph/0504100>`_, we can express an :math:`n`-
            qubit unitary as four :math:`n - 1`-qubit unitaries and three multiplexed rotations
            via (:class:`~.labs.resource_estimation.ResourceSelectPauliRot`). Specifically, the cost
            is given by:

            * 1-qubit unitary, the cost is approximated as a single :code:`RZ` rotation.

            * 2-qubit unitary, the cost is approximated as four single qubit rotations and three :code:`CNOT` gates.

            * 3-qubit unitary or more, the cost is given according to the reference above, recursively.

        Returns:
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_lst = []

        one_qubit_decomp_cost = [GateCount(resource_rep(re.ResourceRZ, {"precision": precision}))]
        two_qubit_decomp_cost = [
            GateCount(resource_rep(re.ResourceRZ, {"precision": precision}), 4),
            GateCount(resource_rep(re.ResourceCNOT), 3),
        ]

        if num_wires == 1:
            return one_qubit_decomp_cost

        if num_wires == 2:
            return two_qubit_decomp_cost

        for gc in two_qubit_decomp_cost:
            gate_lst.append(4 ** (num_wires - 2) * gc)

        for index in range(2, num_wires):
            multiplex_z = resource_rep(
                ResourceSelectPauliRot,
                {
                    "num_ctrl_wires": index,
                    "rotation_axis": "Z",
                    "precision": precision,
                },
            )
            multiplex_y = resource_rep(
                ResourceSelectPauliRot,
                {
                    "num_ctrl_wires": index,
                    "rotation_axis": "Y",
                    "precision": precision,
                },
            )

            gate_lst.append(GateCount(multiplex_z, 2 * 4 ** (num_wires - (1 + index))))
            gate_lst.append(GateCount(multiplex_y, 4 ** (num_wires - (1 + index))))

        return gate_lst


class ResourceSelectPauliRot(ResourceOperator):
    r"""Resource class for the SelectPauliRot gate.

    Args:
        rotation_axis (str): the rotation axis used in the multiplexer
        num_ctrl_wires (int): the number of control wires of the multiplexer
        precision (float): the precision used in the single qubit rotations
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from the construction scheme given in `Möttönen and Vartiainen
        (2005), Fig 7a <https://arxiv.org/abs/quant-ph/0504100>`_. Specifically, the resources
        for an :math:`n` qubit unitary are given as :math:`2^{n}` instances of the :code:`CNOT`
        gate and :math:`2^{n}` instances of the single qubit rotation gate (:code:`RX`,
        :code:`RY` or :code:`RZ`) depending on the :code:`rotation_axis`.

    .. seealso:: :class:`~.SelectPauliRot`

    **Example**

    The resources for this operation are computed using:

    >>> mltplxr = plre.ResourceSelectPauliRot(
    ...     rotation_axis = "Y",
    ...     num_ctrl_wires = 4,
    ...     precision = 1e-3,
    ... )
    >>> print(plre.estimate(mltplxr, plre.StandardGateSet))
    --- Resources: ---
     Total qubits: 5
     Total gates : 32
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 5
     Gate breakdown:
      {'RY': 16, 'CNOT': 16}
    """

    resource_keys = {"num_ctrl_wires", "rotation_axis", "precision"}

    def __init__(self, rotation_axis: str, num_ctrl_wires: int, precision=None, wires=None) -> None:
        if rotation_axis not in ("X", "Y", "Z"):
            raise ValueError("The `rotation_axis` argument must be one of ('X', 'Y', 'Z')")

        self.num_ctrl_wires = num_ctrl_wires
        self.rotation_axis = rotation_axis
        self.precision = precision

        self.num_wires = num_ctrl_wires + 1
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * rotation_axis (str): the rotation axis used in the multiplexer
            * num_ctrl_wires (int): the number of control wires of the multiplexer
            * precision (float): the precision used in the single qubit rotations
        """
        return {
            "num_ctrl_wires": self.num_ctrl_wires,
            "rotation_axis": self.rotation_axis,
            "precision": self.precision,
        }

    @classmethod
    def resource_rep(cls, num_ctrl_wires, rotation_axis, precision=None):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            rotation_axis (str): the rotation axis used in the multiplexer
            num_ctrl_wires (int): the number of control wires of the multiplexer
            precision (float): the precision used in the single qubit rotations

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        num_wires = num_ctrl_wires + 1
        return CompressedResourceOp(
            cls,
            num_wires,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "rotation_axis": rotation_axis,
                "precision": precision,
            },
        )

    @classmethod
    def resource_decomp(cls, num_ctrl_wires, rotation_axis, precision, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            rotation_axis (str): the rotation axis used in the multiplexer
            num_ctrl_wires (int): the number of control wires of the multiplexer
            precision (float): the precision used in the single qubit rotations
            wires (Sequence[int], optional): the wires the operation acts on

        Resources:
            The resources are obtained from the construction scheme given in `Möttönen and Vartiainen
            (2005), Fig 7a <https://arxiv.org/abs/quant-ph/0504100>`_. Specifically, the resources
            for an :math:`n` qubit unitary are given as :math:`2^{n}` instances of the :code:`CNOT`
            gate and :math:`2^{n}` instances of the single qubit rotation gate (:code:`RX`,
            :code:`RY` or :code:`RZ`) depending on the :code:`rotation_axis`.

        Returns:
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        rotation_gate_map = {
            "X": re.ResourceRX,
            "Y": re.ResourceRY,
            "Z": re.ResourceRZ,
        }

        gate = resource_rep(rotation_gate_map[rotation_axis], {"precision": precision})
        cnot = resource_rep(re.ResourceCNOT)

        gate_lst = [
            GateCount(gate, 2**num_ctrl_wires),
            GateCount(cnot, 2**num_ctrl_wires),
        ]

        return gate_lst

    @classmethod
    def phase_grad_resource_decomp(cls, num_ctrl_wires, rotation_axis, precision, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            rotation_axis (str): the rotation axis used in the multiplexer
            num_ctrl_wires (int): the number of control wires of the multiplexer
            precision (float): the precision used in the single qubit rotations
            wires (Sequence[int], optional): the wires the operation acts on

        Resources:
            The resources are obtained from the construction scheme given in `O'Brien and Sünderhauf
            (2025), Fig 4 <https://arxiv.org/pdf/2409.07332>`_. Specifically, the resources
            use two :code:`~.labs.resource_estimation.ResourceQROM`s to digitally load and unload
            the phase angles up to some precision. These are then applied using a single controlled
            :code:`~.labs.resource_estimation.ResourceSemiAdder`.

            .. note::

                This method assumes a phase gradient state is prepared on an auxiliary register.

        Returns:
            list[~.pennylane.labs.resource_estimation.GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        num_prec_wires = math.ceil(math.log2(math.pi / precision)) + 1
        gate_lst = []

        qrom = resource_rep(
            re.ResourceQROM,
            {
                "num_bitstrings": 2**num_ctrl_wires,
                "num_bit_flips": 2**num_ctrl_wires * num_prec_wires // 2,
                "size_bitstring": num_prec_wires,
                "clean": False,
            },
        )

        gate_lst.append(AllocWires(num_prec_wires))
        gate_lst.append(GateCount(qrom))
        gate_lst.append(
            GateCount(
                resource_rep(
                    re.ResourceControlled,
                    {
                        "base_cmpr_op": resource_rep(
                            re.ResourceSemiAdder,
                            {"max_register_size": num_prec_wires},
                        ),
                        "num_ctrl_wires": 1,
                        "num_ctrl_values": 0,
                    },
                )
            )
        )
        gate_lst.append(GateCount(resource_rep(re.ResourceAdjoint, {"base_cmpr_op": qrom})))
        gate_lst.append(FreeWires(num_prec_wires))

        h = resource_rep(re.ResourceHadamard)
        s = resource_rep(re.ResourceS)
        s_dagg = resource_rep(re.ResourceAdjoint, {"base_cmpr_op": s})

        if rotation_axis == "X":
            gate_lst.append(GateCount(h, 2))
        if rotation_axis == "Y":
            gate_lst.append(GateCount(h, 2))
            gate_lst.append(GateCount(s))
            gate_lst.append(GateCount(s_dagg))

        return gate_lst
