# Copyright 2025-2026 Xanadu Quantum Technologies Inc.

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
from pennylane import math as pl_math
from pennylane import numpy as qnp
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    _dequeue,
    resource_rep,
)
from pennylane.estimator.wires_manager import Allocate, Deallocate
from pennylane.math import ceil_log2
from pennylane.wires import Wires, WiresLike

# pylint: disable=arguments-differ,too-many-arguments,unused-argument,super-init-not-called, signature-differs


class IQP(ResourceOperator):
    r"""Resource class for the Instantaneous Quantum Polynomial (IQP) template.

    Args:
        num_wires (int): the number of qubits the operation acts upon
        pattern (list[list[list[int]]]): Specification of the trainable gates. Each element of gates corresponds to a
            unique trainable parameter. Each sublist specifies the generators to which that parameter applies.
            Generators are specified by listing the qubits on which an X operator acts.
        spin_sym (bool, optional): If True, the circuit is equivalent to one where the initial state
            :math:`\frac{1}{\sqrt(2)}(|00\dots0> + |11\dots1>)` is used in place of :math:`|00\dots0>`.
        wires (Sequence[int], optional): the wires the operation acts on

    **Example:**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> iqp = qre.IQP(num_wires=4, pattern=[[[0]], [[1]], [[2]], [[3]]])
    >>> print(qre.estimate(iqp))
    --- Resources: ---
     Total wires: 4
       algorithmic wires: 4
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 184
       'T': 176,
       'CNOT': 0,
       'Hadamard': 8

    .. seealso:: :class:`~.IQP`

    """

    resource_keys = {"spin_sym", "pattern", "num_wires"}

    def __init__(self, num_wires, pattern, spin_sym=False, wires=None) -> None:
        self.num_wires = num_wires
        self.spin_sym = spin_sym
        self.pattern = pattern
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int): the number of qubits the operation acts upon
        """
        return {
            "spin_sym": self.spin_sym,
            "pattern": self.pattern,
            "num_wires": self.num_wires,
        }

    @classmethod
    def resource_rep(cls, num_wires, pattern, spin_sym) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            num_wires (int): the number of qubits the operation acts upon
            pattern (list[list[list[int]]]): Specification of the trainable gates. Each element of gates corresponds to a
                unique trainable parameter. Each sublist specifies the generators to which that parameter applies.
                Generators are specified by listing the qubits on which an X operator acts.
            spin_sym (bool, optional): If True, the circuit is equivalent to one where the initial state
                :math:`\frac{1}{\sqrt(2)}(|00\dots0> + |11\dots1>)` is used in place of :math:`|00\dots0>`.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        return CompressedResourceOp(
            cls,
            num_wires,
            {
                "spin_sym": spin_sym,
                "pattern": pattern,
                "num_wires": num_wires,
            },
        )

    @classmethod
    def resource_decomp(cls, num_wires, pattern, spin_sym) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_wires (int): the number of qubits the operation acts upon
            pattern (list[list[list[int]]]): Specification of the trainable gates. Each element of gates corresponds to a
                unique trainable parameter. Each sublist specifies the generators to which that parameter applies.
                Generators are specified by listing the qubits on which an X operator acts.
            spin_sym (bool, optional): If True, the circuit is equivalent to one where the initial state
                :math:`\frac{1}{\sqrt(2)}(|00\dots0> + |11\dots1>)` is used in place of :math:`|00\dots0>`.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        hadamard = resource_rep(qre.Hadamard)
        pauli_rot = resource_rep(qre.PauliRot, {"pauli_string": "Y" + "X" * (num_wires - 1)})

        hadamard_counts = 2 * num_wires
        multi_rz_counts = defaultdict(int)

        for gate in pattern:
            for gen in gate:
                multi_rz_counts[len(gen)] += 1

        ret = [GateCount(hadamard, hadamard_counts)]

        if spin_sym:
            ret.append(GateCount(pauli_rot, 1))

        return ret + [
            GateCount(resource_rep(qre.MultiRZ, {"num_wires": wires}), counts)
            for (wires, counts) in multi_rz_counts.items()
        ]

    @staticmethod
    def tracking_name(num_wires, pattern, spin_sym) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return f"IQP({num_wires}, {pattern}, {spin_sym})"


class SelectOnlyQRAM(ResourceOperator):
    r"""Resource class for SelectOnlyQRAM.

    Args:
        data (TensorLike | Sequence[str]):
            The classical memory array to retrieve values from.
        num_wires (int):
            The number of qubits the operation acts upon.
        num_control_wires (int):
            The number of ``control_wires``.
        num_select_wires (int):
            The number of ``select_wires``.
        control_wires (WiresLike, optional):
            The register that stores the index for the entry of the classical data we want to
            access.
        target_wires (WiresLike, optional):
            The register in which the classical data gets loaded. The size of this register must
            equal each bitstring length in ``bitstrings``.
        select_wires (WiresLike, optional):
            Wires used to perform the selection.
        select_value (int, optional):
            If provided, only entries whose select bits match this value are loaded.
            The ``select_value`` must be an integer in :math:`[0, 2^{\texttt{len(select_wires)}}]`,
            and cannot be used if no ``select_wires`` are provided.

    Raises:
        ValueError: if the number of wires provided does not match ``num_wires``

    Resources:
        The resources are obtained from the SelectOnlyQRAM implementation in PennyLane.

    .. seealso:: :class:`~.SelectOnlyQRAM`
    """

    resource_keys = {
        "data",
        "num_control_wires",
        "select_value",
        "num_select_wires",
    }

    def __init__(
        self,
        data,
        num_wires,
        num_control_wires,
        num_select_wires,
        control_wires=None,
        target_wires=None,
        select_wires=None,
        select_value=None,
    ):
        all_wires = None
        if control_wires and target_wires and select_wires:
            all_wires = list(control_wires) + list(target_wires) + list(select_wires)
            if len(all_wires) != num_wires:
                raise ValueError(f"Expected {num_wires} wires, got {len(all_wires)}.")

        if isinstance(data, (list, tuple)):
            data = pl_math.array(data)
        if isinstance(data[0], str):
            data = pl_math.array(list(map(lambda bitstring: [int(bit) for bit in bitstring], data)))
        self.data = data

        self.num_wires = num_wires
        self.select_value = select_value
        self.num_control_wires = num_control_wires
        self.num_select_wires = num_select_wires
        super().__init__(wires=all_wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * data (TensorLike | Sequence[str]): the classical memory array to retrieve values from
                * num_wires (int): the number of qubits the operation acts upon
                * select_value (int or None): if provided, only entries whose select bits match this value are loaded
                * num_select_wires (int): the number of ``select_wires``
                * num_control_wires (int): the number of ``control_wires``
        """
        return {
            "data": self.data,
            "num_wires": self.num_wires,
            "select_value": self.select_value,
            "num_select_wires": self.num_select_wires,
            "num_control_wires": self.num_control_wires,
        }

    @classmethod
    def resource_rep(cls, data, num_wires, select_value, num_select_wires, num_control_wires):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            data (TensorLike | Sequence[str]): the classical memory array to retrieve values from
            num_wires (int): the number of qubits the operation acts upon
            select_value (int or None): if provided, only entries whose select bits match this value are loaded
            num_select_wires (int): the number of ``select_wires``
            num_control_wires (int): the number of ``control_wires``

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        params = {
            "data": data,
            "num_wires": num_wires,
            "select_value": select_value,
            "num_select_wires": num_select_wires,
            "num_control_wires": num_control_wires,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, data, num_wires, select_value, num_select_wires, num_control_wires):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            data (TensorLike | Sequence[str]): the classical memory array to retrieve values from
            num_wires (int): the number of qubits the operation acts upon
            select_value (int or None): if provided, only entries whose select bits match this value are loaded
            num_select_wires (int): the number of ``select_wires``
            num_control_wires (int): the number of ``control_wires``

        Resources:
            The resources are obtained from the SelectOnlyQRAM implementation in PennyLane.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears in the decomposition.
        """
        n_total = num_control_wires + num_select_wires

        basis_embedding = resource_rep(qre.BasisEmbedding, {"num_wires": num_select_wires})
        paulix = resource_rep(qre.X)
        mcx = qre.Controlled.resource_rep(resource_rep(qre.X), n_total, 0)

        basis_embedding_count = 0
        if select_value is not None and num_select_wires > 0:
            basis_embedding_count = 1

        mcx_count = 0
        paulix_count = 0

        for addr, bits in enumerate(data):
            if (
                select_value is not None
                and num_select_wires > 0
                and (addr >> num_control_wires) != select_value
            ):
                continue

            control_values = [(addr >> (n_total - 1 - i)) & 1 for i in range(n_total)]
            paulix_count += control_values.count(0) * 2

            for j in range(data.shape[1]):
                if bits[j] == 1:
                    mcx_count += 1

        ret = []
        if paulix_count > 0:
            ret.append(GateCount(paulix, paulix_count))
        if mcx_count > 0:
            ret.append(GateCount(mcx, mcx_count))
        if basis_embedding_count > 0:
            ret.append(GateCount(basis_embedding, basis_embedding_count))

        return ret

    @staticmethod
    def tracking_name(data, num_wires, select_value, num_select_wires, num_control_wires) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return f"SelectOnlyQRAM({data}, {num_wires}, {select_value}, {num_select_wires}, {num_control_wires})"


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
                raise ValueError("Must provide at least one of `num_wires` and `wires`.")
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


class UnaryIterationQPE(ResourceOperator):
    r"""Resource class for Quantum Phase Estimation (QPE) using the unary iteration
    technique.

    This form of QPE, as described in `arXiv.2011.03494 <https://arxiv.org/pdf/2011.03494>`_,
    requires the unitary operator to be a quantum walk operator constructed from ``Select`` and ``Prepare``
    subroutines. In this approach, unary iteration is used to construct successive powers of the walk operator,
    which reduces :class:`~.pennylane.estimator.ops.qubit.non_parametric_ops.T` and
    :class:`~.pennylane.estimator.ops.op_math.controlled_ops.Toffoli` gate counts in their decomposition at the cost of
    increasing the number of auxiliary qubits required.

    For a detailed explanation of unary iteration, see
    `here <https://pennylane.ai/compilation/unary-iteration>`_. Note that users can also provide
    a custom adjoint Quantum Fourier Transform (QFT) implementation, which can be used to further
    optimize the resource requirements.

    Args:
        walk_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): the quantum
            walk operator to apply the phase estimation protocol on
        num_iterations (int): The total number of times the quantum walk operator
            is applied in order to reach a target precision in the eigenvalue estimate.
        adj_qft_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator` | None): An optional
            argument to set the subroutine used to perform the adjoint QFT operation.
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        The resources are obtained from Figure 2. in Section III of `arXiv.2011.03494 <https://arxiv.org/pdf/2011.03494>`_.

    Raises:
        ValueError: ``num_iterations`` must be an integer greater than zero
        TypeError: ``walk_op`` must be an instance of
            :class:`~.pennylane.estimator.templates.subroutines.Qubitization` or
            :class:`~.pennylane.estimator.templates.qubitize.QubitizeTHC`

    .. seealso:: Related PennyLane operation :class:`~.pennylane.QuantumPhaseEstimation` and explanation of `Unary Iteration <https://pennylane.ai/compilation/unary-iteration>`_.

    **Example**

    The resources for this operation are computed as follows:

    >>> import pennylane.estimator as qre
    >>> thc_ham = qre.THCHamiltonian(num_orbitals=20, tensor_rank=40)
    >>> num_iter, walk_op = (11, qre.QubitizeTHC(thc_ham))
    >>> res = qre.estimate(qre.UnaryIterationQPE(walk_op, num_iter))
    >>> print(res)
    --- Resources: ---
     Total wires: 402
       algorithmic wires: 101
       allocated wires: 301
         zero state: 301
         any state: 0
     Total gates : 5.821E+5
       'Toffoli': 3.546E+4,
       'T': 792,
       'CNOT': 4.262E+5,
       'X': 1.833E+4,
       'Z': 475,
       'S': 880,
       'Hadamard': 9.995E+4
    """

    resource_keys = {"cmpr_walk_op", "num_iterations", "adj_qft_cmpr_op"}

    def __init__(
        self,
        walk_op: ResourceOperator,
        num_iterations: int,
        adj_qft_op: ResourceOperator | None = None,
        wires: WiresLike | None = None,
    ):
        remove_ops = [walk_op, adj_qft_op] if adj_qft_op is not None else [walk_op]
        _dequeue(remove_ops)
        self.queue()

        if not (isinstance(num_iterations, int) and num_iterations > 1):
            raise ValueError(
                f"Expected 'num_iterations' to be an integer greater than zero, got {num_iterations}"
            )

        if not isinstance(walk_op, (qre.Qubitization, qre.QubitizeTHC)):
            raise ValueError(
                f"Expected the 'walk_op' to be a qubitization type operator (an instance of 'Qubitization' or 'QubitizeTHC'), got {type(walk_op)}"
            )

        self.walk_op = walk_op.resource_rep_from_op()
        adj_qft_cmpr_op = None if adj_qft_op is None else adj_qft_op.resource_rep_from_op()

        self.adj_qft_cmpr_op = adj_qft_cmpr_op
        self.num_iterations = num_iterations

        self.num_wires = ceil_log2(num_iterations + 1) + walk_op.num_wires

        wires = Wires([]) if wires is None else Wires(wires)
        walk_wires = Wires([]) if walk_op.wires is None else walk_op.wires
        adj_qft_wires = (
            Wires([]) if (adj_qft_op is None or adj_qft_op.wires is None) else adj_qft_op.wires
        )

        all_wires = Wires.all_wires((wires, walk_wires, adj_qft_wires))
        if len(all_wires) == 0:
            self.wires = None
        elif len(all_wires) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(all_wires)}.")
        else:
            self.wires = all_wires

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * cmpr_walk_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                  A compressed resource operator corresponding to the quantum walk operator
                  to apply the phase estimation protocol on.
                * num_iterations (int): The total number of times the quantum walk operator
                  is applied in order to reach a target precision in the eigenvalue
                  estimate.
                * adj_qft_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp` | None):
                  An optional compressed resource operator corresponding to the adjoint QFT routine.
                  If :code:`None`, the default :class:`~.pennylane.estimator.templates.subroutines.QFT`
                  will be used.
        """

        return {
            "cmpr_walk_op": self.walk_op,
            "num_iterations": self.num_iterations,
            "adj_qft_cmpr_op": self.adj_qft_cmpr_op,
        }

    @classmethod
    def resource_rep(
        cls,
        cmpr_walk_op: CompressedResourceOp,
        num_iterations: int,
        adj_qft_cmpr_op: CompressedResourceOp | None = None,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            cmpr_walk_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                A compressed resource operator corresponding to the quantum walk operator
                to apply the phase estimation protocol on.
            num_iterations (int): The total number of times the quantum walk operator
                is applied in order to reach a target precision in the eigenvalue estimate.
            adj_qft_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp` | None):
                An optional compressed resource operator corresponding to the adjoint QFT routine.
                If :code:`None`, the default :class:`~.pennylane.estimator.templates.subroutines.QFT`
                will be used.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        params = {
            "cmpr_walk_op": cmpr_walk_op,
            "num_iterations": num_iterations,
            "adj_qft_cmpr_op": adj_qft_cmpr_op,
        }
        num_wires = ceil_log2(num_iterations + 1) + cmpr_walk_op.num_wires
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        cmpr_walk_op: CompressedResourceOp,
        num_iterations: int,
        adj_qft_cmpr_op: CompressedResourceOp | None = None,
    ) -> list[GateCount | Allocate | Deallocate]:
        r"""Returns the resources for Quantum Phase Estimation implemented using unary iteration.

        Args:
            cmpr_walk_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                A compressed resource operator corresponding to the quantum walk operator
                to apply the phase estimation protocol on.
            num_iterations (int): The total number of times the quantum walk operator
                is applied in order to reach a target precision in the eigenvalue estimate.
            adj_qft_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp` | None):
                An optional compressed resource operator corresponding to the adjoint QFT routine.
                If :code:`None`, the default :class:`~.pennylane.estimator.templates.subroutines.QFT`
                will be used.

        Resources:
            The resources are obtained from Figure 2. in Section III of `arXiv.2011.03494 <https://arxiv.org/pdf/2011.03494>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        num_wires = ceil_log2(num_iterations + 1)

        # extract prep and select from walk operator:
        prep_op = cmpr_walk_op.params["prep_op"]
        select_op = cmpr_walk_op.params["select_op"]

        # build controlled reflection:
        reflection_operator = resource_rep(
            qre.Reflection,
            {
                "num_wires": prep_op.num_wires,
                "alpha": math.pi,
                "cmpr_U": prep_op,
            },
        )
        ctrl_ref_operator = resource_rep(
            qre.Controlled,
            {"base_cmpr_op": reflection_operator, "num_ctrl_wires": 1, "num_zero_ctrl": 0},
        )

        hadamard = resource_rep(qre.Hadamard)
        x = resource_rep(qre.X)
        cnot = resource_rep(qre.CNOT)
        left_elbow = resource_rep(qre.Toffoli, {"elbow": "left"})
        right_elbow = resource_rep(qre.Toffoli, {"elbow": "right"})

        if adj_qft_cmpr_op is None:
            adj_qft_cmpr_op = resource_rep(
                qre.Adjoint,
                {
                    "base_cmpr_op": resource_rep(QFT, {"num_wires": num_wires}),
                },
            )

        return [
            Allocate(num_wires - 1),
            GateCount(hadamard, num_wires),
            GateCount(left_elbow, num_iterations - 1),
            GateCount(cnot, num_iterations - 1),
            GateCount(x, 2 * (num_iterations - 1) + 2),
            GateCount(ctrl_ref_operator, num_iterations + 1),
            GateCount(select_op, num_iterations),
            GateCount(right_elbow, num_iterations - 1),
            GateCount(adj_qft_cmpr_op),
            Deallocate(num_wires - 1),
        ]

    @staticmethod
    def tracking_name(
        cmpr_walk_op: CompressedResourceOp,
        num_iterations: int,
        adj_qft_cmpr_op: CompressedResourceOp | None = None,
    ) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = cmpr_walk_op.name
        adj_qft_name = None if adj_qft_cmpr_op is None else adj_qft_cmpr_op.name
        return f"UnaryIterationQPE({base_name}, {num_iterations}, adj_qft={adj_qft_name})"


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
                raise ValueError("Must provide at least one of `num_wires` and `wires`.")
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
                raise ValueError("Must provide at least one of `num_wires` and `wires`.")
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
                raise ValueError("Must provide at least one of `dim` and `wires`.")
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


class HybridQRAM(ResourceOperator):
    r"""Resource class for HybridQRAM.

    Args:
        data (TensorLike | Sequence[str]):
            The classical memory to retrieve values from.
        num_wires (int):
            The number of qubits the operation acts upon.
        num_select_wires (int):
            The number of "select" bits taken from ``control_wires``.
        num_control_wires (int):
            The number of ``control_wires`` including select and tree control wires.
        control_wires (WiresLike):
            The register that stores the index for the entry of the classical data we want to
            access.
        target_wires (WiresLike):
            The register in which the classical data gets loaded. The size of this register must
            equal each bitstring length in ``bitstrings``.
        work_wires (WiresLike):
            The additional wires required to funnel the desired entry of ``bitstrings`` into the
            ``target_wires`` register. The ``work_wires`` register includes the signal, bus,
            direction, left port and right port wires in that order for a tree of depth
            :math:`(n-k)`. For more details, consult
            `section 3 of arXiv:2306.03242 <https://arxiv.org/abs/2306.03242>`__.

    Raises:
        ValueError: if the number of wires provided does not match ``num_wires``

    Resources:
        The resources are obtained from the HybridQRAM implementation in PennyLane. Please find more
        details about the algorithm in `Systems Architecture for Quantum Random Access Memory <https://arxiv.org/abs/2306.03242>`_.

    .. seealso:: :class:`~.HybridQRAM`
    """

    resource_keys = {"data", "num_target_wires", "num_select_wires", "num_control_wires"}

    def __init__(
        self,
        data,
        num_wires,
        num_select_wires,
        num_control_wires,
        control_wires=None,
        target_wires=None,
        work_wires=None,
    ):
        all_wires = None
        if control_wires and target_wires and work_wires:
            all_wires = list(control_wires) + list(target_wires) + list(work_wires)
            assert num_control_wires == len(control_wires)
            if len(all_wires) != num_wires:
                raise ValueError(f"Expected {num_wires} wires, got {len(all_wires)}.")

        if isinstance(data, (list, tuple)):
            data = pl_math.array(data)
        if isinstance(data[0], str):
            data = pl_math.array(list(map(lambda bitstring: [int(bit) for bit in bitstring], data)))
        self.data = data

        self.num_wires = num_wires
        self.num_select_wires = num_select_wires
        self.num_control_wires = num_control_wires
        super().__init__(wires=all_wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters.
                * data (TensorLike | Sequence[str]): the classical memory to retrieve values from
                * num_wires (int): the number of qubits the operation acts upon
                * num_select_wires (int): the number of select wires
                * num_tree_control_wires (int): the number of ``work_wires`` minus the number of select wires
        """
        return {
            "data": self.data,
            "num_wires": self.num_wires,
            "num_select_wires": self.num_select_wires,
            "num_tree_control_wires": self.num_control_wires - self.num_select_wires,
        }

    @classmethod
    def resource_rep(cls, data, num_wires, num_select_wires, num_tree_control_wires):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            data (TensorLike | Sequence[str]): the classical memory to retrieve values from
            num_wires (int): the number of qubits the operation acts upon
            num_select_wires (int): the number of select wires
            num_tree_control_wires (int): the number of ``work_wires`` minus the number of select wires

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        params = {
            "data": data,
            "num_wires": num_wires,
            "num_select_wires": num_select_wires,
            "num_tree_control_wires": num_tree_control_wires,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, data, num_wires, num_select_wires, num_tree_control_wires):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            data (TensorLike | Sequence[str]): the classical memory to retrieve values from
            num_wires (int): the number of qubits the operation acts upon
            num_select_wires (int): the number of select wires
            num_tree_control_wires (int): the number of ``work_wires`` minus the number of select wires

        Resources:
            The resources are obtained from the HybridQRAM implementation in PennyLane. Please find more
            details about the algorithm in `Systems Architecture for Quantum Random Access Memory <https://arxiv.org/abs/2306.03242>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears in the decomposition.
        """
        num_blocks = 1 << num_select_wires
        num_target_wires = data.shape[1]

        paulix = resource_rep(qre.X)
        cnot = resource_rep(qre.CNOT)
        cswap_one = resource_rep(qre.CSWAP)
        cswap_zero = qre.Controlled.resource_rep(resource_rep(qre.SWAP), 1, 1)
        ccswap_zero = qre.Controlled.resource_rep(cswap_zero, 1, 0)
        ccswap_one = qre.Controlled.resource_rep(cswap_one, 1, 0)
        ch = qre.Controlled.resource_rep(resource_rep(qre.Hadamard), 1, 0)
        cz = qre.Controlled.resource_rep(resource_rep(qre.Z), 1, 0)

        paulix_counts = (num_select_wires <= 0) * num_blocks * 2
        cswap_counts = (
            (num_tree_control_wires + (1 << num_tree_control_wires) - 1) * 2 + 2 * num_target_wires
        ) * num_blocks
        ccswap_count = (
            (
                ((1 << num_tree_control_wires) - 1 - num_tree_control_wires)
                + ((1 << num_tree_control_wires) - 1) * num_target_wires
            )
            * num_blocks
            * 2
        )
        ch_count = num_target_wires * num_blocks * 2

        cnot_count = 0
        cz_count = 0
        cnot_zeroes = defaultdict(int)

        for block_index in range(num_blocks):
            zero_control_values = [
                (block_index >> (num_select_wires - 1 - i)) & 1 for i in range(num_select_wires)
            ].count(0)
            if zero_control_values == 0:
                cnot_count += (num_select_wires > 0) * 2
            else:
                cnot_zeroes[
                    qre.Controlled.resource_rep(
                        resource_rep(qre.X), num_select_wires, zero_control_values
                    )
                ] += (num_select_wires > 0) * 2

        cz_count = int(pl_math.sum(data))

        ret = [
            GateCount(cswap_one, cswap_counts),
            GateCount(ccswap_zero, ccswap_count),
            GateCount(ccswap_one, ccswap_count),
            GateCount(ch, ch_count),
            GateCount(cz, cz_count),
        ]

        for rep, count in cnot_zeroes.items():
            ret.append(GateCount(rep, count))
        if cnot_count != 0:
            ret.append(GateCount(cnot, cnot_count))
        if paulix_counts != 0:
            ret.append(GateCount(paulix, paulix_counts))

        return ret

    @staticmethod
    def tracking_name(data, num_wires, num_select_wires, num_tree_control_wires) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return f"HybridQRAM({data}, {num_wires}, {num_select_wires}, {num_tree_control_wires})"


class BBQRAM(ResourceOperator):
    r"""Resource class for BBQRAM.

    Args:
        num_bitstrings (int): the size of the classical memory array to retrieve values from
        size_bitstring (int): the length of the individual bitstrings in the classical memory
        num_bit_flips (int): the number of 1s in the classical memory
        num_wires (int): the number of qubits the operation acts upon
        control_wires (WiresLike): The register that stores the index for the entry of the classical data we want to
            access.
        target_wires (WiresLike):
            The register in which the classical data gets loaded. The size of this register must
            equal each bitstring length in ``bitstrings``.
        work_wires (WiresLike): The additional wires required to funnel the desired entry of ``bitstrings`` into the
            target register.

    Raises:
        ValueError: if the number of wires provided does not match ``num_wires``

    Resources:
        The resources are obtained from the BBQRAM implementation in PennyLane. The original publication of
        the algorithm can be found in `Quantum Random Access Memory <https://arxiv.org/abs/0708.1879>`_.

    .. seealso:: :class:`~.BBQRAM`
    """

    resource_keys = {"num_bitstrings", "size_bitstring", "num_bit_flips", "num_wires"}

    def __init__(
        self,
        num_bitstrings,
        size_bitstring,
        num_wires,
        num_bit_flips=None,
        control_wires=None,
        target_wires=None,
        work_wires=None,
    ):
        all_wires = None
        if control_wires and target_wires and work_wires:
            all_wires = list(control_wires) + list(target_wires) + list(work_wires)
            if len(all_wires) != num_wires:
                raise ValueError(f"Expected {num_wires} wires, got {len(all_wires)}.")
        if num_bit_flips is None:
            num_bit_flips = num_bitstrings * size_bitstring // 2
        self.num_wires = num_wires
        self.num_bitstrings = num_bitstrings
        self.size_bitstring = size_bitstring
        self.num_bit_flips = num_bit_flips
        super().__init__(wires=all_wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int): the number of qubits the operation acts upon
                * num_bitstrings (int): the size of the classical memory array to retrieve values from
                * size_bitstring (int): the length of the individual bitstrings in the classical memory
                * num_bit_flips (int): the number of 1s in the classical memory
        """
        return {
            "num_wires": self.num_wires,
            "num_bitstrings": self.num_bitstrings,
            "size_bitstring": self.size_bitstring,
            "num_bit_flips": self.num_bit_flips,
        }

    @classmethod
    def resource_rep(cls, num_bitstrings, size_bitstring, num_bit_flips, num_wires):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            num_bitstrings (int): the size of the classical memory array to retrieve values from
            size_bitstring (int): the length of the individual bitstrings in the classical memory
            num_bit_flips (int): the number of 1s in the classical memory
            num_wires (int): the number of qubits the operation acts upon

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        params = {
            "num_bitstrings": num_bitstrings,
            "size_bitstring": size_bitstring,
            "num_bit_flips": num_bit_flips,
            "num_wires": num_wires,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, num_bitstrings, size_bitstring, num_bit_flips, num_wires):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_bitstrings (int): the size of the classical memory array to retrieve values from
            size_bitstring (int): the length of the individual bitstrings in the classical memory
            num_bit_flips (int): the number of 1s in the classical memory
            num_wires (int): the number of qubits the operation acts upon

        Resources:
            The resources are obtained from the BBQRAM implementation in PennyLane. The original publication of
            the algorithm can be found in `Quantum Random Access Memory <https://arxiv.org/abs/0708.1879>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        num_target_wires = size_bitstring
        n_k = (num_bitstrings - 1).bit_length()

        swap = resource_rep(qre.SWAP)
        cswap = resource_rep(qre.CSWAP)
        hadamard = resource_rep(qre.Hadamard)
        pauliz = resource_rep(qre.Z)

        swap_counts = ((1 << n_k) - 1 + n_k) * 2 + num_target_wires * 2
        cswap_counts = ((1 << n_k) - 1) * num_target_wires * 4 + ((1 << n_k) - 1 - n_k) * 4
        hadamard_counts = num_target_wires * 2

        pauliz_counts = num_bit_flips

        return [
            GateCount(swap, swap_counts),
            GateCount(hadamard, hadamard_counts),
            GateCount(cswap, cswap_counts),
            GateCount(pauliz, pauliz_counts),
        ]

    @staticmethod
    def tracking_name(num_bitstrings, size_bitstring, num_bit_flips, num_wires) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return f"BBQRAM({num_bitstrings}, {size_bitstring}, {num_bit_flips}, {num_wires})"


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
        num_ctrl_wires = ceil_log2(num_select_ops)

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
        work_qubits = ceil_log2(num_ops) - 1

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
        num_ctrl_wires = ceil_log2(num_ops)
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
        num_ctrl_wires = ceil_log2(len(cmpr_ops))
        fewest_unique_wires = max(op.num_wires for op in cmpr_ops)

        num_wires = num_wires or fewest_unique_wires + num_ctrl_wires
        params = {"cmpr_ops": cmpr_ops, "num_wires": num_wires}
        return CompressedResourceOp(cls, num_wires, params)


class QROM(ResourceOperator):
    r"""Resource class for the Quantum Read-Only Memory (QROM) template.

    Args:
        num_bitstrings (int): the number of bitstrings that are to be encoded
        size_bitstring (int): the length of each bitstring
        num_bit_flips (int | None): The total number of :math:`1`'s in the dataset. Defaults to
            :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
        restored (bool): Determine if allocated qubits should be reset after the computation
            (at the cost of higher gate counts). Defaults to :code:`True`.
        select_swap_depth (int | None): A parameter :math:`\lambda` that determines
            if data will be loaded in parallel by adding more rows following Figure 1.C of
            `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be :code:`None`,
            :code:`1` or a positive integer power of two. Defaults to ``None``, which sets the
            depth that minimizes T-gate count.
        wires (WiresLike | None): The wires the operation acts on (control and target), excluding
            any additional qubits allocated during the decomposition (e.g select-swap wires).

    Resources:
        The resources for QROM are derived from the following references:

        * :code:`restored=False`: Uses the Select-Swap tree decomposition from Figure 1.C of
          `Low et al. (2018) <https://arxiv.org/abs/1812.00954>`_, further optimized using the
          measurement-based uncomputation technique described in
          `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`__.

        * :code:`restored=True`: Uses the standard QROM resource accounting from Figure 4 of
          `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`__.

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
    Total gates : 85
    'Toffoli': 8,
    'CNOT': 36,
    'X': 17,
    'Hadamard': 24
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
        if opt_width_continuous < 1:
            # The continuous solution could be non-physical
            w1 = w2 = 1
        else:
            w1 = 2 ** int(math.floor(math.log2(opt_width_continuous)))
            w2 = 2 ** ceil_log2(opt_width_continuous)

        def t_cost_func(w):
            return 4 * (math.ceil(num_bitstrings / w) - 2) + 6 * (w - 1) * size_bitstring

        if t_cost_func(w2) < t_cost_func(w1):
            return w2
        return w1

    def __init__(
        self,
        num_bitstrings: int,
        size_bitstring: int,
        num_bit_flips: int | None = None,
        restored: bool = True,
        select_swap_depth: int | None = None,
        wires: WiresLike | None = None,
    ) -> None:
        self.restored = restored
        self.num_bitstrings = num_bitstrings
        self.size_bitstring = size_bitstring
        self.num_bit_flips = num_bit_flips or (num_bitstrings * size_bitstring // 2)

        self.num_control_wires = ceil_log2(num_bitstrings)
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
        num_bitstrings: int,
        size_bitstring: int,
        num_bit_flips: int | None = None,
        select_swap_depth: int | None = None,
        restored: bool = True,
    ) -> list[GateCount]:
        r"""Returns a list of ``GateCount`` objects representing the operator's resources.

        Args:
            num_bitstrings (int): the number of bitstrings that are to be encoded
            size_bitstring (int): the length of each bitstring
            num_bit_flips (int | None): The total number of :math:`1`'s in the dataset. Defaults to
                :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
            select_swap_depth (int | None): A parameter :math:`\lambda` that determines
                if data will be loaded in parallel by adding more rows following Figure 1.C of
                `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be :code:`None`,
                :code:`1` or a positive integer power of two. Defaults to ``None``, which sets the
                depth that minimizes T-gate count.
            restored (bool): Determine if allocated qubits should be reset after the computation
                (at the cost of higher gate counts). Defaults to :code:`True`.

        Resources:
            The resources for QROM are derived from the following references:

            * :code:`restored=False`: Uses the Select-Swap tree decomposition from Figure 1.C of
              `Low et al. (2018) <https://arxiv.org/abs/1812.00954>`_, further optimized using the
              measurement-based uncomputation technique described in
              `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`__.

            * :code:`restored=True`: Uses the standard QROM resource accounting from Figure 4 of
              `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`__.

            Note: we use the unary iterator trick to implement the ``Select``. This
            implementation assumes we have access to :math:`n - 1` additional
            work qubits, where :math:`n = \left\lceil \log_{2}(N) \right\rceil` and :math:`N` is
            the number of batches of unitaries to select.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        if select_swap_depth:
            max_depth = 2 ** ceil_log2(num_bitstrings)
            select_swap_depth = min(max_depth, select_swap_depth)  # truncate depth beyond max depth

        W_opt = select_swap_depth or cls._t_optimized_select_swap_width(
            num_bitstrings, size_bitstring
        )
        L_opt = math.ceil(num_bitstrings / W_opt)
        l = ceil_log2(L_opt)

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

        if restored and (W_opt > 1):
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
        if W_opt > 1:
            ctrl_swap = resource_rep(qre.CSWAP)
            gate_cost.append(
                GateCount(ctrl_swap, swap_restored_prefactor * (W_opt - 1) * size_bitstring)
            )

            if not restored:
                gate_cost.append(GateCount(x, (W_opt - 1) * size_bitstring))  # measure and reset
            gate_cost.append(Deallocate((W_opt - 1) * size_bitstring))  # release Swap registers

        return gate_cost

    @classmethod
    def single_controlled_res_decomp(
        cls,
        num_bitstrings: int,
        size_bitstring: int,
        num_bit_flips: int | None = None,
        select_swap_depth: int | None = None,
        restored: bool = True,
    ):
        r"""The resource decomposition for QROM controlled on a single wire."""
        if select_swap_depth:
            max_depth = 2 ** ceil_log2(num_bitstrings)
            select_swap_depth = min(max_depth, select_swap_depth)  # truncate depth beyond max depth

        W_opt = select_swap_depth or qre.QROM._t_optimized_select_swap_width(
            num_bitstrings, size_bitstring
        )
        L_opt = math.ceil(num_bitstrings / W_opt)
        l = ceil_log2(L_opt)

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

        if restored and (W_opt > 1):
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
        if W_opt > 1:
            w = ceil_log2(W_opt)
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

            Note: we use the single-controlled unary iterator trick to implement the ``Select``. This
            implementation assumes we have access to :math:`n` additional work qubits,
            where :math:`n = \lceil \log_{2}(N) \rceil` and :math:`N` is the number of batches of
            unitaries to select.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
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
                * num_bit_flips (int | None): The total number of :math:`1`'s in the dataset.
                  Defaults to :code:`(num_bitstrings * size_bitstring) // 2`, which is half the
                  dataset.
                * restored (bool): Determine if allocated qubits should be reset after the
                  computation (at the cost of higher gate counts). Defaults to :code:`True`.
                * select_swap_depth (int | None): A parameter :math:`\lambda` that
                  determines if data will be loaded in parallel by adding more rows following
                  Figure 1.C of `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be
                  :code:`None`, :code:`1` or a positive integer power of two. Defaults to None,
                  which sets the depth that minimizes T-gate count.

        """

        return {
            "num_bitstrings": self.num_bitstrings,
            "size_bitstring": self.size_bitstring,
            "num_bit_flips": self.num_bit_flips,
            "select_swap_depth": self.select_swap_depth,
            "restored": self.restored,
        }

    @classmethod
    def _ctrl_T(cls, num_data_blocks: int, num_bit_flips: int, count: int = 1) -> list[GateCount]:
        """Constructs the control-``T`` subroutine as defined in Appendices A and B of
        `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`_.

        Args:
            num_data_blocks(int): The number of data blocks formed by partitioning the total bitstrings based on select-swap depth.
            num_bit_flips (int): The total number of :math:`1`'s in the dataset.
            count (int): The number of times to repeat the subroutine.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: The resource decomposition of the control- :math:`T` subroutine.
        """

        x = resource_rep(qre.X)
        cnot = resource_rep(qre.CNOT)
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})

        gate_cost = []

        if num_data_blocks > 1:
            gate_cost.append(
                GateCount(x, count * (2 * (num_data_blocks - 2) + 1))
            )  # conjugate 0 controlled toffolis + 1 extra X gate from un-controlled unary iterator decomp
            gate_cost.append(
                GateCount(
                    cnot,
                    count * (num_data_blocks - 2) + count * num_bit_flips,
                )  # num CNOTs in unary iterator trick + each unitary in the select is just a CNOT
            )
            gate_cost.append(GateCount(l_elbow, count * (num_data_blocks - 2)))
            gate_cost.append(GateCount(r_elbow, count * (num_data_blocks - 2)))

        else:
            gate_cost.append(
                GateCount(
                    x, count * num_bit_flips
                )  # each unitary in the select is just an X gate to load the data
            )
        return gate_cost

    @classmethod
    def _ctrl_S(cls, num_ctrl_wires: int, count: int = 1) -> list[GateCount]:
        """Constructs the control-S subroutine as defined in Figure 8 of
        `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`_ excluding the initial ``X`` gate.

        Args:
            num_ctrl_wires (int): The number of control wires.
            count (int): The number of times to repeat the subroutine.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: The resource decomposition of the control- :math:`S` subroutine.
        """
        num_ctrl_swaps = 2**num_ctrl_wires - 1
        return [qre.GateCount(qre.resource_rep(qre.CSWAP), count * num_ctrl_swaps)]

    @classmethod
    def _ctrl_S_adj(cls, num_ctrl_wires: int, count: int = 1) -> list[GateCount]:
        r"""Constructs the control-S^adj subroutine as defined in Figure 10
        of `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`_ excluding the terminal ``X`` gate.

        Args:
            num_ctrl_wires (int): The number of control wires.
            count (int): The number of times to repeat the subroutine.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: The resource decomposition of the control- :math:`S^{\dagger}` subroutine.

        """
        h = qre.resource_rep(qre.Hadamard)
        cz = qre.resource_rep(qre.CZ)
        cnot = qre.resource_rep(qre.CNOT)

        num_ops = 2**num_ctrl_wires - 1
        return [
            qre.GateCount(h, count * num_ops),
            qre.GateCount(cz, count * num_ops),
            qre.GateCount(cnot, count * num_ops),
        ]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources of the adjoint of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Args:
            target_resource_params(dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            This resources are based on Appendix C of `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        num_bitstrings = target_resource_params["num_bitstrings"]
        size_bitstring = target_resource_params["size_bitstring"]
        num_bit_flips = target_resource_params.get("num_bit_flips", None)
        select_swap_depth = target_resource_params.get("select_swap_depth", None)
        restored = target_resource_params.get("restored", True)

        gate_lst = []
        x = resource_rep(qre.X)
        z = resource_rep(qre.Z)
        had = qre.resource_rep(qre.Hadamard)

        # Compute the width (output + swap registers) and length (unary iter entries) of the QROM
        if select_swap_depth:
            max_depth = 2 ** ceil_log2(num_bitstrings)
            select_swap_depth = min(max_depth, select_swap_depth)  # truncate depth beyond max depth

        k = select_swap_depth or qre.QROM._t_optimized_select_swap_width(
            num_bitstrings, size_bitstring
        )
        num_qubits_l = ceil_log2(k)  # number of qubits in |l> register

        num_cols = math.ceil(num_bitstrings / k)  # number of columns of data
        num_qubits_h = ceil_log2(num_cols)  # number of qubits in |h> register

        ## Measure output register, reset qubits and construct fixup table
        gate_lst.append(qre.GateCount(had, size_bitstring))  # Figure 5.

        ## Allocate auxiliary qubits
        num_alloc_wires = k  # Swap registers
        if num_cols > 1:
            num_alloc_wires += num_qubits_h - 1  # + work_wires for UI trick

        gate_lst.append(qre.Allocate(num_alloc_wires))

        ## Cost assuming clean auxiliary qubits (Figure 6)
        if not restored:
            gate_lst.append(GateCount(x, 2))
            gate_lst.append(GateCount(had, 2 * k))

            num_bit_flips = (k * num_cols) // 2

            ctrl_S_decomp = cls._ctrl_S(num_ctrl_wires=num_qubits_l)
            ctrl_S_adj_decomp = cls._ctrl_S_adj(num_ctrl_wires=num_qubits_l)
            ctrl_T_decomp = cls._ctrl_T(num_data_blocks=num_cols, num_bit_flips=num_bit_flips)

            gate_lst.extend(ctrl_S_decomp)
            gate_lst.extend(ctrl_S_adj_decomp)
            gate_lst.extend(ctrl_T_decomp)

        ## Cost assuming dirty auxiliary qubits (Figure 7)
        else:
            gate_lst.append(GateCount(z, 2))
            gate_lst.append(GateCount(had, 2))

            num_bit_flips = (k * num_cols) // 2
            count = 1 if k == 1 else 2
            ctrl_S_decomp = cls._ctrl_S(num_ctrl_wires=num_qubits_l, count=count)
            ctrl_S_adj_decomp = cls._ctrl_S_adj(num_ctrl_wires=num_qubits_l, count=count)
            ctrl_T_decomp = cls._ctrl_T(
                num_data_blocks=num_cols, num_bit_flips=num_bit_flips, count=count
            )

            gate_lst.extend(ctrl_S_decomp)
            gate_lst.extend(ctrl_S_adj_decomp)
            gate_lst.extend(ctrl_T_decomp)

        gate_lst.append(qre.Deallocate(num_alloc_wires))

        return gate_lst

    @classmethod
    def resource_rep(
        cls,
        num_bitstrings: int,
        size_bitstring: int,
        num_bit_flips: int | None = None,
        restored: bool = True,
        select_swap_depth: int | None = None,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_bitstrings (int): the number of bitstrings that are to be encoded
            size_bitstring (int): the length of each bitstring
            num_bit_flips (int | None): The total number of :math:`1`'s in the dataset. Defaults to
                :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
            restored (bool): Determine if allocated qubits should be reset after the computation
                (at the cost of higher gate counts). Defaults to :code:`True`.
            select_swap_depth (int | None): A parameter :math:`\lambda` that determines
                if data will be loaded in parallel by adding more rows following Figure 1.C of
                `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be :code:`None`,
                :code:`1` or a positive integer power of two. Defaults to ``None``, which sets the
                depth that minimizes T-gate count.

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
        num_wires = size_bitstring + ceil_log2(num_bitstrings)
        return CompressedResourceOp(cls, num_wires, params)


class SelectPauliRot(ResourceOperator):
    r"""Resource class for the uniformly controlled rotation gate.

    Args:
        rot_axis (str): the rotation axis used in the multiplexer
        num_ctrl_wires (int): the number of control wires of the multiplexer
        precision (float | None): the precision used in the single qubit rotations
        wires (WiresLike, None): the wires the operation acts on

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
        num_prec_wires = ceil_log2(math.pi / precision) + 1
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


class Reflection(ResourceOperator):
    r"""Resource class for the Reflection operator. Apply a reflection about a state :math:`|\Psi\rangle`.

    This operator works by providing an operation, :math:`U`, that prepares the desired state, :math:`\vert \Psi \rangle`,
    that we want to reflect about. We can also provide a reflection angle :math:`\alpha`
    to define the operation in a more generic form:

    .. math::

       R(U, \alpha) = -I + (1 - e^{i\alpha}) |\Psi\rangle \langle \Psi|

    Args:
        num_wires (int | None): The number of wires the operator acts on. If ``None`` is provided, the
            number of wires are inferred from the ``U`` operator.
        U (:class:`~.pennylane.estimator.resource_operator.ResourceOperator` | None): the operator that prepares the state :math:`|\Psi\rangle`
        alpha (float | None): the angle of the operator, should be between :math:`[0, 2\pi]`. Default is :math:`\pi`.
        wires (WiresLike | None): The wires the operation acts on.

    Resources:
        The resources are derived from the decomposition :math:`R(U, \alpha) = U R(\alpha) U^\dagger`.
        The center block :math:`R(\alpha) = -I + (1 - e^{i\alpha})|0\rangle\langle 0|` is implemented
        using a multi-controlled ``PhaseShift``.

        In the special case where :math:`\alpha = \pi`, the ``PhaseShift`` becomes a ``Z`` gate.
        If :math:`\alpha = 0` or :math:`\alpha = 2\pi`, the center block cancels out, leaving :math:`-I`.
        The cost for :math:`-I` is calculated as :math:`X Z X Z = -I`.

    Raises:
        ValueError: if ``alpha`` is not a float within the range ``[0, 2pi]``
        ValueError: if at least one of ``num_wires`` or ``U`` is not provided
        ValueError: if the wires provided don't match the number of wires expected by the operator

    .. seealso:: :class:`~.pennylane.Reflection`

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> U = qre.Hadamard(wires=0)
    >>> ref_op = qre.Reflection(U=U, alpha=0.1)
    >>> print(qre.estimate(ref_op))
    --- Resources: ---
     Total wires: 1
       algorithmic wires: 1
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 52
       'T': 44,
       'X': 4,
       'Z': 2,
       'Hadamard': 2

    """

    resource_keys = {"alpha", "num_wires", "cmpr_U"}

    def __init__(
        self,
        num_wires: int | None = None,
        U: ResourceOperator | None = None,
        alpha: float = math.pi,
        wires: WiresLike = None,
    ) -> None:
        self.queue()

        if not 0 <= alpha <= 2 * qnp.pi:
            raise ValueError(f"alpha must be within [0, 2pi], got {alpha}")
        self.alpha = alpha

        if U is None and num_wires is None:
            raise ValueError("Must provide at least one of `num_wires` or `U`")

        if U is not None:
            _dequeue([U])
            self.cmpr_U = U.resource_rep_from_op()
        else:
            self.cmpr_U = qre.resource_rep(qre.Identity)

        self.num_wires = num_wires
        if num_wires is None:
            self.num_wires = self.cmpr_U.num_wires

        if wires:
            self.wires = Wires(wires)
            if len(self.wires) != num_wires:
                raise ValueError(f"Expected {num_wires} wires, got {len(self.wires)}.")
        else:
            self.wires = None

    @classmethod
    def resource_decomp(
        cls,
        num_wires: int | None = None,
        alpha: float = math.pi,
        cmpr_U: CompressedResourceOp | None = None,
    ):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_wires (int): number of wires the operator acts on
            alpha (float): the angle of the operator, default is :math:`\pi`
            cmpr_U (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): the operator that prepares the state :math:`|\Psi\rangle`

        Resources:
            The resources are derived from the decomposition :math:`R(U, \alpha) = U R(\alpha) U^\dagger`.
            The center block :math:`R(\alpha)` is implemented as a multi-controlled ``PhaseShift`` sandwiched
            by ``X`` gates on the target wire.

            If :math:`\alpha = \pi`, the phase shift is replaced by a ``Z`` gate.
            If :math:`\alpha = 0` or :math:`\alpha = 2\pi`, the operator simplifies to :math:`-I`,
            which costs :math:`X Z X Z`.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_types = []
        x = qre.X.resource_rep()
        z = qre.Z.resource_rep()

        # -1 * Identity = GlobalPhase(Pi) == X * Z * X * Z
        gate_types.append(qre.GateCount(x, 2))
        gate_types.append(qre.GateCount(z, 2))
        if qnp.isclose(alpha % (2 * math.pi), 0):
            return gate_types

        base_op = z if qnp.isclose(alpha, math.pi) else qre.PhaseShift.resource_rep()
        adj_cmpr_U = qre.Adjoint.resource_rep(cmpr_U)

        gate_types.append(qre.GateCount(cmpr_U))  # conjugate with U, U dagger
        gate_types.append(qre.GateCount(x, 2))  # conjugate base op with Xs
        if num_wires > 1:
            ctrl_base_op = qre.Controlled.resource_rep(
                base_cmpr_op=base_op,
                num_ctrl_wires=num_wires - 1,
                num_zero_ctrl=num_wires - 1,
            )
            gate_types.append(qre.GateCount(ctrl_base_op))
        else:
            gate_types.append(qre.GateCount(base_op))

        gate_types.append(qre.GateCount(adj_cmpr_U))
        return gate_types

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params):
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            ``Reflection`` operators are always self-inverse operators. This together with the fact
            that this is a unitary operator implies that it is self-adjoint.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(**target_resource_params))]

    @classmethod
    def controlled_resource_decomp(cls, num_ctrl_wires, num_zero_ctrl, target_resource_params):
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in
                the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The controlled decomposition simplifies by observing that :math:`R(U, \alpha) = U R(\alpha) U^\dagger`
            is a change of basis. Thus, we only need to control the center block :math:`R(\alpha)`,
            not the :math:`U` or :math:`U^\dagger` operations.

            Controlling :math:`R(\alpha)` involves controlling the global phase :math:`-I` and the
            multi-controlled ``PhaseShift``. The global phase :math:`-I` is controlled using
            :math:`MCX \cdot Z \cdot MCX \cdot Z`.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_types = []
        alpha = target_resource_params["alpha"]
        cmpr_U = target_resource_params["cmpr_U"]
        num_wires = target_resource_params["num_wires"]

        base_op = (
            qre.Z.resource_rep() if qnp.isclose(alpha, math.pi) else qre.PhaseShift.resource_rep()
        )

        x = qre.X.resource_rep()
        mcx = qre.MultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires, num_zero_ctrl=num_zero_ctrl
        )
        z = qre.Z.resource_rep()
        adj_cmpr_U = qre.Adjoint.resource_rep(cmpr_U)

        # Controlled-GlobalPhase(Pi) == MCX * Z * MCX * Z
        gate_types.append(qre.GateCount(mcx, 2))
        gate_types.append(qre.GateCount(z, 2))
        if qnp.isclose(alpha % (2 * math.pi), 0):
            return gate_types

        gate_types.append(qre.GateCount(cmpr_U))

        if num_zero_ctrl == num_ctrl_wires:  # all zero controls
            gate_types.append(qre.GateCount(x, 2))  # conjugate base op with Xs
        else:
            num_zero_ctrl += 1  # else absorbe Xs into a one control to make it a zero control

        ctrl_base_op = qre.Controlled.resource_rep(  # extended the controls here
            base_cmpr_op=base_op,
            num_ctrl_wires=num_wires - 1 + num_ctrl_wires,
            num_zero_ctrl=num_wires - 1 + num_zero_ctrl,
        )
        gate_types.append(qre.GateCount(ctrl_base_op))

        gate_types.append(qre.GateCount(adj_cmpr_U))
        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int | None): number of wires the operator acts on
                * alpha (float | None): the angle of the operator, default is :math:`\pi`
                * cmpr_U (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp` | None): the operator that prepares the
                  state :math:`|\Psi\rangle`

        """
        return {"alpha": self.alpha, "num_wires": self.num_wires, "cmpr_U": self.cmpr_U}

    @classmethod
    def resource_rep(
        cls,
        num_wires: int | None = None,
        alpha: float = math.pi,
        cmpr_U: CompressedResourceOp | None = None,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_wires (int): number of wires the operator acts on
            alpha (float): the angle of the operator, default is :math:`\pi`
            cmpr_U (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): the operator that prepares the state :math:`|\Psi\rangle`

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        if not 0 <= alpha <= 2 * qnp.pi:
            raise ValueError(f"alpha must be within [0, 2pi], got {alpha}")

        if cmpr_U is None and num_wires is None:
            raise ValueError("Must provide atleast one of `num_wires` or `U`")

        params = {"alpha": alpha, "num_wires": num_wires, "cmpr_U": cmpr_U}
        return CompressedResourceOp(cls, num_wires, params)


class Qubitization(ResourceOperator):
    r"""Resource class for the Qubitization operator. The operator is also referred to as the Quantum Walk operator.

    The operator is constructed by encoding a Hamiltonian, written as a linear combination of unitaries, into a block encoding (see Figure 1 in
    `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_).

    .. math::
        Q =  \text{Prep}_{H}(2|0\rangle\langle 0| - I)\text{Prep}_{H}^{\dagger} \text{Sel}_{H}.

    Args:
        prep_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): the operator that prepares the coefficients of the LCU
        select_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): the operator that selectively applies the unitaries of the LCU
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        The resources are obtained from Equation 9 in `Babbush et al. (2018) <https://arxiv.org/abs/1805.03662>`_.
        Specifically, the walk operator is defined as :math:`W = R \cdot S`, where :math:`R` is a reflection about the state prepared by
        the ``Prepare`` operator, and :math:`S` is the ``Select`` operator. The cost is therefore one ``Select`` and one ``Reflection``.

    Raises:
        ValueError: if the wires provided don't match the number of wires expected by the operator

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> prep_op = qre.Hadamard(wires=0)
    >>> select_op = qre.Z(wires=0)
    >>> qw_op = qre.Qubitization(prep_op, select_op)
    >>> print(qre.estimate(qw_op))
    --- Resources: ---
     Total wires: 1
       algorithmic wires: 1
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 10
       'X': 4,
       'Z': 4,
       'Hadamard': 2

    """

    resource_keys = {"prep_op", "select_op"}

    def __init__(
        self, prep_op: ResourceOperator, select_op: ResourceOperator, wires: WiresLike = None
    ) -> None:
        self.queue()
        _dequeue([prep_op, select_op])

        self.prep_op = prep_op.resource_rep_from_op()
        self.select_op = select_op.resource_rep_from_op()
        self.num_wires = (
            select_op.num_wires
        )  # The Walk operator acts on the same set of wires as sel

        prep_wires = prep_op.wires or Wires([])
        sel_wires = select_op.wires or Wires([])
        prep_sel_wires = prep_wires + sel_wires

        if wires:
            self.wires = Wires(wires)
            if len(self.wires) != self.num_wires:
                raise ValueError(f"Expected {self.num_wires} wires, got {len(self.wires)}.")
        elif len(prep_sel_wires) == self.num_wires:  # inherit wires from prep and sel
            self.wires = prep_sel_wires
        else:
            self.wires = None

    @classmethod
    def resource_decomp(cls, prep_op: CompressedResourceOp, select_op: CompressedResourceOp):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            prep_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed representation for the operator that prepares
                the coefficients of the LCU.
            select_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed representation for the operator that selectively
                applies the unitaries of the LCU.

        Resources:
            The resources are obtained from Equation 9 in `Babbush et al. (2018) <https://arxiv.org/abs/1805.03662>`_.
            Specifically, the walk operator is defined as :math:`W = R \cdot S`, where :math:`R` is a reflection about the state prepared by
            the ``Prepare`` operator, and :math:`S` is the ``Select`` operator.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        num_wires = prep_op.num_wires  # reflection happens over the prep register
        ref_op = Reflection.resource_rep(num_wires=num_wires, alpha=math.pi, cmpr_U=prep_op)

        return [GateCount(select_op), GateCount(ref_op)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params):
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            ``Reflection`` operators are self-adjoint, and the ``Select`` operator is also self-adjoint.
            Thus the adjoint of this operator has the same resources, just applied in reverse order.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        prep_op = target_resource_params["prep_op"]
        select_op = target_resource_params["select_op"]

        num_wires = prep_op.num_wires  # reflection happens over the prep register
        ref_op = Reflection.resource_rep(num_wires=num_wires, alpha=math.pi, cmpr_U=prep_op)

        return [GateCount(ref_op), GateCount(select_op)]

    @classmethod
    def controlled_resource_decomp(cls, num_ctrl_wires, num_zero_ctrl, target_resource_params):
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are obtained directly from Figure 1 in
            `Babbush et al. (2018) <https://arxiv.org/abs/1805.03662>`_.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        prep_op = target_resource_params["prep_op"]
        select_op = target_resource_params["select_op"]

        num_wires = prep_op.num_wires  # reflection happens over the prep register
        ref_op = Reflection.resource_rep(num_wires=num_wires, alpha=math.pi, cmpr_U=prep_op)

        ctrl_select_op = qre.Controlled.resource_rep(
            base_cmpr_op=select_op,
            num_ctrl_wires=num_ctrl_wires,
            num_zero_ctrl=num_zero_ctrl,
        )
        ctrl_ref = qre.Controlled.resource_rep(
            base_cmpr_op=ref_op,
            num_ctrl_wires=num_ctrl_wires,
            num_zero_ctrl=num_zero_ctrl,
        )

        return [GateCount(ctrl_select_op), GateCount(ctrl_ref)]

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * prep_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): a compressed representation for the operator that
                  prepares the coefficients of the LCU
                * select_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): a compressed representation for the operator that
                  selectively applies the unitaries of the LCU
        """
        return {"prep_op": self.prep_op, "select_op": self.select_op}

    @classmethod
    def resource_rep(
        cls, prep_op: CompressedResourceOp, select_op: CompressedResourceOp
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            prep_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed representation for the operator that prepares
                the coefficients of the LCU.
            select_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed representation for the operator that selectively
                applies the unitaries of the LCU.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = select_op.num_wires
        params = {"prep_op": prep_op, "select_op": select_op}
        return CompressedResourceOp(cls, num_wires, params)
