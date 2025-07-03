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
from typing import Dict

from pennylane import numpy as qnp
from pennylane.labs import resource_estimation as re
from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

# pylint: disable=arguments-differ,protected-access,too-many-arguments,unused-argument,super-init-not-called


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
    >>> print(plre.estimate_resources(out_square))
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
        return CompressedResourceOp(cls, {"register_size": register_size})

    @classmethod
    def default_resource_decomp(cls, register_size, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            register_size (int): the size of the input register

        Resources:
            The resources are obtained from appendix G, lemma 7 in `PRX Quantum, 2, 040332 (2021)
            <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332>`_. Specifically,
            the resources are given as :math:`(n - 1)^2` Toffoli gates, and :math:`n` CNOT gates.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
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
        The phase gradient state is defined as an
        equal superposition of phaseshifts where each shift is progressively more precise. This
        is achieved by applying Hadamard gates to each qubit and then applying RZ-rotations to each
        qubit with progressively smaller rotation angle. The first three rotations can be compiled to
        a Z-gate, S-gate and a T-gate.

    **Example**

    The resources for this operation are computed using:

    >>> phase_grad = plre.ResourcePhaseGradient(num_wires=5)
    >>> gate_set={"Z", "S", "T", "RZ", "Hadamard"}
    >>> print(plre.estimate_resources(phase_grad, gate_set))
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
        return CompressedResourceOp(cls, {"num_wires": num_wires})

    @classmethod
    def default_resource_decomp(cls, num_wires, **kwargs):
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
            list[GateCount]: A list of GateCount objects, where each object
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
    >>> print(plre.estimate_resources(out_mul))
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
        return CompressedResourceOp(
            cls, {"a_num_qubits": a_num_qubits, "b_num_qubits": b_num_qubits}
        )

    @classmethod
    def default_resource_decomp(cls, a_num_qubits, b_num_qubits, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            a_num_qubits (int): the size of the first input register
            b_num_qubits (int): the size of the second input register
            wires (Sequence[int], optional): the wires the operation acts on

        Resources:
            The resources are obtained from appendix G, lemma 10 in `PRX Quantum, 2, 040332 (2021)
            <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332>`_.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
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
    >>> print(plre.estimate_resources(semi_add))
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
        return CompressedResourceOp(cls, {"max_register_size": max_register_size})

    @classmethod
    def default_resource_decomp(cls, max_register_size, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            max_register_size (int): the size of the larger of the two registers being added together

        Resources:
            The resources are obtained from figures 1 and 2 in `Gidney (2018)
            <https://quantum-journal.org/papers/q-2018-06-18-74/pdf/>`_.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
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
    def default_controlled_resource_decomp(
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
            list[GateCount]: A list of GateCount objects, where each object
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
    >>> print(plre.estimate_resources(basis_rot))
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
    def default_resource_decomp(cls, dim_N, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

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
            list[GateCount]: A list of GateCount objects, where each object
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
        return CompressedResourceOp(cls, params)

    @classmethod
    def tracking_name(cls, dim_N) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return f"BasisRotation({dim_N})"


class ResourceSelect(ResourceOperator):
    r"""Resource class for the Select gate.

    Args:
        select_ops (list[~.ResourceOperator]): the set of operations to select over
        wires (Sequence[int], optional): the wires the operation acts on

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
    >>> print(plre.estimate_resources(select_op))
    --- Resources: ---
    Total qubits: 4
    Total gates : 24
    Qubit breakdown:
     clean qubits: 1, dirty qubits: 0, algorithmic qubits: 3
    Gate breakdown:
     {'CNOT': 7, 'S': 2, 'Z': 1, 'Hadamard': 8, 'X': 4, 'Toffoli': 2}
    """

    resource_keys = {"cmpr_ops"}

    def __init__(self, select_ops, wires=None) -> None:
        self.queue(select_ops)
        num_select_ops = len(select_ops)
        num_ctrl_wires = math.ceil(math.log2(num_select_ops))

        try:
            cmpr_ops = tuple(op.resource_rep_from_op() for op in select_ops)
            self.cmpr_ops = cmpr_ops
        except AttributeError as error:
            raise ValueError(
                "All factors of the Select must be instances of `ResourceOperator` in order to obtain resources."
            ) from error

        if wires is not None:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            ops_wires = [op.wires for op in select_ops if op.wires is not None]
            if len(ops_wires) == 0:
                self.wires = None
                self.num_wires = max((op.num_wires for op in select_ops)) + num_ctrl_wires
            else:
                self.wires = Wires.all_wires(ops_wires)
                self.num_wires = len(self.wires) + num_ctrl_wires

    def queue(self, ops_to_remove, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        for op in ops_to_remove:
            context.remove(op)
        context.append(self)
        return self

    @classmethod
    def default_resource_decomp(cls, cmpr_ops, **kwargs):  # pylint: disable=unused-argument
        r"""The resources for a select implementation taking advantage of the unary iterator trick.

        Args:
            cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed
                representation, to be applied according to the selected qubits.

        Resources:
            The resources are based on the analysis in `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_ section III.A,
            'Unary Iteration and Indexed Operations'. See Figures 4, 6, and 7.

            Note: This implementation assumes we have access to :math:`n - 1` additional work qubits,
            where :math:`n = \ceil{log_{2}(N)}` and :math:`N` is the number of batches of unitaries
            to select.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
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
    def textbook_resources(cmpr_ops, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed
                representation, to be applied according to the selected qubits.

        Resources:
            The resources correspond directly to the definition of the operation. Specifically,
            for each operator in :code:`cmpr_ops`, the cost is given as a controlled version of the operator
            controlled on the associated bitstring.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
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

        """
        return {"cmpr_ops": self.cmpr_ops}

    @classmethod
    def resource_rep(cls, cmpr_ops) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed
                representation, to be applied according to the selected qubits.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"cmpr_ops": cmpr_ops}
        return CompressedResourceOp(cls, params)


class ResourceQROM(ResourceOperator):
    """Resource class for the QROM template.

    Args:
        num_bitstrings (int): the number of bitstrings that are to be encoded
        size_bitstring (int): the length of each bitstring
        num_bit_flips (int, optional): The total number of :math:`1`'s in the dataset. Defaults to
            :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
        clean (bool, optional): Determine if allocated qubits should be reset after the computation
            (at the cost of higher gate counts). Defaults to :code`True`.
        select_swap_depth (Union[int, None], optional): A natural number that determines if data
            will be loaded in parallel by adding more rows following Figure 1.C of `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_.
            Defaults to :code:`None`, which internally determines the optimal depth.
        wires (Sequence[int], optional): the wires the operation acts on

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
    >>> print(plre.estimate_resources(qrom))
    --- Resources: ---
    Total qubits: 11
    Total gates : 178.0
    Qubit breakdown:
     clean qubits: 3, dirty qubits: 0, algorithmic qubits: 8
    Gate breakdown:
     {'Hadamard': 56, 'X': 34, 'CNOT': 72.0, 'Toffoli': 16}

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

        if w1 < 1 and w2 < 1:
            return 1

        def t_cost_func(w):
            return 4 * (math.ceil(num_bitstrings / w) - 2) + 6 * (w - 1) * size_bitstring

        if t_cost_func(w2) < t_cost_func(w1) and w2 >= 1:
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
        self.num_bit_flips = num_bit_flips or (num_bitstrings * size_bitstring / 2)

        if wires is not None:
            self.num_wires = len(wires)
            assert self.num_wires > size_bitstring
            self.num_control_wires = self.num_wires - size_bitstring
            assert self.num_control_wires >= math.ceil(math.log2(num_bitstrings))

        else:
            self.num_control_wires = math.ceil(math.log2(num_bitstrings))
            self.num_wires = size_bitstring + self.num_control_wires

        self.select_swap_depth = select_swap_depth
        super().__init__(wires=wires)

    @classmethod
    def default_resource_decomp(
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
            select_swap_depth (Union[int, None], optional): A natural number that determines if data
                will be loaded in parallel by adding more rows following Figure 1.C of `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_.
                Defaults to :code:`None`, which internally determines the optimal depth.
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
            select_swap_depth = 2 ** math.floor(math.log2(select_swap_depth))
        W_opt = select_swap_depth or ResourceQROM._t_optimized_select_swap_width(
            num_bitstrings, size_bitstring
        )
        L_opt = math.ceil(num_bitstrings / W_opt)
        l = math.ceil(math.log2(L_opt))

        gate_cost = []
        gate_cost.append(
            AllocWires((W_opt - 1) * size_bitstring + (l - 1))
        )  # Swap registers + work_wires for UI trick

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
        gate_cost.append(
            GateCount(x, select_clean_prefactor * (2 * (L_opt - 2) + 1))
        )  # conjugate 0 controlled toffolis + 1 extra X gate from un-controlled unary iterator decomp
        gate_cost.append(
            GateCount(
                cnot, select_clean_prefactor * (L_opt - 2) + select_clean_prefactor * num_bit_flips
            )  # num CNOTs in unary iterator trick   +   each unitary in the select is just a CNOT
        )
        gate_cost.append(GateCount(l_elbow, select_clean_prefactor * (L_opt - 2)))
        gate_cost.append(GateCount(r_elbow, select_clean_prefactor * (L_opt - 2)))

        gate_cost.append(FreeWires(l - 1))  # release UI trick work wires

        # # SWAP cost:
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
        W_opt = select_swap_depth or ResourceQROM._t_optimized_select_swap_width(
            num_bitstrings, size_bitstring
        )
        L_opt = math.ceil(num_bitstrings / W_opt)
        l = math.ceil(math.log2(L_opt))

        gate_cost = []
        gate_cost.append(
            FreeWires((W_opt - 1) * size_bitstring + l)
        )  # Swap registers + work_wires for UI trick

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
        gate_cost.append(
            GateCount(x, select_clean_prefactor * (2 * (L_opt - 1)))
        )  # conjugate 0 controlled toffolis
        gate_cost.append(
            GateCount(
                cnot, select_clean_prefactor * (L_opt - 1) + select_clean_prefactor * num_bit_flips
            )  # num CNOTs in unary iterator trick   +   each unitary in the select is just a CNOT
        )
        gate_cost.append(GateCount(l_elbow, select_clean_prefactor * (L_opt - 1)))
        gate_cost.append(GateCount(r_elbow, select_clean_prefactor * (L_opt - 1)))

        gate_cost.append(FreeWires(l))  # release UI trick work wires

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
    def default_controlled_resource_decomp(
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
            select_swap_depth (Union[int, None], optional): A natural number that determines if data
                will be loaded in parallel by adding more rows following Figure 1.C of `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_.
                Defaults to :code:`None`, which internally determines the optimal depth.

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
            list[GateCount]: A list of GateCount objects, where each object
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
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_bitstrings (int): the number of bitstrings that are to be encoded
                * size_bitstring (int): the length of each bitstring
                * num_bit_flips (int, optional): The total number of :math:`1`'s in the dataset. Defaults to :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
                * clean (bool, optional): Determine if allocated qubits should be reset after the computation (at the cost of higher gate counts). Defaults to :code`True`.
                * select_swap_depth (Union[int, None], optional): A natural number that determines if data will be loaded in parallel by adding more rows following Figure 1.C of `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Defaults to :code:`None`, which internally determines the optimal depth.

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
    ) -> CompressedResourceOp:  # pylint: disable=too-many-arguments
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_bitstrings (int): the number of bitstrings that are to be encoded
            size_bitstring (int): the length of each bitstring
            num_bit_flips (int, optional): The total number of :math:`1`'s in the dataset. Defaults to
                :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
            clean (bool, optional): Determine if allocated qubits should be reset after the computation
                (at the cost of higher gate counts). Defaults to :code`True`.
            select_swap_depth (Union[int, None], optional): A natural number that determines if data
                will be loaded in parallel by adding more rows following Figure 1.C of `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_.
                Defaults to :code:`None`, which internally determines the optimal depth.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        if num_bit_flips is None:
            num_bit_flips = num_bitstrings * size_bitstring // 2

        params = {
            "num_bitstrings": num_bitstrings,
            "num_bit_flips": num_bit_flips,
            "size_bitstring": size_bitstring,
            "select_swap_depth": select_swap_depth,
            "clean": clean,
        }
        return CompressedResourceOp(cls, params)


class ResourceQubitUnitary(ResourceOperator):
    r"""Resource class for the QubitUnitary template.

    Args:
        num_wires (int): the number of qubits the operation acts upon
        precision (Union[float, None], optional): The precision used when preparing the single qubit
            rotations used to prepare the entries of the qubit unitary.
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are defined by combining the two equalities in `Möttönen and Vartiainen
        (2005), Fig 14 <https://arxiv.org/pdf/quant-ph/0504100>`_, we can express an :math:`n`-
        qubit unitary as four :math:`n - 1`-qubit unitaries and three multiplexed rotations
        via (:class:`~.labs.resource_estimation.ResourceSelectPauliRot`). Specifically, the cost
        is given by:

        * 1-qubit unitary, the cost is approximated as a single :code:`RZ` rotation.

        * 2-qubit unitary, the cost is approximated as four single qubit rotations and three :code:`CNOT`s.

        * 3-qubit unitary or more, the cost is given according to the reference above, recurrsively.


    .. seealso:: :class:`~.QubitUnitary`

    **Example**

    The resources for this operation are computed using:

    >>> mltplxr = plre.ResourceSelectPauliRot(
    ...     rotation_axis = "Y",
    ...     num_ctrl_wires = 4,
    ...     precision = 1e-3,
    ... )
    >>> print(plre.estimate_resources(mltplxr, plre.StandardGateSet))
    --- Resources: ---
     Total qubits: 5
     Total gates : 32
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 5
     Gate breakdown:
      {'RY': 16, 'CNOT': 16}
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
                * precision (Union[float, None], optional): The precision used when preparing the single qubit rotations used to prepare the entries of the qubit unitary.
        """
        return {"num_wires": self.num_wires, "precision": self.precision}

    @classmethod
    def resource_rep(cls, num_wires, precision) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_wires (int): the number of qubits the operation acts upon
            precision (Union[float, None], optional): The precision used when preparing the single
                qubit rotations used to prepare the entries of the qubit unitary.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"num_wires": num_wires, "precision": precision}
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(
        cls, num_wires, precision=None, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_wires (int): the number of qubits the operation acts upon
            precision (Union[float, None], optional): The precision used when preparing the single
                qubit rotations used to prepare the entries of the qubit unitary.

        Resources:
            The resources are defined by combining the two equalities in `Möttönen and Vartiainen
            (2005), Fig 14 <https://arxiv.org/pdf/quant-ph/0504100>`_, we can express an :math:`n`-
            qubit unitary as four :math:`n - 1`-qubit unitaries and three multiplexed rotations
            via (:class:`~.labs.resource_estimation.ResourceSelectPauliRot`). Specifically, the cost
            is given by:

            * 1-qubit unitary, the cost is approximated as a single :code:`RZ` rotation.

            * 2-qubit unitary, the cost is approximated as four single qubit rotations and three :code:`CNOT`s.

            * 3-qubit unitary or more, the cost is given according to the reference above, recurrsively.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_lst = []
        precision = precision or kwargs["config"]["precision_qubit_unitary"]

        one_qubit_decomp_cost = [GateCount(resource_rep(re.ResourceRZ, {"eps": precision}))]
        two_qubit_decomp_cost = [
            GateCount(resource_rep(re.ResourceRZ, {"eps": precision}), 4),
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
                },
            )
            multiplex_y = resource_rep(
                ResourceSelectPauliRot,
                {
                    "num_ctrl_wires": index,
                    "rotation_axis": "Y",
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
        (2005), Fig 7a <https://arxiv.org/abs/quant-ph/0504100>`_. Specifically, the resources are
        given as :math:`2^{n}` instances of the :code:`CNOT` gate and :math:`2^{n}` instances
        of the single qubit rotation gate (:code:`RX`, :code:`RY` or :code:`RZ`) depending on the
        :code:`rotation_axis`.

    .. seealso:: :class:`~.SelectPauliRot`

    **Example**

    The resources for this operation are computed using:

    >>> mltplxr = plre.ResourceSelectPauliRot(
    ...     rotation_axis = "Y",
    ...     num_ctrl_wires = 4,
    ...     precision = 1e-3,
    ... )
    >>> print(plre.estimate_resources(mltplxr, plre.StandardGateSet))
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
        the Operator that are needed to compute a resource estimation.

        Args:
            rotation_axis (str): the rotation axis used in the multiplexer
            num_ctrl_wires (int): the number of control wires of the multiplexer
            precision (float): the precision used in the single qubit rotations

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(
            cls,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "rotation_axis": rotation_axis,
                "precision": precision,
            },
        )

    @classmethod
    def default_resource_decomp(cls, num_ctrl_wires, rotation_axis, precision, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            rotation_axis (str): the rotation axis used in the multiplexer
            num_ctrl_wires (int): the number of control wires of the multiplexer
            precision (float): the precision used in the single qubit rotations
            wires (Sequence[int], optional): the wires the operation acts on

        Resources:
            The resources are obtained from the construction scheme given in `Möttönen and Vartiainen
            (2005), Fig 7a <https://arxiv.org/abs/quant-ph/0504100>`_. Specifically, the resources are
            given as :math:`2^{n}` instances of the :code:`CNOT` gate and :math:`2^{n}` instances
            of the single qubit rotation gate (:code:`RX`, :code:`RY` or :code:`RZ`) depending on the
            :code:`rotation_axis`.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        rotation_gate_map = {
            "X": re.ResourceRX,
            "Y": re.ResourceRY,
            "Z": re.ResourceRZ,
        }
        precision = precision or kwargs["config"]["precision_select_pauli_rot"]

        gate = resource_rep(rotation_gate_map[rotation_axis], {"eps": precision})
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
            (2025), Fig 4 <https://arxiv.org/pdf/2409.07332>`_. Specifically, the resources are
            use two :code:`~.labs.resource_estimation.ResourceQROM`'s to digitally load and unload
            the phase angles up to some precision. These are then applied using a single controlled
            :code:`~.labs.resource_estimation.ResourceSemiAdder`.

            .. note::

                This method assumes a phase gradient state is prepared on an auxiliary register.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = precision or kwargs["config"]["precision_select_pauli_rot"]
        num_prec_wires = abs(math.floor(math.log2(precision)))
        gate_lst = []

        qrom = resource_rep(
            re.ResourceQROM,
            {
                "num_bitstrings": 2**num_ctrl_wires,
                "num_bit_flips": 2**num_ctrl_wires * num_prec_wires / 2,
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
