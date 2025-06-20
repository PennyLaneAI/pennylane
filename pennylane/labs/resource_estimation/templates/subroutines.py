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

import pennylane as qml
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

# pylint: disable=arguments-differ, protected-access


class ResourceOutOfPlaceSquare(ResourceOperator):
    resource_keys = {"register_size"}

    def __init__(self, register_size: int, wires=None):
        self.register_size = register_size
        self.num_wires = 3 * register_size
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        return {"register_size": self.register_size}

    @classmethod
    def resource_rep(cls, register_size):
        return CompressedResourceOp(cls, {"register_size": register_size})

    @classmethod
    def default_resource_decomp(cls, register_size, **kwargs):
        gate_lst = []

        gate_lst.append(GateCount(resource_rep(re.ResourceToffoli), (register_size - 1) ** 2))
        gate_lst.append(GateCount(resource_rep(re.ResourceCNOT), register_size))

        return gate_lst


class ResourcePhaseGradient(ResourceOperator):
    resource_keys = {"num_wires"}

    def __init__(self, num_wires, wires=None):
        self.num_wires = num_wires
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        return {"num_wires": self.num_wires}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        return CompressedResourceOp(cls, {"num_wires": num_wires})

    @classmethod
    def default_resource_decomp(cls, num_wires, **kwargs):
        gate_counts = []
        if num_wires > 0:
            gate_counts.append(GateCount(resource_rep(re.ResourceZ)))

        if num_wires > 1:
            gate_counts.append(GateCount(resource_rep(re.ResourceS)))

        if num_wires > 2:
            gate_counts.append(GateCount(resource_rep(re.ResourceT)))

        if num_wires > 3:
            gate_counts.append(GateCount(resource_rep(re.ResourceRZ), num_wires - 3))

        return gate_counts


class ResourceQFT(ResourceOperator):
    r"""Resource class for QFT.

    Args:
        num_wires (int): the number of qubits the operation acts upon
        wires (Iterable[Number, str], optional]): the wire(s) the operation acts on

    Resources:
        The resources are obtained from the standard decomposition of QFT as presented
        in (chapter 5) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum Information
        <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.

    .. seealso:: :class:`~.QFT`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceQFT.resources(num_wires=3)
    {Hadamard: 3, SWAP: 1, ControlledPhaseShift: 3}
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
        the Operator that are needed to compute a resource estimation.

        Args:
            num_wires (int): the number of qubits the operation acts upon

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"num_wires": num_wires}
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(cls, num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            num_wires (int): the number of qubits the operation acts upon

        Resources:
            The resources are obtained from the standard decomposition of QFT as presented
            in (Chapter 5) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum Information
            <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.

        """
        hadamard = resource_rep(re.ResourceHadamard)
        swap = resource_rep(re.ResourceSWAP)
        ctrl_phase_shift = resource_rep(re.ResourceControlledPhaseShift)

        return [
            GateCount(hadamard, num_wires),
            GateCount(swap, num_wires // 2),
            GateCount(ctrl_phase_shift, num_wires * (num_wires - 1) // 2),
        ]

    @classmethod
    def resources_via_phase_gradient(cls, num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.
        The number of wires used is 2 * num_wires.

        Args:
            num_wires (int): the number of qubits the operation acts upon

        Resources:
            The resources are obtained from the `Efficient Controlled Phase Gradients post
            <https://algassert.com/post/1708>`_.

        """
        t = resource_rep(re.ResourceT)
        t_counter = 0

        for k in range(1, num_wires):
            # SemiAdder T-cost estimation. Deduce based in image 1 and non-simetrics cnots: https://arxiv.org/pdf/1709.06648
            # TODO: Update once we have qml.SemiAdder
            t_counter += 2 * (2 * (k - 1)) + 4 * (2 * k - 1)

        hadamard = resource_rep(re.ResourceHadamard)
        return [GateCount(hadamard, num_wires), GateCount(t, t_counter)]

    @staticmethod
    def tracking_name(num_wires) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return f"QFT({num_wires})"


class ResourceOutMultiplier(ResourceOperator):
    resource_keys = {"a_num_qubits", "b_num_qubits"}

    def __init__(self, a_num_qubits, b_num_qubits, wires=None) -> None:
        self.num_wires = a_num_qubits + b_num_qubits + 2 * max((a_num_qubits, b_num_qubits))
        self.a_num_qubits = a_num_qubits
        self.b_num_qubits = b_num_qubits
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        return {"a_num_qubits": self.a_num_qubits, "b_num_qubits": self.b_num_qubits}

    @classmethod
    def resource_rep(cls, a_num_qubits, b_num_qubits) -> CompressedResourceOp:
        return CompressedResourceOp(
            cls, {"a_num_qubits": a_num_qubits, "b_num_qubits": b_num_qubits}
        )

    @classmethod
    def default_resource_decomp(
        cls, a_num_qubits, b_num_qubits, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
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
    resource_keys = {"max_register_size"}

    def __init__(self, max_register_size, wires=None):
        self.max_register_size = max_register_size
        self.num_wires = 2 * max_register_size
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        return {"max_register_size": self.max_register_size}

    @classmethod
    def resource_rep(cls, max_register_size):
        return CompressedResourceOp(cls, {"max_register_size": max_register_size})

    @classmethod
    def default_resource_decomp(cls, max_register_size, **kwargs):
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
        if (max_register_size > 2) and (ctrl_num_ctrl_wires == 1):
            cnot_count = (7 * (max_register_size - 2)) + 3
            elbow_count = 2 * (max_register_size - 1)

            x = resource_rep(re.ResourceX)
            cnot = resource_rep(re.ResourceCNOT)
            l_elbow = resource_rep(re.ResourceTempAND)
            r_elbow = resource_rep(re.ResourceAdjoint, {"base_cmpr_op": l_elbow})
            gate_lst = [
                AllocWires(max_register_size - 1),
                GateCount(cnot, cnot_count),
                GateCount(l_elbow, elbow_count),
                GateCount(r_elbow, elbow_count),
                FreeWires(max_register_size - 1),
            ]

            if ctrl_num_ctrl_values:
                gate_lst.append(GateCount(x, 2 * ctrl_num_ctrl_values))
            return gate_lst  # Obtained resource from Fig 4a https://quantum-journal.org/papers/q-2018-06-18-74/pdf/

        raise re.ResourcesNotDefined


class ResourceQuantumPhaseEstimation(ResourceOperator):
    r"""Resource class for QuantumPhaseEstimation (QPE).

    Args:
        unitary (array or Operator): the phase estimation unitary, specified as a matrix or an
            :class:`~.Operator`
        target_wires (Union[Wires, Sequence[int], or int]): the target wires to apply the unitary.
            If the unitary is specified as an operator, the target wires should already have been
            defined as part of the operator. In this case, target_wires should not be specified.
        estimation_wires (Union[Wires, Sequence[int], or int]): the wires to be used for phase
            estimation

    Resource Parameters:
        * base_class (ResourceOperator): The type of the operation corresponding to the phase estimation unitary.
        * base_params (dict): A dictionary of parameters required to obtain the resources for the phase estimation unitary.
        * num_estimation_wires (int): the number of wires used for measuring out the phase

    Resources:
        The resources are obtained from the standard decomposition of QPE as presented
        in (Section 5.2) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum
        Information <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.

    .. seealso:: :class:`~.QuantumPhaseEstimation`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceQuantumPhaseEstimation.resources(
    ...     base_class=re.ResourceQFT,
    ...     base_params={"num_wires": 3},
    ...     num_estimation_wires=3,
    ... )
    {Hadamard: 3, Adjoint(QFT(3)): 1, C(QFT(3),1,0,0): 7}
    """

    resource_keys = {"base_cmpr_op", "num_estimation_wires"}

    def __init__(self, U: ResourceOperator, precision, wires=None):
        self.queue(remove_op=U)
        base_op = U.resource_rep_from_op()

        self.base_op = base_op
        self.estimation_wires = abs(math.floor(math.log2(precision)))

        if wires is not None:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            self.wires = None
            self.num_wires = self.estimation_wires + U.num_wires

    def queue(self, remove_op=None, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        if remove_op:
            context.remove(remove_op)
        context.append(self)
        return self

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * base_class (Type(ResourceOperator)): The type of the operation corresponding to the phase estimation unitary.
                * base_params (dict): A dictionary of parameters required to obtain the resources for the phase estimation unitary.
                * num_estimation_wires (int): the number of wires used for measuring out the phase
        """

        return {
            "base_cmpr_op": self.base_op,
            "num_estimation_wires": self.estimation_wires,
        }

    @classmethod
    def resource_rep(
        cls,
        base_cmpr_op,
        num_estimation_wires,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            base_class (Type(ResourceOperator)): The type of the operation corresponding to the
                phase estimation unitary.
            base_params (dict): A dictionary of parameters required to obtain the resources for
                the phase estimation unitary.
            num_estimation_wires (int): the number of wires used for measuring out the phase

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "base_cmpr_op": base_cmpr_op,
            "num_estimation_wires": num_estimation_wires,
        }
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(
        cls, base_cmpr_op, num_estimation_wires, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            base_class (Type(ResourceOperator)): The type of the operation corresponding to the
                phase estimation unitary.
            base_params (dict): A dictionary of parameters required to obtain the resources for
                the phase estimation unitary.
            num_estimation_wires (int): the number of wires used for measuring out the phase

        Resources:
            The resources are obtained from the standard decomposition of QPE as presented
            in (section 5.2) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum
            Information <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.
        """
        hadamard = resource_rep(re.ResourceHadamard)
        adj_qft = resource_rep(
            re.ResourceAdjoint,
            {
                "base_cmpr_op": resource_rep(ResourceQFT, {"num_wires": num_estimation_wires}),
            },
        )
        ctrl_op = resource_rep(
            re.ResourceControlled,
            {
                "base_cmpr_op": base_cmpr_op,
                "num_ctrl_wires": 1,
                "num_ctrl_values": 0,
            },
        )

        return [
            GateCount(hadamard, num_estimation_wires),
            GateCount(ctrl_op, (2**num_estimation_wires) - 1),
            GateCount(adj_qft, 1),
        ]

    @staticmethod
    def tracking_name(base_cmpr_op, num_estimation_wires) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_cmpr_op.name
        return f"QPE({base_name}, {num_estimation_wires})"


ResourceQPE = ResourceQuantumPhaseEstimation  # Alias for ease of typing
r"""Resource class for QuantumPhaseEstimation (QPE).

Args:
    unitary (array or Operator): the phase estimation unitary, specified as a matrix or an
        :class:`~.Operator`
    target_wires (Union[Wires, Sequence[int], or int]): the target wires to apply the unitary.
        If the unitary is specified as an operator, the target wires should already have been
        defined as part of the operator. In this case, target_wires should not be specified.
    estimation_wires (Union[Wires, Sequence[int], or int]): the wires to be used for phase
        estimation

Resource Parameters:
    * base_class (ResourceOperator): The type of the operation corresponding to the phase estimation unitary.
    * base_params (dict): A dictionary of parameters required to obtain the resources for the phase estimation unitary.
    * num_estimation_wires (int): the number of wires used for measuring out the phase

Resources:
    The resources are obtained from the standard decomposition of QPE as presented
    in (Section 5.2) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum
    Information <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.

.. seealso:: :class:`~.QuantumPhaseEstimation`

**Example**

The resources for this operation are computed using:

>>> re.ResourceQuantumPhaseEstimation.resources(
...     base_class=re.ResourceQFT,
...     base_params={"num_wires": 3},
...     num_estimation_wires=3,
... )
{Hadamard: 3, Adjoint(QFT(3)): 1, C(QFT(3),1,0,0): 7}
"""


class ResourceBasisRotation(ResourceOperator):
    r"""Resource class for the BasisRotation gate.

    Args:
        dim_N (int): The dimensions of the input :code:`unitary_matrix`. This is computed as the 
            number of columns of the matrix.
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from the construction scheme given in `Optica, 3, 1460 (2016)
        <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_. Specifically,
        the resources are given as :math:`dim_N * (dim_N - 1) / 2` instances of the
        :class:`~.ResourceSingleExcitation` gate, and :math:`dim_N * (1 + (dim_N - 1) / 2)` instances
        of the :class:`~.ResourcePhaseShift` gate.

    .. seealso:: :class:`~.BasisRotation`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceBasisRotation.resources(dim_N=3)
    {PhaseShift: 6.0, SingleExcitation: 3.0}
    """

    resource_keys = {"dim_N"}

    def __init__(self, dim_N, wires=None):
        self.num_wires = dim_N
        super().__init__(wires=wires)

    @classmethod
    def default_resource_decomp(cls, dim_N, **kwargs) -> Dict[CompressedResourceOp, int]:
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
        """
        phase_shift = resource_rep(re.ResourcePhaseShift)
        single_excitation = resource_rep(re.ResourceSingleExcitation)

        se_count = dim_N * (dim_N - 1) / 2
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
        where :math:`n = \ceil{log_{2}(N)}` and :math:`N` is the number of batches of unitaries
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
    def textbook_resources(cmpr_ops, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

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
