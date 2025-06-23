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


class ResourceQFT(qml.QFT, ResourceOperator):
    r"""Resource class for QFT.

    Args:
        wires (int or Iterable[Number, str]]): the wire(s) the operation acts on

    Resource Parameters:
        * num_wires (int): the number of qubits the operation acts upon

    Resources:
        The resources are obtained from the standard decomposition of QFT as presented
        in (chapter 5) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum Information
        <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.

    .. seealso:: :class:`~.QFT`

    """

    @staticmethod
    def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            num_wires (int): the number of qubits the operation acts upon

        Resources:
            The resources are obtained from the standard decomposition of QFT as presented
            in (Chapter 5) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum Information
            <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.

        """
        gate_types = {}

        hadamard = re.ResourceHadamard.resource_rep()
        swap = re.ResourceSWAP.resource_rep()
        ctrl_phase_shift = re.ResourceControlledPhaseShift.resource_rep()

        gate_types[hadamard] = num_wires
        gate_types[swap] = num_wires // 2
        gate_types[ctrl_phase_shift] = num_wires * (num_wires - 1) // 2

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int): the number of qubits the operation acts upon
        """
        return {"num_wires": len(self.wires)}

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

    @staticmethod
    def tracking_name(num_wires) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return f"QFT({num_wires})"


class ResourcePhaseAdder(qml.PhaseAdder, re.ResourceOperator):
    r"""Resource class for the PhaseAdder template.

    Args:
        k (int): the number that needs to be added
        x_wires (Sequence[int]): the wires the operation acts on. The number of wires must be enough
            for a binary representation of the value being targeted, :math:`x`. In some cases an additional
            wire is needed, see usage details below. The number of wires also limits the maximum
            value for `mod`.
        mod (int): the modulo for performing the addition. If not provided, it will be set to its maximum value, :math:`2^{\text{len(x_wires)}}`.
        work_wire (Sequence[int] or int): the auxiliary wire to use for the addition. Optional
            when `mod` is :math:`2^{len(x\_wires)}`. Defaults to empty tuple.

    Resource Parameters:
        * mod (int): the module for performing the addition
        * num_x_wires (int): the number of wires the operation acts on

    Resources:
        The resources are obtained from the standard decomposition of :class:`~.PhaseAdder`.

    .. seealso:: :class:`~.PhaseAdder`

    """

    @staticmethod
    def _resource_decomp(mod, num_x_wires, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            mod (int): the module for performing the addition
            num_x_wires (int): the number of wires the operation acts on

        Resources:
            The resources are obtained from the standard decomposition of :class:`~.PhaseAdder`.

        """
        if mod == 2**num_x_wires:
            return {re.ResourcePhaseShift.resource_rep(): num_x_wires}

        qft = ResourceQFT.resource_rep(num_x_wires)
        qft_dag = re.ResourceAdjoint.resource_rep(
            ResourceQFT,
            {"num_wires": num_x_wires},
        )

        phase_shift = re.ResourcePhaseShift.resource_rep()
        phase_shift_dag = re.ResourceAdjoint.resource_rep(
            re.ResourcePhaseShift,
            {},
        )
        ctrl_phase_shift = re.ResourceControlledPhaseShift.resource_rep()

        cnot = re.ResourceCNOT.resource_rep()
        multix = re.ResourceMultiControlledX.resource_rep(1, 0, 1)

        gate_types = {}
        gate_types[qft] = 2
        gate_types[qft_dag] = 2
        gate_types[phase_shift] = 2 * num_x_wires
        gate_types[phase_shift_dag] = 2 * num_x_wires
        gate_types[ctrl_phase_shift] = num_x_wires
        gate_types[cnot] = 1
        gate_types[multix] = 1

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Resource Parameters:
            mod (int): the module for performing the addition
            num_x_wires (int): the number of wires the operation acts on

        Returns:
            dict: A dictionary containing the resource parameters:
                * mod (int): the module for performing the addition
                * num_x_wires (int): the number of wires the operation acts on
        """
        return {
            "mod": self.hyperparameters["mod"],
            "num_x_wires": len(self.hyperparameters["x_wires"]),
        }

    @classmethod
    def resource_rep(cls, mod, num_x_wires) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            mod (int): the module for performing the addition
            num_x_wires (int): the number of wires the operation acts on

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """

        return re.CompressedResourceOp(cls, {"mod": mod, "num_x_wires": num_x_wires})

class ResourceQuantumPhaseEstimation(qml.QuantumPhaseEstimation, ResourceOperator):
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

    """

    @staticmethod
    def _resource_decomp(
        base_class, base_params, num_estimation_wires, **kwargs
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
        gate_types = {}

        hadamard = re.ResourceHadamard.resource_rep()
        adj_qft = re.ResourceAdjoint.resource_rep(ResourceQFT, {"num_wires": num_estimation_wires})
        ctrl_op = re.ResourceControlled.resource_rep(base_class, base_params, 1, 0, 0)

        gate_types[hadamard] = num_estimation_wires
        gate_types[adj_qft] = 1
        gate_types[ctrl_op] = (2**num_estimation_wires) - 1

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * base_class (Type(ResourceOperator)): The type of the operation corresponding to the phase estimation unitary.
                * base_params (dict): A dictionary of parameters required to obtain the resources for the phase estimation unitary.
                * num_estimation_wires (int): the number of wires used for measuring out the phase
        """
        op = self.hyperparameters["unitary"]
        num_estimation_wires = len(self.hyperparameters["estimation_wires"])

        if not isinstance(op, re.ResourceOperator):
            raise TypeError(
                f"Can't obtain QPE resources when the base unitary {op} isn't an instance"
                " of ResourceOperator"
            )

        return {
            "base_class": type(op),
            "base_params": op.resource_params,
            "num_estimation_wires": num_estimation_wires,
        }

    @classmethod
    def resource_rep(
        cls,
        base_class,
        base_params,
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
            "base_class": base_class,
            "base_params": base_params,
            "num_estimation_wires": num_estimation_wires,
        }
        return CompressedResourceOp(cls, params)

    @staticmethod
    def tracking_name(base_class, base_params, num_estimation_wires) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_class.tracking_name(**base_params)
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

"""

class ResourceOutOfPlaceSquare(ResourceOperator):
    # reference: Appendix G, Lemma 7 https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332
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
    # reference: Appendix G, Lemma 10 https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332

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
    def textbook_resources(cmpr_ops, **kwargs) -> list[GateCount]:
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
        elif t_cost_func(w2) >= t_cost_func(w1) and w1 >= 1:
            return w1
        else:
            return 1

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
            work qubits, where :math:`n = \ceil{log_{2}(N)}` and :math:`N` is
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
                * num_bit_flips (int, optional): The total number of :math:`1`'s in the dataset. Defaults to
                :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
                * clean (bool, optional): Determine if allocated qubits should be reset after the computation 
                (at the cost of higher gate counts). Defaults to :code`True`.
                * select_swap_depth (Union[int, None], optional): A natural number that determines if data 
                will be loaded in parallel by adding more rows following Figure 1.C of `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_.
                Defaults to :code:`None`, which internally determines the optimal depth.
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
