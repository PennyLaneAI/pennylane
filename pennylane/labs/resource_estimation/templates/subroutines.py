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
from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceOperator

# pylint: disable=arguments-differ, protected-access


class ResourceQFT(qml.QFT, ResourceOperator):
    r"""Resource class for the QFT template.

    Args:
        num_wires (int): the number of qubits the operation acts upon

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
            in (chapter 5) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum Information
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

        Resource parameters:
            num_wires (int): the number of qubits the operation acts upon

        Returns:
            dict: dictionary containing the resource parameters
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


class ResourceControlledSequence(qml.ControlledSequence, re.ResourceOperator):
    """Resource class for the ControlledSequence template."""

    @staticmethod
    def _resource_decomp(
        base_class, base_params, num_ctrl_wires, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        return {
            re.ResourceControlled.resource_rep(base_class, base_params, 1, 0, 0): 2**num_ctrl_wires
            - 1
        }

    @property
    def resource_params(self) -> dict:
        return {
            "base_class": type(self.base),
            "base_params": self.base.resource_params,
            "num_ctrl_wires": len(self.control_wires),
        }

    @classmethod
    def resource_rep(cls, base_class, base_params, num_ctrl_wires) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(
            cls,
            {
                "base_class": base_class,
                "base_params": base_params,
                "num_ctrl_wires": num_ctrl_wires,
            },
        )

    @staticmethod
    def tracking_name(base_class, base_params, num_ctrl_wires) -> str:
        base_name = base_class.tracking_name(**base_params)
        return f"ControlledSequence({base_name}, {num_ctrl_wires})"


class ResourcePhaseAdder(qml.PhaseAdder, re.ResourceOperator):
    """Resource class for the PhaseAdder template."""

    @staticmethod
    def _resource_decomp(mod, num_x_wires, **kwargs) -> Dict[re.CompressedResourceOp, int]:
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
        ctrl_phase_shift = re.ResourceControlled.resource_rep(
            re.ResourcePhaseShift,
            {},
            1,
            0,
            0,
        )

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
        return {
            "mod": self.hyperparameters["mod"],
            "num_x_wires": len(self.hyperparameters["x_wires"]),
        }

    @classmethod
    def resource_rep(cls, mod, num_x_wires) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {"mod": mod, "num_x_wires": num_x_wires})


class ResourceMultiplier(qml.Multiplier, re.ResourceOperator):
    """Resource class for the Multiplier template."""

    @staticmethod
    def _resource_decomp(
        mod, num_work_wires, num_x_wires, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        if mod == 2**num_x_wires:
            num_aux_wires = num_x_wires
            num_aux_swap = num_x_wires
        else:
            num_aux_wires = num_work_wires - 1
            num_aux_swap = num_aux_wires - 1

        qft = ResourceQFT.resource_rep(num_aux_wires)
        qft_dag = re.ResourceAdjoint.resource_rep(
            ResourceQFT,
            {"num_wires": num_aux_wires},
        )

        sequence = ResourceControlledSequence.resource_rep(
            ResourcePhaseAdder,
            {},
            num_x_wires,
        )

        sequence_dag = re.ResourceAdjoint.resource_rep(
            ResourceControlledSequence,
            {
                "base_class": ResourcePhaseAdder,
                "base_params": {},
                "num_ctrl_wires": num_x_wires,
            },
        )

        cnot = re.ResourceCNOT.resource_rep()

        gate_types = {}
        gate_types[qft] = 2
        gate_types[qft_dag] = 2
        gate_types[sequence] = 1
        gate_types[sequence_dag] = 1
        gate_types[cnot] = min(num_x_wires, num_aux_swap)

        return gate_types

    @property
    def resource_params(self) -> dict:
        return {
            "mod": self.hyperparameters["mod"],
            "num_work_wires": len(self.hyperparameters["work_wires"]),
            "num_x_wires": len(self.hyperparameters["x_wires"]),
        }

    @classmethod
    def resource_rep(cls, mod, num_work_wires, num_x_wires) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(
            cls, {"mod": mod, "num_work_wires": num_work_wires, "num_x_wires": num_x_wires}
        )


class ResourceModExp(qml.ModExp, re.ResourceOperator):
    """Resource class for the ModExp template."""

    @staticmethod
    def _resource_decomp(
        mod, num_output_wires, num_work_wires, num_x_wires, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        mult_resources = ResourceMultiplier._resource_decomp(mod, num_work_wires, num_output_wires)
        gate_types = {}

        for comp_rep, _ in mult_resources.items():
            new_rep = re.ResourceControlled.resource_rep(comp_rep.op_type, comp_rep.params, 1, 0, 0)

            # cancel out QFTs from consecutive Multipliers
            if comp_rep._name in ("QFT", "Adjoint(QFT)"):
                gate_types[new_rep] = 1
            else:
                gate_types[new_rep] = mult_resources[comp_rep] * ((2**num_x_wires) - 1)

        return gate_types

    @property
    def resource_params(self) -> dict:
        return {
            "mod": self.hyperparameters["mod"],
            "num_output_wires": len(self.hyperparameters["output_wires"]),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
            "num_x_wires": len(self.hyperparameters["x_wires"]),
        }

    @classmethod
    def resource_rep(
        cls, mod, num_output_wires, num_work_wires, num_x_wires
    ) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(
            cls,
            {
                "mod": mod,
                "num_output_wires": num_output_wires,
                "num_work_wires": num_work_wires,
                "num_x_wires": num_x_wires,
            },
        )


class ResourceQuantumPhaseEstimation(qml.QuantumPhaseEstimation, ResourceOperator):
    r"""Resource class for QuantumPhaseEstimation (QPE).

    Args:
        base_class (ResourceOperator): The type of the operation corresponding to the
            phase estimation unitary.
        base_params (dict): A dictionary of parameters required to obtain the resources for
            the phase estimation unitary.
        num_estimation_wires (int): the number of wires used for measuring out the phase

    Resources:
        The resources are obtained from the standard decomposition of QPE as presented
        in (section 5.2) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum
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

        Resource parameters:
            base_class (Type(ResourceOperator)): The type of the operation corresponding to the
                phase estimation unitary.
            base_params (dict): A dictionary of parameters required to obtain the resources for
                the phase estimation unitary.
            num_estimation_wires (int): the number of wires used for measuring out the phase

        Returns:
            dict: dictionary containing the resource parameters
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


class ResourceBasisRotation(qml.BasisRotation, ResourceOperator):
    r"""Resource class for the BasisRotation gate.

    Args:
        dim_N (int): The dimensions of the input :code:`unitary_matrix`. This is computed
            as the number of columns of the matrix.

    Resources:
        The resources are obtained from the construction scheme given in `Optica, 3, 1460 (2016)
        <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_. Specifically,
        the resources are given as :math:`dim_N * (dim_N - 1) / 2` instances of the
        :class:`~.ResourceSingleExcitation` gate, and :math:`dim_N * (1 + (dim_N - 1) / 2)` instances
        of the :class:`~.ResourcePhaseShift` gate.

    .. seealso:: :class:`~.BasisRotation`

    """

    @staticmethod
    def _resource_decomp(dim_N, **kwargs) -> Dict[CompressedResourceOp, int]:
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
        gate_types = {}
        phase_shift = re.ResourcePhaseShift.resource_rep()
        single_excitation = re.ResourceSingleExcitation.resource_rep()

        se_count = dim_N * (dim_N - 1) / 2
        ps_count = dim_N + se_count

        gate_types[phase_shift] = ps_count
        gate_types[single_excitation] = se_count
        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Resource parameters:
            dim_N (int): The dimensions of the input :code:`unitary_matrix`. This is computed
                as the number of columns of the matrix.

        Returns:
            dict: dictionary containing the resource parameters
        """
        unitary_matrix = self.parameters[0]
        return {"dim_N": qml.math.shape(unitary_matrix)[0]}

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


class ResourceSelect(qml.Select, ResourceOperator):
    r"""Resource class for the Select gate.

    Args:
        cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed representation,
            to be applied according to the selected qubits.

    Resources:
        The resources correspond directly to the definition of the operation. Specifically,
        for each operator in :code:`cmpr_ops`, the cost is given as a controlled version of the operator
        controlled on the associated bitstring.

    .. seealso:: :class:`~.Select`

    """

    @staticmethod
    def _resource_decomp(cmpr_ops, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed
                representation, to be applied according to the selected qubits.

        Resources:
            The resources correspond directly to the definition of the operation. Specifically,
            for each operator in :code:`cmpr_ops`, the cost is given as a controlled version of the operator
            controlled on the associated bitstring.
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
                cmp_rep.op_type, cmp_rep.params, num_ctrl_wires, 0, 0
            )
            gate_types[ctrl_op] += 1

        return gate_types

    @staticmethod
    def resources_for_ui(cmpr_ops, **kwargs):
        r"""The resources for a select implementation taking advantage of the unary iterator trick.
        The resources are based on the analysis in https://arxiv.org/pdf/1805.03662 section III.A, 'Unary Iteration and Indexed
        Operations'. See Figures 4, 6, and 7.

        Note: This implementation assumes we have access to :math:`S + 1` additional work qubits,
        where :math:`S = \ceil{log_{2}(N)}` and :math:`N` is the number of batches of unitaries
        to select.
        """
        gate_types = defaultdict(int)
        x = re.ResourceX.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()
        toffoli = re.ResourceToffoli.resource_rep()

        num_ops = len(cmpr_ops)

        for cmp_rep in cmpr_ops:
            ctrl_op = re.ResourceControlled.resource_rep(cmp_rep.op_type, cmp_rep.params, 1, 0, 0)
            gate_types[ctrl_op] += 1

        gate_types[x] = 2 * (num_ops - 1)  # conjugate 0 controlled toffolis
        gate_types[cnot] = num_ops - 1
        gate_types[toffoli] = 2 * (num_ops - 1)

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Resource parameters:
            cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed
                representation, to be applied according to the selected qubits.

        Returns:
            dict: dictionary containing the resource parameters
        """
        ops = self.hyperparameters["ops"]
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)
        return {"cmpr_ops": cmpr_ops}

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


class ResourcePrepSelPrep(qml.PrepSelPrep, ResourceOperator):
    r"""Resource class for PrepSelPrep gate.

    Args:
        cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed representation,
            which correspond to the unitaries in the LCU to be blockencoded.

    Resources:
        The resources correspond directly to the definition of the operation. Specifically,
        the resources are given as one instance of :class:`~.ResourceSelect`, which is conjugated by
        a pair of :class:`~.ResourceStatePrep` operations.

    .. seealso:: :class:`~.PrepSelPrep`

    """

    @staticmethod
    def _resource_decomp(cmpr_ops, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed
                representation, which correspond to the unitaries in the LCU to be blockencoded.

        Resources:
            The resources correspond directly to the definition of the operation. Specifically,
            the resources are given as one instance of :class:`~.ResourceSelect`, which is conjugated by
            a pair of :class:`~.ResourceStatePrep` operations.
        """
        gate_types = {}

        num_ops = len(cmpr_ops)
        num_wires = int(math.ceil(math.log2(num_ops)))

        prep = re.ResourceStatePrep.resource_rep(num_wires)
        sel = ResourceSelect.resource_rep(cmpr_ops)
        prep_dag = re.ResourceAdjoint.resource_rep(re.ResourceStatePrep, {"num_wires": num_wires})

        gate_types[prep] = 1
        gate_types[sel] = 1
        gate_types[prep_dag] = 1
        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Resource parameters:
            cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed
                representation, which correspond to the unitaries in the LCU to be blockencoded.

        Returns:
            dict: dictionary containing the resource parameters
        """
        ops = self.hyperparameters["ops"]
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)
        return {"cmpr_ops": cmpr_ops}

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

    @classmethod
    def pow_resource_decomp(cls, z, cmpr_ops) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to
            cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed
                representation, to be applied according to the selected qubits.

        Resources:
            The resources are derived from the following identity. If an operation :math:`\hat{A}`
            can be expressed as :math:`\hat{A} \ = \ \hat{U} \cdot \hat{B} \cdot \hat{U}^{\dagger}`
            then the operation squared can be expressed as:

            .. math::

                \begin{align}
                    \hat{A}^{2} \ &= \ \hat{U} \cdot \hat{B} \cdot \hat{U}^{\dagger} \cdot \hat{U} \cdot \hat{B} \cdot \hat{U}^{\dagger} \\
                    \hat{A}^{2} \ &= \ \hat{U} \cdot \hat{B} \cdot \hat{B} \cdot \hat{U}^{\dagger} \\
                    \hat{A}^{2} \ &= \ \hat{U} \cdot \hat{B}^{2} \cdot \hat{U}^{\dagger},
                \end{align}

            this holds for any integer power :math:`z`. In general, the resources are given by :math:`z`
            instances of :class:`~.ResourceSelect` conjugated by a pair of :class:`~.ResourceStatePrep`
            operations.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        gate_types = {}

        num_ops = len(cmpr_ops)
        num_wires = int(math.ceil(math.log2(num_ops)))

        prep = re.ResourceStatePrep.resource_rep(num_wires)
        pow_sel = re.ResourcePow.resource_rep(ResourceSelect, {"cmpr_ops": cmpr_ops}, z)
        prep_dag = re.ResourceAdjoint.resource_rep(re.ResourceStatePrep, {"num_wires": num_wires})

        gate_types[prep] = 1
        gate_types[pow_sel] = 1
        gate_types[prep_dag] = 1
        return gate_types


class ResourceReflection(qml.Reflection, ResourceOperator):
    r"""Resource class for the Reflection gate.

    Args:
        base_class (Type(ResourceOperator)): The type of the operation used to prepare the
            state we will be reflecting over.
        base_params (dict): A dictionary of parameters required to obtain the resources for
            the state preparation operator.
        num_ref_wires (int): The number of qubits for the subsystem on which the reflection is
            applied.

    Resources:
        The resources correspond directly to the definition of the operation. The operator is
        built as follows:

        .. math::

            \text{R}(U, \alpha) = -I + (1 - e^{i\alpha}) |\Psi\rangle \langle \Psi| = U(-I + (1 - e^{i\alpha}) |0\rangle \langle 0|)U^{\dagger}.

        The central block is obtained through a controlled :class:`~.ResourcePhaseShift` operator and
        a :class:`~.ResourceGlobalPhase` which are conjugated with a pair of :class:`~.ResourceX` gates.
        Finally, the block is conjugated with the state preparation unitary :math:`U`.

    .. seealso:: :class:`~.Reflection`

    """

    @staticmethod
    def _resource_decomp(
        base_class, base_params, num_ref_wires, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            base_class (Type(ResourceOperator)): The type of the operation used to prepare the
                state we will be reflecting over.
            base_params (dict): A dictionary of parameters required to obtain the resources for
                the state preparation operator.
            num_ref_wires (int): The number of qubits for the subsystem on which the reflection is
                applied.

        Resources:
            The resources correspond directly to the definition of the operation. The operator is
            built as follows:

            .. math::

                \text{R}(U, \alpha) = -I + (1 - e^{i\alpha}) |\Psi\rangle \langle \Psi| = U(-I + (1 - e^{i\alpha}) |0\rangle \langle 0|)U^{\dagger}.

            The central block is obtained through a controlled :class:`~.ResourcePhaseShift` operator and
            a :class:`~.ResourceGlobalPhase` which are conjugated with a pair of :class:`~.ResourceX` gates.
            Finally, the block is conjugated with the state preparation unitary :math:`U`.
        """
        gate_types = {}
        base = base_class.resource_rep(**base_params)

        x = re.ResourceX.resource_rep()
        gp = re.ResourceGlobalPhase.resource_rep()
        adj_base = re.ResourceAdjoint.resource_rep(base_class, base_params)
        ps = (
            re.ResourceControlled.resource_rep(
                re.ResourcePhaseShift, {}, num_ref_wires - 1, num_ref_wires - 1, 0
            )
            if num_ref_wires > 1
            else re.ResourcePhaseShift.resource_rep()
        )

        gate_types[x] = 2
        gate_types[gp] = 1
        gate_types[base] = 1
        gate_types[adj_base] = 1
        gate_types[ps] = 1

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Resource parameters:
            base_class (Type(ResourceOperator)): The type of the operation used to prepare the
                state we will be reflecting over.
            base_params (dict): A dictionary of parameters required to obtain the resources for
                the state preparation operator.
            num_ref_wires (int): The number of qubits for the subsystem on which the reflection is
                applied.

        Returns:
            dict: dictionary containing the resource parameters
        """
        base_cmpr_rep = self.hyperparameters["base"].resource_rep_from_op()
        num_ref_wires = len(self.hyperparameters["reflection_wires"])

        return {
            "base_class": base_cmpr_rep.op_type,
            "base_params": base_cmpr_rep.params,
            "num_ref_wires": num_ref_wires,
        }

    @classmethod
    def resource_rep(cls, base_class, base_params, num_ref_wires) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            base_class (Type(ResourceOperator)): The type of the operation used to prepare the
                state we will be reflecting over.
            base_params (dict): A dictionary of parameters required to obtain the resources for
                the state preparation operator.
            num_ref_wires (int): The number of qubits for the subsystem on which the reflection is
                applied.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "base_class": base_class,
            "base_params": base_params,
            "num_ref_wires": num_ref_wires,
        }
        return CompressedResourceOp(cls, params)


class ResourceQubitization(qml.Qubitization, ResourceOperator):
    r"""Resource class for the Qubitization gate.

    Args:
        cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed representation, corresponding to the unitaries of the LCU representation of the hamiltonian being qubitized.
        num_ctrl_wires (int): The number of qubits used to prepare the coefficients vector of the LCU.

    Resources:
        The resources are obtained from the definition of the operation as described in (section III. C)
        `Simulating key properties of lithium-ion batteries with a fault-tolerant quantum computer
        <https://arxiv.org/abs/2204.11890>`_:

        .. math::

            Q =  \text{Prep}_{\mathcal{H}}^{\dagger} \text{Sel}_{\mathcal{H}} \text{Prep}_{\mathcal{H}}(2|0\rangle\langle 0| - I).

        Specifically, the resources are given by one :class:`~.ResourcePrepSelPrep` gate and one
        :class:`~.ResourceReflection` gate.

    .. seealso:: :class:`~.Qubitization`

    """

    @staticmethod
    def _resource_decomp(cmpr_ops, num_ctrl_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed representation,
                corresponding to the unitaries of the LCU representation of the hamiltonian being qubitized.
            num_ctrl_wires (int): The number of qubits used to prepare the coefficients vector of the LCU.

        Resources:
            The resources are obtained from the definition of the operation as described in (section III. C)
            `Simulating key properties of lithium-ion batteries with a fault-tolerant quantum computer
            <https://arxiv.org/abs/2204.11890>`_:

            .. math::

                Q =  \text{Prep}_{\mathcal{H}}^{\dagger} \text{Sel}_{\mathcal{H}} \text{Prep}_{\mathcal{H}}(2|0\rangle\langle 0| - I).

            Specifically, the resources are given by one :class:`~.ResourcePrepSelPrep` gate and one
            :class:`~.ResourceReflection` gate.
        """
        gate_types = {}
        ref = ResourceReflection.resource_rep(re.ResourceIdentity, {}, num_ctrl_wires)
        psp = ResourcePrepSelPrep.resource_rep(cmpr_ops)

        gate_types[ref] = 1
        gate_types[psp] = 1
        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Resource parameters:
            cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed representation, corresponding to the unitaries of the LCU representation of the hamiltonian being qubitized.
            num_ctrl_wires (int): The number of qubits used to prepare the coefficients vector of the LCU.

        Returns:
            dict: dictionary containing the resource parameters
        """
        lcu = self.hyperparameters["hamiltonian"]
        _, ops = lcu.terms()

        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)
        num_ctrl_wires = len(self.hyperparameters["control"])
        return {"cmpr_ops": cmpr_ops, "num_ctrl_wires": num_ctrl_wires}

    @classmethod
    def resource_rep(cls, cmpr_ops, num_ctrl_wires) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            cmpr_ops (list[CompressedResourceOp]): The list of operators, in the compressed representation, corresponding to the unitaries of the LCU representation of the hamiltonian being qubitized.
            num_ctrl_wires (int): The number of qubits used to prepare the coefficients vector of the LCU.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"cmpr_ops": cmpr_ops, "num_ctrl_wires": num_ctrl_wires}
        return CompressedResourceOp(cls, params)


class ResourceQROM(qml.QROM, ResourceOperator):
    """Resource class for the QROM template."""

    @staticmethod
    def _resource_decomp(
        num_bitstrings,
        num_bit_flips,
        num_control_wires,
        num_work_wires,
        size_bitstring,
        clean,
        **kwargs,
    ) -> Dict[CompressedResourceOp, int]:
        r"""The resources for QROM are taken from the following two papers:
        (https://arxiv.org/pdf/1812.00954, figure 1.c) and
        (https://arxiv.org/pdf/1902.02134, figure 4).

        We use the one-auxillary qubit version of select, instead of the built-in select
        resources.
        """
        gate_types = {}
        x = re.ResourceX.resource_rep()

        if num_control_wires == 0:
            gate_types[x] = num_bit_flips
            return gate_types

        cnot = re.ResourceCNOT.resource_rep()
        hadamard = re.ResourceHadamard.resource_rep()

        num_parallel_computations = (num_work_wires + size_bitstring) // size_bitstring
        num_parallel_computations = min(num_parallel_computations, num_bitstrings)

        num_swap_wires = math.floor(math.log2(num_parallel_computations))
        num_select_wires = math.ceil(math.log2(math.ceil(num_bitstrings / (2**num_swap_wires))))
        assert num_swap_wires + num_select_wires <= num_control_wires

        swap_clean_prefactor = 1
        select_clean_prefactor = 1

        if clean:
            gate_types[hadamard] = 2 * size_bitstring
            swap_clean_prefactor = 4
            select_clean_prefactor = 2

        # SELECT cost:
        gate_types[cnot] = num_bit_flips  # each unitary in the select is just a CNOT

        multi_x = re.ResourceMultiControlledX.resource_rep(num_select_wires, 0, 0)
        num_total_ctrl_possibilities = 2**num_select_wires
        gate_types[multi_x] = select_clean_prefactor * (
            2 * num_total_ctrl_possibilities  # two applications targetting the aux qubit
        )
        num_zero_controls = (2 * num_total_ctrl_possibilities * num_select_wires) // 2
        gate_types[x] = select_clean_prefactor * (
            num_zero_controls * 2  # conjugate 0 controls on the multi-qubit x gates from above
        )
        # SWAP cost:
        ctrl_swap = re.ResourceCSWAP.resource_rep()
        gate_types[ctrl_swap] = swap_clean_prefactor * ((2**num_swap_wires) - 1) * size_bitstring

        return gate_types

    @property
    def resource_params(self) -> Dict:
        bitstrings = self.hyperparameters["bitstrings"]
        num_bitstrings = len(bitstrings)

        num_bit_flips = 0
        for bit_string in bitstrings:
            num_bit_flips += bit_string.count("1")

        num_work_wires = len(self.hyperparameters["work_wires"])
        size_bitstring = len(self.hyperparameters["target_wires"])
        num_control_wires = len(self.hyperparameters["control_wires"])
        clean = self.hyperparameters["clean"]

        return {
            "num_bitstrings": num_bitstrings,
            "num_bit_flips": num_bit_flips,
            "num_control_wires": num_control_wires,
            "num_work_wires": num_work_wires,
            "size_bitstring": size_bitstring,
            "clean": clean,
        }

    @classmethod
    def resource_rep(
        cls, num_bitstrings, num_bit_flips, num_control_wires, num_work_wires, size_bitstring, clean
    ) -> CompressedResourceOp:
        params = {
            "num_bitstrings": num_bitstrings,
            "num_bit_flips": num_bit_flips,
            "num_control_wires": num_control_wires,
            "num_work_wires": num_work_wires,
            "size_bitstring": size_bitstring,
            "clean": clean,
        }
        return CompressedResourceOp(cls, params)


class ResourceAmplitudeAmplification(qml.AmplitudeAmplification, ResourceOperator):
    """Resource class for the AmplitudeAmplification template."""

    @staticmethod
    def _resource_decomp(
        U_op,
        U_params,
        O_op,
        O_params,
        iters,
        num_work_wires,
        num_ref_wires,
        fixed_point,
        **kwargs,
    ) -> Dict[CompressedResourceOp, int]:
        r"""The resources for Amplitude Amplifcation are according to the decomposition found
        in qml.AmplitudeAmplification.
        """
        gate_types = {}
        ctrl = re.ResourceControlled.resource_rep(
            base_class=O_op,
            base_params=O_params,
            num_ctrl_wires=num_work_wires,
            num_ctrl_values=0,
            num_work_wires=0,
        )
        phase_shift = re.ResourcePhaseShift.resource_rep()
        hadamard = re.ResourceHadamard.resource_rep()
        reflection = re.ResourceReflection.resource_rep(
            base_class=U_op, base_params=U_params, num_ref_wires=num_ref_wires
        )

        if not fixed_point:
            oracles = re.CompressedResourceOp(O_op, params=O_params)
            gate_types[oracles] = iters
            gate_types[reflection] = iters

            return gate_types

        iters = iters // 2

        gate_types[ctrl] = iters * 2
        gate_types[phase_shift] = iters
        gate_types[hadamard] = iters * 4
        gate_types[reflection] = iters

        return gate_types

    @property
    def resource_params(self) -> Dict:
        U_op = self.hyperparameters["U"]
        O_op = self.hyperparameters["O"]
        U_params = U_op.resource_params if hasattr(U_op, "resource_params") else {}
        O_params = O_op.resource_params if hasattr(O_op, "resource_params") else {}
        iters = self.hyperparameters["iters"]
        fixed_point = self.hyperparameters["fixed_point"]
        num_ref_wires = len(self.hyperparameters["reflection_wires"])

        return {
            "U_op": type(U_op),
            "U_params": U_params,
            "O_op": type(O_op),
            "O_params": O_params,
            "iters": iters,
            "num_work_wires": 1,
            "num_ref_wires": num_ref_wires,
            "fixed_point": fixed_point,
        }

    @classmethod
    def resource_rep(
        cls, U_op, U_params, O_op, O_params, iters, num_work_wires, num_ref_wires, fixed_point
    ) -> CompressedResourceOp:
        params = {
            "U_op": U_op,
            "U_params": U_params,
            "O_op": O_op,
            "O_params": O_params,
            "iters": iters,
            "num_work_wires": num_work_wires,
            "num_ref_wires": num_ref_wires,
            "fixed_point": fixed_point,
        }
        return CompressedResourceOp(cls, params)
