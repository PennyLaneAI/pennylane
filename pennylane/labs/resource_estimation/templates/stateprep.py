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
r"""Resource operators for PennyLane state preparation templates."""
import itertools
import math
from collections import defaultdict
from typing import Dict

import pennylane as qml
from pennylane import numpy as np
from pennylane.labs import resource_estimation as re
from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceOperator
from pennylane.operation import Operation

# pylint: disable=arguments-differ, protected-access


class ResourceStatePrep(qml.StatePrep, ResourceOperator):
    """Resource class for StatePrep.

    Args:
        state (array[complex] or csr_matrix): the state vector to prepare
        wires (Sequence[int] or int): the wire(s) the operation acts on
        pad_with (float or complex): if not ``None``, ``state`` is padded with this constant to be of size :math:`2^n`, where
            :math:`n` is the number of wires.
        normalize (bool): whether to normalize the state vector. To represent a valid quantum state vector, the L2-norm
            of ``state`` must be one. The argument ``normalize`` can be set to ``True`` to normalize the state automatically.
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
        validate_norm (bool): whether to validate the norm of the input state

    Resource Parameters:
        * num_wires (int): the number of wires that the operation acts on

    Resources:
        Uses the resources as defined in the :class:`~.ResourceMottonenStatePreperation` template.

    .. seealso:: :class:`~.StatePrep`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceStatePrep.resources(num_wires=3)
    {MottonenStatePrep(3): 1}
    """

    def __init__(self, state, wires, pad_with=None, normalize=False, id=None, validate_norm=True):
        # Overriding the default init method to allow for CompactState as an input.

        if isinstance(state, re.CompactState):
            self.compact_state = state
            Operation.__init__(self, state, wires=wires)
            return

        self.compact_state = None
        super().__init__(state, wires, pad_with, normalize, id, validate_norm)

    @staticmethod
    def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            num_wires (int): the number of wires that the operation acts on

        Resources:
            Uses the resources as defined in the :class:`~.ResourceMottonenStatePreperation` template.
        """
        return {re.ResourceMottonenStatePreparation.resource_rep(num_wires): 1}

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int): the number of wires that the operation acts on
        """
        if self.compact_state:
            return {"num_wires": self.compact_state.num_qubits}

        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_wires (int): the number of wires that the operation acts on

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"num_wires": num_wires}
        return CompressedResourceOp(cls, params)

    @classmethod
    def tracking_name(cls, num_wires) -> str:
        return f"StatePrep({num_wires})"


class ResourceMottonenStatePreparation(qml.MottonenStatePreparation, ResourceOperator):
    """Resource class for the MottonenStatePreparation template.

    Args:
        state_vector (tensor_like): Input array of shape ``(2^n,)``, where ``n`` is the number of wires
            the state preparation acts on. The input array must be normalized.
        wires (Iterable): wires that the template acts on

    Resource Parameters:
        * num_wires(int): the number of wires that the operation acts on

    Resources:
        Using the resources as described in `Mottonen et al. (2008) <https://arxiv.org/pdf/quant-ph/0407010>`_.
        The resources are defined as :math:`2^{N+2} - 5` :class:`~.ResourceRZ` gates and
        :math:`2^{N+2} - 4N - 4` :class:`~.ResourceCNOT` gates.

    .. seealso:: :class:`~.MottonenStatePreperation`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceMottonenStatePreparation.resources(num_wires=3)
    {RZ: 27, CNOT: 16}
    """

    def __init__(self, state_vector, wires, id=None):
        # Overriding the default init method to allow for CompactState as an input.

        if isinstance(state_vector, re.CompactState):
            self.compact_state = state_vector
            Operation.__init__(self, state_vector, wires=wires)
            return

        self.compact_state = None
        super().__init__(state_vector, wires, id)


    @staticmethod
    def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            num_wires(int): the number of wires that the operation acts on

        Resources:
            Using the resources as described in `Mottonen et al. (2008) <https://arxiv.org/pdf/quant-ph/0407010>`_.
            The resources are defined as :math:`2^{N+2} - 5` :class:`~.ResourceRZ` gates and
            :math:`2^{N+2} - 4N - 4` :class:`~.ResourceCNOT` gates.
        """
        gate_types = {}
        rz = re.ResourceRZ.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        r_count = 2 ** (num_wires + 2) - 5
        cnot_count = 2 ** (num_wires + 2) - 4 * num_wires - 4

        if r_count:
            gate_types[rz] = r_count

        if cnot_count:
            gate_types[cnot] = cnot_count
        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires(int): the number of wires that the operation acts on
        """
        if self.compact_state:
            return {"num_wires": self.compact_state.num_qubit
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_wires(int): the number of wires that the operation acts on

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"num_wires": num_wires}
        return CompressedResourceOp(cls, params)

    @classmethod
    def tracking_name(cls, num_wires) -> str:
        return f"MottonenStatePrep({num_wires})"


class ResourceSuperposition(qml.Superposition, ResourceOperator):
    """Resource class for the Superposition template.

    Args:
        coeffs (tensor-like[float]): normalized coefficients of the superposition
        bases (tensor-like[int]): basis states of the superposition
        wires (Sequence[int]): wires that the operator acts on
        work_wire (Union[Wires, int, str]): the auxiliary wire used for the permutation

    Resource Parameters:
        * num_stateprep_wires (int): the number of wires used for the operation
        * num_basis_states (int): the number of basis states of the superposition
        * size_basis_state (int): the size of each basis state

    Resources:
        The resources are computed following the PennyLane decomposition of
        the class :class:`~.Superposition`.

        We use the following (somewhat naive) assumptions to approximate the
        resources:

        -   The MottonenStatePreparation routine is assumed for the state prep
            component.
        -   The permutation block requires 2 multi-controlled X gates and a
            series of CNOT gates. On average we will be controlling on and flipping
            half the number of bits in :code:`size_basis`. (i.e for any given basis
            state, half will be ones and half will be zeros).
        -   If the number of basis states provided spans the set of all basis states,
            then we don't need to permute. In general, there is a probability associated
            with not needing to permute wires if the basis states happen to match, we
            estimate this quantity aswell.

    .. seealso:: :class:`~.Superposition`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceSuperposition.resources(num_stateprep_wires=3, num_basis_states=3, size_basis_state=3)
    {MottonenStatePrep(3): 1, CNOT: 2, MultiControlledX: 4}

    """

    def __init__(
        self, coeffs=None, bases=None, wires=None, work_wire=None, state_vect=None, id=None
    ):
        # Overriding the default init method to allow for CompactState as an input.

        if isinstance(state_vect, re.CompactState):
            self.compact_state = state_vect
            Operation.__init__(self, state_vect, wires=wires)
            return

        self.compact_state = None
        super().__init__(coeffs, bases, wires, work_wire, id)

    @staticmethod
    def _resource_decomp(
        num_stateprep_wires, num_basis_states, size_basis_state, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            num_stateprep_wires (int): the number of wires used for the operation
            num_basis_states (int): the number of basis states of the superposition
            size_basis_state (int): the size of each basis state

        Resources:
            The resources are computed following the PennyLane decomposition of
            the class :class:`~.Superposition`.

            We use the following (somewhat naive) assumptions to approximate the
            resources:

            -   The MottonenStatePreparation routine is assumed for the state prep
                component.
            -   The permutation block requires 2 multi-controlled X gates and a
                series of CNOT gates. On average we will be controlling on and flipping
                half the number of bits in :code:`size_basis`. (i.e for any given basis
                state, half will be ones and half will be zeros).
            -   If the number of basis states provided spans the set of all basis states,
                then we don't need to permute. In general, there is a probability associated
                with not needing to permute wires if the basis states happen to match, we
                estimate this quantity aswell.

        """
        gate_types = {}
        msp = re.ResourceMottonenStatePreparation.resource_rep(num_stateprep_wires)
        gate_types[msp] = 1

        cnot = re.ResourceCNOT.resource_rep()
        num_zero_ctrls = size_basis_state // 2
        multi_x = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=size_basis_state,
            num_ctrl_values=num_zero_ctrls,
            num_work_wires=0,
        )

        basis_size = 2**size_basis_state
        prob_matching_basis_states = num_basis_states / basis_size
        num_permutes = round(num_basis_states * (1 - prob_matching_basis_states))

        if num_permutes:
            gate_types[cnot] = num_permutes * (
                size_basis_state // 2
            )  # average number of bits to flip
            gate_types[multi_x] = 2 * num_permutes  # for compute and uncompute

        return gate_types

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_stateprep_wires (int): the number of wires used for the operation
                * num_basis_states (int): the number of basis states of the superposition
                * size_basis_state (int): the size of each basis state
        """
        if self.compact_state:
            num_basis_states = self.compact_state.num_coeffs
            size_basis_state = self.compact_state.num_qubits
            num_stateprep_wires = math.ceil(math.log2(num_basis_states))

        else:
            bases = self.hyperparameters["bases"]
            num_basis_states = len(bases)
            size_basis_state = len(bases[0])  # assuming they are all the same size
            num_stateprep_wires = math.ceil(math.log2(len(self.coeffs)))

        return {
            "num_stateprep_wires": num_stateprep_wires,
            "num_basis_states": num_basis_states,
            "size_basis_state": size_basis_state,
        }

    @classmethod
    def resource_rep(
        cls, num_stateprep_wires, num_basis_states, size_basis_state
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_stateprep_wires (int): the number of wires used for the operation
            num_basis_states (int): the number of basis states of the superposition
            size_basis_state (int): the size of each basis state

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "num_stateprep_wires": num_stateprep_wires,
            "num_basis_states": num_basis_states,
            "size_basis_state": size_basis_state,
        }
        return CompressedResourceOp(cls, params)


class ResourceQROMStatePreparation(Operation, re.ResourceOperator):
    r"""Prepares a quantum state using a Quantum Read-Only Memory (QROM) based approach.

    This operation decomposes the state preparation into a sequence of QROM operations and controlled rotations.

    Args:
        state_vector (TensorLike): The state vector to prepare.
        wires (Sequence[int]): The wires on which to prepare the state.
        precision_wires (Sequence[int]): The wires used for storing the binary representations of the
            amplitudes and phases.
        work_wires (Sequence[int], optional):  The wires used as work wires for the QROM operations. Defaults to ``None``.

    **Example**

    .. code-block::

        dev = qml.device("default.qubit", wires=6)
        state_vector = np.array([1,0,0,0]) / 2.0
        wires = [0, 1]
        precision_wires = [2, 3, 4]
        work_wires = [5]

        @qml.qnode(dev)
        def circuit():
            qml.QROMStatePreparation(state_vector, wires, precision_wires, work_wires)
            return qml.state()

        print(circuit())

    .. details::
        :title: Usage Details

        This operation implements the state preparation method described in
        `arXiv:quant-ph/0208112 <https://arxiv.org/abs/quant-ph/0208112>`.  It uses a QROM to store
        the binary representations of the amplitudes and phases of the target state, and then uses
        controlled rotations to apply these values to the target qubits.

        The input `state_vector` must have a length that is a power of 2. The number of `wires`
        must be at least :math:`\log_2(\text{len}(state\_vector))`. The number of `precision_wires` determines the
        precision with which the amplitudes and phases are encoded.

        The `work_wires` are used as auxiliary qubits in the QROM operation. They should be distinct
        from the `wires` and `precision_wires`.

        The decomposition involves encoding the probabilities and phases of the state vector using
        QROMs and then applying controlled rotations based on the values stored in the `precision_wires`.
        The decomposition applies CRY rotations for amplitude encoding and controlled GlobalPhase rotations for the phase encoding.

        The user must ensure that the number of precision wires is enough to store the values. The relation between the number of precision wires `n_p` and the precision `p` is given by :math:`p = 2^{-n_p}`.
    """

    def __init__(self, state_vector, wires, precision_wires, work_wires=None, id=None):

        self.state_vector = state_vector
        self.hyperparameters["input_wires"] = qml.wires.Wires(wires)
        self.hyperparameters["precision_wires"] = qml.wires.Wires(precision_wires)
        self.hyperparameters["work_wires"] = qml.wires.Wires(
            () if work_wires is None else work_wires
        )

        all_wires = (
            self.hyperparameters["input_wires"]
            + self.hyperparameters["precision_wires"]
            + self.hyperparameters["work_wires"]
        )

        super().__init__(state_vector, wires=all_wires, id=id)

    def decomposition(self):  # pylint: disable=arguments-differ
        filtered_hyperparameters = {
            key: value for key, value in self.hyperparameters.items() if key != "input_wires"
        }
        return self.compute_decomposition(
            self.parameters[0],
            wires=self.hyperparameters["input_wires"],
            **filtered_hyperparameters,
        )

    @staticmethod
    def compute_decomposition(
        state_vector, wires, precision_wires, work_wires
    ):  # pylint: disable=arguments-differ
        """
        Computes the decomposition operations for the given state vector.

        Args:
            state_vector (array-like): The state vector.
            wires (list): List of control wires.
            precision_wires (list): List of precision wires.
            work_wires (list or None): List of work wires (can be None).

        Returns:
            list: List of decomposition operations.
        """
        # Square each component of the state vector
        probs = qml.math.abs(state_vector) ** 2
        phases = qml.math.angle(state_vector) % (2 * np.pi)

        decomp_ops = []
        num_iterations = int(qml.math.log2(len(probs)))
        for i in range(num_iterations):

            # Calculation of the numerator and denominator of the function f (Eq.5 [arXiv:quant-ph/0208112])
            prefixes = get_basis_state_list(n_wires=i)
            probs_denominator = qml.math.array([sum_by_prefix(probs, prefix=p) for p in prefixes])

            prefixes_with_zero = get_basis_state_list(n_wires=i, add_zero=True)
            probs_numerator = qml.math.array(
                [sum_by_prefix(probs, prefix=p) for p in prefixes_with_zero]
            )

            eps = 1e-8  # Small constant to avoid division by zero

            # Compute the binary representations of the angles θi
            func = lambda x: 2 * qml.math.arccos(qml.math.sqrt(x)) / np.pi
            thetas_binary = [
                func_to_binary(
                    len(precision_wires), probs_numerator[j] / (probs_denominator[j] + eps), func
                )
                for j in range(len(probs_numerator))
            ]

            # Apply the QROM operation to encode the thetas binary representation
            decomp_ops.append(
                qml.QROM(
                    bitstrings=thetas_binary,
                    target_wires=precision_wires,
                    control_wires=wires[:i],
                    work_wires=work_wires,
                    clean=False,
                )
            )

            # Turn binary representation into proper rotation
            for ind, wire in enumerate(precision_wires):
                rotation_angle = 2 ** (-ind - 1)
                decomp_ops.append(qml.CRY(np.pi * rotation_angle, wires=[wire, wires[i]]))

            # Clean wires used to store the theta values
            decomp_ops.append(
                qml.adjoint(qml.QROM)(
                    bitstrings=thetas_binary,
                    target_wires=precision_wires,
                    control_wires=wires[:i],
                    work_wires=work_wires,
                    clean=False,
                )
            )

        # Compute the binary representations of the phases
        func = lambda x: (x) / (2 * np.pi)
        thetas_binary = [func_to_binary(len(precision_wires), phase, func) for phase in phases]

        # Apply the QROM operation to encode the thetas binary representation
        decomp_ops.append(
            qml.QROM(
                bitstrings=thetas_binary,
                target_wires=precision_wires,
                control_wires=wires,
                work_wires=work_wires,
                clean=False,
            )
        )

        # Turn binary representation into proper rotation
        for ind, wire in enumerate(precision_wires):
            rotation_angle = 2 ** (-ind - 1)
            decomp_ops.append(
                qml.ctrl(
                    qml.GlobalPhase((2 * np.pi) * (-rotation_angle), wires=wires[0]), control=wire
                )
            )

        # Clean wires used to store the theta values
        decomp_ops.append(
            qml.adjoint(qml.QROM)(
                bitstrings=thetas_binary,
                target_wires=precision_wires,
                control_wires=wires,
                work_wires=work_wires,
                clean=False,
            )
        )

        return decomp_ops

    @staticmethod
    def _resource_decomp(
        num_state_qubits, num_precision_wires, num_work_wires, positive_and_real, **kwargs
    ):
        """The resources associated with a single QROMPrep"""
        gate_types = defaultdict(int)

        for j in range(num_state_qubits):
            num_bitstrings = 2**j
            num_bit_flips = 2 ** (j - 1)
            num_control_wires = j

            gate_types[
                re.ResourceQROM.resource_rep(
                    num_bitstrings,
                    num_bit_flips,
                    num_control_wires,
                    num_work_wires,
                    num_precision_wires,
                    clean=False,
                )
            ] += 1

            gate_types[
                re.ResourceAdjoint.resource_rep(
                    base_class=re.ResourceQROM,
                    base_params={
                        "num_bitstrings": num_bitstrings,
                        "num_bit_flips": num_bit_flips,
                        "num_control_wires": num_control_wires,
                        "num_work_wires": num_work_wires,
                        "size_bitstring": num_precision_wires,
                        "clean": False,
                    },
                )
            ] += 1

        c_ry = re.ResourceCRY.resource_rep()
        gate_types[c_ry] = num_precision_wires * num_state_qubits

        c_gp = re.ResourceControlled.resource_rep(re.ResourceGlobalPhase, {}, 1, 0, 0)

        if not positive_and_real:
            gate_types[
                re.ResourceQROM.resource_rep(
                    2**num_state_qubits,
                    2 ** (num_state_qubits - 1),
                    num_state_qubits,
                    num_work_wires,
                    num_precision_wires,
                    clean=False,
                )
            ] += 1

            gate_types[c_gp] = num_precision_wires

            gate_types[
                re.ResourceAdjoint.resource_rep(
                    base_class=re.ResourceQROM,
                    base_params={
                        "num_bitstrings": 2**num_state_qubits,
                        "num_bit_flips": 2 ** (num_state_qubits - 1),
                        "num_control_wires": num_state_qubits,
                        "num_work_wires": num_work_wires,
                        "size_bitstring": num_precision_wires,
                        "clean": False,
                    },
                )
            ] += 1

        return gate_types
    
    @property
    def resource_params(self) -> dict:
        """The key parameters required to expand the resources of QROMPrep."""
        state_vector = self.state_vector
        positive_and_real = True

        for c in state_vector:
            if c.imag != 0 or c.real < 0:
                positive_and_real = False
                break

        num_state_qubits = int(math.log2(len(self.state_vector)))
        num_precision_wires = len(self.hyperparameters["precision_wires"])
        num_work_wires = len(self.hyperparameters["work_wires"])

        return {
            "num_state_qubits": num_state_qubits,
            "num_precision_wires": num_precision_wires,
            "num_work_wires": num_work_wires,
            "positive_and_real": positive_and_real,
        }

    @classmethod
    def resource_rep(cls, num_state_qubits, num_precision_wires, num_work_wires, positive_and_real):
        params = {
            "num_state_qubits": num_state_qubits,
            "num_precision_wires": num_precision_wires,
            "num_work_wires": num_work_wires,
            "positive_and_real": positive_and_real,
        }
        return re.CompressedResourceOp(cls, params)


class ResourceBasisState(qml.BasisState, ResourceOperator):
    r"""Resource class for the BasisState template.

    Args:
        state (tensor_like): Binary input of shape ``(len(wires), )``. For example, if ``state=np.array([0, 1, 0])`` or ``state=2`` (equivalent to 010 in binary), the quantum system will be prepared in the state :math:`|010 \rangle`.

        wires (Sequence[int] or int): the wire(s) the operation acts on
        id (str): Custom label given to an operator instance. Can be useful for some applications where the instance has to be identified.

    Resource Parameters:
        * num_bit_flips (int): number of qubits in the :math:`|1\rangle` state

    Resources:
        The resources for BasisState are according to the decomposition found in qml.BasisState.

    .. seealso:: :class:`~.BasisState`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceBasisState.resources(num_bit_flips = 6)
    {X: 6}
    """

    def __init__(self, state, wires, id=None):
        # Overriding the default init method to allow for CompactState as an input.

        if isinstance(state, re.CompactState):
            self.compact_state = state
            Operation.__init__(self, state, wires=wires)
            return

        self.compact_state = None
        super().__init__(state, wires, id)

    @staticmethod
    def _resource_decomp(
        num_bit_flips,
        **kwargs,
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            num_bit_flips (int): number of qubits in the :math:`|1\rangle` state

        Resources:
            The resources for BasisState are according to the decomposition found in qml.BasisState.
        """
        gate_types = {}
        x = re.ResourceX.resource_rep()
        gate_types[x] = num_bit_flips

        return gate_types

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_bit_flips (int): number of qubits in the :math:`|1\rangle` state
        """
        if self.compact_state:
            return {"num_bit_flips": self.compact_state.num_bit_flips}

        num_bit_flips = sum(self.parameters[0])
        return {"num_bit_flips": num_bit_flips}

    @classmethod
    def resource_rep(cls, num_bit_flips) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_bit_flips (int): number of qubits in the :math:`|1\rangle` state

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """

        params = {"num_bit_flips": num_bit_flips}
        return CompressedResourceOp(cls, params)

    @classmethod
    def tracking_name(cls, num_bit_flips) -> str:
        return f"BasisState({num_bit_flips})"


class ResourceMPSPrep(qml.MPSPrep, ResourceOperator):
    r"""Resource class for the MPSPrep template.

    Args:
        num_wires (int): number of qubits corresponding to the state preparation register
        num_work_wires (int): number of additional qubits matching the bond dimension of the MPS.

    Resources:
        The resources for MPSPrep are according to the decomposition.

    .. seealso:: :class:`~.MPSPrep`

    """

    def __init__(self, mps, wires, work_wires=None, right_canonicalize=False, id=None):
        # Overriding the default init method to allow for CompactState as an input.

        if isinstance(mps, re.CompactState):
            self.compact_state = mps
            Operation.__init__(self, mps, wires=wires)
            return

        self.compact_state = None
        super().__init__(mps, wires, work_wires, right_canonicalize, id)

    @staticmethod
    def _resource_decomp(
        num_wires,
        num_work_wires,
        **kwargs,
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            num_wires (int): number of qubits corresponding to the state preparation register
            num_work_wires (int): number of additional qubits matching the bond dimension of the MPS.

        Resources:
            The resources for MPSPrep are according to the decomposition.
        """
        gate_types = {}

        qubit_unitary = re.ResourceQubitUnitary.resource_rep(num_wires=num_work_wires + 1)
        gate_types[qubit_unitary] = num_wires
        return gate_types

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Resource parameters:
            num_wires (int): number of qubits corresponding to the state preparation register
            num_work_wires (int): number of additional qubits matching the bond dimension of the MPS.

        Returns:
            dict: dictionary containing the resource parameters
        """
        if self.compact_state:
            return {
                "num_wires": self.compact_state.num_qubits,
                "num_work_wires": self.compact_state.num_work_wires,
            }

        ww = self.hyperparameters["work_wires"]
        num_work_wires = len(ww) if ww else 0
        num_wires = self.hyperparameters["input_wires"]

        return {"num_wires": num_wires, "num_work_wires": num_work_wires}

    @classmethod
    def resource_rep(cls, num_wires, num_work_wires) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_wires (int): number of qubits corresponding to the state preparation register
            num_work_wires (int): number of additional qubits matching the bond dimension of the MPS.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"num_wires": num_wires, "num_work_wires": num_work_wires}
        return CompressedResourceOp(cls, params)

    @classmethod
    def tracking_name(cls, num_wires, num_work_wires) -> str:
        return f"MPSPrep({num_wires}, {num_work_wires})"
