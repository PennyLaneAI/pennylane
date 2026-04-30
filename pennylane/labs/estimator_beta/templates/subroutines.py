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
r"""Resource operators for PennyLane subroutine templates."""

import math

import pennylane.labs.estimator_beta as qre
from pennylane.estimator import CompressedResourceOp, GateCount, ResourceOperator, resource_rep
from pennylane.labs.estimator_beta.wires_manager.base_classes import Allocate, Deallocate
from pennylane.math import ceil_log2
from pennylane.wires import WiresLike

# pylint: disable=unused-argument


def selectpaulirot_controlled_resource_decomp(
    num_ctrl_wires: int, num_zero_ctrl: int, target_resource_params: dict
) -> list[GateCount]:
    r"""Returns a list representing the resources of the controlled version of the :class:`~pennylane.estimator.templates.SelectPauliRot` operator.
    Each object in the list
    represents a gate and the number of times it occurs in the circuit.

    Args:
        num_ctrl_wires (int): the number of qubits the operation is controlled on
        num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
        target_resource_params (dict): A dictionary containing the resource parameters
            of the target operator.

    Resources:
        The resources are obtained from the construction scheme given in `Möttönen and Vartiainen
        (2005), Fig 7a <https://arxiv.org/abs/quant-ph/0504100>`_. Specifically, the resources
        for an :math:`n` qubit unitary are given as :math:`2^{n}` instances of the :code:`CNOT`
        gate and :math:`2^{n}` instances of the controlled single qubit rotation gate (:code:`RX`,
        :code:`RY` or :code:`RZ`) depending on the :code:`rot_axis`.

    Returns:
        list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
        represents a specific quantum gate and the number of times it appears
        in the decomposition.
    """

    num_ctrl_wires_base = target_resource_params["num_ctrl_wires"]
    rot_axis = target_resource_params["rot_axis"]
    precision = target_resource_params["precision"]

    rotation_gate_map = {
        "X": qre.RX,
        "Y": qre.RY,
        "Z": qre.RZ,
    }
    gate_lst = []

    gate = resource_rep(
        qre.Controlled,
        {
            "base_cmpr_op": resource_rep(rotation_gate_map[rot_axis], {"precision": precision}),
            "num_ctrl_wires": num_ctrl_wires,
            "num_zero_ctrl": num_zero_ctrl,
        },
    )
    cnot = resource_rep(qre.CNOT)

    gate_lst.append(GateCount(gate, 2**num_ctrl_wires_base))
    gate_lst.append(GateCount(cnot, 2**num_ctrl_wires_base))

    return gate_lst


def qft_phase_grad_resource_decomp(num_wires) -> list[GateCount]:
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

    # Use qubits from phase gradient register
    phase_grad_reg = Allocate(num_wires=num_wires - 1, state="any", restored=True)

    gate_types = [
        phase_grad_reg,
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

    gate_types.append(Deallocate(allocated_register=phase_grad_reg))
    return gate_types


def aqft_resource_decomp(order, num_wires) -> list[GateCount]:
    r"""Returns a list representing the resources of the operator. Each object in the list
    represents a gate and the number of times it occurs in the circuit.

    Args:
        order (int): the maximum number of controlled phase shifts to which the operation is truncated
        num_wires (int): the number of qubits the operation acts upon

    Resources:
        The resources are obtained from (Fig. 4) `arXiv:1803.04933 <https://arxiv.org/abs/1803.04933>`_
        excluding the gate cost of preparing the phase gradient state. The phased Toffoli gates and the
        classical measure-and-reset (Fig. 2) are accounted for as :code:`TemporaryAND` operations.

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

    if order > 1:
        # Use qubits from the phase gradient register
        phase_grad_register = Allocate(order - 1, state="any", restored=True)
        gate_types.append(phase_grad_register)

        if num_wires > 1:
            gate_types.append(GateCount(cs, num_wires - 1))

            for index in range(2, order):
                addition_reg_size = index - 1
                temp_and_register = Allocate(addition_reg_size, state="zero", restored=True)

                temp_and = resource_rep(qre.TemporaryAND)
                temp_and_dag = qre.Adjoint.resource_rep(temp_and)
                in_place_add = qre.SemiAdder.resource_rep(addition_reg_size)

                cost_iter = [
                    temp_and_register,
                    GateCount(temp_and, addition_reg_size),
                    GateCount(in_place_add),
                    GateCount(hadamard),
                    GateCount(temp_and_dag, addition_reg_size),
                    Deallocate(allocated_register=temp_and_register),
                ]
                gate_types.extend(cost_iter)

            addition_reg_size = order - 1
            temp_and_register = Allocate(addition_reg_size, state="zero", restored=True)
            repetitions = num_wires - order

            temp_and = resource_rep(qre.TemporaryAND)
            temp_and_dag = qre.Adjoint.resource_rep(temp_and)
            in_place_add = qre.SemiAdder.resource_rep(addition_reg_size)

            cost_iter = [
                temp_and_register,
                GateCount(temp_and, addition_reg_size * repetitions),
                GateCount(in_place_add, repetitions),
                GateCount(hadamard, repetitions),
                GateCount(temp_and_dag, addition_reg_size * repetitions),
                Deallocate(allocated_register=temp_and_register),
            ]
            gate_types.extend(cost_iter)

            gate_types.append(GateCount(swap, num_wires // 2))

        gate_types.append(Deallocate(allocated_register=phase_grad_register))

    return gate_types


def select_thc_resource_decomp(
    thc_ham: qre.THCHamiltonian,
    num_batches: int = 1,
    rotation_precision: int = 15,
    select_swap_depth: int | None = None,
) -> list[GateCount]:
    r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
    and the number of times it occurs in the decomposition.

    .. note::

        This decomposition assumes that an appropriately sized phase gradient state is available.
        Users should ensure that the cost of constructing this state has been accounted for.
        See also :class:`~.pennylane.estimator.templates.subroutines.PhaseGradient`.

    Args:
        thc_ham (:class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`): A tensor hypercontracted
            Hamiltonian on which this ``Select`` operator is being applied.
        num_batches (int): The number of batches for loading Givens rotation angles
            into temporary quantum registers. Must be less than the number of orbitals in ``thc_ham``.
            The default value of ``1`` loads all angles in one batch.
        rotation_precision (int): The number of bits used to represent the precision for loading
            the rotation angles for basis rotation. The default value is set to ``15`` bits.
        select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.subroutines.QROM`
            used to trade off extra wires for reduced circuit depth. Defaults to :code:`None`, in which
            case, the ``select_swap_depth`` is set to the optimal depth that minimizes the total
            ``T``-gate count.

    Resources:
        The resources are calculated based on Figure 5 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_ and
        Figure 4 in `arXiv:2501.06165 <https://arxiv.org/abs/2501.06165>`_.

    Returns:
        list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
        represents a specific quantum gate and the number of times it appears
        in the decomposition.
    """

    num_orb = thc_ham.num_orbitals
    tensor_rank = thc_ham.tensor_rank

    gate_list = []
    # Total select cost from Eq. 43 in arXiv:2011.03494

    # access the phase gradient register:
    phase_grad_reg = Allocate(rotation_precision - 1, state="any", restored=True)
    gate_list.append(phase_grad_reg)

    # 4 swaps on state registers controlled on spin qubits
    cswap = resource_rep(qre.CSWAP)
    gate_list.append(GateCount(cswap, 4 * num_orb))

    restore_qrom = num_batches != 1

    batched_rotations = int(math.ceil((num_orb - 1) / num_batches))

    # Data output for rotations
    data_reg = Allocate(rotation_precision * batched_rotations, state="zero", restored=True)
    gate_list.append(data_reg)

    # QROM to load rotation angles for both 1-body and 2-body integrals
    qrom_full = resource_rep(
        qre.QROM,
        {
            "num_bitstrings": tensor_rank + num_orb,
            "size_bitstring": rotation_precision * batched_rotations,
            "borrow_qubits": restore_qrom,
            "select_swap_depth": select_swap_depth,
        },
    )
    gate_list.append(GateCount(qrom_full, num_batches))

    # Cost for rotations by adding the rotations into the phase gradient state
    semiadder = resource_rep(
        qre.Controlled,
        {
            "base_cmpr_op": resource_rep(
                qre.SemiAdder,
                {"max_register_size": rotation_precision - 1},
            ),
            "num_ctrl_wires": 1,
            "num_zero_ctrl": 0,
        },
    )
    gate_list.append(GateCount(semiadder, num_orb - 1))

    # Adjoint of QROM for loading both 1-body and 2-body integrals Eq. 34 in arXiv:2011.03494
    gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_full})))
    # Adjoint of semiadder for 1-body and 2-body integrals
    gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": semiadder}), num_orb - 1))

    # QROM to load rotation angles for two body integrals
    qrom_twobody = resource_rep(
        qre.QROM,
        {
            "num_bitstrings": tensor_rank,
            "size_bitstring": rotation_precision * batched_rotations,
            "borrow_qubits": restore_qrom,
            "select_swap_depth": select_swap_depth,
        },
    )
    gate_list.append(GateCount(qrom_twobody, num_batches))

    # Cost for rotations by adding the rotations into the phase gradient state
    gate_list.append(GateCount(semiadder, num_orb - 1))

    # Clifford cost for rotations
    h = resource_rep(qre.Hadamard)
    s = resource_rep(qre.S)
    s_dagg = resource_rep(qre.Adjoint, {"base_cmpr_op": s})

    gate_list.append(GateCount(h, 4 * (num_orb)))
    gate_list.append(GateCount(s, 2 * num_orb))
    gate_list.append(GateCount(s_dagg, 2 * num_orb))

    # Adjoint of QROM for two body integrals Eq. 35 in arXiv:2011.03494
    gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_twobody})))

    # Adjoint of semiadder for two body integrals
    gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": semiadder}), num_orb - 1))

    # Z gate in the center of rotations
    gate_list.append(qre.GateCount(resource_rep(qre.Z)))

    cz = resource_rep(qre.CZ)
    gate_list.append(qre.GateCount(cz, 1))

    # 1 cswap between the spin registers
    gate_list.append(qre.GateCount(cswap, 1))
    gate_list.append(Deallocate(allocated_register=data_reg))  # release data register

    gate_list.append(Deallocate(allocated_register=phase_grad_reg))  # release phase grad register
    return gate_list


def select_thc_controlled_resource_decomp(
    num_ctrl_wires: int, num_zero_ctrl: int, target_resource_params: dict
) -> list[GateCount]:
    r"""Returns a list representing the resources for the controlled version of the operator.

    .. note::

        This decomposition assumes that an appropriately sized phase gradient state is available.
        Users should ensure that the cost of constructing this state has been accounted for.
        See also :class:`~.pennylane.estimator.templates.subroutines.PhaseGradient`.

    Args:
        num_ctrl_wires (int): the number of wires the operation is controlled on
        num_zero_ctrl (int): the number of control wires, that are controlled when in the :math:`|0\rangle` state
        target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

    Resources:
        The resources are calculated based on Figure 5 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_.

    Returns:
        list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
        represents a specific quantum gate and the number of times it appears
        in the decomposition.
    """

    thc_ham = target_resource_params["thc_ham"]
    rotation_precision = target_resource_params["rotation_precision"]
    select_swap_depth = target_resource_params["select_swap_depth"]
    num_batches = target_resource_params["num_batches"]

    num_orb = thc_ham.num_orbitals
    tensor_rank = thc_ham.tensor_rank

    gate_list = []

    # access the phase gradient register:
    phase_grad_reg = Allocate(rotation_precision - 1, state="any", restored=True)
    gate_list.append(phase_grad_reg)

    if num_ctrl_wires > 1:
        mcx = resource_rep(
            qre.MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        aux_reg = Allocate(1, state="zero", restored=True)
        gate_list.append(aux_reg)
        gate_list.append(GateCount(mcx, 2))

    # 4 swaps on state registers controlled on spin qubits
    cswap = resource_rep(qre.CSWAP)
    gate_list.append(GateCount(cswap, 4 * num_orb))

    restore_qrom = num_batches != 1

    batched_rotations = int(math.ceil((num_orb - 1) / num_batches))

    # Data output for rotations
    data_reg = Allocate(rotation_precision * batched_rotations, state="zero", restored=True)
    gate_list.append(data_reg)

    # QROM for loading rotation angles for 1-body and 2-body integrals
    qrom_full = resource_rep(
        qre.QROM,
        {
            "num_bitstrings": tensor_rank + num_orb,
            "size_bitstring": rotation_precision * batched_rotations,
            "borrow_qubits": restore_qrom,
            "select_swap_depth": select_swap_depth,
        },
    )
    gate_list.append(GateCount(qrom_full, num_batches))

    # Cost for rotations by adding the rotations into the phase gradient state
    semiadder = resource_rep(
        qre.Controlled,
        {
            "base_cmpr_op": resource_rep(
                qre.SemiAdder,
                {"max_register_size": rotation_precision - 1},
            ),
            "num_ctrl_wires": 1,
            "num_zero_ctrl": 0,
        },
    )
    gate_list.append(GateCount(semiadder, num_orb - 1))

    # Adjoint of QROM for 1-body and 2-body integrals Eq. 34 in arXiv:2011.03494
    gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_full})))
    # Adjoint of semiadder for 1-body and 2-body integrals
    gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": semiadder}), num_orb - 1))

    # QROM for loading rotation angles for two body integrals
    qrom_twobody = resource_rep(
        qre.QROM,
        {
            "num_bitstrings": tensor_rank,
            "size_bitstring": rotation_precision * batched_rotations,
            "borrow_qubits": restore_qrom,
            "select_swap_depth": select_swap_depth,
        },
    )
    gate_list.append(GateCount(qrom_twobody, num_batches))

    # Cost for rotations by adding the rotations into the phase gradient state
    gate_list.append(GateCount(semiadder, num_orb - 1))

    # Clifford cost for rotations
    h = resource_rep(qre.Hadamard)
    s = resource_rep(qre.S)
    s_dagg = resource_rep(qre.Adjoint, {"base_cmpr_op": s})

    gate_list.append(GateCount(h, 4 * (num_orb)))
    gate_list.append(GateCount(s, 2 * num_orb))
    gate_list.append(GateCount(s_dagg, 2 * num_orb))

    # Adjoint of QROM for two body integrals Eq. 35 in arXiv:2011.03494
    gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_twobody})))
    # Adjoint of semiadder for two body integrals
    gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": semiadder}), num_orb - 1))

    # Z gate in the center of rotations
    cz = resource_rep(qre.CZ)
    gate_list.append(qre.GateCount(cz, 1))

    ccz = resource_rep(
        qre.Controlled,
        {
            "base_cmpr_op": qre.Z.resource_rep(),
            "num_ctrl_wires": 2,
            "num_zero_ctrl": 1,
        },
    )
    gate_list.append(qre.GateCount(ccz, 1))

    # 1 cswap between the spin registers
    gate_list.append(qre.GateCount(cswap, 1))

    gate_list.append(Deallocate(allocated_register=data_reg))  # release data register

    if num_ctrl_wires > 1:
        gate_list.append(Deallocate(allocated_register=aux_reg))
    elif num_zero_ctrl > 0:
        gate_list.append(GateCount(resource_rep(qre.X), 2 * num_zero_ctrl))

    gate_list.append(Deallocate(allocated_register=phase_grad_reg))  # release phase grad register
    return gate_list


def qrom_state_preparation_resource_decomp(
    num_state_qubits: int,
    positive_and_real: bool,
    precision: float | None = None,
    selswap_depths=1,
) -> list[GateCount]:
    r"""Returns a list representing the resources of the operator. Each object in the list
    represents a gate and the number of times it occurs in the circuit.

    Args:
        num_state_qubits (int): number of qubits required to represent the state-vector
        positive_and_real (bool): Flag that the coefficients of the statevector are all real
            and positive.
        precision (float): The precision threshold for loading in the binary representation
            of the rotation angles.
        selswap_depths (int | Iterable(int) | None): A parameter of :code:`QROM`
            used to trade-off extra qubits for reduced circuit depth.

    Resources:
        The resources for QROMStatePreparation are according to the decomposition as described
        in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_, using
        :class:`~.pennylane.labs.estimator_beta.templates.subroutines.LabsQROM` to dynamically
        load the rotation angles. Controlled-RY (and phase shifts) gates are used to apply all of
        the rotations coherently.

    Returns:
        list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of
        ``GateCount`` objects, where each object represents a specific quantum gate and the
        number of times it appears in the decomposition.
    """
    gate_counts = []
    cry = qre.CRY.resource_rep()
    phase_shift = qre.PhaseShift.resource_rep()

    expected_size = num_state_qubits if positive_and_real else num_state_qubits + 1
    if isinstance(selswap_depths, int) or selswap_depths is None:
        selswap_depths = [selswap_depths] * expected_size

    num_precision_wires = ceil_log2(math.pi / precision)
    load_reg = Allocate(num_precision_wires, state="zero", restored=True)

    gate_counts.append(load_reg)  # allocate load register

    for j in range(num_state_qubits):
        num_bitstrings = 2**j
        num_bit_flips = num_bitstrings * num_precision_wires // 2

        gate_counts.append(
            GateCount(
                qre.ChangeOpBasis.resource_rep(
                    cmpr_compute_op=LabsQROM.resource_rep(
                        num_bitstrings=num_bitstrings,
                        size_bitstring=num_precision_wires,
                        num_bit_flips=num_bit_flips,
                        borrow_qubits=True,
                        select_swap_depth=selswap_depths[j],
                    ),
                    cmpr_target_op=qre.Prod.resource_rep(
                        cmpr_factors_and_counts=((cry, num_precision_wires),),
                        num_wires=num_precision_wires + 1,
                    ),
                    num_wires=j + num_precision_wires + 1,
                ),
            )
        )

    if not positive_and_real:
        gate_counts.append(
            GateCount(
                qre.ChangeOpBasis.resource_rep(
                    cmpr_compute_op=LabsQROM.resource_rep(
                        num_bitstrings=2**num_state_qubits,
                        size_bitstring=num_precision_wires,
                        num_bit_flips=((2**num_state_qubits) * num_precision_wires // 2),
                        borrow_qubits=True,
                        select_swap_depth=selswap_depths[-1],
                    ),
                    cmpr_target_op=qre.Prod.resource_rep(
                        cmpr_factors_and_counts=((phase_shift, num_precision_wires),),
                        num_wires=num_precision_wires,
                    ),
                ),
            )
        )

    gate_counts.append(Deallocate(allocated_register=load_reg))  # free load register
    return gate_counts


def qrom_state_preparation_phase_grad_resource_decomp(
    num_state_qubits: int,
    positive_and_real: bool,
    precision: float | None = None,
    selswap_depths=1,
) -> list[GateCount]:
    r"""Returns a list representing the resources of the operator. Each object in the list
    represents a gate and the number of times it occurs in the circuit.

    .. note::

        This decomposition assumes an appropriately sized phase gradient state is available.
        Users should ensure the cost of constructing such a state has been accounted for.
        See also :class:`~.pennylane.pennylane.estimator.PhaseGradient`.

    Args:
        num_state_qubits (int): number of qubits required to represent the state-vector
        positive_and_real (bool): Flag that the coefficients of the statevector are all real
            and positive.
        precision (float): The precision threshold for loading in the binary representation
            of the rotation angles.
        selswap_depths (int | Iterable(int) | None): A parameter of :code:`QROM`
            used to trade-off extra qubits for reduced circuit depth.

    Resources:
        The resources for QROMStatePreparation are according to the decomposition as described
        in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_, using
        :class:`~.pennylane.labs.estimator_beta.templates.subroutines.LabsQROM` to dynamically
        load the rotation angles. These rotations gates are implmented using an inplace
        controlled-adder operation (see figure 4. of `arXiv:2409.07332 <https://arxiv.org/abs/2409.07332>`_)
        to phase gradient.

    Returns:
        list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of
        ``GateCount`` objects, where each object represents a specific quantum gate and the
        number of times it appears in the decomposition.
    """
    gate_counts = []
    h = qre.Hadamard.resource_rep()
    s = qre.S.resource_rep()
    s_dagg = qre.Adjoint.resource_rep(base_cmpr_op=s)

    expected_size = num_state_qubits if positive_and_real else num_state_qubits + 1
    if isinstance(selswap_depths, int) or selswap_depths is None:
        selswap_depths = [selswap_depths] * expected_size

    num_precision_wires = ceil_log2(math.pi / precision)
    semi_adder = qre.SemiAdder.resource_rep(max_register_size=num_precision_wires)
    ctrl_semi_adder = qre.Controlled.resource_rep(semi_adder, 1, 0)

    load_reg = Allocate(num_precision_wires, state="zero", restored=True)
    phase_grad_reg = Allocate(num_precision_wires, state="any", restored=True)

    gate_counts.append(load_reg)  # allocate load register
    gate_counts.append(phase_grad_reg)  # grab qubits from phase grad register

    # map RY rotations to RZ for phase grad
    gate_counts.append(GateCount(h, num_precision_wires))
    gate_counts.append(GateCount(s, num_precision_wires))

    for j in range(num_state_qubits):
        num_bitstrings = 2**j
        num_bit_flips = num_bitstrings * num_precision_wires // 2

        gate_counts.append(
            GateCount(
                qre.ChangeOpBasis.resource_rep(
                    cmpr_compute_op=LabsQROM.resource_rep(
                        num_bitstrings=num_bitstrings,
                        size_bitstring=num_precision_wires,
                        num_bit_flips=num_bit_flips,
                        borrow_qubits=True,
                        select_swap_depth=selswap_depths[j],
                    ),
                    cmpr_target_op=ctrl_semi_adder,
                    num_wires=j + 2 * num_precision_wires,
                )
            )
        )

    if not positive_and_real:
        gate_counts.append(
            GateCount(
                qre.ChangeOpBasis.resource_rep(
                    cmpr_compute_op=LabsQROM.resource_rep(
                        num_bitstrings=2**num_state_qubits,
                        size_bitstring=num_precision_wires,
                        num_bit_flips=((2**num_state_qubits) * num_precision_wires // 2),
                        borrow_qubits=True,
                        select_swap_depth=selswap_depths[-1],
                    ),
                    cmpr_target_op=ctrl_semi_adder,
                    num_wires=num_state_qubits + 2 * num_precision_wires,
                )
            )
        )

    # map RY rotations to RZ for phase grad
    gate_counts.append(GateCount(h, num_precision_wires))
    gate_counts.append(GateCount(s_dagg, num_precision_wires))

    gate_counts.append(Deallocate(allocated_register=load_reg))  # free load register
    gate_counts.append(Deallocate(allocated_register=phase_grad_reg))  # free phase grad register
    return gate_counts


# pylint: disable=arguments-differ,too-many-arguments
class LabsQROM(ResourceOperator):
    r"""Resource class for the Quantum Read-Only Memory (QROM) template.

    Args:
        num_bitstrings (int): the number of bitstrings that are to be encoded
        size_bitstring (int): the length of each bitstring
        num_bit_flips (int | None): The total number of :math:`1`'s in the dataset. Defaults to
            :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
        borrow_qubits (bool): Determine whether the auxiliary qubits should be borrowed (higher gate
            cost) or freshly allocated (higher qubit cost). Defaults to :code:`True`.
        select_swap_depth (int | None): A parameter :math:`\lambda` that determines
            if data will be loaded in parallel by adding more rows following Figure 1.C of
            `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be :code:`None`,
            :code:`1` or a positive integer power of two. Defaults to ``None``, which sets the
            depth that minimizes T-gate count.
        wires (WiresLike | None): The wires the operation acts on (control and target), excluding
            any additional qubits allocated during the decomposition (e.g select-swap wires).

    Resources:
        The resources for QROM are derived from Appendix A, B from `Berry et al. (2019)
        <https://arxiv.org/abs/1902.02134>`_.

        * :code:`borrow_qubits=True`: Uses the borrowed qubit decomposition from Figure 4 of Appendix A in
          `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`_.

        * :code:`borrow_qubits=False`: Uses the clean qubit decomposition from Appendix B in
          `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`_.

    .. seealso::
        The associated PennyLane operation :class:`~.pennylane.QROM` and the resource operator
        :class:`~.pennylane.estimator.templates.subroutines.QROM`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.labs.estimator_beta as qre
    >>> qrom = qre.LabsQROM(
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
    Total gates : 86
    'Toffoli': 8,
    'CNOT': 36,
    'X': 18,
    'Hadamard': 24
    """

    resource_keys = {
        "num_bitstrings",
        "size_bitstring",
        "num_bit_flips",
        "select_swap_depth",
        "borrow_qubits",
    }

    @staticmethod
    def _t_optimized_select_swap_width(num_bitstrings, size_bitstring, borrow):
        pre_factor = 1 / 2 if borrow else 1
        opt_width_continuous = math.sqrt(pre_factor * (num_bitstrings / size_bitstring))

        if opt_width_continuous < 1:
            # The continuous solution could be non-physical
            w1 = w2 = 1
        else:
            w1 = 2 ** int(math.floor(math.log2(opt_width_continuous)))
            w2 = 2 ** ceil_log2(opt_width_continuous)

        def t_cost_func(w, borrow):
            sel_factor, swap_factor = (2, 4) if borrow else (1, 1)
            return (
                sel_factor * (math.ceil(num_bitstrings / w) - 2)
                + swap_factor * (w - 1) * size_bitstring
            )

        if t_cost_func(w2, borrow) < t_cost_func(w1, borrow):
            return w2
        return w1

    def __init__(
        self,
        num_bitstrings: int,
        size_bitstring: int,
        num_bit_flips: int | None = None,
        borrow_qubits: bool = True,
        select_swap_depth: int | None = None,
        wires: WiresLike | None = None,
        **kwargs,
    ) -> None:
        if "restored" in kwargs:
            raise ValueError(
                "'restored' is no longer a supported argument for 'labs.estimator_beta.LabsQROM'."
                "Use 'borrow_qubits = True' instead. Alternatively import 'QROM' from 'pennylane.estimator'."
            )

        self.borrow_qubits = borrow_qubits
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
                * borrow_qubits (bool): Determine whether the auxiliary qubits should be borrowed
                  (higher gate cost) or freshly allocated (higher qubit cost). Defaults to :code:`True`.
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
            "borrow_qubits": self.borrow_qubits,
        }

    @classmethod
    def resource_rep(
        cls,
        num_bitstrings: int,
        size_bitstring: int,
        num_bit_flips: int | None = None,
        borrow_qubits: bool = True,
        select_swap_depth: int | None = None,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_bitstrings (int): the number of bitstrings that are to be encoded
            size_bitstring (int): the length of each bitstring
            num_bit_flips (int | None): The total number of :math:`1`'s in the dataset. Defaults to
                :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
            borrow_qubits (bool): Determine whether the auxiliary qubits should be borrowed (higher gate
                cost) or freshly allocated (higher qubit cost). Defaults to :code:`True`.
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
            "borrow_qubits": borrow_qubits,
        }
        num_wires = size_bitstring + ceil_log2(num_bitstrings)
        return CompressedResourceOp(cls, num_wires, params)

    # pylint: disable=protected-access
    @classmethod
    def resource_decomp(
        cls,
        num_bitstrings: int,
        size_bitstring: int,
        num_bit_flips: int | None = None,
        borrow_qubits: bool = True,
        select_swap_depth: int | None = None,
    ) -> list[GateCount]:
        r"""Returns a list of ``GateCount`` objects representing the operator's resources.

        Args:
            num_bitstrings (int): the number of bitstrings that are to be encoded
            size_bitstring (int): the length of each bitstring
            num_bit_flips (int | None): The total number of :math:`1`'s in the dataset. Defaults to
                :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
            borrow_qubits (bool): Determine whether the auxiliary qubits should be borrowed (higher gate
                cost) or freshly allocated (higher qubit cost). Defaults to :code:`True`.
            select_swap_depth (int | None): A parameter :math:`\lambda` that determines
                if data will be loaded in parallel by adding more rows following Figure 1.C of
                `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be :code:`None`,
                :code:`1` or a positive integer power of two. Defaults to ``None``, which sets the
                depth that minimizes T-gate count.

        Resources:
            The resources for QROM are derived from Appendix A, B from `Berry et al. (2019)
            <https://arxiv.org/abs/1902.02134>`_.

            * :code:`borrow_qubits=True`: Uses the borrowed qubit decomposition from Figure 4 of Appendix A in
              `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`_.

            * :code:`borrow_qubits=False`: Uses the clean qubit decomposition from Appendix B in
              `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`_.

            Note: we use the unary iterator trick to implement the ``Select``. This
            implementation assumes we have access to :math:`n - 1` additional
            work qubits, where :math:`n = \left\lceil \log_{2}(N) \right\rceil` and :math:`N` is
            the number of batches of unitaries to select.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if select_swap_depth:
            max_depth = 2 ** ceil_log2(num_bitstrings)
            select_swap_depth = min(max_depth, select_swap_depth)  # truncate depth beyond max depth

        W_opt = select_swap_depth or cls._t_optimized_select_swap_width(
            num_bitstrings,
            size_bitstring,
            borrow_qubits,
        )
        L_opt = math.ceil(num_bitstrings / W_opt)
        l = ceil_log2(L_opt)

        gate_cost = []
        if L_opt > 2:
            aux_wires_select = Allocate(l - 1, state="zero", restored=True)
            gate_cost.append(aux_wires_select)  # Aux wires for Select (UI)

        if W_opt > 1:
            aux_wires_swap = Allocate(
                num_wires=(W_opt - 1) * size_bitstring,
                state="any" if borrow_qubits else "zero",
                restored=True,
            )
            gate_cost.append(aux_wires_swap)

        hadamard = resource_rep(qre.Hadamard)

        swap_restored_prefactor = 1
        select_restored_prefactor = 1

        if borrow_qubits and (W_opt > 1):
            gate_cost.append(GateCount(hadamard, 2 * size_bitstring))
            swap_restored_prefactor = 4
            select_restored_prefactor = 2

        # SELECT cost:
        gate_cost.extend(cls._select_cost(L_opt, num_bit_flips, select_restored_prefactor))

        if L_opt > 2:
            gate_cost.append(
                Deallocate(allocated_register=aux_wires_select)
            )  # release Select aux wires

        # SWAP cost:
        if W_opt > 1:
            gate_cost.extend(
                cls._swap_cost(size_bitstring, ceil_log2(W_opt), swap_restored_prefactor)
            )

            if (
                not borrow_qubits
            ):  # X-axis measurement & reset (Figure 5 https://arxiv.org/abs/1902.02134)
                gate_cost.append(GateCount(hadamard, (W_opt - 1) * size_bitstring))

            gate_cost.append(Deallocate(allocated_register=aux_wires_swap))

        return gate_cost

    @classmethod
    def single_controlled_res_decomp(
        cls,
        num_bitstrings: int,
        size_bitstring: int,
        num_bit_flips: int | None = None,
        select_swap_depth: int | None = None,
        borrow_qubits: bool = True,
    ):
        r"""The resource decomposition for LabsQROM controlled on a single wire."""
        if select_swap_depth:
            max_depth = 2 ** ceil_log2(num_bitstrings)
            select_swap_depth = min(max_depth, select_swap_depth)  # truncate depth beyond max depth

        W_opt = select_swap_depth or cls._t_optimized_select_swap_width(
            num_bitstrings,
            size_bitstring,
            borrow_qubits,
        )
        L_opt = math.ceil(num_bitstrings / W_opt)
        l = ceil_log2(L_opt)

        gate_cost = []
        if L_opt > 1:
            aux_wires_select = Allocate(l, state="zero", restored=True)
            gate_cost.append(aux_wires_select)  # Aux wires for Select (UI)

        if W_opt > 1:
            aux_wires_swap = Allocate(
                num_wires=(W_opt - 1) * size_bitstring,
                state="any" if borrow_qubits else "zero",
                restored=True,
            )
            gate_cost.append(aux_wires_swap)

        hadamard = resource_rep(qre.Hadamard)

        swap_restored_prefactor = 1
        select_restored_prefactor = 1

        if borrow_qubits and (W_opt > 1):
            gate_cost.append(GateCount(hadamard, 2 * size_bitstring))
            swap_restored_prefactor = 4
            select_restored_prefactor = 2

        # SELECT cost:
        gate_cost.extend(
            cls._single_ctrl_select_cost(L_opt, num_bit_flips, select_restored_prefactor)
        )

        if L_opt > 1:
            gate_cost.append(
                Deallocate(allocated_register=aux_wires_select)
            )  # release Select aux wires

        # SWAP cost:
        if W_opt > 1:
            gate_cost.extend(
                cls._single_ctrl_swap_cost(
                    size_bitstring, ceil_log2(W_opt), swap_restored_prefactor
                )
            )

            if (
                not borrow_qubits
            ):  # X-axis measurement & reset (Figure 5 https://arxiv.org/abs/1902.02134)
                gate_cost.append(
                    GateCount(resource_rep(qre.Hadamard), (W_opt - 1) * size_bitstring)
                )

            gate_cost.append(Deallocate(allocated_register=aux_wires_swap))

        return gate_cost

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
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            The resources for QROM are derived from Appendix A, B from `Berry et al. (2019)
            <https://arxiv.org/abs/1902.02134>`_.

            * :code:`borrow_qubits=True`: Uses the borrowed qubit decomposition from Figure 4 of Appendix A in
              `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`_.

            * :code:`borrow_qubits=False`: Uses the clean qubit decomposition from Appendix B in
              `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`_.

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
        borrow_qubits = target_resource_params.get("borrow_qubits", True)
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
            borrow_qubits,
        )

        if num_ctrl_wires == 1:
            gate_cost.extend(single_ctrl_cost)
            return gate_cost

        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})
        aux_reg = Allocate(num_ctrl_wires - 1, state="zero", restored=True)

        gate_cost.append(aux_reg)
        gate_cost.append(GateCount(l_elbow, num_ctrl_wires - 1))
        gate_cost.extend(single_ctrl_cost)
        gate_cost.append(GateCount(r_elbow, num_ctrl_wires - 1))
        gate_cost.append(Deallocate(allocated_register=aux_reg))
        return gate_cost

    @classmethod
    def _select_cost(
        cls, num_data_blocks: int, num_bit_flips: int, repeat: int = 1
    ) -> list[GateCount]:
        r"""The gate cost of a partial Select subroutine. This decomposition was obtained directly from
        the implementation in `pennylane.templates.subroutines.select._select_resources_unary` assuming
        `partial = True`.

        Args:
            num_data_blocks(int): The number of data blocks formed by partitioning the
                total bitstrings based on select-swap depth.
            num_bit_flips (int): the total number of :math:`1`'s in the dataset
            repeat (int): The number of times to repeat the subroutine.

        Returns:
            list[GateCount]: the resource decomposition of the Select subroutine
        """
        x = resource_rep(qre.X)
        cnot = resource_rep(qre.CNOT)
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})

        gate_cost = []
        if num_data_blocks == 1:
            gate_cost.append(
                GateCount(
                    x,
                    num_bit_flips * repeat,
                )  # each unitary in the select is just an X gate to load the data
            )

        elif num_data_blocks == 2:
            gate_cost.append(
                GateCount(x, 2 * repeat),
            )  # for the 0-control value in the CNOTs
            gate_cost.append(
                GateCount(
                    cnot,
                    num_bit_flips * repeat,
                )  # each unitary in the select is just a CNOT
            )

        elif num_data_blocks / 2 ** ceil_log2(num_data_blocks) > 3 / 4:
            gate_cost.append(
                GateCount(x, repeat * (2 * (num_data_blocks - 3 + 1)))
            )  # conjugate 0-control in left-elbows + 1 extra 0-control CNOT from unary iterator decomp
            gate_cost.append(
                GateCount(
                    cnot,
                    repeat * num_data_blocks + repeat * num_bit_flips,
                )  # num CNOTs in unary iterator trick + each unitary in the select is just a CNOT
            )
            gate_cost.append(GateCount(l_elbow, repeat * (num_data_blocks - 3)))
            gate_cost.append(GateCount(r_elbow, repeat * (num_data_blocks - 3)))

        else:
            gate_cost.append(
                GateCount(x, repeat * (2 * (num_data_blocks - 2 + 1)))
            )  # conjugate 0-control in left-elbows + 1 extra 0-control CNOT from unary iterator decomp
            gate_cost.append(
                GateCount(
                    cnot,
                    repeat * (num_data_blocks - 2) + repeat * num_bit_flips,
                )  # num CNOTs in unary iterator trick + each unitary in the select is just a CNOT
            )
            gate_cost.append(GateCount(l_elbow, repeat * (num_data_blocks - 2)))
            gate_cost.append(GateCount(r_elbow, repeat * (num_data_blocks - 2)))

        return gate_cost

    @classmethod
    def _single_ctrl_select_cost(
        cls, num_data_blocks: int, num_bit_flips: int, repeat: int = 1
    ) -> list[GateCount]:
        r"""The decomposition of a controlled Select operation using unary iteration as
        described in Figure 7. of `Babbush et al. (2018) <https://arxiv.org/abs/1805.0366>`_.

        Args:
            num_data_blocks(int): The number of data blocks formed by partitioning the
                total bitstrings based on select-swap depth.
            num_bit_flips (int): the total number of :math:`1`'s in the dataset
            repeat (int): The number of times to repeat the subroutine.

        Returns:
            list[GateCount]: the resource decomposition of the Select subroutine
        """
        x = resource_rep(qre.X)
        cnot = resource_rep(qre.CNOT)
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})

        gate_cost = []
        if num_data_blocks == 1:
            gate_cost.append(
                GateCount(
                    cnot,
                    num_bit_flips * repeat,
                )  # each unitary in the select is just a CNOT gate to load the data
            )

        else:  # num_data_blocks > 1
            gate_cost.append(
                GateCount(x, repeat * (2 * (num_data_blocks - 1)))
            )  # conjugate 0-control in left-elbows
            gate_cost.append(
                GateCount(
                    cnot,
                    repeat * ((num_data_blocks - 1) + num_bit_flips),
                )  # num CNOTs in unary iterator trick + each unitary in the select is just a CNOT
            )
            gate_cost.append(GateCount(l_elbow, repeat * (num_data_blocks - 1)))
            gate_cost.append(GateCount(r_elbow, repeat * (num_data_blocks - 1)))

        return gate_cost

    @classmethod
    def _swap_cost(
        cls, register_size: int, num_swap_ctrls: int, repeat: int = 1
    ) -> list[GateCount]:
        r"""Constructs the control-S subroutine as defined in Figure 8 of
        `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`_ excluding the initial ``X`` gate.

        Args:
            register_size (int): The length of the bitstrings being encoded.
            num_swap_ctrls (int): The number of control wires to be used for the swapping subroutine.
                Should be equal to :math:`\log_{2}(\text{select_swap_depth})`.
            repeat (int): The number of times to repeat the subroutine.

        Returns:
            list[GateCount]: the resource decomposition of the control- :math:`S` subroutine
        """
        width = 2**num_swap_ctrls
        ctrl_swap = resource_rep(qre.CSWAP)
        return [GateCount(ctrl_swap, repeat * (width - 1) * register_size)]

    @classmethod
    def _swap_adj_cost(
        cls, register_size: int, num_swap_ctrls: int, repeat: int = 1
    ) -> list[GateCount]:
        r"""Constructs the control-S^adj subroutine as defined in Figure 8 to Figure 10
        of `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`_ excluding the terminal ``X`` gate.

        Args:
            register_size (int): The length of the bitstrings being encoded.
            num_swap_ctrls (int): The number of control wires to be used for the swapping subroutine.
                Should be equal to :math:`\log_{2}(\text{select_swap_depth})`.
            repeat (int): The number of times to repeat the subroutine.

        Returns:
            list[GateCount]: the resource decomposition of the control- :math:`S^{\dagger}` subroutine

        """
        h = qre.resource_rep(qre.Hadamard)
        cz = qre.resource_rep(qre.CZ)
        cnot = qre.resource_rep(qre.CNOT)

        width = 2**num_swap_ctrls - 1
        return [
            qre.GateCount(h, repeat * width * register_size),
            qre.GateCount(cz, repeat * width * register_size),
            qre.GateCount(cnot, repeat * width * register_size),
        ]

    @classmethod
    def _single_ctrl_swap_cost(
        cls, register_size: int, num_swap_ctrls: int, repeat: int = 1
    ) -> list[GateCount]:
        r"""This is a combination of the standard control-SWAP decomposition from Figure 1.b of
        `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_, with the observation that each
        set of swaps acts on one of the swap-control wires at a time. We can control this decomposition
        on a single qubit using a single auxiliary qubit and a pair of elbow gates for each swap-control
        wires. This is because we can recycle the same auxiliary qubit for each set of elbows.

        Args:
            register_size (int): The length of the bitstrings being encoded.
            num_swap_ctrls (int): The number of control wires to be used for the swapping subroutine.
                Should be equal to :math:`\log_{2}(\text{select_swap_depth})`.
            repeat (int): The number of times to repeat the subroutine.

        Returns:
            list[GateCount]: the resource decomposition of the control- :math:`S` subroutine
        """
        width = 2**num_swap_ctrls

        ctrl_swap = qre.CSWAP.resource_rep()
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})

        alloc_reg = Allocate(1, state="zero", restored=True)
        gate_cost = [
            alloc_reg,  # need one temporary qubit for l/r-elbow to control SWAP
            GateCount(l_elbow, repeat * num_swap_ctrls),
            GateCount(ctrl_swap, repeat * (width - 1) * register_size),
            GateCount(r_elbow, repeat * num_swap_ctrls),
            Deallocate(allocated_register=alloc_reg),  # release temporary qubit to control SWAP
        ]
        return gate_cost

    @classmethod
    def qrom_clean_auxiliary_adjoint_resource_decomp(
        cls, target_resource_params: dict
    ) -> list[GateCount]:
        """Returns a list representing the resources of the adjoint of the operator.

        Args:
            target_resource_params(dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            This is an alternate decomposition for the adjoint of QROM which uses a measurement and phase
            fixup algorithm. This decomposition requires one clean auxiliary qubit. The resources are
            based on Figure 7 in Appendix C of `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears in the decomposition.
        """
        d = target_resource_params["num_bitstrings"]
        M = target_resource_params["size_bitstring"]
        num_bit_flips = target_resource_params.get("num_bit_flips", None)

        # Set optimal Select-Swap depth
        k_approx = math.sqrt(
            d
        )  # minimizes Toffoli cost Appendix C. https://arxiv.org/abs/1902.02134
        k = 2 ** round(math.log2(k_approx))  # must be a power of 2

        gate_lst = []
        x = resource_rep(qre.X)
        had = qre.resource_rep(qre.Hadamard)

        # Measure + Reset output register in the X-basis:
        gate_lst.append(GateCount(had, M))

        if M < k:
            aux_reg = Allocate(
                k - M, state="zero", restored=True
            )  #  we can re-use the M output qubits
            gate_lst.append(aux_reg)

        new_address_size = int(math.ceil(d / k))
        swap_ctrl_register_size = int(math.log2(k))

        # Apply phase fixup (Fig. 6 Appendix C. https://arxiv.org/abs/1902.02134)
        gate_lst.append(GateCount(x))
        gate_lst.extend(cls._swap_cost(register_size=1, num_swap_ctrls=swap_ctrl_register_size))
        gate_lst.append(GateCount(had, k))
        gate_lst.extend(cls._select_cost(new_address_size, num_bit_flips))
        gate_lst.append(GateCount(had, k))
        gate_lst.extend(cls._swap_adj_cost(register_size=1, num_swap_ctrls=swap_ctrl_register_size))
        gate_lst.append(GateCount(x))

        if M < k:
            gate_lst.append(Deallocate(allocated_register=aux_reg))  # all qubits restored

        return gate_lst

    @classmethod
    def qrom_dirty_auxiliary_adjoint_resource_decomp(
        cls, target_resource_params: dict
    ) -> list[GateCount]:
        """Returns a list representing the resources of the adjoint of the operator.

        Args:
            target_resource_params(dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            This is an alternate decomposition for the adjoint of QROM which uses a measurement and phase
            fixup algorithm. This decomposition requires one borrowed auxiliary qubit. The resources are
            based on Figure 7 in Appendix C of `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears in the decomposition.
        """
        d = target_resource_params["num_bitstrings"]
        M = target_resource_params["size_bitstring"]
        num_bit_flips = target_resource_params.get("num_bit_flips", None)

        # Set optimal Select-Swap depth
        k_approx = math.sqrt(
            d / 2
        )  # minimizes Toffoli cost Appendix C. https://arxiv.org/abs/1902.02134
        k = 2 ** round(math.log2(k_approx))  # must be a power of 2

        gate_lst = []
        z = resource_rep(qre.Z)
        had = qre.resource_rep(qre.Hadamard)

        # Measure + Reset output register in the X-basis:
        gate_lst.append(GateCount(had, M))

        if M < k:
            aux_reg = Allocate(
                k - M, state="any", restored=True
            )  #  we can re-use the M output qubits
            gate_lst.append(aux_reg)

        new_address_size = int(math.ceil(d / k))
        swap_ctrl_register_size = int(math.log2(k))

        t = cls._select_cost(new_address_size, num_bit_flips, repeat=2)
        s = cls._swap_cost(register_size=1, num_swap_ctrls=swap_ctrl_register_size, repeat=4)

        # Apply phase fixup (Fig. 7 Appendix C. https://arxiv.org/abs/1902.02134)
        gate_lst.append(GateCount(z, 2))
        gate_lst.append(GateCount(had, 2))
        gate_lst.extend(s)
        gate_lst.extend(t)

        if M < k:
            gate_lst.append(Deallocate(allocated_register=aux_reg))  # all qubits restored

        return gate_lst

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources of the adjoint of the operator.

        Args:
            target_resource_params(dict): A dictionary containing the resource parameters of the target operator.

        Resources:

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears in the decomposition.
        """
        return cls.qrom_clean_auxiliary_adjoint_resource_decomp(target_resource_params)

    @staticmethod
    def tracking_name(
        num_bitstrings: int,
        size_bitstring: int,
        num_bit_flips: int | None = None,
        borrow_qubits: bool = True,
        select_swap_depth: int | None = None,
    ):
        return "QROM"
