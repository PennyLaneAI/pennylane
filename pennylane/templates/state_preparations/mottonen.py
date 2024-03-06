# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Contains the MottonenStatePreparation template.
"""
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires


# pylint: disable=len-as-condition,arguments-out-of-order,consider-using-enumerate
def gray_code(rank):
    """Generates the Gray code of given rank.

    Args:
        rank (int): rank of the Gray code (i.e. number of bits)
    """

    def gray_code_recurse(g, rank):
        k = len(g)
        if rank <= 0:
            return

        for i in range(k - 1, -1, -1):
            char = "1" + g[i]
            g.append(char)
        for i in range(k - 1, -1, -1):
            g[i] = "0" + g[i]

        gray_code_recurse(g, rank - 1)

    g = ["0", "1"]
    gray_code_recurse(g, rank - 1)

    return g


def _matrix_M_entry(row, col):
    """Returns one entry for the matrix that maps alpha to theta.

    See Eq. (3) in `Möttönen et al. (2004) <https://arxiv.org/abs/quant-ph/0407010>`_.

    Args:
        row (int): one-based row number
        col (int): one-based column number

    Returns:
        (float): transformation matrix entry at given row and column
    """
    # (col >> 1) ^ col is the Gray code of col
    b_and_g = row & ((col >> 1) ^ col)
    sum_of_ones = 0
    while b_and_g > 0:
        if b_and_g & 0b1:
            sum_of_ones += 1

        b_and_g = b_and_g >> 1

    return (-1) ** sum_of_ones


def compute_theta(alpha):
    """Maps the angles alpha of the multi-controlled rotations decomposition of a uniformly controlled rotation
     to the rotation angles used in the Gray code implementation.

    Args:
        alpha (tensor_like): alpha parameters

    Returns:
        (tensor_like): rotation angles theta
    """
    ln = alpha.shape[-1]
    k = np.log2(ln)

    M_trans = np.zeros(shape=(ln, ln))
    for i in range(len(M_trans)):
        for j in range(len(M_trans[0])):
            M_trans[i, j] = _matrix_M_entry(j, i)

    theta = qml.math.transpose(qml.math.dot(M_trans, qml.math.transpose(alpha)))

    return theta / 2**k


def _apply_uniform_rotation_dagger(gate, alpha, control_wires, target_wire):
    r"""Applies a uniformly-controlled rotation to the target qubit.

    A uniformly-controlled rotation is a sequence of multi-controlled
    rotations, each of which is conditioned on the control qubits being in a different state.
    For example, a uniformly-controlled rotation with two control qubits describes a sequence of
    four multi-controlled rotations, each applying the rotation only if the control qubits
    are in states :math:`|00\rangle`, :math:`|01\rangle`, :math:`|10\rangle`, and :math:`|11\rangle`, respectively.

    To implement a uniformly-controlled rotation using single qubit rotations and CNOT gates,
    a decomposition based on Gray codes is used. For this purpose, the multi-controlled rotation
    angles alpha have to be converted into a set of non-controlled rotation angles theta.

    For more details, see `Möttönen and Vartiainen (2005), Fig 7a<https://arxiv.org/abs/quant-ph/0504100>`_.

    Args:
        gate (.Operation): gate to be applied, needs to have exactly one parameter
        alpha (tensor_like): angles to decompose the uniformly-controlled rotation into multi-controlled rotations
        control_wires (array[int]): wires that act as control
        target_wire (int): wire that acts as target

    Returns:
          list[.Operator]: sequence of operators defined by this function
    """
    op_list = []
    theta = compute_theta(alpha)

    gray_code_rank = len(control_wires)

    if gray_code_rank == 0:
        if qml.math.is_abstract(theta) or qml.math.all(theta[..., 0] != 0.0):
            op_list.append(gate(theta[..., 0], wires=[target_wire]))
        return op_list

    code = gray_code(gray_code_rank)
    num_selections = len(code)

    control_indices = [
        int(np.log2(int(code[i], 2) ^ int(code[(i + 1) % num_selections], 2)))
        for i in range(num_selections)
    ]

    for i, control_index in enumerate(control_indices):
        if qml.math.is_abstract(theta) or qml.math.all(theta[..., i] != 0.0):
            op_list.append(gate(theta[..., i], wires=[target_wire]))
        op_list.append(qml.CNOT(wires=[control_wires[control_index], target_wire]))
    return op_list


def _get_alpha_z(omega, n, k):
    r"""Computes the rotation angles required to implement the uniformly-controlled Z rotation
    applied to the :math:`k`th qubit.

    The :math:`j`th angle is related to the phases omega of the desired amplitudes via:

    .. math:: \alpha^{z,k}_j = \sum_{l=1}^{2^{k-1}} \frac{\omega_{(2j-1) 2^{k-1}+l} - \omega_{(2j-2) 2^{k-1}+l}}{2^{k-1}}

    Args:
        omega (tensor_like): phases of the state to prepare
        n (int): total number of qubits for the uniformly-controlled rotation
        k (int): index of current qubit

    Returns:
        array representing :math:`\alpha^{z,k}`
    """
    indices1 = [
        [(2 * j - 1) * 2 ** (k - 1) + l - 1 for l in range(1, 2 ** (k - 1) + 1)]
        for j in range(1, 2 ** (n - k) + 1)
    ]
    indices2 = [
        [(2 * j - 2) * 2 ** (k - 1) + l - 1 for l in range(1, 2 ** (k - 1) + 1)]
        for j in range(1, 2 ** (n - k) + 1)
    ]

    term1 = qml.math.take(omega, indices=indices1, axis=-1)
    term2 = qml.math.take(omega, indices=indices2, axis=-1)
    diff = (term1 - term2) / 2 ** (k - 1)

    return qml.math.sum(diff, axis=-1)


def _get_alpha_y(a, n, k):
    r"""Computes the rotation angles required to implement the uniformly controlled Y rotation
    applied to the :math:`k`th qubit.

    The :math:`j`-th angle is related to the absolute values, a, of the desired amplitudes via:

    .. math:: \alpha^{y,k}_j = 2 \arcsin \sqrt{ \frac{ \sum_{l=1}^{2^{k-1}} a_{(2j-1)2^{k-1} +l}^2  }{ \sum_{l=1}^{2^{k}} a_{(j-1)2^{k} +l}^2  } }

    Args:
        a (tensor_like): absolute values of the state to prepare
        n (int): total number of qubits for the uniformly-controlled rotation
        k (int): index of current qubit

    Returns:
        array representing :math:`\alpha^{y,k}`
    """
    indices_numerator = [
        [(2 * (j + 1) - 1) * 2 ** (k - 1) + l for l in range(2 ** (k - 1))]
        for j in range(2 ** (n - k))
    ]
    numerator = qml.math.take(a, indices=indices_numerator, axis=-1)
    numerator = qml.math.sum(qml.math.abs(numerator) ** 2, axis=-1)

    indices_denominator = [[j * 2**k + l for l in range(2**k)] for j in range(2 ** (n - k))]
    denominator = qml.math.take(a, indices=indices_denominator, axis=-1)
    denominator = qml.math.sum(qml.math.abs(denominator) ** 2, axis=-1)

    # Divide only where denominator is zero, else leave initial value of zero.
    # The equation guarantees that the numerator is also zero in the corresponding entries.

    with np.errstate(divide="ignore", invalid="ignore"):
        division = numerator / denominator

    # Cast the numerator and denominator to ensure compatibility with interfaces
    division = qml.math.cast(division, np.float64)
    denominator = qml.math.cast(denominator, np.float64)

    division = qml.math.where(denominator != 0.0, division, 0.0)

    return 2 * qml.math.arcsin(qml.math.sqrt(division))


class MottonenStatePreparation(Operation):
    r"""
    Prepares an arbitrary state on the given wires using a decomposition into gates developed
    by `Möttönen et al. (2004) <https://arxiv.org/abs/quant-ph/0407010>`_.

    The state is prepared via a sequence
    of uniformly controlled rotations. A uniformly controlled rotation on a target qubit is
    composed from all possible controlled rotations on the qubit and can be used to address individual
    elements of the state vector.

    In the work of Möttönen et al., inverse state preparation
    is executed by first equalizing the phases of the state vector via uniformly controlled Z rotations,
    and then rotating the now real state vector into the direction of the state :math:`|0\rangle` via
    uniformly controlled Y rotations.

    This code is adapted from code written by Carsten Blank for PennyLane-Qiskit.

    .. warning::

        Due to non-trivial classical processing of the state vector,
        this template is not always fully differentiable.

    Args:
        state_vector (tensor_like): Input array of shape ``(2^n,)``, where ``n`` is the number of wires
            the state preparation acts on. The input array must be normalized.
        wires (Iterable): wires that the template acts on

    Example:

        ``MottonenStatePreparation`` creates any arbitrary state on the given wires depending on the input state vector.

        .. code-block:: python

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit(state):
                qml.MottonenStatePreparation(state_vector=state, wires=range(3))
                return qml.state()

            state = np.array([1, 2j, 3, 4j, 5, 6j, 7, 8j])
            state = state / np.linalg.norm(state)

            print(qml.draw(circuit, expansion_strategy="device", max_length=80)(state))

        .. code-block::

            0: ──RY(2.35)─╭●───────────╭●──────────────╭●────────────────────────╭●
            1: ──RY(2.09)─╰X──RY(0.21)─╰X─╭●───────────│────────────╭●───────────│─
            2: ──RY(1.88)─────────────────╰X──RY(0.10)─╰X──RY(0.08)─╰X──RY(0.15)─╰X

            ──╭●────────╭●────╭●────╭●─╭GlobalPhase(-0.79)─┤ ╭State
            ──╰X────────╰X─╭●─│──╭●─│──├GlobalPhase(-0.79)─┤ ├State
            ───RZ(1.57)────╰X─╰X─╰X─╰X─╰GlobalPhase(-0.79)─┤ ╰State

        The state preparation can be checked by running:

        >>> print(np.allclose(state, circuit(state)))
        True

    """

    num_wires = AnyWires
    grad_method = None
    ndim_params = (1,)

    def __init__(self, state_vector, wires, id=None):
        # check if the `state_vector` param is batched
        batched = len(qml.math.shape(state_vector)) > 1

        state_batch = state_vector if batched else [state_vector]

        # apply checks to each state vector in the batch
        for i, state in enumerate(state_batch):
            shape = qml.math.shape(state)

            if len(shape) != 1:
                raise ValueError(
                    f"State vectors must be one-dimensional; vector {i} has shape {shape}."
                )

            n_amplitudes = shape[0]
            if n_amplitudes != 2 ** len(qml.wires.Wires(wires)):
                raise ValueError(
                    f"State vectors must be of length {2 ** len(wires)} or less; vector {i} has length {n_amplitudes}."
                )

            if not qml.math.is_abstract(state):
                norm = qml.math.sum(qml.math.abs(state) ** 2)
                if not qml.math.allclose(norm, 1.0, atol=1e-3):
                    raise ValueError(
                        f"State vectors have to be of norm 1.0, vector {i} has norm {norm}"
                    )

        super().__init__(state_vector, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(state_vector, wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.MottonenStatePreparation.decomposition`.

        Args:
            state_vector (tensor_like): Normalized state vector of shape ``(2^len(wires),)``
            wires (Any or Iterable[Any]): wires that the operator acts on

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> state_vector = torch.tensor([0.5, 0.5, 0.5, 0.5])
        >>> qml.MottonenStatePreparation.compute_decomposition(state_vector, wires=["a", "b"])
        [RY(array(1.57079633), wires=['a']),
        RY(array(1.57079633), wires=['b']),
        CNOT(wires=['a', 'b']),
        CNOT(wires=['a', 'b'])]
        """
        if len(qml.math.shape(state_vector)) > 1:
            raise ValueError(
                "Broadcasting with MottonenStatePreparation is not supported. Please use the "
                "qml.transforms.broadcast_expand transform to use broadcasting with "
                "MottonenStatePreparation."
            )

        a = qml.math.abs(state_vector)
        omega = qml.math.angle(state_vector)
        # change ordering of wires, since original code
        # was written for IBM machines
        wires_reverse = wires[::-1]

        op_list = []

        # Apply inverse y rotation cascade to prepare correct absolute values of amplitudes
        for k in range(len(wires_reverse), 0, -1):
            alpha_y_k = _get_alpha_y(a, len(wires_reverse), k)
            control = wires_reverse[k:]
            target = wires_reverse[k - 1]
            op_list.extend(_apply_uniform_rotation_dagger(qml.RY, alpha_y_k, control, target))

        # If necessary, apply inverse z rotation cascade to prepare correct phases of amplitudes
        if qml.math.is_abstract(omega) or not qml.math.allclose(omega, 0):
            for k in range(len(wires_reverse), 0, -1):
                alpha_z_k = _get_alpha_z(omega, len(wires_reverse), k)
                control = wires_reverse[k:]
                target = wires_reverse[k - 1]
                if len(alpha_z_k) > 0:
                    op_list.extend(
                        _apply_uniform_rotation_dagger(qml.RZ, alpha_z_k, control, target)
                    )

            global_phase = qml.math.sum(-1 * qml.math.angle(state_vector) / len(state_vector))
            op_list.extend([qml.GlobalPhase(global_phase, wires=wires)])

        return op_list
