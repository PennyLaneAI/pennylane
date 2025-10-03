# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This submodule contains the discrete-variable quantum operations that are the
core parametrized gates.
"""
# pylint: disable=arguments-differ
import functools
from collections import Counter
from operator import matmul

import numpy as np

import pennylane as qml
from pennylane import math, queuing
from pennylane.decomposition import add_decomps, controlled_resource_rep, register_resources
from pennylane.decomposition.symbolic_decomposition import adjoint_rotation, pow_rotation
from pennylane.math.decomposition import decomp_int_to_powers_of_two
from pennylane.operation import FlatPytree, Operation, Operator
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

from .non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ
from .parametric_ops_single_qubit import RX, RY, RZ, PhaseShift, _can_replace, stack_last


class MultiRZ(Operation):
    r"""
    Arbitrary multi Z rotation.

    .. math::

        MultiRZ(\theta) = \exp\left(-i \frac{\theta}{2} Z^{\otimes n}\right)

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\theta}f(MultiRZ(\theta)) = \frac{1}{2}\left[f(MultiRZ(\theta +\pi/2)) - f(MultiRZ(\theta-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`MultiRZ(\theta)`.

    .. note::

        If the ``MultiRZ`` gate is not supported on the targeted device, PennyLane
        will decompose the gate using :class:`~.RZ` and :class:`~.CNOT` gates.

    Args:
        theta (TensorLike): rotation angle :math:`\theta`
        wires (Sequence[int] or int): the wires the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = {"num_wires"}

    grad_method = "A"
    parameter_frequencies = [(1,)]

    def _flatten(self) -> FlatPytree:
        return self.data, (self.wires, tuple())

    def __init__(self, theta: TensorLike, wires: WiresLike, id: str | None = None):
        wires = Wires(wires)
        self.hyperparameters["num_wires"] = len(wires)
        super().__init__(theta, wires=wires, id=id)
        if not self._wires:
            raise ValueError(
                f"{self.name}: wrong number of wires. At least one wire has to be provided."
            )

    @staticmethod
    def compute_matrix(theta: TensorLike, num_wires: int) -> TensorLike:
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.MultiRZ.matrix`

        Args:
            theta (TensorLike): rotation angle
            num_wires (int): number of wires the rotation acts on

        Returns:
            TensorLike: canonical matrix

        **Example**

        >>> qml.MultiRZ.compute_matrix(torch.tensor(0.1), 2)
        tensor([[0.9988-0.0500j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.9988+0.0500j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.9988+0.0500j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9988-0.0500j]],
               dtype=torch.complex128)
        """
        eigs = math.convert_like(qml.pauli.pauli_eigs(num_wires), theta)

        if (
            math.get_interface(theta) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            theta = math.cast_like(theta, 1j)
            eigs = math.cast_like(eigs, 1j)

        if math.ndim(theta) == 0:
            return math.diag(math.exp(-0.5j * theta * eigs))

        diags = math.exp(math.outer(-0.5j * theta, eigs))
        return diags[:, :, np.newaxis] * math.cast_like(math.eye(2**num_wires, like=diags), diags)

    def generator(self) -> "qml.Hamiltonian":
        return qml.Hamiltonian([-0.5], [functools.reduce(matmul, [PauliZ(w) for w in self.wires])])

    @staticmethod
    def compute_eigvals(theta: TensorLike, num_wires: int) -> TensorLike:
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.MultiRZ.eigvals`


        Args:
            theta (TensorLike): rotation angle
            num_wires (int): number of wires the rotation acts on

        Returns:
            TensorLike: eigenvalues

        **Example**

        >>> qml.MultiRZ.compute_eigvals(torch.tensor(0.5), 3)
        tensor([0.9689-0.2474j, 0.9689+0.2474j, 0.9689+0.2474j, 0.9689-0.2474j,
                0.9689+0.2474j, 0.9689-0.2474j, 0.9689-0.2474j, 0.9689+0.2474j],
               dtype=torch.complex128)
        """
        eigs = math.convert_like(qml.pauli.pauli_eigs(num_wires), theta)

        if (
            math.get_interface(theta) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            theta = math.cast_like(theta, 1j)
            eigs = math.cast_like(eigs, 1j)

        if math.ndim(theta) == 0:
            return math.exp(-0.5j * theta * eigs)

        return math.exp(math.outer(-0.5j * theta, eigs))

    @staticmethod
    def compute_decomposition(  # pylint: disable=arguments-differ,unused-argument
        theta: TensorLike, wires: WiresLike, **kwargs
    ) -> list[Operator]:
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.MultiRZ.decomposition`.

        Args:
            theta (TensorLike): rotation angle :math:`\theta`
            wires (Iterable, Wires): the wires the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.MultiRZ.compute_decomposition(1.2, wires=(0,1))
        [CNOT(wires=[1, 0]), RZ(1.2, wires=[0]), CNOT(wires=[1, 0])]

        """
        ops = [qml.CNOT(wires=(w0, w1)) for w0, w1 in zip(wires[~0:0:-1], wires[~1::-1])]
        ops.append(RZ(theta, wires=wires[0]))
        ops += [qml.CNOT(wires=(w0, w1)) for w0, w1 in zip(wires[1:], wires[:~0])]

        return ops

    @property
    def resource_params(self) -> dict:
        return {"num_wires": self.hyperparameters["num_wires"]}

    def adjoint(self) -> "MultiRZ":
        return MultiRZ(-self.parameters[0], wires=self.wires)

    def pow(self, z: int | float) -> list[Operator]:
        return [MultiRZ(self.data[0] * z, wires=self.wires)]

    def simplify(self) -> "MultiRZ":
        theta = self.data[0] % (4 * np.pi)

        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires[0])

        return MultiRZ(theta, wires=self.wires)


def _multi_rz_decomposition_resources(num_wires):
    return {qml.RZ: 1, qml.CNOT: 2 * (num_wires - 1)}


@register_resources(_multi_rz_decomposition_resources)
def _multi_rz_decomposition(theta: TensorLike, wires: WiresLike, **__):

    @qml.for_loop(len(wires) - 1, 0, -1)
    def _pre_cnot(i):
        qml.CNOT(wires=(wires[i], wires[i - 1]))

    @qml.for_loop(1, len(wires), 1)
    def _post_cnot(i):
        qml.CNOT(wires=(wires[i], wires[i - 1]))

    _pre_cnot()  # pylint: disable=no-value-for-parameter
    qml.RZ(theta, wires=wires[0])
    _post_cnot()  # pylint: disable=no-value-for-parameter


add_decomps(MultiRZ, _multi_rz_decomposition)
add_decomps("Adjoint(MultiRZ)", adjoint_rotation)
add_decomps("Pow(MultiRZ)", pow_rotation)


class PauliRot(Operation):
    r"""
    Arbitrary Pauli word rotation.

    .. math::

        RP(\theta, P) = \exp\left(-i \frac{\theta}{2} P\right)

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\theta}f(RP(\theta)) = \frac{1}{2}\left[f(RP(\theta +\pi/2)) - f(RP(\theta-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`RP(\theta)`.

    .. note::

        If the ``PauliRot`` gate is not supported on the targeted device, PennyLane
        will decompose the gate using :class:`~.RX`, :class:`~.Hadamard`, :class:`~.RZ`
        and :class:`~.CNOT` gates.

    Args:
        theta (float): rotation angle :math:`\theta`
        pauli_word (string): the Pauli word defining the rotation
        wires (Sequence[int] or int): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    **Example**

    >>> dev = qml.device('default.qubit', wires=1)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.PauliRot(0.5, 'X',  wires=0)
    ...     return qml.expval(qml.Z(0))
    >>> print(example_circuit())
    0.8775825618903724
    """

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    do_check_domain = False
    grad_method = "A"
    parameter_frequencies = [(1,)]

    resource_keys = {
        "pauli_word",
    }

    _ALLOWED_CHARACTERS = "IXYZ"

    _PAULI_CONJUGATION_MATRICES = {
        "X": Hadamard.compute_matrix(),
        "Y": RX.compute_matrix(np.pi / 2),
        "Z": np.array([[1, 0], [0, 1]]),
    }

    @classmethod
    def _primitive_bind_call(cls, theta, pauli_word, wires=None, id=None):
        return super()._primitive_bind_call(theta, pauli_word=pauli_word, wires=wires, id=id)

    def __init__(
        self,
        theta: TensorLike,
        pauli_word: str,
        wires: WiresLike,
        id: str | None = None,
    ):
        super().__init__(theta, wires=wires, id=id)

        if not self._wires:
            raise ValueError(
                f"{self.name}: wrong number of wires. At least one wire has to be provided."
            )

        self.hyperparameters["pauli_word"] = pauli_word
        if not PauliRot._check_pauli_word(pauli_word):
            raise ValueError(
                f'The given Pauli word "{pauli_word}" contains characters that are not allowed. '
                "Allowed characters are I, X, Y and Z"
            )

        num_wires = 1 if isinstance(wires, int) else len(wires)

        if not len(pauli_word) == num_wires:
            raise ValueError(
                f"The number of wires must be equal to the length of the Pauli word. "
                f"The Pauli word {pauli_word} has length {len(pauli_word)}, and "
                f"{num_wires} wires were given {wires}."
            )

    def __repr__(self) -> str:
        return f"PauliRot({self.data[0]}, {self.hyperparameters['pauli_word']}, wires={self.wires.tolist()})"

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        r"""A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label
            cache=None (dict): dictionary that caries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings

        **Example:**

        >>> op = qml.PauliRot(0.1, "XYY", wires=(0,1,2))
        >>> op.label()
        'RXYY'
        >>> op.label(decimals=2)
        'RXYY\n(0.10)'
        >>> op.label(base_label="PauliRot")
        'PauliRot'

        """
        pauli_word = self.hyperparameters["pauli_word"]
        op_label = base_label or ("R" + pauli_word)

        # TODO[dwierichs]: Implement a proper label for parameter-broadcasted operators
        if decimals is not None and self.batch_size is None:
            param_string = f"\n({math.asarray(self.parameters[0]):.{decimals}f})"
            op_label += param_string

        return op_label

    @property
    def resource_params(self) -> dict:
        return {"pauli_word": self.hyperparameters["pauli_word"]}

    @staticmethod
    def _check_pauli_word(pauli_word) -> bool:
        """Check that the given Pauli word has correct structure.

        Args:
            pauli_word (str): Pauli word to be checked

        Returns:
            bool: Whether the Pauli word has correct structure.
        """
        return all(pauli in PauliRot._ALLOWED_CHARACTERS for pauli in set(pauli_word))

    @staticmethod
    def compute_matrix(theta: TensorLike, pauli_word: str) -> TensorLike:
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.PauliRot.matrix`


        Args:
            theta (TensorLike): rotation angle
            pauli_word (str): string representation of Pauli word

        Returns:
            TensorLike: canonical matrix

        **Example**

        >>> qml.PauliRot.compute_matrix(0.5, 'X')
        array([[0.96891242+0.j        , 0.        -0.24740396j],
               [0.        -0.24740396j, 0.96891242+0.j        ]])
        """
        if not PauliRot._check_pauli_word(pauli_word):
            raise ValueError(
                f'The given Pauli word "{pauli_word}" contains characters that are not allowed. '
                "Allowed characters are I, X, Y and Z"
            )

        interface = math.get_interface(theta)

        if (
            interface == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            theta = math.cast_like(theta, 1j)

        # Simplest case is if the Pauli is the identity matrix
        if set(pauli_word) == {"I"}:
            return qml.GlobalPhase.compute_matrix(0.5 * theta, n_wires=len(pauli_word))

        # We first generate the matrix excluding the identity parts and expand it afterwards.
        # To this end, we have to store on which wires the non-identity parts act
        non_identity_wires, non_identity_gates = zip(
            *[(wire, gate) for wire, gate in enumerate(pauli_word) if gate != "I"]
        )

        multi_Z_rot_matrix = MultiRZ.compute_matrix(theta, len(non_identity_gates))

        # now we conjugate with Hadamard and RX to create the Pauli string
        conjugation_matrix = functools.reduce(
            math.kron,
            [PauliRot._PAULI_CONJUGATION_MATRICES[gate] for gate in non_identity_gates],
        )
        if (
            interface == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            conjugation_matrix = math.cast_like(conjugation_matrix, 1j)
        # Note: we use einsum with reverse arguments here because it is not multi-dispatched
        # and the tensordot containing multi_Z_rot_matrix should decide about the interface
        return math.expand_matrix(
            math.einsum(
                "...jk,ij->...ik",
                math.tensordot(multi_Z_rot_matrix, conjugation_matrix, axes=[[-1], [0]]),
                math.conj(conjugation_matrix),
            ),
            non_identity_wires,
            list(range(len(pauli_word))),
        )

    def generator(self) -> "qml.Hamiltonian":
        pauli_word = self.hyperparameters["pauli_word"]
        wire_map = {w: i for i, w in enumerate(self.wires)}

        return qml.Hamiltonian(
            [-0.5], [qml.pauli.string_to_pauli_word(pauli_word, wire_map=wire_map)]
        )

    @staticmethod
    def compute_eigvals(theta: TensorLike, pauli_word: str) -> TensorLike:
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.PauliRot.eigvals`


        Returns:
            TensorLike: eigenvalues

        **Example**

        >>> qml.PauliRot.compute_eigvals(torch.tensor(0.5), "X")
        tensor([0.9689-0.2474j, 0.9689+0.2474j], dtype=torch.complex128)
        """
        if (
            math.get_interface(theta) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            theta = math.cast_like(theta, 1j)

        # Identity must be treated specially because its eigenvalues are all the same
        if set(pauli_word) == {"I"}:
            return qml.GlobalPhase.compute_eigvals(0.5 * theta, n_wires=len(pauli_word))

        return MultiRZ.compute_eigvals(theta, len(pauli_word))

    @staticmethod
    def compute_decomposition(
        theta: TensorLike, wires: WiresLike, pauli_word: str
    ) -> list[Operator]:
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.PauliRot.decomposition`.

        Args:
            theta (TensorLike): rotation angle :math:`\theta`
            wires (Iterable, Wires): the wires the operation acts on
            pauli_word (string): the Pauli word defining the rotation

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.PauliRot.compute_decomposition(1.2, wires=(0,1), pauli_word="XY")
        [H(0), RX(1.5707963267948966, wires=[1]), MultiRZ(1.2, wires=[0, 1]), H(0), RX(-1.5707963267948966, wires=[1])]

        """
        if isinstance(wires, int):  # Catch cases when the wire is passed as a single int.
            wires = [wires]

        # Check for identity and do nothing
        if set(pauli_word) == {"I"}:
            return [qml.GlobalPhase(phi=theta / 2)]

        active_wires, active_gates = zip(
            *[(wire, gate) for wire, gate in zip(wires, pauli_word) if gate != "I"]
        )

        ops = []
        for wire, gate in zip(active_wires, active_gates):
            if gate == "X":
                ops.append(Hadamard(wires=[wire]))
            elif gate == "Y":
                ops.append(RX(np.pi / 2, wires=[wire]))

        ops.append(MultiRZ(theta, wires=list(active_wires)))

        for wire, gate in zip(active_wires, active_gates):
            if gate == "X":
                ops.append(Hadamard(wires=[wire]))
            elif gate == "Y":
                ops.append(RX(-np.pi / 2, wires=[wire]))
        return ops

    def adjoint(self):
        return PauliRot(-self.parameters[0], self.hyperparameters["pauli_word"], wires=self.wires)

    def pow(self, z):
        return [PauliRot(self.data[0] * z, self.hyperparameters["pauli_word"], wires=self.wires)]


def _pauli_rot_resources(pauli_word):
    if set(pauli_word) == {"I"}:
        return {qml.GlobalPhase: 1}
    num_active_wires = len(pauli_word.replace("I", ""))
    return {
        qml.Hadamard: 2 * pauli_word.count("X"),
        qml.RX: 2 * pauli_word.count("Y"),
        qml.resource_rep(qml.MultiRZ, num_wires=num_active_wires): 1,
    }


@register_resources(_pauli_rot_resources)
def _pauli_rot_decomposition(theta, pauli_word, wires, **__):
    if set(pauli_word) == {"I"}:
        qml.GlobalPhase(theta / 2)
        return
    active_wires, active_gates = zip(
        *[(wire, gate) for wire, gate in zip(wires, pauli_word) if gate != "I"]
    )
    for wire, gate in zip(active_wires, active_gates):
        if gate == "X":
            qml.Hadamard(wires=[wire])
        elif gate == "Y":
            qml.RX(np.pi / 2, wires=[wire])
    qml.MultiRZ(theta, wires=list(active_wires))
    for wire, gate in zip(active_wires, active_gates):
        if gate == "X":
            qml.Hadamard(wires=[wire])
        elif gate == "Y":
            qml.RX(-np.pi / 2, wires=[wire])


add_decomps(PauliRot, _pauli_rot_decomposition)
add_decomps("Adjoint(PauliRot)", adjoint_rotation)
add_decomps("Pow(PauliRot)", pow_rotation)


class PCPhase(Operation):
    r"""PCPhase(phi, dim, wires)
    A projector-controlled phase gate.

    This gate applies a complex phase :math:`e^{i\phi}` to the first :math:`dim`
    basis vectors of the input state while applying a complex phase :math:`e^{-i \phi}`
    to the remaining basis vectors. For example, consider the 2-qubit case where ``dim = 3``:

    .. math:: \Pi(\phi) = \begin{bmatrix}
                e^{i\phi} & 0 & 0 & 0 \\
                0 & e^{i\phi} & 0 & 0 \\
                0 & 0 & e^{i\phi} & 0 \\
                0 & 0 & 0 & e^{-i\phi}
            \end{bmatrix}.

    This can also be written as :math:`\Pi(\phi) = \exp(i\phi(2\Pi-\mathbb{I}_N))`, where
    :math:`N=2^n` is the Hilbert space dimension for :math:`n` qubits and :math:`\Pi` is
    the diagonal projector with ``dim`` ones and ``N-dim`` zeros.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)

    Args:
        phi (float): rotation angle :math:`\phi`
        dim (int): the dimension of the subspace
        wires (Iterable[int, str], Wires): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    **Example:**

    We can define a circuit using :class:`~.PCPhase` as follows:

    >>> op_3 = qml.PCPhase(0.27, dim = 3, wires=range(3))

    The resulting operation applies a complex phase :math:`e^{0.27i}` to the first :math:`dim = 3`
    basis vectors and :math:`e^{-0.27i}` to the remaining basis vectors, as we can see from
    the diagonal of the matrix for this circuit.

    >>> print(np.round(np.diag(qml.matrix(op_3)),2))
    [0.96+0.27j 0.96+0.27j 0.96+0.27j 0.96-0.27j 0.96-0.27j 0.96-0.27j
     0.96-0.27j 0.96-0.27j]

    We can also choose a different ``dim`` value to apply the phase shift to a different set of
    basis vectors as follows:

    >>> op_7 = qml.PCPhase(1.23, dim=7, wires=[1, 2, 3])
    >>> print(np.round(np.diag(qml.matrix(op_7)),2))
    [0.33+0.94j 0.33+0.94j 0.33+0.94j 0.33+0.94j 0.33+0.94j 0.33+0.94j
     0.33+0.94j 0.33-0.94j]

    ``PCPhase`` operations are decomposed into (multi-)controlled :class:`~.PhaseShift`
    operations which share the same control values on common control wires, and Pauli-X operations,
    possibly complemented by a global phase.

    >>> op_13 = qml.PCPhase(1.23, dim=13, wires=[1, 2, 3, 4])
    >>> print(qml.draw(op_13.decomposition)())
    1: ─╭●─────────╭●───────────╭GlobalPhase(-1.23)─┤  
    2: ─╰Rϕ(-2.46)─├●───────────├GlobalPhase(-1.23)─┤  
    3: ────────────├○───────────├GlobalPhase(-1.23)─┤  
    4: ──X─────────╰Rϕ(2.46)──X─╰GlobalPhase(-1.23)─┤

    If ``dim`` is a power of two, a single (multi-controlled) ``PhaseShift`` gate is sufficient:

    >>> op_16 = qml.PCPhase(1.23, dim=16, wires=range(6))
    >>> print(qml.draw(op_16.decomposition, wire_order=range(6), show_all_wires=True)())
    0: ────╭○───────────╭GlobalPhase(1.23)─┤  
    1: ──X─╰Rϕ(2.46)──X─├GlobalPhase(1.23)─┤  
    2: ─────────────────├GlobalPhase(1.23)─┤  
    3: ─────────────────├GlobalPhase(1.23)─┤  
    4: ─────────────────├GlobalPhase(1.23)─┤  
    5: ─────────────────╰GlobalPhase(1.23)─┤

    """

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""
    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "Z"
    grad_method = "A"
    parameter_frequencies = [(2,)]

    resource_keys = {"num_wires", "dim"}

    def generator(self) -> "qml.Hermitian":
        r"""Generator of the ``PCPhase`` operator, which is in single-parameter-form.
        The operator reads

        .. math:: \Pi(\phi) = e^{i\phi (2\Pi - \mathbb{I}_N)},

        where :math:`\Pi` is the projector onto the first :math`d` (``dim``) computational basis
        states and :math:`N=2^n` is the Hilbert space dimension for :math:`n` qubits.

        Correspondingly, the generator is
        :math:`2\Pi - \mathbb{I}_N=\text{diag}(\underset{d\text{ times}}{\underbrace{1, \dots, 1}},\underset{(N-d)\text{ times}}{\underbrace{-1, \dots, -1}})`:

        >>> qml.PCPhase(0.5, dim=3, wires=[0, 1]).generator()
        Hermitian(array([[ 1,  0,  0,  0],
           [ 0,  1,  0,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0, -1]]), wires=[0, 1])
        """
        dim, N = self.hyperparameters["dimension"]
        mat = np.diag([1] * dim + [-1] * (N - dim))
        return qml.Hermitian(mat, wires=self.wires)

    def _flatten(self) -> FlatPytree:
        hyperparameter = (("dim", self.hyperparameters["dimension"][0]),)
        return tuple(self.data), (self.wires, hyperparameter)

    def __init__(self, phi: TensorLike, dim: int, wires: WiresLike, id: str | None = None):
        wires = wires if isinstance(wires, Wires) else Wires(wires)

        if not (isinstance(dim, int) and (dim <= 2 ** len(wires))):
            raise ValueError(
                f"The projected dimension {dim} must be an integer that is less than or equal to "
                f"the max size of the matrix {2 ** len(wires)}. Try adding more wires."
            )

        super().__init__(phi, wires=wires, id=id)
        self.hyperparameters["dimension"] = (dim, 2 ** len(wires))

    @property
    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires), "dim": self.hyperparameters["dimension"][0]}

    @staticmethod
    def compute_matrix(phi: TensorLike, dimension: tuple[int, int]) -> TensorLike:
        """Get the matrix representation of Pi-controlled phase unitary."""
        d, t = dimension

        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            p = math.exp(1j * math.cast_like(phi, 1j))
            minus_p = math.exp(-1j * math.cast_like(phi, 1j))
            zeros = math.zeros_like(p)

            columns = []
            for i in range(t):
                columns.append(
                    [p if j == i else zeros for j in range(t)]
                    if i < d
                    else [minus_p if j == i else zeros for j in range(t)]
                )
            r = math.stack(columns, like="tensorflow", axis=-2)
            return r

        arg = 1j * phi
        prefactors = math.array([1] * d + [-1] * (t - d), like=phi)

        if math.ndim(arg) == 0:
            return math.diag(math.exp(arg * prefactors))

        diags = math.exp(math.outer(arg, prefactors))
        return math.stack([math.diag(d) for d in diags])

    @staticmethod
    def compute_eigvals(*params: TensorLike, **hyperparams) -> TensorLike:
        """Get the eigvals for the Pi-controlled phase unitary."""
        phi = params[0]
        d, t = hyperparams["dimension"]

        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phase = math.exp(1j * math.cast_like(phi, 1j))
            minus_phase = math.exp(-1j * math.cast_like(phi, 1j))
            return stack_last([phase if index < d else minus_phase for index in range(t)])

        arg = 1j * phi
        prefactors = math.array([1] * d + [-1] * (t - d), like=phi)

        if math.ndim(phi) == 0:
            product = arg * prefactors
        else:
            product = math.outer(arg, prefactors)
        return math.exp(product)

    @staticmethod
    def compute_decomposition(
        *params: TensorLike, wires: WiresLike, **hyperparams
    ) -> list[Operator]:
        r"""Representation of the PCPhase operator as a product of other operators (static method).

        Args:
            *params (list): trainable parameters of the operator, as stored in the
                ``parameters`` attribute
            wires (Iterable[Any], Wires): wires that the operator acts on
            **hyperparams (dict): non-trainable hyper-parameters of the operator,
                as stored in the ``hyperparameters`` attribute

        Returns:
            list[Operator]: decomposition of the operator

        In short, this decomposition relies on decomposing the generator (see :meth:`~.generator`)
        of the ``PCPhase`` gate into generators of multicontrolled :class:`~.PhaseShift` gates,
        potentially complemented with (non-controlled) Pauli-X gates and/or a global phase.
        For example, for ``dim=13`` on four qubits:

        >>> op_13 = qml.PCPhase(1.23, dim=13, wires=[1, 2, 3, 4])
        >>> print(qml.draw(op_13.decomposition)())
        1: ─╭●─────────╭●───────────╭GlobalPhase(-1.23)─┤  
        2: ─╰Rϕ(-2.46)─├●───────────├GlobalPhase(-1.23)─┤  
        3: ────────────├○───────────├GlobalPhase(-1.23)─┤  
        4: ──X─────────╰Rϕ(2.46)──X─╰GlobalPhase(-1.23)─┤

        In the following we provide a detailed example for illustration purposes.

        **Detailed example**

        Consider the projector-controlled phase gate on :math:`n=4` qubits and with
        :math:`d=\texttt{dim}=3`, i.e,

        >>> op_3 = qml.PCPhase(1.23, dim=3, wires=[0, 1, 2, 3])

        It acts on :math:`N=2^n=16`-dimensional vectors and is described by

        .. math:: \Pi(\phi) = \exp(i\phi G) = \exp(i\phi(2\Pi-\mathbb{I}_N)),

        where :math:`G` is a diagonal matrix with :math:`d=3` ones, followed by
        :math:`2^n-d = 16 - 3=13` negative ones. Accordingly, :math:`\Pi` is diagonal with
        :math:`3` ones and :math:`13` zeros.

        First, we implement the global phase generated by :math:`\mathbb{I}_N` with
        a :class:`~.GlobalPhase` gate with angle :math:`-\phi`.
        Then we decompose :math:`d` into powers of two with positive or negative sign, via
        :math:`d=3=4-1 = 2^2-2^0`. This decomposition tells us that we can write the
        target gate with two (multi-)controlled phase shift gates. For this, we rewrite
        the projector :math:`\Pi` according to the decomposition as

        .. math::

            \Pi &= \text{diag}(1, 1, 1, 0, 0, \dots, 0)\\
            &=\text{diag}(1, 1, 1, 1, 0, \dots, 0)
            -\text{diag}(0, 0, 0, 1, 0, \dots, 0)

        where :math:`0,\dots, 0` indicates :math:`12` zeros each time.
        How do we realize this projector decomposition on the gate level?

        A singly-controlled phase shift gate applies a phase to a quarter of all computational
        basis states (the control filters by the state of one qubit, and the phase shift gate
        itself filters by the :math:`|1\rangle` state of the target qubit, cutting the number
        of states we are acting on in half each time).
        For :math:`n=4`, this amounts to :math:`2^4/4=4` states, which is exactly
        what we need for the first term above. To apply the phase to the *first* four states,
        :math:`|0000\rangle`, :math:`|0001\rangle`, :math:`|0010\rangle`, and :math:`|0011\rangle`,
        we want to "filter by" the first two qubits being in the :math:`|0\rangle` state.
        For qubit :math:`0`, we do this by controlling on the :math:`|0\rangle` state.
        For qubit :math:`1`, we pick it as the target of the controlled phase shift operation.
        Generically, this would make it act on the :math:`|1\rangle` state, so we simply flip
        qubit :math:`1` before and after the operation to apply the phase to the :math:`|0\rangle`
        state instead.
        Thus, we conclude this first step by applying the gates
        ``qml.X(1)``, ``qml.ctrl(qml.PhaseShift(2 * phi, 1), control=[0], control_values=[0])``,
        and ``qml.X(1)``.

        Next, we implement the second term in the projector decomposition, applying a phase
        to a single computational basis state. This requires us to fully control a phase shift
        gate, i.e., we use the last qubit as target and the other three as controls (there is
        some freedom of choice here, but this is a convenient choice).
        We want to apply the phase to the state :math:`|3\rangle=|0011\rangle`. So the controls
        :math:`0` and :math:`1` are set to zero and the control :math:`2` is set to one.
        As we want to effect the phase onto the :math:`|1\rangle` state of qubit :math:`3`,
        we don't need to flip the target bit as we did before. However, given the negative sign
        in the projector decomposition, we need to multiply the phase with :math:`-1`.
        Overall, we apply the gate
        ``qml.ctrl(qml.PhaseShift(-2 * phi, 3), control=[0, 1, 2], control_values=[0, 0, 1])``,
        which concludes the decomposition, now reading:

        >>> print(qml.draw(op_3.decomposition)())
        0: ────╭○───────────╭○─────────╭GlobalPhase(1.23)─┤  
        1: ──X─╰Rϕ(2.46)──X─├○─────────├GlobalPhase(1.23)─┤  
        2: ─────────────────├●─────────├GlobalPhase(1.23)─┤  
        3: ─────────────────╰Rϕ(-2.46)─╰GlobalPhase(1.23)─┤

        """
        with queuing.AnnotatedQueue() as q:
            _decompose_pcphase(*params, wires=wires, **hyperparams)

        if queuing.QueuingManager.recording():
            for op in q.queue:
                queuing.apply(op)

        return q.queue

    def adjoint(self) -> "PCPhase":
        """Computes the adjoint of the operator."""
        phi = self.parameters[0]
        dim, _ = self.hyperparameters["dimension"]
        return PCPhase(-1 * phi, dim=dim, wires=self.wires)

    def pow(self, z: int | float) -> list[Operator]:
        """Computes the operator raised to z."""
        phi = self.parameters[0]
        dim, _ = self.hyperparameters["dimension"]
        return [PCPhase(phi * z, dim=dim, wires=self.wires)]

    def simplify(self) -> "PCPhase":
        """Simplifies the operator if possible."""
        phi = self.parameters[0] % (2 * np.pi)
        dim, _ = self.hyperparameters["dimension"]

        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])

        return PCPhase(phi, dim=dim, wires=self.wires)

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        """The label of the operator when displayed in a circuit."""
        return super().label(decimals=decimals, base_label=base_label or "∏_ϕ", cache=cache)


def _ctrl_phase_shift_resource(subspace, n_control_wires, n_zero_control_values, n_work_wires):
    if n_control_wires == 0:
        return {qml.PhaseShift: 1}
    return {
        controlled_resource_rep(
            qml.PhaseShift,
            {},
            num_control_wires=n_control_wires,
            num_zero_control_values=n_zero_control_values,
            num_work_wires=n_work_wires,
        ): 1,
        qml.X: 2 * (1 - subspace),
    }


def _ctrl_phase_shift(
    phi, target_wire, subspace, control_wires, control_values, work_wires
):  # pylint: disable=too-many-arguments
    r"""Implement a ((multi-)controlled) phase shift on the specified subspace of a
    target qubit/wire.

    Args:
        phi (float): Phase shift angle
        target_wire (~.Wires): the target wire to apply phase shift to.
        subspace (int): which subspace of the target wire the phase shift is applied to. 0 or 1.
        control_wires (WiresLike): the control wires
        control_values (Iterable[bool | int]): the control values.
        work_wires (WiresLike): the work wires

    Returns:
        float: any global phase produced in the process.

    The decomposition for subspace=1 always is a simple ``PhaseShift`` gate, or its controlled
    counterpart. The decomposition of a non-controlled phase shift for subspace=0 can be achieved
    in two ways: The first is to flip the angle of the phase shift and complementing it with a
    global phase, so that the (diagonal of the) gate is decomposed as

    .. math::

        (\exp(i\phi), 1) = (1, \exp(-i\phi)) (\exp(i\phi), \exp(i\phi)).

    The second is to conjugate a phase shift by Pauli-X operators on the same qubit, decomposing
    the gate matrix as

    / e^(i\phi)   0 \ -- / 0   1 \/ 1     0     \/ 0   1 \
    \     0       1 / -- \ 1   0 /\ 0 e^(i\phi) /\ 1   0 /.

    Without controls, the first approach is nicer, because global phases usually are free
    operations. With controls, however, this approach would lead to a controlled global
    phase, which is equivalent to a phase shift on top of the controlled phase shift. For the
    second approach, we may use
    `the ctrl(compute-uncompute) pattern <https://iopscience.iop.org/article/10.1088/2058-9565/aaa5cc>`__
    to avoid controlling the Pauli-X operations, yielding two non-controlled Pauli-X gates and the
    "main" controlled phase shift operation. We deem this decomposition to be better.

    """

    if subspace == 1:
        # If there are no control wires, we are dealing with the very first phase shift of
        # the decomposition, which should be adding projectors. So subspace should have been 0.
        assert len(control_wires) > 0
        qml.ctrl(
            qml.PhaseShift(phi, wires=target_wire),
            control=control_wires,
            control_values=control_values,
            work_wires=work_wires,
        )
        return 0.0

    if len(control_wires) == 0:
        # Flip angle for phase_shift(subspace=0) = phase_shift(subspace=1)*global_phase
        qml.PhaseShift(-phi, wires=target_wire)
        return -phi

    qml.X(target_wire)
    qml.ctrl(
        qml.PhaseShift(phi, wires=target_wire),
        control=control_wires,
        control_values=control_values,
        work_wires=work_wires,
    )
    qml.X(target_wire)
    return 0.0


def _decompose_pcphase_resource(num_wires, dim):
    """Decompose the PCPhase operation into controlled phase shifts and Pauli-X gates."""

    gate_count = Counter()
    flipped, *powers_of_two = decomp_int_to_powers_of_two(dim, num_wires + 1)
    sigma = (-1) ** flipped
    powers_of_two = [sigma * val for val in powers_of_two]

    n_zero_control_values = 0
    for i, c_i in enumerate(powers_of_two):

        if c_i != 0:
            subspace = int(c_i < 0)
            if flipped:
                subspace = 1 - subspace
            gate_count.update(
                _ctrl_phase_shift_resource(
                    subspace,
                    n_control_wires=i,
                    n_zero_control_values=n_zero_control_values,
                    n_work_wires=num_wires - i - 1,
                )
            )

        d_i = next(iter(val for val in powers_of_two[i + 1 :] if val != 0), None)
        next_cval = d_i == 1
        if c_i == 0:
            next_cval = not next_cval
        if flipped:
            next_cval = not next_cval
        if not next_cval:
            n_zero_control_values += 1

    gate_count[qml.GlobalPhase] += 1
    return dict(gate_count)


@register_resources(_decompose_pcphase_resource)
def _decompose_pcphase(phi, wires, dimension):
    """Decompose the PCPhase operation into controlled phase shifts and Pauli-X gates."""

    dim, _ = dimension

    # Use one more bit than there are wires, according to flipping all relevant bits for the
    # projector decomposition, or a global phase on the gate level. Afterwards, we have
    # dim <= 2**len(wires)=2**(n-1), which is a requirement of decomp_int_to_powers_of_two.
    flipped, *powers_of_two = decomp_int_to_powers_of_two(dim, len(wires) + 1)

    sigma = (-1) ** flipped

    # Overall global phase to implement I_N part of the generator
    global_phase = sigma * phi

    phi = 2 * sigma * phi

    # If in flipped (sigma=-1) mode, reverse the sign of all coefficients
    powers_of_two = [sigma * val for val in powers_of_two]

    assert len(powers_of_two) == len(wires)

    control_values = []
    for i, c_i in enumerate(powers_of_two):

        if c_i != 0:
            # Projector with rank 2**(n-1-i) needs to be added/subtracted
            subspace = int(c_i < 0)  # If c_i < 0, target |1> subspace, else target |0> subspace
            if flipped:  # Flip subspace if in flipped (sigma=-1) mode
                subspace = 1 - subspace
            global_phase += _ctrl_phase_shift(
                c_i * phi,
                target_wire=wires[i],
                subspace=subspace,
                control_wires=wires[:i],
                control_values=control_values,
                work_wires=wires[i + 1 :],  # Unused wires of PCPhase can be used as work wires
            )

        # The control value to be used on the current wire (it will be the same for all
        # subsequent operations) depends on whether we add or subtract next)
        d_i = next(iter(val for val in powers_of_two[i + 1 :] if val != 0), None)
        # If we add next, control into the |1> subspace, otherwise into |0> subspace
        # The control value is modified both if we are in the flipped global phase mode and if
        # the current loop iteration did not add a bit string/projector/gate (c_i==0)
        next_cval = d_i == 1
        if c_i == 0:
            next_cval = not next_cval
        if flipped:
            next_cval = not next_cval
        control_values.append(next_cval)

    qml.GlobalPhase(global_phase)


add_decomps(PCPhase, _decompose_pcphase)


class IsingXX(Operation):
    r"""
    Ising XX coupling gate

    .. math:: XX(\phi) = \exp\left(-i \frac{\phi}{2} (X \otimes X)\right) =
        \begin{bmatrix} =
            \cos(\phi / 2) & 0 & 0 & -i \sin(\phi / 2) \\
            0 & \cos(\phi / 2) & -i \sin(\phi / 2) & 0 \\
            0 & -i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
            -i \sin(\phi / 2) & 0 & 0 & \cos(\phi / 2)
        \end{bmatrix}.

    .. note::

        Special cases of using the :math:`XX` operator include:

        * :math:`XX(0) = I`;
        * :math:`XX(\pi) = i (X \otimes X)`.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(XX(\phi)) = \frac{1}{2}\left[f(XX(\phi +\pi/2)) - f(XX(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`XX(\phi)`.

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self) -> "qml.Hamiltonian":
        return qml.Hamiltonian([-0.5], [PauliX(wires=self.wires[0]) @ PauliX(wires=self.wires[1])])

    def __init__(self, phi: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(phi, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {}

    @staticmethod
    def compute_matrix(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.

        .. seealso:: :meth:`~.IsingXX.matrix`


        Args:
           phi (TensorLike): phase angle

        Returns:
           TensorLike: canonical matrix

        **Example**

        >>> qml.IsingXX.compute_matrix(torch.tensor(0.5))
        tensor([[0.9689+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000-0.2474j],
                [0.0000+0.0000j, 0.9689+0.0000j, 0.0000-0.2474j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000-0.2474j, 0.9689+0.0000j, 0.0000+0.0000j],
                [0.0000-0.2474j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9689+0.0000j]],
               dtype=torch.complex128)
        """
        c = math.cos(phi / 2)
        s = math.sin(phi / 2)

        eye = math.eye(4, like=phi)
        rev_eye = math.convert_like(np.eye(4)[::-1].copy(), phi)
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            c = math.cast_like(c, 1j)
            s = math.cast_like(s, 1j)
            eye = math.cast_like(eye, 1j)
            rev_eye = math.cast_like(rev_eye, 1j)

        # The following avoids casting an imaginary quantity to reals when backpropagating
        js = -1j * s
        if math.ndim(phi) == 0:
            return c * eye + js * rev_eye

        return math.tensordot(c, eye, axes=0) + math.tensordot(js, rev_eye, axes=0)

    @staticmethod
    def compute_decomposition(phi: TensorLike, wires: WiresLike) -> list[Operator]:
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.IsingXX.decomposition`.

        Args:
            phi (TensorLike): the phase angle
            wires (Iterable, Wires): the subsystem the gate acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.IsingXX.compute_decomposition(1.23, wires=(0,1))
        [CNOT(wires=[0, 1]), RX(1.23, wires=[0]), CNOT(wires=[0, 1])]

        """
        decomp_ops = [
            qml.CNOT(wires=wires),
            RX(phi, wires=[wires[0]]),
            qml.CNOT(wires=wires),
        ]
        return decomp_ops

    def adjoint(self) -> "IsingXX":
        (phi,) = self.parameters
        return IsingXX(-phi, wires=self.wires)

    def pow(self, z: int | float) -> list[Operator]:
        return [IsingXX(self.data[0] * z, wires=self.wires)]

    def simplify(self) -> "IsingXX":
        phi = self.data[0] % (4 * np.pi)

        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])

        return IsingXX(phi, wires=self.wires)


def _isingxx_to_cnot_rx_cnot_resources():
    return {qml.CNOT: 2, qml.RX: 1}


@register_resources(_isingxx_to_cnot_rx_cnot_resources)
def _isingxx_to_cnot_rx_cnot(phi: TensorLike, wires: WiresLike, **__):
    qml.CNOT(wires=wires)
    qml.RX(phi, wires=[wires[0]])
    qml.CNOT(wires=wires)


add_decomps(IsingXX, _isingxx_to_cnot_rx_cnot)
add_decomps("Adjoint(IsingXX)", adjoint_rotation)
add_decomps("Pow(IsingXX)", pow_rotation)


class IsingYY(Operation):
    r"""
    Ising YY coupling gate

    .. math:: \mathtt{YY}(\phi) = \exp\left(-i \frac{\phi}{2} (Y \otimes Y)\right) =
        \begin{bmatrix}
            \cos(\phi / 2) & 0 & 0 & i \sin(\phi / 2) \\
            0 & \cos(\phi / 2) & -i \sin(\phi / 2) & 0 \\
            0 & -i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
            i \sin(\phi / 2) & 0 & 0 & \cos(\phi / 2)
        \end{bmatrix}.

    .. note::

        Special cases of using the :math:`YY` operator include:

        * :math:`YY(0) = I`;
        * :math:`YY(\pi) = i (Y \otimes Y)`.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(YY(\phi)) = \frac{1}{2}\left[f(YY(\phi +\pi/2)) - f(YY(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`YY(\phi)`.

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self) -> "qml.Hamiltonian":
        return qml.Hamiltonian([-0.5], [PauliY(wires=self.wires[0]) @ PauliY(wires=self.wires[1])])

    def __init__(self, phi: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(phi, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {}

    @staticmethod
    def compute_decomposition(phi: TensorLike, wires: WiresLike) -> list[Operator]:
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.IsingYY.decomposition`.

        Args:
            phi (float): the phase angle
            wires (Iterable, Wires): the subsystem the gate acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.IsingYY.compute_decomposition(1.23, wires=(0,1))
        [CY(wires=[0, 1]), RY(1.23, wires=[0]), CY(wires=[0, 1])]

        """
        return [
            qml.CY(wires=wires),
            RY(phi, wires=[wires[0]]),
            qml.CY(wires=wires),
        ]

    @staticmethod
    def compute_matrix(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.IsingYY.matrix`


        Args:
           phi (TensorLike): phase angle

        Returns:
           TensorLike: canonical matrix

        **Example**

        >>> qml.IsingYY.compute_matrix(torch.tensor(0.5))
        tensor([[0.9689+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.2474j],
                [0.0000+0.0000j, 0.9689+0.0000j, 0.0000-0.2474j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000-0.2474j, 0.9689+0.0000j, 0.0000+0.0000j],
                [0.0000+0.2474j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9689+0.0000j]],
               dtype=torch.complex128)
        """
        c = math.cos(phi / 2)
        s = math.sin(phi / 2)

        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            c = math.cast_like(c, 1j)
            s = math.cast_like(s, 1j)

        js = 1j * s
        r_term = math.cast_like(
            math.array(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
                like=js,
            ),
            1j,
        )
        if math.ndim(phi) == 0:
            return c * math.cast_like(math.eye(4, like=c), c) + js * r_term

        return math.tensordot(c, np.eye(4), axes=0) + math.tensordot(js, r_term, axes=0)

    def adjoint(self) -> "IsingYY":
        (phi,) = self.parameters
        return IsingYY(-phi, wires=self.wires)

    def pow(self, z: int | float) -> list[Operator]:
        return [IsingYY(self.data[0] * z, wires=self.wires)]

    def simplify(self) -> "IsingYY":
        phi = self.data[0] % (4 * np.pi)

        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])

        return IsingYY(phi, wires=self.wires)


def _isingyy_to_cy_ry_cy_resources():
    return {qml.CY: 2, RY: 1}


@register_resources(_isingyy_to_cy_ry_cy_resources)
def _isingyy_to_cy_ry_cy(phi: TensorLike, wires: WiresLike, **__):
    qml.CY(wires=wires)
    RY(phi, wires=[wires[0]])
    qml.CY(wires=wires)


add_decomps(IsingYY, _isingyy_to_cy_ry_cy)
add_decomps("Adjoint(IsingYY)", adjoint_rotation)
add_decomps("Pow(IsingYY)", pow_rotation)


class IsingZZ(Operation):
    r"""
    Ising ZZ coupling gate

    .. math:: ZZ(\phi) = \exp\left(-i \frac{\phi}{2} (Z \otimes Z)\right) =
        \begin{bmatrix}
            e^{-i \phi / 2} & 0 & 0 & 0 \\
            0 & e^{i \phi / 2} & 0 & 0 \\
            0 & 0 & e^{i \phi / 2} & 0 \\
            0 & 0 & 0 & e^{-i \phi / 2}
        \end{bmatrix}.

    .. note::

        Special cases of using the :math:`ZZ` operator include:

        * :math:`ZZ(0) = I`;
        * :math:`ZZ(\pi) = - (Z \otimes Z)`;
        * :math:`ZZ(2\pi) = - I`;

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(ZZ(\phi)) = \frac{1}{2}\left[f(ZZ(\phi +\pi/2)) - f(ZZ(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`ZZ(\theta)`.

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self) -> "qml.Hamiltonian":
        return qml.Hamiltonian([-0.5], [PauliZ(wires=self.wires[0]) @ PauliZ(wires=self.wires[1])])

    def __init__(self, phi: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(phi, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {}

    @staticmethod
    def compute_decomposition(phi: TensorLike, wires: WiresLike):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.IsingZZ.decomposition`.

        Args:
            phi (float): the phase angle
            wires (Iterable, Wires): the subsystem the gate acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.IsingZZ.compute_decomposition(1.23, wires=[0, 1])
        [CNOT(wires=[0, 1]), RZ(1.23, wires=[1]), CNOT(wires=[0, 1])]

        """
        return [
            qml.CNOT(wires=wires),
            RZ(phi, wires=[wires[1]]),
            qml.CNOT(wires=wires),
        ]

    @staticmethod
    def compute_matrix(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.IsingZZ.matrix`


        Args:
           phi (TensorLike): phase angle

        Returns:
           TensorLike: canonical matrix

        **Example**

        >>> qml.IsingZZ.compute_matrix(torch.tensor(0.5))
        tensor([[0.9689-0.2474j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.9689+0.2474j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.9689+0.2474j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9689-0.2474j]])
        """
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            p = math.exp(-0.5j * math.cast_like(phi, 1j))
            if math.ndim(p) == 0:
                return math.diag([p, math.conj(p), math.conj(p), p])

            diags = stack_last([p, math.conj(p), math.conj(p), p])
            return diags[:, :, np.newaxis] * math.cast_like(math.eye(4, like=diags), diags)

        signs = math.array([1, -1, -1, 1], like=phi)
        arg = -0.5j * phi

        if math.ndim(arg) == 0:
            return math.diag(math.exp(arg * signs))

        diags = math.exp(math.outer(arg, signs))
        return diags[:, :, np.newaxis] * math.cast_like(math.eye(4, like=diags), diags)

    @staticmethod
    def compute_eigvals(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.IsingZZ.eigvals`


        Args:
            phi (TensorLike) phase angle

        Returns:
            TensorLike: eigenvalues

        **Example**

        >>> qml.IsingZZ.compute_eigvals(torch.tensor(0.5))
        tensor([0.9689-0.2474j, 0.9689+0.2474j, 0.9689+0.2474j, 0.9689-0.2474j])
        """
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phase = math.exp(-0.5j * math.cast_like(phi, 1j))
            return stack_last([phase, math.conj(phase), math.conj(phase), phase])

        prefactors = math.array([-0.5j, 0.5j, 0.5j, -0.5j], like=phi)
        if math.ndim(phi) == 0:
            product = phi * prefactors
        else:
            product = math.outer(phi, prefactors)
        return math.exp(product)

    def adjoint(self) -> "IsingZZ":
        (phi,) = self.parameters
        return IsingZZ(-phi, wires=self.wires)

    def pow(self, z: int | float) -> list[Operator]:
        return [IsingZZ(self.data[0] * z, wires=self.wires)]

    def simplify(self) -> "IsingZZ":
        phi = self.data[0] % (4 * np.pi)

        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])

        return IsingZZ(phi, wires=self.wires)


def _isingzz_to_cnot_rz_cnot_resources():
    return {qml.CNOT: 2, RZ: 1}


@register_resources(_isingzz_to_cnot_rz_cnot_resources)
def _isingzz_to_cnot_rz_cnot(phi: TensorLike, wires: WiresLike, **__):
    qml.CNOT(wires=wires)
    RZ(phi, wires=[wires[1]])
    qml.CNOT(wires=wires)


add_decomps(IsingZZ, _isingzz_to_cnot_rz_cnot)
add_decomps("Adjoint(IsingZZ)", adjoint_rotation)
add_decomps("Pow(IsingZZ)", pow_rotation)


class IsingXY(Operation):
    r"""
    Ising (XX + YY) coupling gate

    .. math:: \mathtt{XY}(\phi) = \exp\left(i \frac{\phi}{4} (X \otimes X + Y \otimes Y)\right) =
        \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & \cos(\phi / 2) & i \sin(\phi / 2) & 0 \\
            0 & i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    .. note::

        Special cases of using the :math:`XY` operator include:

        * :math:`XY(0) = I`;
        * :math:`XY(\frac{\pi}{2}) = \sqrt{iSWAP}`;
        * :math:`XY(\pi) = iSWAP`;

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: The XY operator satisfies a four-term parameter-shift rule

      .. math::
          \frac{d}{d \phi} f(XY(\phi))
          = c_+ \left[ f(XY(\phi + a)) - f(XY(\phi - a)) \right]
          - c_- \left[ f(XY(\phi + b)) - f(XY(\phi - b)) \right]

      where :math:`f` is an expectation value depending on :math:`XY(\phi)`, and

      - :math:`a = \pi / 2`
      - :math:`b = 3 \pi / 2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4 \sqrt{2}}`

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    grad_method = "A"
    parameter_frequencies = [(0.5, 1.0)]

    def generator(self) -> "qml.Hamiltonian":

        return qml.Hamiltonian(
            [0.25, 0.25],
            [
                qml.X(wires=self.wires[0]) @ qml.X(wires=self.wires[1]),
                qml.Y(wires=self.wires[0]) @ qml.Y(wires=self.wires[1]),
            ],
        )

    def __init__(self, phi: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(phi, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {}

    @staticmethod
    def compute_decomposition(phi: TensorLike, wires: WiresLike) -> list[Operator]:
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.IsingXY.decomposition`.

        Args:
            phi (float): the phase angle
            wires (Iterable, Wires): the subsystem the gate acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.IsingXY.compute_decomposition(1.23, wires=(0,1))
        [H(0), CY(wires=[0, 1]), RY(0.615, wires=[0]), RX(-0.615, wires=[1]), CY(wires=[0, 1]), H(0)]

        """
        return [
            Hadamard(wires=[wires[0]]),
            qml.CY(wires=wires),
            RY(phi / 2, wires=[wires[0]]),
            RX(-phi / 2, wires=[wires[1]]),
            qml.CY(wires=wires),
            Hadamard(wires=[wires[0]]),
        ]

    @staticmethod
    def compute_matrix(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.IsingXY.matrix`


        Args:
           phi (TensorLike): phase angle

        Returns:
           TensorLike: canonical matrix

        **Example**

        >>> qml.IsingXY.compute_matrix(0.5)
        array([[1.        +0.j        , 0.        +0.j        ,        0.        +0.j        , 0.        +0.j        ],
               [0.        +0.j        , 0.96891242+0.j        ,        0.        +0.24740396j, 0.        +0.j        ],
               [0.        +0.j        , 0.        +0.24740396j,        0.96891242+0.j        , 0.        +0.j        ],
               [0.        +0.j        , 0.        +0.j        ,        0.        +0.j        , 1.        +0.j        ]])
        """
        c = math.cos(phi / 2)
        s = math.sin(phi / 2)

        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            c = math.cast_like(c, 1j)
            s = math.cast_like(s, 1j)

        js = 1j * s
        off_diag = math.cast_like(
            math.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                like=js,
            ),
            1j,
        )
        if math.ndim(phi) == 0:
            return math.diag([1, c, c, 1]) + js * off_diag

        ones = math.ones_like(c)
        diags = stack_last([ones, c, c, ones])[:, :, np.newaxis]
        return diags * np.eye(4) + math.tensordot(js, off_diag, axes=0)

    @staticmethod
    def compute_eigvals(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.IsingXY.eigvals`


        Args:
            phi (TensorLike): phase angle

        Returns:
            TensorLike: eigenvalues

        **Example**

        >>> qml.IsingXY.compute_eigvals(0.5)
        array([0.96891242+0.24740396j, 0.96891242-0.24740396j,       1.        +0.j        , 1.        +0.j        ])
        """
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phi = math.cast_like(phi, 1j)

        signs = np.array([1, -1, 0, 0])
        if math.ndim(phi) == 0:
            return math.exp(0.5j * phi * signs)

        return math.exp(math.tensordot(0.5j * phi, signs, axes=0))

    def adjoint(self) -> "IsingXY":
        (phi,) = self.parameters
        return IsingXY(-phi, wires=self.wires)

    def pow(self, z: int | float) -> list[Operator]:
        return [IsingXY(self.data[0] * z, wires=self.wires)]

    def simplify(self) -> "IsingXY":
        phi = self.data[0] % (4 * np.pi)

        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])

        return IsingXY(phi, wires=self.wires)


def _isingxy_to_h_cy_resources():
    return {Hadamard: 2, qml.CY: 2, RY: 1, RX: 1}


@register_resources(_isingxy_to_h_cy_resources)
def _isingxy_to_h_cy(phi: TensorLike, wires: WiresLike, **__):
    Hadamard(wires=[wires[0]])
    qml.CY(wires=wires)
    RY(phi / 2, wires=[wires[0]])
    RX(-phi / 2, wires=[wires[1]])
    qml.CY(wires=wires)
    Hadamard(wires=[wires[0]])


add_decomps(IsingXY, _isingxy_to_h_cy)
add_decomps("Adjoint(IsingXY)", adjoint_rotation)
add_decomps("Pow(IsingXY)", pow_rotation)


class PSWAP(Operation):
    r"""Phase SWAP gate

    .. math:: PSWAP(\phi) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & e^{i \phi} & 0 \\
            0 & e^{i \phi} & 0 & 0 \\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe:

    .. math::
        \frac{d}{d \phi} PSWAP(\phi)
        = \frac{1}{2} \left[ PSWAP(\phi + \pi / 2) - PSWAP(\phi - \pi / 2) \right]

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    grad_method = "A"
    grad_recipe = ([[0.5, 1, np.pi / 2], [-0.5, 1, -np.pi / 2]],)

    def __init__(self, phi: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(phi, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {}

    @staticmethod
    def compute_decomposition(phi: TensorLike, wires: WiresLike) -> list[Operator]:
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.PSWAP.decomposition`.

        Args:
            phi (float): the phase angle
            wires (Iterable, Wires): the subsystem the gate acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.PSWAP.compute_decomposition(1.23, wires=(0,1))
        [SWAP(wires=[0, 1]), CNOT(wires=[0, 1]), PhaseShift(1.23, wires=[1]), CNOT(wires=[0, 1])]
        """
        return [
            qml.SWAP(wires=wires),
            qml.CNOT(wires=wires),
            PhaseShift(phi, wires=[wires[1]]),
            qml.CNOT(wires=wires),
        ]

    @staticmethod
    def compute_matrix(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.PSWAP.matrix`


        Args:
           phi (TensorLike): phase angle

        Returns:
           TensorLike: canonical matrix

        **Example**

        >>> qml.PSWAP.compute_matrix(0.5)
        array([[1.        +0.j        , 0.        +0.j        ,
                0.        +0.j        , 0.        +0.j        ],
               [0.        +0.j        , 0.        +0.j        ,
                0.87758256+0.47942554j, 0.        +0.j        ],
               [0.        +0.j        , 0.87758256+0.47942554j,
                0.        +0.j        , 0.        +0.j        ],
               [0.        +0.j        , 0.        +0.j        ,
                0.        +0.j        , 1.        +0.j        ]])
        """
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phi = math.cast_like(phi, 1j)

        e = math.exp(1j * phi)
        zero = math.zeros_like(phi)
        one = math.ones_like(phi)

        return math.stack(
            [
                stack_last([one, zero, zero, zero]),
                stack_last([zero, zero, e, zero]),
                stack_last([zero, e, zero, zero]),
                stack_last([zero, zero, zero, one]),
            ],
            axis=-2,
        )

    @staticmethod
    def compute_eigvals(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.PSWAP.eigvals`


        Args:
            phi (TensorLike): phase angle

        Returns:
            TensorLike: eigenvalues

        **Example**

        >>> qml.PSWAP.compute_eigvals(0.5)
        array([ 1.        +0.j        ,  1.        +0.j        ,
               -0.87758256-0.47942554j,  0.87758256+0.47942554j])
        """
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phi = math.cast_like(phi, 1j)

        e = math.exp(1j * phi)
        one = math.ones_like(phi)
        return math.transpose(math.stack([one, one, -e, e]))

    def adjoint(self) -> "PSWAP":
        (phi,) = self.parameters
        return PSWAP(-phi, wires=self.wires)

    def simplify(self) -> "PSWAP":
        phi = self.data[0] % (2 * np.pi)

        if _can_replace(phi, 0):
            return qml.SWAP(wires=self.wires)

        return PSWAP(phi, wires=self.wires)


def _pswap_to_swap_cnot_phaseshift_cnot_resources():
    return {qml.SWAP: 1, qml.CNOT: 2, PhaseShift: 1}


@register_resources(_pswap_to_swap_cnot_phaseshift_cnot_resources)
def _pswap_to_swap_cnot_phaseshift_cnot(phi: TensorLike, wires: WiresLike, **__):
    qml.SWAP(wires=wires)
    qml.CNOT(wires=wires)
    PhaseShift(phi, wires=[wires[1]])
    qml.CNOT(wires=wires)


add_decomps(PSWAP, _pswap_to_swap_cnot_phaseshift_cnot)
add_decomps("Adjoint(PSWAP)", adjoint_rotation)


class CPhaseShift00(Operation):
    r"""
    A qubit controlled phase shift.

    .. math:: CR_{00}(\phi) = \begin{bmatrix}
                e^{i\phi} & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit** and controls
        on the zero state :math:`|0\rangle`.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe:

    .. math::
        \frac{d}{d \phi} CR_{00}(\phi)
        = \frac{1}{2} \left[ CR_{00}(\phi + \pi / 2)
            - CR_{00}(\phi - \pi / 2) \right]

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self) -> "qml.Projector":
        return qml.Projector(np.array([0, 0]), wires=self.wires)

    resource_keys = set()

    def __init__(self, phi: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(phi, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {}

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        return super().label(decimals=decimals, base_label="Rϕ(00)", cache=cache)

    @staticmethod
    def compute_matrix(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CPhaseShift00.matrix`

        Args:
            phi (TensorLike): phase shift

        Returns:
            TensorLike: canonical matrix

        **Example**

        >>> qml.CPhaseShift00.compute_matrix(torch.tensor(0.5))
        tensor([[0.8776+0.4794j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 1.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 1.0000+0.0000j]])
        """
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phi = math.cast_like(phi, 1j)

        exp_part = math.exp(1j * phi)

        if math.ndim(phi) > 0:
            ones = math.ones_like(exp_part)
            zeros = math.zeros_like(exp_part)
            matrix = [
                [exp_part, zeros, zeros, zeros],
                [zeros, ones, zeros, zeros],
                [zeros, zeros, ones, zeros],
                [zeros, zeros, zeros, ones],
            ]

            return math.stack([stack_last(row) for row in matrix], axis=-2)

        return math.diag([exp_part, 1, 1, 1])

    @staticmethod
    def compute_eigvals(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.CPhaseShift00.eigvals`


        Args:
            phi (TensorLike): phase shift

        Returns:
            TensorLike: eigenvalues

        **Example**

        >>> qml.CPhaseShift00.compute_eigvals(torch.tensor(0.5))
        tensor([0.8776+0.4794j, 1.0000+0.0000j, 1.0000+0.0000j, 1.0000+0.0000j])
        """
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phi = math.cast_like(phi, 1j)

        exp_part = math.exp(1j * phi)
        ones = math.ones_like(exp_part)
        return stack_last([exp_part, ones, ones, ones])

    @staticmethod
    def compute_decomposition(phi: TensorLike, wires: WiresLike) -> list[Operator]:
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.CPhaseShift00.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CPhaseShift00.compute_decomposition(1.234, wires=(0,1))
        [X(0),
        X(1),
        PhaseShift(0.617, wires=[0]),
        PhaseShift(0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        PhaseShift(-0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        X(1),
        X(0)]

        """
        decomp_ops = [
            PauliX(wires[0]),
            PauliX(wires[1]),
            PhaseShift(phi / 2, wires=[wires[0]]),
            PhaseShift(phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            PhaseShift(-phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            PauliX(wires[1]),
            PauliX(wires[0]),
        ]
        return decomp_ops

    def adjoint(self) -> "CPhaseShift00":
        return CPhaseShift00(-self.data[0], wires=self.wires)

    def pow(self, z: int | float) -> "CPhaseShift00":
        return [CPhaseShift00(self.data[0] * z, wires=self.wires)]

    @property
    def control_values(self) -> str:
        """str: The control values of the operation"""
        return "0"

    @property
    def control_wires(self) -> Wires:
        return self.wires[0:1]


def _cphaseshift00_resources():
    return {PauliX: 4, PhaseShift: 3, qml.CNOT: 2}


@register_resources(_cphaseshift00_resources)
def _cphaseshift00(phi: TensorLike, wires: WiresLike, **__):
    PauliX(wires[0])
    PauliX(wires[1])
    PhaseShift(phi / 2, wires=[wires[0]])
    PhaseShift(phi / 2, wires=[wires[1]])
    qml.CNOT(wires=wires)
    PhaseShift(-phi / 2, wires=[wires[1]])
    qml.CNOT(wires=wires)
    PauliX(wires[1])
    PauliX(wires[0])


add_decomps(CPhaseShift00, _cphaseshift00)
add_decomps("Adjoint(CPhaseShift00)", adjoint_rotation)
add_decomps("Pow(CPhaseShift00)", pow_rotation)


class CPhaseShift01(Operation):
    r"""
    A qubit controlled phase shift.

    .. math:: CR_{01\phi}(\phi) = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & e^{i\phi} & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit** and controls
        on the zero state :math:`|0\rangle`.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe:

    .. math::
        \frac{d}{d \phi} CR_{01}(\phi)
        = \frac{1}{2} \left[ CR_{01}(\phi + \pi / 2)
            - CR_{01}(\phi - \pi / 2) \right]

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self) -> "qml.Projector":
        return qml.Projector(np.array([0, 1]), wires=self.wires)

    resource_keys = set()

    def __init__(self, phi: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(phi, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {}

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        return super().label(decimals=decimals, base_label="Rϕ(01)", cache=cache)

    @staticmethod
    def compute_matrix(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CPhaseShift01.matrix`

        Args:
            phi (TensorLike): phase shift

        Returns:
            TensorLike: canonical matrix

        **Example**

        >>> qml.CPhaseShift01.compute_matrix(torch.tensor(0.5))
        tensor([[1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.8776+0.4794j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 1.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 1.0000+0.0000j]])
        """
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phi = math.cast_like(phi, 1j)

        exp_part = math.exp(1j * phi)

        if math.ndim(phi) > 0:
            ones = math.ones_like(exp_part)
            zeros = math.zeros_like(exp_part)
            matrix = [
                [ones, zeros, zeros, zeros],
                [zeros, exp_part, zeros, zeros],
                [zeros, zeros, ones, zeros],
                [zeros, zeros, zeros, ones],
            ]

            return math.stack([stack_last(row) for row in matrix], axis=-2)

        return math.diag([1, exp_part, 1, 1])

    @staticmethod
    def compute_eigvals(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.CPhaseShift01.eigvals`


        Args:
            phi (TensorLike): phase shift

        Returns:
            TensorLike: eigenvalues

        **Example**

        >>> qml.CPhaseShift01.compute_eigvals(torch.tensor(0.5))
        tensor([1.0000+0.0000j, 0.8776+0.4794j, 1.0000+0.0000j, 1.0000+0.0000j])
        """
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phi = math.cast_like(phi, 1j)

        exp_part = math.exp(1j * phi)
        ones = math.ones_like(exp_part)
        return stack_last([ones, exp_part, ones, ones])

    @staticmethod
    def compute_decomposition(phi: TensorLike, wires: WiresLike) -> list[Operator]:
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.
        .. seealso:: :meth:`~.CPhaseShift01.decomposition`.

        Args:
            phi (Tensorlike): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CPhaseShift01.compute_decomposition(1.234, wires=(0,1))
        [X(0),
        PhaseShift(0.617, wires=[0]),
        PhaseShift(0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        PhaseShift(-0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        X(0)]

        """
        decomp_ops = [
            PauliX(wires[0]),
            PhaseShift(phi / 2, wires=[wires[0]]),
            PhaseShift(phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            PhaseShift(-phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            PauliX(wires[0]),
        ]
        return decomp_ops

    def adjoint(self) -> "CPhaseShift01":
        return CPhaseShift01(-self.data[0], wires=self.wires)

    def pow(self, z: int | float) -> "CPhaseShift01":
        return [CPhaseShift01(self.data[0] * z, wires=self.wires)]

    @property
    def control_values(self) -> str:
        """str: The control values of the operation"""
        return "0"

    @property
    def control_wires(self) -> Wires:
        return self.wires[0:1]


def _cphaseshift01_resources():
    return {PauliX: 2, PhaseShift: 3, qml.CNOT: 2}


@register_resources(_cphaseshift01_resources)
def _cphaseshift01(phi: TensorLike, wires: WiresLike, **__):
    PauliX(wires[0])
    PhaseShift(phi / 2, wires=[wires[0]])
    PhaseShift(phi / 2, wires=[wires[1]])
    qml.CNOT(wires=wires)
    PhaseShift(-phi / 2, wires=[wires[1]])
    qml.CNOT(wires=wires)
    PauliX(wires[0])


add_decomps(CPhaseShift01, _cphaseshift01)
add_decomps("Adjoint(CPhaseShift01)", adjoint_rotation)
add_decomps("Pow(CPhaseShift01)", pow_rotation)


class CPhaseShift10(Operation):
    r"""
    A qubit controlled phase shift.

    .. math:: CR_{10\phi}(\phi) = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & e^{i\phi} & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe:

    .. math::
        \frac{d}{d \phi} CR_{10}(\phi)
        = \frac{1}{2} \left[ CR_{10}(\phi + \pi / 2)
            - CR_{10}(\phi - \pi / 2) \right]

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Any, Wires): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self) -> "qml.Projector":
        return qml.Projector(np.array([1, 0]), wires=self.wires)

    resource_keys = set()

    def __init__(self, phi: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(phi, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {}

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        return super().label(decimals=decimals, base_label="Rϕ(10)", cache=cache)

    @staticmethod
    def compute_matrix(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CPhaseShift10.matrix`

        Args:
            phi (TensorLike): phase shift

        Returns:
            TensorLike: canonical matrix

        **Example**

        >>> qml.CPhaseShift10.compute_matrix(torch.tensor(0.5))
        tensor([[1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.8776+0.4794j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 1.0000+0.0000j]])
        """
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phi = math.cast_like(phi, 1j)

        exp_part = math.exp(1j * phi)

        if math.ndim(phi) > 0:
            ones = math.ones_like(exp_part)
            zeros = math.zeros_like(exp_part)
            matrix = [
                [ones, zeros, zeros, zeros],
                [zeros, ones, zeros, zeros],
                [zeros, zeros, exp_part, zeros],
                [zeros, zeros, zeros, ones],
            ]

            return math.stack([stack_last(row) for row in matrix], axis=-2)

        return math.diag([1, 1, exp_part, 1])

    @staticmethod
    def compute_eigvals(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.CPhaseShift10.eigvals`


        Args:
            phi (TensorLike): phase shift

        Returns:
            TensorLike: eigenvalues

        **Example**

        >>> qml.CPhaseShift10.compute_eigvals(torch.tensor(0.5))
        tensor([1.0000+0.0000j, 1.0000+0.0000j, 0.8776+0.4794j, 1.0000+0.0000j])
        """
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phi = math.cast_like(phi, 1j)

        exp_part = math.exp(1j * phi)
        ones = math.ones_like(exp_part)
        return stack_last([ones, ones, exp_part, ones])

    @staticmethod
    def compute_decomposition(phi: TensorLike, wires: WiresLike) -> list[Operator]:
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.
        .. seealso:: :meth:`~.CPhaseShift10.decomposition`.

        Args:
            phi (TensorLike): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CPhaseShift10.compute_decomposition(1.234, wires=(0,1))
        [X(1),
        PhaseShift(0.617, wires=[0]),
        PhaseShift(0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        PhaseShift(-0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        X(1)]

        """
        decomp_ops = [
            PauliX(wires[1]),
            PhaseShift(phi / 2, wires=[wires[0]]),
            PhaseShift(phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            PhaseShift(-phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            PauliX(wires[1]),
        ]
        return decomp_ops

    def adjoint(self) -> "CPhaseShift10":
        return CPhaseShift10(-self.data[0], wires=self.wires)

    def pow(self, z: int | float):
        return [CPhaseShift10(self.data[0] * z, wires=self.wires)]

    @property
    def control_wires(self) -> Wires:
        return self.wires[0:1]


def _cphaseshift10_resources():
    return {PauliX: 2, PhaseShift: 3, qml.CNOT: 2}


@register_resources(_cphaseshift10_resources)
def _cphaseshift10(phi: TensorLike, wires: WiresLike, **__):
    PauliX(wires[1])
    PhaseShift(phi / 2, wires=[wires[0]])
    PhaseShift(phi / 2, wires=[wires[1]])
    qml.CNOT(wires=wires)
    PhaseShift(-phi / 2, wires=[wires[1]])
    qml.CNOT(wires=wires)
    PauliX(wires[1])


add_decomps(CPhaseShift10, _cphaseshift10)
add_decomps("Adjoint(CPhaseShift10)", adjoint_rotation)
add_decomps("Pow(CPhaseShift10)", pow_rotation)
