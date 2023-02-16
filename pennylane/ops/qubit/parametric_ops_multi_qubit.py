# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=too-many-arguments
"""
This submodule contains the discrete-variable quantum operations that are the
core parameterized gates.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access,invalid-overridden-method
import functools
from operator import matmul
import numpy as np

import pennylane as qml
from pennylane.math import expand_matrix
from pennylane.operation import AnyWires, Operation
from pennylane.utils import pauli_eigs
from pennylane.wires import Wires

from .parametric_ops_single_qubit import _can_replace, stack_last


class MultiRZ(Operation):
    r"""
    Arbitrary multi Z rotation.

    .. math::

        MultiRZ(\theta) = \exp(-i \frac{\theta}{2} Z^{\otimes n})

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
        theta (tensor_like or float): rotation angle :math:`\theta`
        wires (Sequence[int] or int): the wires the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = AnyWires
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,)]

    def __init__(self, theta, wires=None, do_queue=True, id=None):
        wires = Wires(wires)
        self.hyperparameters["num_wires"] = len(wires)
        super().__init__(theta, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_matrix(theta, num_wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.MultiRZ.matrix`

        Args:
            theta (tensor_like or float): rotation angle
            num_wires (int): number of wires the rotation acts on

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.MultiRZ.compute_matrix(torch.tensor(0.1), 2)
        tensor([[0.9988-0.0500j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.9988+0.0500j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.9988+0.0500j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9988-0.0500j]])
        """
        eigs = qml.math.convert_like(pauli_eigs(num_wires), theta)

        if qml.math.get_interface(theta) == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)
            eigs = qml.math.cast_like(eigs, 1j)

        if qml.math.ndim(theta) == 0:
            return qml.math.diag(qml.math.exp(-0.5j * theta * eigs))

        diags = qml.math.exp(qml.math.outer(-0.5j * theta, eigs))
        return diags[:, :, np.newaxis] * qml.math.cast_like(
            qml.math.eye(2**num_wires, like=diags), diags
        )

    def generator(self):
        return -0.5 * functools.reduce(matmul, [qml.PauliZ(w) for w in self.wires])

    @staticmethod
    def compute_eigvals(theta, num_wires):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.MultiRZ.eigvals`


        Args:
            theta (tensor_like or float): rotation angle
            num_wires (int): number of wires the rotation acts on

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.MultiRZ.compute_eigvals(torch.tensor(0.5), 3)
        tensor([0.9689-0.2474j, 0.9689+0.2474j, 0.9689+0.2474j, 0.9689-0.2474j,
                0.9689+0.2474j, 0.9689-0.2474j, 0.9689-0.2474j, 0.9689+0.2474j])
        """
        eigs = qml.math.convert_like(pauli_eigs(num_wires), theta)

        if qml.math.get_interface(theta) == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)
            eigs = qml.math.cast_like(eigs, 1j)

        if qml.math.ndim(theta) == 0:
            return qml.math.exp(-0.5j * theta * eigs)

        return qml.math.exp(qml.math.outer(-0.5j * theta, eigs))

    @staticmethod
    def compute_decomposition(
        theta, wires, **kwargs
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.MultiRZ.decomposition`.

        Args:
            theta (float): rotation angle :math:`\theta`
            wires (Iterable, Wires): the wires the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.MultiRZ.compute_decomposition(1.2, wires=(0,1))
        [CNOT(wires=[1, 0]), RZ(1.2, wires=[0]), CNOT(wires=[1, 0])]

        """
        ops = [qml.CNOT(wires=(w0, w1)) for w0, w1 in zip(wires[~0:0:-1], wires[~1::-1])]
        ops.append(qml.RZ(theta, wires=wires[0]))
        ops += [qml.CNOT(wires=(w0, w1)) for w0, w1 in zip(wires[1:], wires[:~0])]

        return ops

    def adjoint(self):
        return MultiRZ(-self.parameters[0], wires=self.wires)

    def pow(self, z):
        return [MultiRZ(self.data[0] * z, wires=self.wires)]

    def simplify(self):
        theta = self.data[0] % (4 * np.pi)

        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires[0])

        return MultiRZ(theta, wires=self.wires)


class PauliRot(Operation):
    r"""
    Arbitrary Pauli word rotation.

    .. math::

        RP(\theta, P) = \exp(-i \frac{\theta}{2} P)

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
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)

    **Example**

    >>> dev = qml.device('default.qubit', wires=1)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.PauliRot(0.5, 'X',  wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> print(example_circuit())
    0.8775825618903724
    """
    num_wires = AnyWires
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    do_check_domain = False
    grad_method = "A"
    parameter_frequencies = [(1,)]

    _ALLOWED_CHARACTERS = "IXYZ"

    _PAULI_CONJUGATION_MATRICES = {
        "X": qml.Hadamard.compute_matrix(),
        "Y": qml.RX.compute_matrix(np.pi / 2),
        "Z": np.array([[1, 0], [0, 1]]),
    }

    def __init__(self, theta, pauli_word, wires=None, do_queue=True, id=None):
        super().__init__(theta, wires=wires, do_queue=do_queue, id=id)
        self.hyperparameters["pauli_word"] = pauli_word

        if not PauliRot._check_pauli_word(pauli_word):
            raise ValueError(
                f'The given Pauli word "{pauli_word}" contains characters that are not allowed.'
                " Allowed characters are I, X, Y and Z"
            )

        num_wires = 1 if isinstance(wires, int) else len(wires)

        if not len(pauli_word) == num_wires:
            raise ValueError(
                f"The given Pauli word has length {len(pauli_word)}, length "
                f"{num_wires} was expected for wires {wires}"
            )

    def __repr__(self):
        return f"PauliRot({self.data[0]}, {self.hyperparameters['pauli_word']}, wires={self.wires.tolist()})"

    def label(self, decimals=None, base_label=None, cache=None):
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
        'PauliRot\n(0.10)'

        """
        pauli_word = self.hyperparameters["pauli_word"]
        op_label = base_label or ("R" + pauli_word)

        # TODO[dwierichs]: Implement a proper label for parameter-broadcasted operators
        if decimals is not None and self.batch_size is None:
            param_string = f"\n({qml.math.asarray(self.parameters[0]):.{decimals}f})"
            op_label += param_string

        return op_label

    @staticmethod
    def _check_pauli_word(pauli_word):
        """Check that the given Pauli word has correct structure.

        Args:
            pauli_word (str): Pauli word to be checked

        Returns:
            bool: Whether the Pauli word has correct structure.
        """
        return all(pauli in PauliRot._ALLOWED_CHARACTERS for pauli in set(pauli_word))

    @staticmethod
    def compute_matrix(theta, pauli_word):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.PauliRot.matrix`


        Args:
            theta (tensor_like or float): rotation angle
            pauli_word (str): string representation of Pauli word

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.PauliRot.compute_matrix(0.5, 'X')
        [[9.6891e-01+4.9796e-18j 2.7357e-17-2.4740e-01j]
         [2.7357e-17-2.4740e-01j 9.6891e-01+4.9796e-18j]]
        """
        if not PauliRot._check_pauli_word(pauli_word):
            raise ValueError(
                f'The given Pauli word "{pauli_word}" contains characters that are not allowed.'
                " Allowed characters are I, X, Y and Z"
            )

        interface = qml.math.get_interface(theta)

        if interface == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)

        # Simplest case is if the Pauli is the identity matrix
        if set(pauli_word) == {"I"}:
            exp = qml.math.exp(-0.5j * theta)
            iden = qml.math.eye(2 ** len(pauli_word), like=theta)
            if qml.math.get_interface(theta) == "tensorflow":
                iden = qml.math.cast_like(iden, 1j)
            if qml.math.get_interface(theta) == "torch":
                td = exp.device
                iden = iden.to(td)

            if qml.math.ndim(theta) == 0:
                return exp * iden

            return qml.math.stack([e * iden for e in exp])

        # We first generate the matrix excluding the identity parts and expand it afterwards.
        # To this end, we have to store on which wires the non-identity parts act
        non_identity_wires, non_identity_gates = zip(
            *[(wire, gate) for wire, gate in enumerate(pauli_word) if gate != "I"]
        )

        multi_Z_rot_matrix = MultiRZ.compute_matrix(theta, len(non_identity_gates))

        # now we conjugate with Hadamard and RX to create the Pauli string
        conjugation_matrix = functools.reduce(
            qml.math.kron,
            [PauliRot._PAULI_CONJUGATION_MATRICES[gate] for gate in non_identity_gates],
        )
        if interface == "tensorflow":
            conjugation_matrix = qml.math.cast_like(conjugation_matrix, 1j)
        # Note: we use einsum with reverse arguments here because it is not multi-dispatched
        # and the tensordot containing multi_Z_rot_matrix should decide about the interface
        return expand_matrix(
            qml.math.einsum(
                "...jk,ij->...ik",
                qml.math.tensordot(multi_Z_rot_matrix, conjugation_matrix, axes=[[-1], [0]]),
                qml.math.conj(conjugation_matrix),
            ),
            non_identity_wires,
            list(range(len(pauli_word))),
        )

    def generator(self):
        pauli_word = self.hyperparameters["pauli_word"]
        wire_map = {w: i for i, w in enumerate(self.wires)}
        return -0.5 * qml.pauli.string_to_pauli_word(pauli_word, wire_map=wire_map)

    @staticmethod
    def compute_eigvals(theta, pauli_word):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.PauliRot.eigvals`


        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.PauliRot.compute_eigvals(torch.tensor(0.5), "X")
        tensor([0.9689-0.2474j, 0.9689+0.2474j])
        """
        if qml.math.get_interface(theta) == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)

        # Identity must be treated specially because its eigenvalues are all the same
        if set(pauli_word) == {"I"}:
            exp = qml.math.exp(-0.5j * theta)
            ones = qml.math.ones(2 ** len(pauli_word), like=theta)
            if qml.math.get_interface(theta) == "tensorflow":
                ones = qml.math.cast_like(ones, 1j)

            if qml.math.ndim(theta) == 0:
                return exp * ones

            return qml.math.tensordot(exp, ones, axes=0)

        return MultiRZ.compute_eigvals(theta, len(pauli_word))

    @staticmethod
    def compute_decomposition(theta, wires, pauli_word):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.PauliRot.decomposition`.

        Args:
            theta (float): rotation angle :math:`\theta`
            wires (Iterable, Wires): the wires the operation acts on
            pauli_word (string): the Pauli word defining the rotation

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.PauliRot.compute_decomposition(1.2, "XY", wires=(0,1))
        [Hadamard(wires=[0]),
        RX(1.5707963267948966, wires=[1]),
        MultiRZ(1.2, wires=[0, 1]),
        Hadamard(wires=[0]),
        RX(-1.5707963267948966, wires=[1])]

        """
        if isinstance(wires, int):  # Catch cases when the wire is passed as a single int.
            wires = [wires]

        # Check for identity and do nothing
        if set(pauli_word) == {"I"}:
            return []

        active_wires, active_gates = zip(
            *[(wire, gate) for wire, gate in zip(wires, pauli_word) if gate != "I"]
        )

        ops = []
        for wire, gate in zip(active_wires, active_gates):
            if gate == "X":
                ops.append(qml.Hadamard(wires=[wire]))
            elif gate == "Y":
                ops.append(qml.RX(np.pi / 2, wires=[wire]))

        ops.append(MultiRZ(theta, wires=list(active_wires)))

        for wire, gate in zip(active_wires, active_gates):
            if gate == "X":
                ops.append(qml.Hadamard(wires=[wire]))
            elif gate == "Y":
                ops.append(qml.RX(-np.pi / 2, wires=[wire]))
        return ops

    def adjoint(self):
        return PauliRot(-self.parameters[0], self.hyperparameters["pauli_word"], wires=self.wires)

    def pow(self, z):
        return [PauliRot(self.data[0] * z, self.hyperparameters["pauli_word"], wires=self.wires)]


class IsingXX(Operation):
    r"""
    Ising XX coupling gate

    .. math:: XX(\phi) = \exp(-i \frac{\phi}{2} (X \otimes X)) =
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
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self):
        return -0.5 * qml.PauliX(wires=self.wires[0]) @ qml.PauliX(wires=self.wires[1])

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.

        .. seealso:: :meth:`~.IsingXX.matrix`


        Args:
           phi (tensor_like or float): phase angle

        Returns:
           tensor_like: canonical matrix

        **Example**

        >>> qml.IsingXX.compute_matrix(torch.tensor(0.5))
        tensor([[0.9689+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000-0.2474j],
                [0.0000+0.0000j, 0.9689+0.0000j, 0.0000-0.2474j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000-0.2474j, 0.9689+0.0000j, 0.0000+0.0000j],
                [0.0000-0.2474j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9689+0.0000j]],
               dtype=torch.complex128)
        """
        c = qml.math.cos(phi / 2)
        s = qml.math.sin(phi / 2)

        eye = qml.math.eye(4, like=phi)
        rev_eye = qml.math.convert_like(np.eye(4)[::-1].copy(), phi)
        if qml.math.get_interface(phi) == "tensorflow":
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)
            eye = qml.math.cast_like(eye, 1j)
            rev_eye = qml.math.cast_like(rev_eye, 1j)

        # The following avoids casting an imaginary quantity to reals when backpropagating
        js = -1j * s
        if qml.math.ndim(phi) == 0:
            return c * eye + js * rev_eye

        return qml.math.tensordot(c, eye, axes=0) + qml.math.tensordot(js, rev_eye, axes=0)

    @staticmethod
    def compute_decomposition(phi, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.IsingXX.decomposition`.

        Args:
            phi (float): the phase angle
            wires (Iterable, Wires): the subsystem the gate acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.IsingXX.compute_decomposition(1.23, wires=(0,1))
        [CNOT(wires=[0, 1]), RX(1.23, wires=[0]), CNOT(wires=[0, 1]]

        """
        decomp_ops = [
            qml.CNOT(wires=wires),
            qml.RX(phi, wires=[wires[0]]),
            qml.CNOT(wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        (phi,) = self.parameters
        return IsingXX(-phi, wires=self.wires)

    def pow(self, z):
        return [IsingXX(self.data[0] * z, wires=self.wires)]

    def simplify(self):
        phi = self.data[0] % (4 * np.pi)

        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])

        return IsingXX(phi, wires=self.wires)


class IsingYY(Operation):
    r"""
    Ising YY coupling gate

    .. math:: \mathtt{YY}(\phi) = \exp(-i \frac{\phi}{2} (Y \otimes Y)) =
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
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self):
        return -0.5 * qml.PauliY(wires=self.wires[0]) @ qml.PauliY(wires=self.wires[1])

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(phi, wires):
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
            qml.RY(phi, wires=[wires[0]]),
            qml.CY(wires=wires),
        ]

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.IsingYY.matrix`


        Args:
           phi (tensor_like or float): phase angle

        Returns:
           tensor_like: canonical matrix

        **Example**

        >>> qml.IsingYY.compute_matrix(torch.tensor(0.5))
        tensor([[0.9689+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.2474j],
                [0.0000+0.0000j, 0.9689+0.0000j, 0.0000-0.2474j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000-0.2474j, 0.9689+0.0000j, 0.0000+0.0000j],
                [0.0000+0.2474j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9689+0.0000j]])
        """
        c = qml.math.cos(phi / 2)
        s = qml.math.sin(phi / 2)

        if qml.math.get_interface(phi) == "tensorflow":
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)

        js = 1j * s
        r_term = qml.math.cast_like(
            qml.math.array(
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
        if qml.math.ndim(phi) == 0:
            return c * qml.math.cast_like(qml.math.eye(4, like=c), c) + js * r_term

        return qml.math.tensordot(c, np.eye(4), axes=0) + qml.math.tensordot(js, r_term, axes=0)

    def adjoint(self):
        (phi,) = self.parameters
        return IsingYY(-phi, wires=self.wires)

    def pow(self, z):
        return [IsingYY(self.data[0] * z, wires=self.wires)]

    def simplify(self):
        phi = self.data[0] % (4 * np.pi)

        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])

        return IsingYY(phi, wires=self.wires)


class IsingZZ(Operation):
    r"""
    Ising ZZ coupling gate

    .. math:: ZZ(\phi) = \exp(-i \frac{\phi}{2} (Z \otimes Z)) =
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
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self):
        return -0.5 * qml.PauliZ(wires=self.wires[0]) @ qml.PauliZ(wires=self.wires[1])

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(phi, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.IsingZZ.decomposition`.

        Args:
            phi (float): the phase angle
            wires (Iterable, Wires): the subsystem the gate acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.IsingZZ.compute_decomposition(1.23, wires=[0,1])
        [CNOT(wires=[0, 1]), RZ(1.23, wires=[1]), CNOT(wires=[0, 1])]

        """
        return [
            qml.CNOT(wires=wires),
            qml.RZ(phi, wires=[wires[1]]),
            qml.CNOT(wires=wires),
        ]

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.IsingZZ.matrix`


        Args:
           phi (tensor_like or float): phase angle

        Returns:
           tensor_like: canonical matrix

        **Example**

        >>> qml.IsingZZ.compute_matrix(torch.tensor(0.5))
        tensor([[0.9689-0.2474j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.9689+0.2474j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.9689+0.2474j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9689-0.2474j]])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            p = qml.math.exp(-0.5j * qml.math.cast_like(phi, 1j))
            if qml.math.ndim(p) == 0:
                return qml.math.diag([p, qml.math.conj(p), qml.math.conj(p), p])

            diags = stack_last([p, qml.math.conj(p), qml.math.conj(p), p])
            return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)

        signs = qml.math.array([1, -1, -1, 1], like=phi)
        arg = -0.5j * phi

        if qml.math.ndim(arg) == 0:
            return qml.math.diag(qml.math.exp(arg * signs))

        diags = qml.math.exp(qml.math.outer(arg, signs))
        return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)

    @staticmethod
    def compute_eigvals(phi):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.IsingZZ.eigvals`


        Args:
            phi (tensor_like or float): phase angle

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.IsingZZ.compute_eigvals(torch.tensor(0.5))
        tensor([0.9689-0.2474j, 0.9689+0.2474j, 0.9689+0.2474j, 0.9689-0.2474j])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            phase = qml.math.exp(-0.5j * qml.math.cast_like(phi, 1j))
            return stack_last([phase, qml.math.conj(phase), qml.math.conj(phase), phase])

        prefactors = qml.math.array([-0.5j, 0.5j, 0.5j, -0.5j], like=phi)
        if qml.math.ndim(phi) == 0:
            product = phi * prefactors
        else:
            product = qml.math.outer(phi, prefactors)
        return qml.math.exp(product)

    def adjoint(self):
        (phi,) = self.parameters
        return IsingZZ(-phi, wires=self.wires)

    def pow(self, z):
        return [IsingZZ(self.data[0] * z, wires=self.wires)]

    def simplify(self):
        phi = self.data[0] % (4 * np.pi)

        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])

        return IsingZZ(phi, wires=self.wires)


class IsingXY(Operation):
    r"""
    Ising (XX + YY) coupling gate

    .. math:: \mathtt{XY}(\phi) = \exp(i \frac{\theta}{4} (X \otimes X + Y \otimes Y)) =
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
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(0.5, 1.0)]

    def generator(self):
        return 0.25 * (
            qml.PauliX(wires=self.wires[0]) @ qml.PauliX(wires=self.wires[1])
            + qml.PauliY(wires=self.wires[0]) @ qml.PauliY(wires=self.wires[1])
        )

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(phi, wires):
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
        [Hadamard(wires=[0]), CY(wires=[0, 1]), RY(0.615, wires=[0]), RX(-0.615, wires=[1]), CY(wires=[0, 1]), Hadamard(wires=[0])]

        """
        return [
            qml.Hadamard(wires=[wires[0]]),
            qml.CY(wires=wires),
            qml.RY(phi / 2, wires=[wires[0]]),
            qml.RX(-phi / 2, wires=[wires[1]]),
            qml.CY(wires=wires),
            qml.Hadamard(wires=[wires[0]]),
        ]

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.IsingXY.matrix`


        Args:
           phi (tensor_like or float): phase angle

        Returns:
           tensor_like: canonical matrix

        **Example**

        >>> qml.IsingXY.compute_matrix(0.5)
        array([[1.        +0.j        , 0.        +0.j        ,        0.        +0.j        , 0.        +0.j        ],
               [0.        +0.j        , 0.96891242+0.j        ,        0.        +0.24740396j, 0.        +0.j        ],
               [0.        +0.j        , 0.        +0.24740396j,        0.96891242+0.j        , 0.        +0.j        ],
               [0.        +0.j        , 0.        +0.j        ,        0.        +0.j        , 1.        +0.j        ]])
        """
        c = qml.math.cos(phi / 2)
        s = qml.math.sin(phi / 2)

        if qml.math.get_interface(phi) == "tensorflow":
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)

        js = 1j * s
        off_diag = qml.math.cast_like(
            qml.math.array(
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
        if qml.math.ndim(phi) == 0:
            return qml.math.diag([1, c, c, 1]) + js * off_diag

        ones = qml.math.ones_like(c)
        diags = stack_last([ones, c, c, ones])[:, :, np.newaxis]
        return diags * np.eye(4) + qml.math.tensordot(js, off_diag, axes=0)

    @staticmethod
    def compute_eigvals(phi):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.IsingXY.eigvals`


        Args:
            phi (tensor_like or float): phase angle

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.IsingXY.compute_eigvals(0.5)
        array([0.96891242+0.24740396j, 0.96891242-0.24740396j,       1.        +0.j        , 1.        +0.j        ])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        signs = np.array([1, -1, 0, 0])
        if qml.math.ndim(phi) == 0:
            return qml.math.exp(0.5j * phi * signs)

        return qml.math.exp(qml.math.tensordot(0.5j * phi, signs, axes=0))

    def adjoint(self):
        (phi,) = self.parameters
        return IsingXY(-phi, wires=self.wires)

    def pow(self, z):
        return [IsingXY(self.data[0] * z, wires=self.wires)]

    def simplify(self):
        phi = self.data[0] % (4 * np.pi)

        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])

        return IsingXY(phi, wires=self.wires)


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
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    grad_method = "A"
    grad_recipe = ([[0.5, 1, np.pi / 2], [-0.5, 1, -np.pi / 2]],)

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(phi, wires):
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
            qml.PhaseShift(phi, wires=[wires[1]]),
            qml.CNOT(wires=wires),
        ]

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.PSWAP.matrix`


        Args:
           phi (tensor_like or float): phase angle

        Returns:
           tensor_like: canonical matrix

        **Example**

        >>> qml.PSWAP.compute_matrix(0.5)
        array([[1.        +0.j, 0.        +0.j        , 0.        +0.j        , 0.        +0.j],
              [0.        +0.j, 0.        +0.j        , 0.87758256+0.47942554j, 0.        +0.j],
              [0.        +0.j, 0.87758256+0.47942554j, 0.        +0.j        , 0.        +0.j],
              [0.        +0.j, 0.        +0.j        , 0.        +0.j        , 1.        +0.j]])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        e = qml.math.exp(1j * phi)

        return qml.math.stack(
            [
                stack_last([1, 0, 0, 0]),
                stack_last([0, 0, e, 0]),
                stack_last([0, e, 0, 0]),
                stack_last([0, 0, 0, 1]),
            ],
            axis=-2,
        )

    @staticmethod
    def compute_eigvals(phi):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.PSWAP.eigvals`


        Args:
            phi (tensor_like or float): phase angle

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.PSWAP.compute_eigvals(0.5)
        array([ 1.        +0.j        ,  1.        +0.j,       -0.87758256-0.47942554j,  0.87758256+0.47942554j])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        return qml.math.stack([1, 1, -qml.math.exp(1j * phi), qml.math.exp(1j * phi)])

    def adjoint(self):
        (phi,) = self.parameters
        return PSWAP(-phi, wires=self.wires)

    def simplify(self):
        phi = self.data[0] % (2 * np.pi)

        if _can_replace(phi, 0):
            return qml.SWAP(wires=self.wires)

        return PSWAP(phi, wires=self.wires)
