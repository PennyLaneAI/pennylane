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
import numpy as np

import pennylane as qml
from pennylane.operation import Operation
from .non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ

stack_last = functools.partial(qml.math.stack, axis=-1)


def _can_replace(x, y):
    """
    Convenience function that returns true if x is close to y and if
    x does not require grad
    """
    return not qml.math.is_abstract(x) and not qml.math.requires_grad(x) and qml.math.allclose(x, y)


class RX(Operation):
    r"""
    The single qubit X rotation

    .. math:: R_x(\phi) = e^{-i\phi\sigma_x/2} = \begin{bmatrix}
                \cos(\phi/2) & -i\sin(\phi/2) \\
                -i\sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R_x(\phi)) = \frac{1}{2}\left[f(R_x(\phi+\pi/2)) - f(R_x(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`R_x(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "X"
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self):
        return qml.Hamiltonian([-0.5], [PauliX(wires=self.wires)])

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(theta):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.RX.matrix`

        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.RX.compute_matrix(torch.tensor(0.5))
        tensor([[0.9689+0.0000j, 0.0000-0.2474j],
                [0.0000-0.2474j, 0.9689+0.0000j]])
        """
        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        if qml.math.get_interface(theta) == "tensorflow":
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)

        # The following avoids casting an imaginary quantity to reals when backpropagating
        c = (1 + 0j) * c
        js = -1j * s
        return qml.math.stack([stack_last([c, js]), stack_last([js, c])], axis=-2)

    def adjoint(self):
        return RX(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [RX(self.data[0] * z, wires=self.wires)]

    def _controlled(self, wire):
        return qml.CRX(*self.parameters, wires=wire + self.wires)

    def simplify(self):
        theta = self.data[0] % (4 * np.pi)

        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires)

        return RX(theta, wires=self.wires)

    def single_qubit_rot_angles(self):
        # RX(\theta) = RZ(-\pi/2) RY(\theta) RZ(\pi/2)
        pi_half = qml.math.ones_like(self.data[0]) * (np.pi / 2)
        return [pi_half, self.data[0], -pi_half]


class RY(Operation):
    r"""
    The single qubit Y rotation

    .. math:: R_y(\phi) = e^{-i\phi\sigma_y/2} = \begin{bmatrix}
                \cos(\phi/2) & -\sin(\phi/2) \\
                \sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R_y(\phi)) = \frac{1}{2}\left[f(R_y(\phi+\pi/2)) - f(R_y(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`R_y(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "Y"
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self):
        return qml.Hamiltonian([-0.5], [PauliY(wires=self.wires)])

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(theta):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.RY.matrix`


        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.RY.compute_matrix(torch.tensor(0.5))
        tensor([[ 0.9689, -0.2474],
                [ 0.2474,  0.9689]])
        """

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)
        if qml.math.get_interface(theta) == "tensorflow":
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)
        # The following avoids casting an imaginary quantity to reals when backpropagating
        c = (1 + 0j) * c
        s = (1 + 0j) * s
        return qml.math.stack([stack_last([c, -s]), stack_last([s, c])], axis=-2)

    def adjoint(self):
        return RY(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [RY(self.data[0] * z, wires=self.wires)]

    def _controlled(self, wire):
        return qml.CRY(*self.parameters, wires=wire + self.wires)

    def simplify(self):
        theta = self.data[0] % (4 * np.pi)

        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires)

        return RY(theta, wires=self.wires)

    def single_qubit_rot_angles(self):
        # RY(\theta) = RZ(0) RY(\theta) RZ(0)
        return [0.0, self.data[0], 0.0]


class RZ(Operation):
    r"""
    The single qubit Z rotation

    .. math:: R_z(\phi) = e^{-i\phi\sigma_z/2} = \begin{bmatrix}
                e^{-i\phi/2} & 0 \\
                0 & e^{i\phi/2}
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R_z(\phi)) = \frac{1}{2}\left[f(R_z(\phi+\pi/2)) - f(R_z(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`R_z(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "Z"
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self):
        return qml.Hamiltonian([-0.5], [PauliZ(wires=self.wires)])

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(theta):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.RZ.matrix`

        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.RZ.compute_matrix(torch.tensor(0.5))
        tensor([[0.9689-0.2474j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.9689+0.2474j]])
        """
        if qml.math.get_interface(theta) == "tensorflow":
            p = qml.math.exp(-0.5j * qml.math.cast_like(theta, 1j))
            z = qml.math.zeros_like(p)

            return qml.math.stack([stack_last([p, z]), stack_last([z, qml.math.conj(p)])], axis=-2)

        signs = qml.math.array([-1, 1], like=theta)
        arg = 0.5j * theta

        if qml.math.ndim(arg) == 0:
            return qml.math.diag(qml.math.exp(arg * signs))

        diags = qml.math.exp(qml.math.outer(arg, signs))
        return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(2, like=diags), diags)

    @staticmethod
    def compute_eigvals(theta):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.RZ.eigvals`


        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.RZ.compute_eigvals(torch.tensor(0.5))
        tensor([0.9689-0.2474j, 0.9689+0.2474j])
        """
        if qml.math.get_interface(theta) == "tensorflow":
            phase = qml.math.exp(-0.5j * qml.math.cast_like(theta, 1j))
            return qml.math.stack([phase, qml.math.conj(phase)], axis=-1)

        prefactors = qml.math.array([-0.5j, 0.5j], like=theta)
        if qml.math.ndim(theta) == 0:
            product = theta * prefactors
        else:
            product = qml.math.outer(theta, prefactors)
        return qml.math.exp(product)

    def adjoint(self):
        return RZ(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [RZ(self.data[0] * z, wires=self.wires)]

    def _controlled(self, wire):
        return qml.CRZ(*self.parameters, wires=wire + self.wires)

    def simplify(self):
        theta = self.data[0] % (4 * np.pi)

        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires)

        return RZ(theta, wires=self.wires)

    def single_qubit_rot_angles(self):
        # RZ(\theta) = RZ(\theta) RY(0) RZ(0)
        return [self.data[0], 0.0, 0.0]


class PhaseShift(Operation):
    r"""
    Arbitrary single qubit local phase shift

    .. math:: R_\phi(\phi) = e^{i\phi/2}R_z(\phi) = \begin{bmatrix}
                1 & 0 \\
                0 & e^{i\phi}
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R_\phi(\phi)) = \frac{1}{2}\left[f(R_\phi(\phi+\pi/2)) - f(R_\phi(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`R_{\phi}(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "Z"
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self):
        return qml.Projector(np.array([1]), wires=self.wires)

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "Rϕ", cache=cache)

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.PhaseShift.matrix`


        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.PhaseShift.compute_matrix(torch.tensor(0.5))
        tensor([[0.9689-0.2474j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.9689+0.2474j]])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            p = qml.math.exp(1j * qml.math.cast_like(phi, 1j))
            ones = qml.math.ones_like(p)
            zeros = qml.math.zeros_like(p)

            return qml.math.stack([stack_last([ones, zeros]), stack_last([zeros, p])], axis=-2)

        signs = qml.math.array([0, 1], like=phi)
        arg = 1j * phi

        if qml.math.ndim(arg) == 0:
            return qml.math.diag(qml.math.exp(arg * signs))

        diags = qml.math.exp(qml.math.outer(arg, signs))
        return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(2, like=diags), diags)

    @staticmethod
    def compute_eigvals(phi):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.PhaseShift.eigvals`


        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.PhaseShift.compute_eigvals(torch.tensor(0.5))
        tensor([1.0000+0.0000j, 0.8776+0.4794j])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            phase = qml.math.exp(1j * qml.math.cast_like(phi, 1j))
            return stack_last([qml.math.ones_like(phase), phase])

        prefactors = qml.math.array([0, 1j], like=phi)
        if qml.math.ndim(phi) == 0:
            product = phi * prefactors
        else:
            product = qml.math.outer(phi, prefactors)
        return qml.math.exp(product)

    @staticmethod
    def compute_decomposition(phi, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.PhaseShift.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Any, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.PhaseShift.compute_decomposition(1.234, wires=0)
        [RZ(1.234, wires=[0]), GlobalPhase(-0.617, wires=[])]

        """
        return [RZ(phi, wires=wires), qml.GlobalPhase(-phi / 2)]

    def adjoint(self):
        return PhaseShift(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [PhaseShift(self.data[0] * z, wires=self.wires)]

    def _controlled(self, wire):
        return qml.ControlledPhaseShift(*self.parameters, wires=wire + self.wires)

    def simplify(self):
        phi = self.data[0] % (2 * np.pi)

        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires)

        return PhaseShift(phi, wires=self.wires)

    def single_qubit_rot_angles(self):
        # PhaseShift(\theta) = RZ(\theta) RY(0) RZ(0)
        return [self.data[0], 0.0, 0.0]


class Rot(Operation):
    r"""
    Arbitrary single qubit rotation

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 3
    * Number of dimensions per parameter: (0, 0, 0)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R(\phi, \theta, \omega)) = \frac{1}{2}\left[f(R(\phi+\pi/2, \theta, \omega)) - f(R(\phi-\pi/2, \theta, \omega))\right]`
      where :math:`f` is an expectation value depending on :math:`R(\phi, \theta, \omega)`.
      This gradient recipe applies for each angle argument :math:`\{\phi, \theta, \omega\}`.

    .. note::

        If the ``Rot`` gate is not supported on the targeted device, PennyLane
        will attempt to decompose the gate into :class:`~.RZ` and :class:`~.RY` gates.

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        omega (float): rotation angle :math:`\omega`
        wires (Any, Wires): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 1
    num_params = 3
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0, 0, 0)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,), (1,), (1,)]

    def __init__(self, phi, theta, omega, wires, id=None):
        super().__init__(phi, theta, omega, wires=wires, id=id)

    @staticmethod
    def compute_matrix(phi, theta, omega):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Rot.matrix`


        Args:
            phi (tensor_like or float): first rotation angle
            theta (tensor_like or float): second rotation angle
            omega (tensor_like or float): third rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.Rot.compute_matrix(torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3))
        tensor([[ 0.9752-0.1977j, -0.0993+0.0100j],
                [ 0.0993+0.0100j,  0.9752+0.1977j]])

        """
        # It might be that they are in different interfaces, e.g.,
        # Rot(0.2, 0.3, tf.Variable(0.5), wires=0)
        # So we need to make sure the matrix comes out having the right type
        interface = qml.math.get_interface(phi, theta, omega)

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        # If anything is not tensorflow, it has to be casted and then
        if interface == "tensorflow":
            phi = qml.math.cast_like(qml.math.asarray(phi, like=interface), 1j)
            omega = qml.math.cast_like(qml.math.asarray(omega, like=interface), 1j)
            c = qml.math.cast_like(qml.math.asarray(c, like=interface), 1j)
            s = qml.math.cast_like(qml.math.asarray(s, like=interface), 1j)

        # The following variable is used to assert the all terms to be stacked have same shape
        one = qml.math.ones_like(phi) * qml.math.ones_like(omega)
        c = c * one
        s = s * one

        mat = [
            [
                qml.math.exp(-0.5j * (phi + omega)) * c,
                -qml.math.exp(0.5j * (phi - omega)) * s,
            ],
            [
                qml.math.exp(-0.5j * (phi - omega)) * s,
                qml.math.exp(0.5j * (phi + omega)) * c,
            ],
        ]

        return qml.math.stack([stack_last(row) for row in mat], axis=-2)

    @staticmethod
    def compute_decomposition(phi, theta, omega, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.Rot.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            theta (float): rotation angle :math:`\theta`
            omega (float): rotation angle :math:`\omega`
            wires (Any, Wires): the wire the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.Rot.compute_decomposition(1.2, 2.3, 3.4, wires=0)
        [RZ(1.2, wires=[0]), RY(2.3, wires=[0]), RZ(3.4, wires=[0])]

        """
        return [
            RZ(phi, wires=wires),
            RY(theta, wires=wires),
            RZ(omega, wires=wires),
        ]

    def adjoint(self):
        phi, theta, omega = self.parameters
        return Rot(-omega, -theta, -phi, wires=self.wires)

    def _controlled(self, wire):
        return qml.CRot(*self.parameters, wires=wire + self.wires)

    def single_qubit_rot_angles(self):
        return self.data

    def simplify(self):
        """Simplifies into single-rotation gates or a Hadamard if possible.

        >>> qml.Rot(np.pi / 2, 0.1, -np.pi / 2, wires=0).simplify()
        RX(0.1, wires=[0])
        >>> qml.Rot(np.pi, np.pi/2, 0, 0).simplify()
        Hadamard(wires=[0])

        """
        p0, p1, p2 = [p % (4 * np.pi) for p in self.data]

        if _can_replace(p0, 0) and _can_replace(p1, 0) and _can_replace(p2, 0):
            return qml.Identity(wires=self.wires)
        if _can_replace(p0, np.pi / 2) and _can_replace(p2, 7 * np.pi / 2):
            return RX(p1, wires=self.wires)
        if _can_replace(p0, 0) and _can_replace(p2, 0):
            return RY(p1, wires=self.wires)
        if _can_replace(p1, 0):
            return RZ((p0 + p2) % (4 * np.pi), wires=self.wires)
        if _can_replace(p0, np.pi) and _can_replace(p1, np.pi / 2) and _can_replace(p2, 0):
            return Hadamard(wires=self.wires)

        return Rot(p0, p1, p2, wires=self.wires)


class U1(Operation):
    r"""
    U1 gate.

    .. math:: U_1(\phi) = e^{i\phi/2}R_z(\phi) = \begin{bmatrix}
                1 & 0 \\
                0 & e^{i\phi}
            \end{bmatrix}.

    .. note::

        The ``U1`` gate is an alias for the phase shift operation :class:`~.PhaseShift`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_1(\phi)) = \frac{1}{2}\left[f(U_1(\phi+\pi/2)) - f(U_1(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`U_1(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self):
        return qml.Projector(np.array([1]), wires=self.wires)

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.U1.matrix`

        Args:
            phi (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.U1.compute_matrix(torch.tensor(0.5))
        tensor([[1.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.8776+0.4794j]])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)
            fac = qml.math.cast_like([0, 1], 1j)
        else:
            fac = np.array([0, 1])

        fac = qml.math.convert_like(fac, phi)

        arg = 1j * phi
        if qml.math.ndim(arg) == 0:
            return qml.math.diag(qml.math.exp(arg * fac))

        diags = qml.math.exp(qml.math.outer(arg, fac))
        return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(2, like=diags), diags)

    @staticmethod
    def compute_decomposition(phi, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.U1.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Any, Wires): Wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.U1.compute_decomposition(1.234, wires=0)
        [PhaseShift(1.234, wires=[0])]

        """
        return [PhaseShift(phi, wires=wires)]

    def adjoint(self):
        return U1(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [U1(self.data[0] * z, wires=self.wires)]

    def simplify(self):
        phi = self.data[0] % (2 * np.pi)

        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires)

        return U1(phi, wires=self.wires)


class U2(Operation):
    r"""
    U2 gate.

    .. math::

        U_2(\phi, \delta) = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & -\exp(i \delta)
        \\ \exp(i \phi) & \exp(i (\phi + \delta)) \end{bmatrix}

    The :math:`U_2` gate is related to the single-qubit rotation :math:`R` (:class:`Rot`) and the
    :math:`R_\phi` (:class:`PhaseShift`) gates via the following relation:

    .. math::

        U_2(\phi, \delta) = R_\phi(\phi+\delta) R(\delta,\pi/2,-\delta)

    .. note::

        If the ``U2`` gate is not supported on the targeted device, PennyLane
        will attempt to decompose the gate into :class:`~.Rot` and :class:`~.PhaseShift` gates.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2
    * Number of dimensions per parameter: (0, 0)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_2(\phi, \delta)) = \frac{1}{2}\left[f(U_2(\phi+\pi/2, \delta)) - f(U_2(\phi-\pi/2, \delta))\right]`
      where :math:`f` is an expectation value depending on :math:`U_2(\phi, \delta)`.
      This gradient recipe applies for each angle argument :math:`\{\phi, \delta\}`.

    Args:
        phi (float): azimuthal angle :math:`\phi`
        delta (float): quantum phase :math:`\delta`
        wires (Sequence[int] or int): the subsystem the gate acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 1
    num_params = 2
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0, 0)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,), (1,)]

    def __init__(self, phi, delta, wires, id=None):
        super().__init__(phi, delta, wires=wires, id=id)

    @staticmethod
    def compute_matrix(phi, delta):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.U2.matrix`

        Args:
            phi (tensor_like or float): azimuthal angle
            delta (tensor_like or float): quantum phase

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.U2.compute_matrix(torch.tensor(0.1), torch.tensor(0.2))
        tensor([[ 0.7071+0.0000j, -0.6930-0.1405j],
                [ 0.7036+0.0706j,  0.6755+0.2090j]])
        """
        interface = qml.math.get_interface(phi, delta)

        # If anything is not tensorflow, it has to be casted and then
        if interface == "tensorflow":
            phi = qml.math.cast_like(qml.math.asarray(phi, like=interface), 1j)
            delta = qml.math.cast_like(qml.math.asarray(delta, like=interface), 1j)

        one = qml.math.ones_like(phi) * qml.math.ones_like(delta)
        mat = [
            [one, -qml.math.exp(1j * delta) * one],
            [qml.math.exp(1j * phi) * one, qml.math.exp(1j * (phi + delta))],
        ]

        return qml.math.stack([stack_last(row) for row in mat], axis=-2) / np.sqrt(2)

    @staticmethod
    def compute_decomposition(phi, delta, wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.U2.decomposition`.

        Args:
            phi (float): azimuthal angle :math:`\phi`
            delta (float): quantum phase :math:`\delta`
            wires (Iterable, Wires): the subsystem the gate acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.U2.compute_decomposition(1.23, 2.34, wires=0)
        [Rot(2.34, 1.5707963267948966, -2.34, wires=[0]),
        PhaseShift(2.34, wires=[0]),
        PhaseShift(1.23, wires=[0])]

        """
        pi_half = qml.math.ones_like(delta) * (np.pi / 2)
        return [
            Rot(delta, pi_half, -delta, wires=wires),
            PhaseShift(delta, wires=wires),
            PhaseShift(phi, wires=wires),
        ]

    def adjoint(self):
        phi, delta = self.parameters
        new_delta = qml.math.mod((np.pi - phi), (2 * np.pi))
        new_phi = qml.math.mod((np.pi - delta), (2 * np.pi))
        return U2(new_phi, new_delta, wires=self.wires)

    def simplify(self):
        """Simplifies the gate into RX or RY gates if possible."""
        wires = self.wires

        phi, delta = [p % (2 * np.pi) for p in self.data]

        if _can_replace(delta, 0) and _can_replace(phi, 0):
            return RY(np.pi / 2, wires=wires)
        if _can_replace(delta, np.pi / 2) and _can_replace(phi, 3 * np.pi / 2):
            return RX(np.pi / 2, wires=wires)
        if _can_replace(delta, 3 * np.pi / 2) and _can_replace(phi, np.pi / 2):
            return RX(3 * np.pi / 2, wires=wires)

        return U2(phi, delta, wires=wires)


class U3(Operation):
    r"""
    Arbitrary single qubit unitary.

    .. math::

        U_3(\theta, \phi, \delta) = \begin{bmatrix} \cos(\theta/2) & -\exp(i \delta)\sin(\theta/2) \\
        \exp(i \phi)\sin(\theta/2) & \exp(i (\phi + \delta))\cos(\theta/2) \end{bmatrix}

    The :math:`U_3` gate is related to the single-qubit rotation :math:`R` (:class:`Rot`) and the
    :math:`R_\phi` (:class:`PhaseShift`) gates via the following relation:

    .. math::

        U_3(\theta, \phi, \delta) = R_\phi(\phi+\delta) R(\delta,\theta,-\delta)

    .. note::

        If the ``U3`` gate is not supported on the targeted device, PennyLane
        will attempt to decompose the gate into :class:`~.PhaseShift` and :class:`~.Rot` gates.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 3
    * Number of dimensions per parameter: (0, 0, 0)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_3(\theta, \phi, \delta)) = \frac{1}{2}\left[f(U_3(\theta+\pi/2, \phi, \delta)) - f(U_3(\theta-\pi/2, \phi, \delta))\right]`
      where :math:`f` is an expectation value depending on :math:`U_3(\theta, \phi, \delta)`.
      This gradient recipe applies for each angle argument :math:`\{\theta, \phi, \delta\}`.

    Args:
        theta (float): polar angle :math:`\theta`
        phi (float): azimuthal angle :math:`\phi`
        delta (float): quantum phase :math:`\delta`
        wires (Sequence[int] or int): the subsystem the gate acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 1
    num_params = 3
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0, 0, 0)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,), (1,), (1,)]

    def __init__(self, theta, phi, delta, wires, id=None):
        super().__init__(theta, phi, delta, wires=wires, id=id)

    @staticmethod
    def compute_matrix(theta, phi, delta):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.U3.matrix`

        Args:
            theta (tensor_like or float): polar angle
            phi (tensor_like or float): azimuthal angle
            delta (tensor_like or float): quantum phase

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.U3.compute_matrix(torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3))
        tensor([[ 0.9988+0.0000j, -0.0477-0.0148j],
                [ 0.0490+0.0099j,  0.8765+0.4788j]])

        """
        # It might be that they are in different interfaces, e.g.,
        # U3(0.2, 0.3, tf.Variable(0.5), wires=0)
        # So we need to make sure the matrix comes out having the right type
        interface = qml.math.get_interface(theta, phi, delta)

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        # If anything is not tensorflow, it has to be casted and then
        if interface == "tensorflow":
            phi = qml.math.cast_like(qml.math.asarray(phi, like=interface), 1j)
            delta = qml.math.cast_like(qml.math.asarray(delta, like=interface), 1j)
            c = qml.math.cast_like(qml.math.asarray(c, like=interface), 1j)
            s = qml.math.cast_like(qml.math.asarray(s, like=interface), 1j)

        # The following variable is used to assert the all terms to be stacked have same shape
        one = qml.math.ones_like(phi) * qml.math.ones_like(delta)
        c = c * one
        s = s * one

        mat = [
            [c, -s * qml.math.exp(1j * delta)],
            [s * qml.math.exp(1j * phi), c * qml.math.exp(1j * (phi + delta))],
        ]

        return qml.math.stack([stack_last(row) for row in mat], axis=-2)

    @staticmethod
    def compute_decomposition(theta, phi, delta, wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.U3.decomposition`.

        Args:
            theta (float): polar angle :math:`\theta`
            phi (float): azimuthal angle :math:`\phi`
            delta (float): quantum phase :math:`\delta`
            wires (Iterable, Wires): the subsystem the gate acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.U3.compute_decomposition(1.23, 2.34, 3.45, wires=0)
        [Rot(3.45, 1.23, -3.45, wires=[0]),
        PhaseShift(3.45, wires=[0]),
        PhaseShift(2.34, wires=[0])]

        """
        return [
            Rot(delta, theta, -delta, wires=wires),
            PhaseShift(delta, wires=wires),
            PhaseShift(phi, wires=wires),
        ]

    def adjoint(self):
        theta, phi, delta = self.parameters
        new_delta = qml.math.mod((np.pi - phi), (2 * np.pi))
        new_phi = qml.math.mod((np.pi - delta), (2 * np.pi))
        return U3(theta, new_phi, new_delta, wires=self.wires)

    def simplify(self):
        """Simplifies into :class:`~.RX`, :class:`~.RY`, or :class:`~.PhaseShift` gates
        if possible.

        >>> qml.U3(0.1, 0, 0, wires=0).simplify()
        RY(0.1, wires=[0])

        """
        wires = self.wires
        params = self.parameters

        p0 = params[0] % (4 * np.pi)
        p1, p2 = [p % (2 * np.pi) for p in params[1:]]

        if _can_replace(p0, 0) and _can_replace(p1, 0) and _can_replace(p2, 0):
            return qml.Identity(wires=wires)
        if _can_replace(p0, 0) and not _can_replace(p1, 0) and _can_replace(p2, 0):
            return PhaseShift(p1, wires=wires)
        if (
            _can_replace(p2, np.pi / 2)
            and _can_replace(p1, 3 * np.pi / 2)
            and not _can_replace(p0, 0)
        ):
            return RX(p0, wires=wires)
        if not _can_replace(p0, 0) and _can_replace(p1, 0) and _can_replace(p2, 0):
            return RY(p0, wires=wires)

        return U3(p0, p1, p2, wires=wires)
