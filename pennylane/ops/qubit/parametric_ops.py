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
# pylint: disable=too-many-arguments
"""
This submodule contains the discrete-variable quantum operations that are the
core parameterized gates.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access,invalid-overridden-method
import functools
import math
from operator import matmul

import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, Operation, expand_matrix
from pennylane.ops.qubit.non_parametric_ops import PauliX, PauliY, PauliZ, Hadamard
from pennylane.utils import pauli_eigs
from pennylane.wires import Wires

INV_SQRT2 = 1 / math.sqrt(2)

stack_last = functools.partial(qml.math.stack, axis=-1)


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
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
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
        return -0.5 * PauliX(wires=self.wires)

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

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
        CRX(*self.parameters, wires=wire + self.wires)

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
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
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
        return -0.5 * PauliY(wires=self.wires)

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

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
        CRY(*self.parameters, wires=wire + self.wires)

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
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
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
        return -0.5 * PauliZ(wires=self.wires)

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

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

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
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
        CRZ(*self.parameters, wires=wire + self.wires)

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
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
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

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

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

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
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
        [RZ(1.234, wires=[0])]

        """
        return [RZ(phi, wires=wires)]

    def adjoint(self):
        return PhaseShift(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [PhaseShift(self.data[0] * z, wires=self.wires)]

    def _controlled(self, wire):
        ControlledPhaseShift(*self.parameters, wires=wire + self.wires)

    def single_qubit_rot_angles(self):
        # PhaseShift(\theta) = RZ(\theta) RY(0) RZ(0)
        return [self.data[0], 0.0, 0.0]


class ControlledPhaseShift(Operation):
    r"""
    A qubit controlled phase shift.

    .. math:: CR_\phi(\phi) = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i\phi}
            \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(CR_\phi(\phi)) = \frac{1}{2}\left[f(CR_\phi(\phi+\pi/2)) - f(CR_\phi(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`CR_{\phi}(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "Z"
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self):
        return qml.Projector(np.array([1, 1]), wires=self.wires)

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "Rϕ", cache=cache)

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.ControlledPhaseShift.matrix`

        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.ControlledPhaseShift.compute_matrix(torch.tensor(0.5))
            tensor([[1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0000+0.0000j],
                    [0.0+0.0j, 1.0+0.0j, 0.0+0.0j, 0.0000+0.0000j],
                    [0.0+0.0j, 0.0+0.0j, 1.0+0.0j, 0.0000+0.0000j],
                    [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.8776+0.4794j]])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            p = qml.math.exp(1j * qml.math.cast_like(phi, 1j))
            if qml.math.ndim(p) == 0:
                return qml.math.diag([1, 1, 1, p])

            ones = qml.math.ones_like(p)
            diags = stack_last([ones, ones, ones, p])
            return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)

        signs = qml.math.array([0, 0, 0, 1], like=phi)
        arg = 1j * phi

        if qml.math.ndim(arg) == 0:
            return qml.math.diag(qml.math.exp(arg * signs))

        diags = qml.math.exp(qml.math.outer(arg, signs))
        return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)

    @staticmethod
    def compute_eigvals(phi):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.ControlledPhaseShift.eigvals`


        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.ControlledPhaseShift.compute_eigvals(torch.tensor(0.5))
        tensor([1.0000+0.0000j, 1.0000+0.0000j, 1.0000+0.0000j, 0.8776+0.4794j])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            phase = qml.math.exp(1j * qml.math.cast_like(phi, 1j))
            ones = qml.math.ones_like(phase)
            return stack_last([ones, ones, ones, phase])

        prefactors = qml.math.array([0, 0, 0, 1j], like=phi)
        if qml.math.ndim(phi) == 0:
            product = phi * prefactors
        else:
            product = qml.math.outer(phi, prefactors)
        return qml.math.exp(product)

    @staticmethod
    def compute_decomposition(phi, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.ControlledPhaseShift.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.ControlledPhaseShift.compute_decomposition(1.234, wires=(0,1))
        [PhaseShift(0.617, wires=[0]),
        CNOT(wires=[0, 1]),
        PhaseShift(-0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        PhaseShift(0.617, wires=[1])]

        """
        decomp_ops = [
            qml.PhaseShift(phi / 2, wires=wires[0]),
            qml.CNOT(wires=wires),
            qml.PhaseShift(-phi / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            qml.PhaseShift(phi / 2, wires=wires[1]),
        ]
        return decomp_ops

    def adjoint(self):
        return ControlledPhaseShift(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [ControlledPhaseShift(self.data[0] * z, wires=self.wires)]

    @property
    def control_wires(self):
        return Wires(self.wires[0])


CPhase = ControlledPhaseShift


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
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 1
    num_params = 3
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0, 0, 0)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,), (1,), (1,)]

    def __init__(self, phi, theta, omega, wires, do_queue=True, id=None):
        super().__init__(phi, theta, omega, wires=wires, do_queue=do_queue, id=id)

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
        interface = qml.math._multi_dispatch([phi, theta, omega])

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
        decomp_ops = [
            RZ(phi, wires=wires),
            RY(theta, wires=wires),
            RZ(omega, wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        phi, theta, omega = self.parameters
        return Rot(-omega, -theta, -phi, wires=self.wires)

    def _controlled(self, wire):
        CRot(*self.parameters, wires=wire + self.wires)

    def single_qubit_rot_angles(self):
        return self.data

    def simplify(self):
        """Simplifies into single-rotation gates or a Hadamard if possible.

        >>> qml.Rot(np.pi / 2, 0.1, -np.pi / 2, wires=0).simplify()
        RX(0.1, wires=[0])
        >>> qml.Rot(np.pi, np.pi/2, 0, 0).simplify()
        Hadamard(wires=[0])

        """
        p0, p1, p2 = np.mod(self.data, 2 * np.pi)

        if np.allclose(p0, np.pi / 2) and np.allclose(np.mod(self.data[2], -2 * np.pi), -np.pi / 2):
            return qml.RX(self.data[1], wires=self.wires)
        if np.allclose(p0, 0) and np.allclose(p2, 0):
            return qml.RY(self.data[1], wires=self.wires)
        if np.allclose(p1, 0):
            return qml.RZ(self.data[0] + self.data[2], wires=self.wires)
        if np.allclose(p0, np.pi) and np.allclose(p1, np.pi / 2) and np.allclose(p2, 0):
            return qml.Hadamard(wires=self.wires)

        return self


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

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
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
        ops.append(RZ(theta, wires=wires[0]))
        ops += [qml.CNOT(wires=(w0, w1)) for w0, w1 in zip(wires[1:], wires[:~0])]

        return ops

    def adjoint(self):
        return MultiRZ(-self.parameters[0], wires=self.wires)

    def pow(self, z):
        return [MultiRZ(self.data[0] * z, wires=self.wires)]


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
        "X": Hadamard.compute_matrix(),
        "Y": RX.compute_matrix(np.pi / 2),
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

        if self.inverse:
            op_label += "⁻¹"

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
        return -0.5 * qml.grouping.string_to_pauli_word(pauli_word, wire_map=wire_map)

    @staticmethod
    def compute_eigvals(theta, pauli_word):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
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
            pauli_word (string): the Pauli word defining the rotation
            wires (Iterable, Wires): the wires the operation acts on

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


class CRX(Operation):
    r"""
    The controlled-RX operator

    .. math::

        \begin{align}
            CR_x(\phi) &=
            \begin{bmatrix}
            & 1 & 0 & 0 & 0 \\
            & 0 & 1 & 0 & 0\\
            & 0 & 0 & \cos(\phi/2) & -i\sin(\phi/2)\\
            & 0 & 0 & -i\sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.
        \end{align}

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: The controlled-RX operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://doi.org/10.1088/1367-2630/ac2cb3):

      .. math::

          \frac{d}{d\phi}f(CR_x(\phi)) = c_+ \left[f(CR_x(\phi+a)) - f(CR_x(\phi-a))\right] - c_- \left[f(CR_x(\phi+b)) - f(CR_x(\phi-b))\right]

      where :math:`f` is an expectation value depending on :math:`CR_x(\phi)`, and

      - :math:`a = \pi/2`
      - :math:`b = 3\pi/2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "X"
    grad_method = "A"
    parameter_frequencies = [(0.5, 1.0)]

    def generator(self):
        return -0.5 * qml.Projector(np.array([1]), wires=self.wires[0]) @ qml.PauliX(self.wires[1])

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "RX", cache=cache)

    @staticmethod
    def compute_matrix(theta):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CRX.matrix`

        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.CRX.compute_matrix(torch.tensor(0.5))
        tensor([[1.0+0.0j, 0.0+0.0j,    0.0+0.0j,    0.0+0.0j],
                [0.0+0.0j, 1.0+0.0j,    0.0+0.0j,    0.0+0.0j],
                [0.0+0.0j, 0.0+0.0j, 0.9689+0.0j, 0.0-0.2474j],
                [0.0+0.0j, 0.0+0.0j, 0.0-0.2474j, 0.9689+0.0j]])
        """
        interface = qml.math.get_interface(theta)

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        if interface == "tensorflow":
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)

        # The following avoids casting an imaginary quantity to reals when backpropagating
        c = (1 + 0j) * c
        js = -1j * s
        ones = qml.math.ones_like(js)
        zeros = qml.math.zeros_like(js)
        matrix = [
            [ones, zeros, zeros, zeros],
            [zeros, ones, zeros, zeros],
            [zeros, zeros, c, js],
            [zeros, zeros, js, c],
        ]

        return qml.math.stack([stack_last(row) for row in matrix], axis=-2)

    @staticmethod
    def compute_decomposition(phi, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.CRot.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Iterable, Wires): the wires the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CRX.compute_decomposition(1.2, wires=(0,1))
        [RZ(1.5707963267948966, wires=[1]),
        RY(0.6, wires=[1]),
        CNOT(wires=[0, 1]),
        RY(-0.6, wires=[1]),
        CNOT(wires=[0, 1]),
        RZ(-1.5707963267948966, wires=[1])]

        """
        pi_half = qml.math.ones_like(phi) * (np.pi / 2)
        decomp_ops = [
            RZ(pi_half, wires=wires[1]),
            RY(phi / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            RY(-phi / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            RZ(-pi_half, wires=wires[1]),
        ]
        return decomp_ops

    def adjoint(self):
        return CRX(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [CRX(self.data[0] * z, wires=self.wires)]

    @property
    def control_wires(self):
        return Wires(self.wires[0])


class CRY(Operation):
    r"""
    The controlled-RY operator

    .. math::

        \begin{align}
            CR_y(\phi) &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0\\
                0 & 0 & \cos(\phi/2) & -\sin(\phi/2)\\
                0 & 0 & \sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.
        \end{align}

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: The controlled-RY operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://doi.org/10.1088/1367-2630/ac2cb3):

      .. math::

          \frac{d}{d\phi}f(CR_y(\phi)) = c_+ \left[f(CR_y(\phi+a)) - f(CR_y(\phi-a))\right] - c_- \left[f(CR_y(\phi+b)) - f(CR_y(\phi-b))\right]

      where :math:`f` is an expectation value depending on :math:`CR_y(\phi)`, and

      - :math:`a = \pi/2`
      - :math:`b = 3\pi/2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "Y"
    grad_method = "A"
    parameter_frequencies = [(0.5, 1.0)]

    def generator(self):
        return -0.5 * qml.Projector(np.array([1]), wires=self.wires[0]) @ qml.PauliY(self.wires[1])

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "RY", cache=cache)

    @staticmethod
    def compute_matrix(theta):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CRY.matrix`


        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.CRY.compute_matrix(torch.tensor(0.5))
        tensor([[ 1.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  1.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.9689, -0.2474],
                [ 0.0000,  0.0000,  0.2474,  0.9689]], dtype=torch.float64)
        """
        interface = qml.math.get_interface(theta)

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        if interface == "tensorflow":
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)

        # The following avoids casting an imaginary quantity to reals when backpropagating
        c = (1 + 0j) * c
        s = (1 + 0j) * s
        ones = qml.math.ones_like(s)
        zeros = qml.math.zeros_like(s)
        matrix = [
            [ones, zeros, zeros, zeros],
            [zeros, ones, zeros, zeros],
            [zeros, zeros, c, -s],
            [zeros, zeros, s, c],
        ]

        return qml.math.stack([stack_last(row) for row in matrix], axis=-2)

    @staticmethod
    def compute_decomposition(phi, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.CRY.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CRY.compute_decomposition(1.2, wires=(0,1))
        [RY(0.6, wires=[1]),
        CNOT(wires=[0, 1]),
        RY(-0.6, wires=[1]),
        CNOT(wires=[0, 1])]

        """
        decomp_ops = [
            RY(phi / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            RY(-phi / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        return CRY(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [CRY(self.data[0] * z, wires=self.wires)]

    @property
    def control_wires(self):
        return Wires(self.wires[0])


class CRZ(Operation):
    r"""
    The controlled-RZ operator

    .. math::

        \begin{align}
             CR_z(\phi) &=
             \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0\\
                0 & 0 & e^{-i\phi/2} & 0\\
                0 & 0 & 0 & e^{i\phi/2}
            \end{bmatrix}.
        \end{align}


    .. note:: The subscripts of the operations in the formula refer to the wires they act on, e.g. 1 corresponds to the first element in ``wires`` that is the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: The controlled-RZ operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://doi.org/10.1088/1367-2630/ac2cb3):

      .. math::

          \frac{d}{d\phi}f(CR_z(\phi)) = c_+ \left[f(CR_z(\phi+a)) - f(CR_z(\phi-a))\right] - c_- \left[f(CR_z(\phi+b)) - f(CR_z(\phi-b))\right]

      where :math:`f` is an expectation value depending on :math:`CR_z(\phi)`, and

      - :math:`a = \pi/2`
      - :math:`b = 3\pi/2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "Z"
    grad_method = "A"
    parameter_frequencies = [(0.5, 1.0)]

    def generator(self):
        return -0.5 * qml.Projector(np.array([1]), wires=self.wires[0]) @ qml.PauliZ(self.wires[1])

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "RZ", cache=cache)

    @staticmethod
    def compute_matrix(theta):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CRZ.matrix`

        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.CRZ.compute_matrix(torch.tensor(0.5))
        tensor([[1.0+0.0j, 0.0+0.0j,       0.0+0.0j,       0.0+0.0j],
                [0.0+0.0j, 1.0+0.0j,       0.0+0.0j,       0.0+0.0j],
                [0.0+0.0j, 0.0+0.0j, 0.9689-0.2474j,       0.0+0.0j],
                [0.0+0.0j, 0.0+0.0j,       0.0+0.0j, 0.9689+0.2474j]])
        """
        if qml.math.get_interface(theta) == "tensorflow":
            p = qml.math.exp(-0.5j * qml.math.cast_like(theta, 1j))
            if qml.math.ndim(p) == 0:
                return qml.math.diag([1, 1, p, qml.math.conj(p)])

            ones = qml.math.ones_like(p)
            diags = stack_last([ones, ones, p, qml.math.conj(p)])
            return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)

        signs = qml.math.array([0, 0, 1, -1], like=theta)
        arg = -0.5j * theta

        if qml.math.ndim(arg) == 0:
            return qml.math.diag(qml.math.exp(arg * signs))

        diags = qml.math.exp(qml.math.outer(arg, signs))
        return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)

    @staticmethod
    def compute_eigvals(theta):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.CRZ.eigvals`


        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.CRZ.compute_eigvals(torch.tensor(0.5))
        tensor([1.0000+0.0000j, 1.0000+0.0000j, 0.9689-0.2474j, 0.9689+0.2474j])
        """
        if qml.math.get_interface(theta) == "tensorflow":
            phase = qml.math.exp(-0.5j * qml.math.cast_like(theta, 1j))
            ones = qml.math.ones_like(phase)
            return stack_last([ones, ones, phase, qml.math.conj(phase)])

        prefactors = qml.math.array([0, 0, -0.5j, 0.5j], like=theta)
        if qml.math.ndim(theta) == 0:
            product = theta * prefactors
        else:
            product = qml.math.outer(theta, prefactors)
        return qml.math.exp(product)

    @staticmethod
    def compute_decomposition(phi, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.CRZ.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CRZ.compute_decomposition(1.2, wires=(0,1))
        [PhaseShift(0.6, wires=[1]),
        CNOT(wires=[0, 1]),
        PhaseShift(-0.6, wires=[1]),
        CNOT(wires=[0, 1])]

        """
        decomp_ops = [
            PhaseShift(phi / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            PhaseShift(-phi / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        return CRZ(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [CRZ(self.data[0] * z, wires=self.wires)]

    @property
    def control_wires(self):
        return Wires(self.wires[0])


class CRot(Operation):
    r"""
    The controlled-Rot operator

    .. math:: CR(\phi, \theta, \omega) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
            0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 3
    * Number of dimensions per parameter: (0, 0, 0)
    * Gradient recipe: The controlled-Rot operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://doi.org/10.1088/1367-2630/ac2cb3):

      .. math::

          \frac{d}{d\mathbf{x}_i}f(CR(\mathbf{x}_i)) = c_+ \left[f(CR(\mathbf{x}_i+a)) - f(CR(\mathbf{x}_i-a))\right] - c_- \left[f(CR(\mathbf{x}_i+b)) - f(CR(\mathbf{x}_i-b))\right]

      where :math:`f` is an expectation value depending on :math:`CR(\mathbf{x}_i)`, and

      - :math:`\mathbf{x} = (\phi, \theta, \omega)` and `i` is an index to :math:`\mathbf{x}`
      - :math:`a = \pi/2`
      - :math:`b = 3\pi/2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        omega (float): rotation angle :math:`\omega`
        wires (Sequence[int]): the wire the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    num_params = 3
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0, 0, 0)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(0.5, 1.0), (0.5, 1.0), (0.5, 1.0)]

    def __init__(self, phi, theta, omega, wires, do_queue=True, id=None):
        super().__init__(phi, theta, omega, wires=wires, do_queue=do_queue, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "Rot", cache=cache)

    @staticmethod
    def compute_matrix(phi, theta, omega):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CRot.matrix`


        Args:
            phi(tensor_like or float): first rotation angle
            theta (tensor_like or float): second rotation angle
            omega (tensor_like or float): third rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

         >>> qml.CRot.compute_matrix(torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3))
         tensor([[ 1.0+0.0j,  0.0+0.0j,        0.0+0.0j,        0.0+0.0j],
                [ 0.0+0.0j,  1.0+0.0j,        0.0+0.0j,        0.0+0.0j],
                [ 0.0+0.0j,  0.0+0.0j,  0.9752-0.1977j, -0.0993+0.0100j],
                [ 0.0+0.0j,  0.0+0.0j,  0.0993+0.0100j,  0.9752+0.1977j]])
        """
        # It might be that they are in different interfaces, e.g.,
        # CRot(0.2, 0.3, tf.Variable(0.5), wires=[0, 1])
        # So we need to make sure the matrix comes out having the right type
        interface = qml.math._multi_dispatch([phi, theta, omega])

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        # If anything is not tensorflow, it has to be casted
        if interface == "tensorflow":
            phi = qml.math.cast_like(qml.math.asarray(phi, like=interface), 1j)
            omega = qml.math.cast_like(qml.math.asarray(omega, like=interface), 1j)
            c = qml.math.cast_like(qml.math.asarray(c, like=interface), 1j)
            s = qml.math.cast_like(qml.math.asarray(s, like=interface), 1j)

        # The following variable is used to assert the all terms to be stacked have same shape
        one = qml.math.ones_like(phi) * qml.math.ones_like(omega)
        c = c * one
        s = s * one

        o = qml.math.ones_like(c)
        z = qml.math.zeros_like(c)
        mat = [
            [o, z, z, z],
            [z, o, z, z],
            [
                z,
                z,
                qml.math.exp(-0.5j * (phi + omega)) * c,
                -qml.math.exp(0.5j * (phi - omega)) * s,
            ],
            [
                z,
                z,
                qml.math.exp(-0.5j * (phi - omega)) * s,
                qml.math.exp(0.5j * (phi + omega)) * c,
            ],
        ]

        return qml.math.stack([stack_last(row) for row in mat], axis=-2)

    @staticmethod
    def compute_decomposition(phi, theta, omega, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.CRot.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            theta (float): rotation angle :math:`\theta`
            omega (float): rotation angle :math:`\omega`
            wires (Iterable, Wires): the wires the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.PhaseShift.compute_decomposition(1.234, wires=0)
        [RZ(-1.1, wires=[1]),
        CNOT(wires=[0, 1]),
        RZ(-2.3, wires=[1]),
        RY(-1.15, wires=[1]),
        CNOT(wires=[0, 1]),
        RY(1.15, wires=[1]),
        RZ(3.4, wires=[1])]

        """
        decomp_ops = [
            RZ((phi - omega) / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            RZ(-(phi + omega) / 2, wires=wires[1]),
            RY(-theta / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            RY(theta / 2, wires=wires[1]),
            RZ(omega, wires=wires[1]),
        ]
        return decomp_ops

    def adjoint(self):
        phi, theta, omega = self.parameters
        return CRot(-omega, -theta, -phi, wires=self.wires)

    @property
    def control_wires(self):
        return Wires(self.wires[0])

    def simplify(self):
        """Simplifies into single controlled rotations or a controlled-Hadamard if possible.

        >>> qml.CRot(np.pi / 2, 0.1, -np.pi / 2, wires=(0,1)).simplify()
        CRX(0.1, wires=[0, 1])
        >>> qml.CRot(0, 0.2, 0, wires=(0,1)).simplify()
        CRY(0.2, wires=[0, 1])

        """
        target_wires = [w for w in self.wires if w not in self.control_wires]
        wires = self.wires
        params = self.parameters

        p0, p1, p2 = np.mod(params, 2 * np.pi)

        if np.allclose(p0, np.pi / 2) and np.allclose(np.mod(self.data[2], -2 * np.pi), -np.pi / 2):
            return qml.CRX(self.data[1], wires=wires)
        if np.allclose(p0, 0) and np.allclose(p2, 0):
            return qml.CRY(self.data[1], wires=wires)
        if np.allclose(p1, 0):
            return qml.CRZ(self.data[0] + self.data[2], wires=wires)
        if np.allclose(p0, np.pi) and np.allclose(p1, np.pi / 2) and np.allclose(p2, 0):
            hadamard = qml.Hadamard
            return qml.ctrl(hadamard, control=self.control_wires)(wires=target_wires)

        return self


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
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
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

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

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
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 1
    num_params = 2
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0, 0)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,), (1,)]

    def __init__(self, phi, delta, wires, do_queue=True, id=None):
        super().__init__(phi, delta, wires=wires, do_queue=do_queue, id=id)

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
        interface = qml.math._multi_dispatch([phi, delta])

        # If anything is not tensorflow, it has to be casted and then
        if interface == "tensorflow":
            phi = qml.math.cast_like(qml.math.asarray(phi, like=interface), 1j)
            delta = qml.math.cast_like(qml.math.asarray(delta, like=interface), 1j)

        one = qml.math.ones_like(phi) * qml.math.ones_like(delta)
        mat = [
            [one, -qml.math.exp(1j * delta) * one],
            [qml.math.exp(1j * phi) * one, qml.math.exp(1j * (phi + delta))],
        ]

        return INV_SQRT2 * qml.math.stack([stack_last(row) for row in mat], axis=-2)

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
        decomp_ops = [
            Rot(delta, pi_half, -delta, wires=wires),
            PhaseShift(delta, wires=wires),
            PhaseShift(phi, wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        phi, delta = self.parameters
        new_delta = qml.math.mod((np.pi - phi), (2 * np.pi))
        new_phi = qml.math.mod((np.pi - delta), (2 * np.pi))
        return U2(new_phi, new_delta, wires=self.wires)

    def simplify(self):
        """Simplifies the gate into RX or RY gates if possible."""
        wires = self.wires

        if np.allclose(np.mod(self.data[1], 2 * np.pi), 0) and np.allclose(
            np.mod(self.data[0] + self.data[1], 2 * np.pi), 0
        ):
            return qml.RY(np.pi / 2, wires=wires)
        if np.allclose(np.mod(self.data[1], np.pi / 2), 0) and np.allclose(
            np.mod(self.data[0] + self.data[1], 2 * np.pi), 0
        ):
            return qml.RX(self.data[1], wires=wires)

        return self


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
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 1
    num_params = 3
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0, 0, 0)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,), (1,), (1,)]

    def __init__(self, theta, phi, delta, wires, do_queue=True, id=None):
        super().__init__(theta, phi, delta, wires=wires, do_queue=do_queue, id=id)

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
        interface = qml.math._multi_dispatch([theta, phi, delta])

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
        decomp_ops = [
            Rot(delta, theta, -delta, wires=wires),
            PhaseShift(delta, wires=wires),
            PhaseShift(phi, wires=wires),
        ]
        return decomp_ops

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

        p0, p1, p2 = np.mod(params, 2 * np.pi)

        if np.allclose(p0, 0) and not np.allclose(p1, 0) and np.allclose(p2, 0):
            return qml.PhaseShift(self.data[1], wires=wires)
        if (
            np.allclose(p2, np.pi / 2)
            and np.allclose(np.mod(self.data[1] + self.data[2], 2 * np.pi), 0)
            and not np.allclose(p0, 0)
        ):
            return qml.RX(self.data[0], wires=wires)
        if not np.allclose(p0, 0) and np.allclose(p1, 0) and np.allclose(p2, 0):
            return qml.RY(self.data[0], wires=wires)

        return self


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
        return -0.5 * PauliX(wires=self.wires[0]) @ PauliX(wires=self.wires[1])

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
            RX(phi, wires=[wires[0]]),
            qml.CNOT(wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        (phi,) = self.parameters
        return IsingXX(-phi, wires=self.wires)

    def pow(self, z):
        return [IsingXX(self.data[0] * z, wires=self.wires)]


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
        return -0.5 * PauliY(wires=self.wires[0]) @ PauliY(wires=self.wires[1])

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
        return -0.5 * PauliZ(wires=self.wires[0]) @ PauliZ(wires=self.wires[1])

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

        >>> qml.IsingZZ.compute_decomposition(1.23, wires=0)
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

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
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
        return 0.25 * qml.PauliX(wires=self.wires[0]) @ qml.PauliX(
            wires=self.wires[1]
        ) + 0.25 * qml.PauliY(wires=self.wires[0]) @ qml.PauliY(wires=self.wires[1])

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
        if qml.math.ndim(phi) == 0:
            return qml.math.diag([1, c, c, 1]) + qml.math.diag([0, js, js, 0])[::-1]

        ones = qml.math.ones_like(c)
        diags = stack_last([ones, c, c, ones])[:, :, np.newaxis]
        return diags * np.eye(4) + qml.math.tensordot(js, np.diag([0, 1, 1, 0])[::-1], axes=0)

    @staticmethod
    def compute_eigvals(phi):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
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
