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
import numpy as np

import pennylane as qml
from pennylane.operation import Operation
from pennylane.wires import Wires

from .non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ
from .parametric_ops_single_qubit import _can_replace, stack_last, RY, RZ, PhaseShift


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

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

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

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
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
            PhaseShift(phi / 2, wires=wires[0]),
            qml.CNOT(wires=wires),
            PhaseShift(-phi / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            PhaseShift(phi / 2, wires=wires[1]),
        ]
        return decomp_ops

    def adjoint(self):
        return ControlledPhaseShift(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [ControlledPhaseShift(self.data[0] * z, wires=self.wires)]

    def simplify(self):
        phi = self.data[0] % (2 * np.pi)

        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])

        return ControlledPhaseShift(phi, wires=self.wires)

    @property
    def control_wires(self):
        return Wires(self.wires[0])


CPhase = ControlledPhaseShift


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

    def generator(self):
        return qml.Projector(np.array([0, 0]), wires=self.wires)

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label="Rϕ(00)", cache=cache)

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CPhaseShift00.matrix`

        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.CPhaseShift00.compute_matrix(torch.tensor(0.5))
            tensor([[0.8776+0.4794j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
                    [0.0000+0.0000j, 1.0+0.0j, 0.0+0.0j, 0.0+0.0j],
                    [0.0000+0.0000j, 0.0+0.0j, 1.0+0.0j, 0.0+0.0j],
                    [0.0000+0.0000j, 0.0+0.0j, 0.0+0.0j, 1.0+0.0j]])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        exp_part = qml.math.exp(1j * phi)

        if qml.math.ndim(phi) > 0:
            ones = qml.math.ones_like(exp_part)
            zeros = qml.math.zeros_like(exp_part)
            matrix = [
                [exp_part, zeros, zeros, zeros],
                [zeros, ones, zeros, zeros],
                [zeros, zeros, ones, zeros],
                [zeros, zeros, zeros, ones],
            ]

            return qml.math.stack([stack_last(row) for row in matrix], axis=-2)

        return qml.math.diag([exp_part, 1, 1, 1])

    @staticmethod
    def compute_eigvals(phi):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.CPhaseShift00.eigvals`


        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.CPhaseShift00.compute_eigvals(torch.tensor(0.5))
        tensor([0.8776+0.4794j, 1.0000+0.0000j, 1.0000+0.0000j, 1.0000+0.0000j])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        exp_part = qml.math.exp(1j * phi)
        ones = qml.math.ones_like(exp_part)
        return stack_last([exp_part, ones, ones, ones])

    @staticmethod
    def compute_decomposition(phi, wires):
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
        [PauliX(wires=[0]),
        PauliX(wires=[1]),
        PhaseShift(0.617, wires=[0]),
        PhaseShift(0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        PhaseShift(-0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        PauliX(wires=[1]),
        PauliX(wires=[0])]

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

    def adjoint(self):
        return CPhaseShift00(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [CPhaseShift00(self.data[0] * z, wires=self.wires)]

    @property
    def control_values(self):
        """str: The control values of the operation"""
        return "0"

    @property
    def control_wires(self):
        return self.wires[0:1]


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

    def generator(self):
        return qml.Projector(np.array([0, 1]), wires=self.wires)

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label="Rϕ(01)", cache=cache)

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CPhaseShift01.matrix`

        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.CPhaseShift01.compute_matrix(torch.tensor(0.5))
            tensor([[1.0+0.0j, 0.0000+0.0000j, 0.0+0.0j, 0.0+0.0j],
                    [0.0+0.0j, 0.8776+0.4794j, 0.0+0.0j, 0.0+0.0j],
                    [0.0+0.0j, 0.0000+0.0000j, 1.0+0.0j, 0.0+0.0j],
                    [0.0+0.0j, 0.0000+0.0000j, 0.0+0.0j, 1.0+0.0j]])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        exp_part = qml.math.exp(1j * phi)

        if qml.math.ndim(phi) > 0:
            ones = qml.math.ones_like(exp_part)
            zeros = qml.math.zeros_like(exp_part)
            matrix = [
                [ones, zeros, zeros, zeros],
                [zeros, exp_part, zeros, zeros],
                [zeros, zeros, ones, zeros],
                [zeros, zeros, zeros, ones],
            ]

            return qml.math.stack([stack_last(row) for row in matrix], axis=-2)

        return qml.math.diag([1, exp_part, 1, 1])

    @staticmethod
    def compute_eigvals(phi):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.CPhaseShift01.eigvals`


        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.CPhaseShift01.compute_eigvals(torch.tensor(0.5))
        tensor([1.0000+0.0000j, 0.8776+0.4794j, 1.0000+0.0000j, 1.0000+0.0000j])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        exp_part = qml.math.exp(1j * phi)
        ones = qml.math.ones_like(exp_part)
        return stack_last([ones, exp_part, ones, ones])

    @staticmethod
    def compute_decomposition(phi, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.
        .. seealso:: :meth:`~.CPhaseShift01.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CPhaseShift01.compute_decomposition(1.234, wires=(0,1))
        [PauliX(wires=[0]),
        PhaseShift(0.617, wires=[0]),
        PhaseShift(0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        PhaseShift(-0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        PauliX(wires=[0])]

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

    def adjoint(self):
        return CPhaseShift01(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [CPhaseShift01(self.data[0] * z, wires=self.wires)]

    @property
    def control_values(self):
        """str: The control values of the operation"""
        return "0"

    @property
    def control_wires(self):
        return self.wires[0:1]


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

    def generator(self):
        return qml.Projector(np.array([1, 0]), wires=self.wires)

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label="Rϕ(10)", cache=cache)

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CPhaseShift10.matrix`

        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.CPhaseShift10.compute_matrix(torch.tensor(0.5))
            tensor([[1.0+0.0j, 0.0+0.0j, 0.0000+0.0000j, 0.0+0.0j],
                    [0.0+0.0j, 1.0+0.0j, 0.0000+0.0000j, 0.0+0.0j],
                    [0.0+0.0j, 0.0+0.0j, 0.8776+0.4794j, 0.0+0.0j],
                    [0.0+0.0j, 0.0+0.0j, 0.0000+0.0000j, 1.0+0.0j]])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        exp_part = qml.math.exp(1j * phi)

        if qml.math.ndim(phi) > 0:
            ones = qml.math.ones_like(exp_part)
            zeros = qml.math.zeros_like(exp_part)
            matrix = [
                [ones, zeros, zeros, zeros],
                [zeros, ones, zeros, zeros],
                [zeros, zeros, exp_part, zeros],
                [zeros, zeros, zeros, ones],
            ]

            return qml.math.stack([stack_last(row) for row in matrix], axis=-2)

        return qml.math.diag([1, 1, exp_part, 1])

    @staticmethod
    def compute_eigvals(phi):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.CPhaseShift10.eigvals`


        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.CPhaseShift10.compute_eigvals(torch.tensor(0.5))
        tensor([1.0000+0.0000j, 1.0000+0.0000j, 0.8776+0.4794j, 1.0000+0.0000j])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        exp_part = qml.math.exp(1j * phi)
        ones = qml.math.ones_like(exp_part)
        return stack_last([ones, ones, exp_part, ones])

    @staticmethod
    def compute_decomposition(phi, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.
        .. seealso:: :meth:`~.CPhaseShift10.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.CPhaseShift10.compute_decomposition(1.234, wires=(0,1))
        [PauliX(wires=[1]),
        PhaseShift(0.617, wires=[0]),
        PhaseShift(0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        PhaseShift(-0.617, wires=[1]),
        CNOT(wires=[0, 1]),
        PauliX(wires=[1])]

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

    def adjoint(self):
        return CPhaseShift10(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [CPhaseShift10(self.data[0] * z, wires=self.wires)]

    @property
    def control_wires(self):
        return self.wires[0:1]


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
        return -0.5 * qml.Projector(np.array([1]), wires=self.wires[0]) @ PauliX(self.wires[1])

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

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

    def simplify(self):
        phi = self.data[0] % (4 * np.pi)

        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])

        return CRX(phi, wires=self.wires)

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
        return -0.5 * qml.Projector(np.array([1]), wires=self.wires[0]) @ PauliY(self.wires[1])

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

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

    def simplify(self):
        phi = self.data[0] % (4 * np.pi)

        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])

        return CRY(phi, wires=self.wires)

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
        return -0.5 * qml.Projector(np.array([1]), wires=self.wires[0]) @ PauliZ(self.wires[1])

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

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

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
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

    def simplify(self):
        phi = self.data[0] % (4 * np.pi)

        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])

        return CRZ(phi, wires=self.wires)

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
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    num_params = 3
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0, 0, 0)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(0.5, 1.0), (0.5, 1.0), (0.5, 1.0)]

    def __init__(self, phi, theta, omega, wires, id=None):
        super().__init__(phi, theta, omega, wires=wires, id=id)

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
        interface = qml.math.get_interface(phi, theta, omega)

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

        p0, p1, p2 = [p % (4 * np.pi) for p in params]

        if _can_replace(p0, 0) and _can_replace(p1, 0) and _can_replace(p2, 0):
            return qml.Identity(wires=wires[0])
        if _can_replace(p0, np.pi / 2) and _can_replace(p2, 7 * np.pi / 2):
            return qml.CRX(p1, wires=wires)
        if _can_replace(p0, 0) and _can_replace(p2, 0):
            return qml.CRY(p1, wires=wires)
        if _can_replace(p1, 0):
            return qml.CRZ((p0 + p2) % (4 * np.pi), wires=wires)
        if _can_replace(p0, np.pi) and _can_replace(p1, np.pi / 2) and _can_replace(p2, 0):
            return qml.ctrl(Hadamard(wires=target_wires), control=self.control_wires)

        return CRot(p0, p1, p2, wires=wires)
