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
"""
This submodule contains the discrete-variable quantum operations that come
from quantum chemistry applications.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access
import functools
import numpy as np
from scipy.sparse import csr_matrix

import pennylane as qml
from pennylane.operation import Operation

I4 = np.eye(4)
I16 = np.eye(16)

stack_last = functools.partial(qml.math.stack, axis=-1)


class SingleExcitation(Operation):
    r"""
    Single excitation rotation.

    .. math:: U(\phi) = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}.

    This operation performs a rotation in the two-dimensional subspace :math:`\{|01\rangle,
    |10\rangle\}`. The name originates from the occupation-number representation of
    fermionic wavefunctions, where the transformation  from :math:`|10\rangle` to :math:`|01\rangle`
    is interpreted as "exciting" a particle from the first qubit to the second.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: The ``SingleExcitation`` operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://doi.org/10.1088/1367-2630/ac2cb3):

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wires the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)

    **Example**

    The following circuit performs the transformation :math:`|10\rangle\rightarrow \cos(
    \phi/2)|10\rangle -\sin(\phi/2)|01\rangle`:

    .. code-block::

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.SingleExcitation(phi, wires=[0, 1])
            return qml.state()

        circuit(0.1)
    """

    num_wires = 2
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    """Gradient computation method."""

    parameter_frequencies = [(0.5, 1.0)]
    """Frequencies of the operation parameter with respect to an expectation value."""

    def generator(self):
        w1, w2 = self.wires
        return 0.25 * qml.PauliX(w1) @ qml.PauliY(w2) - 0.25 * qml.PauliY(w1) @ qml.PauliX(w2)

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SingleExcitation.matrix`


        Args:
          phi (tensor_like or float): rotation angle

        Returns:
          tensor_like: canonical matrix

        **Example**

        >>> qml.SingleExcitation.compute_matrix(torch.tensor(0.5))
        tensor([[ 1.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.9689, -0.2474,  0.0000],
                [ 0.0000,  0.2474,  0.9689,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  1.0000]])
        """
        interface = qml.math.get_interface(phi)
        if interface == "tensorflow":
            c = qml.math.cos(phi / 2)
            s = qml.math.sin(phi / 2)
            ones = qml.math.ones_like(phi)
            zeros = qml.math.zeros_like(phi)
            rows = [
                [ones, zeros, zeros, zeros],
                [zeros, c, -s, zeros],
                [zeros, s, c, zeros],
                [zeros, zeros, zeros, ones],
            ]
            return qml.math.stack([stack_last(row) for row in rows], axis=-2)

        c_minus_1 = qml.math.cos(phi / 2) - 1
        s = qml.math.sin(phi / 2)

        mask_c = np.diag([0, 1, 1, 0])
        mask_s = np.array([[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        eye = np.eye(4)
        if interface == "torch":
            mask_c = qml.math.convert_like(mask_c, phi)
            mask_s = qml.math.convert_like(mask_s, phi)
            eye = qml.math.convert_like(eye, phi)

        if qml.math.ndim(phi) == 0:
            return c_minus_1 * mask_c + s * mask_s + eye

        return (
            qml.math.einsum("i,jk->ijk", c_minus_1, mask_c)
            + qml.math.einsum("i,jk->ijk", s, mask_s)
            + eye
        )

    @staticmethod
    def compute_decomposition(phi, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.SingleExcitation.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.SingleExcitation.compute_decomposition(1.23, wires=(0,1))
        [CNOT(wires=[0, 1]), CRY(1.23, wires=[1, 0]), CNOT(wires=[0, 1])]

        """
        decomp_ops = [
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.CRY(phi, wires=[wires[1], wires[0]]),
            qml.CNOT(wires=[wires[0], wires[1]]),
        ]
        return decomp_ops

    def adjoint(self):
        (phi,) = self.parameters
        return SingleExcitation(-phi, wires=self.wires)

    def pow(self, z):
        return [SingleExcitation(self.data[0] * z, wires=self.wires)]

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "G", cache=cache)


class SingleExcitationMinus(Operation):
    r"""
    Single excitation rotation with negative phase-shift outside the rotation subspace.

    .. math:: U_-(\phi) = \begin{bmatrix}
                e^{-i\phi/2} & 0 & 0 & 0 \\
                0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                0 & 0 & 0 & e^{-i\phi/2}
            \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_-(\phi)) = \frac{1}{2}\left[f(U_-(\phi+\pi/2)) - f(U_-(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`U_-(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wires the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)

    """
    num_wires = 2
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    """Gradient computation method."""

    parameter_frequencies = [(1,)]
    """Frequencies of the operation parameter with respect to an expectation value."""

    def generator(self):
        w1, w2 = self.wires
        return (
            -0.25 * qml.Identity(w1)
            + 0.25 * qml.PauliX(w1) @ qml.PauliY(w2)
            - 0.25 * qml.PauliY(w1) @ qml.PauliX(w2)
            - 0.25 * qml.PauliZ(w1) @ qml.PauliZ(w2)
        )

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SingleExcitationMinus.matrix`


        Args:
          phi (tensor_like or float): rotation angle

        Returns:
          tensor_like: canonical matrix

        **Example**

        >>> qml.SingleExcitationMinus.compute_matrix(torch.tensor(0.5))
        tensor([[ 0.9689-0.2474j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
                [ 0.0000+0.0000j,  0.9689+0.0000j, -0.2474+0.0000j,  0.0000+0.0000j],
                [ 0.0000+0.0000j,  0.2474+0.0000j,  0.9689+0.0000j,  0.0000+0.0000j],
                [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,  0.9689-0.2474j]])
        """
        interface = qml.math.get_interface(phi)
        if interface == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)
            c = qml.math.cos(phi / 2)
            s = qml.math.sin(phi / 2)
            e = qml.math.exp(-0.5j * phi)
            zeros = qml.math.zeros_like(phi)
            rows = [
                [e, zeros, zeros, zeros],
                [zeros, c, -s, zeros],
                [zeros, s, c, zeros],
                [zeros, zeros, zeros, e],
            ]
            return qml.math.stack([stack_last(row) for row in rows], axis=-2)

        c = qml.math.cos(phi / 2)
        s = qml.math.sin(phi / 2)
        e = qml.math.exp(-0.5j * phi)

        mask_e = np.diag([1, 0, 0, 1])
        mask_c = np.diag([0, 1, 1, 0])
        mask_s = np.array([[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])

        if qml.math.ndim(phi) == 0:
            if interface == "torch":
                mask_e = qml.math.convert_like(mask_e, phi)
                mask_c = qml.math.convert_like(mask_c, phi)
                mask_s = qml.math.convert_like(mask_s, phi)
            return e * mask_e + c * mask_c + s * mask_s

        return (
            qml.math.einsum("i,jk->ijk", e, mask_e)
            + qml.math.einsum("i,jk->ijk", c, mask_c)
            + qml.math.einsum("i,jk->ijk", s, mask_s)
        )

    @staticmethod
    def compute_decomposition(phi, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.SingleExcitationMinus.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.SingleExcitationMinus.compute_decomposition(1.23, wires=(0,1))
        [PauliX(wires=[0]),
        PauliX(wires=[1]),
        ControlledPhaseShift(-0.615, wires=[1, 0]),
        PauliX(wires=[0]),
        PauliX(wires=[1]),
        ControlledPhaseShift(-0.615, wires=[0, 1]),
        CNOT(wires=[0, 1]),
        CRY(1.23, wires=[1, 0]),
        CNOT(wires=[0, 1])]

        """
        decomp_ops = [
            qml.PauliX(wires=wires[0]),
            qml.PauliX(wires=wires[1]),
            qml.ControlledPhaseShift(-phi / 2, wires=[wires[1], wires[0]]),
            qml.PauliX(wires=wires[0]),
            qml.PauliX(wires=wires[1]),
            qml.ControlledPhaseShift(-phi / 2, wires=[wires[0], wires[1]]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.CRY(phi, wires=[wires[1], wires[0]]),
            qml.CNOT(wires=[wires[0], wires[1]]),
        ]
        return decomp_ops

    def adjoint(self):
        (phi,) = self.parameters
        return SingleExcitationMinus(-phi, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "G₋", cache=cache)


class SingleExcitationPlus(Operation):
    r"""
    Single excitation rotation with positive phase-shift outside the rotation subspace.

    .. math:: U_+(\phi) = \begin{bmatrix}
                e^{i\phi/2} & 0 & 0 & 0 \\
                0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                0 & 0 & 0 & e^{i\phi/2}
            \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_+(\phi)) = \frac{1}{2}\left[f(U_+(\phi+\pi/2)) - f(U_+(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`U_+(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wires the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)

    """
    num_wires = 2
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    """Gradient computation method."""

    parameter_frequencies = [(1,)]
    """Frequencies of the operation parameter with respect to an expectation value."""

    def generator(self):
        w1, w2 = self.wires
        return (
            0.25 * qml.Identity(w1)
            + 0.25 * qml.PauliX(w1) @ qml.PauliY(w2)
            - 0.25 * qml.PauliY(w1) @ qml.PauliX(w2)
            + 0.25 * qml.PauliZ(w1) @ qml.PauliZ(w2)
        )

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SingleExcitationPlus.matrix`


        Args:
          phi (tensor_like or float): rotation angle

        Returns:
          tensor_like: canonical matrix

        **Example**

        >>> qml.SingleExcitationPlus.compute_matrix(torch.tensor(0.5))
        tensor([[ 0.9689+0.2474j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
                [ 0.0000+0.0000j,  0.9689+0.0000j, -0.2474+0.0000j,  0.0000+0.0000j],
                [ 0.0000+0.0000j,  0.2474+0.0000j,  0.9689+0.0000j,  0.0000+0.0000j],
                [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,  0.9689+0.2474j]])
        """
        interface = qml.math.get_interface(phi)
        if interface == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)
            c = qml.math.cos(phi / 2)
            s = qml.math.sin(phi / 2)
            e = qml.math.exp(0.5j * phi)
            zeros = qml.math.zeros_like(phi)
            rows = [
                [e, zeros, zeros, zeros],
                [zeros, c, -s, zeros],
                [zeros, s, c, zeros],
                [zeros, zeros, zeros, e],
            ]
            return qml.math.stack([stack_last(row) for row in rows], axis=-2)

        c = qml.math.cos(phi / 2)
        s = qml.math.sin(phi / 2)
        e = qml.math.exp(0.5j * phi)

        mask_e = np.diag([1, 0, 0, 1])
        mask_c = np.diag([0, 1, 1, 0])
        mask_s = np.array([[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])

        if qml.math.ndim(phi) == 0:
            if interface == "torch":
                mask_e = qml.math.convert_like(mask_e, phi)
                mask_c = qml.math.convert_like(mask_c, phi)
                mask_s = qml.math.convert_like(mask_s, phi)
            return e * mask_e + c * mask_c + s * mask_s

        return (
            qml.math.einsum("i,jk->ijk", e, mask_e)
            + qml.math.einsum("i,jk->ijk", c, mask_c)
            + qml.math.einsum("i,jk->ijk", s, mask_s)
        )

    @staticmethod
    def compute_decomposition(phi, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.SingleExcitationPlus.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.SingleExcitationPlus.compute_decomposition(1.23, wires=(0,1))
        [PauliX(wires=[0]),
        PauliX(wires=[1]),
        ControlledPhaseShift(0.615, wires=[1, 0]),
        PauliX(wires=[0]),
        PauliX(wires=[1]),
        ControlledPhaseShift(0.615, wires=[0, 1]),
        CNOT(wires=[0, 1]),
        CRY(1.23, wires=[1, 0]),
        CNOT(wires=[0, 1])]

        """
        decomp_ops = [
            qml.PauliX(wires=wires[0]),
            qml.PauliX(wires=wires[1]),
            qml.ControlledPhaseShift(phi / 2, wires=[wires[1], wires[0]]),
            qml.PauliX(wires=wires[0]),
            qml.PauliX(wires=wires[1]),
            qml.ControlledPhaseShift(phi / 2, wires=[wires[0], wires[1]]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.CRY(phi, wires=[wires[1], wires[0]]),
            qml.CNOT(wires=[wires[0], wires[1]]),
        ]
        return decomp_ops

    def adjoint(self):
        (phi,) = self.parameters
        return SingleExcitationPlus(-phi, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "G₊", cache=cache)


class DoubleExcitation(Operation):
    r"""
    Double excitation rotation.

    This operation performs an :math:`SO(2)` rotation in the two-dimensional subspace :math:`\{
    |1100\rangle,|0011\rangle\}`. More precisely, it performs the transformation

    .. math::

        &|0011\rangle \rightarrow \cos(\phi/2) |0011\rangle + \sin(\phi/2) |1100\rangle\\
        &|1100\rangle \rightarrow \cos(\phi/2) |1100\rangle - \sin(\phi/2) |0011\rangle,

    while leaving all other basis states unchanged.

    The name originates from the occupation-number representation of fermionic wavefunctions, where
    the transformation from :math:`|1100\rangle` to :math:`|0011\rangle` is interpreted as
    "exciting" two particles from the first pair of qubits to the second pair of qubits.

    **Details:**

    * Number of wires: 4
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: The ``DoubleExcitation`` operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://doi.org/10.1088/1367-2630/ac2cb3):

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wires the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)

    **Example**

    The following circuit performs the transformation :math:`|1100\rangle\rightarrow \cos(
    \phi/2)|1100\rangle - \sin(\phi/2)|0011\rangle)`:

    .. code-block::

        dev = qml.device('default.qubit', wires=4)

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            qml.DoubleExcitation(phi, wires=[0, 1, 2, 3])
            return qml.state()

        circuit(0.1)
    """
    num_wires = 4
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    """Gradient computation method."""

    parameter_frequencies = [(0.5, 1.0)]
    """Frequencies of the operation parameter with respect to an expectation value."""

    def generator(self):
        w0, w1, w2, w3 = self.wires
        coeffs = [0.0625, 0.0625, -0.0625, 0.0625, -0.0625, 0.0625, -0.0625, -0.0625]
        obs = [
            qml.PauliX(w0) @ qml.PauliX(w1) @ qml.PauliX(w2) @ qml.PauliY(w3),
            qml.PauliX(w0) @ qml.PauliX(w1) @ qml.PauliY(w2) @ qml.PauliX(w3),
            qml.PauliX(w0) @ qml.PauliY(w1) @ qml.PauliX(w2) @ qml.PauliX(w3),
            qml.PauliX(w0) @ qml.PauliY(w1) @ qml.PauliY(w2) @ qml.PauliY(w3),
            qml.PauliY(w0) @ qml.PauliX(w1) @ qml.PauliX(w2) @ qml.PauliX(w3),
            qml.PauliY(w0) @ qml.PauliX(w1) @ qml.PauliY(w2) @ qml.PauliY(w3),
            qml.PauliY(w0) @ qml.PauliY(w1) @ qml.PauliX(w2) @ qml.PauliY(w3),
            qml.PauliY(w0) @ qml.PauliY(w1) @ qml.PauliY(w2) @ qml.PauliX(w3),
        ]
        return qml.Hamiltonian(coeffs, obs)

    def pow(self, z):
        return [DoubleExcitation(self.data[0] * z, wires=self.wires)]

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    mask_s = np.zeros((16, 16))
    mask_s[3, 12] = -1
    mask_s[12, 3] = 1

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.DoubleExcitation.matrix`


        Args:
          phi (tensor_like or float): rotation angle

        Returns:
          tensor_like: canonical matrix
        """
        c = qml.math.cos(phi / 2)
        s = qml.math.sin(phi / 2)

        if qml.math.ndim(phi) == 0:
            diag = qml.math.diag([1.0] * 3 + [c] + [1.0] * 8 + [c] + [1.0] * 3)
            if qml.math.get_interface(phi) == "torch":
                return diag + s * qml.math.convert_like(DoubleExcitation.mask_s, phi)
            return diag + s * DoubleExcitation.mask_s

        ones = qml.math.ones_like(c)
        diag = qml.math.stack([ones] * 3 + [c] + [ones] * 8 + [c] + [ones] * 3, axis=-1)
        diag = qml.math.einsum("ij,jk->ijk", diag, I16)
        off_diag = qml.math.einsum("i,jk->ijk", s, DoubleExcitation.mask_s)
        return diag + off_diag

    @staticmethod
    def compute_decomposition(phi, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.DoubleExcitation.decomposition`.

        For the source of this decomposition, see page 17 of
        `"Local, Expressive, Quantum-Number-Preserving VQE Ansatze for Fermionic Systems" <https://doi.org/10.1088/1367-2630/ac2cb3>`_ .

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.DoubleExcitation.compute_decomposition(1.23, wires=(0,1,2,3))
        [CNOT(wires=[2, 3]),
        CNOT(wires=[0, 2]),
        Hadamard(wires=[3]),
        Hadamard(wires=[0]),
        CNOT(wires=[2, 3]),
        CNOT(wires=[0, 1]),
        RY(0.15375, wires=[1]),
        RY(-0.15375, wires=[0]),
        CNOT(wires=[0, 3]),
        Hadamard(wires=[3]),
        CNOT(wires=[3, 1]),
        RY(0.15375, wires=[1]),
        RY(-0.15375, wires=[0]),
        CNOT(wires=[2, 1]),
        CNOT(wires=[2, 0]),
        RY(-0.15375, wires=[1]),
        RY(0.15375, wires=[0]),
        CNOT(wires=[3, 1]),
        Hadamard(wires=[3]),
        CNOT(wires=[0, 3]),
        RY(-0.15375, wires=[1]),
        RY(0.15375, wires=[0]),
        CNOT(wires=[0, 1]),
        CNOT(wires=[2, 0]),
        Hadamard(wires=[0]),
        Hadamard(wires=[3]),
        CNOT(wires=[0, 2]),
        CNOT(wires=[2, 3])]

        """
        # This decomposition is the "upside down" version of that on p17 of https://arxiv.org/abs/2104.05695
        decomp_ops = [
            qml.CNOT(wires=[wires[2], wires[3]]),
            qml.CNOT(wires=[wires[0], wires[2]]),
            qml.Hadamard(wires=wires[3]),
            qml.Hadamard(wires=wires[0]),
            qml.CNOT(wires=[wires[2], wires[3]]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.RY(phi / 8, wires=wires[1]),
            qml.RY(-phi / 8, wires=wires[0]),
            qml.CNOT(wires=[wires[0], wires[3]]),
            qml.Hadamard(wires=wires[3]),
            qml.CNOT(wires=[wires[3], wires[1]]),
            qml.RY(phi / 8, wires=wires[1]),
            qml.RY(-phi / 8, wires=wires[0]),
            qml.CNOT(wires=[wires[2], wires[1]]),
            qml.CNOT(wires=[wires[2], wires[0]]),
            qml.RY(-phi / 8, wires=wires[1]),
            qml.RY(phi / 8, wires=wires[0]),
            qml.CNOT(wires=[wires[3], wires[1]]),
            qml.Hadamard(wires=wires[3]),
            qml.CNOT(wires=[wires[0], wires[3]]),
            qml.RY(-phi / 8, wires=wires[1]),
            qml.RY(phi / 8, wires=wires[0]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.CNOT(wires=[wires[2], wires[0]]),
            qml.Hadamard(wires=wires[0]),
            qml.Hadamard(wires=wires[3]),
            qml.CNOT(wires=[wires[0], wires[2]]),
            qml.CNOT(wires=[wires[2], wires[3]]),
        ]

        return decomp_ops

    def adjoint(self):
        (theta,) = self.parameters
        return DoubleExcitation(-theta, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "G²", cache=cache)


class DoubleExcitationPlus(Operation):
    r"""
    Double excitation rotation with positive phase-shift outside the rotation subspace.

    This operation performs an :math:`SO(2)` rotation in the two-dimensional subspace :math:`\{
    |1100\rangle,|0011\rangle\}` while applying a phase-shift on other states. More precisely,
    it performs the transformation

    .. math::

        &|0011\rangle \rightarrow \cos(\phi/2) |0011\rangle - \sin(\phi/2) |1100\rangle\\
        &|1100\rangle \rightarrow \cos(\phi/2) |1100\rangle + \sin(\phi/2) |0011\rangle\\
        &|x\rangle \rightarrow e^{i\phi/2} |x\rangle,

    for all other basis states :math:`|x\rangle`.

    **Details:**

    * Number of wires: 4
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_+(\phi)) = \frac{1}{2}\left[f(U_+(\phi+\pi/2)) - f(U_+(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`U_+(\phi)`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wires the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 4
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    """Gradient computation method."""

    parameter_frequencies = [(1,)]
    """Frequencies of the operation parameter with respect to an expectation value."""

    def generator(self):
        G = -1 * np.eye(16, dtype=complex64)
        G[3, 3] = G[12, 12] = 0
        G[3, 12] = -1j  # 3 (dec) = 0011 (bin)
        G[12, 3] = 1j  # 12 (dec) = 1100 (bin)
        H = csr_matrix(-0.5 * G)
        return qml.SparseHamiltonian(H, wires=self.wires)

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.DoubleExcitationPlus.matrix`

        Args:
          phi (tensor_like or float): rotation angle

        Returns:
          tensor_like: canonical matrix

        """
        c = qml.math.cos(phi / 2)
        s = qml.math.sin(phi / 2)

        interface = qml.math.get_interface(phi)

        if interface == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)

        e = qml.math.exp(0.5j * phi)
        c = (1 + 0j) * c

        if qml.math.ndim(phi) == 0:
            diag = qml.math.diag([e] * 3 + [c] + [e] * 8 + [c] + [e] * 3)
            if qml.math.get_interface(phi) == "torch":
                return diag + s * qml.math.convert_like(DoubleExcitation.mask_s, phi)
            return diag + s * DoubleExcitation.mask_s

        diag = qml.math.stack([e] * 3 + [c] + [e] * 8 + [c] + [e] * 3, axis=-1)
        diag = qml.math.einsum("ij,jk->ijk", diag, I16)
        off_diag = qml.math.einsum("i,jk->ijk", s, DoubleExcitation.mask_s)
        return diag + off_diag

    def adjoint(self):
        (theta,) = self.parameters
        return DoubleExcitationPlus(-theta, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "G²₊", cache=cache)


class DoubleExcitationMinus(Operation):
    r"""
    Double excitation rotation with negative phase-shift outside the rotation subspace.

    This operation performs an :math:`SO(2)` rotation in the two-dimensional subspace :math:`\{
    |1100\rangle,|0011\rangle\}` while applying a phase-shift on other states. More precisely,
    it performs the transformation

    .. math::

        &|0011\rangle \rightarrow \cos(\phi/2) |0011\rangle - \sin(\phi/2) |1100\rangle\\
        &|1100\rangle \rightarrow \cos(\phi/2) |1100\rangle + \sin(\phi/2) |0011\rangle\\
        &|x\rangle \rightarrow e^{-i\phi/2} |x\rangle,

    for all other basis states :math:`|x\rangle`.

    **Details:**

    * Number of wires: 4
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_-(\phi)) = \frac{1}{2}\left[f(U_-(\phi+\pi/2)) - f(U_-(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`U_-(\phi)`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wires the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 4
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    """Gradient computation method."""

    parameter_frequencies = [(1,)]
    """Frequencies of the operation parameter with respect to an expectation value."""

    def generator(self):
        G = np.eye(16, dtype=complex64)
        G[3, 3] = G[12, 12] = 0
        G[3, 12] = -1j  # 3 (dec) = 0011 (bin)
        G[12, 3] = 1j  # 12 (dec) = 1100 (bin)
        H = csr_matrix(-0.5 * G)
        return qml.SparseHamiltonian(H, wires=self.wires)

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.DoubleExcitationMinus.matrix`


        Args:
          phi (tensor_like or float): rotation angle

        Returns:
          tensor_like: canonical matrix

        """
        c = qml.math.cos(phi / 2)
        s = qml.math.sin(phi / 2)

        interface = qml.math.get_interface(phi)

        if interface == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)

        e = qml.math.exp(-0.5j * phi)
        c = (1 + 0j) * c

        if qml.math.ndim(phi) == 0:
            diag = qml.math.diag([e] * 3 + [c] + [e] * 8 + [c] + [e] * 3)
            if qml.math.get_interface(phi) == "torch":
                return diag + s * qml.math.convert_like(DoubleExcitation.mask_s, phi)
            return diag + s * DoubleExcitation.mask_s

        diag = qml.math.stack([e] * 3 + [c] + [e] * 8 + [c] + [e] * 3, axis=-1)
        diag = qml.math.einsum("ij,jk->ijk", diag, I16)
        off_diag = qml.math.einsum("i,jk->ijk", s, DoubleExcitation.mask_s)
        return diag + off_diag

    def adjoint(self):
        (theta,) = self.parameters
        return DoubleExcitationMinus(-theta, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "G²₋", cache=cache)


class OrbitalRotation(Operation):
    r"""
    Spin-adapted spatial orbital rotation.

    For two neighbouring spatial orbitals :math:`\{|\Phi_{0}\rangle, |\Phi_{1}\rangle\}`, this operation
    performs the following transformation

    .. math::
        &|\Phi_{0}\rangle = \cos(\phi/2)|\Phi_{0}\rangle - \sin(\phi/2)|\Phi_{1}\rangle\\
        &|\Phi_{1}\rangle = \cos(\phi/2)|\Phi_{0}\rangle + \sin(\phi/2)|\Phi_{1}\rangle,

    with the same orbital operation applied in the :math:`\alpha` and :math:`\beta` spin orbitals.

    .. figure:: ../../_static/qchem/orbital_rotation_decomposition_extended.png
        :align: center
        :width: 100%
        :target: javascript:void(0);

    Here, :math:`G(\phi)` represents a single-excitation Givens rotation, implemented in PennyLane
    as the :class:`~.SingleExcitation` operation.

    **Details:**

    * Number of wires: 4
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: The ``OrbitalRotation`` operator has 4 equidistant frequencies
      :math:`\{0.5, 1, 1.5, 2\}`, and thus permits an 8-term parameter-shift rule.
      (see `Wierichs et al. (2022) <https://doi.org/10.22331/q-2022-03-30-677>`__).

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wires the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)

    **Example**

    .. code-block::

        >>> dev = qml.device('default.qubit', wires=4)
        >>> @qml.qnode(dev)
        ... def circuit(phi):
        ...     qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
        ...     qml.OrbitalRotation(phi, wires=[0, 1, 2, 3])
        ...     return qml.state()
        >>> circuit(0.1)
        array([ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                0.00249792+0.j,  0.        +0.j,  0.        +0.j,
               -0.04991671+0.j,  0.        +0.j,  0.        +0.j,
               -0.04991671+0.j,  0.        +0.j,  0.        +0.j,
                0.99750208+0.j,  0.        +0.j,  0.        +0.j,
                0.        +0.j])
    """
    num_wires = 4
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    """Gradient computation method."""

    parameter_frequencies = [(0.5, 1.0, 1.5, 2.0)]
    """Frequencies of the operation parameter with respect to an expectation value."""

    def generator(self):
        w0, w1, w2, w3 = self.wires
        return (
            0.25 * qml.PauliX(w0) @ qml.PauliY(w2)
            - 0.25 * qml.PauliY(w0) @ qml.PauliX(w2)
            + 0.25 * qml.PauliX(w1) @ qml.PauliY(w3)
            - 0.25 * qml.PauliY(w1) @ qml.PauliX(w3)
        )

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    mask_s = np.zeros((16, 16))
    mask_s[1, 4] = mask_s[2, 8] = mask_s[7, 13] = mask_s[11, 14] = -1
    mask_s[4, 1] = mask_s[8, 2] = mask_s[13, 7] = mask_s[14, 11] = 1

    mask_cs = np.zeros((16, 16))
    mask_cs[3, 6] = mask_cs[3, 9] = mask_cs[6, 12] = mask_cs[9, 12] = -1
    mask_cs[6, 3] = mask_cs[9, 3] = mask_cs[12, 6] = mask_cs[12, 9] = 1

    mask_s2 = np.zeros((16, 16))
    mask_s2[3, 12] = mask_s2[12, 3] = 1
    mask_s2[6, 9] = mask_s2[9, 6] = -1

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.OrbitalRotation.matrix`


        Args:
          phi (tensor_like or float): rotation angle

        Returns:
          tensor_like: canonical matrix
        """
        # This matrix is the "sign flipped" version of that on p18 of https://arxiv.org/abs/2104.05695,
        # where the sign flip is to adjust for the opposite convention used by authors for naming wires.
        # Additionally, there was a typo in the sign of a matrix element "s" at [2, 8], which is fixed here.

        c = qml.math.cos(phi / 2)
        s = qml.math.sin(phi / 2)

        if qml.math.ndim(phi) == 0:
            diag = qml.math.diag(
                [1.0, c, c, c**2, c, 1.0, c**2, c, c, c**2, 1.0, c, c**2, c, c, 1.0]
            )
            if qml.math.get_interface(phi) == "torch":
                mask_s = qml.math.convert_like(OrbitalRotation.mask_s, phi)
                mask_cs = qml.math.convert_like(OrbitalRotation.mask_cs, phi)
                mask_s2 = qml.math.convert_like(OrbitalRotation.mask_s2, phi)
                return diag + s * mask_s + (c * s) * mask_cs + s**2 * mask_s2
            return (
                diag
                + s * OrbitalRotation.mask_s
                + (c * s) * OrbitalRotation.mask_cs
                + s**2 * OrbitalRotation.mask_s2
            )

        ones = qml.math.ones_like(c)
        diag = qml.math.stack(
            [ones, c, c, c**2, c, ones, c**2, c, c, c**2, ones, c, c**2, c, c, ones],
            axis=-1,
        )

        diag = qml.math.einsum("ij,jk->ijk", diag, I16)
        off_diag = (
            qml.math.einsum("i,jk->ijk", s, OrbitalRotation.mask_s)
            + qml.math.einsum("i,jk->ijk", c * s, OrbitalRotation.mask_cs)
            + qml.math.einsum("i,jk->ijk", s**2, OrbitalRotation.mask_s2)
        )

        return diag + off_diag

    @staticmethod
    def compute_decomposition(phi, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.OrbitalRotation.decomposition`.

        For the source of this decomposition, see page 18 of
        `"Local, Expressive, Quantum-Number-Preserving VQE Ansatze for Fermionic Systems" <https://doi.org/10.1088/1367-2630/ac2cb3>`_ .

        Args:
            phi (float): rotation angle :math:`\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.OrbitalRotation.compute_decomposition(1.23, wires=(0,1,2,3))
        [Hadamard(wires=[3]),
        Hadamard(wires=[2]),
        CNOT(wires=[3, 1]),
        CNOT(wires=[2, 0]),
        RY(0.615, wires=[3]),
        RY(0.615, wires=[2]),
        RY(0.615, wires=[1]),
        RY(0.615, wires=[0]),
        CNOT(wires=[3, 1]),
        CNOT(wires=[2, 0]),
        Hadamard(wires=[3]),
        Hadamard(wires=[2])]

        """
        # This decomposition is the "upside down" version of that on p18 of https://arxiv.org/abs/2104.05695
        decomp_ops = [
            qml.Hadamard(wires=wires[3]),
            qml.Hadamard(wires=wires[2]),
            qml.CNOT(wires=[wires[3], wires[1]]),
            qml.CNOT(wires=[wires[2], wires[0]]),
            qml.RY(phi / 2, wires=wires[3]),
            qml.RY(phi / 2, wires=wires[2]),
            qml.RY(phi / 2, wires=wires[1]),
            qml.RY(phi / 2, wires=wires[0]),
            qml.CNOT(wires=[wires[3], wires[1]]),
            qml.CNOT(wires=[wires[2], wires[0]]),
            qml.Hadamard(wires=wires[3]),
            qml.Hadamard(wires=wires[2]),
        ]
        return decomp_ops

    def adjoint(self):
        (phi,) = self.parameters
        return OrbitalRotation(-phi, wires=self.wires)
