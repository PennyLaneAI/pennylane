# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
core parameterized gates for qutrits.
"""
import itertools
import numpy as np
import pennylane as qml
from pennylane.operation import Operation


class TRX(Operation):
    r"""
    The single qutrit X rotation

    Performs the RX operation on the specified 2D subspace. The subspace is
    given as a keyword argument and determines which two of three single-qutrit
    basis states the operation applies to.

    The construction of this operator is based on section 3 of
    `Di et al. (2012) <https://arxiv.org/abs/1105.5485>`_.

    .. math:: TRX^{jk}(\phi) = \exp(-i\phi\sigma_x^{jk}/2),

    where :math:`\sigma_x^{jk} = |j\rangle\langle k| + |k\rangle\langle j|;`
    :math:`j, k \in \{0, 1, 2\}, j < k`.

    .. seealso:: :class:`~.RX`

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        subspace (Sequence[int]): the 2D subspace on which to apply operation
        id (str or None): String representing the operation (optional)

    **Example**

    The specified subspace will determine which basis states the operation actually
    applies to:

    >>> qml.TRX(0.5, wires=0, subspace=(0, 1)).matrix()
    array([[0.96891242+0.j        , 0.        -0.24740396j, 0.        +0.j        ],
           [0.        -0.24740396j, 0.96891242+0.j        , 0.        +0.j        ],
           [0.        +0.j        , 0.        +0.j        , 1.        +0.j        ]])

    >>> qml.TRX(0.5, wires=0, subspace=(0, 2)).matrix()
    array([[0.96891242+0.j        , 0.        +0.j        , 0.        -0.24740396j],
           [0.        +0.j        , 1.        +0.j        , 0.        +0.j        ],
           [0.        -0.24740396j, 0.        +0.j        , 0.96891242+0.j        ]])

    >>> qml.TRX(0.5, wires=0, subspace=(1, 2)).matrix()
    array([[1.        +0.j        , 0.        +0.j        , 0.        +0.j        ],
           [0.        +0.j        , 0.96891242+0.j        , 0.        -0.24740396j],
           [0.        +0.j        , 0.        -0.24740396j, 0.96891242+0.j        ]])
    """
    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(0.5, 1)]

    # Internal dictionary to map subpsaces to Gell-Mann observable for the generator
    _index_dict = {(0, 1): 1, (0, 2): 4, (1, 2): 6}

    def generator(self):
        return qml.s_prod(-0.5, qml.GellMann(self.wires, index=self._index_dict[self.subspace]))

    def __init__(self, phi, wires, subspace=(0, 1), id=None):
        self._subspace = self.validate_subspace(subspace)
        self._hyperparameters = {
            "subspace": self._subspace,
        }
        super().__init__(phi, wires=wires, id=id)

    @property
    def subspace(self):
        """The single-qutrit basis states which the operator acts on

        This subspace determines which two single-qutrit basis states the operator acts on.
        The remaining basis state is not affected by the operator.

        Returns:
            tuple[int]: subspace on which operator acts
        """
        return self._subspace

    @staticmethod
    def compute_matrix(
        theta, subspace=(0, 1)
    ):  # pylint: disable=arguments-differ,dangerous-default-value
        r"""Representation of the operator as a canonical matrix in the computational basis.

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TRX.matrix`

        Args:
            theta (tensor_like or float): rotation angle
            subspace (Sequence[int]): the 2D subspace on which to apply the operation

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.TRX.compute_matrix(torch.tensor(0.5), subspace=(0, 2))
        tensor([[0.9689+0.0000j, 0.0000+0.0000j, 0.0000-0.2474j],
                [0.0000+0.0000j, 1.0000+0.0000j, 0.0000+0.0000j],
                [0.0000-0.2474j, 0.0000+0.0000j, 0.9689+0.0000j]])
        """
        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        if qml.math.get_interface(theta) == "tensorflow":
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)

        # The following avoids casting an imaginary quantity to reals when backpropagating
        c = (1 + 0j) * c
        js = -1j * s

        shape = qml.math.shape(theta)
        is_broadcasted = len(shape) != 0
        # Construct identity matrices and cast to complex type
        mat = (
            qml.math.tensordot([1] * shape[0], qml.math.eye(3), axes=0)
            if is_broadcasted
            else qml.math.eye(3)
        )
        mat = qml.math.cast_like(mat, js)

        # Create slices that determine the indices at which the rotation terms of the matrix should be
        slices = tuple(itertools.product(subspace, subspace))
        if is_broadcasted:
            slices = [(Ellipsis, *s) for s in slices]

        # Put rotation terms in the appropriate indices using the slices
        mat[slices[0]] = mat[slices[3]] = c
        mat[slices[1]] = mat[slices[2]] = js

        return qml.math.convert_like(mat, theta)

    def adjoint(self):
        return TRX(-self.data[0], wires=self.wires, subspace=self.subspace)

    def pow(self, z):
        return [TRX(self.data[0] * z, wires=self.wires, subspace=self.subspace)]


class TRY(Operation):
    r"""
    The single qutrit Y rotation

    Performs the RY operation on the specified 2D subspace. The subspace is
    given as a keyword argument and determines which two of three single-qutrit
    basis states the operation applies to.

    The construction of this operator is based on section 3 of
    `Di et al. (2012) <https://arxiv.org/abs/1105.5485>`_.

    .. math:: TRY^{jk}(\phi) = \exp(-i\phi\sigma_y^{jk}/2),

    where :math:`\sigma_y^{jk} = -i |j\rangle\langle k| + i |k\rangle\langle j|;`
    :math:`j, k \in \{0, 1, 2\}, j < k`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        subspace (Sequence[int]): the 2D subspace on which to apply operation
        id (str or None): String representing the operation (optional)

    **Example**

    The specified subspace will determine which basis states the operation actually
    applies to:

    >>> qml.TRY(0.5, wires=0, subspace=(0, 1)).matrix()
    array([[ 0.96891242+0.j, -0.24740396-0.j,  0.        +0.j],
           [ 0.24740396+0.j,  0.96891242+0.j,  0.        +0.j],
           [ 0.        +0.j,  0.        +0.j,  1.        +0.j]])

    >>> qml.TRY(0.5, wires=0, subspace=(0, 2)).matrix()
    array([[ 0.96891242+0.j,  0.        +0.j, -0.24740396-0.j],
           [ 0.        +0.j,  1.        +0.j,  0.        +0.j],
           [ 0.24740396+0.j,  0.        +0.j,  0.96891242+0.j]])

    >>> qml.TRY(0.5, wires=0, subspace=(1, 2)).matrix()
    array([[ 1.        +0.j,  0.        +0.j,  0.        +0.j],
           [ 0.        +0.j,  0.96891242+0.j, -0.24740396-0.j],
           [ 0.        +0.j,  0.24740396+0.j,  0.96891242+0.j]])
    """
    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(0.5, 1)]

    # Internal dictionary to map subpsaces to Gell-Mann observable for the generator
    _index_dict = {(0, 1): 2, (0, 2): 5, (1, 2): 7}

    def generator(self):
        return qml.s_prod(-0.5, qml.GellMann(self.wires, index=self._index_dict[self.subspace]))

    def __init__(self, phi, wires, subspace=(0, 1), id=None):
        self._subspace = self.validate_subspace(subspace)
        self._hyperparameters = {
            "subspace": self._subspace,
        }
        super().__init__(phi, wires=wires, id=id)

    @property
    def subspace(self):
        """The single-qutrit basis states which the operator acts on

        This subspace determines which two single-qutrit basis states the operator acts on.
        The remaining basis state is not affected by the operator.

        Returns:
            tuple[int]: subspace on which operator acts
        """
        return self._subspace

    @staticmethod
    def compute_matrix(theta, subspace=(0, 1)):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TRY.matrix`

        Args:
            theta (tensor_like or float): rotation angle
            subspace (Sequence[int]): the 2D subspace on which to apply operation

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.TRY.compute_matrix(torch.tensor(0.5), subspace=(0, 2))
        tensor([[ 0.9689+0.j,  0.0000+0.j, -0.2474-0.j],
                [ 0.0000+0.j,  1.0000+0.j,  0.0000+0.j],
                [ 0.2474+0.j,  0.0000+0.j,  0.9689+0.j]])
        """
        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        if qml.math.get_interface(theta) == "tensorflow":
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)

        # The following avoids casting an imaginary quantity to reals when backpropagating
        c = (1 + 0j) * c
        s = (1 + 0j) * s

        shape = qml.math.shape(theta)
        is_broadcasted = len(shape) != 0
        # Construct identity matrices and cast to complex type
        mat = (
            qml.math.tensordot([1] * qml.math.shape(theta)[0], qml.math.eye(3), axes=0)
            if is_broadcasted
            else qml.math.eye(3)
        )
        mat = qml.math.cast_like(mat, s)

        slices = tuple(itertools.product(subspace, subspace))
        if is_broadcasted:
            slices = [(Ellipsis, *s) for s in slices]

        # Put rotation terms in the appropriate indices using the slices
        mat[slices[0]] = mat[slices[3]] = c
        mat[slices[1]] = -s
        mat[slices[2]] = s

        return qml.math.convert_like(mat, theta)

    def adjoint(self):
        return TRY(-self.data[0], wires=self.wires, subspace=self.subspace)

    def pow(self, z):
        return [TRY(self.data[0] * z, wires=self.wires, subspace=self.subspace)]


class TRZ(Operation):
    r"""The single qutrit Z rotation

    Performs the RZ operation on the specified 2D subspace. The subspace is
    given as a keyword argument and determines which two of three single-qutrit
    basis states the operation applies to.

    The construction of this operator is based on section 3 of
    `Di et al. (2012) <https://arxiv.org/abs/1105.5485>`_.

    .. math:: TRZ^{jk}(\phi) = \exp(-i\phi\sigma_z^{jk}/2),

    where :math:`\sigma_z^{jk} = |j\rangle\langle j| - |k\rangle\langle k|;`
    :math:`j, k \in \{0, 1, 2\}, j < k`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        subspace (Sequence[int]): the 2D subspace on which to apply operation
        id (str or None): String representing the operation (optional)

    **Example**

    The specified subspace will determine which basis states the operation actually
    applies to:

    >>> qml.TRZ(0.5, wires=0, subspace=(0, 1)).matrix()
    array([[0.96891242-0.24740396j, 0.        +0.j        , 0.        +0.j        ],
           [0.        +0.j        , 0.96891242+0.24740396j, 0.        +0.j        ],
           [0.        +0.j        , 0.        +0.j        , 1.        +0.j        ]])

    >>> qml.TRZ(0.5, wires=0, subspace=(0, 2)).matrix()
    array([[0.96891242-0.24740396j, 0.        +0.j        , 0.        +0.j        ],
           [0.        +0.j        , 1.        +0.j        , 0.        +0.j        ],
           [0.        +0.j        , 0.        +0.j        , 0.96891242+0.24740396j]])

    >>> qml.TRZ(0.5, wires=0, subspace=(1, 2)).matrix()
    array([[1.        +0.j        , 0.        +0.j        , 0.        +0.j        ],
           [0.        +0.j        , 0.96891242-0.24740396j, 0.        +0.j        ],
           [0.        +0.j        , 0.        +0.j        , 0.96891242+0.24740396j]])
    """
    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(0.5, 1)]

    def generator(self):
        if self.subspace == (0, 1):
            return qml.s_prod(-0.5, qml.GellMann(wires=self.wires, index=3))

        if self.subspace == (0, 2):
            coeffs = [-0.25, -0.25 * np.sqrt(3)]
            obs = [qml.GellMann(wires=self.wires, index=3), qml.GellMann(wires=self.wires, index=8)]
            return qml.dot(coeffs, obs)

        coeffs = [0.25, -0.25 * np.sqrt(3)]
        obs = [qml.GellMann(wires=self.wires, index=3), qml.GellMann(wires=self.wires, index=8)]
        return qml.dot(coeffs, obs)

    def __init__(self, phi, wires, subspace=(0, 1), id=None):
        self._subspace = self.validate_subspace(subspace)
        self._hyperparameters = {
            "subspace": self._subspace,
        }
        super().__init__(phi, wires=wires, id=id)

    @property
    def subspace(self):
        """The single-qutrit basis states which the operator acts on

        This subspace determines which two single-qutrit basis states the operator acts on.
        The remaining basis state is not affected by the operator.

        Returns:
            tuple[int]: subspace on which operator acts
        """
        return self._subspace

    @staticmethod
    def compute_matrix(theta, subspace=(0, 1)):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TRZ.matrix`

        Args:
            theta (tensor_like or float): rotation angle
            subspace (Sequence[int]): the 2D subspace on which to apply operation

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.TRZ.compute_matrix(torch.tensor(0.5), subspace=(0, 2))
        tensor([[0.9689-0.2474j, 0.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 1.0000+0.0000j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000+0.0000j, 0.9689+0.2474j]])
        """
        if qml.math.get_interface(theta) == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)
        p = qml.math.exp(-1j * theta / 2)

        shape = qml.math.shape(theta)
        is_broadcasted = len(shape) != 0
        # Construct identity matrices and cast to complex type
        mat = (
            qml.math.tensordot([1] * shape[0], qml.math.eye(3), axes=0)
            if is_broadcasted
            else qml.math.eye(3)
        )
        mat = qml.math.cast_like(mat, p)

        slices = [(Ellipsis, i, i) for i in subspace]

        mat[slices[0]] = p
        mat[slices[1]] = qml.math.conj(p)

        return qml.math.convert_like(mat, theta)

    def adjoint(self):
        return TRZ(-self.data[0], wires=self.wires, subspace=self.subspace)

    def pow(self, z):
        return [TRZ(self.data[0] * z, wires=self.wires, subspace=self.subspace)]
