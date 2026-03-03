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
"""Tree representation of coefficients of a realspace operator"""

from __future__ import annotations

from enum import Enum
from itertools import product

import numpy as np
from numpy import allclose, isclose, ndarray, zeros


class RealspaceCoeffs:
    """Lightweight representation of a tensor of coefficients.

    The :class:`~.pennylane.labs.trotter_error.RealspaceCoeffs` object is initialized with an array
    and can be used to represent coefficients of a real space operator. A real space operator
    is constrcuted from position and momentum operators, e.g., Eq. 4
    of `arXiv:1703.09313 <https://arxiv.org/abs/1703.09313>`_ which represents a vibrational
    Hamiltonian.

    Args:
        tensor (ndarray): a numpy tensor
        label (string): name of the tensor

    **Examples**

    >>> import numpy as np
    >>> from pennylane.labs.trotter_error import RealspaceCoeffs
    >>> coeffs = np.array([[1, 0], [0, 1]])
    >>> rs_coeffs = RealspaceCoeffs(coeffs, label="alpha")
    >>> rs_coeffs.shape
    (2, 2)

    .. details::
         :title: Usage Details

         The :class:`~.pennylane.labs.trotter_error.RealspaceCoeffs` object allows arithmetic
         operations such as addition, subtraction, multiplication and matrix multiplication.
         Printing the resulting objects displays the expression that is used to compute each entry
         of the tensor.

         >>> coeffs1 = RealspaceCoeffs(np.array([[1, 0], [0, 1]]), label="alpha")
         >>> coeffs2 = RealspaceCoeffs(np.array([[2, 1], [1, 3]]), label="beta")
         >>> expr1 = coeffs1 + 2 * coeffs2
         >>> coeffs3 = RealspaceCoeffs(np.array([3, 2]), label="omega")
         >>> expr2 = expr1 @ coeffs3
         >>> expr2
         ((alpha[idx0,idx1]) + (2 * (beta[idx0,idx1]))) * (omega[idx2])
    """

    def __init__(self, tensor: np.ndarray, label: str = None):
        if label is None and tensor.shape != ():
            raise ValueError(f"A label is required for a tensor of shape {tensor.shape}.")

        self._tree = _RealspaceTree.tensor_node(tensor, label)

    @classmethod
    def _from_tree(cls, tree: _RealspaceTree):
        """Initialize directly from a ``_RealspaceTree`` object."""
        rs_coeffs = cls.__new__(cls)
        rs_coeffs._tree = tree
        return rs_coeffs

    def __add__(self, other: RealspaceCoeffs) -> RealspaceCoeffs:
        tree = _RealspaceTree.sum_node(self._tree, other._tree)
        return RealspaceCoeffs._from_tree(tree)

    def __sub__(self, other: RealspaceCoeffs) -> RealspaceCoeffs:
        tree = _RealspaceTree.sum_node(self._tree, _RealspaceTree.scalar_node(-1, other._tree))
        return RealspaceCoeffs._from_tree(tree)

    def __mul__(self, scalar: float) -> RealspaceCoeffs:
        tree = _RealspaceTree.scalar_node(scalar, self._tree)
        return RealspaceCoeffs._from_tree(tree)

    __rmul__ = __mul__

    def __matmul__(self, other: RealspaceCoeffs) -> RealspaceCoeffs:
        tree = _RealspaceTree.outer_node(self._tree, other._tree)
        return RealspaceCoeffs._from_tree(tree)

    def __repr__(self) -> str:
        return self._tree.__repr__()

    def __getitem__(self, index) -> float:
        return self._tree.compute(index)

    def __eq__(self, other: RealspaceCoeffs) -> bool:
        return self._tree == other._tree

    @property
    def is_zero(self) -> bool:
        """Determine if the :class:`~.pennylane.labs.trotter_error.RealspaceCoeffs` objects
        represents the zero tensor.

        Returns:
            bool: returns ``True`` when the tensor is zero, otherwise returns ``False``
        """
        return self._tree.is_zero

    @property
    def shape(self) -> tuple[int]:
        """Return the shape of the tensor."""
        return self._tree.shape

    def nonzero(self, threshold: float = 0.0):
        """Return the nonzero coefficients in a dictionary.

        Args:
            threshold (float): tolerance to return coefficients with magnitude greater than ``threshold``

        Returns:
            dict: a dictionary representation of the coefficient tensor

        **Example**

        >>> from pennylane.labs.trotter_error import RealspaceCoeffs
        >>> import numpy as np
        >>> node = RealspaceCoeffs(np.array([[1, 0, 0, 1], [0, 0, 1, 1]]), label="alpha")
        >>> node.nonzero()
        {(0, 0): 1, (0, 3): 1, (1, 2): 1, (1, 3): 1}
        """

        return self._tree.nonzero(threshold)


class _NodeType(Enum):
    """Enum containing the types of nodes"""

    SUM = 1
    OUTER = 2
    SCALAR = 3
    TENSOR = 4
    FLOAT = 5


class _RealspaceTree:  # pylint: disable=too-many-instance-attributes
    """
     A tree representing an expression that computes the coefficients of a :class:`~.pennylane.labs.trotter_error.RealspaceOperator`.
     This class should be instantiated from the following class methods:

        * ``tensor_node(tensor)``: a leaf node containing the coefficients as a tensor
        * ``outer_node(l_child, r_child)``: a node representing the outer product of two ``RealspaceCoeffs`` objects
        * ``sum_node(l_child, r_child)``: a node representing the sum of two ``RealspaceCoeffs`` objects
        * ``scalar_node(scalar, child)``: a node representing the product of a ``RealspaceCoeffs`` object by a scalar

    **Examples**

    Building a node representing a tensor of coefficients

    >>> from pennylane.labs.trotter_error import _RealspaceTree
    >>> import numpy as np
    >>> node = _RealspaceTree.tensor_node(np.array([[1, 2, 3], [4, 5, 6]]), label="alpha")
    >>> node
    alpha[idx0,idx1]

    Building a node representing the outer product of its children

    >>> from pennylane.labs.trotter_error import _RealspaceTree
    >>> import numpy as np
    >>> left_child = _RealspaceTree.tensor_node(np.array([1, 2, 3]), label="alpha")
    >>> right_child = _RealspaceTree.tensor_node(np.array([[1, 3, 4], [4, 5, 6]]), label="beta")
    >>> parent = _RealspaceTree.outer_node(left_child, right_child)
    >>> parent
    (alpha[idx0]) * (beta[idx1,idx2])

    Building a node representing the sum of its children

    >>> from pennylane.labs.trotter_error import _RealspaceTree
    >>> import numpy as np
    >>> left_child = _RealspaceTree.tensor_node(np.array([1, 2, 3]), label="alpha")
    >>> right_child = _RealspaceTree.tensor_node(np.array([4, 5, 6]), label="beta")
    >>> parent = _RealspaceTree.sum_node(left_child, right_child)
    >>> parent
    (alpha[idx0]) + (beta[idx0])

    Building a node representing the multiplication of its child by a scalar

    >>> from pennylane.labs.trotter_error import _RealspaceTree
    >>> import numpy as np
    >>> child = _RealspaceTree.tensor_node(np.array([[1, 2, 3], [4, 5, 6]]), label="alpha")
    >>> parent = _RealspaceTree.scalar_node(5, child)
    >>> parent
    5 * (alpha[idx0,idx1])
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        node_type: _NodeType,
        l_child: _RealspaceTree = None,
        r_child: _RealspaceTree = None,
        tensor: ndarray = None,
        scalar: float = None,
        value: float = None,
        label: str = None,
    ) -> _RealspaceTree:

        self.node_type = node_type
        self.l_child = l_child
        self.r_child = r_child
        self.tensor = tensor
        self.scalar = scalar
        self.value = value
        self.label = label

        if node_type == _NodeType.SUM:
            self.shape = l_child.shape
            self.is_zero = l_child.is_zero and r_child.is_zero
        elif node_type == _NodeType.OUTER:
            self.shape = l_child.shape + r_child.shape
            self.is_zero = l_child.is_zero or r_child.is_zero
        elif node_type == _NodeType.SCALAR:
            self.shape = l_child.shape
            self.is_zero = l_child.is_zero or isclose(scalar, 0)
        elif node_type == _NodeType.TENSOR:
            self.shape = tensor.shape
            self.is_zero = allclose(tensor, zeros(tensor.shape))
        elif node_type == _NodeType.FLOAT:
            self.shape = ()
            self.is_zero = isclose(value, 0)
        else:
            raise ValueError(f"Got invalid node type {node_type}.")

    @classmethod
    def sum_node(cls, l_child: _RealspaceTree, r_child: _RealspaceTree) -> _RealspaceTree:
        """Returns a ``_RealspaceTree`` representing the sum of the two children nodes.

        Args:
            l_child (_RealspaceTree): the left child
            r_child (_RealspaceTree): the right child

        Returns:
            _RealspaceTree: a `RealspaceCoeff` object representing the sum of `l_child` and `r_child`

        **Example**

        >>> from pennylane.labs.trotter_error import _RealspaceTree
        >>> import numpy as np
        >>> left_child = _RealspaceTree.tensor_node(np.array([1, 2, 3]), label="alpha")
        >>> right_child = _RealspaceTree.tensor_node(np.array([4, 5, 6]), label="beta")
        >>> parent = _RealspaceTree.sum_node(left_child, right_child)
        >>> parent
        >>> (alpha[idx0]) + (beta[idx0])

        """

        if l_child.shape != r_child.shape:
            raise ValueError(
                f"Cannot add _RealspaceTree of shape {l_child.shape} with _RealspaceTree of shape {r_child.shape}."
            )

        return cls(
            node_type=_NodeType.SUM,
            l_child=l_child,
            r_child=r_child,
        )

    @classmethod
    def outer_node(
        cls,
        l_child: _RealspaceTree,
        r_child: _RealspaceTree,
    ) -> _RealspaceTree:
        """Returns a ``_RealspaceTree`` representing the outer product of the two children nodes.

        Args:
            l_child (_RealspaceTree): the left child
            r_child (RealspaceCOeffs): the right child

        Returns:
            _RealspaceTree: a ``_RealspaceTree`` object representing the outer product of ``l_child`` and ``r_child``

        **Example**

        >>> from pennylane.labs.trotter_error import _RealspaceTree
        >>> import numpy as np
        >>> left_child = _RealspaceTree.tensor_node(np.array([1, 2, 3]), label="alpha")
        >>> right_child = _RealspaceTree.tensor_node(np.array([[1, 3, 4], [4, 5, 6]]), label="beta")
        >>> parent = _RealspaceTree.outer_node(left_child, right_child)
        >>> parent
        (alpha[idx0]) * (beta[idx1,idx2])
        """

        return cls(
            node_type=_NodeType.OUTER,
            l_child=l_child,
            r_child=r_child,
        )

    @classmethod
    def tensor_node(cls, tensor: ndarray, label: str = None) -> _RealspaceTree:
        """Returns a ``_RealspaceTree`` leaf node storing a tensor of coefficients.

        Args:
            tensor (ndarray): a tensor of coefficients
            label (string): a label for the tensor to be used when displaying the ``_RealspaceTree`` object as an expression

        Returns:
            _RealspaceTree: a ``_RealspaceTree`` object representing containing the tensor

        **Example**

        >>> from pennylane.labs.trotter_error import _RealspaceTree
        >>> import numpy as np
        >>> node = _RealspaceTree.tensor_node(np.array([[1, 2, 3], [4, 5, 6]]), label="alpha")
        >>> node
        alpha[idx0,idx1]
        """

        if len(tensor.shape):
            return cls(
                node_type=_NodeType.TENSOR,
                tensor=tensor,
                label=label,
            )

        return cls(
            node_type=_NodeType.FLOAT,
            value=tensor,
        )

    @classmethod
    def scalar_node(cls, scalar: float, child: _RealspaceTree) -> _RealspaceTree:
        """Returns a ``_RealspaceTree`` representing the scalar product of ``scalar`` and ``child``.

        Args:
            scalar (float): a scalar to multiply ``child`` by
            child (_RealspaceTree): the ``_RealspaceTree`` object to be multiplied by ``scalar``

        Returns:
            _RealspaceTree: a ``_RealspaceTree`` object representing the coefficients of ``child`` multiplied by ``scalar``

        **Example**

        >>> from pennylane.labs.trotter_error import _RealspaceTree
        >>> import numpy as np
        >>> child = _RealspaceTree.tensor_node(np.array([[1, 2, 3], [4, 5, 6]]), label="alpha")
        >>> parent = _RealspaceTree.scalar_node(5, child)
        >>> parent
        5 * (alpha[idx0,idx1])
        """

        return cls(
            node_type=_NodeType.SCALAR,
            l_child=child,
            scalar=scalar,
        )

    def __add__(self, other: _RealspaceTree) -> _RealspaceTree:
        return self.__class__.sum_node(self, other)

    def __mul__(self, scalar: float) -> _RealspaceTree:
        return self.__class__.scalar_node(scalar, self)

    __rmul__ = __mul__

    def __matmul__(self, other: _RealspaceTree) -> _RealspaceTree:
        return self.__class__.outer_node(self, other)

    def __eq__(self, other: _RealspaceTree) -> bool:  # pylint: disable=too-many-return-statements
        if self.node_type != other.node_type:
            return False

        if self.shape != other.shape:
            return False

        if self.node_type == _NodeType.OUTER:
            if self.l_child.shape != other.l_child.shape:
                return False

            if self.r_child.shape != other.r_child.shape:
                return False

            return (self.l_child == other.l_child) and (self.r_child == other.r_child)

        if self.node_type == _NodeType.SCALAR:
            if self.scalar != other.scalar:
                return False

            return self.l_child == other.l_child

        if self.node_type == _NodeType.SUM:
            return (self.l_child == other.l_child) and (self.r_child == other.r_child)

        if self.node_type == _NodeType.TENSOR:
            return allclose(self.tensor, other.tensor)

        if self.node_type == _NodeType.FLOAT:
            return self.value == other.value

        raise ValueError(f"_RealspaceTree was constructed with invalid _NodeType {self.node_type}.")

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        indices = [f"idx{i}" for i in range(len(self.shape))]

        return str(self._str(indices))

    # pylint: disable=protected-access
    def _str(self, indices) -> str:

        if self.node_type == _NodeType.TENSOR:
            return f"{self.label}[{','.join(indices)}]"
        if self.node_type == _NodeType.FLOAT:
            return f"{self.value}"
        if self.node_type == _NodeType.SCALAR:
            return f"{self.scalar} * ({self.l_child._str(indices)})"
        if self.node_type == _NodeType.OUTER:
            l_indices = indices[: len(self.l_child.shape)]
            r_indices = indices[len(self.l_child.shape) :]
            return f"({self.l_child._str(l_indices)}) * ({self.r_child._str(r_indices)})"
        if self.node_type == _NodeType.SUM:
            return f"({self.l_child._str(indices)}) + ({self.r_child._str(indices)})"

        raise ValueError(f"_RealspaceTree was constructed with invalid _NodeType {self.node_type}.")

    def compute(self, index: tuple[int]) -> float:
        """Evaluate the tree on a given ``index``.

        Args:
            index (Tuple[int]): the index of the coefficient to be computed

        Returns:
            float: the coefficient at the given index

        **Example**

        >>> from pennylane.labs.trotter_error import _RealspaceTree
        >>> import numpy as np
        >>> left_child = _RealspaceTree.tensor_node(np.array([1, 2, 3]), label="alpha")
        >>> right_child = _RealspaceTree.tensor_node(np.array([[1, 3, 4], [4, 5, 6]]), label="beta")
        >>> parent = _RealspaceTree.outer_node(left_child, right_child)
        >>> parent.compute((1, 1, 2))
        12
        """

        if not self._validate_index(index):
            raise ValueError(f"Given index {index} is not compatible with shape {self.shape}")

        if self.node_type == _NodeType.TENSOR:
            return self.tensor[index]
        if self.node_type == _NodeType.FLOAT:
            return self.value
        if self.node_type == _NodeType.SCALAR:
            return self.scalar * self.l_child.compute(index)
        if self.node_type == _NodeType.SUM:
            return self.l_child.compute(index) + self.r_child.compute(index)
        if self.node_type == _NodeType.OUTER:
            l_index = index[: len(self.l_child.shape)]
            r_index = index[len(self.l_child.shape) :]
            return self.l_child.compute(l_index) * self.r_child.compute(r_index)

        raise ValueError(f"_RealspaceTree was constructed with invalid _NodeType {self.node_type}.")

    def _validate_index(self, index: tuple[int]) -> bool:
        """Validate the shape of an index.

        Args:
            index (Tuple[int]): an index

        Returns:
            bool: True if ``index`` corresponds to a valid index of the tensor, False otherwise
        """

        if len(index) != len(self.shape):
            return False

        for x, y in zip(index, self.shape):
            if x < 0:
                return False

            if x >= y:
                return False

        return True

    def nonzero(self, threshold: float = 0.0):
        """Return the nonzero coefficients in a dictionary.

        Args:
            threshold (float): only return coefficients with magnitude greater than ``threshold``

        Returns:
            dict: a dictionary representation of the coefficient tensor

        **Example**

        >>> from pennylane.labs.trotter_error import _RealspaceTree
        >>> import numpy as np
        >>> node = _RealspaceTree.tensor_node(np.array([[1, 0, 0, 1], [0, 0, 1, 1]]), label="alpha")
        >>> node.nonzero()
        {(0, 0): 1, (0, 3): 1, (1, 2): 1, (1, 3): 1}
        """

        if self.node_type == _NodeType.TENSOR:
            return _numpy_to_dict(self.tensor, threshold)
        if self.node_type == _NodeType.FLOAT:
            return {(): self.value}
        if self.node_type == _NodeType.SCALAR:
            return _scale_dict(self.scalar, self.l_child.nonzero(threshold), threshold)
        if self.node_type == _NodeType.SUM:
            return _add_dicts(
                self.l_child.nonzero(threshold),
                self.r_child.nonzero(threshold),
                threshold,
            )
        if self.node_type == _NodeType.OUTER:
            return _mul_dicts(
                self.l_child.nonzero(threshold),
                self.r_child.nonzero(threshold),
                threshold,
            )

        raise ValueError(f"_RealspaceTree was constructed with invalid _NodeType {self.node_type}.")


def _add_dicts(d1: dict, d2: dict, threshold: float):
    """Add two coefficient dictionaries

    Args:
        d1 (dict): the first dictionary to be added
        d2 (dict): the second dictionary to be added
        threshold (float): only return coefficients with magnitude greater than ``threshold``

    Returns:
        dict: the sum of ``d1`` and ``d2``
    """
    add_dict = {}

    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())

    for key in d1_keys.intersection(d2_keys):
        if abs(d1[key] + d2[key]) > threshold:
            add_dict[key] = d1[key] + d2[key]

    for key in d1_keys.difference(d2_keys):
        add_dict[key] = d1[key]

    for key in d2_keys.difference(d1_keys):
        add_dict[key] = d2[key]

    return add_dict


def _mul_dicts(d1, d2, threshold):
    """Multiply two coefficient dictionaries

    Args:
        d1 (dict): the first dictionary to be multiplied
        d2 (dict): the second dictionary to be multiplied
        threshold (float): only return coefficients with magnitude greater than ``threshold``

    Returns:
        dict: the outer product of ``d1`` and ``d2``
    """
    mul_dict = {}

    for key1, key2 in product(d1.keys(), d2.keys()):
        if abs(d1[key1] * d2[key2]) > threshold:
            mul_dict[key1 + key2] = d1[key1] * d2[key2]

    return mul_dict


def _scale_dict(scalar, d, threshold):
    """Multiply a coefficient dictionary by a scalar

    Args:
        d (dict): the dictionary to be scaled
        scalar (float): the scalar to multiply ``d`` by
        threshold (float): only return coefficients with magnitude greater than ``threshold``

    Returns:
        dict: the product of ``d`` and ``scalar``
    """
    scaled = {}
    for key in d.keys():
        if abs(scalar * d[key]) > threshold:
            scaled[key] = scalar * d[key]

    return scaled


def _numpy_to_dict(arr, threshold):
    """Returns a dictionary representation of a numpy array

    Args:
        arr (ndarray): a numpy array
        threshold (float): only return coefficients with magnitude greater than ``threshold``

    Returns:
        dict: a dictionary representation of the numpy array
    """
    nz = arr.nonzero()
    d = {}

    for index in zip(*nz):
        if abs(arr[index]) > threshold:
            index = tuple(map(int, index))
            d[index] = float(arr[index])

    return d
