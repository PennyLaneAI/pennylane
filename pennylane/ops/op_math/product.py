from functools import reduce
from multiprocessing.sharedctypes import Value

import numpy as np
import pennylane as qml

from pennylane import math
from pennylane.operation import Operator


def op_prod(*Prods, do_queue=True, id=None):
    r"""This constructs an operator that is the product of operators.
    All operators like qml.RX, qml.RZ and qml.RY are rotation matrices, thus the product is commutative and
    the order of the Operators does not matter.

    Args:
        *Prods (tuple[~.operation.Operator]): the operators we want to multiply together.

    Keyword Args:
        do_queue (bool): determines if the product operator will be queued (currently not supported).
            Default is True.
        id (str or None): id for the product operator. Default is None.

    Returns:
        ~ops.op_math.Product: Operator representign the product of operators

    ..seealso:: :class:`~.ops.op_math.prod`

    **Example**
    The product of qml.RX and qml.RZ must be equal to -iqml.RY. Which means that the matrix must be equal to 
    -i*array([[0,-i],[i,0]])
    >>> product_op = op_product(qml.PauliX(0), qml.PauliZ(0))
    >>> product_op
    PauliX(wires=[0]) * PauliZ(wires=[0])
    >>> prouct_op.matrix()
    array([[ 0,  -1],
           [ 1, 0]])
        
    """
    return Product(*Prods, id=id)


def _prod(mats_gen, dtype=None, cast_like=None):
    """
    Function that gives the multiplication of matrices from operators.
        Args:
        mats_gen (Generator): a python generator which produces the matrices which
        will be multiplied together.

    Keyword Args:
        dtype (str): a string representing the data type of the entries in the result.
        cast_like (Tensor): a tensor with the desired data type in its entries.

    """

    try:
        res = reduce(np.dot, mats_gen)
    except ValueError:
        print("The operators you have defined are not in the expected form")
    if dtype is not None:
        res = math.cast(res, dtype)
    if cast_like is not None:
        res = math.cast_like(res, cast_like)

    return res


class Product(Operator):
    r"""Symbolic operator representing the product of operators.

    Args:
        prods (tuple[~.operation.Operator]): a tuple of operators which will be multiplied together.

    Keyword Args:
        do_queue (bool): determines if the product operator will be queued (currently not supported).
            Default is True.
        id (str or None): id for the product operator. Default is None.

    
    **Example**
    The product of qml.PauliX and qml.PauliZ must be equal to -i*qml.PauliY. Which means that the matrix must be equal to 
    -i*array([[0,-i],[i,0]])
    >>> product_op = op_product(qml.PauliX(0), qml.PauliZ(0))
    >>> product_op
    PauliX(wires=[0]) * PauliZ(wires=[0])
    >>> prouct_op.matrix()
    array([[ 0,  -1],
           [ 1, 0]])
    >>> summed_op.terms()
    ([1.0, 1.0], (PauliX(wires=[0]), PauliZ(wires=[0])))

    .. details::
        :title: Usage Details

        We can have products of operators in the same Hilbert Space. 
        For example, multiplying operators whose matrices have the same dimensions. 
        Support for operators with different dimensions is not available yet.
    """

    _eigs = {}  # cache eigen vectors and values like in qml.Hermitian

    def __init__(self, *products, do_queue=True, id=None):  # pylint: disable=super-init-not-called
        """Initialize a Symbolic Operator class corresponding to the Product of operations."""
        self._name = "Prod"
        self._id = id
        self.queue_idx = None

        if len(products) < 2:
            raise ValueError(f"Require at least two operators to multiply; got {len(products)}")

        self.products = products
        self.data = [s.parameters for s in products]
        self._wires = qml.wires.Wires.all_wires([s.wires for s in products])

        if do_queue:
            self.queue()

    def __repr__(self):
        """Constructor-call-like representation."""
        return " * ".join([f"{f}" for f in self.products])

    def __copy__(self):
        cls = self.__class__
        copied_op = cls.__new__(cls)
        copied_op.data = self.data.copy()  # copies the combined parameters
        copied_op.products = tuple(s.__copy__() for s in self.products)

        for attr, value in vars(self).items():
            if attr not in {"data", "products"}:
                setattr(copied_op, attr, value)

        return copied_op

    @property
    def batch_size(self):
        """Batch size of input parameters."""
        raise ValueError("Batch size is not defined for Product operators.")

    @property
    def ndim_params(self):
        """ndim_params of input parameters."""
        raise ValueError(
            "Dimension of parameters is not currently implemented for Product operators."
        )

    @property
    def num_wires(self):
        return len(self.wires)

    @property
    def num_params(self):
        return sum(op.num_params for op in self.products)

    @property
    def is_hermitian(self):
        """If all of the terms in the product are hermitian, then the product is hermitian for rotation matrices"""
        return all(s.is_hermitian for s in self.products)

    @property
    def eigendecomposition(self):
        r"""Return the eigendecomposition of the matrix specified by the Hermitian observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        It transforms the input operator according to the wires specified.

        Returns:
            dict[str, array]: dictionary containing the eigenvalues and the eigenvectors of the operator
        """
        Hmat = self.matrix()
        Hmat = qml.math.to_numpy(Hmat)
        Hkey = tuple(Hmat.flatten().tolist())
        if Hkey not in self._eigs:
            w, U = np.linalg.eigh(Hmat)
            self._eigs[Hkey] = {"eigvec": U, "eigval": w}

        return self._eigs[Hkey]

    def eigvals(self):
        r"""Return the eigenvalues of the specified Hermitian observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        Returns:
            array: array containing the eigenvalues of the Hermitian observable
        """
        return self.eigendecomposition["eigval"]

    def matrix(self, wire_order=None):
        r"""Representation of the operator as a matrix in the computational basis.

        If ``wire_order`` is provided, the numerical representation considers the position of the
        operator's wires in the global wire order. Otherwise, the wire order defaults to the
        operator's wires.

        If the matrix depends on trainable parameters, the result
        will be cast in the same autodifferentiation framework as the parameters.

        A ``MatrixUndefinedError`` is raised if the matrix representation has not been defined.

        .. seealso:: :meth:`~.Operator.compute_matrix`

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels from the operator's wires

        Returns:
            tensor_like: matrix representation
        """

        def matrix_gen(products, wire_order=None):
            """Helper function to construct a generator of matrices"""
            for op in products:
                yield op.matrix(wire_order=wire_order)

        if wire_order is None:
            wire_order = self.wires

        return _prod(matrix_gen(self.products, wire_order))

    @property
    def _queue_category(self):  # don't queue Sum instances because it may not be unitary!
        """Used for sorting objects into their respective lists in `QuantumTape` objects.
        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Returns: None
        """
        return None

    def queue(self, context=qml.QueuingContext):
        """Updates each operator in the products owner to Sum, this ensures
        that the products are not applied to the circuit repeatedly."""
        for op in self.products:
            context.safe_update_info(op, owner=self)
        return self
