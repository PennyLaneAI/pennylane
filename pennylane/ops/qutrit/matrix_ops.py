import warnings
import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, Operation, DecompositionUndefinedError
from pennylane.wires import Wires


class QutritUnitary(Operation):
    r"""Apply an arbitrary, fixed unitary matrix.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        U (array[complex]): square unitary matrix
        wires(Sequence[int] or int): the wire(s) the operation acts on
    """
    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, *params, wires, do_queue=True):
        wires = Wires(wires)

        # For pure QutritUnitary operations (not controlled), check that the number
        # of wires fits the dimensions of the matrix
        U = params[0]

        dim = 3 ** len(wires)

        if qml.math.shape(U) != (dim, dim):
            raise ValueError(
                f"Input unitary must be of shape {(dim, dim)} to act on {len(wires)} wires."
            )

        # Check for unitarity; due to variable precision across the different ML frameworks,
        # here we issue a warning to check the operation, instead of raising an error outright.
        if not qml.math.is_abstract(U) and not qml.math.allclose(
            qml.math.dot(U, qml.math.T(qml.math.conj(U))),
            qml.math.eye(qml.math.shape(U)[0]),
            atol=1e-6,
        ):
            warnings.warn(
                f"Operator {U}\n may not be unitary."
                "Verify unitarity of operation, or use a datatype with increased precision.",
                UserWarning,
            )

        super().__init__(*params, wires=wires, do_queue=do_queue)

    @staticmethod
    def compute_matrix(U):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Args:
            U (tensor_like): unitary matrix

        Returns:
            tensor_like: canonical matrix
        """
        return U

    def adjoint(self):
        U = self.matrix()
        return QutritUnitary(qml.math.moveaxis(qml.math.conj(U), -2, -1), wires=self.wires)

    def pow(self, z):
        if isinstance(z, int):
            return [QutritUnitary(qml.math.linalg.matrix_power(self.matrix(), z), wires=self.wires)]
        return super().pow(z)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "U", cache=cache)
