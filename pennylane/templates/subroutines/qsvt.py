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
"""
Contains the QSVT template and qsvt wrapper function.
"""

import copy
import math
from collections.abc import Sequence
from functools import reduce
from typing import Literal

import numpy as np
import scipy
from numpy.polynomial import Polynomial, chebyshev

from pennylane import math, ops
from pennylane.operation import Operation, Operator
from pennylane.queuing import QueuingManager, apply
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .fable import FABLE
from .prepselprep import PrepSelPrep
from .qubitization import Qubitization


def _pauli_rep_process(A, poly, encoding_wires, block_encoding, angle_solver="root-finding"):

    if block_encoding not in ["prepselprep", "qubitization", None]:
        raise ValueError(
            f"block_encoding = {block_encoding} not supported for A of type {type(A)}. "
            "When A is a Hamiltonian or has a Pauli decomposition, block_encoding should "
            "take the value 'prepselprep' or 'qubitization'. Otherwise, please provide the "
            "matrix of the Hamiltonian as input. For more details, see the 'qml.matrix' function."
        )

    if any(wire in Wires(encoding_wires) for wire in A.wires):
        raise ValueError(
            f"Control wires in '{block_encoding}' should be different from the hamiltonian wires"
        )

    # compute angles
    angles = poly_to_angles(poly, "QSVT", angle_solver=angle_solver)

    encoding = (
        Qubitization(A, control=encoding_wires)
        if block_encoding == "qubitization"
        else PrepSelPrep(A, control=encoding_wires)
    )

    projectors = [
        ops.PCPhase(angle, dim=2 ** len(A.wires), wires=encoding_wires + A.wires)
        for angle in angles
    ]
    return encoding, projectors


def _tensorlike_process(A, poly, encoding_wires, block_encoding, angle_solver="root-finding"):
    if block_encoding not in ["embedding", "fable", None]:
        raise ValueError(
            f"block_encoding = {block_encoding} not supported for A of type {type(A)}."
            "When A is a matrix block_encoding should take the value 'embedding' or 'fable'. "
            "Otherwise, please provide an input with a Pauli decomposition. For more details, "
            "see the 'qml.pauli_decompose' function."
        )

    # compute angles
    angles = poly_to_angles(poly, "QSVT", angle_solver=angle_solver)

    A = math.atleast_2d(A)
    max_dimension = 1 if len(math.array(A).shape) == 0 else max(A.shape)

    if block_encoding == "fable":
        if math.linalg.norm(max_dimension * math.ravel(A), np.inf) > 1:
            raise ValueError(
                "The subnormalization factor should be lower than 1. Ensure that the product"
                " of the maximum dimension of A and its square norm is less than 1."
            )

        # FABLE encodes A / 2^n, need to rescale to obtain desired block-encoding

        fable_norm = int(np.ceil(np.log2(max_dimension)))
        encoding = FABLE(2**fable_norm * A, wires=encoding_wires)

        projectors = [ops.PCPhase(angle, dim=len(A), wires=encoding_wires) for angle in angles]

    else:
        c, r = math.shape(A)

        projectors = []
        for idx, phi in enumerate(angles):
            dim = c if idx % 2 else r
            projectors.append(ops.PCPhase(phi, dim=dim, wires=encoding_wires))

        encoding = ops.BlockEncode(A, wires=encoding_wires)

    return encoding, projectors


def qsvt(
    A: Operator | TensorLike,
    poly: TensorLike,
    encoding_wires: Sequence,
    block_encoding: Literal[None, "prepselprep", "qubitization", "embedding", "fable"] = None,
    angle_solver="root-finding",
):
    r"""
    Implements the Quantum Singular Value Transformation (QSVT) for a matrix or Hamiltonian ``A``,
    using a polynomial defined by ``poly`` and a block encoding specified by ``block_encoding``.

    .. math::

        \begin{pmatrix}
        A & * \\
        * & * \\
        \end{pmatrix}
        \xrightarrow{QSVT}
        \begin{pmatrix}
        \text{poly}(A) + i \dots & * \\
        * & * \\
        \end{pmatrix}

    The polynomial transformation is encoded as the real part of the top left term after applying the operator.

    This function calculates the required phase angles from the polynomial using :func:`~.poly_to_angles`.

    .. note::

        The function :func:`~.poly_to_angles`, used within ``qsvt``, is not JIT-compatible, which
        prevents ``poly`` from being traceable in ``qsvt``. However, ``A`` is traceable
        and can be optimized by JIT within this function.

    Args:

        A (Union[tensor_like, Operator]): The matrix on which the QSVT will be applied.
            This can be an array or an object that has a Pauli representation. See :func:`~.pauli_decompose`.

        poly (tensor_like): coefficients of the polynomial ordered from lowest to highest power

        encoding_wires (Sequence[int]): The qubit wires used for the block encoding. See Usage Details below for
            more information on ``encoding_wires`` depending on the block encoding used.

        block_encoding (str): Specifies the type of block encoding to use. Options include:

            - ``"prepselprep"``: Embeds the Hamiltonian ``A`` using :class:`~pennylane.PrepSelPrep`.
              Default encoding for Hamiltonians.
            - ``"qubitization"``: Embeds the Hamiltonian ``A`` using :class:`~pennylane.Qubitization`.
            - ``"embedding"``: Embeds the matrix ``A`` using :class:`~pennylane.BlockEncode`.
              Template not hardware compatible. Default encoding for matrices.
            - ``"fable"``: Embeds the matrix ``A`` using :class:`~pennylane.FABLE`. Template hardware compatible.

        angle_solver (str): Specifies the method used to calculate the angles of the routine
            via :func:`poly_to_angles <pennylane.poly_to_angles>`. Options include:

            - ``"root-finding"``: effective for polynomials of degree up to :math:`\sim 1000`
            - ``"iterative"``: effective for polynomials of degree higher than :math:`\sim 1000`

    Returns:
        (Operator): A quantum operator implementing QSVT on the matrix ``A`` with the
        specified encoding and projector phases.

    .. seealso:: :class:`~.QSVT`

    Example:

    .. code-block:: python

        # P(x) = -x + 0.5 x^3 + 0.5 x^5
        poly = np.array([0, -1, 0, 0.5, 0, 0.5])

        hamiltonian = qml.dot([0.3, 0.7], [qml.Z(1), qml.X(1) @ qml.Z(2)])

        dev = qml.device("default.qubit")


        @qml.qnode(dev)
        def circuit():
            qml.qsvt(hamiltonian, poly, encoding_wires=[0], block_encoding="prepselprep")
            return qml.state()


        matrix = qml.matrix(circuit, wire_order=[0, 1, 2])()

    >>> print(matrix[:4, :4].real) # doctest: +SKIP
    [[-0.1625  0.     -0.3793  0.    ]
     [ 0.     -0.1625  0.      0.3793]
     [-0.3793  0.      0.1625  0.    ]
     [ 0.      0.3793  0.      0.1625]]


    .. details::
        :title: Usage Details

        If the input ``A`` is a Hamiltonian, the valid ``block_encoding`` values are
        ``"prepselprep"`` and ``"qubitization"``. In this case, ``encoding_wires`` refers to the
        ``control`` parameter in the templates :class:`~pennylane.PrepSelPrep` and :class:`~pennylane.Qubitization`,
        respectively. These wires represent the auxiliary qubits necessary for the block encoding of
        the Hamiltonian. The number of ``encoding_wires`` required must be :math:`\lceil \log_2(m) \rceil`,
        where :math:`m` is the number of terms in the Hamiltonian.

        .. code-block:: python

            # P(x) = -1 + 0.2 x^2 + 0.5 x^4
            poly = np.array([-1, 0, 0.2, 0, 0.5])

            hamiltonian = qml.dot([0.3, 0.4, 0.3], [qml.Z(2), qml.X(2) @ qml.Z(3), qml.X(2)])

            dev = qml.device("default.qubit")

            @qml.qnode(dev)
            def circuit():
                qml.qsvt(hamiltonian, poly, encoding_wires=[0, 1], block_encoding="prepselprep")
                return qml.state()


            matrix = qml.matrix(circuit, wire_order=[0, 1, 2, 3])()

        >>> print(np.round(matrix[:4, :4], 4).real) # doctest: +SKIP
        [[-0.7158  0.     -0.      0.    ]
         [ 0.     -0.975   0.     -0.    ]
         [ 0.      0.     -0.7158  0.    ]
         [ 0.      0.      0.     -0.975 ]]


        Alternatively, if the input ``A`` is a matrix, the valid values for ``block_encoding`` are
        ``"embedding"`` and ``"fable"``. In this case, the ``encoding_wires`` parameter corresponds to
        the ``wires`` attribute in the templates :class:`~pennylane.BlockEncode` and :class:`~pennylane.FABLE`,
        respectively. Note that for QSVT to work, the input matrix must be Hermitian.

        .. code-block:: python

            # P(x) = -1 + 0.2 x^2 + 0.5 x^4
            poly = np.array([-0.1, 0, 0.2, 0, 0.5])

            A = np.array([[-0.1, 0, 0, 0.1], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.1, 0, -0.2, -0.1]])

            dev = qml.device("default.qubit")

            @qml.qnode(dev)
            def circuit():
                qml.qsvt(A, poly, encoding_wires=[0, 1, 2, 3, 4], block_encoding="fable")
                return qml.state()

            matrix = qml.matrix(circuit, wire_order=[0, 1, 2, 3, 4])()

        >>> print(np.round(matrix[:4, :4], 4).real) # doctest: +SKIP
        [[-0.0954  0.     -0.0056 -0.0054]
         [-0.     -0.0912  0.      0.    ]
         [-0.0056  0.     -0.0788  0.0164]
         [-0.0054  0.      0.0164 -0.0842]]

        Note that for the FABLE block encoding to function correctly, it must comply with the following:

        .. math::

                d \|A\|^2 \leq 1,

        where :math:`d` is the maximum dimension of :math:`A` and :math:`\|A\|` is the 2-norm of :math:`A`.
        In the previous example this is satisfied since :math:`d = 4` and :math:`\|A\|^2 = 0.2`:

        >>> print(4 * np.linalg.norm(A, ord='fro')**2)
        0.80...


    """

    # If the input A is a Hamiltonian
    if hasattr(A, "pauli_rep"):
        encoding, projectors = _pauli_rep_process(
            A, poly, encoding_wires, block_encoding, angle_solver=angle_solver
        )
    else:
        encoding, projectors = _tensorlike_process(
            A, poly, encoding_wires, block_encoding, angle_solver=angle_solver
        )

    return QSVT(encoding, projectors)


class QSVT(Operation):
    r"""QSVT(UA,projectors)
    Implements the
    `quantum singular value transformation <https://arxiv.org/abs/1806.01838>`__ (QSVT) circuit.

    .. note ::

        This template allows users to define hardware-compatible block encoding and
        projector-controlled phase shift circuits. For a QSVT implementation that is
        tailored to work directly with an input matrix and a transformation polynomial
        see :func:`~.qsvt`.

    Given an :class:`~.Operator` :math:`U`, which block encodes the matrix :math:`A`, and a list of
    projector-controlled phase shift operations :math:`\vec{\Pi}_\phi`, this template applies a
    circuit for the quantum singular value transformation as follows.

    When the number of projector-controlled phase shifts is even (:math:`d` is odd), the QSVT
    circuit is defined as:

    .. math::

        U_{QSVT} = \tilde{\Pi}_{\phi_1}U\left[\prod^{(d-1)/2}_{k=1}\Pi_{\phi_{2k}}U^\dagger
        \tilde{\Pi}_{\phi_{2k+1}}U\right]\Pi_{\phi_{d+1}}.


    And when the number of projector-controlled phase shifts is odd (:math:`d` is even):

    .. math::

        U_{QSVT} = \left[\prod^{d/2}_{k=1}\Pi_{\phi_{2k-1}}U^\dagger\tilde{\Pi}_{\phi_{2k}}U\right]
        \Pi_{\phi_{d+1}}.

    This circuit applies a polynomial transformation (:math:`Poly^{SV}`) to the singular values of
    the block encoded matrix:

    .. math::

        \begin{align}
             U_{QSVT}(A, \vec{\phi}) &=
             \begin{bmatrix}
                Poly^{SV}(A) & \cdot \\
                \cdot & \cdot
            \end{bmatrix}.
        \end{align}

    .. seealso::

        :func:`~.qsvt` and `A Grand Unification of Quantum Algorithms <https://arxiv.org/pdf/2105.02859.pdf>`_.

    Args:
        UA (Operator): the block encoding circuit, specified as an :class:`~.Operator`,
            like :class:`~.BlockEncode`
        projectors (Sequence[Operator]): a list of projector-controlled phase
            shifts that implement the desired polynomial

    Raises:
        ValueError: if the input block encoding is not an operator

    **Example**

    To implement QSVT in a circuit, we can use the following method:

    >>> dev = qml.device("default.qubit", wires=[0])
    >>> block_encoding = qml.Hadamard(wires=0)  # note H is a block encoding of 1/sqrt(2)
    >>> phase_shifts = [qml.RZ(-2 * theta, wires=0) for theta in (1.23, -0.5, 4)]  # -2*theta to match convention

    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.QSVT(block_encoding, phase_shifts)
    ...     return qml.expval(qml.Z(0))
    ... 
    
    >>> example_circuit()
    np.float64(0.5403...)

    We can visualize the circuit as follows:

    >>> print(qml.draw(example_circuit)())
    0: ──QSVT─┤  <Z>

    To see the implementation details, we can expand the circuit:

    >>> q_script = qml.tape.QuantumScript(ops=[qml.QSVT(block_encoding, phase_shifts)])
    >>> print(q_script.expand().draw(decimals=2))
    0: ──RZ(-2.46)──H──RZ(1.00)──H†──RZ(-8.00)─┤

    See the Usage Details section for more examples on implementing QSVT with different block
    encoding methods.

    .. details::
        :title: Usage Details

        The QSVT operation can be used with different block encoding methods, depending on the
        initial operator for which the singular value transformation is applied and the desired
        backend device. Examples are provided below.

        If we want to transform the singular values of a matrix,
        the matrix can be block-encoded with either the :class:`~.BlockEncode` or :class:`~.FABLE`
        operations. Note that :class:`~.BlockEncode` is more efficient on simulator devices but
        it cannot be used with hardware backends because it currently has no gate decomposition.
        The :class:`~.FABLE` operation is less efficient on simulator devices but is hardware
        compatible.

        The following example applies the polynomial :math:`p(x) = -x + 0.5x^3 + 0.5x^5` to an
        arbitrary hermitian matrix using :class:`~.BlockEncode` for block encoding.

        .. code-block:: python

            poly = np.array([0, -1, 0, 0.5, 0, 0.5])
            angles = qml.poly_to_angles(poly, "QSVT")
            input_matrix = np.array([[0.2, 0.1], [0.1, -0.1]])

            wires = [0, 1]
            block_encode = qml.BlockEncode(input_matrix, wires=wires)
            projectors = [
                qml.PCPhase(angles[i], dim=len(input_matrix), wires=wires)
                for i in range(len(angles))
            ]

            dev = qml.device("default.qubit")
            @qml.qnode(dev)
            def circuit():
                qml.QSVT(block_encode, projectors)
                return qml.state()

        >>> circuit() # doctest: +SKIP
        array([-0.1942+0.6665j, -0.0979+0.3583j,  0.332 -0.5105j, -0.0955+0.0104j])

        If we want to transform the singular values of a linear
        combination of unitaries, e.g., a Hamiltonian, it can be block-encoded with operations
        such as :class:`~.PrepSelPrep` or :class:`~.Qubitization`. Note that both of these operations
        have a gate decomposition and can be implemented on hardware. The following example applies the polynomial
        :math:`p(x) = -x + 0.5x^3 + 0.5x^5` to the Hamiltonian :math:`H = 0.1X_3 - 0.7X_3Z_4 - 0.2Z_3Y_4`,
        block-encoded with :class:`~.PrepSelPrep`.

        .. code-block:: python

            poly = np.array([0, -1, 0, 0.5, 0, 0.5])
            H = 0.1 * qml.X(2) - 0.7 * qml.X(2) @ qml.Z(3) - 0.2 * qml.Z(2)

            control_wires = [0, 1]
            block_encode = qml.PrepSelPrep(H, control=control_wires)
            angles = qml.poly_to_angles(poly, "QSVT")

            projectors = [
                qml.PCPhase(angles[i], dim=2 ** len(H.wires), wires=control_wires + H.wires)
                for i in range(len(angles))
            ]

            dev = qml.device("default.qubit")

            @qml.qnode(dev)
            def circuit():
                qml.QSVT(block_encode, projectors)
                return qml.state()

        >>> circuit() # doctest: +SKIP
        array([ 1.44000000e-01+1.01511390e-01j,  0.00000000e+00+0.00000000e+00j,
                4.32000000e-01+3.04534169e-01j,  0.00000000e+00+0.00000000e+00j,
                -4.14503215e-17+7.27402636e-17j,  0.00000000e+00+0.00000000e+00j,
                5.59003542e-01+9.65699229e-02j,  0.00000000e+00+0.00000000e+00j,
                4.22566958e-01+7.30000000e-02j,  0.00000000e+00+0.00000000e+00j,
                -3.16925218e-01-5.47500000e-02j,  0.00000000e+00+0.00000000e+00j,
                5.20486781e-18-4.91300614e-17j,  0.00000000e+00+0.00000000e+00j,
                -2.79501771e-01-4.82849614e-02j,  0.00000000e+00+0.00000000e+00j])
    """

    grad_method = None
    """Gradient computation method."""

    def _flatten(self):
        data = (self.hyperparameters["UA"], self.hyperparameters["projectors"])
        return data, tuple()

    # pylint: disable=arguments-differ
    @classmethod
    def _primitive_bind_call(cls, UA, projectors, **kwargs):  # kwarg is id
        return cls._primitive.bind(UA, *projectors, **kwargs)

    @classmethod
    def _unflatten(cls, data, _) -> "QSVT":
        return cls(*data)

    def __init__(self, UA, projectors, id=None):
        if not isinstance(UA, Operator):
            raise ValueError("Input block encoding must be an Operator")

        self._hyperparameters = {
            "UA": UA,
            "projectors": projectors,
        }

        total_wires = Wires.all_wires([proj.wires for proj in projectors]) + Wires(UA.wires)

        super().__init__(wires=total_wires, id=id)

    def map_wires(self, wire_map: dict):
        # pylint: disable=protected-access
        new_op = copy.deepcopy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        new_op._hyperparameters["UA"] = new_op._hyperparameters["UA"].map_wires(wire_map)
        new_op._hyperparameters["projectors"] = [
            proj.map_wires(wire_map) for proj in new_op._hyperparameters["projectors"]
        ]
        return new_op

    @property
    def data(self):
        r"""Flattened list of operator data in this QSVT operation.

        This ensures that the backend of a ``QuantumScript`` which contains a
        ``QSVT`` operation can be inferred with respect to the types of the
        ``QSVT`` block encoding and projector-controlled phase shift data.
        """
        return tuple(datum for op in self._operators for datum in op.data)

    @data.setter
    def data(self, new_data):
        # We need to check if ``new_data`` is empty because ``Operator.__init__()``  will attempt to
        # assign the QSVT data to an empty tuple (since no positional arguments are provided).
        if new_data:
            for op in self._operators:
                if op.num_params > 0:
                    op.data = new_data[: op.num_params]
                    new_data = new_data[op.num_params :]

    def __copy__(self):
        # Override Operator.__copy__() to avoid setting the "data" property before the new instance
        # is assigned hyper-parameters since QSVT data is derived from the hyper-parameters.
        clone = QSVT.__new__(QSVT)

        # Ensure the operators in the hyper-parameters are copied instead of aliased.
        clone._hyperparameters = {
            "UA": copy.copy(self._hyperparameters["UA"]),
            "projectors": list(map(copy.copy, self._hyperparameters["projectors"])),
        }

        for attr, value in vars(self).items():
            if attr != "_hyperparameters":
                setattr(clone, attr, value)

        return clone

    @property
    def _operators(self) -> list[Operator]:
        """Flattened list of operators that compose this QSVT operation."""
        return [self._hyperparameters["UA"], *self._hyperparameters["projectors"]]

    @staticmethod
    def compute_decomposition(
        *_data, UA, projectors, **_kwargs
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        The :class:`~.QSVT` is decomposed into alternating block encoding
        and projector-controlled phase shift operators. This is defined by the following
        equations, where :math:`U` is the block encoding operation and both :math:`\Pi_\phi` and
        :math:`\tilde{\Pi}_\phi` are projector-controlled phase shifts with angle :math:`\phi`.

        When the number of projector-controlled phase shifts is even (:math:`d` is odd), the QSVT
        circuit is defined as:

        .. math::

            U_{QSVT} = \Pi_{\phi_1}U\left[\prod^{(d-1)/2}_{k=1}\Pi_{\phi_{2k}}U^\dagger
            \tilde{\Pi}_{\phi_{2k+1}}U\right]\Pi_{\phi_{d+1}}.


        And when the number of projector-controlled phase shifts is odd (:math:`d` is even):

        .. math::

            U_{QSVT} = \left[\prod^{d/2}_{k=1}\Pi_{\phi_{2k-1}}U^\dagger\tilde{\Pi}_{\phi_{2k}}U\right]
            \Pi_{\phi_{d+1}}.

        .. seealso:: :meth:`~.QSVT.decomposition`.

        Args:
            UA (Operator): the block encoding circuit, specified as a :class:`~.Operator`
            projectors (list[Operator]): a list of projector-controlled phase
                shift circuits that implement the desired polynomial

        Returns:
            list[.Operator]: decomposition of the operator
        """

        op_list = []
        UA_adj = copy.copy(UA)

        for idx, op in enumerate(projectors[:-1]):
            if QueuingManager.recording():
                apply(op)
            op_list.append(op)

            if idx % 2 == 0:
                if QueuingManager.recording():
                    apply(UA)
                op_list.append(UA)

            else:
                op_list.append(ops.adjoint(UA_adj))

        if QueuingManager.recording():
            apply(projectors[-1])
        op_list.append(projectors[-1])

        return op_list

    def label(self, decimals=None, base_label=None, cache=None):
        op_label = base_label or self.__class__.__name__
        return op_label

    def queue(self, context=QueuingManager):
        context.remove(self._hyperparameters["UA"])
        for op in self._hyperparameters["projectors"]:
            context.remove(op)
        context.append(self)
        return self

    @staticmethod
    def compute_matrix(*args, **kwargs):
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Operator.matrix` and :func:`~.matrix`

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            **hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            tensor_like: matrix representation
        """
        # pylint: disable=unused-argument
        op_list = []
        UA = kwargs["UA"]
        projectors = kwargs["projectors"]

        # incase this method is called in a queue context, this prevents queuing ops unnecessarily
        with QueuingManager.stop_recording():
            UA_copy = copy.copy(UA)

            for idx, op in enumerate(projectors[:-1]):
                op_list.append(op)
                if idx % 2 == 0:
                    op_list.append(UA)
                else:
                    op_list.append(ops.adjoint(UA_copy))

            op_list.append(projectors[-1])
            mat = ops.functions.matrix(ops.prod(*tuple(op_list[::-1])))

        return mat


# pylint: disable=protected-access
if QSVT._primitive is not None:

    @QSVT._primitive.def_impl
    def _(UA, *projectors, **kwargs):  # kwarg might be id
        return type.__call__(QSVT, UA, projectors, **kwargs)


def _complementary_poly(poly_coeffs):
    r"""
    Computes the complementary polynomial Q given a polynomial P.

    The polynomial Q is complementary to P if it satisfies the following equation:

    .. math:

        |P(e^{i\theta})|^2 + |Q(e^{i\theta})|^2 = 1, \quad \forall \quad \theta \in \left[0, 2\pi\right]

    The method is based on computing an auxiliary polynomial R, finding its roots,
    and reconstructing Q by using information extracted from the roots.
    For more details see `arXiv:2308.01501 <https://arxiv.org/abs/2308.01501>`_.

    Args:
        poly_coeffs (tensor-like): coefficients of the complex polynomial P

    Returns:
        tensor-like: coefficients of the complementary polynomial Q
    """
    poly_degree = len(poly_coeffs) - 1

    # Build the polynomial R(z) = z^degree * (1 - conj(P(1/z)) * P(z)), deduced from (eq.33) and (eq.34) of
    # `arXiv:2308.01501 <https://arxiv.org/abs/2308.01501>`_.
    # Note that conj(P(1/z)) * P(z) could be expressed as z^-degree * conj(P(z)[::-1]) * P(z)
    R = Polynomial.basis(poly_degree) - Polynomial(poly_coeffs) * Polynomial(
        np.conj(poly_coeffs[::-1])
    )
    r_roots = R.roots()

    inside_circle = [root for root in r_roots if np.abs(root) <= 1]
    outside_circle = [root for root in r_roots if np.abs(root) > 1]

    scale_factor = np.sqrt(np.abs(R.coef[-1] * np.prod(outside_circle)))
    Q_poly = scale_factor * Polynomial.fromroots(inside_circle)

    return Q_poly.coef


def _compute_qsp_angle(poly_coeffs):
    r"""
    Computes the Quantum Signal Processing (QSP) angles given the coefficients of a polynomial F.

    The method for computing the QSP angles is adapted from the approach described in [`arXiv:2406.04246 <https://arxiv.org/abs/2406.04246>`_] for Generalized-QSP.

    Args:
        poly_coeffs (tensor-like): coefficients of the input polynomial F

    Returns:
        (tensor-like): QSP angles corresponding to the input polynomial F

    .. details::
        :title: Implementation details

        Based on the appendix A in `arXiv:2406.04246 <https://arxiv.org/abs/2406.04246>`_, the target polynomial :math:`F`
        is transformed into a new polynomial :math:`P` by following the steps below:

        0. The input to the function are the coefficients of :math:`F`, e.g. :math:`[c_0, 0, c_1, 0, c_2]`.
        1. We express the polynomial in the Chebyshev basis by applying the Chebyshev transform. This generates
           a new representation :math:`[a_0, 0, a_1, 0, a_2]`.
        2. We generate :math:`P` by reordering the array, moving the zeros to the initial positions.
           :math:`P = [0, 0, a_0, a_1, a_2]`.

        The polynomial :math:`P` can now be used in Algorithm 1 of [`arXiv:2308.01501 <https://arxiv.org/abs/2308.01501>`_]
        in order to find the desired angles.

        The above algorithm is specific to Generalized-QSP so an adaptation has been made to return the required angles:

            - The :math:`R(\theta, \phi, \lambda)` gate, is simplified into a :math:`R_Y(\theta)` gate.

            - The calculation of :math:`\theta_d` is updated to :math:`\theta_d = \tan^{-1}(\frac{a_d}{b_d})`.
              In this way, the sign introduced by :math:`\phi_d` and :math:`\lambda_d` is absorbed
              in the :math:`\theta_d` value.
    """

    parity = (len(poly_coeffs) - 1) % 2

    P = np.concatenate(
        [np.zeros(len(poly_coeffs) // 2), chebyshev.poly2cheb(poly_coeffs)[parity::2]]
    ) * (1 - 1e-12)

    complementary = _complementary_poly(P)

    polynomial_matrix = np.array([P, complementary])
    num_terms = polynomial_matrix.shape[1]
    rotation_angles = np.zeros(num_terms)

    # Adaptation of Algorithm 1 of [arXiv:2308.01501]
    with QueuingManager.stop_recording():
        for idx in range(num_terms - 1, -1, -1):

            poly_a, poly_b = polynomial_matrix[:, idx]
            rotation_angles[idx] = np.arctan2(poly_b.real, poly_a.real)

            rotation_op = ops.RY.compute_matrix(-2 * rotation_angles[idx])

            updated_poly_matrix = rotation_op @ polynomial_matrix
            polynomial_matrix = np.array(
                [updated_poly_matrix[0][1 : idx + 1], updated_poly_matrix[1][0:idx]]
            )

    return rotation_angles


def _cheby_pol(x, degree):
    r"""Return the value of the Chebyshev polynomial cos(degree*arcos(x)) at point x

    Args:
        x (float): |x| \leq 1 is point at which to evaluate cos(degree * cos(\cdot))
        degree (int): degree of the Chebyshev polynomial

    Returns:
        float: value of cos(degree*arcos(x))
    """
    return math.cos(degree * math.arccos(x))


def _poly_func(coeffs, parity, x):
    r"""Evaluate a polynomial function of a given parity expressed in the Chebyshev basis at value x

    Args:
        coeffs (tensor_like): coefficient of the polynomial function in the Chebyshev basis
        parity (int): 0 or 1 for odd/even polynomials respectively
        x (float): point at which to evaluate the polynomial function

    Returns:
        float: \sum c_kT_{2k} if even else \sum c_kT_{2k+1} if odd where T_k(x)=cos(k \arccos(x))
    """

    ind = math.arange(len(coeffs))
    return sum(coeffs[i] * _cheby_pol(x, degree=2 * i + parity) for i in ind)


def _z_rotation(phi, interface):
    r"""Returns the matrix of the `RZ(2 \phi)` gate.

    Args:
        phi (float): angle parameter

    Returns:
        tensor_like: Z rotation matrix
    """

    return math.array([[math.exp(1j * phi), 0.0], [0.0, math.exp(-1j * phi)]], like=interface)


def _W_of_x(x, interface):
    r"""Returns the matrix of the operator W(x) defined in Theorem (1) of https://arxiv.org/pdf/2002.11649

    Args:
        x (float): point at which to evaluate the parametric operator W

    Returns:
        tensor_like: 2x2 matrix of W(x)
    """

    return math.array(
        [
            [_cheby_pol(x=x, degree=1.0), 1j * math.sqrt(1 - _cheby_pol(x=x, degree=1.0) ** 2)],
            [1j * math.sqrt(1 - _cheby_pol(x=x, degree=1.0) ** 2), _cheby_pol(x=x, degree=1.0)],
        ],
        like=interface,
    )


def _qsp_iterate(phi, x, interface):
    r"""
    Signal operator defined as the product of RZ(phi) and W(x)

    Args:
        phi (float): angle parameter
        x (float): point at which to evaluate the parametric operator W

    Returns:
        tensor_like: 2x2 matrix of operator defined in Theorem (1) of https://arxiv.org/pdf/2002.11649
    """

    a = math.dot(_W_of_x(x=x, interface=interface), _z_rotation(phi=phi, interface=interface))
    return a


def _qsp_iterate_broadcast(phis, x, interface):
    r"""Eq (13) Resulting unitary of the QSP circuit (on reduced invariant subspace ofc)

    Args:
        phis (tensor_like): array of QSP angles implementing a given polynomial
        x (float):point at which to evaluate the polynomial
    Returns:
        tensor_like: 2x2 block-encoding of polynomial implemented by the angles phi
    """

    # pylint: disable=import-outside-toplevel
    try:
        from jax import vmap

        interface = "jax"
        qsp_iterate_list = vmap(_qsp_iterate, in_axes=(0, None, None))(phis[1:], x, interface)
    except ModuleNotFoundError:
        qsp_iterate_list = math.vectorize(_qsp_iterate, excluded=(1, 2), signature="()->(m,n)")(
            phis[1:], x, interface
        )

    matrix_iterate = reduce(math.dot, qsp_iterate_list)
    matrix_iterate = math.dot(_z_rotation(phi=phis[0], interface=interface), matrix_iterate)

    return math.real(matrix_iterate[0, 0])


def _grid_pts(degree, interface):
    r"""Generate the grid: x_j = cos(\frac{(2j-1)\pi}{4\tilde{d}}) over which the polynomials
    are evaluated and the optimization is carried defined in page 8 (https://arxiv.org/pdf/2002.11649)

    Args:
        degree (int): degree of polynomial function

    Returns:
        tensor_like: optimization grid points
    """

    d = (degree + 1) // 2 + (degree + 1) % 2
    return math.array(
        [math.cos((2 * j - 1) * np.pi / (4 * d)) for j in range(1, d + 1)], like=interface
    )


def _qsp_optimization(degree, coeffs_target_func, interface=None):
    r"""
    Algorithm 1 in https://arxiv.org/pdf/2002.11649 produces the angle parameters by minimizing the distance between the target and qsp polynomial over the grid

    Args:
        degree (int): degree of polynomial function
        coeffs_target_func (tensor_like): coefficients of the polynomial function in ascending index order

    Returns:
        tuple[tensor_like, float]: A tuple containing QSP angles and the converged cost function value at QSP angles
    """
    parity = degree % 2

    # pylint: disable=import-outside-toplevel
    try:
        from jax import jacobian

        interface = "jax"

    except ModuleNotFoundError:
        from autograd import jacobian

    grid_points = _grid_pts(degree, interface=interface)

    initial_guess = [np.pi / 4] + [0.0] * (degree - 1) + [np.pi / 4]
    initial_guess = math.array(initial_guess, like=interface)

    targets = [_poly_func(coeffs=coeffs_target_func, x=x, parity=parity) for x in grid_points]
    targets = math.array(targets, like=interface)

    def obj_function(phi):
        # Equation (23) in https://arxiv.org/pdf/2002.11649

        # pylint: disable=import-outside-toplevel
        try:
            from jax import jit, vmap

            qsp_iterates = jit(_qsp_iterate_broadcast, static_argnames=["interface"])

            obj_func = (
                vmap(qsp_iterates, in_axes=(None, 0, None))(phi, grid_points, interface) - targets
            )
        except ModuleNotFoundError:
            obj_func = (
                math.vectorize(_qsp_iterate_broadcast, excluded=(0, 2))(phi, grid_points, interface)
                - targets
            )

        obj_func = math.dot(obj_func, obj_func)

        return 1 / len(grid_points) * obj_func

    try:
        from jax import jit

        obj_function = jit(obj_function)
    except ModuleNotFoundError:
        pass

    results = scipy.optimize.minimize(
        fun=obj_function,
        x0=initial_guess,
        method="L-BFGS-B",  # More efficient than Newton method
        jac=jacobian(obj_function),
        tol=1e-15,
    )
    phis = results.x
    cost_func = results.fun

    return phis, cost_func


def _compute_qsp_angles_iteratively(poly):
    """Calculates the angles given a polynomial in canonical base

    Args:
        poly (tensor_like): coefficients of the polynomial ordered from lowest to highest power
    """
    poly_cheb = chebyshev.poly2cheb(poly)
    degree = len(poly_cheb) - 1

    coeffs_odd = poly_cheb[1::2]
    coeffs_even = poly_cheb[0::2]

    if np.allclose(coeffs_odd, np.zeros_like(coeffs_odd)):
        coeffs_target_func = math.array(coeffs_even)
    else:
        coeffs_target_func = math.array(coeffs_odd)

    angles, *_ = _qsp_optimization(degree=degree, coeffs_target_func=coeffs_target_func)

    return angles


def _gqsp_u3_gate(theta, phi, lambd):
    r"""
    Computes the U3 gate matrix for Generalized Quantum Signal Processing (GQSP) as described
    in [`arXiv:2406.04246 <https://arxiv.org/abs/2406.04246>`_]
    """

    exp_phi = math.exp(1j * phi)
    exp_lambda = math.exp(1j * lambd)
    exp_lambda_phi = math.exp(1j * (lambd + phi))

    matrix = np.array(
        [
            [exp_lambda_phi * math.cos(theta), exp_phi * math.sin(theta)],
            [exp_lambda * math.sin(theta), -math.cos(theta)],
        ],
        dtype=complex,
    )

    return matrix


def _compute_gqsp_angles(poly_coeffs):
    r"""
    Computes the Generalized Quantum Signal Processing (GQSP) angles given the coefficients of a polynomial P.

    The method for computing the GQSP angles is based on the algorithm described in [`arXiv:2406.04246 <https://arxiv.org/abs/2406.04246>`_].
    The complementary polynomial is calculated using root-finding methods.

    Args:
        poly_coeffs (tensor-like): Coefficients of the input polynomial P.

    Returns:
        (tensor-like): QSP angles corresponding to the input polynomial P. The shape is (3, P-degree)
    """

    complementary = _complementary_poly(poly_coeffs)

    # Algorithm 1 in [arXiv:2308.01501]
    input_data = math.array([poly_coeffs, complementary])
    num_elements = input_data.shape[1]

    angles_theta, angles_phi, angles_lambda = math.zeros([3, num_elements])

    for idx in range(num_elements - 1, -1, -1):

        component_a, component_b = input_data[:, idx]
        angles_theta[idx] = math.arctan2(np.abs(component_b), math.abs(component_a))
        angles_phi[idx] = (
            0
            if math.isclose(math.abs(component_b), 0, atol=1e-10)
            else math.angle(component_a * math.conj(component_b))
        )

        if idx == 0:
            angles_lambda[0] = math.angle(component_b)
        else:
            updated_matrix = (
                _gqsp_u3_gate(angles_theta[idx], angles_phi[idx], 0).conj().T @ input_data
            )
            input_data = math.array([updated_matrix[0][1 : idx + 1], updated_matrix[1][0:idx]])

    return angles_theta, angles_phi, angles_lambda


def transform_angles(angles, routine1, routine2):
    r"""
    Converts angles for quantum signal processing (QSP) and quantum singular value transformation (QSVT) routines.

    The transformation is based on Appendix A.2 of `arXiv:2105.02859 <https://arxiv.org/abs/2105.02859>`_. Note that QSVT is equivalent to taking the reflection convention of QSP.

    Args:
        angles (tensor-like): angles to be transformed
        routine1 (str): the current routine for which the angles are obtained, must be either ``"QSP"`` or ``"QSVT"``
        routine2 (str): the target routine for which the angles should be transformed,
            must be either ``"QSP"`` or ``"QSVT"``

    Returns:
        tensor-like: the transformed angles as an array

    **Example**

    .. code-block::

        >>> qsp_angles = np.array([0.2, 0.3, 0.5])
        >>> qsvt_angles = qml.transform_angles(qsp_angles, "QSP", "QSVT")
        >>> print(qsvt_angles)
        [-6.868...  1.870... -0.285...]


    .. details::
        :title: Usage Details

        This example applies the polynomial :math:`P(x) = x - \frac{x^3}{2} + \frac{x^5}{3}` to a block-encoding
        of :math:`x = 0.2`.

        .. code-block::

            poly = np.array([0, 1.0, 0, -1/2, 0, 1/3])

            qsp_angles = qml.poly_to_angles(poly, "QSP")
            qsvt_angles = qml.transform_angles(qsp_angles, "QSP", "QSVT")

            x = 0.2

            # Encodes x in the top left of the matrix
            block_encoding = qml.RX(2 * np.arccos(x), wires=0)

            projectors = [qml.PCPhase(angle, dim=1, wires=0) for angle in qsvt_angles]

            @qml.qnode(qml.device("default.qubit"))
            def circuit_qsvt():
                qml.QSVT(block_encoding, projectors)
                return qml.state()

            output = qml.matrix(circuit_qsvt, wire_order=[0])()[0, 0]
            expected = sum(coef * (x**i) for i, coef in enumerate(poly))

            print("output qsvt: ", output.real)
            print("P(x) =       ", expected)

        .. code-block:: pycon

            output qsvt:  0.19610666666647059
            P(x) =        0.19610666666666668
    """

    if routine1 == "QSP" and routine2 == "QSVT":
        num_angles = len(angles)
        update_vals = np.empty(num_angles)

        update_vals[0] = 3 * np.pi / 4 - (3 + num_angles % 4) * np.pi / 2
        update_vals[1:-1] = np.pi / 2
        update_vals[-1] = -np.pi / 4
        update_vals = math.convert_like(update_vals, angles)

        return angles + update_vals

    if routine1 == "QSVT" and routine2 == "QSP":
        num_angles = len(angles)
        update_vals = np.empty(num_angles)

        update_vals[0] = 3 * np.pi / 4 - (3 + num_angles % 4) * np.pi / 2
        update_vals[1:-1] = np.pi / 2
        update_vals[-1] = -np.pi / 4
        update_vals = math.convert_like(update_vals, angles)

        return angles - update_vals

    raise AssertionError(
        f"Invalid conversion. The conversion between {routine1} --> {routine2} is not defined."
    )


def poly_to_angles(poly, routine, angle_solver: Literal["root-finding"] = "root-finding"):
    r"""
    Computes the angles needed to implement a polynomial transformation with quantum signal processing (QSP),
    quantum singular value transformation (QSVT) or generalized quantum signal processing (GQSP).

    The polynomial :math:`P(x) = \sum_n a_n x^n` must satisfy :math:`|P(x)| \leq 1` for :math:`x \in [-1, 1]`.
    In QSP and QSVT, the coefficients :math:`a_n` must be real and the exponents :math:`n` must be either all even or all odd.
    For more details see `arXiv:2105.02859 <https://arxiv.org/abs/2105.02859>`_.

    Args:
        poly (tensor_like): coefficients of the polynomial ordered from lowest to highest power

        routine (str): the routine for which the angles are computed. Must be either ``"QSP"``, ``"QSVT"`` or ``"GQSP"``.

        angle_solver (str): Specifies the method used to calculate the angles. Options include:

            - ``"root-finding"``: effective for polynomials of degree up to :math:`\sim 1000`
            - ``"iterative"``: effective for polynomials of degree higher than :math:`\sim 1000` for
              the ``"QSP"`` and ``"QSVT"`` routines.

    Returns:
        (tensor-like): computed angles for the specified routine

    Raises:
        AssertionError: if ``poly`` is not valid
        AssertionError: if ``routine`` or ``angle_solver`` is not supported

    **Example**

    This example generates the ``QSVT`` angles for the polynomial :math:`P(x) = x - \frac{x^3}{2} + \frac{x^5}{3}`.

    .. code-block::

        >>> poly = np.array([0, 1.0, 0, -1/2, 0, 1/3])
        >>> qsvt_angles = qml.poly_to_angles(poly, "QSVT")
        >>> print(qsvt_angles)
        [-5.497...  1.570...  1.570...  0.583...   1.61...  0.747...]


    .. details::
        :title: Usage Details

        This example applies the polynomial :math:`P(x) = x - \frac{x^3}{2} + \frac{x^5}{3}` to a block-encoding
        of :math:`x = 0.2`.

        .. code-block::

            poly = np.array([0, 1.0, 0, -1/2, 0, 1/3])

            qsvt_angles = qml.poly_to_angles(poly, "QSVT")

            x = 0.2

            # Encode x in the top left of the matrix
            block_encoding = qml.RX(2 * np.arccos(x), wires=0)
            projectors = [qml.PCPhase(angle, dim=1, wires=0) for angle in qsvt_angles]

            @qml.qnode(qml.device("default.qubit"))
            def circuit_qsvt():
                qml.QSVT(block_encoding, projectors)
                return qml.state()

            output = qml.matrix(circuit_qsvt, wire_order=[0])()[0, 0]
            expected = sum(coef * (x**i) for i, coef in enumerate(poly))

            print("output qsvt: ", output.real)
            print("P(x) =       ", expected)

        .. code-block:: pycon

            output qsvt:  0.19610666666647059
            P(x) =        0.19610666666666668

    """

    poly = math.trim_zeros(math.array(poly, like="numpy"), trim="b")

    if len(poly) == 1:
        raise AssertionError("The polynomial must have at least degree 1.")

    for x in [-1, 0, 1]:
        if math.abs(math.sum(coeff * x**i for i, coeff in enumerate(poly))) > 1:
            # Check that |P(x)| ≤ 1. Only points -1, 0, 1 will be checked.
            raise AssertionError("The polynomial must satisfy that |P(x)| ≤ 1 for all x in [-1, 1]")

    if routine in ["QSVT", "QSP"]:
        if not (
            np.isclose(math.sum(math.abs(poly[::2])), 0.0)
            or np.isclose(math.sum(math.abs(poly[1::2])), 0.0)
        ):
            raise AssertionError(
                "The polynomial has no definite parity. All odd or even entries in the array must take a value of zero."
            )
        assert np.allclose(
            np.array(poly, dtype=np.complex128).imag, 0
        ), "Array must not have an imaginary part"

    if routine == "QSVT":
        if angle_solver == "root-finding":
            return transform_angles(_compute_qsp_angle(poly), "QSP", "QSVT")
        if angle_solver == "iterative":
            return transform_angles(_compute_qsp_angles_iteratively(poly), "QSP", "QSVT")

        raise AssertionError(
            "Invalid angle solver method. We currently support 'root-finding' and 'iterative'"
        )

    if routine == "QSP":
        if angle_solver == "root-finding":
            return _compute_qsp_angle(poly)
        if angle_solver == "iterative":
            return _compute_qsp_angles_iteratively(poly)
        raise AssertionError(
            "Invalid angle solver method. Valid value is 'root-finding' and 'iterative'"
        )

    if routine == "GQSP":
        return _compute_gqsp_angles(poly)
    raise AssertionError("Invalid routine. Valid values are 'GQSP', 'QSP' and 'QSVT'")
