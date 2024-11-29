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
"""
Contains the QSVT template and qsvt wrapper function.
"""
# pylint: disable=too-many-arguments
import copy

import numpy as np
from numpy.polynomial import Polynomial, chebyshev

import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops import BlockEncode, PCPhase
from pennylane.ops.op_math import adjoint
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires


def qsvt(A, angles, wires, convention=None):
    r"""Implements the
    `quantum singular value transformation <https://arxiv.org/abs/1806.01838>`__ (QSVT) circuit.

    .. note ::

        :class:`~.BlockEncode` and :class:`~.PCPhase` used in this implementation of QSVT
        are matrix-based operators and well-suited for simulators.
        To implement QSVT with user-defined circuits for the block encoding and
        projector-controlled phase shifts, use the :class:`~.QSVT` template.

    Given a matrix :math:`A`, and a list of angles :math:`\vec{\phi}`, this function applies a
    circuit for the quantum singular value transformation using :class:`~.BlockEncode` and
    :class:`~.PCPhase`.

    When the number of angles is even (:math:`d` is odd), the QSVT circuit is defined as:

    .. math::

        U_{QSVT} = \tilde{\Pi}_{\phi_1}U\left[\prod^{(d-1)/2}_{k=1}\Pi_{\phi_{2k}}U^\dagger
        \tilde{\Pi}_{\phi_{2k+1}}U\right]\Pi_{\phi_{d+1}},


    and when the number of angles is odd (:math:`d` is even):

    .. math::

        U_{QSVT} = \left[\prod^{d/2}_{k=1}\Pi_{\phi_{2k-1}}U^\dagger\tilde{\Pi}_{\phi_{2k}}U\right]
        \Pi_{\phi_{d+1}}.

    Here, :math:`U` denotes a block encoding of :math:`A` via :class:`~.BlockEncode` and
    :math:`\Pi_\phi` denotes a projector-controlled phase shift with angle :math:`\phi`
    via :class:`~.PCPhase`.

    This circuit applies a polynomial transformation (:math:`Poly^{SV}`) to the singular values of
    the block encoded matrix:

    .. math::

        \begin{align}
             U_{QSVT}(A, \phi) &=
             \begin{bmatrix}
                Poly^{SV}(A) & \cdot \\
                \cdot & \cdot
            \end{bmatrix}.
        \end{align}

    The polynomial transformation is determined by a combination of the block encoding and choice of angles,
    :math:`\vec{\phi}`. The convention used by :class:`~.BlockEncode` is commonly refered to as the
    reflection convention or :math:`R` convention. Another equivalent convention for the block encoding is
    the :math:`Wx` or rotation convention.

    Depending on the choice of convention for blockencoding, the same phase angles will produce different
    polynomial transformations. We provide the functionality to swap between blockencoding conventions and
    to transform the phase angles accordingly using the :code:`convention` keyword argument.

    .. seealso::

        :class:`~.QSVT` and `A Grand Unification of Quantum Algorithms <https://arxiv.org/pdf/2105.02859.pdf>`_.

    Args:
        A (tensor_like): the general :math:`(n \times m)` matrix to be encoded
        angles (tensor_like): a list of angles by which to shift to obtain the desired polynomial
        wires (Iterable[int, str], Wires): the wires the template acts on
        convention (string): can be set to ``"Wx"`` to convert quantum signal processing angles in the
            `Wx` convention to QSVT angles.

    **Example**

    To implement QSVT in a circuit, we can use the following method:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> A = np.array([[0.1, 0.2], [0.3, 0.4]])
    >>> angles = np.array([0.1, 0.2, 0.3])
    >>> @qml.qnode(dev)
    ... def example_circuit(A):
    ...     qml.qsvt(A, angles, wires=[0, 1])
    ...     return qml.expval(qml.Z(0))

    The resulting circuit implements QSVT.

    >>> print(qml.draw(example_circuit)(A))
    0: ─╭QSVT─┤  <Z>
    1: ─╰QSVT─┤

    To see the implementation details, we can expand the circuit:

    >>> q_script = qml.tape.QuantumScript(ops=[qml.qsvt(A, angles, wires=[0, 1])])
    >>> print(q_script.expand().draw(decimals=2))
    0: ─╭∏_ϕ(0.30)─╭BlockEncode(M0)─╭∏_ϕ(0.20)─╭BlockEncode(M0)†─╭∏_ϕ(0.10)─┤
    1: ─╰∏_ϕ(0.30)─╰BlockEncode(M0)─╰∏_ϕ(0.20)─╰BlockEncode(M0)†─╰∏_ϕ(0.10)─┤
    """
    if qml.math.shape(A) == () or qml.math.shape(A) == (1,):
        A = qml.math.reshape(A, [1, 1])

    c, r = qml.math.shape(A)
    global_phase, global_phase_op = None, None

    with qml.QueuingManager.stop_recording():
        UA = BlockEncode(A, wires=wires)
    projectors = []

    if convention == "Wx":
        angles = _qsp_to_qsvt(angles)
        global_phase = (len(angles) - 1) % 4

        if global_phase:
            with qml.QueuingManager.stop_recording():
                global_phase_op = qml.GlobalPhase(-0.5 * np.pi * (4 - global_phase), wires=wires)

    for idx, phi in enumerate(angles):
        dim = c if idx % 2 else r
        with qml.QueuingManager.stop_recording():
            projectors.append(PCPhase(phi, dim=dim, wires=wires))

    projectors = projectors[::-1]  # reverse order to match equation

    if convention == "Wx" and global_phase:
        return qml.prod(global_phase_op, QSVT(UA, projectors))
    return QSVT(UA, projectors)


class QSVT(Operation):
    r"""QSVT(UA,projectors)
    Implements the
    `quantum singular value transformation <https://arxiv.org/abs/1806.01838>`__ (QSVT) circuit.

    .. note ::

        This template allows users to define hardware-compatible block encoding and
        projector-controlled phase shift circuits. For a QSVT implementation that is
        tailored for simulators see :func:`~.qsvt` .

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

    >>> dev = qml.device("default.qubit", wires=2)
    >>> A = np.array([[0.1]])
    >>> block_encode = qml.BlockEncode(A, wires=[0, 1])
    >>> shifts = [qml.PCPhase(i + 0.1, dim=1, wires=[0, 1]) for i in range(3)]
    >>> @qml.qnode(dev)
    >>> def example_circuit():
    ...    qml.QSVT(block_encode, shifts)
    ...    return qml.expval(qml.Z(0))

    The resulting circuit implements QSVT.

    >>> print(qml.draw(example_circuit)())
    0: ─╭QSVT─┤  <Z>
    1: ─╰QSVT─┤

    To see the implementation details, we can expand the circuit:

    >>> q_script = qml.tape.QuantumScript(ops=[qml.QSVT(block_encode, shifts)])
    >>> print(q_script.expand().draw(decimals=2))
    0: ─╭∏_ϕ(0.10)─╭BlockEncode(M0)─╭∏_ϕ(1.10)─╭BlockEncode(M0)†─╭∏_ϕ(2.10)─┤
    1: ─╰∏_ϕ(0.10)─╰BlockEncode(M0)─╰∏_ϕ(1.10)─╰BlockEncode(M0)†─╰∏_ϕ(2.10)─┤

    When working with this class directly, we can make use of any PennyLane operation
    to represent our block-encoding and our phase-shifts.

    >>> dev = qml.device("default.qubit", wires=[0])
    >>> block_encoding = qml.Hadamard(wires=0)  # note H is a block encoding of 1/sqrt(2)
    >>> phase_shifts = [qml.RZ(-2 * theta, wires=0) for theta in (1.23, -0.5, 4)]  # -2*theta to match convention
    >>>
    >>> @qml.qnode(dev)
    >>> def example_circuit():
    ...     qml.QSVT(block_encoding, phase_shifts)
    ...     return qml.expval(qml.Z(0))
    >>>
    >>> example_circuit()
    tensor(0.54030231, requires_grad=True)

    Once again, we can visualize the circuit as follows:

    >>> print(qml.draw(example_circuit)())
    0: ──QSVT─┤  <Z>

    To see the implementation details, we can expand the circuit:

    >>> q_script = qml.tape.QuantumScript(ops=[qml.QSVT(block_encoding, phase_shifts)])
    >>> print(q_script.expand().draw(decimals=2))
    0: ──RZ(-2.46)──H──RZ(1.00)──H†──RZ(-8.00)─┤
    """

    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    grad_method = None
    """Gradient computation method."""

    def _flatten(self):
        data = (self.hyperparameters["UA"], self.hyperparameters["projectors"])
        return data, tuple()

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @classmethod
    def _unflatten(cls, data, _) -> "QSVT":
        return cls(*data)

    def __init__(self, UA, projectors, id=None):
        if not isinstance(UA, qml.operation.Operator):
            raise ValueError("Input block encoding must be an Operator")

        self._hyperparameters = {
            "UA": UA,
            "projectors": projectors,
        }

        total_wires = qml.wires.Wires.all_wires(
            [proj.wires for proj in projectors]
        ) + qml.wires.Wires(UA.wires)

        super().__init__(wires=total_wires, id=id)

    def map_wires(self, wire_map: dict):
        # pylint: disable=protected-access
        new_op = copy.deepcopy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        new_op._hyperparameters["UA"] = qml.map_wires(new_op._hyperparameters["UA"], wire_map)
        new_op._hyperparameters["projectors"] = [
            qml.map_wires(proj, wire_map) for proj in new_op._hyperparameters["projectors"]
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
    def _operators(self) -> list[qml.operation.Operator]:
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
            if qml.QueuingManager.recording():
                qml.apply(op)
            op_list.append(op)

            if idx % 2 == 0:
                if qml.QueuingManager.recording():
                    qml.apply(UA)
                op_list.append(UA)

            else:
                op_list.append(adjoint(UA_adj))

        if qml.QueuingManager.recording():
            qml.apply(projectors[-1])
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

        with (
            QueuingManager.stop_recording()
        ):  # incase this method is called in a queue context, this prevents
            UA_copy = copy.copy(UA)  # us from queuing operators unnecessarily

            for idx, op in enumerate(projectors[:-1]):
                op_list.append(op)
                if idx % 2 == 0:
                    op_list.append(UA)
                else:
                    op_list.append(adjoint(UA_copy))

            op_list.append(projectors[-1])
            mat = qml.matrix(qml.prod(*tuple(op_list[::-1])))

        return mat


def _qsp_to_qsvt(angles):
    r"""Converts qsp angles to qsvt angles."""
    num_angles = len(angles)
    update_vals = np.empty(num_angles)

    update_vals[0] = 3 * np.pi / 4
    update_vals[1:-1] = np.pi / 2
    update_vals[-1] = -np.pi / 4
    update_vals = qml.math.convert_like(update_vals, angles)

    return angles + update_vals


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
    with qml.QueuingManager.stop_recording():
        for idx in range(num_terms - 1, -1, -1):

            poly_a, poly_b = polynomial_matrix[:, idx]
            rotation_angles[idx] = np.arctan2(poly_b.real, poly_a.real)

            rotation_op = qml.matrix(qml.RY(-2 * rotation_angles[idx], wires=0))

            updated_poly_matrix = rotation_op @ polynomial_matrix
            polynomial_matrix = np.array(
                [updated_poly_matrix[0][1 : idx + 1], updated_poly_matrix[1][0:idx]]
            )

    return rotation_angles


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
        [-6.86858347  1.87079633 -0.28539816]


    .. details::
        :title: Usage Details

        This example applies the polynomial :math:`P(x) = x - \frac{x^3}{2} + \frac{x^5}{3}` to a block-encoding
        of :math:`x = 0.2`.

        .. code-block::

            poly = [0, 1.0, 0, -1/2, 0, 1/3]

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
        update_vals = qml.math.convert_like(update_vals, angles)

        return angles + update_vals

    if routine1 == "QSVT" and routine2 == "QSP":
        num_angles = len(angles)
        update_vals = np.empty(num_angles)

        update_vals[0] = 3 * np.pi / 4 - (3 + num_angles % 4) * np.pi / 2
        update_vals[1:-1] = np.pi / 2
        update_vals[-1] = -np.pi / 4
        update_vals = qml.math.convert_like(update_vals, angles)

        return angles - update_vals

    raise AssertionError(
        f"Invalid conversion. The conversion between {routine1} --> {routine2} is not defined."
    )


def poly_to_angles(poly, routine, angle_solver="root-finding"):
    r"""
    Computes the angles needed to implement a polynomial transformation with quantum signal processing (QSP)
    or quantum singular value transformation (QSVT).

    The polynomial :math:`P(x) = \sum_n a_n x^n` must satisfy :math:`|P(x)| \leq 1` for :math:`x \in [-1, 1]`, the
    coefficients :math:`a_n` must be real and the exponents :math:`n` must be either all even or all odd.
    For more details see `arXiv:2105.02859 <https://arxiv.org/abs/2105.02859>`_.

    Args:
        poly (tensor-like): coefficients of the polynomial ordered from lowest to highest power

        routine (str): the routine for which the angles are computed. Must be either ``"QSP"`` or ``"QSVT"``.

        angle_solver (str): the method used to calculate the angles. Default is ``"root-finding"``.

    Returns:
        (tensor-like): computed angles for the specified routine

    Raises:
        AssertionError: if ``poly`` is not valid
        AssertionError: if ``routine`` or ``angle_solver`` is not supported

    **Example**

    This example generates the ``QSVT`` angles for the polynomial :math:`P(x) = x - \frac{x^3}{2} + \frac{x^5}{3}`.

    .. code-block::

        >>> poly = [0, 1.0, 0, -1/2, 0, 1/3]
        >>> qsvt_angles = qml.poly_to_angles(poly, "QSVT")
        >>> print(qsvt_angles)
        [-5.49778714  1.57079633  1.57079633  0.5833829   1.61095884  0.74753829]


    .. details::
        :title: Usage Details

        This example applies the polynomial :math:`P(x) = x - \frac{x^3}{2} + \frac{x^5}{3}` to a block-encoding
        of :math:`x = 0.2`.

        .. code-block::

            poly = [0, 1.0, 0, -1/2, 0, 1/3]

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

    # Trailing zeros are removed from the array
    for _ in range(len(poly)):
        if not np.isclose(poly[-1], 0.0):
            break
        poly.pop()

    if len(poly) == 1:
        raise AssertionError("The polynomial must have at least degree 1.")

    for x in [-1, 0, 1]:
        if qml.math.abs(qml.math.sum(coeff * x**i for i, coeff in enumerate(poly))) > 1:
            # Check that |P(x)| ≤ 1. Only points -1, 0, 1 will be checked.
            raise AssertionError("The polynomial must satisfy that |P(x)| ≤ 1 for all x in [-1, 1]")

    if routine in ["QSVT", "QSP"]:
        if not (
            np.isclose(qml.math.sum(qml.math.abs(poly[::2])), 0.0)
            or np.isclose(qml.math.sum(qml.math.abs(poly[1::2])), 0.0)
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
        raise AssertionError("Invalid angle solver method. We currently support 'root-finding'")

    if routine == "QSP":
        if angle_solver == "root-finding":
            return _compute_qsp_angle(poly)
        raise AssertionError("Invalid angle solver method. Valid value is 'root-finding'")
    raise AssertionError("Invalid routine. Valid values are 'QSP' and 'QSVT'")
