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
Contains templates for QSVT based subroutines.
"""
import copy
from collections import defaultdict
from functools import wraps
from typing import Dict

import pennylane as qml
import numpy as np

from numpy.polynomial import Polynomial, chebyshev

from pennylane.labs import resource_estimation as re
from pennylane.labs.resource_estimation import (
    CompressedResourceOp,
    ResourceExp,
    ResourceOperator,
    ResourcesNotDefined,
)
from pennylane.queuing import QueuingManager
from pennylane.templates import QSVT
from pennylane.wires import Wires
from pennylane.operation import Operation, AnyWires


# pylint: disable=arguments-differ


class ResourcePCPhase(qml.PCPhase, re.ResourceOperator):
    r"""Resource class for the Projector-Controlled Phaseshift gate.

    Resources:
        The resources are derived from the pennylane decomposition.

    .. seealso:: :class:`~.PCPhase`
    
    """

    @staticmethod
    def _resource_decomp(dim, num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""
        Resources:
            The resources are derived from the pennylane decomposition.
        """
        res = {}

        binary_dim = format(dim, f"0{num_wires}b")
        max_ctrls = 0 
        for index, val in enumerate(binary_dim):
            if val == "1":
                max_ctrls = index

        for num_ctrls in range(max_ctrls + 1):
            if num_ctrls == 0:
                res[re.ResourceRZ.resource_rep()] = 1
            
            res[re.ResourceControlled.resource_rep(re.ResourceRZ, {}, num_ctrls, 0, 1)] = 1

        return res

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Resource parameters:
            dim: ...
            num_wires: ...

        Returns:
            dict: dictionary containing the resource parameters
        """
        dim = self.hyperparameters["dimension"][0]
        num_wires = len(self.wires)
        return {"dim": dim, "num_wires": num_wires}
    
    @classmethod
    def resource_rep(cls, dim, num_wires) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {"dim": dim, "num_wires":num_wires})
    

class ResourceQSVT(Operation, re.ResourceOperator):
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
    >>>
    >>> @qml.qnode(dev)
    >>> def example_circuit():
    ...     qml.QSVT(block_encoding, phase_shifts)
    ...     return qml.expval(qml.Z(0))
    >>>
    >>> example_circuit()
    0.5403023058681395

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

        .. code-block::

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

        .. code-block:: pycon

            >>> circuit()
            array([-0.194205  +0.66654551j, -0.097905  +0.35831418j,
                    0.3319832 -0.51047262j, -0.09551437+0.01043668j])

        If we want to transform the singular values of a linear
        combination of unitaries, e.g., a Hamiltonian, it can be block-encoded with operations
        such as :class:`~.PrepSelPrep` or :class:`~.Qubitization`. Note that both of these operations
        have a gate decomposition and can be implemented on hardware. The following example applies the polynomial
        :math:`p(x) = -x + 0.5x^3 + 0.5x^5` to the Hamiltonian :math:`H = 0.1X_3 - 0.7X_3Z_4 - 0.2Z_3Y_4`,
        block-encoded with :class:`~.PrepSelPrep`.

        .. code-block::

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

        .. code-block:: pycon

            >>> circuit()
            array([ 1.44000000e-01+1.01511390e-01j,  0.00000000e+00+0.00000000e+00j,
                    4.32000000e-01+3.04534169e-01j,  0.00000000e+00+0.00000000e+00j,
                    1.92998954e-17+5.00377363e-17j,  0.00000000e+00+0.00000000e+00j,
                    5.59003542e-01+9.65699229e-02j,  0.00000000e+00+0.00000000e+00j,
                    4.22566958e-01+7.30000000e-02j,  0.00000000e+00+0.00000000e+00j,
                   -3.16925218e-01-5.47500000e-02j,  0.00000000e+00+0.00000000e+00j,
                   -2.98448441e-17-3.10878188e-17j,  0.00000000e+00+0.00000000e+00j,
                   -2.79501771e-01-4.82849614e-02j,  0.00000000e+00+0.00000000e+00j])
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

    def __init__(self, UA, projectors, num_projectors=None, id=None):
        if not isinstance(UA, qml.operation.Operator):
            raise ValueError("Input block encoding must be an Operator")

        try:
            _ = len(projectors)
        except TypeError:
            projectors = [projectors]

        if num_projectors is None:
            num_projectors = len(projectors)

        self._hyperparameters = {
            "UA": UA,
            "projectors": projectors,
            "num_projectors": num_projectors,
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
                op_list.append(qml.adjoint(UA_adj))

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
                    op_list.append(qml.adjoint(UA_copy))

            op_list.append(projectors[-1])
            mat = qml.matrix(qml.prod(*tuple(op_list[::-1])))

        return mat

    @staticmethod
    def _resource_decomp(block_encode, proj, num_proj, **kwargs) -> Dict[CompressedResourceOp, int]:
        res = {} 
        u_dag_count = num_proj // 2
        u_count = u_dag_count + (num_proj % 2)

        adjoint_block_encode = re.ResourceAdjoint.resource_rep(block_encode.op_type, block_encode.params)

        res[proj] = num_proj
        res[block_encode] = u_count
        res[adjoint_block_encode] = u_dag_count
        return res

    @property
    def resource_params(self):

        block_encoding = self.hyperparameters["UA"]
        proj = self.hyperparameters["projectors"][0]
        num_proj = self.hyperparameters["num_projectors"]
    
        block_encoding_cmpr_rep = block_encoding.resource_rep_from_op()
        proj_cmpr_rep = proj.resource_rep_from_op()
        return {"block_encode": block_encoding_cmpr_rep, "proj":proj_cmpr_rep, "num_proj":num_proj}
    
    @classmethod
    def resource_rep(cls, block_encode, proj, num_proj):
        return re.CompressedResourceOp(cls, {"block_encode": block_encode, "proj":proj, "num_proj":num_proj})


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


def _gqsp_u3_gate(theta, phi, lambd):
    r"""
    Computes the U3 gate matrix for Generalized Quantum Signal Processing (GQSP) as described
    in [`arXiv:2406.04246 <https://arxiv.org/abs/2406.04246>`_]
    """

    exp_phi = qml.math.exp(1j * phi)
    exp_lambda = qml.math.exp(1j * lambd)
    exp_lambda_phi = qml.math.exp(1j * (lambd + phi))

    matrix = np.array(
        [
            [exp_lambda_phi * qml.math.cos(theta), exp_phi * qml.math.sin(theta)],
            [exp_lambda * qml.math.sin(theta), -qml.math.cos(theta)],
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
    input_data = qml.math.array([poly_coeffs, complementary])
    num_elements = input_data.shape[1]

    angles_theta, angles_phi, angles_lambda = qml.math.zeros([3, num_elements])

    for idx in range(num_elements - 1, -1, -1):

        component_a, component_b = input_data[:, idx]
        angles_theta[idx] = qml.math.arctan2(np.abs(component_b), qml.math.abs(component_a))
        angles_phi[idx] = (
            0
            if qml.math.isclose(qml.math.abs(component_b), 0, atol=1e-10)
            else qml.math.angle(component_a * qml.math.conj(component_b))
        )

        if idx == 0:
            angles_lambda[0] = qml.math.angle(component_b)
        else:
            updated_matrix = (
                _gqsp_u3_gate(angles_theta[idx], angles_phi[idx], 0).conj().T @ input_data
            )
            input_data = qml.math.array([updated_matrix[0][1 : idx + 1], updated_matrix[1][0:idx]])

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
        [-6.86858347  1.87079633 -0.28539816]


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
