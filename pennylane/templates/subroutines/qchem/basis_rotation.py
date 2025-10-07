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
This module contains the template for performing basis transformation defined by a set of fermionic ladder operators.
"""
import numpy as np

from pennylane import math
from pennylane.decomposition import add_decomps, register_resources
from pennylane.operation import Operation
from pennylane.ops import PhaseShift, SingleExcitation, cond
from pennylane.wires import WiresLike


def _adjust_determinant(matrix):
    """Given an orthogonal (real-valued unitary) matrix, adjust its determinant to be 1
    and queue a phase shift that is equivalent to this adjustment in the context of BasisRotation.

    Args:
        matrix (array): orthogonal matrix to adjust the determinant of.

    Returns:
        tuple[float or None, array]: The angle to be passed into a PhaseShift gate on the first
        wire to perform the determinant adjustment on the quantum circuit level, which is ``None``
        if no adjustment is needed, as well as the new matrix with adjusted determinant :math:`+1`.

    """
    det = math.linalg.det(matrix)
    if math.is_abstract(matrix) or det < 0:
        # Adjust determinant to make unitary matrix special orthogonal; multiplication of
        # the first column with -1 is equal to prepending the decomposition with a phase shift
        matrix = (
            math.copy(matrix)
            if math.get_interface(matrix) == "jax"
            else math.toarray(matrix).copy()
        )
        matrix = math.T(math.set_index(math.T(matrix), 0, -matrix[:, 0]))
        return np.pi * (1 - det) / 2, matrix
    return None, matrix


class BasisRotation(Operation):
    r"""Implements a circuit that performs an exact single-body basis rotation using Givens
    rotations and phase shifts.

    The :class:`~.BasisRotation` template performs the following unitary transformation
    :math:`U(u)` determined by the single-particle fermionic
    generators as given in `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ :

    .. math::

        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.

    The unitary :math:`U(u)` is implemented efficiently by performing its Givens decomposition into a sequence of
    :class:`~.PhaseShift` and :class:`~.SingleExcitation` gates using the construction scheme given in
    `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ .
    For real-valued, i.e., orthogonal :math:`u`, only ``SingleExcitation`` gates are required,
    except for a :class:`~.PauliZ` phase flip for :math:`\operatorname{det}(u)=-1`.
    This can be expressed concisely by applying ``PhaseShift((1-det(u)) * π / 2)``.

    .. seealso:: :func:`~.math.decomposition.givens_decomposition` for the underlying matrix factorization.

    Args:
        wires (Iterable[Any]): wires that the operator acts on
        unitary_matrix (array): matrix specifying the basis transformation
        check (bool): test unitarity of the provided `unitary_matrix`

    Raises:
        ValueError: if the provided matrix is not square.
        ValueError: if length of the wires is less than two.

    .. details::
        :title: Usage Details
        :href: usage-basis-rotation

        The :class:`~.pennylane.BasisRotation` template can be used to implement the evolution :math:`e^{iH}` where
        :math:`H = \sum_{pq} V_{pq} a^\dagger_p a_q` and :math:`V` is an :math:`N \times N` Hermitian matrix.
        When the unitary matrix :math:`u` is the transformation matrix that diagonalizes :math:`V`, the evolution is:

        .. math::

            e^{i \sum_{pq} V_{pq} a^\dagger_p a_q} = U(u)^\dagger \prod_k e^{i\lambda_k \sigma_z^k} U(u),

        where :math:`\lambda_k` denotes the eigenvalues of matrix :math:`V`, the Hamiltonian coefficients matrix.

        >>> V = np.array([[ 0.53672126+0.j        , -0.1126064 -2.41479668j],
        ...               [-0.1126064 +2.41479668j,  1.48694623+0.j        ]])
        >>> eigen_vals, eigen_vecs = np.linalg.eigh(V)
        >>> umat = eigen_vecs.T
        >>> wires = range(len(umat))
        >>> def circuit():
        ...    qml.adjoint(qml.BasisRotation(wires=wires, unitary_matrix=umat))
        ...    for idx, eigenval in enumerate(eigen_vals):
        ...        qml.RZ(eigenval, wires=[idx])
        ...    qml.BasisRotation(wires=wires, unitary_matrix=umat)
        >>> circ_unitary = qml.matrix(circuit, wire_order=wires)()
        >>> np.round(circ_unitary/circ_unitary[0][0], 3)
        array([[ 1.   -0.j   , -0.   +0.j   , -0.   +0.j   , -0.   +0.j   ],
               [-0.   +0.j   , -0.516-0.596j, -0.302-0.536j, -0.   +0.j   ],
               [-0.   +0.j   ,  0.35 +0.506j, -0.311-0.724j, -0.   +0.j   ],
               [-0.   +0.j   , -0.   +0.j   , -0.   +0.j   , -0.438+0.899j]])

        The ``BasisRotation`` is implemented with :class:`~.PhaseShift` and
        :class:`~.SingleExcitation` gates:

        >>> print(qml.draw(qml.BasisRotation(wires=wires, unitary_matrix=umat).decomposition)())
        0: ──Rϕ(-1.52)─╭G(1.38)──Rϕ(-1.62)─┤
        1: ──Rϕ(1.62)──╰G(1.38)────────────┤

        For real-valued matrices, the decomposition only consists of ``SingleExcitation`` gates,
        except for one phase gate to account for negative determinants:

        >>> from scipy.stats import ortho_group
        >>> O = ortho_group.rvs(4, random_state=51)
        >>> print(qml.draw(qml.BasisRotation(wires=range(4), unitary_matrix=O).decomposition)())
        0: ──Rϕ(3.14)─╭G(-3.19)──────────╭G(2.63)─┤
        1: ─╭G(-3.13)─╰G(-3.19)─╭G(2.68)─╰G(2.63)─┤
        2: ─╰G(-3.13)─╭G(-2.98)─╰G(2.68)─╭G(5.70)─┤
        3: ───────────╰G(-2.98)──────────╰G(5.70)─┤

    .. details::
        :title: Theory
        :href: theory-basis-rotation

        The basis rotation performed by ``BasisRotation`` implements a transformation
        in the qubit Hilbert space that corresponds to a simple basis change of
        fermionic creation operators, translated to qubits via the Jordan-Wigner mapping.
        The old fermionic creation operators :math:`a_p^\dagger` and the new creation
        operators :math:`b_p^\dagger` are related to each other by the following equation:

        .. math::

            b_p^\dagger = \sum_{q}u_{pq} a_q^\dagger.

        The effect of :math:`U(u)`, the rotation in qubit Hilbert space, is then

        .. math::

            U(u) A_p^\dagger U(u)^\dagger = B_p^\dagger,

        where :math:`A_p^\dagger` and :math:`B_p^\dagger` are the original and transformed
        creation operators under the Jordan-Wigner transformation, respectively.

        **Underlying matrix factorization**

        The rotation is irreducibly represented as a unitary :math:`N\times N` matrix :math:`u`,
        which can be factorized into nearest-neighbour Givens rotations and individual phase
        shifts. Such a factorization of :math:`u` is implemented
        in :func:`~.math.decomposition.givens_decomposition`.

        The Givens rotations take the form

        .. math::

            T_{k}(\theta) = \begin{pmatrix}
                1 & 0 & \cdots & 0 & 0 & \cdots & 0 & 0 \\
                0 & 1 & & & & & 0 & 0 \\
                \vdots & & \ddots & & & & & \vdots \\
                0 & & & \cos(\theta) & -\sin(\theta) & & & 0 \\
                0 & & & \sin(\theta) & \cos(\theta) & & & 0 \\
                \vdots & & & & & \ddots & & \vdots \\
                0 & 0 & & & & & 1 & 0 \\
                0 & 0 & \cdots & 0 & 0  & \cdots & 0 & 1 \\
            \end{pmatrix},

        where the four non-trivial entries are at indices :math:`k` and :math:`k+1`.
        It will also be useful to look at the generator of :math:`T_{k}`:

        .. math::

            T_k(\theta) = \exp(\theta E_{k,k+1}),

        where :math:`E_{k,\ell}` is a matrix that is zero everywhere except for a
        :math:`-1` in position :math:`(k,\ell)` and a :math:`1` in position :math:`(\ell,k)`.
        The phase shifts in the decomposition read

        .. math::

            P_{j}(\phi) = \operatorname{diag}(1,\cdots, 1, e^{i\phi}, 1, \cdots, 1),

        with the single non-trivial entry at index :math:`j`.
        Such a phase shift is generated in the following form:

        .. math::

            P_j(\phi) = \exp(i \phi D_{j}),

        where :math:`D_j` is zero everywhere except for a :math:`1` in position :math:`(j,j)`.

        Next, we consider multiple Lie algebras generated by the :math:`E_{k,k+1}` and :math:`D_j`,
        and in particular the full Lie algebra :math:`\mathfrak{g}` generated by all of them:

        .. math::

            \langle\{E_{k,k+1}\}_k\rangle_\text{Lie} &= \text{span}_{\mathbb{R}}\{E_{k,\ell} | 1\leq k<\ell\leq N\}\cong\mathfrak{so}(N)\\
            \langle\{D_{j}\}_j\rangle_\text{Lie} &= \text{span}_{\mathbb{R}}\{D_{j} | 1\leq j\leq N\}\cong\mathfrak{u}(1)^N\\
            \mathfrak{g}=\langle\{D_{j}\}_j\cup\{E_{k,k+1}\}_k\rangle_\text{Lie}
            &=\text{span}_{\mathbb{R}}\left(\{E_{k,\ell}, F_{k,\ell}| 1\leq k<\ell \leq N\}\cup\{D_{j}\}_j\right)
            \cong \mathfrak{u}(N).

        The full algebra :math:`\mathfrak{g}` contains the matrices :math:`F_{k,\ell}` that
        look like the matrices :math:`E_{k,\ell}`, but without any
        minus sign. There are ``N(N-1) / 2`` matrices :math:`E` and :math:`F` each,
        and ``N`` matrices :math:`D`. So that all taken together span the
        :math:`N^2`-dimensional algebra :math:`\mathfrak{u}(N)`.

        **Mapping the matrix factorization to a circuit**

        The factorization of :math:`u` can be mapped to a quantum circuit by identifying:

        .. math::

            T_{k}(\theta) &\mapsto \texttt{SingleExcitation}(2\theta,\texttt{wires=[k, k+1]}),\\
            P_{j}(\phi) &\mapsto \texttt{PhaseShift}(\phi, \texttt{wires=[j]}).

        This identification is a group homomorphism, which is proven in
        `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_. We can understand this
        by looking at the algebra to which this mapping sends the algebra :math:`\mathfrak{g}`
        from above.
        The ``SingleExcitation`` gates have the generators
        :math:`\hat{E}_{k,k+1}=\tfrac{i}{2}(X_k Y_{k+1} - Y_k X_{k+1})`
        (note the additional prefactor of :math:`2` from the mapping):

        >>> qml.generator(qml.SingleExcitation(0.2512, [0, 1]))
        (X(0) @ Y(1) + -1.0 * (Y(0) @ X(1)), np.float64(0.25))

        Similarly, the ``PhaseShift`` gates have the generators
        :math:`\hat{D}_j=\tfrac{i}{2}(\mathbb{I}-Z_j)=i|1\rangle\langle 1|_j`:

        >>> qml.generator(qml.PhaseShift(0.742, [0]))
        (Projector(array([1]), wires=[0]), 1.0)

        It turns out that these generators :math:`\hat{E}_{k,k+1}` and :math:`\hat{D}_j`
        have commutation relations equivalent to those of
        the irreducible matrices above, with a crucial feature in how the identification
        :math:`E_{k,k+1}\mapsto \hat{E}_{k,k+1}` generalizes.
        One could try to map, say, :math:`E_{2, 4}` to :math:`\tfrac{i}{2}(X_2 Y_4 -Y_2 X_4)`,
        but this will not be consistent with the operators and commutation relations in the
        algebra. Instead, we need to insert strings of Pauli :math:`Z` operators whenever the
        interaction encoded by the generator is not between nearest neighbours, so that
        :math:`E_{2,4}` maps to :math:`\tfrac{i}{2}(X_2 Z_3 Y_4 -Y_2 Z_3 X_4)`,
        which we also denote as :math:`\tfrac{i}{2}(\overline{X_2 Y_4} -\overline{Y_2 X_4})`.
        Then we have

        .. math::

            E_{k,\ell} & \mapsto \hat{E}_{k,\ell} = \tfrac{i}{2}(\overline{X_kY_\ell}-\overline{Y_kX_\ell}),\\
            F_{k,\ell} & \mapsto \hat{F}_{k,\ell} = \tfrac{i}{2}(\overline{X_kX_\ell}+\overline{Y_kY_\ell}),\\
            D_{j} & \mapsto \hat{D}_{j} = \tfrac{i}{2}(\mathbb{I}-Z_j).

        The fact that we need to use :math:`\overline{X_kY_\ell}` instead of :math:`X_kY_\ell`
        is a consequence of mapping fermions onto qubits via the Jordan-Wigner transformation.
        Depending on the application, the relative signs between operators in this mapping need
        to be evaluated with extra care.

        **Real rotation matrices**

        It is common for the orbital rotation matrix implemented by ``BasisRotation`` to be
        real-valued and thus orthogonal. The generators :math:`E_{k,k+1}` generate the Lie
        algebra :math:`\mathfrak{so}(N)`, which means that the Givens rotations
        :math:`T_k` are sufficient to factorize the full rotation, and only ``SingleExcitation``
        gates are needed in the quantum circuit. A small exception occurs for orthogonal matrices
        that are not *special* orthogonal (unit determinant), i.e., that have
        determinant :math:`-1`. For those, the sign of one row or column of the matrix can be
        inverted, corresponding to a phase flip gate :class:`~.PauliZ` in the circuit.
        The new matrix then has a unit determinant and can be
        implemented exclusively with ``SingleExcitation`` gates.

    """

    grad_method = None

    resource_keys = {"dim", "is_real"}

    @classmethod
    def _primitive_bind_call(cls, wires, unitary_matrix, check=False, id=None):
        # pylint: disable=arguments-differ
        if cls._primitive is None:
            # guard against this being called when primitive is not defined.
            return type.__call__(cls, wires, unitary_matrix, check=check, id=id)  # pragma: no cover

        return cls._primitive.bind(*wires, unitary_matrix, check=check, id=id)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(wires=metadata[0], unitary_matrix=data[0])

    def __init__(self, wires, unitary_matrix, check=False, id=None):
        M, N = math.shape(unitary_matrix)

        if M != N:
            raise ValueError(f"The unitary matrix should be of shape NxN, got {(M, N)}")

        if check:
            if not math.is_abstract(unitary_matrix) and not math.allclose(
                unitary_matrix @ math.conj(unitary_matrix).T,
                math.eye(M, dtype=complex),
                atol=1e-4,
            ):
                raise ValueError("The provided transformation matrix should be unitary.")

        if len(wires) < 2:
            raise ValueError(f"This template requires at least two wires, got {len(wires)}")

        super().__init__(unitary_matrix, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "dim": math.shape(self.data[0])[0],
            "is_real": math.is_real_obj_or_close(self.data[0]),
        }

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(
        unitary_matrix, wires, check=False
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.BasisRotation.decomposition`.

        Args:
            wires (Any or Iterable[Any]): wires that the operator acts on
            unitary_matrix (array): matrix specifying the basis transformation
            check (bool): test unitarity of the provided `unitary_matrix`

        Returns:
            list[.Operator]: decomposition of the operator
        """

        if check:
            if not math.is_abstract(unitary_matrix) and not math.allclose(
                unitary_matrix @ math.conj(unitary_matrix).T,
                math.eye(len(unitary_matrix), dtype=complex),
                atol=1e-4,
            ):
                raise ValueError("The provided transformation matrix should be unitary.")

        if len(wires) < 2:
            raise ValueError(f"This template requires at least two wires, got {len(wires)}")

        op_list = []

        if math.is_real_obj_or_close(unitary_matrix):
            angle, unitary_matrix = _adjust_determinant(unitary_matrix)
            if angle is not None:
                op_list.append(PhaseShift(angle, wires=wires[0]))

            _, givens_list = math.decomposition.givens_decomposition(unitary_matrix)
            for grot_mat, (i, j) in givens_list:
                theta = math.arctan2(grot_mat[0, 1], grot_mat[0, 0])
                op_list.append(SingleExcitation(2 * theta, wires=[wires[i], wires[j]]))
            return op_list

        phase_list, givens_list = math.decomposition.givens_decomposition(unitary_matrix)

        for idx, phase in enumerate(phase_list):
            op_list.append(PhaseShift(math.angle(phase), wires=wires[idx]))

        for grot_mat, (i, j) in givens_list:
            theta = math.arccos(math.real(grot_mat[1, 1]))
            phi = math.angle(grot_mat[0, 0])

            op_list.append(SingleExcitation(2 * theta, wires=[wires[i], wires[j]]))

            if math.is_abstract(phi) or not math.isclose(phi, phi * 0.0):
                op_list.append(PhaseShift(phi, wires=wires[i]))

        return op_list


def _basis_rotation_decomp_resources(dim, is_real):
    se_count = dim * (dim - 1) / 2
    if is_real:
        return {PhaseShift: 1, SingleExcitation: se_count}

    ps_count = dim + se_count
    return {PhaseShift: ps_count, SingleExcitation: se_count}


# Not exact because PhaseShift(s) might be skipped
@register_resources(_basis_rotation_decomp_resources, exact=False)
def _basis_rotation_decomp(unitary_matrix, wires: WiresLike, **__):

    if math.is_real_obj_or_close(unitary_matrix):
        angle, unitary_matrix = _adjust_determinant(unitary_matrix)
        if angle is not None:
            PhaseShift(angle, wires=wires[0])

        _, givens_list = math.decomposition.givens_decomposition(unitary_matrix)
        for grot_mat, (i, j) in givens_list:
            theta = math.arctan2(grot_mat[0, 1], grot_mat[0, 0])
            SingleExcitation(2 * theta, wires=[wires[i], wires[j]])
        return

    phase_list, givens_list = math.decomposition.givens_decomposition(unitary_matrix)

    for idx, phase in enumerate(phase_list):
        PhaseShift(math.angle(phase), wires=wires[idx])

    for grot_mat, (i, j) in givens_list:
        theta = math.arccos(math.real(grot_mat[1, 1]))
        phi = math.angle(grot_mat[0, 0])
        SingleExcitation(2 * theta, wires=[wires[i], wires[j]])
        cond(not math.allclose(phi, 0.0), PhaseShift)(phi, wires[i])


add_decomps(BasisRotation, _basis_rotation_decomp)


# Program capture needs to unpack and re-pack the wires to support dynamic wires. For
# BasisRotation, the unconventional argument ordering requires custom def_impl code.
# See capture module for more information on primitives
# If None, jax isn't installed so the class never got a primitive.
if BasisRotation._primitive is not None:  # pylint: disable=protected-access

    @BasisRotation._primitive.def_impl  # pylint: disable=protected-access
    def _(*args, **kwargs):
        # If there are more than two args, we are calling with unpacked wires, so that
        # we have to repack them. This replaces the n_wires logic in the general case.
        if len(args) != 2:
            args = (args[:-1], args[-1])
        return type.__call__(BasisRotation, *args, **kwargs)
