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
Contains templates for Quantum Signal Processing (QSP) based subroutines.
"""
import scipy.special as sps

import pennylane.numpy as qnp
from pennylane.estimator.ops.op_math.symbolic import Adjoint, Controlled
from pennylane.estimator.ops.qubit.parametric_ops_multi_qubit import PCPhase
from pennylane.estimator.ops.qubit.parametric_ops_single_qubit import RX, RZ, Rot
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    _dequeue,
    resource_rep,
)
from pennylane.wires import Wires, WiresLike


class GQSP(ResourceOperator):
    r"""Resource class for the Generalized Quantum Signal Processing (GQSP) algorithm.
    As described in theorem 3 of `Generalized Quantum Signal Processing (2024)
    <https://arxiv.org/pdf/2308.01501>`_.

    Args:
        signal_operator (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): The
            signal operator which encodes the target hamiltonian.
        poly_deg (int): the maximum positive degree :math:`d` of the polynomial transformation
        neg_poly_deg (int): the maximum negative degree :math:`k` of the polynomial transformation
        rot_precision (float, None): The precision with which to apply the general SU(2) rotation gates.
        wires (Sequence[int], None): The wires the operation acts on. This includes both the wires of the
            signal operator and the control wire required for block-encoding.

    Resources:
        The resources are obtained as described in theorem 3 of `Generalized Quantum Signal Processing
        (2024) <https://arxiv.org/pdf/2308.01501>`_.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    """

    resource_keys = {"cmpr_signal_op", "poly_deg", "neg_poly_deg", "rot_precision"}

    def __init__(
        self,
        signal_operator: ResourceOperator,
        poly_deg: int,
        neg_poly_deg: int = 0,
        rot_precision: float | None = None,
        wires: WiresLike = None,
    ):
        _dequeue(signal_operator)  # remove operator
        self.queue()

        self.poly_deg = poly_deg
        self.neg_poly_deg = neg_poly_deg
        self.rot_precision = rot_precision
        self.cmpr_signal_op = signal_operator.resource_rep_from_op()

        self.num_wires = signal_operator.num_wires + 1  # add control wire

        if wires:
            self.wires = Wires(wires)
            if base_wires := signal_operator.wires:
                self.wires = Wires.all_wires([self.wires, base_wires])
            if len(self.wires) != self.num_wires:
                raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}.")
        else:
            self.wires = None

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * cmpr_signal_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                  The compressed representation of signal operator which encodes the target hamiltonian.
                * poly_deg (int): the maximum positive degree :math:`d` of the polynomial transformation
                * neg_poly_deg (int): the maximum negative degree :math:`k` of the polynomial
                  transformation
                * rot_precision (float, None): The precision with which to apply the general SU(2)
                  rotation gates.
        """

        return {
            "cmpr_signal_op": self.cmpr_signal_op,
            "poly_deg": self.poly_deg,
            "neg_poly_deg": self.neg_poly_deg,
            "rot_precision": self.rot_precision,
        }

    @classmethod
    def resource_rep(
        cls,
        cmpr_signal_op: CompressedResourceOp,
        poly_deg: int,
        neg_poly_deg: int,
        rot_precision: float | None,
    ):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            cmpr_signal_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                The compressed representation of signal operator which encodes the target hamiltonian.
            poly_deg (int): the maximum positive degree :math:`d` of the polynomial transformation
            neg_poly_deg (int): the maximum negative degree :math:`k` of the polynomial transformation
            is_controlled (bool): Is ``True`` if the provided ``signal_operator`` is already
                block-encoded via a signle qubit control.
            rot_precision (float, None): The precision with which to apply the general SU(2)
                rotation gates.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = cmpr_signal_op.num_wires + 1  # add control wire
        params = {
            "cmpr_signal_op": cmpr_signal_op,
            "poly_deg": poly_deg,
            "neg_poly_deg": neg_poly_deg,
            "rot_precision": rot_precision,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        cmpr_signal_op: CompressedResourceOp,
        poly_deg: int,
        neg_poly_deg: int,
        rot_precision: float | None,
    ):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            cmpr_signal_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                The compressed representation of signal operator which encodes the target hamiltonian.
            poly_deg (int): the maximum positive degree :math:`d` of the polynomial transformation
            neg_poly_deg (int): the maximum negative degree :math:`k` of the polynomial transformation
            rot_precision (float, None): The precision with which to apply the general SU(2) rotation gates.
            wires (Sequence[int], None): The wires the operation acts on. This includes both the wires of the
                signal operator and the control wire required for block-encoding.

        Resources:
            The resources are obtained as described in theorem 3 of `Generalized Quantum Signal Processing
            (2024) <https://arxiv.org/pdf/2308.01501>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        rot = Rot.resource_rep(precision=rot_precision)
        adj_cmpr_signal_op = Adjoint.resource_rep(cmpr_signal_op)

        ctrl_cmpr_signal_op = Controlled.resource_rep(
            base_cmpr_op=cmpr_signal_op,
            num_ctrl_wires=1,
            num_zero_ctrl=1,
        )

        ctrl_adj_cmpr_signal_op = Controlled.resource_rep(
            base_cmpr_op=adj_cmpr_signal_op,
            num_ctrl_wires=1,
            num_zero_ctrl=0,
        )

        if neg_poly_deg == 0:
            return [GateCount(rot, poly_deg + 1), GateCount(ctrl_cmpr_signal_op, poly_deg)]

        return [
            GateCount(rot, poly_deg + neg_poly_deg + 1),
            GateCount(ctrl_cmpr_signal_op, poly_deg),
            GateCount(ctrl_adj_cmpr_signal_op, neg_poly_deg),
        ]


class HamSimGQSP(ResourceOperator):
    r"""Resource class for performing hamiltonian simulation using GQSP.

    Args:
        walk_op (ResourceOperator): the quantum walk operator
        time (float): the simulation time
        one_norm (float): one norm of the hamiltonian
        approximation_error (float): the tolerance for error in the polynomial approximation of :math:`e^{it\cos{theta}}`
        wires (Sequence[int], None): the wires the operation acts on

    Resources:
        The resources are obtained as described in theorem 7 and corollary 8 of `Generalized Quantum Signal Processing
        (2024) <https://arxiv.org/pdf/2308.01501>`_.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    """

    resource_keys = {"walk_op", "time", "one_norm", "approximation_error"}

    def __init__(
        self,
        walk_op: ResourceOperator,
        time: float,
        one_norm: float,
        approximation_error: float,
        wires: WiresLike = None,
    ):
        _dequeue(walk_op)  # remove operator
        self.queue()

        self.walk_op = walk_op.resource_rep_from_op()
        self.time = time
        self.one_norm = one_norm
        self.approximation_error = approximation_error
        self.num_wires = walk_op.num_wires + 1  # control wire

        if wires:
            self.wires = Wires(wires)
            if walk_op_wires := walk_op.wires:
                self.wires = Wires.all_wires([self.wires, walk_op_wires])
            if len(self.wires) != self.num_wires:
                raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}.")
        else:
            self.wires = None

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * walk_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                  The quantum walk operator.
                * time (float): the simulation time
                * one_norm (float): one norm of the hamiltonian
                * approximation_error (float): the tolerance for error in the polynomial
                  approximation of :math:`e^{it\cos{theta}}`
        """

        return {
            "walk_op": self.walk_op,
            "time": self.time,
            "one_norm": self.one_norm,
            "approximation_error": self.approximation_error,
        }

    @classmethod
    def resource_rep(
        cls,
        walk_op: CompressedResourceOp,
        time: float,
        one_norm: float,
        approximation_error: float,
    ):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            walk_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): The
                quantum walk operator.
            time (float): the simulation time
            one_norm (float): one norm of the hamiltonian
            approximation_error (float): the tolerance for error in the polynomial approximation of
                :math:`e^{it\cos{theta}}`

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = walk_op.num_wires + 1
        params = {
            "walk_op": walk_op,
            "time": time,
            "one_norm": one_norm,
            "approximation_error": approximation_error,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        walk_op: CompressedResourceOp,
        time: float,
        one_norm: float,
        approximation_error: float,
    ):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            walk_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): The
                quantum walk operator.
            time (float): the simulation time
            one_norm (float): one norm of the hamiltonian
            approximation_error (float): the tolerance for error in the polynomial approximation of
                :math:`e^{it\cos{theta}}`
        Resources:
            The resources are obtained as described in theorem 7 and corollary 8 of `Generalized Quantum Signal Processing
            (2024) <https://arxiv.org/pdf/2308.01501>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        poly_deg = cls.degree_of_poly_approx(time, one_norm, approximation_error)
        gqsp = GQSP.resource_rep(
            walk_op,
            poly_deg=poly_deg,
            neg_poly_deg=poly_deg,
            rot_precision=None,
        )
        return [GateCount(gqsp)]

    @staticmethod
    def degree_of_poly_approx(time, one_norm, epsilon):
        """Obtain the maximum degree of the polynomial approximation required
        to approximate e^(iht * cos(theta)) within error epsilon

        Args:
            time (float): the simulation time
            one_norm (float): one norm of the hamiltonian
            epsilon (float): the tolerance for error in the polynomial approximation of :math:`e^{it\cos{theta}}`

        Returns:
            int: the minimum degree of the polynomial approximation
        """
        N_0 = int(qnp.ceil(qnp.abs(time * one_norm)))  # initial guess for the degree
        error = qnp.abs(sps.jv(N_0 + 1, time * one_norm))  # initial error

        N = N_0
        while error > (epsilon / 2):
            N += 1
            error = qnp.abs(sps.jv(N + 1, time * one_norm))

        return N


class QSVT(ResourceOperator):
    r"""Implements the `Quantum Singular Value Transformation <https://arxiv.org/abs/1806.01838>`_
    (QSVT) circuit.

    Given a :class:`~.estimator.resource_operator.ResourceOperator` :math:`U`, which block encodes
    the matrix :math:`A` in the top left with subspace dimension ``dims = (A_n, A_m)``; and the degree
    of the target polynomial transformation ``poly_deg = d``, this template applies a
    circuit for the quantum singular value transformation as follows.

    When the degree of the polynomial is odd, the QSVT circuit is defined as:

    .. math::

        U_{QSVT} = \tilde{\Pi}_{\phi_1}U\left[\prod^{(d-1)/2}_{k=1}\Pi_{\phi_{2k}}U^\dagger
        \tilde{\Pi}_{\phi_{2k+1}}U\right].


    And when the number of degree is even:

    .. math::

        U_{QSVT} = \left[\prod^{d/2}_{k=1}\Pi_{\phi_{2k-1}}U^\dagger\tilde{\Pi}_{\phi_{2k}}U\right].

    This circuit applies a polynomial transformation of degree :math:`d` (:math:`Poly^{SV}`) to the
    singular values of the block encoded matrix:

    .. math::

        \begin{align}
             U_{QSVT}(A, \vec{\phi}) &=
             \begin{bmatrix}
                Poly^{SV}(A) & \cdot \\
                \cdot & \cdot
            \end{bmatrix}.
        \end{align}

    .. seealso::

        :func:`~.qsvt` and :class:`~.QSVT`.

    Args:
        block_encoding (ResourceOperator): the block encoding operator
        encoding_dims (int | tuple(int)): The subspace (number of rows and columns) where the operator
            is encoded in the matrix representation of the block encoding operator.
        poly_deg (int): the degree of the polynomial transformation being applied
        wires (Sequence[int], None): the wires the operation acts on

    Resources:
        The resources are obtained as described in theorem 4 of `A Grand Unification of Quantum Algorithms
        (2021) <https://arxiv.org/pdf/2105.02859>`_.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    """

    resource_keys = {"block_encoding", "encoding_dims", "poly_deg"}

    def __init__(
        self,
        block_encoding: ResourceOperator,
        encoding_dims: int | tuple[int],
        poly_deg: int,
        wires: WiresLike = None,
    ):
        _dequeue(block_encoding)  # remove operator
        if not isinstance(encoding_dims, (int, tuple)):
            raise ValueError(
                f"Expected `encoding_dims` to be an int or tuple of int. Got {encoding_dims}"
            )

        if isinstance(encoding_dims, int):
            encoding_dims = (encoding_dims, encoding_dims)

        if len(encoding_dims) == 1:
            dim = encoding_dims[0]
            encoding_dims = (dim, dim)
        elif len(encoding_dims) > 2:
            raise ValueError(
                "Expected `encoding_dims` to be a tuple of two integers, representing the dimensions"
                f" (row, col) of the subspace where the matrix is encoded. Got {encoding_dims}"
            )

        self.block_encoding = block_encoding.resource_rep_from_op()
        self.encoding_dims = encoding_dims
        self.poly_deg = poly_deg

        self.num_wires = block_encoding.num_wires

        if wires is None and block_encoding.wires is not None:
            wires = block_encoding.wires

        if len(wires) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(wires)}.")
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * block_encoding (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                  The block encoding operator.
                * encoding_dims (int | tuple(int)): The subspace (number of rows and columns) where
                  the operator is encoded in the matrix representation of the block encoding operator.
                * poly_deg (int): the degree of the polynomial transformation being applied
        """
        return {
            "block_encoding": self.block_encoding,
            "encoding_dims": self.encoding_dims,
            "poly_deg": self.poly_deg,
        }

    @classmethod
    def resource_rep(
        cls,
        block_encoding: CompressedResourceOp,
        encoding_dims: int,
        poly_deg: int,
    ):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            block_encoding (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                The block encoding operator.
            encoding_dims (int | tuple(int)): The subspace (number of rows and columns) where
                the operator is encoded in the matrix representation of the block encoding operator.
            poly_deg (int): the degree of the polynomial transformation being applied

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = block_encoding.num_wires
        params = {
            "block_encoding": block_encoding,
            "encoding_dims": encoding_dims,
            "poly_deg": poly_deg,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        block_encoding: CompressedResourceOp,
        encoding_dims: int,
        poly_deg: int,
    ):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            block_encoding (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                The block encoding operator.
            encoding_dims (int | tuple(int)): The subspace (number of rows and columns) where
                the operator is encoded in the matrix representation of the block encoding operator.
            poly_deg (int): the degree of the polynomial transformation being applied

        Resources:
            The resources are obtained as described in theorem 4 of `A Grand Unification of Quantum Algorithms
            (2021) <https://arxiv.org/pdf/2105.02859>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        num_rows, num_cols = encoding_dims
        num_wires = block_encoding.num_wires

        pi = PCPhase.resource_rep(num_wires=num_wires, dim=num_cols)
        pi_tilde = PCPhase.resource_rep(num_wires=num_wires, dim=num_rows)
        block_encoding_adj = Adjoint.resource_rep(block_encoding)

        if poly_deg % 2 == 0:  # even degree
            be_counts = poly_deg // 2
            be_adj_counts = poly_deg // 2
            pi_counts = poly_deg // 2
            pi_tilde_counts = poly_deg // 2

        else:  # odd degree
            be_counts = ((poly_deg - 1) // 2) + 1
            be_adj_counts = (poly_deg - 1) // 2
            pi_counts = (poly_deg - 1) // 2
            pi_tilde_counts = ((poly_deg - 1) // 2) + 1

        return [
            GateCount(block_encoding, be_counts),
            GateCount(block_encoding_adj, be_adj_counts),
            GateCount(pi, pi_counts),
            GateCount(pi_tilde, pi_tilde_counts),
        ]


class QSP(ResourceOperator):
    r"""Implements the `Quantum Signal Processing <https://arxiv.org/pdf/2105.02859>`_
    (QSP) circuit.

    Given a single qubit :class:`~.estimator.resource_operator.ResourceOperator` :math:`W(a)`, which
    block encodes the value :math:`a` as the top left entry in the matrix representation; and the
    degree of the target polynomial transformation ``poly_deg = d``, this template estimates the
    cost of the QSP circuit.

    The circuit is given as follows in the z-convention (``convention="z"``):

    .. math::

        U_{QSP} = e^{i\phi_{0}Z}\prod^{d}_{k=1}W(a)e^{i\phi_{k}Z} .

    The circuit can also be expressed in the x-convention (``convention="x"``):

    .. math::

        U_{QSP} = e^{i\phi_{0}X}\prod^{d}_{k=1}W(a)e^{i\phi_{k}X} .

    .. seealso::

        :func:`~.qsvt` and :class:`~.QSVT`.

    Args:
        block_encoding (ResourceOperator): the block encoding operator
        poly_deg (int): the degree of the polynomial transformation being applied
        convention (str): the convention used by the rotations and the block encoding operator
        rotation_precision (float | None): the p
        wires (Sequence[int], None): the wires the operation acts on

    Resources:
        The resources are obtained as described in theorem 1 of `A Grand Unification of Quantum Algorithms
        (2021) <https://arxiv.org/pdf/2105.02859>`_.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    """

    resource_keys = {"block_encoding", "poly_deg", "convention"}

    def __init__(
        self,
        block_encoding: ResourceOperator,
        poly_deg: int,
        convention: str = "z",
        wires: WiresLike = None,
    ):
        _dequeue(block_encoding)  # remove operator
        if block_encoding.num_wires > 1:
            raise ValueError("The block encoding operator should act on a single qubit!")

        self.block_encoding = block_encoding.resource_rep_from_op()
        self.convention = convention
        self.poly_deg = poly_deg

        self.num_wires = 1
        if wires is not None and len(wires) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(wires)}.")

        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * block_encoding (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                  The block encoding operator.
                * encoding_dims (int | tuple(int)): The subspace (number of rows and columns) where
                  the operator is encoded in the matrix representation of the block encoding operator.
                * poly_deg (int): the degree of the polynomial transformation being applied
        """
        return {
            "block_encoding": self.block_encoding,
            "poly_deg": self.poly_deg,
            "convention": self.convention,
        }

    @classmethod
    def resource_rep(
        cls,
        block_encoding: CompressedResourceOp,
        poly_deg: int,
        convention: str,
    ):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            block_encoding (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                The block encoding operator.
            poly_deg (int): the degree of the polynomial transformation being applied
            convention (str): specifies the convention used for QSP

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        params = {
            "block_encoding": block_encoding,
            "poly_deg": poly_deg,
            "convention": convention,
        }
        return CompressedResourceOp(cls, num_wires=1, params=params)

    @classmethod
    def resource_decomp(
        cls,
        block_encoding: CompressedResourceOp,
        encoding_dims: int,
        poly_deg: int,
    ):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            block_encoding (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                The block encoding operator.
            poly_deg (int): the degree of the polynomial transformation being applied
            convention (str): specifies the convention used for QSP

        Resources:
            The resources are obtained as described in theorem 1 of `A Grand Unification of Quantum Algorithms
            (2021) <https://arxiv.org/pdf/2105.02859>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        rot_op = RZ.resource_rep()

        return [
            GateCount(block_encoding, be_counts),
            GateCount(block_encoding_adj, be_adj_counts),
            GateCount(pi, pi_counts),
            GateCount(pi_tilde, pi_tilde_counts),
        ]
