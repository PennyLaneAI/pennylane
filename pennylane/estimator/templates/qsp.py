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
    GateCount,
    ResourceOperator,
    _dequeue,
)
from pennylane.wires import Wires, WiresLike

# pylint: disable=arguments-differ,super-init-not-called, signature-differs, too-many-arguments


class GQSP(ResourceOperator):
    r"""Resource class for the Generalized Quantum Signal Processing (GQSP) algorithm.

    The ``GQSP`` operator is defined based on Theorem 6 of `Generalized Quantum Signal Processing (2024)
    <https://arxiv.org/pdf/2308.01501>`_:

    .. math::

        GQSP = \left( \prod_{j=1}^{d^{-}} R(\theta_{j}, \phi_{j}, 0) \hat{A}^{\prime} \right)
        \left( \prod_{j=1}^{d^{+}} R(\theta_{j + d^{-}}, \phi_{j + d^{-}}, 0) \hat{A} \right) R(\theta_0, \phi_0, \lambda),

    where :math:`R` is the single-qubit rotation operator and :math:`\vec{\phi}`, :math:`\vec{\theta}` and :math:`\lambda`
    are the rotation angles that generate the polynomial transformation. The maximum positive and
    negative polynomial degrees are denoted by :math:`d^{+}` and :math:`d^{-}`, respectively.
    Additionally, :math:`\hat{A}` and :math:`\hat{A}^{\prime}` are given by:

    .. math::

        \begin{align}
            \hat{A} &= \ket{0}\bra{0}\otimes\hat{U} + \ket{1}\bra{1}\otimes\mathbf{I}, \\
            \hat{A}^{\prime} &= \ket{0}\bra{0}\otimes\mathbf{I} + \ket{1}\bra{1}\otimes\hat{U}^{\dagger}, \\ \\
        \end{align}

    where :math:`U` is a signal operator which encodes a target Hamiltonian.

    Args:
        signal_operator (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): the
            signal operator which encodes a target Hamiltonian
        d_plus (int): The largest positive degree :math:`d^{+}` of the polynomial transformation.
        d_minus (int): The largest (in absolute value) negative degree :math:`d^{-}` of the polynomial
            transformation, representing powers of the adjoint of the signal operator.
        rotation_precision (float | None): The precision with which the general rotation gates are applied.
        wires (WiresLike | None): The wires the operation acts on. This includes both the wires of the
            signal operator and the control wire required for block-encoding.

    Resources:
        The resources are obtained as described in Theorem 6 of `Generalized Quantum Signal
        Processing (2024) <https://arxiv.org/pdf/2308.01501>`_.

    Raises:
        ValueError: if ``d_plus`` is not a positive integer greater than zero
        ValueError: if ``d_minus`` is not an integer greater than or equal to zero
        ValueError: if ``rotation_precision`` is not a positive real number greater than zero
        ValueError: if the wires provided don't match the number of wires expected by the operator

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> signal_op = qre.RX(0.1, wires=0)
    >>> d_plus = 5
    >>> d_minus = 3
    >>> gqsp = qre.GQSP(signal_op, d_plus, d_minus)
    >>> print(qre.estimate(gqsp))
    --- Resources: ---
     Total wires: 2
       algorithmic wires: 2
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 1.438E+3
       'T': 1.396E+3,
       'CNOT': 16,
       'X': 10,
       'Hadamard': 16

    """

    resource_keys = {"cmpr_signal_op", "d_plus", "d_minus", "rotation_precision"}

    def __init__(
        self,
        signal_operator: ResourceOperator,
        d_plus: int,
        d_minus: int = 0,
        rotation_precision: float | None = None,
        wires: WiresLike = None,
    ):
        _dequeue(signal_operator)  # remove operator
        self.queue()

        if (not isinstance(d_plus, int)) or d_plus <= 0:
            raise ValueError(f"'d_plus' must be a positive integer greater than zero, got {d_plus}")

        if (not isinstance(d_minus, int)) or d_minus < 0:
            raise ValueError(f"'d_minus' must be a non-negative integer, got {d_minus}")

        if rotation_precision is not None and rotation_precision <= 0:
            raise ValueError(
                f"Expected 'rotation_precision' to be a positive real number greater than zero, got {rotation_precision}"
            )

        self.d_plus = d_plus
        self.d_minus = d_minus
        self.rotation_precision = rotation_precision
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
                * cmpr_signal_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`):
                  the compressed representation of the signal operator which encodes the target Hamiltonian
                * d_plus (int): The largest positive degree :math:`d^{+}` of the polynomial transformation.
                * d_minus (int): The largest (in absolute value) negative degree :math:`d^{-}` of the
                  polynomial transformation, representing powers of the adjoint of the signal operator.
                * rotation_precision (float | None): The precision with which the general
                  rotation gates are applied.
        """

        return {
            "cmpr_signal_op": self.cmpr_signal_op,
            "d_plus": self.d_plus,
            "d_minus": self.d_minus,
            "rotation_precision": self.rotation_precision,
        }

    @classmethod
    def resource_rep(
        cls,
        cmpr_signal_op: ResourceOperator,
        d_plus: int,
        d_minus: int = 0,
        rotation_precision: float | None = None,
    ) -> ResourceOperator:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            cmpr_signal_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`):
                the compressed representation of the signal operator which encodes the target Hamiltonian
            d_plus (int): The largest positive degree :math:`d^{+}` of the polynomial transformation.
            d_minus (int): The largest (in absolute value) negative degree :math:`d^{-}` of the polynomial
                transformation, representing powers of the adjoint of the signal operator.
            rotation_precision (float | None): The precision with which the general rotation gates are applied.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.ResourceOperator`: the operator in a compressed representation
        """
        if (not isinstance(d_plus, int)) or d_plus <= 0:
            raise ValueError(f"'d_plus' must be a positive integer greater than zero, got {d_plus}")

        if (not isinstance(d_minus, int)) or d_minus < 0:
            raise ValueError(f"'d_minus' must be a non-negative integer, got {d_minus}")

        if rotation_precision is not None and rotation_precision <= 0:
            raise ValueError(
                f"Expected 'rotation_precision' to be a positive real number greater than zero, got {rotation_precision}"
            )

        return cls(
            signal_operator=cmpr_signal_op,
            d_plus=d_plus,
            d_minus=d_minus,
            rotation_precision=rotation_precision,
        )

    @classmethod
    def resource_decomp(
        cls,
        cmpr_signal_op: ResourceOperator,
        d_plus: int,
        d_minus: int = 0,
        rotation_precision: float | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            cmpr_signal_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`):
                the compressed representation of the signal operator which encodes the target Hamiltonian
            d_plus (int): The largest positive degree :math:`d^{+}` of the polynomial transformation.
            d_minus (int): The largest (in absolute value) negative degree :math:`d^{-}` of the polynomial
                transformation, representing powers of the adjoint of the signal operator.
            rotation_precision (float | None): The precision with which the general rotation gates are applied.

        Resources:
            The resources are obtained as described in Theorem 6 of
            `Generalized Quantum Signal Processing (2024) <https://arxiv.org/pdf/2308.01501>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        rot = Rot.resource_rep(precision=rotation_precision)
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

        if d_minus == 0:
            return [GateCount(rot, d_plus + 1), GateCount(ctrl_cmpr_signal_op, d_plus)]

        return [
            GateCount(rot, d_plus + d_minus + 1),
            GateCount(ctrl_cmpr_signal_op, d_plus),
            GateCount(ctrl_adj_cmpr_signal_op, d_minus),
        ]


class GQSPTimeEvolution(ResourceOperator):
    r"""Resource class for performing Hamiltonian simulation using GQSP.

    Args:
        walk_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): the quantum walk operator
        time (float): the simulation time
        one_norm (float): one norm of the Hamiltonian
        poly_approx_precision (float): the tolerance for error in the polynomial approximation
        wires (WiresLike | None): The wires the operation acts on. This includes both the wires of the
            signal operator and the control wire required for block-encoding.

    Resources:
        The resources are obtained as described in Theorem 7 and Corollary 8 of
        `Generalized Quantum Signal Processing (2024) <https://arxiv.org/pdf/2308.01501>`_.

    Raises:
        ValueError: if the ``wires`` provided don't match the number of wires expected by the operator
        ValueError: if the ``time`` provided is not a positive real number greater than zero
        ValueError: if the ``one_norm`` provided is not a positive real number greater than zero
        ValueError: if the ``poly_approx_precision`` provided is not a positive real number greater than zero

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> walk_op = qre.RX(0.1, wires=0)
    >>> time = 1.0
    >>> one_norm = 1.0
    >>> approx_error = 0.01
    >>> hamsim = qre.GQSPTimeEvolution(walk_op, time, one_norm, approx_error)
    >>> print(qre.estimate(hamsim))
    --- Resources: ---
     Total wires: 2
       algorithmic wires: 2
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 1.110E+3
       'T': 1.080E+3,
       'CNOT': 12,
       'X': 6,
       'Hadamard': 12

    """

    resource_keys = {"walk_op", "time", "one_norm", "poly_approx_precision"}

    def __init__(
        self,
        walk_op: ResourceOperator,
        time: float,
        one_norm: float,
        poly_approx_precision: float | None = None,
        wires: WiresLike = None,
    ):
        _dequeue(walk_op)  # remove operator
        self.queue()

        if (not isinstance(time, (int, float))) or time <= 0:
            raise (
                ValueError(
                    f"Expected 'time' to be a positive real number greater than zero, got {time}"
                )
            )

        if (not isinstance(one_norm, (int, float))) or one_norm <= 0:
            raise (
                ValueError(
                    f"Expected 'one_norm' to be a positive real number greater than zero, got {one_norm}"
                )
            )

        if poly_approx_precision is not None:
            if (not isinstance(poly_approx_precision, (int, float))) or poly_approx_precision <= 0:
                raise (
                    ValueError(
                        f"Expected 'poly_approx_precision' to be a positive real number greater than zero, got {poly_approx_precision}"
                    )
                )

        self.walk_op = walk_op.resource_rep_from_op()
        self.time = time
        self.one_norm = one_norm
        self.poly_approx_precision = poly_approx_precision
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
                * walk_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`):
                  the quantum walk operator
                * time (float): the simulation time
                * one_norm (float): one norm of the Hamiltonian
                * poly_approx_precision (float): the tolerance for error in the polynomial
                  approximation
        """

        return {
            "walk_op": self.walk_op,
            "time": self.time,
            "one_norm": self.one_norm,
            "poly_approx_precision": self.poly_approx_precision,
        }

    @classmethod
    def resource_rep(
        cls,
        walk_op: ResourceOperator,
        time: float,
        one_norm: float,
        poly_approx_precision: float | None = None,
    ) -> ResourceOperator:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            walk_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): the
                quantum walk operator
            time (float): the simulation time
            one_norm (float): one norm of the Hamiltonian
            poly_approx_precision (float | None): the tolerance for error in the polynomial approximation

        Returns:
            :class:`~.pennylane.estimator.resource_operator.ResourceOperator`: the operator in a compressed representation
        """

        if (not isinstance(time, (int, float))) or time <= 0:
            raise (
                ValueError(
                    f"Expected 'time' to be a positive real number greater than zero, got {time}"
                )
            )

        if (not isinstance(one_norm, (int, float))) or one_norm <= 0:
            raise (
                ValueError(
                    f"Expected 'one_norm' to be a positive real number greater than zero, got {one_norm}"
                )
            )

        if poly_approx_precision is not None:
            if (not isinstance(poly_approx_precision, (int, float))) or poly_approx_precision <= 0:
                raise (
                    ValueError(
                        f"Expected 'poly_approx_precision' to be a positive real number greater than zero, got {poly_approx_precision}"
                    )
                )

        return cls(
            walk_op=walk_op,
            time=time,
            one_norm=one_norm,
            poly_approx_precision=poly_approx_precision,
        )

    @classmethod
    def resource_decomp(
        cls,
        walk_op: ResourceOperator,
        time: float,
        one_norm: float,
        poly_approx_precision: float | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            walk_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): the
                quantum walk operator
            time (float): the simulation time
            one_norm (float): one norm of the Hamiltonian
            poly_approx_precision (float | None): the tolerance for error in the polynomial approximation

        Resources:
            The resources are obtained as described in Theorem 7 and Corollary 8 of
            `Generalized Quantum Signal Processing (2024) <https://arxiv.org/pdf/2308.01501>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        poly_deg = cls.poly_approx(time, one_norm, poly_approx_precision)
        gqsp = GQSP.resource_rep(
            walk_op,
            d_plus=poly_deg,
            d_minus=poly_deg,
            rotation_precision=None,
        )
        return [GateCount(gqsp)]

    @staticmethod
    def poly_approx(time: float, one_norm: float, epsilon: float) -> int:
        r"""Obtain the maximum degree of the polynomial approximation required
        to approximate :math:`e^{(iHt \cos{\theta})}` within some error epsilon.

        Args:
            time (float): the simulation time
            one_norm (float): one norm of the Hamiltonian
            epsilon (float): the tolerance for error in the polynomial approximation

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
    r"""Resource class for Quantum Singular Value Transformation (QSVT).

    This operation uses a :class:`~.estimator.resource_operator.ResourceOperator` :math:`U` that
    block encodes a matrix :math:`A` in its top-left block. This circuit applies a
    polynomial transformation (:math:`Poly^{SV}`) of degree :math:`d` to the singular values of the
    block encoded matrix:

    .. math::

        \begin{align}
             U_{QSVT}(A, \vec{\phi}) &=
             \begin{bmatrix}
                Poly^{SV}(A) & \cdot \\
                \cdot & \cdot
            \end{bmatrix}.
        \end{align}

    When the degree of the polynomial is odd, the QSVT circuit is defined as:

    .. math::

        U_{QSVT} = \tilde{\Pi}_{\phi_1}U\left[\prod^{(d-1)/2}_{k=1}\Pi_{\phi_{2k}}U^\dagger
        \tilde{\Pi}_{\phi_{2k+1}}U\right],


    and when the degree is even,

    .. math::

        U_{QSVT} = \left[\prod^{d/2}_{k=1}\Pi_{\phi_{2k-1}}U^\dagger\tilde{\Pi}_{\phi_{2k}}U\right],

    where :math:`\Pi_{\phi}` and :math:`\tilde{\Pi}_{\phi}` are projector-controlled phase shifts
    (:class:`~.estimator.ops.qubit.parametric_ops_multi_qubit.PCPhase`).

    .. seealso::

        :func:`~.qsvt` and :class:`~.QSVT`.

    Args:
        block_encoding (:class:`~.estimator.resource_operator.ResourceOperator`): the block encoding operator
        encoding_dims (int | tuple(int)): The dimensions of the encoded matrix.
            If an integer is provided, a square matrix is assumed.
        poly_deg (int): the degree of the polynomial transformation being applied
        wires (WiresLike | None): the wires the operation acts on

    Raises:
        ValueError: if ``encoding_dims`` is not a positive integer or a tuple of two positive integers
        ValueError: if ``poly_deg`` is not a positive integer greater than zero
        ValueError: if the ``wires`` provided don't match the number of wires expected by the operator

    Resources:
        The resources are obtained as described in Theorem 4 of `A Grand Unification of Quantum Algorithms
        (2021) <https://arxiv.org/pdf/2105.02859>`_.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> block_encoding = qre.RX(0.1, wires=0)
    >>> encoding_dims = (2, 2)
    >>> poly_deg = 3
    >>> qsvt = qre.QSVT(block_encoding, encoding_dims, poly_deg)
    >>> print(qre.estimate(qsvt))
    --- Resources: ---
     Total wires: 1
       algorithmic wires: 1
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 39
       'T': 39
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
            raise TypeError(
                f"Expected `encoding_dims` to be an integer or tuple of integers. Got {encoding_dims}"
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

        if not all(isinstance(d, int) and d > 0 for d in encoding_dims):
            raise ValueError("Expected elements of `encoding_dims` to be positive integers.")

        self.block_encoding = block_encoding.resource_rep_from_op()
        self.encoding_dims = encoding_dims

        if (not isinstance(poly_deg, int)) or poly_deg <= 0:
            raise ValueError(
                f"'poly_deg' must be a positive integer greater than zero, got {poly_deg}"
            )

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
                * block_encoding (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`):
                  the block encoding operator
                * encoding_dims (int | tuple(int)): The dimensions of the encoded matrix.
                  If an integer is provided, a square matrix is assumed.
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
        block_encoding: ResourceOperator,
        encoding_dims: int,
        poly_deg: int,
    ):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            block_encoding (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`):
                the block encoding operator
            encoding_dims (int | tuple(int)): The dimensions of the encoded matrix.
                If an integer is provided, a square matrix is assumed.
            poly_deg (int): the degree of the polynomial transformation being applied

        Returns:
            :class:`~.pennylane.estimator.resource_operator.ResourceOperator`: the operator in a compressed representation
        """
        if not isinstance(encoding_dims, (int, tuple)):
            raise TypeError(
                f"Expected `encoding_dims` to be an integer or tuple of integers. Got {encoding_dims}"
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

        if not all(isinstance(d, int) and d > 0 for d in encoding_dims):
            raise ValueError("Expected elements of `encoding_dims` to be positive integers.")

        if (not isinstance(poly_deg, int)) or poly_deg <= 0:
            raise ValueError(
                f"'poly_deg' must be a positive integer greater than zero, got {poly_deg}"
            )

        return cls(
            block_encoding=block_encoding,
            encoding_dims=encoding_dims,
            poly_deg=poly_deg,
        )

    @classmethod
    def resource_decomp(
        cls,
        block_encoding: ResourceOperator,
        encoding_dims: int,
        poly_deg: int,
    ):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            block_encoding (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`):
                the block encoding operator
            encoding_dims (int | tuple(int)): The dimensions of the encoded matrix.
                If an integer is provided, a square matrix is assumed.
            poly_deg (int): the degree of the polynomial transformation being applied

        Resources:
            The resources are obtained as described in Theorem 4 of `A Grand Unification of Quantum Algorithms
            (2021) <https://arxiv.org/pdf/2105.02859>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
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

    This template estimates the resources for a QSP circuit of degree :math:`d` (``poly_deg``).
    The circuit uses a single-qubit :class:`~.estimator.resource_operator.ResourceOperator`
    :math:`W(a)` that block encodes a scalar value :math:`a` in its top-left entry.

    The circuit is given as follows in the Z-convention (``convention="Z"``):

    .. math::

        \hat{U}_{QSP} = e^{i\phi_{0}\hat{Z}}\prod^{d}_{k=1}\hat{W}(a)e^{i\phi_{k}\hat{Z}} .

    The circuit can also be expressed in the X-convention (``convention="X"``):

    .. math::

        \hat{U}_{QSP} = e^{i\phi_{0}\hat{X}}\prod^{d}_{k=1}\hat{W}(a)e^{i\phi_{k}\hat{X}} .

    .. seealso::

        :func:`~.qsvt` and :class:`~.QSVT`.

    Args:
        block_encoding (:class:`~.estimator.resource_operator.ResourceOperator`): the block encoding operator
        poly_deg (int): the degree of the polynomial transformation being applied
        convention (str): the basis used for the rotation operators, valid conventions are ``"X"`` or ``"Z"``
        rotation_precision (float | None): The error threshold for the approximate Clifford + T
            decomposition of the single qubit rotation gates used to implement this operation.
        wires (WiresLike | None): the wires the operation acts on

    Raises:
        ValueError: if the block encoding operator acts on more than one qubit
        ValueError: if the convention is not ``"X"`` or ``"Z"``

    Resources:
        The resources are obtained as described in Theorem 1 of `A Grand Unification of Quantum Algorithms
        (2021) <https://arxiv.org/pdf/2105.02859>`_.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> block_encoding = qre.RX(0.1, wires=0)
    >>> poly_deg = 3
    >>> qsp = qre.QSP(block_encoding, poly_deg, convention="Z", rotation_precision=1e-5)
    >>> print(qre.estimate(qsp))
    --- Resources: ---
     Total wires: 1
       algorithmic wires: 1
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 151
       'T': 151
    """

    resource_keys = {"block_encoding", "poly_deg", "convention", "rotation_precision"}

    def __init__(
        self,
        block_encoding: ResourceOperator,
        poly_deg: int,
        convention: str = "Z",
        rotation_precision: float | None = None,
        wires: WiresLike = None,
    ):
        _dequeue(block_encoding)  # remove operator
        if block_encoding.num_wires > 1:
            raise ValueError("The block encoding operator should act on a single qubit!")

        if not (convention in {"Z", "X"}):
            raise ValueError(f"The valid conventions are 'Z' or 'X'. Got {convention}")

        self.block_encoding = block_encoding.resource_rep_from_op()
        self.convention = convention
        self.poly_deg = poly_deg
        self.rotation_precision = rotation_precision

        self.num_wires = 1
        if wires is not None and len(wires) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(wires)}.")

        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * block_encoding (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`):
                  the block encoding operator
                * poly_deg (int): the degree of the polynomial transformation being applied
                * convention (str): the basis used for the rotation operators, valid conventions are ``"X"`` or ``"Z"``
                * rotation_precision (float | None): The error threshold for the approximate Clifford + T
                  decomposition of the single qubit rotation gates used to implement this operation.
        """
        return {
            "block_encoding": self.block_encoding,
            "poly_deg": self.poly_deg,
            "convention": self.convention,
            "rotation_precision": self.rotation_precision,
        }

    @classmethod
    def resource_rep(
        cls,
        block_encoding: ResourceOperator,
        poly_deg: int,
        convention: str = "Z",
        rotation_precision: float | None = None,
    ):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            block_encoding (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`):
                the block encoding operator
            poly_deg (int): the degree of the polynomial transformation being applied
            convention (str):the basis used for the rotation operators, valid conventions are ``"X"`` or ``"Z"``
            rotation_precision (float | None): The error threshold for the approximate Clifford + T
                decomposition of the single qubit rotation gates used to implement this operation.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.ResourceOperator`: the operator in a compressed representation
        """
        if block_encoding.num_wires > 1:
            raise ValueError("The block encoding operator should act on a single qubit!")

        if not (convention in {"Z", "X"}):
            raise ValueError(f"The valid conventions are 'Z' or 'X'. Got {convention}")

        return cls(
            block_encoding=block_encoding,
            poly_deg=poly_deg,
            convention=convention,
            rotation_precision=rotation_precision,
        )

    @classmethod
    def resource_decomp(
        cls,
        block_encoding: ResourceOperator,
        poly_deg: int,
        convention: str = "Z",
        rotation_precision: float | None = None,
    ):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            block_encoding (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`):
                the block encoding operator
            poly_deg (int): the degree of the polynomial transformation being applied
            convention (str): the basis used for the rotation operators, valid conventions are ``"X"`` or ``"Z"``
            rotation_precision (float): The error threshold for the approximate Clifford + T
                decomposition of the single qubit rotation gates used to implement this operation.

        Resources:
            The resources are obtained as described in Theorem 1 of `A Grand Unification of Quantum Algorithms
            (2021) <https://arxiv.org/pdf/2105.02859>`_.

        Raises:
            ValueError: if the convention is not ``"X"`` or ``"Z"``

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if convention == "Z":
            rot_op = RZ.resource_rep(rotation_precision)
        elif convention == "X":
            rot_op = RX.resource_rep(rotation_precision)
        else:
            raise ValueError(f"The valid conventions are 'Z' or 'X'. Got {convention}")

        return [
            GateCount(block_encoding, poly_deg),
            GateCount(rot_op, poly_deg + 1),
        ]
