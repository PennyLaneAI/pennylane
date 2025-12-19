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
from pennylane.estimator.ops.qubit.parametric_ops_single_qubit import Rot
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    _dequeue,
)
from pennylane.wires import Wires, WiresLike

# pylint: disable=arguments-differ,super-init-not-called, signature-differs, too-many-arguments


class GQSP(ResourceOperator):
    r"""Resource class for the Generalized Quantum Signal Processing (GQSP) algorithm.

    The implementation is based on Theorem 6 of `Generalized Quantum Signal Processing (2024)
    <https://arxiv.org/pdf/2308.01501>`_. Given the block-encoded operator ``signal_operator``
    (:math:`\hat{U}`), the maximum positive polynomial degree ``poly_deg`` (:math:`d^{+}`) and
    the maximum negative polynomial degree ``neg_poly_deg`` (:math:`d^{-}`), the ``GQSP`` operator
    is defined according to:

    .. math::

        GQSP = \left( \prod_{j=1}^{d^{-}} R(\theta_{j}, \phi_{j}, 0) \hat{A}^{\prime} \right) 
        \left( \prod_{j=1}^{d^{+}} R(\theta_{j + d^{-}}, \phi_{j + d^{-}}, 0) \hat{A} \right) R(\theta_0, \phi_0, \lambda)

    Where :math:`R` is the general rotation operator 
    :class:`~.estimator.ops.qubit.parametric_ops_single_qubit.Rot`, and :math:`\vec{\phi}`, 
    :math:`vec{\theta}` and :math:`\lambda` are the rotation angles that generate the polynomial transformation.
    Additionally, :math:`\hat{A}` and :math:`\hat{A}^{\prime}` are given by:

    .. math::

        \begin{align}
            \hat{A} &= \ket{0}\bra{0}\otimes\hat{U} + \ket{1}\bra{1}\otimes\mathbf{I}, \\
            \hat{A}^{\prime} &= \ket{0}\bra{0}\otimes\mathbf{I} + \ket{1}\bra{1}\otimes\hat{U}^{\dagger}, \\ \\
        \end{align}

    Args:
        signal_operator (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): the
            signal operator which encodes the target Hamiltonian
        poly_deg (int): the maximum positive degree :math:`d^{+}` of the polynomial transformation
        neg_poly_deg (int): the maximum negative degree :math:`d^{-}` of the polynomial transformation
        rotation_precision (float | None): the precision with which the general rotation gates are applied
        wires (WiresLike | None): The wires the operation acts on. This includes both the wires of the
            signal operator and the control wire required for block-encoding.

    Resources:
        The resources are obtained as described in Theorem 6 of `Generalized Quantum Signal 
        Processing (2024) <https://arxiv.org/pdf/2308.01501>`_. Specifically, the resources are given
        by ``poly_deg`` instances of :math:`\hat{A}`, ``neg_poly_deg`` instances of
        :math:`\hat{A^{\prime}}`, and ``poly_deg + neg_poly_deg + 1`` instances of the general
        ``Rot`` gate.

    Raises:
        ValueError: ``poly_deg`` must be a positive integer greater than zero
        ValueError: ``neg_poly_deg`` must be a positive integer
        ValueError: ``rotation_precision`` must be a positive real number greater than zero
        ValueError: if the wires provided don't match the number of wires expected by the operator              

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> signal_op = qre.RX(0.1, wires=0)
    >>> poly_deg = 5
    >>> neg_poly_deg = 3
    >>> gqsp = qre.GQSP(signal_op, poly_deg, neg_poly_deg)
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

    resource_keys = {"cmpr_signal_op", "poly_deg", "neg_poly_deg", "rotation_precision"}

    def __init__(
        self,
        signal_operator: ResourceOperator,
        poly_deg: int,
        neg_poly_deg: int = 0,
        rotation_precision: float | None = None,
        wires: WiresLike = None,
    ):
        _dequeue(signal_operator)  # remove operator
        self.queue()

        if (not isinstance(poly_deg, int)) or poly_deg <= 0:
            raise ValueError(
                f"'poly_deg' must be a positive integer greater than zero, got {poly_deg}"
            )

        if (not isinstance(neg_poly_deg, int)) or neg_poly_deg < 0:
            raise ValueError(f"'neg_poly_deg' must be a positive integer, got {neg_poly_deg}")

        if rotation_precision is not None and rotation_precision <= 0:
            raise ValueError(
                f"Expected 'rotation_precision' to be a positive real number greater than zero, got {rotation_precision}"
            )

        self.poly_deg = poly_deg
        self.neg_poly_deg = neg_poly_deg
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
                * cmpr_signal_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                  the compressed representation of signal operator which encodes the target Hamiltonian
                * poly_deg (int): the maximum positive degree :math:`d^{+}` of the polynomial transformation
                * neg_poly_deg (int): the maximum negative degree :math:`d^{-}` of the polynomial
                  transformation
                * rotation_precision (float | None): the precision with which the general
                  rotation gates are applied
        """

        return {
            "cmpr_signal_op": self.cmpr_signal_op,
            "poly_deg": self.poly_deg,
            "neg_poly_deg": self.neg_poly_deg,
            "rotation_precision": self.rotation_precision,
        }

    @classmethod
    def resource_rep(
        cls,
        cmpr_signal_op: CompressedResourceOp,
        poly_deg: int,
        neg_poly_deg: int,
        rotation_precision: float | None,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            cmpr_signal_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                the compressed representation of signal operator which encodes the target Hamiltonian
            poly_deg (int): the maximum positive degree :math:`d^{+}` of the polynomial transformation
            neg_poly_deg (int): the maximum negative degree :math:`d^{-}` of the polynomial transformation
            rotation_precision (float | None): the precision with which the general rotation gates are applied

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = cmpr_signal_op.num_wires + 1  # add control wire
        params = {
            "cmpr_signal_op": cmpr_signal_op,
            "poly_deg": poly_deg,
            "neg_poly_deg": neg_poly_deg,
            "rotation_precision": rotation_precision,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        cmpr_signal_op: CompressedResourceOp,
        poly_deg: int,
        neg_poly_deg: int,
        rotation_precision: float | None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            cmpr_signal_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                the compressed representation of signal operator which encodes the target Hamiltonian
            poly_deg (int): the maximum positive degree :math:`d^{+}` of the polynomial transformation
            neg_poly_deg (int): the maximum negative degree :math:`d^{-}` of the polynomial transformation
            rotation_precision (float | None): the precision with which the general rotation gates are applied

        Resources:
            The resources are obtained as described in Theorem 6 of
            `Generalized Quantum Signal Processing (2024) <https://arxiv.org/pdf/2308.01501>`_.
            Specifically, the resources are given by ``poly_deg`` instances of :math:`\hat{A}`,
            ``neg_poly_deg`` instances of :math:`\hat{A^{\prime}}`, and ``poly_deg + neg_poly_deg + 1``
            instances of the general ``Rot`` gate.

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

        if neg_poly_deg == 0:
            return [GateCount(rot, poly_deg + 1), GateCount(ctrl_cmpr_signal_op, poly_deg)]

        return [
            GateCount(rot, poly_deg + neg_poly_deg + 1),
            GateCount(ctrl_cmpr_signal_op, poly_deg),
            GateCount(ctrl_adj_cmpr_signal_op, neg_poly_deg),
        ]


class GQSPTimeEvolution(ResourceOperator):
    r"""Resource class for performing Hamiltonian simulation using GQSP.

    Args:
        walk_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): the quantum walk operator
        time (float): the simulation time
        one_norm (float): one norm of the Hamiltonian
        poly_approx_precision (float): the tolerance for error in the polynomial approximation of :math:`e^{it\cos{(\theta)}}`
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
        poly_approx_precision: float,
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
                * walk_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                  the quantum walk operator
                * time (float): the simulation time
                * one_norm (float): one norm of the Hamiltonian
                * poly_approx_precision (float): the tolerance for error in the polynomial
                  approximation of :math:`e^{it\cos{theta}}`
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
        walk_op: CompressedResourceOp,
        time: float,
        one_norm: float,
        poly_approx_precision: float,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            walk_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): the
                quantum walk operator
            time (float): the simulation time
            one_norm (float): one norm of the Hamiltonian
            poly_approx_precision (float): the tolerance for error in the polynomial approximation of
                :math:`e^{it\cos{\theta}}`

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = walk_op.num_wires + 1
        params = {
            "walk_op": walk_op,
            "time": time,
            "one_norm": one_norm,
            "poly_approx_precision": poly_approx_precision,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        walk_op: CompressedResourceOp,
        time: float,
        one_norm: float,
        poly_approx_precision: float,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            walk_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): the
                quantum walk operator
            time (float): the simulation time
            one_norm (float): one norm of the Hamiltonian
            poly_approx_precision (float): the tolerance for error in the polynomial approximation of
                :math:`e^{it\cos{\theta}}`
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
            poly_deg=poly_deg,
            neg_poly_deg=poly_deg,
            rotation_precision=None,
        )
        return [GateCount(gqsp)]

    @staticmethod
    def poly_approx(time: float, one_norm: float, epsilon: float) -> int:
        r"""Obtain the maximum degree of the polynomial approximation required
        to approximate :math:`e^(iHt * \cos{\theta})` within error epsilon.

        Args:
            time (float): the simulation time
            one_norm (float): one norm of the Hamiltonian
            epsilon (float): the tolerance for error in the polynomial approximation of :math:`e^{it\cos{\theta}}`

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
