# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Resource operators for PennyLane arithmetic subroutines."""

from typing import Dict

import pennylane.estimator as qre
from pennylane.estimator.ops.op_math.symbolic import ChangeOpBasis, Controlled, Prod
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.estimator.wires_manager import Allocate, Deallocate
from pennylane.math import ceil_log2
from pennylane.wires import WiresLike

# pylint: disable=arguments-differ,unused-argument,signature-differs


class PhaseAdder(
    ResourceOperator
):  # PhaseAdd_in(k, N): Inplace Classical-Quantum Modular Adder in QFT basis
    r"""Resource class for the PhaseAdder gate.

    This operator performs the modular addition by an integer :math:`k` modulo :math:`mod` in the
    Fourier basis:

    .. math::

        \text{PhaseAdder}(k,mod) |\phi (x) \rangle = |\phi (x+k \; \text{mod} \; mod) \rangle,

    where :math:`|\phi (x) \rangle` represents the :math:`| x \rangle` state in the Fourier basis,

    .. math::

        \text{QFT} |x \rangle = |\phi (x) \rangle.

    Args:
        num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
        mod (int | None): The modulo for performing the addition. If not provided, it will be set to
            its maximum value, :math:`2^{\text{num_x_wires}}`.

    Resources:
        The resources are based on the quantum Fourier transform method presented in
        `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. It is referred too as the :math:`Sum(k)`
        operator within the Quantum Fourier Adder.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> phase_add = qre.PhaseAdder(5, mod=16)
    >>> print(qre.estimate(phase_add))
    --- Resources: ---
     Total wires: 7
       algorithmic wires: 5
       allocated wires: 2
         zero state: 2
         any state: 0
     Total gates : 9.964E+3
       'T': 9.768E+3,
       'CNOT': 170,
       'X': 2,
       'Hadamard': 24
    """

    resource_keys = {"num_x_wires", "mod"}

    def __init__(self, num_x_wires, mod=None, wires=None):
        self.mod = 2**num_x_wires if mod is None else mod
        self.num_x_wires = num_x_wires

        if (self.mod > 2**num_x_wires) or (self.mod < 1):
            raise ValueError(f"mod must take values inbetween (1, {2**num_x_wires}), got {mod}")

        self.num_wires = self.num_x_wires
        if wires is not None and len(wires) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {wires}")

        super().__init__(wires=wires)

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
                * mod (int): The modulo for performing the addition. If not provided, it will be set to
                  its maximum value, :math:`2^{\text{num_x_wires}}`.
        """
        return {"num_x_wires": self.num_x_wires, "mod": self.mod}

    @classmethod
    def resource_rep(cls, num_x_wires, mod=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
            mod (int | None): The modulo for performing the addition. If not provided, it will be set to
                its maximum value, :math:`2^{\text{num_x_wires}}`.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        mod = 2**num_x_wires if mod is None else mod
        params = {"num_x_wires": num_x_wires, "mod": mod}
        return CompressedResourceOp(cls, num_x_wires, params)

    @classmethod
    def resource_decomp(cls, num_x_wires, mod=None) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
            mod (int | None): The modulo for performing the addition. If not provided, it will be set to
                its maximum value, :math:`2^{\text{num_x_wires}}`.

        Resources:
            The resources are based on the quantum Fourier transform method presented in
            `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. It is referred too as the :math:`Sum(k)`
            operator within the Quantum Fourier Adder.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        mod = 2**num_x_wires if mod is None else mod

        phase_shift = qre.PhaseShift.resource_rep()
        if mod == 2**num_x_wires:
            return [GateCount(phase_shift, num_x_wires)]

        gate_lst = []
        gate_lst.append(Allocate(2))
        gate_lst.append(GateCount(phase_shift, num_x_wires + 1))  # Sum(k)
        gate_lst.append(GateCount(phase_shift, num_x_wires + 1))  # Sum(-N)

        qft = qre.QFT.resource_rep(num_x_wires + 1)
        qft_cnot_qft_dag = ChangeOpBasis.resource_rep(
            cmpr_compute_op=qft,
            cmpr_target_op=qre.CNOT.resource_rep(),
            num_wires=num_x_wires + 2,
        )
        gate_lst.append(GateCount(qft_cnot_qft_dag))  # QFT CNOT QFT^dagger

        prod_phase_shifts = Prod.resource_rep(
            cmpr_factors_and_counts=((phase_shift, num_x_wires + 1),),
            num_wires=num_x_wires + 1,
        )
        ctrl_sum_N = Controlled.resource_rep(prod_phase_shifts, 1, 0)
        gate_lst.append(GateCount(ctrl_sum_N))  # Ctrl-Sum(N)

        qft_sum_k = Prod.resource_rep(
            cmpr_factors_and_counts=((qft, 1), (phase_shift, num_x_wires + 1)),
            num_wires=num_x_wires + 1,
        )
        zero_cnot = Controlled.resource_rep(
            base_cmpr_op=qre.X.resource_rep(),
            num_ctrl_wires=1,
            num_zero_ctrl=1,
        )
        change_op_basis_qft_sum_k_cnot = ChangeOpBasis.resource_rep(
            cmpr_compute_op=qre.Adjoint.resource_rep(qft_sum_k),
            cmpr_target_op=zero_cnot,
            num_wires=num_x_wires + 2,
        )

        gate_lst.append(  # Sum(-k) QFT^dagger CNOT QFT Sum(k)
            GateCount(change_op_basis_qft_sum_k_cnot)
        )
        gate_lst.append(Deallocate(2))
        return gate_lst

    @classmethod
    def pow_resource_decomp(cls, pow_z: int, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            Taking arbitrary powers of the phase addition operation can be simplified and
            expressed as a single phase addition. The resources are one total ``PhaseAdder``.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(**target_resource_params))]


class Adder(ResourceOperator):  # Add_in(k, N): Inplace Quantum-Classical Modular Adder
    r"""Resource class for the inplace quantum-classical modular Adder gate.

    This operator performs the modular addition by an integer :math:`k` modulo :math:`mod` in the
    computational basis:

    .. math::

        \text{Adder}(k, mod) |x \rangle = | x+k \; \text{mod} \; mod \rangle.

    Args:
        num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
        mod (int | None): The modulo for performing the addition. If not provided, it will be set to
            its maximum value, :math:`2^{\text{num_x_wires}}`.

    Resources:
        The resources are based on the quantum Fourier transform method presented in section V.B of
        `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. See figure two in the reference for a
        circuit diagram.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> adder = qre.Adder(5, mod=32)
    >>> print(qre.estimate(adder))
    --- Resources: ---
     Total wires: 5
       algorithmic wires: 5
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 2.922E+3
       'T': 2.860E+3,
       'CNOT': 52,
       'Hadamard': 10
    """

    resource_keys = {"num_x_wires", "mod"}

    def __init__(self, num_x_wires, mod=None, wires=None):
        self.mod = 2**num_x_wires if mod is None else mod
        self.num_x_wires = num_x_wires

        if (self.mod > 2**num_x_wires) or (self.mod < 1):
            raise ValueError(f"mod must take values inbetween (1, {2**num_x_wires}), got {mod}")

        self.num_wires = self.num_x_wires
        if wires is not None and len(wires) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {wires}")

        super().__init__(wires=wires)

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
                * mod (int): The modulo for performing the addition. If not provided, it will be set to
                  its maximum value, :math:`2^{\text{num_x_wires}}`.
        """
        return {"num_x_wires": self.num_x_wires, "mod": self.mod}

    @classmethod
    def resource_rep(cls, num_x_wires, mod=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
            mod (int | None): The modulo for performing the addition. If not provided, it will be set to
                its maximum value, :math:`2^{\text{num_x_wires}}`.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        mod = 2**num_x_wires if mod is None else mod
        params = {"num_x_wires": num_x_wires, "mod": mod}
        return CompressedResourceOp(cls, num_x_wires, params)

    @classmethod
    def resource_decomp(cls, num_x_wires, mod=None) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
            mod (int | None): The modulo for performing the addition. If not provided, it will be set to
                its maximum value, :math:`2^{\text{num_x_wires}}`.

        Resources:
            The resources are based on the quantum Fourier transform method presented in section V.B of
            `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. See figure two in the reference for a
            circuit diagram.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        mod = 2**num_x_wires if mod is None else mod

        phase_shift = qre.PhaseShift.resource_rep()
        if mod == 2**num_x_wires:
            qft = qre.QFT.resource_rep(num_x_wires)
            prod_phase_shifts_n = Prod.resource_rep(
                cmpr_factors_and_counts=((phase_shift, num_x_wires),),
                num_wires=num_x_wires,
            )
            change_op_basis = ChangeOpBasis.resource_rep(
                cmpr_compute_op=qft,
                cmpr_target_op=prod_phase_shifts_n,
                num_wires=num_x_wires,
            )

            return [GateCount(change_op_basis)]

        gate_lst = []
        gate_lst.append(Allocate(2))

        qft_n_plus_one = qre.QFT.resource_rep(num_x_wires + 1)
        prod_phase_shifts_n_plus_one = Prod.resource_rep(
            cmpr_factors_and_counts=((phase_shift, num_x_wires + 1),),
            num_wires=num_x_wires + 1,
        )
        ctrl_sum_N = Controlled.resource_rep(
            prod_phase_shifts_n_plus_one,
            1,
            0,
        )

        qft_cnot_qft_dag = ChangeOpBasis.resource_rep(
            cmpr_compute_op=qft_n_plus_one,
            cmpr_target_op=qre.CNOT.resource_rep(),
            num_wires=num_x_wires + 2,
        )

        qft_sum_k = Prod.resource_rep(
            cmpr_factors_and_counts=((qft_n_plus_one, 1), (phase_shift, num_x_wires + 1)),
            num_wires=num_x_wires + 1,
        )
        zero_cnot = Controlled.resource_rep(
            base_cmpr_op=qre.X.resource_rep(),
            num_ctrl_wires=1,
            num_zero_ctrl=1,
        )

        change_op_basis_qft_sum_k_cnot = ChangeOpBasis.resource_rep(
            cmpr_compute_op=qre.Adjoint.resource_rep(qft_sum_k),
            cmpr_target_op=zero_cnot,
            num_wires=num_x_wires + 2,
        )

        target_circ = Prod.resource_rep(
            cmpr_factors_and_counts=(
                (phase_shift, num_x_wires + 1),
                (phase_shift, num_x_wires + 1),
                (qft_cnot_qft_dag, 1),
                (ctrl_sum_N, 1),
                (change_op_basis_qft_sum_k_cnot, 1),
            ),
            num_wires=num_x_wires + 2,
        )

        ultimate_change_op_basis = ChangeOpBasis.resource_rep(
            cmpr_compute_op=qft_n_plus_one,
            cmpr_target_op=target_circ,
            num_wires=num_x_wires + 2,
        )

        gate_lst.append(GateCount(ultimate_change_op_basis))
        gate_lst.append(Deallocate(2))
        return gate_lst

    @classmethod
    def pow_resource_decomp(cls, pow_z: int, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            Taking arbitrary powers of the addition operation can be simplified and expressed as
            a single addition. The resources are one total ``Adder``.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(**target_resource_params))]


class OutAdder(ResourceOperator):  # Add_out(N): Out-of-place Quantum-Quantum Modular Adder
    r"""Resource class for the Out-of-place Quantum Quantum Modular Adder gate.

    This operator performs the modular addition of two integers :math:`x` and :math:`y` modulo
    :math:`mod` in the computational basis:

    .. math::

        \text{OutAdder}(mod) |x \rangle | y \rangle | b \rangle = |x \rangle | y \rangle | b+x+y \; \text{mod} \; mod \rangle,

    Args:
        num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
        num_y_wires (int): the number of wires used to represent, in binary, the value of :math:`y`
        num_output_wires (int): the number of wires used to store the result of the addition
        mod (int | None): The modulo for performing the addition. If not provided, it will be set to
            its maximum value, :math:`2^{\text{num_output_wires}}`.

    Resources:
        The resources are based on the quantum Fourier transform method presented in section V.E of
        `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. See figure five in the reference for a
        circuit diagram.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> out_add = qre.OutAdder(3, 3, 5, mod=32)
    >>> print(qre.estimate(out_add))
    --- Resources: ---
     Total wires: 11
       algorithmic wires: 11
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 6.722E+3
       'T': 6.600E+3,
       'CNOT': 112,
       'Hadamard': 10
    """

    resource_keys = {"num_x_wires", "num_y_wires", "num_output_wires", "mod"}

    def __init__(self, num_x_wires, num_y_wires, num_output_wires, mod=None, wires=None):
        self.mod = 2**num_output_wires if mod is None else mod
        self.num_x_wires = num_x_wires
        self.num_y_wires = num_y_wires
        self.num_output_wires = num_output_wires

        if (self.mod > 2**num_output_wires) or (self.mod < 1):
            raise ValueError(
                f"mod must take values inbetween (1, {2**num_output_wires}), got {mod}"
            )

        self.num_wires = num_x_wires + num_y_wires + num_output_wires
        if wires is not None and len(wires) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {wires}")

        super().__init__(wires=wires)

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
                * num_y_wires (int): the number of wires used to represent, in binary, the value of :math:`y`
                * num_output_wires (int): the number of wires used to store the result of the addition
                * mod (int): The modulo for performing the addition. If not provided, it will be set to
                  its maximum value, :math:`2^{\text{num_x_wires}}`.
        """
        return {
            "num_x_wires": self.num_x_wires,
            "num_y_wires": self.num_y_wires,
            "num_output_wires": self.num_output_wires,
            "mod": self.mod,
        }

    @classmethod
    def resource_rep(
        cls, num_x_wires, num_y_wires, num_output_wires, mod=None
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
            num_y_wires (int): the number of wires used to represent, in binary, the value of :math:`y`
            num_output_wires (int): the number of wires used to store the result of the addition
            mod (int | None): The modulo for performing the addition. If not provided, it will be set to
                its maximum value, :math:`2^{\text{num_output_wires}}`.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        mod = 2**num_output_wires if mod is None else mod
        params = {
            "num_output_wires": num_output_wires,
            "num_x_wires": num_x_wires,
            "num_y_wires": num_y_wires,
            "mod": mod,
        }
        return CompressedResourceOp(cls, num_x_wires + num_y_wires + num_output_wires, params)

    @classmethod
    def resource_decomp(
        cls, num_x_wires, num_y_wires, num_output_wires, mod=None
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
            num_y_wires (int): the number of wires used to represent, in binary, the value of :math:`y`
            num_output_wires (int): the number of wires used to store the result of the addition
            mod (int | None): The modulo for performing the addition. If not provided, it will be set to
                its maximum value, :math:`2^{\text{num_output_wires}}`.

        Resources:
            The resources are based on the quantum Fourier transform method presented in section V.E of
            `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. See figure five in the reference for a
            circuit diagram.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        mod = 2**num_output_wires if mod is None else mod
        phase_add = PhaseAdder.resource_rep(num_output_wires, mod)
        ctrl_adder = qre.Controlled.resource_rep(phase_add, 1, 0)

        num_qft_wires = num_output_wires
        if mod != 2**num_output_wires:
            num_qft_wires += 1

        qft = qre.QFT.resource_rep(num_wires=num_qft_wires)

        return [
            GateCount(
                qre.ChangeOpBasis.resource_rep(
                    cmpr_compute_op=qft,
                    cmpr_target_op=qre.Prod.resource_rep(
                        cmpr_factors_and_counts=((ctrl_adder, num_x_wires + num_y_wires),),
                        num_wires=num_x_wires + num_y_wires + num_output_wires,
                    ),
                    num_wires=num_x_wires + num_y_wires + num_output_wires,
                )
            ),
        ]


class SemiAdder(ResourceOperator):  # Add_in(N): Inplace Quantum-Quantum Adder
    r"""Resource class for the SemiAdder gate.

    Args:
        max_register_size (int): the size of the larger of the two registers being added together
        wires (Sequence[int], None): the wires the operation acts on

    Resources:
        The resources are obtained from figures 1 and 2 in `Gidney (2018)
        <https://quantum-journal.org/papers/q-2018-06-18-74/pdf/>`_.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.SemiAdder`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> semi_add = qre.SemiAdder(max_register_size=4)
    >>> print(qre.estimate(semi_add))
    --- Resources: ---
    Total wires: 11
        algorithmic wires: 8
        allocated wires: 3
        zero state: 3
        any state: 0
    Total gates : 30
    'Toffoli': 3,
    'CNOT': 18,
    'Hadamard': 9
    """

    resource_keys = {"max_register_size"}

    def __init__(self, max_register_size: int, wires: WiresLike = None):
        self.max_register_size = max_register_size
        self.num_wires = 2 * max_register_size
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * max_register_size (int): the size of the larger of the two registers being added together

        """
        return {"max_register_size": self.max_register_size}

    @classmethod
    def resource_rep(cls, max_register_size):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            max_register_size (int): the size of the larger of the two registers being added together

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = 2 * max_register_size
        return CompressedResourceOp(cls, num_wires, {"max_register_size": max_register_size})

    @classmethod
    def resource_decomp(cls, max_register_size: int):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            max_register_size (int): the size of the larger of the two registers being added together

        Resources:
            The resources are obtained from figures 1 and 2 in `Gidney (2018)
            <https://quantum-journal.org/papers/q-2018-06-18-74/pdf/>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        cnot = resource_rep(qre.CNOT)
        if max_register_size == 1:
            return [GateCount(cnot)]

        x = resource_rep(qre.X)
        toff = resource_rep(qre.Toffoli)
        if max_register_size == 2:
            return [GateCount(cnot, 2), GateCount(x, 2), GateCount(toff)]

        cnot_count = (6 * (max_register_size - 2)) + 3
        elbow_count = max_register_size - 1

        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})
        return [
            Allocate(max_register_size - 1),
            GateCount(cnot, cnot_count),
            GateCount(l_elbow, elbow_count),
            GateCount(r_elbow, elbow_count),
            Deallocate(max_register_size - 1),
        ]  # Obtained resource from Fig1 and Fig2 https://quantum-journal.org/papers/q-2018-06-18-74/pdf/

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires: int, num_zero_ctrl: int, target_resource_params: dict
    ):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): dictionary containing the size of the larger of the two registers being added together

        Resources:
            The resources are obtained from figure 4a in `Gidney (2018)
            <https://quantum-journal.org/papers/q-2018-06-18-74/pdf/>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        max_register_size = target_resource_params["max_register_size"]
        if max_register_size <= 2:
            target_resource_params = target_resource_params or {}
            gate_lst = []
            if num_zero_ctrl != 0:
                x = resource_rep(qre.X)
                gate_lst.append(GateCount(x, 2 * num_zero_ctrl))

            decomp = cls.resource_decomp(**target_resource_params)
            for action in decomp:
                if isinstance(action, GateCount):
                    gate = action.gate
                    c_gate = Controlled.resource_rep(
                        gate,
                        num_ctrl_wires,
                        num_zero_ctrl=0,  # we flipped already and added the X gates above
                    )
                    gate_lst.append(GateCount(c_gate, action.count))

            return gate_lst
        gate_lst = []

        if num_ctrl_wires > 1:
            mcx = resource_rep(
                qre.MultiControlledX,
                {
                    "num_ctrl_wires": num_ctrl_wires,
                    "num_zero_ctrl": num_zero_ctrl,
                },
            )
            gate_lst.append(Allocate(1))
            gate_lst.append(GateCount(mcx, 2))

        cnot_count = (7 * (max_register_size - 2)) + 3
        elbow_count = 2 * (max_register_size - 1)

        x = resource_rep(qre.X)
        cnot = resource_rep(qre.CNOT)
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})
        gate_lst.extend(
            [
                Allocate(max_register_size - 1),
                GateCount(cnot, cnot_count),
                GateCount(l_elbow, elbow_count),
                GateCount(r_elbow, elbow_count),
                Deallocate(max_register_size - 1),
            ],
        )

        if num_ctrl_wires > 1:
            gate_lst.append(Deallocate(1))
        elif num_zero_ctrl > 0:
            gate_lst.append(GateCount(x, 2 * num_zero_ctrl))

        return gate_lst


class ClassicalOutMultiplier(
    ResourceOperator
):  # Mult_out(k, N): Out-of-place Quantum-Classical Modular Multiplier
    r"""Resource class for the out-of-place quantum-classical modular multiplication operation.

    This operator performs the modular multiplication by an integer :math:`k` modulo :math:`mod` in
    the computational basis:

    .. math::

        \text{ClassicalOutMultiplier}(k, mod) |a \rangle |b \rangle = |a \rangle | b  + a \cdot k \; \text{mod} \; mod \rangle.

    Args:
        num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
        num_output_wires (int): the number of wires used to store the result of the multiplication
        mod (int | None): The modulo for performing the multiplication. If not provided, it will be set to
            its maximum value, :math:`2^{\text{num_output_wires}}`.

    Resources:
        The resources are based on the quantum Fourier transform method presented in section V.F of
        `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. See figure six in the reference for a
        circuit diagram.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> mult = qre.ClassicalOutMultiplier(5,6,mod=32)
    >>> print(qre.estimate(mult))
    --- Resources: ---
     Total wires: 14
       algorithmic wires: 11
       allocated wires: 3
         zero state: 3
         any state: 0
     Total gates : 1.089E+5
       'Toffoli': 530,
       'T': 1.063E+5,
       'CNOT': 1.780E+3,
       'X': 10,
       'Hadamard': 280
    """

    resource_keys = {"num_x_wires", "num_output_wires", "mod"}

    def __init__(self, num_x_wires, num_output_wires, mod=None, wires=None):
        self.mod = 2**num_output_wires if mod is None else mod
        self.num_x_wires = num_x_wires
        self.num_output_wires = num_output_wires

        if (self.mod > 2**num_output_wires) or (self.mod < 1):
            raise ValueError(
                f"mod must take values inbetween (1, {2**num_output_wires}), got {mod}"
            )

        self.num_wires = num_x_wires + num_output_wires
        if wires is not None and len(wires) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {wires}")

        super().__init__(wires=wires)

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
                * num_output_wires (int): the number of wires used to store the result of the multiplication
                * mod (int | None): The modulo for performing the multiplication. If not provided, it will be
                  set to its maximum value, :math:`2^{\text{num_output_wires}}`.
        """
        return {
            "num_x_wires": self.num_x_wires,
            "num_output_wires": self.num_output_wires,
            "mod": self.mod,
        }

    @classmethod
    def resource_rep(cls, num_x_wires, num_output_wires, mod=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
            num_output_wires (int): the number of wires used to store the result of the multiplication
            mod (int | None): The modulo for performing the multiplication. If not provided, it will be set to
                its maximum value, :math:`2^{\text{num_output_wires}}`.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        mod = 2**num_output_wires if mod is None else mod
        params = {"mod": mod, "num_x_wires": num_x_wires, "num_output_wires": num_output_wires}
        return CompressedResourceOp(cls, num_x_wires + num_output_wires, params)

    @classmethod
    def resource_decomp(cls, num_x_wires, num_output_wires, mod=None) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
            num_output_wires (int): the number of wires used to store the result of the multiplication
            mod (int | None): The modulo for performing the multiplication. If not provided, it will be set to
                its maximum value, :math:`2^{\text{num_output_wires}}`.

        Resources:
            The resources are based on the quantum Fourier transform method presented in section V.F of
            `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. See figure six in the reference for a
            circuit diagram.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        mod = 2**num_output_wires if mod is None else mod
        base_adder = Adder.resource_rep(num_output_wires, mod)
        ctrl_adder = qre.Controlled.resource_rep(
            base_cmpr_op=base_adder,
            num_ctrl_wires=1,
            num_zero_ctrl=0,
        )
        return [GateCount(ctrl_adder, num_x_wires)]

    @classmethod
    def pow_resource_decomp(cls, pow_z: int, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            Taking arbitrary powers of the multiplication operation can be simplified and expressed as
            a single multiplication. The resources are one total ``ClassicalOutMultiplier``.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(**target_resource_params))]


class Multiplier(ResourceOperator):  # Mult_in(k, N): Inplace Quantum-Classical Modular Multiplier
    r"""Resource class for the inplace quantum-classical modular multiplication operation.

    This operator performs the modular multiplication by an integer :math:`k` modulo :math:`mod` in
    the computational basis:

    .. math::

        \text{Multiplier}(k,mod) |x \rangle = | x \cdot k \; \text{mod} \; mod \rangle.

    Args:
        num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
        mod (int | None): The modulo for performing the multiplication. If not provided, it will be set to
            its maximum value, :math:`2^{\text{num_x_wires}}`.

    Resources:
        The resources are based on the quantum Fourier transform method presented in section V.G of
        `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. See figure seven in the reference for a
        circuit diagram.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> mult = qre.Multiplier(5, mod=32)
    >>> print(qre.estimate(mult))
    --- Resources: ---
     Total wires: 10
       algorithmic wires: 5
       allocated wires: 5
         zero state: 5
         any state: 0
     Total gates : 3.374E+4
       'T': 3.300E+4,
       'CNOT': 635,
       'Hadamard': 100
    """

    resource_keys = {"num_x_wires", "mod"}

    def __init__(self, num_x_wires, mod=None, wires=None):
        self.mod = 2**num_x_wires if mod is None else mod
        self.num_x_wires = num_x_wires

        if (self.mod > 2**num_x_wires) or (self.mod < 1):
            raise ValueError(f"mod must take values inbetween (1, {2**num_x_wires}), got {mod}")

        self.num_wires = num_x_wires
        if wires is not None and len(wires) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {wires}")

        super().__init__(wires=wires)

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
                * mod (int | None): The modulo for performing the multiplication. If not provided, it will
                  be set to its maximum value, :math:`2^{\text{num_x_wires}}`.
        """
        return {"mod": self.mod, "num_x_wires": self.num_x_wires}

    @classmethod
    def resource_rep(cls, num_x_wires, mod=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
            mod (int | None): The modulo for performing the multiplication. If not provided, it will be set to
                its maximum value, :math:`2^{\text{num_x_wires}}`.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        mod = 2**num_x_wires if mod is None else mod
        params = {"mod": mod, "num_x_wires": num_x_wires}
        return CompressedResourceOp(cls, num_x_wires, params)

    @classmethod
    def resource_decomp(cls, num_x_wires, mod=None) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
            mod (int | None): The modulo for performing the multiplication. If not provided, it will be set to
                its maximum value, :math:`2^{\text{num_x_wires}}`.

        Resources:
            The resources are based on the quantum Fourier transform method presented in section V.G of
            `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. See figure seven in the reference for a
            circuit diagram.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        mod = 2**num_x_wires if mod is None else mod
        gate_lst = []
        gate_lst.append(qre.Allocate(num_x_wires))

        swap = qre.SWAP.resource_rep()
        mult = ClassicalOutMultiplier.resource_rep(num_x_wires, num_x_wires, mod)

        gate_lst.append(
            GateCount(
                qre.ChangeOpBasis.resource_rep(
                    cmpr_compute_op=mult,
                    cmpr_target_op=qre.Prod.resource_rep(
                        cmpr_factors_and_counts=((swap, num_x_wires),),
                        num_wires=2 * num_x_wires,
                    ),
                    num_wires=2 * num_x_wires,
                ),
            )
        )
        gate_lst.append(qre.Deallocate(num_x_wires))
        return gate_lst

    @classmethod
    def pow_resource_decomp(cls, pow_z: int, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            Taking arbitrary powers of the multiplication operation can be simplified and expressed as
            a single multiplication. The resources are one total ``Multiplier``.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(**target_resource_params))]


class OutMultiplier(ResourceOperator):  # Out-of-place Quantum-Quantum Multiplication
    r"""Resource class for the OutMultiplier gate.

    Args:
        a_num_wires (int): the size of the first input register
        b_num_wires (int): the size of the second input register
        wires (Sequence[int], None): the wires the operation acts on

    Resources:
        The resources are obtained from appendix G, lemma 10 in `PRX Quantum, 2, 040332 (2021)
        <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332>`_.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.OutMultiplier`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> out_mul = qre.OutMultiplier(4, 4)
    >>> print(qre.estimate(out_mul))
    --- Resources: ---
    Total wires: 16
        algorithmic wires: 16
        allocated wires: 0
        zero state: 0
        any state: 0
    Total gates : 70
    'Toffoli': 14,
    'CNOT': 14,
    'Hadamard': 42
    """

    resource_keys = {"a_num_wires", "b_num_wires"}

    def __init__(self, a_num_wires: int, b_num_wires: int, wires: WiresLike = None) -> None:
        self.num_wires = a_num_wires + b_num_wires + 2 * max((a_num_wires, b_num_wires))
        self.a_num_wires = a_num_wires
        self.b_num_wires = b_num_wires
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * a_num_wires (int): the size of the first input register
                * b_num_wires (int): the size of the second input register
        """
        return {"a_num_wires": self.a_num_wires, "b_num_wires": self.b_num_wires}

    @classmethod
    def resource_rep(cls, a_num_wires, b_num_wires) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            a_num_wires (int): the size of the first input register
            b_num_wires (int): the size of the second input register

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = a_num_wires + b_num_wires + 2 * max((a_num_wires, b_num_wires))
        return CompressedResourceOp(
            cls, num_wires, {"a_num_wires": a_num_wires, "b_num_wires": b_num_wires}
        )

    @classmethod
    def resource_decomp(cls, a_num_wires, b_num_wires) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            a_num_wires (int): the size of the first input register
            b_num_wires (int): the size of the second input register

        Resources:
            The resources are obtained from appendix G, lemma 10 in `PRX Quantum, 2, 040332 (2021)
            <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        l = max(a_num_wires, b_num_wires)

        toff = resource_rep(qre.Toffoli)
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})

        toff_count = 2 * a_num_wires * b_num_wires - l
        elbow_count = toff_count // 2
        toff_count = toff_count - (elbow_count * 2)

        gate_lst = [
            GateCount(l_elbow, elbow_count),
            GateCount(r_elbow, elbow_count),
        ]

        if toff_count:
            gate_lst.append(GateCount(toff))
        return gate_lst


class ModExp(ResourceOperator):  # ModExp(a, N): Out-of-place Modular Exponentiation
    r"""Resource class for the out-of-place modular exponentiation operation.

    This operator performs the modular exponentiation of the integer :math:`base` to the power
    :math:`x` modulo :math:`mod` in the computational basis:

    .. math::

        \text{ModExp}(base,mod) |x \rangle |b \rangle = |x \rangle |b \cdot base^x \; \text{mod} \; mod \rangle,

    Args:
        num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
        num_output_wires (int): the number of wires used to store the result of the exponentiation
        mod (int | None): The modulo for performing the exponentiation. If not provided, it will be set to
            its maximum value, :math:`2^{\text{num_output_wires}}`.

    Resources:
        The resources are based on the quantum Fourier transform method presented in section V.J of
        `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. See figure nine in the reference for a
        circuit diagram.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> modexp = qre.ModExp(5, 10, mod=32)
    >>> print(qre.estimate(modexp))
    --- Resources: ---
     Total wires: 29
       algorithmic wires: 15
       allocated wires: 14
         zero state: 14
         any state: 0
     Total gates : 5.662E+6
       'Toffoli': 9.255E+4,
       'T': 5.372E+6,
       'CNOT': 1.096E+5,
       'X': 100,
       'Hadamard': 8.720E+4
    """

    resource_keys = {"num_x_wires", "num_output_wires", "mod"}

    def __init__(self, num_x_wires, num_output_wires, mod=None, wires=None):
        self.mod = 2**num_output_wires if mod is None else mod
        self.num_x_wires = num_x_wires
        self.num_output_wires = num_output_wires

        if (self.mod > 2**num_output_wires) or (self.mod < 1):
            raise ValueError(
                f"mod must take values inbetween (1, {2**num_output_wires}), got {mod}"
            )

        self.num_wires = num_x_wires + num_output_wires
        if wires is not None and len(wires) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {wires}")

        super().__init__(wires=wires)

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
                * num_output_wires (int): the number of wires used to store the result of the exponentiation
                * mod (int | None): The modulo for performing the exponentiation. If not provided, it will be
                  set to its maximum value, :math:`2^{\text{num_output_wires}}`.
        """
        return {
            "num_x_wires": self.num_x_wires,
            "num_output_wires": self.num_output_wires,
            "mod": self.mod,
        }

    @classmethod
    def resource_rep(cls, num_x_wires, num_output_wires, mod=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
            num_output_wires (int): the number of wires used to store the result of the exponentiation
            mod (int | None): The modulo for performing the exponentiation. If not provided, it will be set to
                its maximum value, :math:`2^{\text{num_output_wires}}`.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        mod = 2**num_output_wires if mod is None else mod
        params = {"num_x_wires": num_x_wires, "num_output_wires": num_output_wires, "mod": mod}
        return CompressedResourceOp(cls, num_x_wires + num_output_wires, params)

    @classmethod
    def resource_decomp(cls, num_x_wires, num_output_wires, mod=None) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
            num_output_wires (int): the number of wires used to store the result of the exponentiation
            mod (int | None): The modulo for performing the exponentiation. If not provided, it will be set to
                its maximum value, :math:`2^{\text{num_output_wires}}`.

        Resources:
            The resources are based on the quantum Fourier transform method presented in section V.J of
            `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. See figure nine in the reference for a
            circuit diagram.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        mod = 2**num_output_wires if mod is None else mod
        mult_in = Multiplier.resource_rep(num_output_wires, mod)
        ctrl_mult = qre.Controlled.resource_rep(mult_in, 1, 0)
        return [GateCount(ctrl_mult, num_x_wires)]


class OutOfPlaceSquare(ResourceOperator):  # Out-of-place Square
    r"""Resource class for the OutofPlaceSquare gate.

    Args:
        register_size (int): the size of the input register
        wires (Sequence[int], None): the wires the operation acts on

    Resources:
        The resources are obtained from appendix G, lemma 7 in `PRX Quantum, 2, 040332 (2021)
        <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332>`_. Specifically,
        the resources are given as :math:`(n - 1)^2` Toffoli gates, and :math:`n` CNOT gates, where
        :math:`n` is the size of the input register.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> out_square = qre.OutOfPlaceSquare(register_size=3)
    >>> print(qre.estimate(out_square))
    --- Resources: ---
    Total wires: 9
        algorithmic wires: 9
        allocated wires: 0
        zero state: 0
        any state: 0
    Total gates : 7
    'Toffoli': 4,
    'CNOT': 3
    """

    resource_keys = {"register_size"}

    def __init__(self, register_size: int, wires: WiresLike = None):
        self.register_size = register_size
        self.num_wires = 3 * register_size
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * register_size (int): the size of the input register
        """
        return {"register_size": self.register_size}

    @classmethod
    def resource_rep(cls, register_size: int):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            register_size (int): the size of the input register

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = 3 * register_size
        return CompressedResourceOp(cls, num_wires, {"register_size": register_size})

    @classmethod
    def resource_decomp(cls, register_size):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            register_size (int): the size of the input register

        Resources:
            The resources are obtained from appendix G, lemma 7 in `PRX Quantum, 2, 040332 (2021)
            <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332>`_. Specifically,
            the resources are given as :math:`(n - 1)^2` Toffoli gates, and :math:`n` CNOT gates.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_lst = []

        gate_lst.append(GateCount(resource_rep(qre.Toffoli), (register_size - 1) ** 2))
        gate_lst.append(GateCount(resource_rep(qre.CNOT), register_size))

        return gate_lst
