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

import pennylane.labs.estimator_beta as qre
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.wires import WiresLike

# pylint: disable=arguments-differ,unused-argument,signature-differs,too-many-arguments


class LabsPhaseAdder(
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
        `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. It is referred to as the :math:`Sum(k)`
        operator within the Quantum Fourier Adder.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.labs.estimator_beta as qre
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

    def __init__(self, num_x_wires: int, mod: int | None = None, wires: WiresLike | None = None):
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
    def resource_rep(cls, num_x_wires: int, mod: int | None = None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

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
    def resource_decomp(cls, num_x_wires: int, mod: int | None = None) -> list[GateCount]:
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
        gate_lst.append(qre.Allocate(2))
        gate_lst.append(GateCount(phase_shift, num_x_wires + 1))  # Sum(k)
        gate_lst.append(GateCount(phase_shift, num_x_wires + 1))  # Sum(-N)

        qft = qre.QFT.resource_rep(num_x_wires + 1)
        qft_cnot_qft_dag = qre.ChangeOpBasis.resource_rep(
            cmpr_compute_op=qft,
            cmpr_target_op=qre.CNOT.resource_rep(),
            num_wires=num_x_wires + 2,
        )
        gate_lst.append(GateCount(qft_cnot_qft_dag))  # QFT CNOT QFT^dagger

        prod_phase_shifts = qre.Prod.resource_rep(
            cmpr_factors_and_counts=((phase_shift, num_x_wires + 1),),
            num_wires=num_x_wires + 1,
        )
        ctrl_sum_N = qre.Controlled.resource_rep(prod_phase_shifts, 1, 0)
        gate_lst.append(GateCount(ctrl_sum_N))  # Ctrl-Sum(N)

        qft_sum_k = qre.Prod.resource_rep(
            cmpr_factors_and_counts=((qft, 1), (phase_shift, num_x_wires + 1)),
            num_wires=num_x_wires + 1,
        )
        zero_cnot = qre.Controlled.resource_rep(
            base_cmpr_op=qre.X.resource_rep(),
            num_ctrl_wires=1,
            num_zero_ctrl=1,
        )
        change_op_basis_qft_sum_k_cnot = qre.ChangeOpBasis.resource_rep(
            cmpr_compute_op=qre.Adjoint.resource_rep(qft_sum_k),
            cmpr_target_op=zero_cnot,
            num_wires=num_x_wires + 2,
        )

        gate_lst.append(  # Sum(-k) QFT^dagger CNOT QFT Sum(k)
            GateCount(change_op_basis_qft_sum_k_cnot)
        )
        gate_lst.append(qre.Deallocate(2))
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


class LabsAdder(ResourceOperator):  # Add_in(k, N): Inplace Quantum-Classical Modular Adder
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

    >>> import pennylane.labs.estimator_beta as qre
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

    def __init__(self, num_x_wires: int, mod: int | None = None, wires: WiresLike | None = None):
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
    def resource_rep(cls, num_x_wires: int, mod: int | None = None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

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
    def resource_decomp(cls, num_x_wires: int, mod: int | None = None) -> list[GateCount]:
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
            prod_phase_shifts_n = qre.Prod.resource_rep(
                cmpr_factors_and_counts=((phase_shift, num_x_wires),),
                num_wires=num_x_wires,
            )
            change_op_basis = qre.ChangeOpBasis.resource_rep(
                cmpr_compute_op=qft,
                cmpr_target_op=prod_phase_shifts_n,
                num_wires=num_x_wires,
            )

            return [GateCount(change_op_basis)]

        gate_lst = []
        gate_lst.append(qre.Allocate(2))

        qft_n_plus_one = qre.QFT.resource_rep(num_x_wires + 1)
        prod_phase_shifts_n_plus_one = qre.Prod.resource_rep(
            cmpr_factors_and_counts=((phase_shift, num_x_wires + 1),),
            num_wires=num_x_wires + 1,
        )
        ctrl_sum_N = qre.Controlled.resource_rep(
            prod_phase_shifts_n_plus_one,
            1,
            0,
        )

        qft_cnot_qft_dag = qre.ChangeOpBasis.resource_rep(
            cmpr_compute_op=qft_n_plus_one,
            cmpr_target_op=qre.CNOT.resource_rep(),
            num_wires=num_x_wires + 2,
        )

        qft_sum_k = qre.Prod.resource_rep(
            cmpr_factors_and_counts=((qft_n_plus_one, 1), (phase_shift, num_x_wires + 1)),
            num_wires=num_x_wires + 1,
        )
        zero_cnot = qre.Controlled.resource_rep(
            base_cmpr_op=qre.X.resource_rep(),
            num_ctrl_wires=1,
            num_zero_ctrl=1,
        )

        change_op_basis_qft_sum_k_cnot = qre.ChangeOpBasis.resource_rep(
            cmpr_compute_op=qre.Adjoint.resource_rep(qft_sum_k),
            cmpr_target_op=zero_cnot,
            num_wires=num_x_wires + 2,
        )

        target_circ = qre.Prod.resource_rep(
            cmpr_factors_and_counts=(
                (phase_shift, num_x_wires + 1),
                (phase_shift, num_x_wires + 1),
                (qft_cnot_qft_dag, 1),
                (ctrl_sum_N, 1),
                (change_op_basis_qft_sum_k_cnot, 1),
            ),
            num_wires=num_x_wires + 2,
        )

        ultimate_change_op_basis = qre.ChangeOpBasis.resource_rep(
            cmpr_compute_op=qft_n_plus_one,
            cmpr_target_op=target_circ,
            num_wires=num_x_wires + 2,
        )

        gate_lst.append(GateCount(ultimate_change_op_basis))
        gate_lst.append(qre.Deallocate(2))
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


class LabsOutAdder(ResourceOperator):  # Add_out(N): Out-of-place Quantum-Quantum Modular Adder
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
        wires (WiresLike | None): the wires the operator acts on

    Resources:
        The resources are based on the quantum Fourier transform method presented in section V.E of
        `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. See figure five in the reference for a
        circuit diagram.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.labs.estimator_beta as qre
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

    def __init__(
        self,
        num_x_wires: int,
        num_y_wires: int,
        num_output_wires: int,
        mod: int | None = None,
        wires: WiresLike | None = None,
    ):
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
        cls, num_x_wires: int, num_y_wires: int, num_output_wires: int, mod: int | None = None
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

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
        cls, num_x_wires: int, num_y_wires: int, num_output_wires: int, mod: int | None = None
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
        phase_add = LabsPhaseAdder.resource_rep(num_output_wires, mod)
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
        wires (WiresLike | None): the wires the operator acts on

    Resources:
        The resources are based on the quantum Fourier transform method presented in section V.F of
        `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. See figure six in the reference for a
        circuit diagram.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.labs.estimator_beta as qre
    >>> mult = qre.ClassicalOutMultiplier(5, 6, mod=32)
    >>> print(qre.estimate(mult))
    --- Resources: ---
     Total wires: 14
       algorithmic wires: 11
       allocated wires: 3
         zero state: 3
         any state: 0
     Total gates : 1.021E+5
       'Toffoli': 80,
       'T': 1.001E+5,
       'CNOT': 1.740E+3,
       'X': 10,
       'Hadamard': 210
    """

    resource_keys = {"num_x_wires", "num_output_wires", "mod"}

    def __init__(
        self,
        num_x_wires: int,
        num_output_wires: int,
        mod: int | None = None,
        wires: WiresLike | None = None,
    ):
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
    def resource_rep(
        cls, num_x_wires: int, num_output_wires: int, mod: int | None = None
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

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
    def resource_decomp(
        cls, num_x_wires: int, num_output_wires: int, mod: int | None = None
    ) -> list[GateCount]:
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
        base_adder = LabsAdder.resource_rep(num_output_wires, mod)
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


class LabsMultiplier(
    ResourceOperator
):  # Mult_in(k, N): Inplace Quantum-Classical Modular Multiplier
    r"""Resource class for the inplace quantum-classical modular multiplication operation.

    This operator performs the modular multiplication by an integer :math:`k` modulo :math:`mod` in
    the computational basis:

    .. math::

        \text{Multiplier}(k,mod) |x \rangle = | x \cdot k \; \text{mod} \; mod \rangle.

    Args:
        num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
        mod (int | None): The modulo for performing the multiplication. If not provided, it will be set to
            its maximum value, :math:`2^{\text{num_x_wires}}`.
        wires (WiresLike | None): the wires the operator acts on

    Resources:
        The resources are based on the quantum Fourier transform method presented in Section V.G of
        `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. See Figure 7 in the reference for a
        circuit diagram.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.labs.estimator_beta as qre
    >>> mult = qre.LabsMultiplier(5, mod=32)
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

    def __init__(self, num_x_wires: int, mod: int | None = None, wires: WiresLike | None = None):
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
    def resource_rep(cls, num_x_wires: int, mod: int | None = None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

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
    def resource_decomp(cls, num_x_wires: int, mod: int | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_x_wires (int): the number of wires used to represent, in binary, the value of :math:`x`
            mod (int | None): The modulo for performing the multiplication. If not provided, it will be set to
                its maximum value, :math:`2^{\text{num_x_wires}}`.

        Resources:
            The resources are based on the quantum Fourier transform method presented in Section V.G of
            `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. See Figure 7 in the reference for a
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


class LabsModExp(ResourceOperator):  # ModExp(a, N): Out-of-place Modular Exponentiation
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
        wires (WiresLike | None): the wires the operator acts on

    Resources:
        The resources are based on the quantum Fourier transform method presented in section V.J of
        `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. See figure nine in the reference for a
        circuit diagram.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.labs.estimator_beta as qre
    >>> modexp = qre.LabsModExp(5, 10, mod=32)
    >>> print(qre.estimate(modexp))
    --- Resources: ---
     Total wires: 28
       algorithmic wires: 15
       allocated wires: 13
         zero state: 13
         any state: 0
     Total gates : 4.979E+6
       'Toffoli': 2.450E+3,
       'T': 4.888E+6,
       'CNOT': 8.170E+4,
       'X': 200,
       'Hadamard': 6.600E+3
    """

    resource_keys = {"num_x_wires", "num_output_wires", "mod"}

    def __init__(
        self,
        num_x_wires: int,
        num_output_wires: int,
        mod: int | None = None,
        wires: WiresLike | None = None,
    ):
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
    def resource_rep(
        cls, num_x_wires: int, num_output_wires: int, mod: int | None = None
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

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
    def resource_decomp(
        cls, num_x_wires: int, num_output_wires: int, mod: int | None = None
    ) -> list[GateCount]:
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
        mult_in = LabsMultiplier.resource_rep(num_output_wires, mod)
        ctrl_mult = qre.Controlled.resource_rep(mult_in, 1, 0)
        return [GateCount(ctrl_mult, num_x_wires)]
