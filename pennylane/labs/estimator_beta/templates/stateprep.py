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
r"""Resource operators for state preparation templates."""
import math

import pennylane.labs.estimator_beta as qre
from pennylane.labs.estimator_beta.compact_hamiltonian import FirstQuantizedHamiltonian
from pennylane.estimator.resource_operator import CompressedResourceOp, GateCount, ResourceOperator, resource_rep
from pennylane.labs.estimator_beta.wires_manager import Allocate, Deallocate
from pennylane.math.utils import ceil_log2
from pennylane.wires import WiresLike

class PrepFirstQuantization(ResourceOperator):
    r"""Resource class for preparing the state for first quantization algorithms.

    This operator customizes the Prepare circuit based on the structure of the first quantizated Hamiltonian.

    Args:
        fq_ham (:class:`~pennylane.labs.estimator_beta.FirstQuantizedHamiltonian`): a first quantized
            Hamiltonian for which the state is being prepared
        coordinates_precision (int): The number of bits used to represent the precision for loading
            the nuclear coordinates. The default value is set to ``15`` bits.
        select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.subroutines.QROM`
            used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.
        wires (WiresLike | None): the wires on which the operator acts

    Resources:
        The resources are calculated based on Section II A of `arXiv:2105.12767 <https://arxiv.org/abs/2105.12767>`_.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> fq_ham = qre.FirstQuantizedHamiltonian(num_orbitals, num_particles, num_nuclei, box_length)
    >>> res = qre.estimate(qre.PrepFirstQuantization(fq_ham, coordinates_precision=15))
    >>> print(res)

    """

    resource_keys = {"thc_ham", "coordinates_precision", "select_swap_depth"}

    def __init__(
        self,
        fq_ham: FirstQuantizedHamiltonian,
        coordinates_precision: int = 15,
        select_swap_depth: int | None = None,
        wires: WiresLike | None = None,
    ):

        if not isinstance(fq_ham, FirstQuantizedHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for PrepFirstQuantization."
                f"This method works with first quantized Hamiltonian, {type(fq_ham)} provided"
            )

        if not isinstance(coordinates_precision, int):
            raise TypeError(
                f"`coordinates_precision` must be an integer, but type {type(coordinates_precision)} was provided."
            )

        self.fq_ham = fq_ham
        self.coordinates_precision = coordinates_precision
        self.select_swap_depth = select_swap_depth

        n_eta = ceil_log2(fq_ham.num_electrons)
        n_p = ceil_log2(fq_ham.num_plane_waves ** (1 / 3) + 1)
        lambda_zeta = fq_ham.num_electrons + fq_ham.charge
        n_eta_lz = ceil_log2(lambda_zeta + 2*lambda_zeta)
        n_M = ceil_log2(4*fq_ham.num_electrons**2/ fq_ham.omega**(1/3))
        # Total number of wires is obtained based on the Appendix C of `arXiv:2105.12767 <https://arxiv.org/abs/2105.12767>`_.
        self.num_wires = n_eta_lz + 2*n_eta + 6*n_p + n_M + 16

        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * fq_ham (:class:`~.pennylane.estimator.compact_hamiltonian.FirstQuantizedHamiltonian`): a first quantized
                  Hamiltonian for which the state is being prepared
                * coordinates_precision (int): The number of bits used to represent the precision for loading
                  the nuclear coordinates. The default value is set to ``15`` bits.
                * select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.QROM`
                  used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.
        """
        return {
            "fq_ham": self.fq_ham,
            "coordinates_precision": self.coordinates_precision,
            "select_swap_depth": self.select_swap_depth,
        }

    @classmethod
    def resource_rep(
        cls,
        fq_ham: FirstQuantizedHamiltonian,
        coordinates_precision: int = 15,
        select_swap_depth: int | None = None,
    ) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            fq_ham (:class:`~pennylane.estimator.compact_hamiltonian.FirstQuantizedHamiltonian`): a first quantized
                Hamiltonian for which the state is being prepared
            coordinates_precision (int): The number of bits used to represent the precision for loading
                the nuclear coordinates. The default value is set to ``15`` bits.
            select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.QROM`
                used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.
        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        if not isinstance(fq_ham, FirstQuantizedHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for PrepFirstQuantization."
                f"This method works with first quantized Hamiltonian, {type(fq_ham)} provided"
            )

        if not isinstance(coordinates_precision, int):
            raise TypeError(
                f"`coordinates_precision` must be an integer, but type {type(coordinates_precision)} was provided."
            )

        n_eta = ceil_log2(fq_ham.num_electrons)
        n_p = ceil_log2(fq_ham.num_plane_waves ** (1 / 3) + 1)
        lambda_zeta = fq_ham.num_electrons + fq_ham.charge
        n_eta_lz = ceil_log2(lambda_zeta + 2*lambda_zeta)
        n_M = ceil_log2(4*fq_ham.num_electrons**2/ fq_ham.omega**(1/3))
        # Total number of wires is obtained based on the Appendix C of `arXiv:2105.12767 <https://arxiv.org/abs/2105.12767>`_.
        num_wires = n_eta_lz + 2*n_eta + 6*n_p + n_M + 16

        params = {
            "fq_ham": fq_ham,
            "coordinates_precision": coordinates_precision,
            "select_swap_depth": select_swap_depth,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @staticmethod
    def _superposition_prep_costs(
        value: int,
        register_size: int,
        ) -> list[GateCount]:
        """Resource costs for preparing an equal superposition over a register.

        The resources are obtained from Appendix A.2 of `arXiv:2011.03494 <https://arxiv.org/pdf/2011.03494>`_.
        Cost per register: 3 * register_size + 2 * br - 9 Toffolis.

        Args:
            value (int): The classical integer for the inequality test.
            register_size (int): Number of qubits in each register.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: List of GateCount objects.
        """
        br = 8 # number of bits for the rotation precision of ancilla for equal superposition preparation
        gate_lst = []

        ineq = resource_rep(
                qre.OutOfPlaceIntegerComparator,
                {"value": value, "register_size": register_size, "geq": False},
        )

        toffoli = resource_rep(qre.Toffoli)
        cz = resource_rep(qre.CZ)

        # Forward inequality test
        gate_lst.append(GateCount(ineq, 1))

        # Rotation on ancilla
        gate_lst.append(GateCount(toffoli, br - 3))

        # CZ
        gate_lst.append(GateCount(cz, 1))

        # Inverse rotation
        gate_lst.append(GateCount(toffoli, br - 3))

        # Adjoint inequality (0 Toffoli cost)
        gate_lst.append(GateCount(
                    resource_rep(qre.Adjoint, {"base_cmpr_op": ineq}),
                    1,
        ))

        # Reflection
        gate_lst.append(GateCount(toffoli, register_size - 1))

        # Compute the inequality again
        gate_lst.append(GateCount(ineq, 1))

        return gate_lst

    @classmethod
    def resource_decomp(
        cls,
        fq_ham: FirstQuantizedHamiltonian,
        coordinates_precision: int = 15,
        select_swap_depth: int | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Args:
            fq_ham (:class:`~pennylane.estimator.compact_hamiltonian.FirstQuantizedHamiltonian`): a first quantized
                Hamiltonian for which the walk operator is being created
            coordinates_precision (int): The number of bits used to represent the precision for loading
                the nuclear coordinates. The default value is set to ``15`` bits.
            select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.QROM`
                used to trade-off extra qubits for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.

        Resources:
            The resources are calculated based on Section II A of `arXiv:2105.12767 <https://arxiv.org/abs/2105.12767>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """

        gate_list = []
        br = 8
        eta = fq_ham.num_electrons
        num_pw = fq_ham.num_plane_waves
        n_eta = ceil_log2(eta)
        omega = fq_ham.omega

        # number of qubits required to store a signed
        # binary representation of one component of the momentum
        # of a single electron, taken from Eq. (22) of PRX Quantum 2, 040332 (2021)
        n_p = ceil_log2(num_pw ** (1 / 3) + 1)

        lambda_zeta = eta + fq_ham.charge # sum of nuclear charges

        # Number of ancilla qubits in inequality testing for preparation of Coulomb potential
        # Taken from https://arxiv.org/pdf/2602.20234
        n_M = ceil_log2(4*eta**2/ omega**(1/3))

        # Step:1: Rotate the ancilla qubit for selecting between T and U+V Preparations
        gate_list.append(GateCount(resource_rep(qre.RY), 1))

        # Step 2a: Prepare an equal superposition over i and j registers
        gate_list.extend(cls._superposition_prep_costs(eta, n_eta))
        gate_list.extend(cls._superposition_prep_costs(eta, n_eta))

        # Step 2b and 2c:
        eq = resource_rep(qre.RegisterEquality, {"register_size": n_eta})
        gate_list.append(GateCount(eq, 1)) # compute the equality for i and j registers
        gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": eq}), 1)) # adjoint of equality test
        toffoli = resource_rep(qre.Toffoli)
        gate_list.append(GateCount(toffoli, 2)) # Extra toffolis to flag and invert the success of the equality test

        # Step 2d: Invert the superposition over i and j registers
        gate_list.extend(cls._superposition_prep_costs(eta, n_eta))
        gate_list.extend(cls._superposition_prep_costs(eta, n_eta))

        # Step 3: Prepare the superposition over w,r,s registers to use for T part of the select operation
        # Over w register: 3 spatial coordinates
        n_w = ceil_log2(3)
        gate_list.extend(cls._superposition_prep_costs(3, n_w))

        # Over r and s registers: we need a cascade of controlled Hadamard gates
        gate_list.extend(qre.ch_toffoli_based_resource_decomp() * (2 * (n_p-2)))

        # Step 4: Prepare the superposition state for selection between U and V registers
        n_eta_lz = ceil_log2(eta + 2*lambda_zeta)
        gate_list.extend(cls._superposition_prep_costs(eta + 2*lambda_zeta, n_eta_lz))
        ineq = resource_rep(
                qre.OutOfPlaceIntegerComparator,
                {"value": eta, "register_size": n_eta_lz, "geq": False},
        )
        gate_list.append(GateCount(ineq, 1))
        gate_list.append(GateCount(toffoli, 1))

        # Step 5: Prepare the superposition over \nu register
        # a) Create a superposition over \mu register
        gate_list.extend(qre.ch_toffoli_based_resource_decomp() * (n_p-1))

        # b) Create a superposition over \nu register for all 3 spatial coordinates
        gate_list.append(GateCount(resource_rep(qre.Hadamard), 6))
        gate_list.extend(qre.ch_toffoli_based_resource_decomp() * (3 * (n_p-1)))

        # c) remove -0 from the representation of \nu in the superposition
        mcx = resource_rep(qre.MultiControlledX, {"num_ctrl_wires":n_p+1, "num_zero_ctrl":0})
        gate_list.append(GateCount(mcx, 3))
        gate_list.append(GateCount(resource_rep(qre.Toffoli), 2)) # Check if any of them returned True

        # d) test whether all of νx, νy , and νz are smaller in absolute value than 2μ−2
        # convert the \mu register to one-hot unary
        cnot = resource_rep(qre.CNOT)
        gate_list.append(GateCount(cnot, n_p-1)) # cascade of CNOTs to convert from binary to unary

        mcx = resource_rep(qre.MultiControlledX, {"num_ctrl_wires":4, "num_zero_ctrl":0})
        gate_list.append(GateCount(mcx, n_p))

        # e) compute m(\nu_x^2 + \nu_y^2 + \nu_z^2)
        # sum of three squares
        gate_list.append(GateCount(toffoli, 3*n_p**2 - n_p - 1))

        # Multiply two numbers of length log(M) and 2*n_p + 2
        mult = resource_rep(qre.OutMultiplier, {"a_num_wires": n_M, "b_num_wires": 2*n_p + 2})
        gate_list.append(GateCount(mult, 1))

        # f) Test inequality
        gate_list.append(GateCount(toffoli, 2*n_p + n_M + 2)) # Cost of testing the inequality using Toffolis

        # g) Toffolis to flag the success
        gate_list.append(GateCount(toffoli, 3))

        # h) uncompute the state prep- rest of the cost is in Cliffords
        # only the controlled-Hadamards used to prepare \mu and \nu registers need to be uncomputed
        gate_list.extend(qre.ch_toffoli_based_resource_decomp() * 4 * (n_p-1))

        # Step 6: Amplitude loading for T, U and V
        qrom = resource_rep(qre.QROM, {"num_bitstrings": lambda_zeta, "size_bitstring": coordinates_precision, "select_swap_depth": select_swap_depth})
        gate_list.append(GateCount(qrom, 1))

        return gate_list

