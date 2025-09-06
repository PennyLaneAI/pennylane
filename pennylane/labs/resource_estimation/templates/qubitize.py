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
r"""Resource operators for PennyLane subroutine templates."""
import math

import numpy as np

from pennylane.labs import resource_estimation as plre
from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.wires import Wires

# pylint: disable=too-many-arguments, arguments-differ


class ResourceQubitizeTHC(ResourceOperator):
    r"""Resource class for Qubitization of THC Hamiltonian.

    Args:
        compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
            Hamiltonian for which the walk operator is being created
        coeff_precision_bits (int, optional): Number of bits used to represent the precision for loading
            the coefficients of the Hamiltonian. If :code:`None` is provided the default value from the
            :code:`resource_config` is used.
        rotation_precision_bits (int, optional): Number of bits used to represent the precision for loading
            the rotation angles for basis rotation. If :code:`None` is provided the default value from the
            :code:`resource_config` is used.
        select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
            used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
            ``select_swap_depth`` individually for :code:`ResourcePrepTHC` and :code:`ResourceSelectTHC` circuits,
            respectively.
        wires (list[int] or optional): the wires on which the operator acts

    Resources:
        The resources are calculated based on `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

    **Example**

    The resources for this operation are computed using:

    >>> compact_ham = plre.CompactHamiltonian.thc(num_orbitals=20, tensor_rank=40)
    >>> res = plre.estimate_resources(plre.ResourceQubitizeTHC(compact_ham))
    >>> print(res)
    --- Resources: ---
     Total qubits: 371
     Total gates : 4.885E+4
     Qubit breakdown:
      clean qubits: 313, dirty qubits: 0, algorithmic qubits: 58
     Gate breakdown:
      {'Toffoli': 3.156E+3, 'CNOT': 3.646E+4, 'X': 1.201E+3, 'Hadamard': 7.913E+3, 'S': 80, 'Z': 41}

    """

    def __init__(
        self,
        compact_ham,
        coeff_precision_bits=None,
        rotation_precision_bits=None,
        select_swap_depths=None,
        wires=None,
    ):
        if compact_ham.method_name != "thc":
            raise TypeError(
                f"Unsupported Hamiltonian representation for ResourceQubitizeTHC."
                f"This method works with thc Hamiltonian, {compact_ham.method_name} provided"
            )

        if isinstance(select_swap_depths, (list, tuple, np.ndarray)):
            if len(select_swap_depths) != 2:
                raise ValueError(
                    f"Expected the length of `select_swap_depths` to be 2, got {len(select_swap_depths)}"
                )
        elif not (isinstance(select_swap_depths, int) or select_swap_depths is None):
            raise TypeError("`select_swap_depths` must be an integer, None or iterable")

        self.compact_ham = compact_ham
        self.coeff_precision_bits = coeff_precision_bits
        self.rotation_precision_bits = rotation_precision_bits
        self.select_swap_depths = select_swap_depths
        if wires is not None:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            num_orb = compact_ham.params["num_orbitals"]
            tensor_rank = compact_ham.params["tensor_rank"]
            self.num_wires = num_orb * 2 + 2 * int(np.ceil(math.log2(tensor_rank + 1))) + 6
            self.wires = None
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                  Hamiltonian for which the walk operator is being created
                * coeff_precision_bits (int, optional): The number of bits used to represent the precision for loading
                  the coefficients of Hamiltonian. If :code:`None` is provided the default value from the
                  :code:`resource_config` is used.
                * rotation_precision_bits (int, optional): The number of bits used to represent the precision for loading
                  the rotation angles for basis rotation. If :code:`None` is provided the default value from the
                  :code:`resource_config` is used.
                * select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
                    used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
                    ``select_swap_depth`` individually for :code:`ResourcePrepTHC` and :code:`ResourceSelectTHC` circuits,
                    respectively.
        """
        return {
            "compact_ham": self.compact_ham,
            "coeff_precision_bits": self.coeff_precision_bits,
            "rotation_precision_bits": self.rotation_precision_bits,
            "select_swap_depths": self.select_swap_depths,
        }

    @classmethod
    def resource_rep(
        cls, compact_ham, coeff_precision_bits, rotation_precision_bits, select_swap_depths
    ) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            coeff_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            rotation_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
                ``select_swap_depth`` individually for :code:`ResourcePrepTHC` and :code:`ResourceSelectTHC` circuits,
                respectively.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """

        if isinstance(select_swap_depths, (list, tuple, np.ndarray)):
            if len(select_swap_depths) != 2:
                raise ValueError(
                    f"Expected the length of `select_swap_depths` to be 2, got {len(select_swap_depths)}"
                )
        elif not (isinstance(select_swap_depths, int) or select_swap_depths is None):
            raise TypeError("`select_swap_depths` must be an integer, None or iterable")

        params = {
            "compact_ham": compact_ham,
            "coeff_precision_bits": coeff_precision_bits,
            "rotation_precision_bits": rotation_precision_bits,
            "select_swap_depths": select_swap_depths,
        }
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(
        cls,
        compact_ham,
        coeff_precision_bits=None,
        rotation_precision_bits=None,
        select_swap_depths=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        .. note::

            This decomposition assumes an appropriately sized phase gradient state is available.
            Users should ensure the cost of constructing such a state has been accounted for.
            See also :class:`~.pennylane.labs.resource_estimation.ResourcePhaseGradient`.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            coeff_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            rotation_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
                ``select_swap_depth`` individually for :code:`ResourcePrepTHC` and :code:`ResourceSelectTHC` circuits,
                respectively.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        gate_list = []

        tensor_rank = compact_ham.params["tensor_rank"]
        m_register = int(np.ceil(np.log2(tensor_rank)))
        coeff_precision_bits = coeff_precision_bits or kwargs["config"]["qubitization_coeff_bits"]

        if isinstance(select_swap_depths, int) or select_swap_depths is None:
            select_swap_depths = [select_swap_depths] * 2

        select = resource_rep(
            plre.ResourceSelectTHC,
            {
                "compact_ham": compact_ham,
                "rotation_precision_bits": rotation_precision_bits,
                "select_swap_depth": select_swap_depths[1],
            },
        )
        gate_list.append(GateCount(select))

        prep = resource_rep(
            plre.ResourcePrepTHC,
            {
                "compact_ham": compact_ham,
                "coeff_precision_bits": coeff_precision_bits,
                "select_swap_depth": select_swap_depths[0],
            },
        )
        gate_list.append(GateCount(prep))
        gate_list.append(GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": prep})))

        # reflection cost
        toffoli = resource_rep(plre.ResourceToffoli)
        gate_list.append(GateCount(toffoli, 2 * m_register + coeff_precision_bits + 4))

        return gate_list

    @classmethod
    def default_controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        compact_ham,
        coeff_precision_bits=None,
        rotation_precision_bits=None,
        select_swap_depths=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for the controlled version of the operator.

        .. note::

            This decomposition assumes an appropriately sized phase gradient state is available.
            Users should ensure the cost of constructing such a state has been accounted for.
            See also :class:`~.pennylane.labs.resource_estimation.ResourcePhaseGradient`.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            coeff_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            rotation_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
                ``select_swap_depth`` individually for :code:`ResourcePrepTHC` and :code:`ResourceSelectTHC` circuits,
                respectively.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        gate_list = []

        tensor_rank = compact_ham.params["tensor_rank"]
        m_register = int(np.ceil(np.log2(tensor_rank)))
        coeff_precision_bits = coeff_precision_bits or kwargs["config"]["qubitization_coeff_bits"]

        if isinstance(select_swap_depths, int) or select_swap_depths is None:
            select_swap_depths = [select_swap_depths] * 2

        if ctrl_num_ctrl_wires > 1:
            mcx = resource_rep(
                plre.ResourceMultiControlledX,
                {
                    "num_ctrl_wires": ctrl_num_ctrl_wires,
                    "num_ctrl_values": ctrl_num_ctrl_values,
                },
            )
            gate_list.append(AllocWires(1))
            gate_list.append(GateCount(mcx, 2))

        select = resource_rep(
            plre.ResourceSelectTHC,
            {
                "compact_ham": compact_ham,
                "rotation_precision": rotation_precision_bits,
                "select_swap_depth": select_swap_depths[1],
            },
        )
        gate_list.append(
            GateCount(
                resource_rep(
                    plre.ResourceControlled,
                    {"base_cmpr_op": select, "num_ctrl_wires": 1, "num_ctrl_values": 0},
                )
            )
        )

        prep = resource_rep(
            plre.ResourcePrepTHC,
            {
                "compact_ham": compact_ham,
                "coeff_precision": coeff_precision_bits,
                "select_swap_depth": select_swap_depths[0],
            },
        )
        gate_list.append(GateCount(prep))
        gate_list.append(GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": prep})))

        # reflection cost
        toffoli = resource_rep(plre.ResourceToffoli)
        gate_list.append(GateCount(toffoli, 2 * m_register + coeff_precision_bits + 4))

        if ctrl_num_ctrl_wires > 1:
            gate_list.append(FreeWires(1))
        elif ctrl_num_ctrl_values > 0:
            gate_list.append(GateCount(resource_rep(plre.ResourceX), 2 * ctrl_num_ctrl_values))

        return gate_list

class ResourceQubitizeDF(ResourceOperator):
    r"""Resource class for Qubitization of Double Factorized Hamiltonian.

    Args:
        compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a double factorized
            Hamiltonian for which the walk operator is being created
        coeff_precision_bits (int, optional): Number of bits used to represent the precision for loading
            the coefficients of the Hamiltonian. If :code:`None` is provided the default value from the
            :code:`resource_config` is used.
        rotation_precision_bits (int, optional): Number of bits used to represent the precision for loading
            the rotation angles for basis rotation. If :code:`None` is provided the default value from the
            :code:`resource_config` is used.
        select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
            used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
            ``select_swap_depth`` individually for :code:`ResourcePrepTHC` and :code:`ResourceSelectTHC` circuits,
            respectively.
        wires (list[int] or optional): the wires on which the operator acts

    Resources:
        The resources are calculated based on `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

    **Example**

    The resources for this operation are computed using:

    >>> compact_ham = plre.CompactHamiltonian.df(num_orbitals=20, num_fragments=10, average_rank=40, max_rank=100)
    >>> res = plre.estimate_resources(plre.ResourceQubitizeDF(compact_ham))
    >>> print(res)

    """

    def __init__(
        self,
        compact_ham,
        coeff_precision_bits=None,
        rotation_precision_bits=None,
        select_swap_depths=None,
        wires=None,
    ):
        if compact_ham.method_name != "df":
            raise TypeError(
                f"Unsupported Hamiltonian representation for ResourceQubitizeDF."
                f"This method works with df Hamiltonian, {compact_ham.method_name} provided"
            )

        if isinstance(select_swap_depths, (list, tuple, np.ndarray)):
            if len(select_swap_depths) != 2:
                raise ValueError(
                    f"Expected the length of `select_swap_depths` to be 2, got {len(select_swap_depths)}"
                )
        elif not (isinstance(select_swap_depths, int) or select_swap_depths is None):
            raise TypeError("`select_swap_depths` must be an integer, None or iterable")

        self.compact_ham = compact_ham
        self.coeff_precision_bits = coeff_precision_bits
        self.rotation_precision_bits = rotation_precision_bits
        self.select_swap_depths = select_swap_depths
        # if wires is not None:
        #     self.wires = Wires(wires)
        #     self.num_wires = len(self.wires)
        # else:
        #     num_orb = compact_ham.params["num_orbitals"]
        #     tensor_rank = compact_ham.params["tensor_rank"]
        #     self.num_wires = num_orb * 2 + 2 * int(np.ceil(math.log2(tensor_rank + 1))) + 6
        #     self.wires = None
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                  Hamiltonian for which the walk operator is being created
                * coeff_precision_bits (int, optional): The number of bits used to represent the precision for loading
                  the coefficients of Hamiltonian. If :code:`None` is provided the default value from the
                  :code:`resource_config` is used.
                * rotation_precision_bits (int, optional): The number of bits used to represent the precision for loading
                  the rotation angles for basis rotation. If :code:`None` is provided the default value from the
                  :code:`resource_config` is used.
                * select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
                    used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
                    ``select_swap_depth`` individually for :code:`ResourcePrepTHC` and :code:`ResourceSelectTHC` circuits,
                    respectively.
        """
        return {
            "compact_ham": self.compact_ham,
            "coeff_precision_bits": self.coeff_precision_bits,
            "rotation_precision_bits": self.rotation_precision_bits,
            "select_swap_depths": self.select_swap_depths,
        }

    @classmethod
    def resource_rep(
        cls, compact_ham, coeff_precision_bits, rotation_precision_bits, select_swap_depths
    ) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            coeff_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            rotation_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
                ``select_swap_depth`` individually for :code:`ResourcePrepTHC` and :code:`ResourceSelectTHC` circuits,
                respectively.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """

        if isinstance(select_swap_depths, (list, tuple, np.ndarray)):
            if len(select_swap_depths) != 2:
                raise ValueError(
                    f"Expected the length of `select_swap_depths` to be 2, got {len(select_swap_depths)}"
                )
        elif not (isinstance(select_swap_depths, int) or select_swap_depths is None):
            raise TypeError("`select_swap_depths` must be an integer, None or iterable")

        params = {
            "compact_ham": compact_ham,
            "coeff_precision_bits": coeff_precision_bits,
            "rotation_precision_bits": rotation_precision_bits,
            "select_swap_depths": select_swap_depths,
        }
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(
        cls,
        compact_ham,
        coeff_precision_bits=None,
        rotation_precision_bits=None,
        select_swap_depths=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        .. note::

            This decomposition assumes an appropriately sized phase gradient state is available.
            Users should ensure the cost of constructing such a state has been accounted for.
            See also :class:`~.pennylane.labs.resource_estimation.ResourcePhaseGradient`.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            coeff_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            rotation_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
                ``select_swap_depth`` individually for :code:`ResourcePrepTHC` and :code:`ResourceSelectTHC` circuits,
                respectively.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """

        coeff_precision_bits = coeff_precision_bits or kwargs["config"]["qubitization_coeff_bits"]
        rotation_precision_bits = rotation_precision_bits or kwargs["config"]["qubitization_rotation_bits"]
        gate_list = []
        num_orbitals = compact_ham.params["num_orbitals"]
        xi = compact_ham.params["num_orbitals"]
        Lxi = compact_ham.params["num_eigenvectors"]
        L = compact_ham.params["num_fragments"]
        num_coeff = Lxi + num_orbitals

        nl_register = int(np.ceil(np.log2(L + 1)))
        nxi = int(np.ceil(np.log2(xi)))
        coeff_register = int(np.ceil(np.log2(num_coeff)))
        #print("coeff: ", num_coeff, coeff_register)

        # Fig 16 in arXiv:2011.03494
        # Step1 : Prep
        # 1a, Prepare equal superposition over L+1 basis states, step 1a

        eta = ((L+1) & -(L+1)).bit_length() - 1
        # paper assumes cost of inequality to be equal to a adder's cost
        comparator = resource_rep(plre.ResourceSemiAdder, {"max_register_size": nl_register-eta})
        gate_list.append(GateCount(comparator, 1))

        # Rotate an ancilla to obtain amplitude for sucess
        # and invert the rotation
        toffoli = resource_rep(plre.ResourceToffoli)
        gate_list.append(AllocWires(coeff_precision_bits))
        gate_list.append(GateCount(toffoli, 2*(coeff_precision_bits-3)))
        gate_list.append(FreeWires(coeff_precision_bits))

        # Reflection on \lceil log(L+1) \rceil - eta - 1 qubits
        gate_list.append(GateCount(toffoli, nl_register-eta-1))

        # Inequality test again
        gate_list.append(GateCount(comparator, 1))

        # step 1b
        qrom_prep1 = resource_rep(
            plre.ResourceQROM,
            {
                "num_bitstrings": L+1,
                "size_bitstring": nl_register+coeff_precision_bits,
                "clean": False,
                "select_swap_depth": select_swap_depths,
            },
        )
        gate_list.append(GateCount(qrom_prep1, 1))

        # step 1c
        ineq = resource_rep(plre.ResourceSemiAdder, {"max_register_size": coeff_precision_bits+1})
        gate_list.append(GateCount(ineq, 1))

        # step 1d
        cswap = resource_rep(plre.ResourceCSWAP)
        gate_list.append(GateCount(cswap, nl_register))

        # Step2 : Output data from the l register
        qrom_output = resource_rep(plre.ResourceQROM, {"num_bitstrings": L+1, "size_bitstring": nxi+coeff_register+coeff_precision_bits+1 , "clean": False, "select_swap_depth": select_swap_depths})
        gate_list.append(GateCount(qrom_output,1))

        # Step3 : Prepare the state on p register controlled on l register
        # step 3a:
        # copy the nxi register: i
        gate_list.append(GateCount(toffoli, 4*nxi-4))

        # controlled Hadamards - catalytic decomposition: ii
        gate_list.append(GateCount(toffoli, 4*nxi))

        # inequality test on xi register: iii
        xi_comparator = resource_rep(plre.ResourceSemiAdder, {"max_register_size": nxi+1})
        gate_list.append(GateCount(xi_comparator,4))

        # rotate and invert the rotation of ancilla: iv and vi
        gate_list.append(AllocWires(coeff_precision_bits))
        gate_list.append(GateCount(toffoli,8*coeff_precision_bits-16))
        gate_list.append(FreeWires(coeff_precision_bits))

        #Reflection on the result of inequality: v
        cz = resource_rep(plre.ResourceCZ)
        gate_list.append(GateCount(cz,4))

        # Controlled Hadamards: vii and ix
        gate_list.append(GateCount(toffoli, 8*nxi))

        # Reflect about the zero state: viii
        gate_list.append(GateCount(toffoli, 4*nxi-4))

        # inequality test again: x
        gate_list.append(GateCount(xi_comparator,4))

        #step3b: Add the offset to the second register
        adder = resource_rep(plre.ResourceSemiAdder, {"max_register_size": coeff_register})
        gate_list.append(GateCount(adder,4))

        # #step3c: QROM to output alt and keep values
        qrom_prep2 = resource_rep(plre.ResourceQROM, {"num_bitstrings": num_coeff, "size_bitstring": nxi+coeff_precision_bits+2 , "clean": False, "select_swap_depth": select_swap_depths})
        gate_list.append(GateCount(qrom_prep2, 1))

        # #step3d: Inequality test and controlled swaps
        gate_list.append(GateCount(ineq,4))
        gate_list.append(GateCount(cswap, 4*nxi))

        # Step 4: Apply number operators via rotations
        #step4a: Add offset to the second register
        gate_list.append(GateCount(adder,2))

        #step4b: QROM for the rotation angles
        # For 2-body
        qrom_rot_twobody = resource_rep(plre.ResourceQROM, {"num_bitstrings": num_coeff, "size_bitstring": num_orbitals*rotation_precision_bits , "clean": False, "select_swap_depth": select_swap_depths})
        gate_list.append(GateCount(qrom_rot_twobody,1))

        #step4c: controlled swaps controlled on spin qubit
        gate_list.append(GateCount(cswap, 2*num_orbitals))

        #step4d: Controlled rotations based on semiadder
        ctrl_adder = resource_rep(plre.ResourceControlled, {"base_cmpr_op": resource_rep(plre.ResourceSemiAdder, {"max_register_size": rotation_precision_bits-1}), "num_ctrl_wires":1, "num_ctrl_values":0})
        gate_list.append(GateCount(ctrl_adder,2*num_orbitals))

        #step4e: Z1 controlled on success of prep of l and p registers
        ccz = resource_rep(plre.ResourceCCZ)
        gate_list.append(GateCount(ccz, 2))

        #step4f: reverse the controlled rotations and cswaps
        gate_list.append(GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": ctrl_adder}), 2*num_orbitals))
        gate_list.append(GateCount(cswap, 2*num_orbitals))

        #step4g: Reverse the qrom
        qrom_rot_twobody_adj = resource_rep(plre.ResourceQROM, {"num_bitstrings": num_coeff, "size_bitstring": num_orbitals, "clean": False, "select_swap_depth": select_swap_depths})
        gate_list.append(GateCount(qrom_rot_twobody_adj,1))

        #step4h: Reverse the addition
        gate_list.append(GateCount(adder,2))

        # Step 5: Invert the state prep cost, same as step: 3, with a different QROM cost
        # Appropriately changed step 3 numbers except for QROM
        qrom_prep2_adj = resource_rep(plre.ResourceQROM, {"num_bitstrings": num_coeff, "size_bitstring": num_orbitals , "clean": False, "select_swap_depth": select_swap_depths})
        gate_list.append(GateCount(qrom_prep2_adj, 1))

        # Step 6: Reflection cost
        gate_list.append(GateCount(toffoli, nxi+coeff_precision_bits+2))

        # Step 7: Repeat steps 2-5 for one-electron integrals
        # Appropriately doubled the resources, adding QROMs here
        print(Lxi, nxi+coeff_precision_bits+2)
        qrom_prep2_onebody = resource_rep(plre.ResourceQROM, {"num_bitstrings": Lxi, "size_bitstring": nxi+coeff_precision_bits+2 , "clean": False, "select_swap_depth": select_swap_depths})
        gate_list.append(GateCount(qrom_prep2_onebody))

        qrom_rot_onebody = resource_rep(plre.ResourceQROM, {"num_bitstrings": Lxi, "size_bitstring": num_orbitals*rotation_precision_bits , "clean": False, "select_swap_depth": select_swap_depths})
        gate_list.append(GateCount(qrom_rot_onebody,1))

        qrom_rot_onebody_adj = resource_rep(plre.ResourceQROM, {"num_bitstrings": Lxi, "size_bitstring": num_orbitals, "clean": False, "select_swap_depth": select_swap_depths})
        gate_list.append(GateCount(qrom_rot_onebody_adj,1))

        qrom_prep2_onebody_adj = resource_rep(plre.ResourceQROM, {"num_bitstrings": Lxi, "size_bitstring": num_orbitals, "clean": False, "select_swap_depth": select_swap_depths})
        gate_list.append(GateCount(qrom_prep2_onebody_adj, 1))

        # Step 8: Invert the QROM in step:2
        qrom_output_adj = resource_rep(plre.ResourceQROM, {"num_bitstrings": L+1, "size_bitstring": nxi , "clean": False, "select_swap_depth": select_swap_depths})
        gate_list.append(GateCount(qrom_output_adj,1))

        # Step 9: Reflection needed for walk operator
        gate_list.append(GateCount(toffoli, nl_register+nxi+2*coeff_precision_bits+1))

        return gate_list
