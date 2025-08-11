import pennylane.labs.resource_estimation as plre
from pennylane.labs.resource_estimation import GateCount, AllocWires, FreeWires
import math

class ResourceSelectTHC(ResourceOperator):
    r"""Resource class for creating the Select operator for THC Hamiltonian.

    Args:
        compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
            Hamiltonian on which the select operator is being applied
        rotation_precision (float, optional): precision for loading the rotation angles
        select_swap_depth (int, optional): A natural number that determines if data
            will be loaded in parallel by adding more rows following Figure 1.C of `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_.
            Defaults to :code:`None`, which internally determines the optimal depth.
        wires (list[int] or optional): the wires on which the operator acts

    Resources:
        The resources are calculated based on Figure 5 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

    **Example**

    The resources for this operation are computed using:

    >>> compact_ham = plre.CompactHamiltonian.thc(num_orbitals=20, tensor_rank=40)
    >>> res = plre.estimate_resources(plre.ResourceSelectTHC(compact_ham))
    >>> print(res)
    """

    def __init__(self, compact_ham, rotation_precision=None, select_swap_depth=None, wires=None):

        if compact_ham.method_name != "thc":
            raise TypeError(
                f"Unsupported Hamiltonian representation for ResourceQubitizeTHC."
                f"This method works with thc Hamiltonian, {compact_ham.method_name} provided"
            )
        self.compact_ham = compact_ham
        self.rotation_precision = rotation_precision
        self.select_swap_depth = select_swap_depth
        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]
        self.num_wires = num_orb*2 +  2 * int(np.ceil(math.log2(tensor_rank+1))) + 6
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                    Hamiltonian on which the select operator is being applied
                * rotation_precision (float, optional): precision for loading the rotation angles
                * select_swap_depth (int, optional): A natural number that determines if data
                    will be loaded in parallel by adding more rows following Figure 1.C of `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_.
                    Defaults to :code:`None`, which internally determines the optimal depth.
        """
        return {"compact_ham": self.compact_ham, "rotation_precision": rotation_precision, "select_swap_depth": self.select_swap_depth}

    @classmethod
    def resource_rep(cls, compact_ham, rotation_precision=None, select_swap_depth=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian on which the select operator is being applied
            rotation_precision (float, optional): precision for loading the rotation angles
            select_swap_depth (int, optional): A natural number that determines if data
                will be loaded in parallel by adding more rows following Figure 1.C of `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_.
                Defaults to :code:`None`, which internally determines the optimal depth.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"compact_ham": compact_ham, "rotation_precision": rotation_precision, "select_swap_depth": select_swap_depth}
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(cls, compact_ham, rotation_precision=None, select_swap_depth=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian on which the select operator is being applied
            coeff_precision (float, optional): precision for loading the rotation angles
            rotation_precision (float, optional): precision for loading the rotation angles for basis rotation
            compare_precision (float, optional): precision for comparing two numbers

        Resources:
            The resources are calculated based on Figure 5 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        rotation_precision = rotation_precision or kwargs["config"]["precision_qubitization_rotation"]
        rot_prec_wires = abs(math.floor(math.log2(rotation_precision)))

        gate_list = []

        # Resource state
        gate_list.append(AllocWires(rot_prec_wires))

        phase_grad = resource_rep(plre.ResourcePhaseGradient, {num_wires: rot_prec_wires})
        gate_list.append(GateCount(phase_grad,1))

        swap = resource_rep(plre.ResourceCSWAP)
        gate_list.append(GateCount(swap, 4*num_orb))

        # For 2-body integrals
        gate_list.append(AllocWires(rot_prec_wires*(num_orb-1)))
        qrom_twobody = resource_rep(
            re.ResourceQROM,
            {
                "num_bitstrings": tensor_rank+num_orb,
                "size_bitstring": rot_prec_wires,
                "clean": False,
                "select_swap_depth": select_swap_depth
            },
        )
        gate_lst.append(GateCount(qrom))

        semiadder = resource_rep(
                        re.ResourceControlled,
                        {
                            "base_cmpr_op": resource_rep(
                                re.ResourceSemiAdder,
                                {"max_register_size": rot_prec_wires},
                            ),
                            "num_ctrl_wires": 1,
                            "num_ctrl_values": 0,
                    },
                    )
        gate_list.append(GateCount(semiadder, num_orb-1))

        gate_list.append(GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": qrom_twobody})))
        gate_list.append(GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op":semiadder}), num_orb-1))


        # For one body integrals
        qrom_onebody = resource_rep(
            re.ResourceQROM,
            {
                "num_bitstrings": tensor_rank,
                "size_bitstring": rot_prec_wires,
                "clean": False,
                "select_swap_depth": select_swap_depth
            },
        )
        gate_lst.append(GateCount(qrom))

        gate_list.append(GateCount(semiadder, num_orb-1))

        h = resource_rep(re.ResourceHadamard)
        s = resource_rep(re.ResourceS)
        s_dagg = resource_rep(re.ResourceAdjoint, {"base_cmpr_op": s})

        gate_lst.append(GateCount(h, 4*(num_orb)))
        gate_lst.append(GateCount(s, 2*num_orb))
        gate_lst.append(GateCount(s_dagg, 2*num_orb))

        gate_list.append(GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": qrom_onebody})))
        gate_list.append(GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op":semiadder}), num_orb-1))

        # Z gate in the center of rotations
        cz = resource_rep(plre.ResourceControlled,
                    {
                       "base_cmpr_op": plre.ResourceZ.resource_rep(),
                        "num_ctrl_wires": 1,
                        "num_ctrl_values": 0,
                    })
        gate_list.append(plre.GateCount(cz, 1))

        ccz = resource_rep(plre.ResourceControlled,
                    {
                       "base_cmpr_op": plre.ResourceZ.resource_rep(),
                        "num_ctrl_wires": 2,
                        "num_ctrl_values": 1,
                    })
        gate_list.append(plre.GateCount(ccz, 1))

        gate_list.append(FreeWires(rot_prec_wires*(num_orb-1)))
        gate_list.append(FreeWires(rot_prec_wires))
        return gate_list
