import pennylane as qml
import pennylane.numpy as np
from pennylane.operation import Operation, AnyWires
import re


def compute_indices_MPS(wires, loc):
    """
    Generate a list of wire indices that quantum gates acts on
    Args:
        loc (int): local wire number of a single quantum gate
        wires (Iterable): the total set of wires
    Returns:
        layers (array): array of wire indices or wire labels for each block
    """
    
    layers = np.array([[wires[idx] for idx in range(j, j+loc)] for j in range(0, len(wires) - loc // 2, loc // 2)])
    return layers


class MPS_from_function(Operation):

    num_params = 1
    num_wires = AnyWires
    par_domain = "A"

    def __init__(
        self,
        wires,
        loc,
        block,
        n_params_block,
        weights=None,
        do_queue=True,
        id=None,
    ):
        n_wires = len(wires)
        assert loc >= 2, f"loc must be larger than or equal to 2; got {loc}"
        assert n_wires >= 3, f"number of wires must be greater than or equal to 3; got {n_wires}"
        assert loc <= n_wires, f"loc must be smaller than or equal to the number of wires; got loc = {loc} and number of wires = {n_wires}"
  
        self.n_blocks = int(n_wires / (loc / 2) - 1)
        self.block = block

        if weights is None:
            self.weights = np.random.rand(n_params_block, int(self.n_blocks))
        else:
            self.weights = weights

        self.ind_gates = compute_indices_MPS(wires, loc)

        # TO DO: raise error if params_block does not match with the input block
        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    def expand(self):

        with qml.tape.QuantumTape() as tape:
            for idx, w in enumerate(self.ind_gates):
                self.block(weights=self.weights[idx], wires=w.tolist())
                # In this ordering, remember we need to measure the last qubit

        return tape

    @staticmethod
    def shape(n_wires, loc, n_params_block):

        r"""Returns the expected shape of the weights tensor.
        Args:
            n_wires (int): number of blocks
            loc (int): local dimension of the block
            n_params_block (int): number of parameters per block
        Returns:
            tuple[int]: shape
        """

        assert n_wires % 2 == 0, "n_wires should be an even integer"
        assert loc % 2 == 0, "loc should be an even integer"
        assert loc <= n_wires, "loc should be smaller or equal than num_wires"

        n_blocks = int(n_wires / (loc / 2) - 1)

        return n_blocks, n_params_block
