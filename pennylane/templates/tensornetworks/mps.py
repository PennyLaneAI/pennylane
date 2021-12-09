import pennylane as qml
import pennylane.numpy as np
from pennylane.operation import Operation, AnyWires


def compute_indices_MPS(wires, loc):
    """
    Generate a list of wire indices that quantum gates acts on
    Args:
        wires (Iterable): the total set of wires
        loc (int): local wire number of a single quantum gate
    Returns:
        layers (array): array of wire indices or wire labels for each block
    """

    if len(wires)%2 != 0:
        raise AssertionError(f"number of wires must be an even integer; got {len(wires)}")

    if loc%2 != 0:
        raise AssertionError(f"loc must be an even integer; got {loc}")

    layers = np.array(
        [
            [wires[idx] for idx in range(j, j + loc)]
            for j in range(0, len(wires) - loc // 2, loc // 2)
        ]
    )
    return layers


class MPS(Operation):
    r""" Quantum circuit consisting on the broadcast of local gates, following the architecture from `arXiv:1803.11537 <https://arxiv.org/abs/1803.11537>`_.

    The argument ``block`` can be either an user-defined quantum function or an existing template. 

        Args:

            wires (Iterable):  wires that the template acts on
            loc (int): number of wires that each  block acts on
            block: quantum circuit that compose each block
            n_params_block (Integer):
            weights (tensor_like): weight tensor
        """

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
        if loc < 2:
            raise ValueError(f"number of wires in each block must be larger than or equal to 2; got loc={loc}")

        if n_wires < 3:
            raise ValueError(f"fnumber of wires must be greater than or equal to 3; got {n_wires}")

        if loc > n_wires:
            raise ValueError(f"loc must be smaller than or equal to the number of wires; got loc = {loc} and number of wires = {n_wires}")


        shape = qml.math.shape(weights)[-4:]  # (n_params_block, n_blocks)
        self.n_params_block = n_params_block
        self.n_blocks = int(n_wires / (loc / 2) - 1)
        self.block = block

        if weights is None:
            self.weights = np.random.rand(n_params_block, int(self.n_blocks))  

        else:

            if shape[-1] != self.n_blocks:
                raise ValueError(
                    f"Weights tensor must have last dimension of length {self.n_blocks}; got {shape[-1]}"
                )
            if shape[0] != self.n_params_block:
                raise ValueError(
                    f"Weights tensor must have first dimension of length {self.n_params_block}; got {shape[0]}"
                )

            self.weights = weights

        self.ind_gates = compute_indices_MPS(wires, loc)

        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    def expand(self):

        with qml.tape.QuantumTape() as tape:
            for idx, w in enumerate(self.ind_gates):
                self.block(weights=self.weights[..., idx], wires=w.tolist())
                # Different ordering compared to [arXiv:1803.11537v2] -> measurement of the last instead of the first qubit

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

        if n_wires % 2 ==1:
            raise ValueError(f"n_wires should be an even integer; got n_wires = {n_wires}")

        if loc > n_wires:
            raise ValueError(f"loc must be smaller than or equal to the number of wires; got loc = {loc} and number of wires = {n_wires}")

        n_blocks = int(n_wires / (loc / 2) - 1)

        return n_blocks, n_params_block
