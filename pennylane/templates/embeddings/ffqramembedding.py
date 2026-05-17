r"""
Contains the ``FFQRAMEmbedding`` template.
"""

from pennylane import math
from pennylane.decomposition import (
    add_decomps,
    controlled_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation
from pennylane.ops import Hadamard, PauliX, RY, ctrl
from pennylane.wires import Wires


class FFQRAMEmbedding(Operation):
    r"""
    Flip-flop QRAM embedding draft with explicit sparse address list.
    """

    grad_method = None

    resource_keys = {"address", "num_address_wires", "num_entries"}

    def _flatten(self):
        hyperparameters = (("address", self._address),)
        return self.data, (self.wires, hyperparameters)

    def __repr__(self):
        return (
            f"FFQRAMEmbedding({self.data[0]}, "
            f"wires={self.wires.tolist()}, "
            f"address={self._address})"
        )

    def __init__(self, amplitudes, wires, address):
        wires = Wires(wires)

        # use same input format as QROM
        if isinstance(address[0], str):
            address = math.array(
                list(map(lambda bitstring: [int(bit) for bit in bitstring], address))
            )

        if isinstance(address, (list, tuple)):
            address = math.array(address)

        num_address_wires = len(wires) - 1

        num_entries = math.shape(amplitudes)[-1]
        assert num_entries == len(address)
        assert num_entries <= 2**num_address_wires

        self._address = address

        self._hyperparameters = {
            "address": address,
        }

        super().__init__(amplitudes, wires=wires)

    @property
    def resource_params(self):
        return {
            "address": self._address,
            "num_address_wires": len(self.wires) - 1,
            "num_entries": len(self._address),
        }

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (1,)

    @staticmethod
    def _normalize_amplitudes(amplitudes):
        """Normalize along the last axis, supporting optional batching."""
        batched = math.ndim(amplitudes) > 1

        if batched:
            norm = math.sqrt(math.sum(amplitudes**2, axis=-1))
            return amplitudes / math.expand_dims(norm, axis=-1)

        norm = math.sqrt(math.sum(amplitudes**2))
        return amplitudes / norm

def _flip_zero_bits(address_wires, addr_bits):
    """
    Apply X gates to where the addr_bits is zero.
    """
    for wire, bit in zip(address_wires, addr_bits):
        if int(bit) == 0:
            PauliX(wires=wire)

def _ffqram_embedding_resources(address, num_address_wires, num_entries):
    resources = defaultdict(int)

    # Hadamard gates to initialize the |+>^n state.
    resources[resource_rep(Hadamard)] += num_address_wires

    for i in range(num_entries):
        addr_bits = address[i]
        num_zero_bits = sum(int(bit) == 0 for bit in addr_bits)
        # one "flip" and one "flop" for 0 bits
        resources[resource_rep(PauliX)] += 2 * num_zero_bits

        # controlled RY for each entry
        resources[
            controlled_resource_rep(
                base_class=RY,
                base_params={},
                num_control_wires=num_address_wires,
                num_zero_control_values=0,
            )
        ] += 1

    return resources

@register_resources(_ffqram_embedding_resources)
def _ffqram_embedding_decomposition(amplitudes, wires, address, **_):
    address_wires = wires[:-1]
    reg_wire = wires[-1]

    amplitudes = FFQRAMEmbedding._normalize_amplitudes(amplitudes)
    angles = 2 * math.arcsin(amplitudes)

    # optional batch dimension: align with AngleEmbedding
    batched = math.ndim(angles) > 1
    angles = math.T(angles) if batched else angles

    # Prepare initial state |+>^n
    for wire in address_wires:
        Hadamard(wires=wire)

    for i, addr_bits in enumerate(address):
        # flip
        _flip_zero_bits(address_wires, addr_bits)
        # register
        ctrl(RY(angles[i], wires=reg_wire), control=address_wires)
        # flop (unflip)
        _flip_zero_bits(address_wires, addr_bits)


add_decomps(FFQRAMEmbedding, _ffqram_embedding_decomposition)