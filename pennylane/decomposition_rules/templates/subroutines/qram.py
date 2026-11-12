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

"""Built-in decomposition rules for templates.subroutines.qram."""

# pylint: disable=consider-using-generator

from collections import defaultdict

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import (
    CH,
    CSWAP,
    CZ,
    RY,
    SWAP,
    BasisState,
    Hadamard,
    MultiControlledX,
    PauliX,
    PauliZ,
    adjoint,
    cond,
    ctrl,
)
from pennylane.templates.subroutines.qram import (
    BBQRAM,
    FFQRAM,
    HybridQRAM,
    SelectOnlyQRAM,
    _node_index,
)
from pennylane.typing import Bool, Float, Wire
from pennylane.wires import concatenate_wires


def _bucket_brigade_qram_resources(num_controls, num_target_wires):
    """
    Calculates the resources, assuming the worst case where bitstrings are all ones.
    """
    n_k = num_controls
    _ctrl_swap = ctrl(SWAP(Wire[2]), Wire[1], Bool[1])
    n_address_swaps = (1 << n_k) - 1 + n_k
    n_routing_swaps = ((1 << n_k) - 1) * num_target_wires + (1 << n_k) - 1 - n_k
    return {
        SWAP: n_address_swaps + num_target_wires * 2,
        adjoint(SWAP(Wire[2])): n_address_swaps,
        CSWAP: n_routing_swaps,
        adjoint(CSWAP(Wire[3])): n_routing_swaps,
        _ctrl_swap: n_routing_swaps,
        adjoint(_ctrl_swap): n_routing_swaps,
        Hadamard: num_target_wires * 2,
        PauliZ: (1 << n_k) * num_target_wires,
    }


def _mark_routers_via_bus(wire_manager, n_k):
    """Write low-order address bits into router directions **layer-by-layer** via the bus.

    For each low bit a_k (k = 0..n_k-1):
      1) SWAP(control_wires[k], bus)
      2) Route bus down k levels (CSWAPs controlled by routers at levels < k)
      3) At node (k, path-prefix), SWAP(bus, dir[k, path-prefix])
    """
    SWAP([wire_manager.control_wires[0], wire_manager.bus_wire[0]])
    SWAP([wire_manager.bus_wire[0], wire_manager.router(0, 0)])
    for k in range(1, n_k):
        # 1) load a_k into the bus
        origin = wire_manager.control_wires[k]
        target = wire_manager.bus_wire[0]
        SWAP(wires=[origin, target])
        # 2) route down k levels
        _route_bus_down_first_k_levels(wire_manager, k)
        # 3) deposit at level-k node on the active path
        for p in range(1 << k):
            # change to  in_wire later
            parent = _node_index(k - 1, p >> 1)
            origin = (
                wire_manager.portL_wires[parent] if p % 2 == 0 else wire_manager.portR_wires[parent]
            )
            target = wire_manager.router(k, p)
            SWAP(wires=[origin, target])


def _route_bus_down_first_k_levels(wire_manager, k_levels):
    """Route the bus down the first `k_levels` of the tree using dir-controlled CSWAPs."""
    for ell in range(k_levels):
        for p in range(1 << ell):
            in_w = wire_manager.node_in_wire(ell, p)
            L = wire_manager.portL(ell, p)
            R = wire_manager.portR(ell, p)
            d = wire_manager.router(ell, p)
            # dir==1 ⇒ SWAP(in, R)
            CSWAP(wires=[d, in_w, R])
            # dir==0 ⇒ SWAP(in, L)
            ctrl(SWAP(wires=[in_w, L]), control=[d], control_values=[0])


def _leaf_ops_for_bit(wire_manager, bitstrings, n_k, j):
    """Apply the leaf write for target bit index j."""
    for p in range(1 << n_k):
        if p % 2 == 0:
            target = wire_manager.portL(n_k - 1, p >> 1)
        else:
            target = wire_manager.portR(n_k - 1, p >> 1)
        bit = bitstrings[p, j]
        cond(bit, PauliZ)(wires=target)


@register_resources(_bucket_brigade_qram_resources, exact=False)
def _bucket_brigade_qram_decomposition(
    bitstrings, wire_manager, **__
):  # pylint: disable=unused-argument
    bus_wire = wire_manager.bus_wire
    control_wires = wire_manager.control_wires
    n_k = len(control_wires)
    # 1) address loading
    _mark_routers_via_bus(wire_manager, n_k)
    # 2) For each target bit: load→route down→leaf op→route up→restore (reuse the route bus function)
    for j, tw in enumerate(wire_manager.target_wires):
        Hadamard(wires=[tw])
        SWAP(wires=[tw, bus_wire[0]])
        _route_bus_down_first_k_levels(wire_manager, len(control_wires))
        _leaf_ops_for_bit(wire_manager, bitstrings, n_k, j)
        adjoint(_route_bus_down_first_k_levels)(wire_manager, len(control_wires))
        SWAP(wires=[tw, bus_wire[0]])
        Hadamard(wires=[tw])
    # 3) address unloading
    adjoint(_mark_routers_via_bus)(wire_manager, n_k)


add_decomps(BBQRAM, _bucket_brigade_qram_decomposition)


def _hybrid_qram_resources(num_target_wires, num_select_wires, num_tree_control_wires):
    resources = defaultdict(int)
    num_blocks = 1 << num_select_wires

    resources[PauliX] += (num_select_wires <= 0) * num_blocks * 2

    ctrl_swap = CSWAP(Wire[3])
    n_address_swaps = num_tree_control_wires + (1 << num_tree_control_wires) - 1
    resources[ctrl_swap] += (n_address_swaps + 2 * num_target_wires) * num_blocks
    resources[adjoint(ctrl_swap)] += n_address_swaps * num_blocks

    ccswap_count = (
        (
            ((1 << num_tree_control_wires) - 1 - num_tree_control_wires)
            + ((1 << num_tree_control_wires) - 1) * num_target_wires
        )
        * num_blocks
        * 2
    )

    resources[ctrl(SWAP(Wire[2]), Wire[2])] += ccswap_count // 2
    resources[adjoint(ctrl(SWAP(Wire[2]), Wire[2]))] += ccswap_count // 2

    resources[ctrl(SWAP(Wire[2]), Wire[2], Bool[2])] += ccswap_count // 2
    resources[adjoint(ctrl(SWAP(Wire[2]), Wire[2], Bool[2]))] += ccswap_count // 2

    resources[CH] += num_target_wires * num_blocks * 2

    if num_select_wires > 0:
        cx_rep = MultiControlledX(Wire[num_select_wires + 1], control_values=Bool[num_select_wires])
        resources[cx_rep] += 2 * num_blocks
    resources[CZ] += (1 << num_tree_control_wires) * num_target_wires * num_blocks

    return resources


def _bits(value: int, length: int) -> list[int]:
    """Return `length` bits of `value` (MSB first)."""
    return [(value >> (length - 1 - i)) & 1 for i in range(length)]


def _tree_leaf_ops_for_bit_block_ctrl(
    bitstrings, j, block_index, tree_wire_manager, n_tree, signal
):  # pylint: disable=too-many-arguments
    """Leaf write for target bit j, for a given select prefix block, controlled on signal."""

    # For each leaf index p of the tree (n_tree bits)
    for p in range(1 << n_tree):
        # physical leaf wire (same pattern as BBQRAM)
        if p % 2 == 0:
            target = tree_wire_manager.portL(n_tree - 1, p >> 1)
        else:
            target = tree_wire_manager.portR(n_tree - 1, p >> 1)

        # Global address index: (block_index << n_tree) + p
        addr = (block_index << n_tree) + p
        bit = bitstrings[addr][j]
        cond(bit, CZ)(wires=[signal, target])


def _tree_route_bus_down_first_k_levels_ctrl(k_levels, tree_wire_manager, signal):
    """Tree routing down for first `k_levels` levels, controlled on signal."""

    for ell in range(k_levels):
        for p in range(1 << ell):
            in_w = tree_wire_manager.node_in_wire(ell, p)
            L = tree_wire_manager.portL(ell, p)
            R = tree_wire_manager.portR(ell, p)
            d = tree_wire_manager.router(ell, p)

            # dir==1: CSWAP(d, in_w, R) — additionally controlled on signal
            ctrl(SWAP(wires=[in_w, R]), control=[signal, d], control_values=[1, 1])

            # dir==0: SWAP(in_w, L) controlled on (d == 0) and signal == 1
            ctrl(
                SWAP(wires=[in_w, L]),
                control=[signal, d],
                control_values=[1, 0],
            )


def _swap_controlled_on_signal(tree_wire_manager, signal, level, k):
    origin = tree_wire_manager.control_wires[k:][level]
    target = tree_wire_manager.bus_wire[0]
    CSWAP(wires=[signal, origin, target])


def _tree_mark_routers_via_bus_ctrl(tree_wire_manager, n_tree, k, signal):
    """Address loading for the tree (n_tree bits), controlled on signal."""

    # SWAP(tree_control_wires[0], bus) controlled on signal
    _swap_controlled_on_signal(tree_wire_manager, signal, 0, k)

    # route down qram wires for level 0
    _tree_route_bus_down_first_k_levels_ctrl(0, tree_wire_manager, signal)

    # deposit into dir[0, *] along active path
    CSWAP(wires=[signal, tree_wire_manager.bus_wire[0], tree_wire_manager.router(0, 0)])

    for level in range(1, n_tree):
        # SWAP(tree_control_wires[level], bus) controlled on signal
        _swap_controlled_on_signal(tree_wire_manager, signal, level, k)

        # route down qram wires for current levels
        _tree_route_bus_down_first_k_levels_ctrl(level, tree_wire_manager, signal)

        # deposit into dir[level, *] along active path
        for p in range(1 << level):
            parent = _node_index(level - 1, p >> 1)
            if p % 2 == 0:
                origin = tree_wire_manager.portL_wires[parent]
            else:
                origin = tree_wire_manager.portR_wires[parent]
            target = tree_wire_manager.router(level, p)
            CSWAP(wires=[signal, origin, target])


def _block_tree_query_ops(
    bitstrings, block_index, tree_wire_manager, n_tree, k, signal
):  # pylint: disable=too-many-arguments
    """One BBQRAM-style query of the (n_tree)-depth tree for a fixed select prefix."""

    # 1) address loading for the tree (controlled on signal)
    _tree_mark_routers_via_bus_ctrl(tree_wire_manager, n_tree, k, signal)

    # 2) per-target data phase, controlled on signal
    for j, tw in enumerate(tree_wire_manager.target_wires):
        # H on target
        CH(wires=[signal, tw])

        # Swap target <-> bus
        CSWAP(wires=[signal, tw, tree_wire_manager.bus_wire[0]])

        # Route down tree
        _tree_route_bus_down_first_k_levels_ctrl(n_tree, tree_wire_manager, signal)

        # Leaf Z ops for this block and bit index j
        _tree_leaf_ops_for_bit_block_ctrl(
            bitstrings, j, block_index, tree_wire_manager, n_tree, signal
        )

        # Route back up
        adjoint(_tree_route_bus_down_first_k_levels_ctrl)(n_tree, tree_wire_manager, signal)

        # Swap back bus -> target
        CSWAP(wires=[signal, tw, tree_wire_manager.bus_wire[0]])

        # Final H on target
        CH(wires=[signal, tw])

    # 3) address unloading for the tree (controlled on signal)
    adjoint(_tree_mark_routers_via_bus_ctrl)(tree_wire_manager, n_tree, k, signal)


@register_resources(_hybrid_qram_resources, exact=False)
def _hybrid_qram_decomposition(
    bitstrings, tree_wire_manager, select_wires, signal_wire, **_
):  # pylint: disable=unused-argument, too-many-arguments
    k = len(select_wires)

    signal = signal_wire[0]
    num_blocks = 1 << k if k > 0 else 1

    for block_index in range(num_blocks):
        # Multi-controlled X to turn signal on when select bits == block_index
        if k > 0:
            sel_pattern = _bits(block_index, k)
            mcx_wires = concatenate_wires(select_wires, [signal])
            MultiControlledX(mcx_wires, control_values=sel_pattern)
        else:
            # No select bits: just flip signal for all addresses
            PauliX(wires=signal)

        # Perform one tree query, driven by lower n_tree bits, controlled on signal
        _block_tree_query_ops(
            bitstrings,
            block_index,
            tree_wire_manager,
            len(tree_wire_manager.control_wires[k:]),
            k,
            signal,
        )

        # Uncompute signal
        if k > 0:
            mcx_wires = concatenate_wires(select_wires, [signal])
            MultiControlledX(mcx_wires, control_values=sel_pattern)
        else:
            PauliX(wires=signal)


add_decomps(HybridQRAM, _hybrid_qram_decomposition)


def _select_only_qram_resources(
    select_value, num_control_wires, num_select_wires, num_target_wires
):
    resources = defaultdict(int)
    n_total = num_control_wires + num_select_wires

    if select_value is not None and num_select_wires > 0:
        resources[BasisState(Bool[num_select_wires], Wire[num_select_wires])] += 1

    for addr in range(2 ** (num_select_wires + num_control_wires)):

        if (
            select_value is not None
            and num_select_wires > 0
            and (addr >> num_control_wires) != select_value
        ):
            continue

        control_values = [(addr >> (n_total - 1 - i)) & 1 for i in range(n_total)]
        resources[PauliX] += control_values.count(0) * 2
        x_rep = MultiControlledX(Wire[n_total + 1]) if n_total else PauliX
        resources[x_rep] += num_target_wires

    return resources


def _flip_controls(control_wires, control_vals):
    for i, control_value in enumerate(control_vals):
        if control_value == 0:
            PauliX(control_wires[i])


@register_resources(_select_only_qram_resources, exact=False)
def _select_only_qram_decomposition(
    bitstrings, select_value, select_wires, control_wires, target_wires, **_
):  # pylint: disable=unused-argument, too-many-arguments
    controls = select_wires + control_wires
    num_select = len(select_wires)
    n_total = num_select + len(control_wires)
    gate = MultiControlledX if n_total else PauliX

    if select_value is not None and num_select > 0:
        BasisState(math.int_to_binary(select_value, num_select), wires=select_wires)

    # Loop over all addresses (0 .. 2^(num_select+num_controls)-1)
    for addr, bits in enumerate(bitstrings):
        # If select_value is specified, only implement entries whose
        # high num_select bits (select part) match that value.
        if select_value is not None and num_select > 0:
            sel_part = addr >> (n_total - num_select)
            if sel_part != select_value:
                continue

        control_values = [(addr >> (n_total - 1 - i)) & 1 for i in range(n_total)]

        _flip_controls(controls, control_values)

        # For each bit position in the bitstrings
        for j in range(bitstrings[0].shape[0]):
            # Multi-controlled X on target_wires[j],
            # controlled on controls matching `control_values`.

            cond(bits[j], gate)(wires=concatenate_wires(controls, [target_wires[j]]))

        _flip_controls(controls, control_values)


add_decomps(SelectOnlyQRAM, _select_only_qram_decomposition)


def _flip_zero_bits(address_wires, addr_bits):
    """
    Apply X gates to where the addr_bits is zero.
    """
    for wire, bit in zip(address_wires, addr_bits, strict=True):
        cond(bit == 0, PauliX)(wires=wire)


def _normalize_amplitudes(amplitudes):
    """Normalize along the last axis, supporting optional batching."""
    norm = math.linalg.norm(amplitudes, axis=-1)

    if not math.is_tracer(norm) and math.any(math.isclose(norm, 0.0)):
        raise ValueError("The amplitudes must have a non-zero norm.")

    # Follow StatePrep's pattern to deal with optional batch dimension.
    return amplitudes / math.reshape(norm, (*math.shape(amplitudes)[:-1], 1))


def _ffqram_resources(num_zero_bits, num_address_wires, num_entries):
    """
    - One Hadamard gate per address wire to initialize the |+>^n state.
    - One "flip" and one "flop" for 0 bits.
    - One controlled RY for each entry.
    """
    return {
        Hadamard: num_address_wires,
        PauliX: 2 * num_zero_bits,
        ctrl(RY(Float, Wire[1]), Wire[num_address_wires]): num_entries,
    }


@register_resources(_ffqram_resources)
def _ffqram_decomposition(amplitudes, wires, address, **_):
    address_wires = wires[:-1]
    reg_wire = wires[-1]

    if capture.enabled():
        amplitudes = math.array(amplitudes, like="jax")
        address_wires = math.array(address_wires, like="jax")
        address = math.array(address, like="jax")

    amplitudes = _normalize_amplitudes(amplitudes)
    angles = 2 * math.arcsin(amplitudes)

    # optional batch dimension: align with AngleEmbedding
    batched = math.ndim(angles) > 1
    angles = math.T(angles) if batched else angles

    @for_loop(len(address_wires))
    def superposition_loop(i, wires):
        Hadamard(wires=wires[i])
        return wires

    superposition_loop(address_wires)  # pylint: disable=no-value-for-parameter

    @for_loop(len(address))
    def main_loop(j, addr):
        addr_bits = addr[j]
        # flip
        _flip_zero_bits(address_wires, addr_bits)
        # register
        ctrl(RY(angles[j], wires=reg_wire), control=address_wires)
        # flop (unflip)
        _flip_zero_bits(address_wires, addr_bits)
        return addr

    main_loop(address)  # pylint: disable=no-value-for-parameter


add_decomps(FFQRAM, _ffqram_decomposition)
