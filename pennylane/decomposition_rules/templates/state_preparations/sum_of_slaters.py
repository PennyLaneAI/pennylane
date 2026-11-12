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

"""Built-in decomposition rules for templates.state_preparations.sum_of_slaters."""

from collections import defaultdict

import numpy as np

import pennylane as qp
from pennylane import allocate, for_loop, math
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.templates.state_preparations.sum_of_slaters import (
    SoSData,
    SumOfSlatersPrep,
    _preprocess,
    _sos_encoding_size,
    select_sos_rows,
)
from pennylane.typing import Bool, Wire


# pylint: disable-next=unused-argument
def _sos_state_prep_resources(coefficients, wires, indices, **_):
    """Compute the resources for _sos_state_prep. It is an upper bound due to
    conditionally applied CNOT and X gates."""

    n = 1 if isinstance(wires, int) else len(wires)
    v_bits = math.int_to_binary(np.array(indices), n).T
    selector_ids, _ = select_sos_rows(v_bits)

    num_entries = len(indices)
    num_bits = len(selector_ids)
    num_wires = n

    if num_entries == 1:
        return {qp.BasisState(Bool[num_wires], Wire[num_wires]): 1}
    d = math.ceil_log2(num_entries)
    m = _sos_encoding_size(num_bits, num_entries)

    identity_encoding = num_bits == m

    resources = defaultdict(int)

    # Step 1 in paper (p.7)
    state_type, _ = math.get_abstract_and_cast_dtype(coefficients)
    resources[qp.MultiplexerStatePreparation(state_type[2**d], wires=Wire[d])] += 1

    # Step 2 in paper (p.7)
    resources[
        qp.QROM(
            bitstrings=Bool[num_entries, num_wires],
            control_wires=Wire[d],
            target_wires=Wire[num_wires],
            work_wires=Wire[d - 1],
            clean=True,
        )
    ] += 1

    if not identity_encoding:
        ## Step 3 & 4 in paper (p.7). This is an upper bound
        resources[qp.CNOT] += m * num_wires  # size {u_k} * bits in u_k

    ## Step 5 in paper (p.7)
    resources[qp.TemporaryAND] += (num_entries - 1) * (m - 1)
    resources[_adjoint_abstract(qp.TemporaryAND)] += (num_entries - 1) * (m - 1)

    # Calculate the bit counts of all integers that need to be uncomputed and sum them up.
    number_of_bits_to_unset = int(np.sum(np.bitwise_count(np.arange(1, num_entries))))
    resources[qp.CNOT] += number_of_bits_to_unset

    # We have to flip at most m control bits between any pair of the `num_entries-1` uncomputing
    # MCX groups (skipping 0 because nothing needs to be done) as well as before the first
    # and after the last group. This amounts to `num_entries` layers of bit flips
    resources[qp.X] += num_entries * m

    if not identity_encoding:
        ## Step 6 in paper (p.7). This is an upper bound
        resources[qp.CNOT] += m * num_wires  # size {u_k} * bits in u_k

    return resources


# pylint: disable=unused-argument,too-many-arguments
def _sos_state_prep_work_wires(
    coefficients,
    wires,
    indices,
    enumeration_wires=(),
    identification_wires=(),
    qrom_work_wires=(),
    mcx_cache_wires=(),
    **_,
):
    """See SumOfSlatersPrep.required_register_sizes for details."""
    if indices is None:
        # invalid rule, just return so we dont get an error
        return {}
    # pylint: disable-next=protected-access
    n = 1 if isinstance(wires, int) else len(wires)
    v_bits = math.int_to_binary(np.array(indices), n).T
    selector_ids, _ = select_sos_rows(v_bits)

    num_entries = len(indices)
    num_bits = len(selector_ids)
    num_wires = n

    num_enum, num_id, num_qrom, num_mcx = (
        len(enumeration_wires),
        len(identification_wires),
        len(qrom_work_wires),
        len(mcx_cache_wires),
    )

    # pylint: disable-next=protected-access
    sizes = SumOfSlatersPrep._required_register_sizes_from_nums(num_entries, num_bits, num_wires)
    return {"zeroed": sum(sizes.values()) - num_wires - num_enum - num_id - num_qrom - num_mcx}


def _pad_and_cast_coeffs(coefficients, enumeration_wires):
    """Pad coefficients into a vector of sizes ``2**len(enumeration_wires)``, because
    ``MultiplexerStatePreparation`` expects a full-dimensional input vector.
    Also casts to the dtype consistent with the input dtype of coefficients and the reported
    dtype in the resource function."""
    missing_dim = 2 ** len(enumeration_wires) - len(coefficients)
    _, cast_dtype = math.get_abstract_and_cast_dtype(coefficients)
    zeros = qp.math.cast_like(qp.math.zeros(missing_dim), coefficients)
    interface = qp.math.get_interface(coefficients)
    coefficients = qp.math.concatenate([coefficients, zeros], like=interface)
    return qp.math.cast(coefficients, cast_dtype)


def _sos_state_prep_with_wires(
    data: tuple[np.ndarray, np.ndarray, SoSData],
    *,
    wires,
    enumeration_wires,
    identification_wires,
    qrom_work_wires,
    mcx_cache_wires,
    selected_wires,
):
    # pylint: disable=too-many-arguments, no-value-for-parameter
    coefficients, v_bits, data = data
    identity_encoding = data.r == data.m
    b_bits = data.b_bits
    num_entries = v_bits.shape[1]

    # Step 1: Dense state preparation in enumeration register
    padded_coefficients = _pad_and_cast_coeffs(coefficients, enumeration_wires)
    qp.MultiplexerStatePreparation(padded_coefficients, wires=enumeration_wires)

    # Step 2 in paper (p.7): QROM to load v_bits into system register
    qp.QROM(
        v_bits.T,
        control_wires=enumeration_wires,
        target_wires=wires,
        work_wires=qrom_work_wires,
    )

    # Step 5) in paper (p.7): Use identification register to uncompute the enumeration register
    mcx_ctrl_wires = selected_wires if identity_encoding else identification_wires

    # Create wires for elbow ladder
    elbow_wires = [mcx_ctrl_wires[0]] + [
        _wires[k] for k in range(data.m - 1) for _wires in [mcx_ctrl_wires[1:], mcx_cache_wires]
    ]
    elbow_triples = [elbow_wires[2 * k : 2 * k + 3] for k in range(data.m - 1)]
    cnot_control_wire = elbow_wires[-1]

    if qp.compiler.active() or qp.capture.enabled():
        enumeration_wires = qp.math.array(enumeration_wires, like="jax")
        b_bits = qp.math.array(b_bits, like="jax")
        mcx_ctrl_wires = qp.math.array(mcx_ctrl_wires, like="jax")
        elbow_triples = qp.math.array(elbow_triples, like="jax")

    if not identity_encoding:
        u_bits = data.u_bits
        if qp.compiler.active() or qp.capture.enabled():
            u_bits = qp.math.array(u_bits, like="jax")
            selected_wires = qp.math.array(selected_wires, like="jax")
            identification_wires = qp.math.array(identification_wires, like="jax")

        # Step 3-4) in paper (p.7): Encode the b_bits from Lemma 1 in the identification
        # register. Note that we skip this step if identity_encoding=True, because the encoding
        # is trivial in this case. This is an additional optimization compared to the paper.
        # Across all iterations, one CNOT is applied per set bit in u_bits.
        encoding_cnot = qp.hint(qp.CNOT, {"branch-prob": np.mean(data.u_bits)})

        @for_loop(data.m)
        def encoding(i):
            u = u_bits[i]

            @for_loop(data.r)
            def inner_loop(j):
                qp.cond(u[j], encoding_cnot)([selected_wires[j], identification_wires[i]])

            inner_loop()

        encoding()

    @for_loop(data.m - 1)
    def left_elbow_ladder(i):
        qp.TemporaryAND(elbow_triples[i])

    # Across all k > 0, one CNOT is applied per set bit in k.
    enumeration_cnot = qp.hint(
        qp.CNOT,
        {"branch-prob": np.mean(qp.math.int_to_binary(np.arange(1, num_entries), data.d))},
    )

    @for_loop(data.d)
    def cnot_loop(j, k):
        bit_is_set = (k >> (data.d - 1 - j)) & 1
        qp.cond(bit_is_set, enumeration_cnot)([cnot_control_wire, enumeration_wires[j]])
        return k

    @for_loop(data.m - 2, -1, -1)
    def right_elbow_ladder(i):
        qp.adjoint(qp.TemporaryAND(elbow_triples[i]))

    # Across all k > 0, the flipped bits are data.b_bits[:, k] ^ prev_bits, where prev_bits
    # is data.b_bits[:, k - 1], or all ones for k = 1.
    prev_bits = np.concatenate([np.ones((data.m, 1), dtype=int), data.b_bits[:, 1:-1]], axis=1)
    flip_x = qp.hint(qp.X, {"branch-prob": np.mean(data.b_bits[:, 1:] ^ prev_bits)})

    @for_loop(data.m)
    def flip(i, bits_to_flip):
        qp.cond(bits_to_flip[i], flip_x)(mcx_ctrl_wires[i])
        return bits_to_flip

    # Start the for loop at 1 because we don't need to do anything for 0 anyway
    @for_loop(1, num_entries)
    def uncompute_enumeration(k, prev_bits):
        bits = b_bits[:, k]
        flip(bits ^ prev_bits)
        if data.m > 1:
            left_elbow_ladder()
        cnot_loop(k)
        if data.m > 1:
            right_elbow_ladder()
        return bits

    last_bits = uncompute_enumeration(np.ones(data.m, dtype=int))

    last_flip_x = qp.hint(qp.X, {"branch-prob": np.mean(1 - data.b_bits[:, -1])})

    @for_loop(data.m)
    def last_flip(i):
        qp.cond(1 - last_bits[i], last_flip_x)(mcx_ctrl_wires[i])

    last_flip()

    # Step 6) in paper (p.7): Uncompute the b_i in the identification register (self-adjoint)
    if not identity_encoding:
        encoding()


@register_condition(
    lambda coefficients, wires, indices, *_, **__: indices is not None
)  # indices must be known
@register_resources(_sos_state_prep_resources, exact=False, work_wires=_sos_state_prep_work_wires)
# pylint: disable-next=too-many-arguments
def _sos_state_prep(
    coefficients,
    wires,
    indices,
    enumeration_wires=(),
    identification_wires=(),
    qrom_work_wires=(),
    mcx_cache_wires=(),
):
    """Compute the decomposition of the sum-of-Slaters state preparation technique."""
    n = 1 if isinstance(wires, int) else len(wires)
    num_enum, num_id, num_qrom, num_mcx = (
        len(enumeration_wires),
        len(identification_wires),
        len(qrom_work_wires),
        len(mcx_cache_wires),
    )
    num_entries = len(indices)
    v_bits = math.int_to_binary(np.array(indices), n).T  # Shape (n, num_entries)
    if num_entries == 1:
        # This is only correct up to a global phase that might be contained in coefficients
        qp.BasisState(v_bits[:, 0], wires=wires)
        return
    assert v_bits.shape == (n, num_entries)

    selected_wires, data = _preprocess(v_bits, wires)
    # pylint: disable-next=protected-access
    sizes = SumOfSlatersPrep._required_register_sizes_from_nums(num_entries, data.r, n)
    all_allocate_wires = sum(sizes.values()) - n - num_enum - num_id - num_qrom - num_mcx

    with allocate(all_allocate_wires, state="zero", restored=True) as allocated:
        start = 0
        # There is no implementation of QROM with allocate yet, so we allocate its work wires here.
        # ``allocate(0)`` records nothing when every register was already provided.
        all_wires = {}

        def _allocate_conditionally(wires_arg, name, start):
            if sizes[name] == 0:
                all_wires[name] = []
            if len(wires_arg) > 0:
                all_wires[name] = wires_arg
            else:
                all_wires[name] = allocated[start : (start := start + sizes[name])]
            return start

        start = _allocate_conditionally(enumeration_wires, "enumeration_wires", start)
        start = _allocate_conditionally(identification_wires, "identification_wires", start)
        start = _allocate_conditionally(qrom_work_wires, "qrom_work_wires", start)
        start = _allocate_conditionally(mcx_cache_wires, "mcx_cache_wires", start)

        all_wires["wires"] = wires
        all_wires["selected_wires"] = selected_wires
        data = (coefficients, v_bits, data)
        _sos_state_prep_with_wires(data, **all_wires)  # pylint: disable=missing-kwoa


add_decomps(SumOfSlatersPrep, _sos_state_prep)
