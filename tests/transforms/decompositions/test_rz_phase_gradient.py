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

"""Tests for ``qp.transforms.decompositions.make_rz_to_phase_gradient_decomp``"""

# pylint: disable=no-value-for-parameter,disable=too-many-arguments

import numpy as np
import pytest

import pennylane as qp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.transforms.decompositions import (
    make_rz_to_phase_gradient_decomp,
    validate_phase_gradient_wires,
)
from pennylane.wires import WireError


def _msb_anchored_width(phi, p):
    """MSB-anchored significant width of ``phi`` at ``p`` bits (0 if it rounds to zero)."""
    bits = [int(b) for b in qp.math.binary_decimals(phi, p, unit=2 * np.pi)]
    set_positions = [i for i, b in enumerate(bits) if b]
    return (max(set_positions) + 1) if set_positions else 0


def _num_set_bits(phi, p):
    """Number of set bits in the ``p``-bit binary representation of ``phi`` (in units of 2*pi)."""
    bits = [int(b) for b in qp.math.binary_decimals(phi, p, unit=2 * np.pi)]
    return sum(bits)


def _full_rz_specs(phi, p):
    """Full-width (``adaptive_precision=False``) specs.

    The compute/uncompute fanout is a ``MultiX`` (one ``X`` per set bit) controlled by the ``RZ``
    target wire, realizing one ``CNOT`` per set bit on each of the two (compute and uncompute)
    passes (a single-control ``MultiX`` fanout that's always controlled on |1>, so it renders
    directly as ``CNOT``), so the count is ``2 * num_set_bits``. Trailing zero bits emit no gate,
    hence ``CNOT`` is absent for an angle that rounds to zero."""
    specs = {"GlobalPhase": 1, "SemiAdder": 1}
    n = _num_set_bits(phi, p)
    if n:
        specs["CNOT"] = 2 * n
    return specs


def _expected_rz_specs(phi, p, adaptive_precision):
    """Expected specs for a single ``RZ``.

    With ``adaptive_precision=False`` the full-width circuit is always emitted. With
    ``adaptive_precision=True`` only trailing zero bits are dropped (narrowing the ``SemiAdder``), so
    the set-bit count - and thus the ``MultiControlledX`` count - is unchanged; an angle that rounds to zero
    emits just the aggregate ``GlobalPhase`` (kept because it uses the true angle and is required to
    match the full-precision result)."""
    if adaptive_precision and _msb_anchored_width(phi, p) == 0:
        return {"GlobalPhase": 1}
    return _full_rz_specs(phi, p)


@pytest.mark.parametrize(
    "n_angle_wires, n_phase_grad_wires, n_work_wires, msg_match",
    [
        [5, 3, 2, "angle_wires and phase_grad wires must be of same size"],
        [3, 4, 2, "angle_wires and phase_grad wires must be of same size"],
        [4, 4, 2, r"work_wires need to be at least of size len\(phase_grad_wires\) - 1"],
    ],
)
def test_validate_phase_gradient_wires(n_angle_wires, n_phase_grad_wires, n_work_wires, msg_match):
    """Test that a WireError is raised correctly for wrongly-sized registers"""
    angle_wires = qp.wires.Wires([f"ang_{i}" for i in range(n_angle_wires)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(n_phase_grad_wires)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(n_work_wires)])

    with pytest.raises(WireError, match=msg_match):
        _ = validate_phase_gradient_wires(angle_wires, phase_grad_wires, work_wires)


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("adaptive_precision", [True, False])
@pytest.mark.parametrize("p", [2, 3, 4])
def test_valid_decomp(p, adaptive_precision):
    """Test that ``make_rz_to_phase_gradient_decomp`` yields a valid decomposition, with capture
    both disabled (concrete angle) and enabled (abstract angle).

    ``_test_decomposition_rule`` checks the emitted circuit against the rule's full-precision
    resource estimate. We use an all-ones angle so the concrete decomposition saturates that
    estimate (every bit is set, so there are no trailing zeros to truncate); other angles emit a
    smaller circuit than the (upper-bound) estimate and are instead covered by
    ``test_as_fixed_decomps``/``test_as_alt_decomps`` (counts) and ``test_adaptive_truncation``
    (statevector exactness)."""

    phi = (1 - 2.0**-p) * 2 * np.pi  # binary all-ones at p bits
    first_free = 1
    angle_wires = list(range(first_free, first_free + p))
    phase_grad_wires = list(range(first_free + p, first_free + 2 * p))
    work_wires = list(range(first_free + 2 * p, first_free + 3 * p - 1))

    custom_decomp = make_rz_to_phase_gradient_decomp(
        angle_wires, phase_grad_wires, work_wires, adaptive_precision=adaptive_precision
    )
    _test_decomposition_rule(qp.RZ(phi, 0), custom_decomp, skip_decomp_matrix_check=True)


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize("adaptive_precision", [True, False])
@pytest.mark.parametrize("phi", [0.5, 0.3, 1 / 2 + 1 / 4 + 1 / 8, 1.0])
@pytest.mark.parametrize("p", [2, 3, 4])
def test_as_fixed_decomps(phi, p, adaptive_precision):
    """Test that the decomposition rule from make_rz_to_phase_gradient_decomp works as expected
    as a fixed decomposition and yields the correct resources. With ``adaptive_precision=False``
    the full-width circuit is always emitted regardless of the angle's significant width."""
    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(p)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(p)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(p - 1)])

    kwargs = {
        "angle_wires": angle_wires,
        "phase_grad_wires": phase_grad_wires,
        "work_wires": work_wires,
    }

    custom_decomp = make_rz_to_phase_gradient_decomp(
        **kwargs, adaptive_precision=adaptive_precision
    )
    gate_set = {"SemiAdder", "CNOT", "PauliX", "GlobalPhase"}

    @qp.transforms.decompose(gate_set=gate_set, fixed_decomps={qp.RZ: custom_decomp})
    @qp.qnode(qp.device("null.qubit"))
    def circuit():
        qp.RZ(phi, 0)
        return qp.state()

    expected = _expected_rz_specs(phi, p, adaptive_precision)
    specs = qp.specs(circuit)()["resources"].quantum_operations
    assert specs == expected


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize("adaptive_precision", [True, False])
@pytest.mark.parametrize("phi", [0.5, 0.3, 1 / 2 + 1 / 4 + 1 / 8, 1.0])
@pytest.mark.parametrize("p", [2, 3, 4])
def test_as_alt_decomps(phi, p, adaptive_precision):
    """Test that the decomposition rule from ``make_rz_to_phase_gradient_decomp works`` as
    expected as an alternative decomposition and yields the correct resources. With
    ``adaptive_precision=False`` the full-width circuit is always emitted."""
    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(p)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(p)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(p - 1)])

    kwargs = {
        "angle_wires": angle_wires,
        "phase_grad_wires": phase_grad_wires,
        "work_wires": work_wires,
    }

    custom_decomp = make_rz_to_phase_gradient_decomp(
        **kwargs, adaptive_precision=adaptive_precision
    )
    gate_set = {"SemiAdder", "CNOT", "PauliX", "GlobalPhase"}

    @qp.transforms.decompose(gate_set=gate_set, alt_decomps={qp.RZ: [custom_decomp]})
    @qp.qnode(qp.device("null.qubit"))
    def circuit():
        qp.RZ(phi, 0)
        return qp.state()

    expected = _expected_rz_specs(phi, p, adaptive_precision)
    specs = qp.specs(circuit)()["resources"].quantum_operations
    assert specs == expected


@pytest.mark.usefixtures("enable_graph_decomposition")
def test_integration_multi_wire(seed):
    """
    Tests that the decomposition correctly realizes the phase gradient decomposition of RZ as described in
    https://pennylane.ai/compilation/phase-gradient/b-rotations
    """

    prec = 3

    phi = (1 / 2 + 0 / 4 + 1 / 8) * 4 * np.pi
    wires = [0]

    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(prec - 1)])

    phase_grad_state = np.exp(-1j * 2 * np.pi * np.arange(2**3) / 2**3) / np.sqrt(2**3)

    all_wires = angle_wires + phase_grad_wires + work_wires + wires

    custom_decomp = make_rz_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires)

    @qp.transforms.decompose(
        gate_set={
            "StatePrep",
            "Adjoint(StatePrep)",
            "SemiAdder",
            "CNOT",
            "PauliX",
            "GlobalPhase",
        },
        fixed_decomps={qp.RZ: custom_decomp},
    )
    @qp.qnode(qp.device("default.qubit", wires=all_wires))
    def circuit(phi, in_state):
        qp.StatePrep(in_state, wires=wires)  # input state
        qp.StatePrep(phase_grad_state, wires=phase_grad_wires)  # phase gradient state
        qp.RZ(phi, wires)
        qp.adjoint(
            qp.StatePrep(phase_grad_state, wires=phase_grad_wires)
        )  # uncompute phase gradient state
        return qp.state()

    # random input state
    rng = np.random.default_rng(seed=seed)
    in_state = rng.random(2 ** len(wires))
    in_state /= np.linalg.norm(in_state)

    # returned output state
    out_state = circuit(phi, in_state)

    # expected output state
    zeros = np.eye(2 ** (prec * 3 - 1), 1)[:, 0]  # |000> on all the aux wires
    out_state_expected = qp.matrix(qp.RZ(phi, wires)) @ in_state
    out_state_expected = np.kron(zeros, out_state_expected)

    assert np.allclose(out_state, out_state_expected)


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize("frac", [0.0, 0.5, 0.25, 0.75, 0.125, 0.875])
def test_adaptive_truncation(frac):
    """A concrete angle is truncated to its MSB-anchored width, yet the decomposition stays exact:
    for angles that are exactly representable at ``prec`` bits it reproduces the ideal ``RZ``,
    including angles with trailing zeros (truncated adder) and the zero angle (identity)."""
    prec = 4
    phi = frac * 2 * np.pi

    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(prec)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(prec - 1)])
    sys = qp.wires.Wires([0])
    all_wires = angle_wires + phase_grad_wires + work_wires + sys

    custom_decomp = make_rz_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires)
    phase_grad_state = np.exp(-1j * 2 * np.pi * np.arange(2**prec) / 2**prec) / np.sqrt(2**prec)

    @qp.transforms.decompose(
        gate_set={
            "StatePrep",
            "Adjoint(StatePrep)",
            "SemiAdder",
            "CNOT",
            "PauliX",
            "GlobalPhase",
        },
        fixed_decomps={qp.RZ: custom_decomp},
    )
    @qp.qnode(qp.device("default.qubit", wires=all_wires))
    def circuit(in_state):
        qp.StatePrep(in_state, wires=sys)
        qp.StatePrep(phase_grad_state, wires=phase_grad_wires)
        qp.RZ(phi, sys)
        qp.adjoint(qp.StatePrep(phase_grad_state, wires=phase_grad_wires))
        return qp.state()

    rng = np.random.default_rng(0)
    in_state = rng.random(2)
    in_state /= np.linalg.norm(in_state)

    out_state = circuit(in_state)
    zeros = np.eye(2 ** (len(all_wires) - 1), 1)[:, 0]
    out_state_expected = np.kron(zeros, qp.matrix(qp.RZ(phi, sys)) @ in_state)
    assert np.allclose(out_state, out_state_expected)
