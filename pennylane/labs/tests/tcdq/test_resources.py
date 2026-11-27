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
"""Resource integration and independent ideal-model checks for TCDQ sampling."""

import math
from itertools import product

import pytest

import pennylane as qp
import pennylane.labs.estimator_beta as qre
from pennylane import math as pm
from pennylane.estimator import GateCount, ResourceConfig
from pennylane.labs.estimator_beta.ops.op_math.controlled_ops import (
    mcx_many_clean_aux_resource_decomp,
)
from pennylane.labs.estimator_beta.templates import tcdq_sampling as ts
from pennylane.labs.estimator_beta.templates.state_prep import LabsSumOfSlatersPrep
from pennylane.labs.tcdq import CircuitConfig, QuditCircuitConfig, shot_resources
from pennylane.labs.tcdq.qudit_expval_functions import _parse_qudit_gate_dict
from pennylane.templates.state_preparations.sum_of_slaters import (
    _sos_state_prep_resources,
    select_sos_rows,
)


def _counts(op, config=None):
    return dict(qre.estimate(op, config=config).gate_counts)


def _fourier(dim):
    labels = pm.arange(dim)
    return pm.exp(2j * math.pi * pm.outer(labels, labels) / dim) / math.sqrt(dim)


@pytest.fixture(name="make_config")
def config_factory():
    jax = pytest.importorskip("jax")

    def make(dims, gates, **kwargs):
        common = {"gates": gates, "n_samples": 10, "key": jax.random.PRNGKey(0), **kwargs}
        if isinstance(dims, int):
            return CircuitConfig(n_qubits=dims, **common)
        _parse_qudit_gate_dict(gates, len(dims), dims)
        return QuditCircuitConfig(dims=dims, n_qudits=len(dims), **common)

    return make


@pytest.mark.parametrize("n", [1, 3])
def test_qubit_shot_matches_existing_iqp(make_config, n):
    gates = {weight - 1: [list(range(weight))] for weight in range(1, n + 1)}
    shot = shot_resources(make_config(n, gates))
    assert _counts(shot) == _counts(qre.IQP(n, list(gates.values())))
    sparse = {key: [dict.fromkeys(gate, 1) for gate in group] for key, group in gates.items()}
    qudit = shot_resources(make_config([2] * n, sparse))
    assert _counts(qudit) == _counts(shot)
    assert _counts(qudit)["T"] > 0
    assert "X" not in _counts(qudit)
    assert qre.estimate(qudit).algo_wires == n


@pytest.mark.parametrize("params", [None, [1e-5, 2e-5]])
def test_negative_qubit_indices_merge(make_config, params):
    aliases = shot_resources(make_config(4, {0: [[-1]], 1: [[3]]}), params=params)
    merged = shot_resources(
        make_config(4, {0: [[3]]}), params=None if params is None else [sum(params)]
    )
    assert _counts(aliases) == _counts(merged)


def test_sparse_qudit_parser_order_and_modulo(make_config):
    gates = {0: [{1: 5, 0: -2}], 1: [{0: 1, 1: 1}]}
    group = _parse_qudit_gate_dict(gates, 2, [3, 4])[0]
    assert group.supports.tolist() == [[0, 1], [0, 1]]
    assert group.powers.tolist() == [[1, 1], [1, 1]]
    aliases = shot_resources(make_config([3, 4], gates), params=[1e-5, 2e-5])
    merged = shot_resources(make_config([3, 4], {0: [{0: 1, 1: 1}]}), params=[3e-5])
    assert _counts(aliases) == _counts(merged)


@pytest.mark.parametrize("dims", [3, [3, 4, 5]])
@pytest.mark.parametrize(
    "amps, expected",
    [
        ([1, 1, 1], "uniform_real"),
        ([1, 1, 1 + 1e-12], "arbitrary"),
        ([1, 1, 1 + 1e-12j], "arbitrary"),
        ([1j, 1j, 1j], "arbitrary"),
    ],
)
def test_input_state_classification_is_exact(make_config, dims, amps, expected):
    elems = pm.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])
    gates = {0: [[0]]} if isinstance(dims, int) else {0: [{0: 1}]}
    shot = shot_resources(
        make_config(dims, gates, init_state_elems=elems, init_state_amps=pm.array(amps))
    )
    bits = (
        elems.T
        if isinstance(dims, int)
        else pm.array(
            [
                (elems[:, site] >> bit) & 1
                for site, dim in enumerate(dims)
                for bit in range((dim - 1).bit_length())
            ]
        )
    )
    num_wires = dims if isinstance(dims, int) else sum((dim - 1).bit_length() for dim in dims)
    condensed = (
        qre.UniformStatePrep(3) if expected == "uniform_real" else qre.QROMStatePreparation(2)
    )
    prep = LabsSumOfSlatersPrep(3, num_wires, len(select_sos_rows(bits)[0]), condensed)
    expected_counts = _counts(shot_resources(make_config(dims, gates)))
    for gate, count in _counts(prep).items():
        expected_counts[gate] = expected_counts.get(gate, 0) + count
    assert _counts(shot) == {gate: count for gate, count in expected_counts.items() if count}
    assert qre.estimate(shot).algo_wires == num_wires


def test_merged_parameters_and_explicit_precision(make_config):
    gates = {0: [[0, 1]], 1: [[1, 0], [2]], 2: [[3]]}
    config = make_config(4, gates, init_state_elems=[[1, 0, 1, 0]], init_state_amps=[1])
    params = [1e-5, 2e-5, 0.3]
    shot = shot_resources(config, params=params, precision=1e-6)
    assert ts.iqp_parities_from_gates(gates, params) == pytest.approx(
        {(0, 1): 3e-5, (2,): 2e-5, (3,): 0.3}
    )
    layer = qre.ParityPhaseLayer(
        4, ((1, 2), (2, 1)), precision=1e-6, gates_params=[2e-5, 3e-5, 0.3]
    )
    assert _counts(shot) == _counts(qre.Prod([(qre.X(), 2), (qre.Hadamard(), 8), layer]))
    assert _counts(shot_resources(config))["T"] > _counts(shot)["T"]


@pytest.mark.parametrize(
    "state", [{"init_state_amps": [1, 1]}, {"init_state_elems": [[0, 0], [1, 1]]}]
)
def test_incomplete_input_state_uses_default(make_config, state):
    assert _counts(shot_resources(make_config(2, {0: [[0]]}, **state))) == _counts(
        shot_resources(make_config(2, {0: [[0]]}))
    )


@pytest.mark.parametrize("dims, gates", [(4, {0: [[0]]}), ([3, 4], {0: [{0: 1}]})])
def test_explicit_phase_cost_and_queue(make_config, dims, gates):
    config = make_config(dims, gates, phase_fn=lambda params, labels: params[0] * labels.sum())
    with pytest.raises(ValueError, match="phase_op"):
        shot_resources(config)
    phase = qre.SumPhaseLayer([2, 2, 2, 2], degree=2)
    base = shot_resources(make_config(dims, gates))
    shot = shot_resources(config, phase_op=phase)
    expected = _counts(base)
    for gate, count in _counts(phase).items():
        expected[gate] = expected.get(gate, 0) + count
    assert _counts(shot) == expected

    def circuit():
        shot_resources(config, phase_op=qre.SumPhaseLayer([2, 2, 2, 2], degree=2))

    assert dict(qre.estimate(circuit)().gate_counts) == expected


def test_invalid_config():
    with pytest.raises(TypeError, match="CircuitConfig"):
        shot_resources({"gates": {}})


def _layer_count(angles, precision=None, config=None):
    layer = qre.ParityPhaseLayer(1, ((1, len(angles)),), precision, gates_params=angles)
    return _counts(layer, config).get("T", 0)


@pytest.mark.parametrize(
    "theta, precision, expected",
    [
        (1e-5, 1e-9, 9.066),
        (1e-6, 1e-9, 0.0892),
        (1e-3, 1e-6, 26.41),
        (1e-4, 1e-6, 0.6586),
        (1e-2, 1e-3, 3.244),
    ],
)
def test_mixed_synthesis_published_values(theta, precision, expected):
    """Rounded aggregate estimates reproduce the values reported in arXiv:2605.31544."""
    assert _layer_count([theta, -theta] * 500, precision) == pytest.approx(
        1000 * expected, rel=1e-3, abs=1
    )


@pytest.mark.parametrize("precision", [1e-12, 1e-6, 1e-3])
def test_mixed_synthesis_limits(precision):
    angles = pm.linspace(-4, 4, 1000)
    assert _layer_count([0.4 * precision] * 1000, precision) == 0
    count = _layer_count(angles, precision)
    assert count <= 1000 * _counts(qre.RZ(precision=precision))["T"]
    assert _layer_count(angles + math.pi / 4, precision) == count
    assert _layer_count([0, math.pi / 4, -math.pi / 2, math.pi / 8], precision) == 1


@pytest.mark.parametrize("theta, precision", [(math.pi / 8, 1e-9), (0.385, 1e-2)])
def test_staircase_boundary(theta, precision):
    assert _layer_count([theta] * 1000, precision) == pytest.approx(
        1000 * math.sqrt(2) * math.sin(2 * theta), abs=1
    )


@pytest.mark.parametrize(
    "target, copies, precision", [(0, 1, 1e-8), (1, 2, 0.1), (1000, 2000, 0.1)]
)
def test_positive_expected_cost_rounds_up(target, copies, precision):
    """Positive costs, including a tiny excess over an integer, must not be erased."""
    theta = 1e-8 if target == 0 else math.asin((target + 2e-7) / copies / math.sqrt(2)) / 2
    assert _layer_count([theta] * copies, precision) == target + 1


@pytest.mark.parametrize("copies", [1, 1000])
def test_exact_t_rotations_do_not_overround(copies):
    assert _layer_count([math.pi / 8] * copies, 1e-9) == copies


@pytest.mark.parametrize("operator", [qre.RZ, qre.MultiRZ, qre.ParityPhaseLayer])
def test_configured_precision_and_cache_isolation(operator):
    loose, tight = qre.LabsResourceConfig(), qre.LabsResourceConfig()
    loose.set_precision(operator, 1e-6)
    tight.set_precision(operator, 1e-9)
    angles = [1e-5] * 1000
    layer = qre.ParityPhaseLayer(1, ((1, 1000),), gates_params=angles)
    assert _counts(layer, loose)["T"] == _layer_count(angles, 1e-6) == 4
    assert _counts(layer, tight)["T"] == _layer_count(angles, 1e-9) == 9067
    loose.set_precision(operator, 1e-9)
    assert _counts(layer, loose)["T"] == 9067
    assert _layer_count(angles, 1e-6, config=tight) == 4


@pytest.mark.parametrize("operator", [qre.RZ, qre.MultiRZ, qre.ParityPhaseLayer])
def test_custom_rotation_decomposition_takes_precedence(operator):
    def custom_rotation(num_wires=1, **_params):
        return [GateCount(qre.T.resource_rep(), 7 * num_wires)]

    config = qre.LabsResourceConfig()
    config.set_decomp(operator, custom_rotation)
    layer = qre.ParityPhaseLayer(2, ((1, 1), (2, 2)), gates_params=[1e-9] * 3)
    expected = {
        qre.RZ: {"CNOT": 4, "T": 21},
        qre.MultiRZ: {"T": 35},
        qre.ParityPhaseLayer: {"T": 14},
    }[operator]
    assert _counts(layer, config) == expected


def test_plain_resource_config_uses_ordinary_rotations():
    layer = qre.ParityPhaseLayer(2, ((2, 3),), gates_params=[1e-9] * 3)
    plain = ResourceConfig()
    plain.set_precision(qre.RZ, 1e-6)
    expected = _counts(qre.Prod([(qre.MultiRZ(2), 3)]), plain)
    assert _counts(layer, plain) == expected
    assert dict(qp.estimator.estimate(layer, config=plain).gate_counts) == expected


def test_parity_ladder_counts_and_validation():
    assert _counts(qre.ParityPhaseLayer(4, ((1, 4), (2, 6)), gates_params=[1e-5] * 10)) == {
        "CNOT": 12,
        "T": 91,
    }
    with pytest.raises(ValueError, match="Parity weight"):
        qre.ParityPhaseLayer(3, ((4, 1),))
    with pytest.raises(ValueError, match="angles"):
        qre.ParityPhaseLayer(3, ((1, 2),), gates_params=[0.1])


def test_transformed_parity_layers():
    layer = qre.ParityPhaseLayer(3, ((1, 3),), gates_params=[math.pi / 4] * 3)
    assert _counts(qre.Controlled(layer, 1, 0)) == {"CNOT": 6, "T": 6}
    controlled = qre.estimate(qre.Controlled(layer, 2, 1))
    assert controlled.gate_counts["Toffoli"] == 2
    assert controlled.zeroed_wires == 1
    assert controlled.algo_wires == 5
    assert _counts(qre.Adjoint(layer)) == _counts(layer)
    t_layer = qre.ParityPhaseLayer(1, ((1, 2),), gates_params=[math.pi / 8] * 2)
    assert _counts(qre.Pow(t_layer, 2)).get("T", 0) == 0
    target = qp.matrix(qp.ctrl(qp.MultiRZ(-0.74, [1, 2]), control=0))
    half = qp.matrix(qp.MultiRZ(-0.37, [1, 2]), wire_order=range(3))
    both = qp.matrix(qp.MultiRZ(0.37, [0, 1, 2]), wire_order=range(3))
    assert pm.allclose(target, half @ both, rtol=0, atol=1e-12)


@pytest.mark.parametrize("n", [1, 3, 8, 100])
@pytest.mark.parametrize("degree", [1, 3, None])
def test_sum_phase_qubit_cost(n, degree):
    res = qre.estimate(qre.SumPhaseLayer([2] * n, degree=degree))
    bits = n.bit_length()
    rotations = sum(math.comb(bits, weight) for weight in range(1, min(degree or bits, bits) + 1))
    adders = n - n.bit_count()
    assert res.gate_counts.get("Toffoli", 0) == adders
    assert res.gate_counts["T"] == rotations * _counts(qre.RZ())["T"]
    assert res.algo_wires == n
    assert res.zeroed_wires == adders


@pytest.mark.parametrize(
    "dims, adders", [([4] * 95, 185), ([16] * 36, 140), ([3] * 5 + [5] * 2, 12)]
)
def test_sum_phase_qudit_cost(dims, adders):
    res = qre.estimate(qre.SumPhaseLayer(dims))
    input_bits = sum((dim - 1).bit_length() for dim in dims)
    bits = sum(2 ** (dim - 1).bit_length() - 1 for dim in dims).bit_length()
    assert res.algo_wires == input_bits
    assert res.zeroed_wires == res.gate_counts["Toffoli"] == adders
    assert res.gate_counts["T"] == (2**bits - 1) * _counts(qre.RZ())["T"]
    with pytest.raises(ValueError, match="degree"):
        qre.SumPhaseLayer(dims, degree=0)


@pytest.mark.parametrize("bits, degree", [(4, 1), (5, 2), (7, 3)])
def test_sum_polynomial_has_bounded_walsh_degree(bits, degree):
    """Independent algebraic check underlying the degree-limited rotation count."""
    for mask in range(1, 2**bits):
        if mask.bit_count() > degree:
            assert (
                sum((-1) ** (mask & label).bit_count() * label**degree for label in range(2**bits))
                == 0
            )


def test_input_preparation_matches_existing_template(make_config):
    indices, n = pm.array([0, 1, 3, 7, 12, 18, 25, 31]), 5
    elems = pm.int_to_binary(indices, n)
    num_bits = len(select_sos_rows(elems.T)[0])
    config = make_config(
        n, {0: [[0]]}, init_state_elems=elems, init_state_amps=pm.ones(len(indices))
    )
    resource_config = qre.LabsResourceConfig()
    resource_config.set_decomp(qre.MultiControlledX, mcx_many_clean_aux_resource_decomp)
    counts = _counts(shot_resources(config), resource_config)
    prep = _counts(qre.UniformStatePrep(len(indices)), resource_config)
    qrom = _counts(qre.LabsQROM(len(indices), n, borrow_qubits=False), resource_config)
    step5 = counts["Toffoli"] - prep.get("Toffoli", 0) - qrom.get("Toffoli", 0)
    template = _sos_state_prep_resources(pm.ones(len(indices)), list(range(n)), indices)
    assert step5 == template[qp.TemporaryAND] == (len(indices) - 1) * (num_bits - 1)


@pytest.mark.parametrize("dim", [2, 3, 4, 5, 6, 8, 16])
def test_site_walsh_spectrum(dim):
    for power in range(1, dim):
        spectrum = ts.qudit_site_spectrum(dim, power)
        for label in range(dim):
            actual = sum(
                value * (-1) ** (mask & label).bit_count() for mask, value in spectrum.items()
            )
            expected = math.sqrt(2) * math.cos(2 * math.pi * power * label / dim + math.pi / 4)
            assert actual == pytest.approx(expected, rel=0, abs=1e-12)


@pytest.mark.parametrize("dim", [6, 10, 12, 20])
def test_good_thomas_fourier_identity(dim):
    """Check the exact CRT identity used to price composite Fourier transforms."""
    even = dim & -dim
    odd = dim // even
    incoming, outgoing = pm.zeros((dim, dim)), pm.zeros((dim, dim))
    for left, right in product(range(even), range(odd)):
        incoming[(odd * left + even * right) % dim, left * odd + right] = 1
        outgoing[
            (odd * pow(odd, -1, even) * left + even * pow(even, -1, odd) * right) % dim,
            left * odd + right,
        ] = 1
    compiled = outgoing @ pm.kron(_fourier(even), _fourier(odd)) @ incoming.T
    assert pm.allclose(compiled, _fourier(dim), rtol=0, atol=1e-12)


@pytest.mark.parametrize("dim", [2, 3, 4, 6])
def test_fourier_cost_branches(dim):
    if dim == 6:
        counts = _counts(ts.fourier_resources(dim))
        assert counts["T"] == 660
        assert counts["Toffoli"] == 10
        assert counts["T"] < _counts(qre.QubitUnitary(3))["T"]
    else:
        expected = {2: qre.Hadamard(), 3: qre.QubitUnitary(2), 4: qre.QFT(2)}[dim]
        assert _counts(ts.fourier_resources(dim)) == _counts(expected)


def test_manual_qubit_compilation_matches_dense_model():
    """An ideal parity compilation matches H D H; this is not a synthesized-circuit test."""
    n, params = 4, [0.17, -0.23, 0.41, 0.09]
    gates = {0: [[0, 1]], 1: [[2]], 2: [[1, 2, 3], [0]], 3: [[1, 0]]}
    labels = pm.array(list(product([0, 1], repeat=n)))
    phase = sum(
        params[j] * (-1.0) ** pm.sum(labels[:, gate], axis=1)
        for j, gate_list in gates.items()
        for gate in gate_list
    )
    initial = pm.zeros(2**n, dtype=complex)
    initial[[6, 9, 15]] = pm.array([1, 2j, -1]) / math.sqrt(6)
    hadamards = qp.matrix(qp.Hadamard(0))
    for _ in range(n - 1):
        hadamards = pm.kron(hadamards, qp.matrix(qp.Hadamard(0)))
    expected = pm.abs(hadamards @ (pm.exp(1j * phase) * (hadamards @ initial))) ** 2
    parities = ts.iqp_parities_from_gates(gates, params)

    @qp.qnode(qp.device("default.qubit", wires=n))
    def circuit():
        qp.StatePrep(initial, wires=range(n))
        for wire in range(n):
            qp.Hadamard(wire)
        for support, angle in parities.items():
            qp.MultiRZ(-2 * angle, wires=support)
        for wire in range(n):
            qp.Hadamard(wire)
        return qp.probs()

    assert pm.allclose(circuit(), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("dims", [[4, 4], [3, 4], [8, 2], [5, 3]])
def test_manual_qudit_compilation_matches_dense_model(dims):
    """Check the binary Walsh encoding against F† D F in the original qudit space."""
    params, gates = [0.17, -0.23, 0.41], {0: [{0: 1}], 1: [{1: 1}], 2: [{0: 1, 1: 1}, {0: 1}]}
    labels = pm.array(list(product(*[range(dim) for dim in dims])))
    phase = pm.zeros(len(labels))
    for j, gate_list in gates.items():
        for gate in gate_list:
            values = pm.ones(len(labels))
            for site, power in gate.items():
                if power:
                    values *= math.sqrt(2) * pm.cos(
                        2 * math.pi * power * labels[:, site] / dims[site] + math.pi / 4
                    )
            phase += params[j] * values
    fourier = pm.kron(*[_fourier(dim) for dim in dims])
    initial = pm.zeros(len(labels), dtype=complex)
    initial[0] = 1
    expected = pm.abs(fourier.conj().T @ (pm.exp(1j * phase) * (fourier @ initial))) ** 2
    bits = [(dim - 1).bit_length() for dim in dims]
    offsets = [0, bits[0], sum(bits)]
    blocks = []
    for dim, width in zip(dims, bits):
        block = pm.eye(2**width, dtype=complex)
        block[:dim, :dim] = _fourier(dim)
        blocks.append(block)
    parities = ts.qudit_parities_from_gates(gates, dims, params)

    @qp.qnode(qp.device("default.qubit", wires=sum(bits)))
    def circuit():
        for site, block in enumerate(blocks):
            qp.QubitUnitary(block, wires=range(offsets[site], offsets[site + 1]))
        for key, angle in parities.items():
            wires = [
                offsets[site] + bits[site] - 1 - bit
                for site, mask in key
                for bit in range(bits[site])
                if mask >> bit & 1
            ]
            qp.MultiRZ(-2 * angle, wires=wires)
        for site, block in enumerate(blocks):
            qp.adjoint(qp.QubitUnitary)(block, wires=range(offsets[site], offsets[site + 1]))
        return qp.probs()

    probabilities = pm.reshape(circuit(), [2**width for width in bits])
    probabilities = probabilities[tuple(slice(dim) for dim in dims)].reshape(-1)
    assert pm.sum(probabilities) == pytest.approx(1, rel=0, abs=1e-12)
    assert pm.allclose(probabilities, expected, rtol=0, atol=1e-12)
