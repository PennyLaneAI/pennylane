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
"""
Tests for the resource operators of one computational-basis shot of the tcdq circuits.
"""

import math
from itertools import product

import numpy as np
import pytest

import pennylane as qp
import pennylane.labs.estimator_beta as qre
from pennylane.estimator import GateCount, resource_rep
from pennylane.labs.estimator_beta.ops.op_math.controlled_ops import (
    mcx_many_clean_aux_resource_decomp,
)
from pennylane.labs.estimator_beta.templates import tcdq_sampling as ts
from pennylane.labs.estimator_beta.templates.state_prep import LabsSumOfSlatersPrep
from pennylane.labs.tcdq import create_local_gates, create_random_gates
from pennylane.templates.state_preparations.sum_of_slaters import (
    _sos_state_prep_resources,
    select_sos_rows,
)

# pylint: disable=no-self-use, too-many-arguments, too-many-locals


def _counts(op, config=None):
    res = qre.estimate(op, config=config)
    return res, dict(res.gate_counts)


def _random_qudit_gates(dims, num_gates, max_weight, rng):
    gates = {}
    for j in range(num_gates):
        w = int(rng.integers(1, max_weight + 1))
        sites = rng.choice(len(dims), size=w, replace=False)
        g = [0] * len(dims)
        for q in sites:
            g[q] = int(rng.integers(1, dims[q]))
        gates[j] = [g]
    return gates


def _all_qudit_gates(dims, max_weight):
    gates, j = {}, 0
    for g in product(*[range(d) for d in dims]):
        if 0 < sum(v != 0 for v in g) <= max_weight:
            gates[j] = [list(g)]
            j += 1
    return gates


class TestParityHelpers:
    """Classical gate-set -> parity-histogram helpers."""

    @pytest.mark.parametrize("n, w", [(5, 1), (6, 2), (7, 3), (4, 4)])
    def test_all_gates_matches_create_local_gates(self, n, w):
        """The closed form equals the histogram of the explicit tcdq gate dictionary."""
        explicit = qre.iqp_parity_weights_from_gates(create_local_gates(n, w))
        assert ts.iqp_parity_weights_all_gates(n, w) == explicit

    def test_duplicate_supports_merge(self):
        """Gates on the same support (any order, repeated qubits) are one rotation."""
        gates = {0: [[0, 1]], 1: [[1, 0]], 2: [[2], [0, 1]], 3: [[3, 3]]}
        assert qre.iqp_parity_weights_from_gates(gates) == ((1, 2), (2, 1))

    def test_merged_angles(self):
        """Merged supports add their angles, and a shared parameter enters every support."""
        gates = {0: [[0, 1]], 1: [[1, 0]], 2: [[2], [0, 1]], 3: [[3, 3]]}
        angles = qre.iqp_parities_from_gates(gates, np.array([0.1, 0.2, 0.4, 0.8]))
        assert angles == pytest.approx({(0, 1): 0.7, (2,): 0.4, (3,): 0.8})
        assert qre.iqp_parities_from_gates(gates) == {(0, 1): 3.0, (2,): 1.0, (3,): 1.0}

    def test_qudit_merged_angles_are_linear(self):
        """The merged Walsh coefficients are the parameter-weighted sum of the per-gate ones,
        with one coefficient per parity of the histogram."""
        dims = [3, 4, 5]
        gates = _random_qudit_gates(dims, 6, 2, np.random.default_rng(0))
        params = np.random.default_rng(1).normal(size=len(gates))
        expected = {}
        for k, gate_list in gates.items():
            for key, c in qre.qudit_parities_from_gates({0: gate_list}, dims).items():
                expected[key] = expected.get(key, 0.0) + params[k] * c
        merged = qre.qudit_parities_from_gates(gates, dims, params)
        assert merged == pytest.approx(expected)
        assert len(merged) == sum(c for _, c in qre.qudit_parity_weights_from_gates(gates, dims))

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("n, m, w", [(6, 10, 2), (8, 40, 3), (5, 31, 5)])
    def test_arbitrary_bound_dominates_random_gates(self, seed, n, m, w):
        """No gate set of m gates of weight <= w has more rotations or CNOTs than the bound."""
        exact = qre.iqp_parity_weights_from_gates(create_random_gates(n, m, 1, w, seed=seed))
        bound = ts.iqp_parity_weights_arbitrary_gates(n, m, w)
        assert sum(c for _, c in exact) <= sum(c for _, c in bound)
        assert sum((k - 1) * c for k, c in exact) <= sum((k - 1) * c for k, c in bound)

    def test_arbitrary_bound_saturates_at_all_gates(self):
        """With more gates than supports, the bound is the all-gates histogram."""
        assert ts.iqp_parity_weights_arbitrary_gates(6, 10**6, 2) == (
            ts.iqp_parity_weights_all_gates(6, 2)
        )

    @pytest.mark.parametrize("d", [2, 3, 4, 5, 6, 8, 16])
    def test_site_spectrum_reproduces_eigenvalues(self, d):
        """The Walsh expansion reproduces sqrt(2) cos(2 pi v z/d + pi/4) (Eq. (50) of the notes)."""
        b = max(1, math.ceil(math.log2(d)))
        for v in range(1, d):
            spec = ts.qudit_site_spectrum(d, v)
            assert 0 not in spec  # no constant term: Q(v) is traceless on the populated labels
            for z in range(d):
                val = sum(a * (-1) ** bin(s & z).count("1") for s, a in spec.items())
                assert np.isclose(val, np.sqrt(2) * np.cos(2 * np.pi * v * z / d + np.pi / 4))
            assert all(0 < s < 2**b for s in spec)

    def test_qubit_site_spectrum(self):
        """For d=2 the generator Q(1) is Pauli Z (Sec. 2.2.6 of the notes)."""
        assert ts.qudit_site_spectrum(2, 1) == pytest.approx({1: 1.0})

    @pytest.mark.parametrize("dims, w", [([3, 4, 5], 2), ([4, 4, 4], 3), ([2, 8, 3, 6], 2)])
    def test_all_gates_dp_matches_enumeration(self, dims, w):
        """The site-by-site closed form equals the merged expansion of the enumerated gate set."""
        explicit = qre.qudit_parity_weights_from_gates(_all_qudit_gates(dims, w), dims)
        assert ts.qudit_parity_weights_all_gates(dims, w) == explicit

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("d, n, m, w", [(4, 5, 6, 2), (3, 4, 10, 3), (8, 4, 3, 2)])
    def test_qudit_arbitrary_bound(self, seed, d, n, m, w):
        """The qudit arbitrary-gate bound dominates random gate sets, per qubit weight."""
        rng = np.random.default_rng(seed)
        exact = dict(
            qre.qudit_parity_weights_from_gates(_random_qudit_gates([d] * n, m, w, rng), d)
        )
        bound = dict(ts.qudit_parity_weights_arbitrary_gates(d, n, m, w))
        assert sum(exact.values()) <= sum(bound.values())
        assert sum((k - 1) * c for k, c in exact.items()) <= sum(
            (k - 1) * c for k, c in bound.items()
        )

    def test_qudit_d2_reduces_to_qubit(self):
        """For d=2 the qudit gate set is the qubit gate set (Sec. 2.2.6)."""
        assert ts.qudit_parity_weights_all_gates(2, 3, num_qudits=7) == (
            ts.iqp_parity_weights_all_gates(7, 3)
        )

    def test_paper_parameter_counts(self):
        """Spectral Born machines Sec. 5: every two-qudit gate on 95 qudits (d=4) is 40470
        parameters; for d=4 each generator is one parity, so the rotation count is the same."""
        pw = ts.qudit_parity_weights_all_gates(4, 2, num_qudits=95)
        assert sum(c for _, c in pw) == 95 * 3 + math.comb(95, 2) * 9 == 40470


def _layer_t_count(angles, precision=None):
    """T count of a layer of single-qubit rotations by ``angles``."""
    layer = qre.ParityPhaseLayer(1, ((1, len(angles)),), precision, gates_params=angles)
    return _counts(layer)[1].get("T", 0)


class TestMixedSynthesis:
    """Expected T count per rotation with the mixed synthesis of Bothe et al., read off layers
    of 1000 equal rotations (the layer rounds the sum up once)."""

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
    def test_paper_values(self, theta, precision, expected):
        """Values quoted in arXiv:2605.31544, for both signs of the angle."""
        t_count = _layer_t_count([theta, -theta] * 500, precision)
        assert t_count == pytest.approx(1000 * expected, rel=1e-3, abs=1)

    @pytest.mark.parametrize("precision", [1e-12, 1e-9, 1e-6, 1e-3])
    def test_limits(self, precision):
        """Free below about precision/2, never dearer than the angle-independent cost, and
        periodic under the Clifford rotations by pi/4."""
        assert _layer_t_count([0.4 * precision] * 1000, precision) == 0
        theta = np.random.default_rng(0).uniform(-4, 4, size=1000)
        t_count = _layer_t_count(theta, precision)
        assert t_count <= 1000 * _counts(qre.RZ(precision=precision))[1]["T"]
        assert _layer_t_count(theta + np.pi / 4, precision) == t_count

    @pytest.mark.parametrize("theta, precision", [(np.pi / 8, 1e-9), (0.385, 1e-2)])
    def test_staircase_t_gate(self, theta, precision):
        """Close to pi/8 the T gate over-rotation is used whenever theta <= phi = pi/8, with
        expected T count sqrt(2) sin(2 theta); tan(0.385) > pi/8, where the paper's script
        falls back to the asymptotic formula."""
        t_count = _layer_t_count([theta] * 1000, precision)
        assert t_count == pytest.approx(1000 * np.sqrt(2) * np.sin(2 * theta), abs=1)


class TestParityPhaseLayer:
    """Tests for ParityPhaseLayer."""

    pw = ((1, 4), (2, 6))

    def test_resource_params_and_rep(self):
        """Resource params, resource_rep and the dispatching class."""
        op = qre.ParityPhaseLayer(4, {2: 6, 1: 4})
        assert op.resource_params == {
            "num_wires": 4,
            "parity_weights": self.pw,
            "precision": None,
            "gates_params": None,
        }
        rep = qre.ParityPhaseLayer.resource_rep(4, self.pw)
        assert rep.op_type is qre.ParityPhaseLayer
        assert rep == op.resource_rep_from_op()

    def test_resources(self):
        """One MultiRZ per parity."""
        assert qre.ParityPhaseLayer.resource_decomp(4, self.pw, 1e-5) == [
            GateCount(resource_rep(qre.MultiRZ, {"num_wires": 1, "precision": 1e-5}), 4),
            GateCount(resource_rep(qre.MultiRZ, {"num_wires": 2, "precision": 1e-5}), 6),
        ]

    def test_matches_core_iqp(self):
        """Oracle: the phase layer plus 2n Hadamards is qre.IQP on the same pattern."""
        gates = create_local_gates(6, 3)
        iqp = _counts(qre.IQP(6, list(gates.values())))[1]
        layer = _counts(qre.ParityPhaseLayer(6, qre.iqp_parity_weights_from_gates(gates)))[1]
        layer["Hadamard"] = 12
        assert layer == iqp

    def test_invalid_weight(self):
        """Weights outside 1..num_wires raise."""
        with pytest.raises(ValueError, match="Parity weight"):
            qre.ParityPhaseLayer(3, ((4, 1),))

    def test_adjoint_and_pow(self):
        """Adjoint and non-zero powers cost the same as the layer."""
        base = _counts(qre.ParityPhaseLayer(4, self.pw))
        for op in (
            qre.Adjoint(qre.ParityPhaseLayer(4, self.pw)),
            qre.Pow(qre.ParityPhaseLayer(4, self.pw), 3),
        ):
            res, counts = _counts(op)
            assert counts == base[1]
            assert res.algo_wires == base[0].algo_wires

    @pytest.mark.parametrize("num_ctrl, num_zero", [(1, 0), (1, 1), (3, 1)])
    def test_controlled(self, num_ctrl, num_zero):
        """One control doubles the rotations; more controls add a clean auxiliary qubit."""
        res, counts = _counts(qre.Controlled(qre.ParityPhaseLayer(4, self.pw), num_ctrl, num_zero))
        assert counts["T"] == 2 * 10 * 44
        assert res.algo_wires == 4 + num_ctrl
        mcx_aux = qre.estimate(qre.MultiControlledX(num_ctrl, num_zero)).zeroed_wires
        assert res.zeroed_wires == (0 if num_ctrl == 1 else 1 + mcx_aux)

    def test_controlled_identity(self):
        """Exact check of the single-control identity used in the controlled decomposition."""
        theta, n = 0.37, 3
        target = qp.matrix(qp.ctrl(qp.MultiRZ(-2 * theta, wires=[1, 2, 3]), control=0))
        half = qp.matrix(qp.MultiRZ(-theta, wires=[1, 2, 3]), wire_order=range(n + 1))
        both = qp.matrix(qp.MultiRZ(theta, wires=[0, 1, 2, 3]), wire_order=range(n + 1))
        assert np.allclose(target, half @ both)

    def test_gates_params_canonical(self):
        """The angles are stored as a sorted tuple, so their order and container do not matter."""
        angles = [0.3, -1e-5, 0.0, 1e-3, 2.0, 0.1, 0.2, 0.4, 0.5, 0.6]
        op = qre.ParityPhaseLayer(4, self.pw, gates_params=np.array(angles))
        assert op.resource_params["gates_params"] == tuple(sorted(angles))
        rep = qre.ParityPhaseLayer.resource_rep(4, self.pw, gates_params=angles[::-1])
        assert rep == op.resource_rep_from_op()

    def test_gates_params_wrong_length(self):
        """One angle per rotation is required."""
        with pytest.raises(ValueError, match="gates_params has 9 angles"):
            qre.ParityPhaseLayer(4, self.pw, gates_params=[0.1] * 9)

    def test_small_angles(self):
        """Ten rotations by 1e-5 at 1e-9 cost 10 x 9.066 expected T gates (Bothe et al. Sec. III),
        rounded up once, and keep their CNOT ladders."""
        counts = _counts(qre.ParityPhaseLayer(4, self.pw, gates_params=[1e-5] * 10))[1]
        assert counts == {"CNOT": 12, "T": 91}

    def test_large_angles_cost_rz(self):
        """At 1e-9 a generic angle is cheaper with RZ, so the layer costs as without angles."""
        angles = np.random.default_rng(0).uniform(0.1, 0.3, size=10)
        with_angles = _counts(qre.ParityPhaseLayer(4, self.pw, gates_params=angles))
        without = _counts(qre.ParityPhaseLayer(4, self.pw))
        assert with_angles[1] == without[1]
        assert with_angles[0].algo_wires == without[0].algo_wires

    def test_clifford_t_angles(self):
        """Multiples of pi/4 are Cliffords and odd multiples of pi/8 are one T gate each."""
        cliffords = [0.0, np.pi / 4, -np.pi / 2, 3 * np.pi / 4]
        t_gates = [np.pi / 8, -np.pi / 8, 3 * np.pi / 8, 9 * np.pi / 8]
        layer = qre.ParityPhaseLayer(2, ((1, 2), (2, 6)), gates_params=cliffords + t_gates)
        assert _counts(layer)[1] == {"CNOT": 12, "T": 4}

    def test_explicit_precision(self):
        """At precision 1e-3 the angle-independent mixed synthesis is cheaper than RZ, so every
        rotation is priced by its expected T count (3.244 for 1e-2, Bothe et al. Sec. III)."""
        layer = qre.ParityPhaseLayer(1, ((1, 1),), precision=1e-3, gates_params=[1e-2])
        assert _counts(layer)[1] == {"T": 4}
        rz_t = _counts(qre.RZ(precision=1e-3))[1]["T"]
        large = qre.ParityPhaseLayer(1, ((1, 1),), precision=1e-3, gates_params=[0.3])
        assert _counts(large)[1]["T"] < rz_t

    def test_adjoint_and_pow_with_angles(self):
        """The adjoint costs the same; a power multiplies every angle."""
        layer = qre.ParityPhaseLayer(4, self.pw, gates_params=[1e-5] * 10)
        assert _counts(qre.Adjoint(layer))[1] == _counts(layer)[1]
        tripled = qre.ParityPhaseLayer(4, self.pw, gates_params=[3e-5] * 10)
        assert _counts(qre.Pow(layer, 3))[1] == _counts(tripled)[1]
        t_layer = qre.ParityPhaseLayer(1, ((1, 2),), gates_params=[np.pi / 8] * 2)
        assert "T" not in _counts(qre.Pow(t_layer, 2))[1]  # T^2 = S

    def test_controlled_with_angles(self):
        """A controlled rotation by c is two rotations by c/2: a controlled S is two T gates."""
        layer = qre.ParityPhaseLayer(3, ((1, 3),), gates_params=[np.pi / 4] * 3)
        assert _counts(qre.Controlled(layer, 1, 0))[1] == {"CNOT": 6, "T": 6}
        res, counts = _counts(qre.Controlled(layer, 2, 1))
        assert counts["T"] == 6
        assert counts["Toffoli"] == 2 * _counts(qre.MultiControlledX(2, 1))[1]["Toffoli"]
        assert res.algo_wires == 5


class TestQuditFourierTransform:
    """Tests for QuditFourierTransform."""

    @pytest.mark.parametrize(
        "d, op",
        [
            (2, resource_rep(qre.Hadamard)),
            (4, resource_rep(qre.QFT, {"num_wires": 2})),
            (16, resource_rep(qre.QFT, {"num_wires": 4})),
            (5, resource_rep(qre.QubitUnitary, {"num_wires": 3, "precision": None})),
        ],
    )
    def test_resources(self, d, op):
        """Hadamard, QFT or a generic unitary depending on d."""
        assert qre.QuditFourierTransform.resource_decomp(d) == [GateCount(op)]

    @pytest.mark.parametrize("d", [2, 3, 8])
    def test_adjoint(self, d):
        """The adjoint costs the same and acts on ceil(log2 d) wires."""
        base, counts = _counts(qre.QuditFourierTransform(d))
        res, adj = _counts(qre.Adjoint(qre.QuditFourierTransform(d)))
        assert adj == counts
        assert res.algo_wires == base.algo_wires == max(1, math.ceil(math.log2(d)))

    @staticmethod
    def _dft(n):
        idx = np.arange(n)
        return np.exp(2j * np.pi * np.outer(idx, idx) / n) / np.sqrt(n)

    @pytest.mark.parametrize("d", [6, 10, 12, 20])
    def test_good_thomas_identity(self, d):
        """F_d = P_out (F_{2^k} (x) F_q) P_in for d = 2^k q with explicit label permutations."""
        two_power = d & -d
        q = d // two_power
        inv_q = pow(q, -1, two_power)
        inv_two = pow(two_power, -1, q)
        # input map (Ruritanian): n = q n1 + 2^k n2 mod d; output map (CRT)
        p_in = np.zeros((d, d))
        p_out = np.zeros((d, d))
        for n1 in range(two_power):
            for n2 in range(q):
                p_in[(q * n1 + two_power * n2) % d, n1 * q + n2] = 1
                p_out[(q * inv_q * n1 + two_power * inv_two * n2) % d, n1 * q + n2] = 1
        kron = np.kron(self._dft(two_power), self._dft(q))
        assert np.allclose(p_out @ kron @ p_in.T, self._dft(d))

    def test_good_thomas_resources(self):
        """d = 6 uses F_2 (x) F_3 with two permutations of 5 transpositions on 3 qubits."""
        decomp = qre.QuditFourierTransform.resource_decomp(6)
        mcx = resource_rep(qre.MultiControlledX, {"num_ctrl_wires": 2, "num_zero_ctrl": 0})
        assert decomp == [
            GateCount(mcx, 10),
            GateCount(resource_rep(qre.CNOT), 40),
            GateCount(resource_rep(qre.X), 40),
            GateCount(qre.QuditFourierTransform.resource_rep(2)),
            GateCount(qre.QuditFourierTransform.resource_rep(3)),
        ]
        _, exact = _counts(qre.QuditFourierTransform(6))
        _, generic = _counts(qre.QubitUnitary(3))
        assert exact["T"] == 660 < generic["T"] == 3168
        assert exact["Toffoli"] == 10

    @pytest.mark.parametrize("d", [3, 5, 7, 9, 15])
    def test_odd_dimension_is_generic(self, d):
        """Odd d has no power-of-two factor and stays a generic unitary."""
        b = math.ceil(math.log2(d))
        unitary = resource_rep(qre.QubitUnitary, {"num_wires": b, "precision": None})
        assert qre.QuditFourierTransform.resource_decomp(d) == [GateCount(unitary)]


class TestIQPSamplingShot:
    """Tests for IQPSamplingShot."""

    pw = ts.iqp_parity_weights_all_gates(20, 2)

    def test_resource_params_and_rep(self):
        """Resource params, resource_rep and the dispatching class."""
        op = qre.IQPSamplingShot(20, self.pw, 64, "arbitrary", select_swap_depth=2)
        rep = op.resource_rep_from_op()
        assert rep.op_type is qre.IQPSamplingShot
        assert rep.num_wires == 20
        assert rep.params["num_bits"] is None

    def test_decomposition(self):
        """SOS input state, 2n Hadamards, one phase layer."""
        decomp = qre.IQPSamplingShot.resource_decomp(20, self.pw, 64, "uniform_real")
        sos = resource_rep(
            LabsSumOfSlatersPrep,
            {
                "num_coeffs": 64,
                "num_wires": 20,
                "num_bits": 20,
                "stateprep_cmpr_op": resource_rep(qre.UniformStatePrep, {"num_states": 64}),
                "select_swap_depth": None,
            },
        )
        assert decomp == [
            GateCount(sos),
            GateCount(resource_rep(qre.Hadamard), 40),
            GateCount(qre.ParityPhaseLayer.resource_rep(20, self.pw)),
        ]

    def test_single_basis_state(self):
        """N = 1 is a BasisState and needs no auxiliary qubits."""
        res, counts = _counts(qre.IQPSamplingShot(20, self.pw, 1, "arbitrary"))
        assert res.algo_wires == res.total_wires == 20
        assert counts.get("Toffoli", 0) == 0
        assert counts["T"] == 44 * sum(c for _, c in self.pw)

    def test_uniform_cheaper_than_arbitrary(self):
        """Real, equal amplitudes never cost more Toffolis than arbitrary amplitudes."""
        arb = _counts(qre.IQPSamplingShot(20, self.pw, 1000, "arbitrary"))[1]
        uni = _counts(qre.IQPSamplingShot(20, self.pw, 1000, "uniform_real"))[1]
        assert uni["Toffoli"] < arb["Toffoli"]

    def test_scaling_in_n(self):
        """Doubling n at w = 2 multiplies the phase-layer T count by about 4 and at most 4."""
        t1 = _counts(
            qre.IQPSamplingShot(50, ts.iqp_parity_weights_all_gates(50, 2), 1, "arbitrary")
        )[1]
        t2 = _counts(
            qre.IQPSamplingShot(100, ts.iqp_parity_weights_all_gates(100, 2), 1, "arbitrary")
        )[1]
        assert 3.5 < t2["T"] / t1["T"] <= 4

    def test_exact_below_arbitrary_bound(self):
        """An explicit gate set never costs more than the arbitrary-gates bound for the same m, w."""
        gates = create_random_gates(30, 200, 1, 3, seed=1)
        exact = _counts(
            qre.IQPSamplingShot(30, qre.iqp_parity_weights_from_gates(gates), 100, "arbitrary")
        )[1]
        bound = _counts(
            qre.IQPSamplingShot(
                30, ts.iqp_parity_weights_arbitrary_gates(30, 200, 3), 100, "arbitrary"
            )
        )[1]
        assert exact["T"] <= bound["T"] and exact["CNOT"] <= bound["CNOT"]

    def test_symbolic_ops(self):
        """Adjoint, Pow and Controlled versions estimate without errors and keep the wires."""
        shot = qre.IQPSamplingShot(10, ts.iqp_parity_weights_all_gates(10, 2), 8, "arbitrary")
        base = qre.estimate(shot)
        for op in (qre.Adjoint(shot), qre.Pow(shot, 2), qre.Controlled(shot, 1, 0)):
            res = qre.estimate(op)
            assert res.algo_wires >= base.algo_wires

    def test_gates_params(self):
        """The angles reach the phase layer, and small angles make the layer cheaper."""
        angles = np.random.default_rng(0).normal(scale=1e-5, size=210)
        shot = qre.IQPSamplingShot(20, self.pw, 1, "arbitrary", gates_params=angles)
        assert shot.resource_params["gates_params"] == tuple(sorted(angles))
        decomp = qre.IQPSamplingShot.resource_decomp(**shot.resource_params)
        assert decomp[-1] == GateCount(
            qre.ParityPhaseLayer.resource_rep(20, self.pw, gates_params=angles)
        )
        res, counts = _counts(shot)
        assert counts["T"] < 44 * 210 / 5
        assert counts["CNOT"] == 2 * 190
        assert res.algo_wires == 20

    def test_gates_params_wrong_length(self):
        """One angle per rotation is required."""
        with pytest.raises(ValueError, match="gates_params has 3 angles"):
            qre.IQPSamplingShot(20, self.pw, 1, "arbitrary", gates_params=[0.1] * 3)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"init_state": "real"}, "init_state"),
            ({"num_init_states": 2**21}, "num_init_states"),
            ({"num_bits": 21}, "num_bits"),
        ],
    )
    def test_invalid(self, kwargs, match):
        """Invalid physics inputs raise."""
        args = {
            "num_wires": 20,
            "parity_weights": self.pw,
            "num_init_states": 8,
            "init_state": "arbitrary",
        }
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            qre.IQPSamplingShot(**args)


class TestQuditSamplingShot:
    """Tests for QuditSamplingShot."""

    def test_d2_equals_qubit_shot(self):
        """Limit: a d = 2 qudit shot is the qubit shot up to the F_2 = H identification."""
        pw = ts.qudit_parity_weights_all_gates(2, 2, num_qudits=12)
        q = _counts(qre.QuditSamplingShot(2, pw, 50, "arbitrary", num_qudits=12))
        b = _counts(qre.IQPSamplingShot(12, pw, 50, "arbitrary"))
        assert q[1] == b[1]
        assert q[0].total_wires == b[0].total_wires

    @pytest.mark.parametrize(
        "dims, wires", [(((4, 95),), 190), (((16, 36),), 144), ([3, 5, 8, 2], 9)]
    )
    def test_wires(self, dims, wires):
        """Binary encoding: 95 ququarts on 190 qubits and 36 d=16 qudits on 144 qubits
        (Spectral Born machines Sec. 5)."""
        res = qre.estimate(qre.QuditSamplingShot(dims, ((1, 1),), 1, "arbitrary"))
        assert res.algo_wires == wires

    def test_dims_formats(self):
        """int + num_qudits, per-site list and histogram give the same operator."""
        reps = {
            qre.QuditSamplingShot(
                4, ((1, 2),), 4, "arbitrary", num_qudits=3
            ).resource_rep_from_op(),
            qre.QuditSamplingShot([4, 4, 4], ((1, 2),), 4, "arbitrary").resource_rep_from_op(),
            qre.QuditSamplingShot(((4, 3),), ((1, 2),), 4, "arbitrary").resource_rep_from_op(),
        }
        assert len(reps) == 1

    def test_decomposition(self):
        """SOS input state on the encoding qubits, F_d and F_d^dagger per site, one layer."""
        pw = ((2, 3),)
        decomp = qre.QuditSamplingShot.resource_decomp(((3, 2), (4, 1)), pw, 1, "arbitrary")
        ft3, ft4 = qre.QuditFourierTransform.resource_rep(
            3
        ), qre.QuditFourierTransform.resource_rep(4)
        assert decomp == [
            GateCount(resource_rep(qre.BasisState, {"num_wires": 6})),
            GateCount(ft3, 2),
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": ft3}), 2),
            GateCount(ft4, 1),
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": ft4}), 1),
            GateCount(qre.ParityPhaseLayer.resource_rep(6, pw)),
        ]

    def test_gates_params_d2_equals_qubit_shot(self):
        """With angles, a d = 2 qudit shot is still the qubit shot."""
        pw = ts.qudit_parity_weights_all_gates(2, 2, num_qudits=12)
        angles = np.random.default_rng(0).normal(scale=1e-4, size=78)
        q = _counts(
            qre.QuditSamplingShot(2, pw, 50, "arbitrary", gates_params=angles, num_qudits=12)
        )
        b = _counts(qre.IQPSamplingShot(12, pw, 50, "arbitrary", gates_params=angles))
        assert q[1] == b[1]
        assert q[1]["T"] < _counts(qre.IQPSamplingShot(12, pw, 50, "arbitrary"))[1]["T"]

    def test_gates_params_reach_layer(self):
        """The angles of the Walsh-expanded layer reach the phase layer."""
        pw = ((2, 3),)
        angles = (1e-4, -2e-4, 0.3)
        decomp = qre.QuditSamplingShot.resource_decomp(
            ((3, 2), (4, 1)), pw, 1, "arbitrary", gates_params=angles
        )
        assert decomp[-1] == GateCount(
            qre.ParityPhaseLayer.resource_rep(6, pw, gates_params=angles)
        )
        with pytest.raises(ValueError, match="gates_params has 2 angles"):
            qre.QuditSamplingShot(((3, 2), (4, 1)), pw, 1, "arbitrary", gates_params=angles[:2])


def _rotations(s, degree):
    t = s if degree is None else min(degree, s)
    return sum(math.comb(s, k) for k in range(1, t + 1))


class TestSumPhaseLayer:
    """Tests for SumPhaseLayer."""

    @pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 100])
    @pytest.mark.parametrize("degree", [1, 3, None])
    def test_qubit_resources(self, n, degree):
        """Hamming weight of n qubits: n - popcount(n) TemporaryAND into as many auxiliary qubits,
        one rotation per Walsh term of degree <= t on the ceil(log2(n + 1)) sum bits."""
        res, gc = _counts(qre.SumPhaseLayer(2, degree=degree, num_qudits=n))
        adders = n - bin(n).count("1")
        s = math.ceil(math.log2(n + 1))
        t_per_rz = _counts(qre.RZ())[1]["T"]
        assert gc.get("Toffoli", 0) == adders
        assert gc["T"] == _rotations(s, degree) * t_per_rz
        assert res.algo_wires == n
        assert res.zeroed_wires == adders

    @pytest.mark.parametrize(
        "dims, adders",
        [(((4, 95),), 185), (((16, 36),), 140), (((3, 5), (5, 2)), 12), (((6, 7),), 18)],
    )
    def test_qudit_resources(self, dims, adders):
        """The sum register holds the sum of the largest encodable labels, and the tree needs at
        least one adder per input bit beyond the register (a full adder removes one bit net)."""
        res, gc = _counts(qre.SumPhaseLayer(dims))
        input_bits = sum(math.ceil(math.log2(d)) * c for d, c in dims)
        s = sum((2 ** math.ceil(math.log2(d)) - 1) * c for d, c in dims).bit_length()
        assert gc["T"] == (2**s - 1) * _counts(qre.RZ())[1]["T"]
        assert input_bits - s <= gc["Toffoli"] == adders < input_bits
        assert res.zeroed_wires == adders

    @pytest.mark.parametrize("s, degree", [(4, 1), (5, 2), (7, 3)])
    def test_polynomial_walsh_degree(self, s, degree):
        """A polynomial of degree t in S = sum_j 2^j x_j only has Walsh terms of weight <= t."""
        values = [sum((x >> j & 1) << j for j in range(s)) ** degree for x in range(2**s)]
        for mask in range(1, 2**s):
            coeff = sum((-1) ** bin(mask & x).count("1") * v for x, v in enumerate(values))
            if bin(mask).count("1") > degree:
                assert coeff == 0

    def test_resource_params_and_rep(self):
        """Dims formats agree and the parameters round-trip."""
        reps = {
            qre.SumPhaseLayer(4, degree=2, num_qudits=3).resource_rep_from_op(),
            qre.SumPhaseLayer([4, 4, 4], degree=2).resource_rep_from_op(),
            qre.SumPhaseLayer.resource_rep(((4, 3),), degree=2),
        }
        assert len(reps) == 1
        op = qre.SumPhaseLayer(((4, 3),), degree=2, precision=1e-6)
        assert op.resource_params == {"qudit_dims": ((4, 3),), "degree": 2, "precision": 1e-6}
        assert op.num_wires == 6

    def test_adjoint(self):
        """The adjoint costs the same as the layer."""
        layer = qre.SumPhaseLayer(2, degree=2, num_qudits=10)
        assert _counts(qre.Adjoint(layer))[1] == _counts(layer)[1]

    def test_invalid_degree(self):
        """A degree below 1 is rejected."""
        with pytest.raises(ValueError, match="degree must be at least 1"):
            qre.SumPhaseLayer(2, degree=0, num_qudits=4)


class TestPhaseOp:
    """The trainable phase layer is priced by the operator passed as ``phase_op``."""

    @pytest.mark.parametrize(
        "make_shot",
        [
            lambda phase: qre.IQPSamplingShot(10, ((2, 45),), 16, "arbitrary", phase_op=phase),
            lambda phase: qre.QuditSamplingShot(
                3, ((1, 6),), 16, "arbitrary", num_qudits=5, phase_op=phase
            ),
        ],
    )
    def test_phase_op_adds_its_cost(self, make_shot):
        """The shot with a phase layer costs the shot without it plus the layer."""
        phase = qre.SumPhaseLayer(2, degree=2, num_qudits=10)
        with_phase, without = make_shot(phase), make_shot(None)
        assert with_phase.resource_params["phase_cmpr_op"] == phase.resource_rep_from_op()
        assert without.resource_params["phase_cmpr_op"] is None
        total, base, layer = _counts(with_phase)[1], _counts(without)[1], _counts(phase)[1]
        assert total == {g: base.get(g, 0) + layer.get(g, 0) for g in set(base) | set(layer)}

    def test_phase_op_is_dequeued(self):
        """The phase operator passed in is not queued as a separate operation."""

        def circuit():
            qre.IQPSamplingShot(
                4, ((1, 4),), 1, "arbitrary", phase_op=qre.SumPhaseLayer(2, num_qudits=4)
            )

        with_phase = _counts(qre.IQPSamplingShot(4, ((1, 4),), 1, "arbitrary"))[1]
        layer = _counts(qre.SumPhaseLayer(2, num_qudits=4))[1]
        expected = {g: with_phase.get(g, 0) + layer.get(g, 0) for g in set(with_phase) | set(layer)}
        assert dict(qre.estimate(circuit)().gate_counts) == expected


class TestInputStateOracles:
    """The input-state cost against the template and Fomichev et al. Sec. III A."""

    @pytest.mark.parametrize("num_states, n, seed", [(100, 24, 0), (1000, 40, 1)])
    def test_step5_matches_template(self, num_states, n, seed):
        """With the m-1 Toffoli MCX decomposition the Step-5 Toffolis equal the template's
        TemporaryAND count; the labs default MCX costs 2m-3 (pinned ratio band)."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(2**n, num_states, replace=False)
        tmpl = _sos_state_prep_resources(np.ones(num_states), list(range(n)), idx)
        step5 = tmpl[qp.TemporaryAND]
        bits = qp.math.int_to_binary(np.array(idx), n).T
        num_bits = len(select_sos_rows(bits)[0])
        k = math.ceil(math.log2(num_states))

        def sos_toffoli(cfg):
            full = _counts(
                qre.IQPSamplingShot(n, ((1, 1),), num_states, "uniform_real", num_bits=num_bits),
                cfg,
            )[1]
            prep = _counts(qre.UniformStatePrep(num_states), cfg)[1]
            qrom = _counts(qre.LabsQROM(num_states, n, borrow_qubits=False), cfg)[1]
            return full["Toffoli"] - prep.get("Toffoli", 0) - qrom.get("Toffoli", 0)

        cfg = qre.LabsResourceConfig()
        cfg.set_decomp(qre.MultiControlledX, mcx_many_clean_aux_resource_decomp)
        m = min(num_bits, 2 * k - 1)
        assert sos_toffoli(cfg) == step5 == (num_states - 1) * (m - 1)
        assert sos_toffoli(qre.LabsResourceConfig()) == (num_states - 1) * (2 * m - 3)

    @pytest.mark.parametrize("num_states", [64, 1000, 4096])
    def test_fomichev_bound(self, num_states):
        """Fomichev et al. Sec. III A: Steps 2-6 cost (2 log D - 2) D + D Toffolis with the
        paper's MCX. Step 1 (QROM state preparation, 2^(log D + 1) in the paper) is priced by
        QROMStatePreparation at finite precision and is checked separately."""
        cfg = qre.LabsResourceConfig()
        cfg.set_decomp(qre.MultiControlledX, mcx_many_clean_aux_resource_decomp)
        k = math.ceil(math.log2(num_states))
        total = _counts(qre.IQPSamplingShot(60, ((1, 1),), num_states, "arbitrary"), cfg)[1]
        step1 = _counts(qre.QROMStatePreparation(k), cfg)[1]
        assert total["Toffoli"] - step1["Toffoli"] <= (2 * k - 2) * num_states + num_states
        assert step1["Toffoli"] >= 2 ** (k + 1)  # pinned: the finite-precision prep is not cheaper


def _qubit_reference(n, gates, theta, elems, amps):
    """Dense model q(x) = |<x| H D H |psi>|^2 from the definition (Eqs. (1), (3) of the notes)."""
    z = np.array(list(product([0, 1], repeat=n)))
    phase = np.zeros(2**n)
    for j, gate_list in gates.items():
        for g in gate_list:
            phase += theta[j] * (-1.0) ** z[:, g].sum(axis=1)
    psi = np.zeros(2**n, complex)
    for e, a in zip(elems, amps):
        psi[int("".join(map(str, e)), 2)] = a
    h = qp.matrix(qp.Hadamard(0))
    hn = h
    for _ in range(n - 1):
        hn = np.kron(hn, h)
    return np.abs(hn @ (np.exp(1j * phase) * (hn @ psi))) ** 2


class TestCompiledCircuits:
    """Exact identities: the counted gate sequence implements the tcdq circuit."""

    def test_qubit_shot_statevector(self):
        """StatePrep, H, one MultiRZ per support, H reproduces the dense qubit model."""
        n = 4
        gates = {0: [[0, 1]], 1: [[2]], 2: [[1, 2, 3], [0]], 3: [[1, 0]]}
        rng = np.random.default_rng(3)
        theta = rng.normal(size=len(gates))
        elems = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 1, 1]])
        amps = rng.normal(size=3) + 1j * rng.normal(size=3)
        amps /= np.linalg.norm(amps)

        angles = qre.iqp_parities_from_gates(gates, theta)
        assert len(angles) == sum(c for _, c in qre.iqp_parity_weights_from_gates(gates))

        dev = qp.device("default.qubit", wires=n)

        @qp.qnode(dev)
        def circuit():
            psi = np.zeros(2**n, complex)
            for e, a in zip(elems, amps):
                psi[int("".join(map(str, e)), 2)] = a
            qp.StatePrep(psi, wires=range(n))
            for q in range(n):
                qp.Hadamard(q)
            for supp, ang in angles.items():
                qp.MultiRZ(-2 * ang, wires=list(supp))
            for q in range(n):
                qp.Hadamard(q)
            return qp.probs(wires=range(n))

        assert np.allclose(circuit(), _qubit_reference(n, gates, theta, elems, amps))

    @pytest.mark.parametrize("dims", [[4, 4], [3, 4], [8, 2], [5, 3]])
    def test_qudit_shot_statevector(self, dims):
        """F_d blocks, merged Walsh MultiRZs and adjoint F_d blocks reproduce F^dag D F on the
        binary-encoded qudits."""
        rng = np.random.default_rng(sum(dims))
        gates = _random_qudit_gates(dims, 4, 2, rng)
        theta = rng.normal(size=len(gates))
        bits = [max(1, math.ceil(math.log2(d))) for d in dims]
        offs = np.cumsum([0] + bits)
        nq = int(offs[-1])

        # dense reference on the qudit space
        zs = np.array(list(product(*[range(d) for d in dims])))
        omega = np.zeros(len(zs))
        for j, gate_list in gates.items():
            for g in gate_list:
                val = np.ones(len(zs))
                for q, v in enumerate(g):
                    if v:
                        val *= np.sqrt(2) * np.cos(2 * np.pi * v * zs[:, q] / dims[q] + np.pi / 4)
                omega += theta[j] * val
        fts = [np.exp(2j * np.pi * np.outer(range(d), range(d)) / d) / np.sqrt(d) for d in dims]
        f = fts[0]
        for m in fts[1:]:
            f = np.kron(f, m)
        x0 = np.zeros(len(zs), complex)
        x0[0] = 1.0
        ref = np.abs(f.conj().T @ (np.exp(1j * omega) * (f @ x0))) ** 2

        # compiled circuit, with the merged rotation angles that gates_params receives
        parities = qre.qudit_parities_from_gates(gates, dims, theta)
        assert len(parities) == sum(c for _, c in qre.qudit_parity_weights_from_gates(gates, dims))

        def ft_block(d, b):
            full = np.eye(2**b, dtype=complex)
            full[:d, :d] = np.exp(2j * np.pi * np.outer(range(d), range(d)) / d) / np.sqrt(d)
            return full

        dev = qp.device("default.qubit", wires=nq)

        @qp.qnode(dev)
        def circuit():
            for q, (d, b) in enumerate(zip(dims, bits)):
                qp.QubitUnitary(ft_block(d, b), wires=range(offs[q], offs[q] + b))
            for key, ang in parities.items():
                wires = [
                    int(offs[q] + bits[q] - 1 - i)
                    for q, mask in key
                    for i in range(bits[q])
                    if mask >> i & 1
                ]
                qp.MultiRZ(-2 * ang, wires=wires)
            for q, (d, b) in enumerate(zip(dims, bits)):
                qp.adjoint(qp.QubitUnitary(ft_block(d, b), wires=range(offs[q], offs[q] + b)))
            return qp.probs(wires=range(nq))

        probs = np.asarray(circuit()).reshape([2**b for b in bits])
        probs = probs[tuple(slice(0, d) for d in dims)].reshape(-1)
        assert np.isclose(probs.sum(), 1.0)
        assert np.allclose(probs, ref, atol=1e-10)

    @pytest.mark.parametrize("b", [1, 2, 3])
    def test_qft_is_qudit_fourier(self, b):
        """For d = 2^b, qp.QFT on b big-endian wires is F_d (Eq. (42) of the notes)."""
        d = 2**b
        fd = np.exp(2j * np.pi * np.outer(range(d), range(d)) / d) / np.sqrt(d)
        assert np.allclose(qp.matrix(qp.QFT(wires=range(b))), fd)
