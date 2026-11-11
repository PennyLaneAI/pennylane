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
r"""Resource models for a computational-basis shot of a TCDQ circuit.

The adapter :func:`pennylane.labs.tcdq.shot_resources` composes existing resources:
input preparation, H D H for qubits or F^dagger D F for binary-encoded qudits.
Measurement is free; K shots multiply gate counts, not the required wires.

Qudit generators use their Walsh expansion (Welch et al., arXiv:1306.3991), with
zero phases on unused labels. Equal parities merge. Fourier transforms use QFT
for powers of two, Good-Thomas factorization for even dimensions, and a generic
unitary for odd dimensions; label permutations use a transposition upper bound.

Known diagonal angles use Bothe et al. (arXiv:2605.31544, Eqs. 9-12): the cheaper
of ordinary RZ and ancilla-free mixed synthesis. T counts are expectations,
rounded up once per layer. CNOT ladders are always counted; synthesis Cliffords
are omitted. The staircase uses theta <= phi (paper footnote 4). Every rotation
gets the same precision; no application error budget is apportioned. Comparing
the mixture's diamond-norm error with RZ precision is a cost-model assumption.
Custom RZ/MultiRZ decompositions take precedence over mixed synthesis.
Mixed pricing uses LabsResourceConfig; a core ResourceConfig uses ordinary MultiRZ.

Input preparation follows Fomichev et al. (arXiv:2310.18410, Sec. III A), using
QROMStatePreparation or UniformStatePrep for the condensed state, followed by
LabsSumOfSlatersPrep. Its default MCX and select-swap QROM costs differ from the
paper's many-clean-ancilla MCX and unary-iteration QROM; estimator configuration
controls those decompositions. T and Toffoli counts remain separate.
"""

import math
from collections import Counter

import pennylane.labs.estimator_beta as qre
from pennylane import math as pl_math
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.wires import WiresLike

# pylint: disable=arguments-differ, too-many-arguments


def _num_bits(dim: int) -> int:
    """Number of qubits used to store one qudit of dimension ``dim``."""
    return max(1, (int(dim) - 1).bit_length())


def _as_histogram(counter) -> tuple[tuple[int, int], ...]:
    """Canonical ``((weight, count), ...)`` tuple, sorted by weight, zero counts dropped."""
    return tuple(sorted((int(w), int(c)) for w, c in dict(counter).items() if c > 0))


def _validate_histogram(parity_weights, num_wires):
    hist = _as_histogram(dict(parity_weights))
    for w, _ in hist:
        if w < 1 or w > num_wires:
            raise ValueError(f"Parity weight {w} must lie between 1 and num_wires={num_wires}.")
    return hist


def iqp_parities_from_gates(
    gates: dict, params=None, num_wires=None
) -> dict[tuple[int, ...], float]:
    """Merge qubit supports into angles of exp(i theta Z); params=None uses unit angles."""
    params = None if params is None else pl_math.to_numpy(params).tolist()
    angles = {}
    for k, gate_list in gates.items():
        theta = 1.0 if params is None else params[k]
        for g in gate_list:
            if g:
                if num_wires is not None:
                    g = [q + num_wires if q < 0 else q for q in g]
                    if min(g) < 0 or max(g) >= num_wires:
                        raise IndexError("Qubit index out of range.")
                key = tuple(sorted(set(g)))
                angles[key] = angles.get(key, 0.0) + theta
    return angles


def qudit_site_spectrum(dim: int, power: int) -> dict[int, float]:
    """Walsh coefficients of sqrt(2) cos(2 pi power z/dim + pi/4), zero outside dim."""
    if not 0 < power < dim:
        raise ValueError(f"power must lie in 1..{dim - 1}, got {power}.")
    size = 2 ** _num_bits(dim)
    vals = [
        math.sqrt(2) * math.cos(2 * math.pi * power * z / dim + math.pi / 4) for z in range(dim)
    ]
    spectrum = {}
    for s in range(size):
        a = sum((-1) ** bin(s & z).count("1") * val for z, val in enumerate(vals)) / size
        if abs(a) > 1e-12:
            spectrum[s] = a
    return spectrum


def _dims_list(dims, num_qudits=None):
    """Expand a repeated local dimension or copy a per-site dimension list."""
    if isinstance(dims, int):
        if num_qudits is None:
            raise ValueError("num_qudits is required when dims is an int.")
        return [dims] * num_qudits
    return list(dims)


def qudit_parities_from_gates(gates: dict, dims, params=None) -> dict[tuple, float]:
    """Expand sparse qudit generators into merged ((site, bit_mask), ...) parity angles."""
    params = None if params is None else pl_math.to_numpy(params).tolist()
    total = {}
    for k, gate_list in gates.items():
        theta = 1.0 if params is None else params[k]
        for g in gate_list:
            terms = {(): theta}
            for q, v in sorted(g.items()):
                v = int(v) % dims[q]
                if v == 0:
                    continue
                spec = qudit_site_spectrum(dims[q], v)
                new = {}
                for key, c in terms.items():
                    for mask, a in spec.items():
                        k2 = key + ((q, mask),) if mask else key
                        new[k2] = new.get(k2, 0.0) + c * a
                terms = new
            for key, c in terms.items():
                if key:  # the all-identity term is a global phase
                    total[key] = total.get(key, 0.0) + c
    return total


# Staircase of optimal over-rotations (Table 1 and Appendix C, exhaustive search up to 35 T),
# copied from the paper's ancillary script small_angle_costing.py in its order (decreasing
# tan(alpha)): tan(alpha), the average T count divided by sin(2 theta), and the over-rotation
# angle phi.
# fmt: off
_STAIRCASE_TANALPHA = pl_math.array([
    1.000000000000000000, 0.414213562373095090, 0.350834874673672581, 0.312876580442609076, 0.212012989773690319,
    0.199053696635809713, 0.137553374901583453, 0.114743436552311937, 0.110968924780318418, 0.106108155871760396,
    0.095599761041589515, 0.093857467246492854, 0.089772198956584240, 0.081007642852468559, 0.080059368825502769,
    0.071823156445232433, 0.068074811618681924, 0.063557891353411389, 0.056565932310567113, 0.054687766545393701,
    0.043853474353372793, 0.043184230632607387, 0.041757558494795996, 0.035525578867165862, 0.026620221579097662,
    0.023740068332829375, 0.023219765631210493, 0.023041225970146049, 0.021820124509931472, 0.021584661590176659,
    0.021095252610628751, 0.020170619047074331, 0.016419959844448571, 0.013250840260523361, 0.013201874372092005,
    0.011678064337090286, 0.011610273857837769, 0.010123357001352220, 0.009178937379670129, 0.008939807749050285,
    0.008911641286055169, 0.008891831473781506, 0.008223547022012874, 0.008134840314959944, 0.005676564243201448,
    0.005490790602226593, 0.005438659165343685, 0.005011421293695884, 0.004445849949711543, 0.003802274290667123,
    0.003414481239748043, 0.003400465772733619, 0.003362918351896612, 0.002623446891891916, 0.002421456525235684,
    0.001942671784383428,
][::-1])

_STAIRCASE_AVG_T = pl_math.array([
    0.000000000000000000, 1.414213562373094923, 8.358523998839725522, 14.90836057126281666, 18.66666666666666430,
    35.41820150062702055, 36.69143914061946532, 45.81060988897981900, 59.62423066692823426, 79.37357444754127300,
    82.26631789232057201, 91.53596103867951683, 100.1969478876198707, 103.3309855545793710, 114.8865692342005786,
    129.9467535443398845, 130.1425954431041134, 137.7083130201563108, 162.2185949989647042, 177.3704252435253466,
    190.7728291979915696, 234.8268288886290804, 253.6524018919697312, 274.1441976350207597, 277.6535538936209377,
    337.6541634661525109, 418.3216273097789895, 546.1909215474321400, 569.3392538941667453, 610.1284182110173333,
    621.5720973965859457, 648.8713606032779353, 670.4052913138149279, 869.0702760915087310, 1061.190329568459902,
    1126.943755563780542, 1206.586675638444831, 1334.124470449861747, 1395.777866999348362, 1472.699658090156618,
    1580.987320181555106, 1690.149558553591305, 1831.306963333395743, 1910.882476918264274, 2018.310491944398791,
    2324.117154310660226, 2486.215289670361472, 3293.182410105669078, 3605.545686445599586, 4083.705682496006830,
    4431.716899082452983, 4707.761035937885026, 5205.974979638611330, 6671.197660262269892, 7123.787980357737979,
    8537.336639973320416,
][::-1])

_STAIRCASE_PHI = pl_math.array([
    0.785398163397448168, 0.392699081698724195, 0.255495373648521762, 0.284924126622062401, 0.192835807949161497,
    0.173552440720655482, 0.124107795648728134, 0.110082423147095237, 0.109906358777430102, 0.101528462773912573,
    0.091710893591349804, 0.093403544968404958, 0.070187845172285837, 0.077752676679104127, 0.078669855732249924,
    0.069493297706496718, 0.057791644573391976, 0.061888637249731267, 0.055597998797776847, 0.053665989739915342,
    0.041987261143899557, 0.042637172618535335, 0.041443167083348088, 0.034682096952168145, 0.019816715253174161,
    0.022220137471257971, 0.022717871336363209, 0.022893844150929561, 0.021083637492522535, 0.021313565565117158,
    0.020116119052836046, 0.020040204171539192, 0.016410933443297120, 0.013234079689553997, 0.013194264811155256,
    0.011536666869366122, 0.011604021540317999, 0.010119687083072504, 0.008597823205676720, 0.008827793348440431,
    0.008855693342139597, 0.008875421608560755, 0.008191240044674157, 0.008111792864880870, 0.005202477715701859,
    0.005378493332280404, 0.005430047050166927, 0.005010434945759594, 0.004437666869261012, 0.003795608808731670,
    0.003384718928535352, 0.003398669385652424, 0.003361547449303978, 0.002623229168641762, 0.002386380112804906,
    0.001932691904085085,
][::-1])
# fmt: on

_K_ASYMPTOTIC = (2 * math.sqrt(2 * math.e**3) / 3) ** (2 / 3)


def _mixed_synthesis_t_counts(angles, precision):
    """Expected T counts for ancilla-free mixed synthesis, Bothe et al. Eqs. 9-12."""
    # rotations by multiples of pi/4 are Cliffords, so only |theta| reduced to [0, pi/8] costs T
    theta = pl_math.abs((pl_math.to_numpy(angles) + math.pi / 8) % (math.pi / 4) - math.pi / 8)
    t_counts = pl_math.zeros_like(theta)
    live = theta > 0
    theta = theta[live]
    sin2, tan1 = pl_math.sin(2 * theta), pl_math.tan(theta)
    row = pl_math.searchsorted(_STAIRCASE_TANALPHA, precision / sin2 + tan1, side="right") - 1
    row_or_0 = pl_math.clip(row, 0, None)
    stair = (row >= 0) & (theta <= _STAIRCASE_PHI[row_or_0])
    live_counts = _STAIRCASE_AVG_T[row_or_0] * sin2
    # off the staircase precision < sin(2 theta), so alpha < 1 + pi/8 < K and all terms are finite
    th = theta[~stair]
    alpha = precision / (2 * th) + th
    phi0 = pl_math.maximum(alpha - alpha / pl_math.log(_K_ASYMPTOTIC / alpha), th)
    live_counts[~stair] = (3 * th / (alpha + 2 * phi0)) * pl_math.log2(
        12 / ((alpha - phi0) ** 2 * (alpha + 2 * phi0))
    )
    worst = 1.52 * math.log2(1 / precision) - 0.01
    t_counts[live] = pl_math.maximum(pl_math.minimum(live_counts, worst), 0.0)
    return t_counts


def _canonical_params(gates_params, parity_weights):
    """Sorted tuple of the rotation angles, one per parity of ``parity_weights``."""
    if gates_params is None:
        return None
    angles = tuple(sorted(pl_math.to_numpy(gates_params).astype(float).ravel().tolist()))
    num_rotations = sum(c for _, c in parity_weights)
    if len(angles) != num_rotations:
        raise ValueError(
            f"gates_params has {len(angles)} angles, but parity_weights has {num_rotations} "
            "rotations; give one angle per rotation."
        )
    return angles


class ParityPhaseLayer(ResourceOperator):
    r"""Resource model for commuting rotations :math:`\exp(i c_P Z_P)`.

    Args:
        num_wires (int): register width
        parity_weights (tuple[tuple[int, int]]): distinct parity counts by Pauli weight
        precision (float | None): per-rotation precision; otherwise use configured MultiRZ/RZ
        gates_params (Sequence[float] | None): one angle per parity; None uses ordinary MultiRZ
        wires (WiresLike | None): optional wire labels

    Known angles use the mixed-synthesis assumptions documented at module level.
    Their order is irrelevant to the count, but their number must match the histogram.
    """

    resource_keys = {"num_wires", "parity_weights", "precision", "gates_params"}

    def __init__(
        self,
        num_wires: int,
        parity_weights,
        precision: float | None = None,
        gates_params=None,
        wires: WiresLike | None = None,
    ):
        self.num_wires = num_wires
        self.parity_weights = _validate_histogram(parity_weights, num_wires)
        self.precision = precision
        self.gates_params = _canonical_params(gates_params, self.parity_weights)
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        """Returns a dictionary containing the minimal information needed to compute the resources."""
        return {
            "num_wires": self.num_wires,
            "parity_weights": self.parity_weights,
            "precision": self.precision,
            "gates_params": self.gates_params,
        }

    @classmethod
    def resource_rep(
        cls, num_wires: int, parity_weights, precision: float | None = None, gates_params=None
    ) -> CompressedResourceOp:
        """Return the resources for this operation."""
        parity_weights = _validate_histogram(parity_weights, num_wires)
        params = {
            "num_wires": num_wires,
            "parity_weights": parity_weights,
            "precision": precision,
            "gates_params": _canonical_params(gates_params, parity_weights),
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        num_wires: int,
        parity_weights,
        precision: float | None = None,
        gates_params=None,
        *,
        config=None,
    ):
        """Price the layer, honoring the calling labs estimator's rotation policy."""
        del num_wires
        if (
            config is None
            or gates_params is None
            or any(op in config.custom_decomps for op in (qre.RZ, qre.MultiRZ))
        ):
            return [
                GateCount(resource_rep(qre.MultiRZ, {"num_wires": w, "precision": precision}), c)
                for w, c in parity_weights
            ]
        if precision is None:
            precision = config.resource_op_precisions[qre.MultiRZ]["precision"]
        if precision is None:
            precision = config.resource_op_precisions[qre.RZ]["precision"]
        if not 0 < precision < 1:
            raise ValueError("Mixed-synthesis precision must lie strictly between 0 and 1.")
        rz = resource_rep(qre.RZ, {"precision": precision})
        rz_t = sum(
            g.count for g in qre.RZ.resource_decomp(precision=precision) if g.gate.name == "T"
        )
        t_counts = _mixed_synthesis_t_counts(gates_params, precision)
        mixed = t_counts < rz_t
        num_cnots = sum(2 * (w - 1) * c for w, c in parity_weights)
        num_rz = len(gates_params) - int(pl_math.sum(mixed))
        expected_t = float(pl_math.sum(t_counts[mixed]))
        nearest = round(expected_t)
        if nearest > 0 and abs(expected_t - nearest) <= 8 * math.ulp(expected_t):
            expected_t = nearest
        num_t = math.ceil(expected_t)
        counts = ((resource_rep(qre.CNOT), num_cnots), (rz, num_rz), (resource_rep(qre.T), num_t))
        return [GateCount(op, n) for op, n in counts if n > 0]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None):
        """The adjoint negates every angle and costs the same as the layer."""
        return [GateCount(cls.resource_rep(**target_resource_params))]

    @classmethod
    def pow_resource_decomp(cls, pow_z: int, target_resource_params: dict | None = None):
        """Scale known angles by the power; a zero power is the identity."""
        if pow_z == 0:
            return [GateCount(resource_rep(qre.Identity))]
        params = dict(target_resource_params)
        if params["gates_params"] is not None:
            params["gates_params"] = [pow_z * c for c in params["gates_params"]]
        return [GateCount(cls.resource_rep(**params))]

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires: int, num_zero_ctrl: int, target_resource_params: dict | None = None
    ):
        """Use two half-angle rotations per parity, and a clean control ancilla if needed."""
        params = target_resource_params
        weights = Counter()
        for w, c in params["parity_weights"]:
            weights[w] += c
            weights[w + 1] += c
        angles = params["gates_params"]
        if angles is not None:
            angles = [s * c / 2 for c in angles for s in (1, -1)]
        layer = cls.resource_rep(params["num_wires"] + 1, weights, params["precision"], angles)
        gates = [GateCount(layer)]
        if num_ctrl_wires == 1:
            if num_zero_ctrl:
                gates.append(GateCount(resource_rep(qre.X), 2))
            return gates
        aux = qre.Allocate(1, restored=True)
        mcx = resource_rep(
            qre.MultiControlledX,
            {"num_ctrl_wires": num_ctrl_wires, "num_zero_ctrl": num_zero_ctrl},
        )
        return [aux, GateCount(mcx, 2)] + gates + [qre.Deallocate(allocated_register=aux)]


def _crt_factors(dim: int) -> tuple[int, int]:
    """Split ``dim = 2^k * q`` with ``q`` odd; returns ``(2^k, q)``."""
    two_power = dim & -dim
    return two_power, dim // two_power


def fourier_resources(dim):
    """Compose a binary-encoded Fourier cost model using existing resource operators."""
    bits = _num_bits(dim)
    if dim == 2:
        return qre.Hadamard()
    if dim == 2**bits:
        return qre.QFT(bits)
    even, odd = _crt_factors(dim)
    if even == 1:
        return qre.QubitUnitary(bits)
    transpositions = 2 * (dim - 1)
    return qre.Prod(
        [
            (qre.MultiControlledX(bits - 1), transpositions),
            (qre.CNOT(), 2 * (bits - 1) * transpositions),
            (qre.X(), 2 * (bits - 1) * transpositions),
            fourier_resources(even),
            fourier_resources(odd),
        ]
    )


def _canonical_dims(qudit_dims, num_qudits=None):
    """``((d, number of sites), ...)`` from a histogram, a per-site list, or an ``int``."""
    if isinstance(qudit_dims, tuple) and qudit_dims and isinstance(qudit_dims[0], tuple):
        return _as_histogram(dict(qudit_dims))
    return _as_histogram(Counter(_dims_list(qudit_dims, num_qudits)))


def _adder_tree(qudit_dims) -> tuple[int, int, int]:
    """Return full adders, half adders and sum bits for a carry-save digit-sum tree."""
    columns = []
    for d, count in qudit_dims:
        for j in range(_num_bits(d)):
            if j == len(columns):
                columns.append(0)
            columns[j] += count
    full = half = 0
    j = 0
    while j < len(columns):
        while columns[j] > 1:
            if j + 1 == len(columns):
                columns.append(0)
            if columns[j] >= 3:
                full += 1
                columns[j] -= 2
            else:
                half += 1
                columns[j] -= 1
            columns[j + 1] += 1
        j += 1
    return full, half, len(columns)


class SumPhaseLayer(ResourceOperator):
    r"""Phase function of the binary-encoded digit sum, with clean adder uncomputation.

    Args:
        qudit_dims (Sequence[int] | int): local dimensions; an int requires num_qudits
        degree (int | None): polynomial degree, or None for an arbitrary function of the sum
        precision (float | None): per-rotation precision, otherwise configured MultiRZ/RZ
        num_qudits (int | None): number of equal-dimension sites
        wires (WiresLike | None): optional wire labels

    A carry-save tree uses TemporaryAND operations; measurement-assisted uncomputation
    restores its ancillas. The sum-register phase contains parities up to the given degree.
    """

    resource_keys = {"qudit_dims", "degree", "precision"}

    def __init__(
        self,
        qudit_dims,
        degree: int | None = None,
        precision: float | None = None,
        num_qudits: int | None = None,
        wires: WiresLike | None = None,
    ):
        if degree is not None and degree < 1:
            raise ValueError(f"degree must be at least 1, got {degree}.")
        self.qudit_dims = _canonical_dims(qudit_dims, num_qudits)
        self.num_wires = sum(_num_bits(d) * c for d, c in self.qudit_dims)
        self.degree = degree
        self.precision = precision
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        """Returns a dictionary containing the minimal information needed to compute the resources."""
        return {"qudit_dims": self.qudit_dims, "degree": self.degree, "precision": self.precision}

    @classmethod
    def resource_rep(
        cls, qudit_dims, degree: int | None = None, precision: float | None = None
    ) -> CompressedResourceOp:
        """Return the resources for this operation."""
        qudit_dims = _canonical_dims(qudit_dims)
        num_wires = sum(_num_bits(d) * c for d, c in qudit_dims)
        params = {"qudit_dims": qudit_dims, "degree": degree, "precision": precision}
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, qudit_dims, degree: int | None = None, precision: float | None = None):
        """Count the adder tree, phase on its sum register, and clean uncomputation."""
        full, half, s = _adder_tree(qudit_dims)
        t = s if degree is None else min(degree, s)
        weights = tuple((k, math.comb(s, k)) for k in range(1, t + 1))
        phase = GateCount(ParityPhaseLayer.resource_rep(s, weights, precision))
        adders = full + half
        if adders == 0:
            return [phase]
        aux = qre.Allocate(adders, restored=True)
        tand = resource_rep(qre.TemporaryAND)
        return [
            aux,
            GateCount(tand, adders),
            GateCount(resource_rep(qre.CNOT), 2 * (5 * full + half)),
            phase,
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": tand}), adders),
            qre.Deallocate(allocated_register=aux),
        ]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None):
        """The adjoint negates the phase and costs the same as the layer."""
        return [GateCount(cls.resource_rep(**target_resource_params))]
