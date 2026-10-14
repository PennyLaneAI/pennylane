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
r"""Resource operators for one computational-basis shot of the ``tcdq`` circuit families.

The ``pennylane.labs.tcdq`` module trains two circuit families classically,

.. math::

    U_{\text{qubit}}(\theta) = H^{\otimes n} D(\theta) H^{\otimes n}, \qquad
    U_{\text{qudit}}(\theta) = F_{\mathbf{d}}^\dagger D(\theta) F_{\mathbf{d}},

acting on an input state that is a superposition of :math:`N` computational basis states
(`IQPopt, arXiv:2501.04776 <https://arxiv.org/abs/2501.04776>`__ for qubits,
`Spectral Born machines, arXiv:2607.06675 <https://arxiv.org/abs/2607.06675>`__ for qudits). The
operators in this module price the quantum part: **one shot**, i.e. preparing the input state,
applying the circuit once and measuring every wire in the computational basis. Use
:func:`~pennylane.labs.tcdq.shot_resources` to build the shot directly from a
:class:`~pennylane.labs.tcdq.CircuitConfig` or :class:`~pennylane.labs.tcdq.QuditCircuitConfig`.

Conventions
-----------

* **Environment**: ``pennylane.labs.estimator_beta`` (labs), its ``estimate`` and
  ``LabsResourceConfig``. The labs namespace is required for :class:`~.LabsSumOfSlatersPrep`.
* **Unit of cost**: one shot. Measurement in the computational basis is free. ``K`` shots cost
  ``K`` times the gates and the same wires.
* **Trainable phase layer**: the optional layer :math:`D'(\xi)` (``phase_fn`` in ``tcdq``) is an
  arbitrary classical function, so its cost cannot be inferred. It is priced by the operator
  passed as ``phase_op``, e.g. a :class:`~.SumPhaseLayer` for functions of the digit sum.
* **Qubit phase layer**: one :class:`~.pennylane.estimator.MultiRZ` per distinct gate support
  (``exp(i theta Z_S) = MultiRZ(-2 theta)``). Gates that share a support merge into one rotation.
* **Qudit encoding**: each qudit of dimension ``d`` is stored in ``b = ceil(log2 d)`` qubits
  (binary encoding of the integer label), as in the qubit counts of arXiv:2607.06675, Sec. 5
  (``36 log2 16 = 144`` and ``95 x 2 = 190`` qubits).
* **Qudit phase layer**: the eigenvalue function of every generator
  :math:`Q(v) = \chi Z^v + \chi^* Z^{-v}` is expanded in Walsh functions on the encoding bits
  (`Welch et al., NJP 16, 033040 (2014), Eq. (9) <https://arxiv.org/abs/1306.3991>`__); each
  non-zero Walsh coefficient is one ``MultiRZ``. All gates are diagonal and commute, so equal
  parities from different gates merge into one rotation. Labels ``z >= d`` are never populated;
  their phases are set to zero (not optimised).
* **Qudit Fourier transform**: ``Hadamard`` for ``d = 2``, :class:`~.pennylane.estimator.QFT`
  on ``b`` qubits for ``d = 2^b``, the prime-factor (Good-Thomas) factorisation
  :math:`F_d = \Pi_{\text{out}} (F_{2^k} \otimes F_q) \Pi_{\text{in}}` for even ``d = 2^k q``
  with ``q`` odd, and a generic :class:`~.pennylane.estimator.QubitUnitary` on ``b`` qubits for
  odd ``d``. :math:`F_d^\dagger` costs the same as :math:`F_d`. The quantum Bluestein algorithm
  (`Kuo and Wong, arXiv:2512.15349 <https://arxiv.org/abs/2512.15349>`__) is not used: its
  convolution step is a block encoding that succeeds with probability 0.17-0.25 per transform,
  and making it deterministic with amplitude amplification costs more than the generic unitary
  for odd ``d < 33``.
* **Input state**: :class:`~.LabsSumOfSlatersPrep`
  (`Fomichev et al., PRX Quantum 5, 040339 (2024), Sec. III A <https://arxiv.org/abs/2310.18410>`__).
  The condensed state on ``k = ceil(log2 N)`` qubits is prepared with
  :class:`~.pennylane.estimator.QROMStatePreparation` for arbitrary complex amplitudes (Step 1 of
  Sec. III A, which cites the QROM state preparation of Low et al.), and with
  :class:`~.pennylane.estimator.UniformStatePrep` for real, equal amplitudes. ``N = 1`` is a
  :class:`~.pennylane.estimator.BasisState`.
* **Identification register**: unless ``num_bits`` is given, the state is assumed to need the
  full ``2k - 1`` identification qubits (non-identity encoding), matching the data-independent
  upper bound of :class:`~.pennylane.SumOfSlatersPrep`.
* **Rotation precision**: per rotation, ``precision`` if given, otherwise the config value
  (``1e-9`` for ``RZ`` in ``LabsResourceConfig``). No error budget is split here.
* **Gate metric**: ``T`` and ``Toffoli`` are reported separately; a combined figure, where
  quoted, is ``T + 4 Toffoli``.

Deviation ledger
----------------

Faithful to the circuits above and to Fomichev et al. Sec. III A, except:

* Step 5 of Fomichev et al. is priced by ``LabsSumOfSlatersPrep`` as ``N - 1``
  :class:`~.pennylane.estimator.MultiControlledX` on ``m = 2k - 1`` controls. Under the default
  ``LabsResourceConfig`` decomposition (one clean auxiliary qubit) each costs ``2m - 3``
  Toffolis, against ``m - 1`` in the paper and in :class:`~.pennylane.SumOfSlatersPrep`
  (1.9x on that step for ``N = 1000``). Setting the ``mcx_many_clean_aux_resource_decomp``
  decomposition in the config recovers ``m - 1`` for ``m - 2`` more auxiliary qubits.
* QROM: the select-swap ``LabsQROM`` with ``borrow_qubits=False`` in place of the paper's
  ``D``-Toffoli unary-iteration QROM.
* ``LabsSumOfSlatersPrep`` defaults its condensed state preparation to
  :class:`~.LabsMottonenStatePreparation`, whose counts (``2^(k+2) - 5`` rotations) are the
  ones Mottonen et al. (quant-ph/0407010) give for a transformation between two arbitrary
  states; from a basis state they state that half suffices. This module does not use that
  default (see the input-state convention above).
* The controlled phase shifts inside ``QFT`` are priced as generic rotations, including the
  ``pi/2`` one that is an exact Clifford+T gate.
* The two label permutations of the Good-Thomas factorisation are priced as ``d - 1``
  transpositions, each one ``MultiControlledX`` on ``b - 1`` controls plus ``2(b - 1)`` CNOT and
  ``X`` gates (an upper bound; the labels ``z >= d`` are free).
"""

import math
from collections import Counter

import pennylane.labs.estimator_beta as qre
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    _dequeue,
    resource_rep,
)
from pennylane.labs.estimator_beta.templates.state_prep import LabsSumOfSlatersPrep
from pennylane.wires import WiresLike

# pylint: disable=arguments-differ, too-many-arguments, too-many-positional-arguments, too-many-instance-attributes

_INIT_STATES = ("arbitrary", "uniform_real")


def _num_bits(dim: int) -> int:
    """Number of qubits used to store one qudit of dimension ``dim``."""
    return max(1, math.ceil(math.log2(dim)))


def _as_histogram(counter) -> tuple[tuple[int, int], ...]:
    """Canonical ``((weight, count), ...)`` tuple, sorted by weight, zero counts dropped."""
    return tuple(sorted((int(w), int(c)) for w, c in dict(counter).items() if c > 0))


def _validate_histogram(parity_weights, num_wires):
    hist = _as_histogram(dict(parity_weights))
    for w, _ in hist:
        if w < 1 or w > num_wires:
            raise ValueError(f"Parity weight {w} must lie between 1 and num_wires={num_wires}.")
    return hist


# ---------------------------------------------------------------------------------------------
# Classical preprocessing: gate sets -> parity-weight histograms
# ---------------------------------------------------------------------------------------------


def iqp_parity_weights_all_gates(num_wires: int, max_weight: int) -> tuple[tuple[int, int], ...]:
    r"""Parity-weight histogram of the qubit gate set with one gate on every support of size
    ``1, ..., max_weight``, as produced by :func:`~pennylane.labs.tcdq.create_local_gates`.

    Args:
        num_wires (int): number of qubits ``n``
        max_weight (int): maximum gate weight ``w``

    Returns:
        tuple[tuple[int, int]]: ``((k, binom(n, k)) for k = 1..w)``

    **Example**

    >>> iqp_parity_weights_all_gates(4, 2)
    ((1, 4), (2, 6))
    """
    return _as_histogram({k: math.comb(num_wires, k) for k in range(1, max_weight + 1)})


def iqp_parity_weights_arbitrary_gates(
    num_wires: int, num_gates: int, max_weight: int
) -> tuple[tuple[int, int], ...]:
    r"""Worst-case parity-weight histogram for ``num_gates`` arbitrary qubit gates of weight at most
    ``max_weight``.

    Gates on the same support merge, so at most ``binom(n, k)`` distinct weight-``k`` rotations
    exist. The worst case fills the highest weights first, which maximises the CNOT count; the
    rotation count is ``min(num_gates, sum_k binom(n, k))`` in every case.

    Args:
        num_wires (int): number of qubits ``n``
        num_gates (int): number of gates ``m`` in the gate set
        max_weight (int): maximum gate weight ``w``

    Returns:
        tuple[tuple[int, int]]: an upper bound on the parity-weight histogram

    **Example**

    >>> iqp_parity_weights_arbitrary_gates(10, 50, 2)
    ((1, 5), (2, 45))
    """
    remaining = num_gates
    hist = {}
    for k in range(min(max_weight, num_wires), 0, -1):
        take = min(remaining, math.comb(num_wires, k))
        hist[k] = take
        remaining -= take
    return _as_histogram(hist)


def iqp_parity_weights_from_gates(gates: dict) -> tuple[tuple[int, int], ...]:
    r"""Exact parity-weight histogram of a qubit ``gates`` dictionary in the format of
    :class:`~pennylane.labs.tcdq.CircuitConfig`.

    Repeated qubits within a gate count once, and gates with identical supports merge into one
    rotation (all gates commute).

    Args:
        gates (dict[int, list[list[int]]]): map from parameter index to a list of supports

    Returns:
        tuple[tuple[int, int]]: the parity-weight histogram

    **Example**

    >>> iqp_parity_weights_from_gates({0: [[0, 1]], 1: [[2]], 2: [[1, 0]]})
    ((1, 1), (2, 1))
    """
    supports = {frozenset(g) for gate_list in gates.values() for g in gate_list if g}
    return _as_histogram(Counter(len(s) for s in supports))


def qudit_site_spectrum(dim: int, power: int) -> dict[int, float]:
    r"""Walsh spectrum of the single-site eigenvalue function of :math:`Q(v)` on the encoding bits.

    The eigenvalue of :math:`Q(v) = \chi Z^v + \chi^* Z^{-v}` on :math:`|z\rangle` is
    :math:`\sqrt{2}\cos(2\pi v z / d + \pi/4)` for :math:`\chi = e^{i\pi/4}/\sqrt{2}`. On
    ``b = ceil(log2 d)`` bits it is expanded as :math:`\sum_s a_s (-1)^{s \cdot z}`, where
    :math:`s \cdot z` is the parity of the bits of ``z`` selected by the mask ``s``. Labels
    ``z >= d`` get the value ``0``.

    Args:
        dim (int): local dimension ``d``
        power (int): generator power ``v`` in ``1, ..., d - 1``

    Returns:
        dict[int, float]: non-zero Walsh coefficients ``{mask: a_mask}``

    **Example**

    >>> qudit_site_spectrum(4, 1)
    {3: 1.0}
    """
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


def _site_masks_union(dim):
    """Union over all powers of the non-constant masks appearing in the site spectra."""
    masks = set()
    for v in range(1, dim):
        masks |= set(qudit_site_spectrum(dim, v))
    masks.discard(0)
    return masks


def _dims_list(dims, num_qudits=None):
    if isinstance(dims, int):
        if num_qudits is None:
            raise ValueError("num_qudits is required when dims is an int.")
        return [dims] * num_qudits
    return list(dims)


def qudit_parities_from_gates(gates: dict, dims) -> dict[tuple, float]:
    r"""Merged parity expansion of a qudit gate set at unit angles.

    Each generator :math:`Q_g` is the tensor product of its site spectra; the products of all gates are summed per parity. The returned coefficients are those
    of :math:`\sum_j Q_{g_j}`; the parity set (keys) is what fixes the cost.

    Args:
        gates (dict[int, list[list[int]]]): map from parameter index to a list of generator
            vectors, in the format of :class:`~pennylane.labs.tcdq.QuditCircuitConfig`
        dims (int | list[int]): local dimensions; an ``int`` applies to every site

    Returns:
        dict[tuple[tuple[int, int]], float]: parity ``((site, mask), ...) -> coefficient``
    """
    flat = [list(g) for gate_list in gates.values() for g in gate_list]
    n = len(flat[0]) if flat else 0
    dims = _dims_list(dims, n)
    total = {}
    for g in flat:
        terms = {(): 1.0}
        for q, v in enumerate(g):
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


def qudit_parity_weights_from_gates(gates: dict, dims) -> tuple[tuple[int, int], ...]:
    r"""Exact parity-weight histogram of a qudit ``gates`` dictionary after merging equal parities.

    Args:
        gates (dict[int, list[list[int]]]): generator vectors, as in
            :class:`~pennylane.labs.tcdq.QuditCircuitConfig`
        dims (int | list[int]): local dimensions

    Returns:
        tuple[tuple[int, int]]: ``((qubit_weight, count), ...)``

    **Example**

    >>> qudit_parity_weights_from_gates({0: [[1, 0]], 1: [[1, 3]]}, 4)
    ((2, 1), (3, 1))
    """
    parities = qudit_parities_from_gates(gates, dims)
    hist = Counter(sum(bin(m).count("1") for _, m in key) for key in parities)
    return _as_histogram(hist)


def _site_weight_poly(masks):
    """Counter {qubit weight: number of masks}."""
    return Counter(bin(m).count("1") for m in masks)


def qudit_parity_weights_all_gates(
    dims, max_weight: int, num_qudits: int | None = None
) -> tuple[tuple[int, int], ...]:
    r"""Parity-weight histogram of the qudit gate set containing every generator with at most
    ``max_weight`` non-zero entries, each in ``1, ..., d_q - 1``.

    This is the gate set behind the parameter counts of Spectral Born machines Sec. 5
    (e.g. ``95 x 3 + binom(95, 2) x 9 = 40470`` for ``d = 4``). After merging, the parities are
    all products over at most ``max_weight`` sites of one non-zero mask per site drawn from the
    union of the site spectra.

    Args:
        dims (int | list[int]): local dimensions; an ``int`` requires ``num_qudits``
        max_weight (int): maximum number of qudits a gate acts on
        num_qudits (int | None): number of qudits when ``dims`` is an ``int``

    Returns:
        tuple[tuple[int, int]]: ``((qubit_weight, count), ...)``

    **Example**

    >>> qudit_parity_weights_all_gates(4, 2, num_qudits=3)
    ((1, 6), (2, 15), (3, 12), (4, 3))
    """
    dims = _dims_list(dims, num_qudits)
    polys = {d: _site_weight_poly(_site_masks_union(d)) for d in set(dims)}
    # dp[k] = Counter{qubit weight: count} over parities touching exactly k sites
    dp = [Counter({0: 1})] + [Counter() for _ in range(max_weight)]
    for d in dims:
        poly = polys[d]
        for k in range(max_weight, 0, -1):
            for w0, c0 in dp[k - 1].items():
                for w1, c1 in poly.items():
                    dp[k][w0 + w1] += c0 * c1
    hist = Counter()
    for k in range(1, max_weight + 1):
        hist.update(dp[k])
    return _as_histogram(hist)


def qudit_parity_weights_arbitrary_gates(
    dim: int, num_qudits: int, num_gates: int, max_weight: int
) -> tuple[tuple[int, int], ...]:
    r"""Upper bound on the parity-weight histogram of ``num_gates`` arbitrary qudit gates of weight
    at most ``max_weight``, all sites of dimension ``dim``.

    One gate is bounded by the worst generator: ``max_weight`` sites, each with the power whose
    site spectrum has the most terms (ties broken by total qubit weight). The per-weight count is
    then capped by the all-gates histogram, since every parity of any such gate set appears there.

    Args:
        dim (int): local dimension ``d`` of every site
        num_qudits (int): number of qudits ``n``
        num_gates (int): number of gates ``m``
        max_weight (int): maximum gate weight ``w``

    Returns:
        tuple[tuple[int, int]]: an upper bound on ``((qubit_weight, count), ...)``

    **Example**

    >>> qudit_parity_weights_arbitrary_gates(4, 10, 5, 2)
    ((4, 5),)
    """
    w = min(max_weight, num_qudits)

    def score(v):
        spec = [m for m in qudit_site_spectrum(dim, v) if m]
        return (len(spec), sum(bin(m).count("1") for m in spec))

    v_star = max(range(1, dim), key=score)
    site = _site_weight_poly(m for m in qudit_site_spectrum(dim, v_star) if m)
    gate = Counter({0: 1})
    for _ in range(w):
        new = Counter()
        for w0, c0 in gate.items():
            for w1, c1 in site.items():
                new[w0 + w1] += c0 * c1
        gate = new
    cap = dict(qudit_parity_weights_all_gates(dim, w, num_qudits=num_qudits))
    return _as_histogram({k: min(num_gates * c, cap.get(k, 0)) for k, c in gate.items()})


# ---------------------------------------------------------------------------------------------
# Resource operators
# ---------------------------------------------------------------------------------------------


class ParityPhaseLayer(ResourceOperator):
    r"""Resource class for a diagonal layer of commuting Pauli-Z rotations.

    The layer implements :math:`\prod_P \exp(i c_P Z_P)` where every parity :math:`Z_P` is a
    tensor product of Pauli-Z operators. Each term is one
    :class:`~.pennylane.estimator.MultiRZ` on ``|P|`` wires (``2(|P| - 1)`` CNOTs and one ``RZ``).

    Args:
        num_wires (int): number of wires of the layer
        parity_weights (tuple[tuple[int, int]]): ``((weight, count), ...)``: the number of
            distinct parities of each Pauli weight. An architecture-independent description of
            the gate set, produced by the ``*_parity_weights_*`` helpers of this module.
        precision (float | None): synthesis precision per rotation. ``None`` uses the
            configured ``MultiRZ``/``RZ`` precision. Architecture knob.
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        One ``MultiRZ(weight)`` per parity. The adjoint and powers negate or scale the angles and
        cost the same. One control is absorbed into the rotations,
        :math:`C[\exp(i c Z_P)] = \exp(i \tfrac{c}{2} Z_P)\exp(-i \tfrac{c}{2} Z_c Z_P)`; more
        controls are first reduced to one with a ``MultiControlledX`` into a clean auxiliary qubit.

    **Example**

    >>> import pennylane.labs.estimator_beta as qre
    >>> layer = qre.ParityPhaseLayer(4, ((1, 4), (2, 6)))
    >>> print(dict(qre.estimate(layer).gate_counts))
    {'CNOT': 12, 'T': 440}
    """

    resource_keys = {"num_wires", "parity_weights", "precision"}

    def __init__(
        self,
        num_wires: int,
        parity_weights,
        precision: float | None = None,
        wires: WiresLike | None = None,
    ):
        self.num_wires = num_wires
        self.parity_weights = _validate_histogram(parity_weights, num_wires)
        self.precision = precision
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int): number of wires of the layer
                * parity_weights (tuple[tuple[int, int]]): number of parities per Pauli weight
                * precision (float | None): synthesis precision per rotation
        """
        return {
            "num_wires": self.num_wires,
            "parity_weights": self.parity_weights,
            "precision": self.precision,
        }

    @classmethod
    def resource_rep(
        cls, num_wires: int, parity_weights, precision: float | None = None
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        params = {
            "num_wires": num_wires,
            "parity_weights": _validate_histogram(parity_weights, num_wires),
            "precision": precision,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, num_wires: int, parity_weights, precision: float | None = None):
        r"""Returns a list of ``GateCount`` objects: one ``MultiRZ`` per parity.

        Args:
            num_wires (int): number of wires of the layer
            parity_weights (tuple[tuple[int, int]]): number of parities per Pauli weight
            precision (float | None): synthesis precision per rotation

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: the decomposition
        """
        del num_wires
        return [
            GateCount(resource_rep(qre.MultiRZ, {"num_wires": w, "precision": precision}), c)
            for w, c in parity_weights
        ]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None):
        r"""The adjoint negates every angle and costs the same as the layer.

        Args:
            target_resource_params (dict): resource parameters of the layer

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: the decomposition
        """
        return [GateCount(cls.resource_rep(**target_resource_params))]

    @classmethod
    def pow_resource_decomp(cls, pow_z: int, target_resource_params: dict | None = None):
        r"""A power scales every angle and costs the same as the layer (identity for ``z = 0``).

        Args:
            pow_z (int): the exponent
            target_resource_params (dict): resource parameters of the layer

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: the decomposition
        """
        if pow_z == 0:
            return [GateCount(resource_rep(qre.Identity))]
        return [GateCount(cls.resource_rep(**target_resource_params))]

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires: int, num_zero_ctrl: int, target_resource_params: dict | None = None
    ):
        r"""Controlled layer: two rotations per parity for one control, plus an
        ``MultiControlledX`` compute/uncompute pair into one auxiliary qubit for more controls.

        Args:
            num_ctrl_wires (int): number of control wires
            num_zero_ctrl (int): number of controls conditioned on :math:`|0\rangle`
            target_resource_params (dict): resource parameters of the layer

        Returns:
            list: the decomposition
        """
        precision = target_resource_params["precision"]
        gates = []
        for w, c in target_resource_params["parity_weights"]:
            for ww in (w, w + 1):
                rz = resource_rep(qre.MultiRZ, {"num_wires": ww, "precision": precision})
                gates.append(GateCount(rz, c))
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


class QuditFourierTransform(ResourceOperator):
    r"""Resource class for the Fourier transform :math:`F_d` over :math:`\mathbb{Z}_d` on one
    binary-encoded qudit.

    :math:`F_d |x\rangle = d^{-1/2}\sum_k e^{2\pi i k x / d}|k\rangle` acts on
    ``b = ceil(log2 d)`` qubits.

    Args:
        dim (int): local dimension ``d``
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        ``Hadamard`` for ``d = 2`` and :class:`~.pennylane.estimator.QFT` on ``b`` qubits for
        ``d = 2^b``. For even ``d = 2^k q`` with ``q > 1`` odd, the prime-factor (Good-Thomas)
        factorisation :math:`F_d = \Pi_{\text{out}} (F_{2^k} \otimes F_q) \Pi_{\text{in}}`:
        two label permutations (``d - 1`` transpositions each, one ``MultiControlledX`` on
        ``b - 1`` controls plus ``2(b - 1)`` CNOT and ``X`` gates per transposition), the
        power-of-two transform on ``k`` qubits and :math:`F_q` on ``ceil(log2 q)`` qubits. For
        odd ``d`` a generic :class:`~.pennylane.estimator.QubitUnitary` on ``b`` qubits (an
        embedding of the ``d x d`` unitary, identity on the unused labels). The adjoint costs the
        same. All variants are exact and deterministic.

    **Example**

    >>> import pennylane.labs.estimator_beta as qre
    >>> print(dict(qre.estimate(qre.QuditFourierTransform(4)).gate_counts))
    {'Hadamard': 2, 'CNOT': 5, 'T': 132}
    >>> print(dict(qre.estimate(qre.QuditFourierTransform(6)).gate_counts))
    {'Toffoli': 10, 'CNOT': 43, 'X': 40, 'Hadamard': 1, 'T': 660}
    """

    resource_keys = {"dim"}

    def __init__(self, dim: int, wires: WiresLike | None = None):
        if dim < 2:
            raise ValueError(f"dim must be at least 2, got {dim}.")
        self.dim = dim
        self.num_wires = _num_bits(dim)
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * dim (int): local dimension
        """
        return {"dim": self.dim}

    @classmethod
    def resource_rep(cls, dim: int) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, _num_bits(dim), {"dim": dim})

    @classmethod
    def resource_decomp(cls, dim: int):
        r"""Returns a list of ``GateCount`` objects for :math:`F_d`.

        Args:
            dim (int): local dimension

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: the decomposition
        """
        b = _num_bits(dim)
        if dim == 2:
            return [GateCount(resource_rep(qre.Hadamard))]
        if dim == 2**b:
            return [GateCount(resource_rep(qre.QFT, {"num_wires": b}))]
        two_power, odd = _crt_factors(dim)
        if two_power == 1:
            return [GateCount(resource_rep(qre.QubitUnitary, {"num_wires": b, "precision": None}))]
        # Good-Thomas: F_d = P_out (F_{2^k} (x) F_q) P_in on the k + ceil(log2 q) = b qubits
        mcx = resource_rep(qre.MultiControlledX, {"num_ctrl_wires": b - 1, "num_zero_ctrl": 0})
        transpositions = 2 * (dim - 1)
        return [
            GateCount(mcx, transpositions),
            GateCount(resource_rep(qre.CNOT), 2 * (b - 1) * transpositions),
            GateCount(resource_rep(qre.X), 2 * (b - 1) * transpositions),
            GateCount(cls.resource_rep(two_power)),
            GateCount(cls.resource_rep(odd)),
        ]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None):
        r""":math:`F_d^\dagger` costs the same as :math:`F_d`.

        Args:
            target_resource_params (dict): resource parameters of the transform

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: the decomposition
        """
        return [GateCount(cls.resource_rep(**target_resource_params))]


def _crt_factors(dim: int) -> tuple[int, int]:
    """Split ``dim = 2^k * q`` with ``q`` odd; returns ``(2^k, q)``."""
    two_power = dim & -dim
    return two_power, dim // two_power


def _canonical_dims(qudit_dims, num_qudits=None):
    """``((d, number of sites), ...)`` from a histogram, a per-site list, or an ``int``."""
    if isinstance(qudit_dims, tuple) and qudit_dims and isinstance(qudit_dims[0], tuple):
        return _as_histogram(dict(qudit_dims))
    return _as_histogram(Counter(_dims_list(qudit_dims, num_qudits)))


def _adder_tree(qudit_dims) -> tuple[int, int, int]:
    """Full adders, half adders and output bits of a carry-save tree summing the qudit labels.

    Column ``j`` holds the bits of weight ``2^j``. A full adder turns three bits of a column into
    one bit of the same column and one carry in the next; a half adder does the same for two.
    """
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
    r"""Resource class for a diagonal phase that depends on a basis state only through the sum of
    its qudit labels.

    The layer implements :math:`|z\rangle \mapsto e^{i f(S(z))}|z\rangle` with
    :math:`S(z) = \sum_q z_q`, the Hamming weight for qubits. Phase functions of the mean or a
    polynomial in the mean of ``z``, as used with ``phase_fn`` in ``pennylane.labs.tcdq``, have
    this form.

    Args:
        qudit_dims (tuple[tuple[int, int]] | list[int] | int): local dimensions as
            ``((d, number of sites), ...)``, a per-site list, or an ``int`` together with
            ``num_qudits``; ``d = 2`` for qubits
        degree (int | None): degree ``t`` of :math:`f` as a polynomial in :math:`S`. ``None``
            allows an arbitrary function of :math:`S`.
        precision (float | None): synthesis precision per rotation. ``None`` uses the configured
            ``MultiRZ``/``RZ`` precision.
        num_qudits (int | None): number of sites when ``qudit_dims`` is an ``int``
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        The sum is computed into ``s`` bits by a carry-save tree of full adders (three bits of
        equal weight to a sum bit in place and a carry into a fresh qubit: five ``CNOT`` and one
        :class:`~.pennylane.estimator.TemporaryAND`) and half adders (two bits: one ``CNOT`` and
        one ``TemporaryAND``), as in Hamming-weight phasing
        (`Kivlichan et al., Quantum 4, 296 (2020) <https://arxiv.org/abs/1902.10673>`__). For
        ``n`` qubits this is ``n - popcount(n)`` ``TemporaryAND`` gates and as many auxiliary
        qubits, with ``s = ceil(log2(n + 1))``. The phase is a :class:`~.ParityPhaseLayer` on the
        ``s`` sum bits: a polynomial of degree ``t`` has Walsh terms of weight at most ``t``, so at
        most :math:`\sum_{k=1}^{\min(t, s)} \binom{s}{k}` rotations (``2^s - 1`` for an arbitrary
        :math:`f`). The tree is then uncomputed; the adjoint ``TemporaryAND`` is measurement
        based and costs no Toffoli
        (`Gidney, Quantum 2, 74 (2018) <https://arxiv.org/abs/1709.06648>`__).

    **Example**

    >>> import pennylane.labs.estimator_beta as qre
    >>> layer = qre.SumPhaseLayer(2, degree=3, num_qudits=100)
    >>> res = qre.estimate(layer)
    >>> res.gate_counts["Toffoli"], res.gate_counts["T"]
    (97, 2772)
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
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * qudit_dims (tuple[tuple[int, int]]): local dimensions and their multiplicities
                * degree (int | None): polynomial degree of the phase function
                * precision (float | None): synthesis precision per rotation
        """
        return {"qudit_dims": self.qudit_dims, "degree": self.degree, "precision": self.precision}

    @classmethod
    def resource_rep(
        cls, qudit_dims, degree: int | None = None, precision: float | None = None
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        qudit_dims = _canonical_dims(qudit_dims)
        num_wires = sum(_num_bits(d) * c for d, c in qudit_dims)
        params = {"qudit_dims": qudit_dims, "degree": degree, "precision": precision}
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, qudit_dims, degree: int | None = None, precision: float | None = None):
        r"""Returns a list of ``GateCount`` objects: the adder tree, its uncomputation and the
        phase on the sum register.

        Args:
            qudit_dims (tuple[tuple[int, int]]): local dimensions and their multiplicities
            degree (int | None): polynomial degree of the phase function
            precision (float | None): synthesis precision per rotation

        Returns:
            list: the decomposition
        """
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
        r"""The adjoint negates the phase and costs the same as the layer.

        Args:
            target_resource_params (dict): resource parameters of the layer

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: the decomposition
        """
        return [GateCount(cls.resource_rep(**target_resource_params))]


def _input_state_decomp(num_wires, num_init_states, init_state, num_bits, select_swap_depth):
    """Gate list preparing an ``N``-sparse input state on ``num_wires`` qubits."""
    if init_state not in _INIT_STATES:
        raise ValueError(f"init_state must be one of {_INIT_STATES}, got {init_state!r}.")
    if num_init_states == 1:
        return [GateCount(resource_rep(qre.BasisState, {"num_wires": num_wires}))]
    k = math.ceil(math.log2(num_init_states))
    if init_state == "arbitrary":
        condensed = resource_rep(
            qre.QROMStatePreparation,
            {
                "num_state_qubits": k,
                "precision": None,
                "positive_and_real": False,
                "selswap_depths": 1,
            },
        )
    else:
        condensed = resource_rep(qre.UniformStatePrep, {"num_states": num_init_states})
    sos = resource_rep(
        LabsSumOfSlatersPrep,
        {
            "num_coeffs": num_init_states,
            "num_wires": num_wires,
            "num_bits": num_wires if num_bits is None else num_bits,
            "stateprep_cmpr_op": condensed,
            "select_swap_depth": select_swap_depth,
        },
    )
    return [GateCount(sos)]


def _validate_state(num_wires, num_init_states, init_state, num_bits):
    if init_state not in _INIT_STATES:
        raise ValueError(f"init_state must be one of {_INIT_STATES}, got {init_state!r}.")
    if not 1 <= num_init_states <= 2**num_wires:
        raise ValueError(f"num_init_states must lie in 1..2^{num_wires}, got {num_init_states}.")
    if num_bits is not None and not 1 <= num_bits <= num_wires:
        raise ValueError(f"num_bits must lie in 1..{num_wires}, got {num_bits}.")


class IQPSamplingShot(ResourceOperator):
    r"""Resource class for one computational-basis shot of a parameterised qubit IQP circuit.

    The shot prepares :math:`|\psi_{\text{in}}\rangle = \sum_{i=1}^N \psi_i |x_i\rangle`, applies
    :math:`U(\theta) = H^{\otimes n} D(\theta) H^{\otimes n}` with
    :math:`D(\theta) = \prod_j \exp(i\theta_j Z_{g_j})`
    (`IQPopt, arXiv:2501.04776 <https://arxiv.org/abs/2501.04776>`__), optionally followed by a
    trainable phase layer :math:`D'(\xi)` priced by ``phase_op``, and measures every qubit.

    Args:
        num_wires (int): number of qubits ``n``. Physics input.
        parity_weights (tuple[tuple[int, int]]): number of distinct gate supports per weight. Physics
            input, built by :func:`iqp_parity_weights_all_gates` (all gates up to weight ``w``),
            :func:`iqp_parity_weights_arbitrary_gates` (``m`` arbitrary gates of weight at most
            ``w``) or :func:`iqp_parity_weights_from_gates` (an explicit ``gates`` dictionary).
        num_init_states (int): number ``N`` of basis states in the input state. Physics input.
            ``N = 1`` is a basis state (the ``tcdq`` default :math:`|0\rangle`).
        init_state (str): ``"arbitrary"`` for arbitrary complex amplitudes or ``"uniform_real"``
            for :math:`\psi_i = 1/\sqrt{N}`. Physics input.
        num_bits (int | None): number of rows of the ``N x n`` bit matrix needed to tell the
            basis states apart (``len(select_sos_rows(bits)[0])`` for known states). ``None``
            assumes ``n``, the data-independent upper bound. Physics input.
        select_swap_depth (int | None): ``LabsQROM`` select-swap depth of the input-state QROM.
            Architecture knob.
        precision (float | None): synthesis precision per phase-layer rotation. Architecture knob.
        phase_op (ResourceOperator | None): the trainable phase layer :math:`D'(\xi)`, e.g. a
            :class:`~.SumPhaseLayer`. ``None`` prices the circuit without it. Physics input.
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        :class:`~.LabsSumOfSlatersPrep` (Fomichev et al. Sec. III A) for the input state, ``2n``
        ``Hadamard`` gates, and a :class:`~.ParityPhaseLayer` with one ``MultiRZ`` per gate
        support, plus ``phase_op`` if given. The measurement is free. ``parity_weights`` from
        :func:`iqp_parity_weights_arbitrary_gates` is an upper bound; the other two helpers are
        exact.

    .. seealso:: :func:`~pennylane.labs.tcdq.shot_resources` to build the shot from a
        :class:`~pennylane.labs.tcdq.CircuitConfig`, :class:`~.QuditSamplingShot`

    **Example**

    >>> import pennylane.labs.estimator_beta as qre
    >>> pw = ((1, 20), (2, 190))  # every gate of weight 1 and 2 on 20 qubits
    >>> shot = qre.IQPSamplingShot(20, pw, num_init_states=64, init_state="uniform_real")
    >>> res = qre.estimate(shot)
    >>> res.algo_wires, res.gate_counts["T"]
    (20, 9240)
    """

    resource_keys = {
        "num_wires",
        "parity_weights",
        "num_init_states",
        "init_state",
        "num_bits",
        "select_swap_depth",
        "precision",
        "phase_cmpr_op",
    }

    def __init__(
        self,
        num_wires: int,
        parity_weights,
        num_init_states: int,
        init_state: str,
        num_bits: int | None = None,
        select_swap_depth: int | None = None,
        precision: float | None = None,
        phase_op: ResourceOperator | None = None,
        wires: WiresLike | None = None,
    ):
        _validate_state(num_wires, num_init_states, init_state, num_bits)
        self.num_wires = num_wires
        self.parity_weights = _validate_histogram(parity_weights, num_wires)
        self.num_init_states = num_init_states
        self.init_state = init_state
        self.num_bits = num_bits
        self.select_swap_depth = select_swap_depth
        self.precision = precision
        if phase_op is not None:
            _dequeue(phase_op)
        self.phase_cmpr_op = phase_op.resource_rep_from_op() if phase_op is not None else None
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters ``num_wires``,
            ``parity_weights``, ``num_init_states``, ``init_state``, ``num_bits``,
            ``select_swap_depth``, ``precision`` and ``phase_cmpr_op`` (see the class docstring).
        """
        return {k: getattr(self, k) for k in self.resource_keys}

    @classmethod
    def resource_rep(
        cls,
        num_wires: int,
        parity_weights,
        num_init_states: int,
        init_state: str,
        num_bits: int | None = None,
        select_swap_depth: int | None = None,
        precision: float | None = None,
        phase_cmpr_op: CompressedResourceOp | None = None,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        _validate_state(num_wires, num_init_states, init_state, num_bits)
        params = {
            "num_wires": num_wires,
            "parity_weights": _validate_histogram(parity_weights, num_wires),
            "num_init_states": num_init_states,
            "init_state": init_state,
            "num_bits": num_bits,
            "select_swap_depth": select_swap_depth,
            "precision": precision,
            "phase_cmpr_op": phase_cmpr_op,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        num_wires: int,
        parity_weights,
        num_init_states: int,
        init_state: str,
        num_bits: int | None = None,
        select_swap_depth: int | None = None,
        precision: float | None = None,
        phase_cmpr_op: CompressedResourceOp | None = None,
    ):
        r"""Returns a list of ``GateCount`` objects for one shot.

        Args:
            See the class docstring.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: the decomposition
        """
        gates = _input_state_decomp(
            num_wires, num_init_states, init_state, num_bits, select_swap_depth
        )
        layer = ParityPhaseLayer.resource_rep(num_wires, parity_weights, precision)
        gates.append(GateCount(resource_rep(qre.Hadamard), 2 * num_wires))
        gates.append(GateCount(layer))
        if phase_cmpr_op is not None:
            gates.append(GateCount(phase_cmpr_op))
        return gates


class QuditSamplingShot(ResourceOperator):
    r"""Resource class for one computational-basis shot of a parameterised qudit circuit
    (spectral Born machine), compiled to qubits.

    The shot prepares an ``N``-sparse input state, applies
    :math:`U(\theta) = F_{\mathbf{d}}^\dagger D(\theta) F_{\mathbf{d}}` with
    :math:`D(\theta) = \prod_j \exp(i\theta_j Q_{g_j})`
    (`Spectral Born machines, arXiv:2607.06675 <https://arxiv.org/abs/2607.06675>`__), optionally
    followed by a trainable phase layer :math:`D'(\xi)` priced by ``phase_op``, and measures every
    qubit; the measured bits are the binary labels of the qudits.

    Args:
        qudit_dims (tuple[tuple[int, int]] | list[int] | int): local dimensions as
            ``((d, number of sites), ...)``, a per-site list, or an ``int`` together with
            ``num_qudits``. Physics input.
        parity_weights (tuple[tuple[int, int]]): number of distinct Walsh parities per qubit weight
            after merging. Physics input, built by :func:`qudit_parity_weights_all_gates`,
            :func:`qudit_parity_weights_arbitrary_gates` (upper bound) or
            :func:`qudit_parity_weights_from_gates`.
        num_init_states (int): number ``N`` of basis states in the input state. Physics input.
        init_state (str): ``"arbitrary"`` or ``"uniform_real"``. Physics input.
        num_bits (int | None): distinguishing rows of the encoded bit matrix; ``None`` assumes all
            encoded qubits. Physics input.
        select_swap_depth (int | None): ``LabsQROM`` select-swap depth. Architecture knob.
        precision (float | None): synthesis precision per phase-layer rotation. Architecture knob.
        phase_op (ResourceOperator | None): the trainable phase layer :math:`D'(\xi)`, e.g. a
            :class:`~.SumPhaseLayer`. ``None`` prices the circuit without it. Physics input.
        num_qudits (int | None): number of sites when ``qudit_dims`` is an ``int``
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        :class:`~.LabsSumOfSlatersPrep` on the ``sum_q ceil(log2 d_q)`` encoding qubits, one
        :class:`~.QuditFourierTransform` and one adjoint per site, and a
        :class:`~.ParityPhaseLayer` for the Walsh-expanded diagonal layer, plus ``phase_op`` if
        given. The measurement is free.

    .. seealso:: :func:`~pennylane.labs.tcdq.shot_resources` to build the shot from a
        :class:`~pennylane.labs.tcdq.QuditCircuitConfig`, :class:`~.IQPSamplingShot`

    **Example**

    >>> import pennylane.labs.estimator_beta as qre
    >>> pw = qre.qudit_parity_weights_from_gates({0: [[1, 0, 0]], 1: [[0, 2, 3]]}, 4)
    >>> shot = qre.QuditSamplingShot(4, pw, num_init_states=1, init_state="arbitrary", num_qudits=3)
    >>> qre.estimate(shot).algo_wires
    6
    """

    resource_keys = {
        "qudit_dims",
        "parity_weights",
        "num_init_states",
        "init_state",
        "num_bits",
        "select_swap_depth",
        "precision",
        "phase_cmpr_op",
    }

    def __init__(
        self,
        qudit_dims,
        parity_weights,
        num_init_states: int,
        init_state: str,
        num_bits: int | None = None,
        select_swap_depth: int | None = None,
        precision: float | None = None,
        phase_op: ResourceOperator | None = None,
        num_qudits: int | None = None,
        wires: WiresLike | None = None,
    ):
        self.qudit_dims = _canonical_dims(qudit_dims, num_qudits)
        _validate_state(self.num_wires, num_init_states, init_state, num_bits)
        self.parity_weights = _validate_histogram(parity_weights, self.num_wires)
        self.num_init_states = num_init_states
        self.init_state = init_state
        self.num_bits = num_bits
        self.select_swap_depth = select_swap_depth
        self.precision = precision
        if phase_op is not None:
            _dequeue(phase_op)
        self.phase_cmpr_op = phase_op.resource_rep_from_op() if phase_op is not None else None
        super().__init__(wires=wires)

    @property
    def num_wires(self) -> int:
        """Number of qubits encoding the qudits."""
        return self._num_qubits(self.qudit_dims)

    @staticmethod
    def _num_qubits(qudit_dims):
        return sum(_num_bits(d) * c for d, c in qudit_dims)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters ``qudit_dims``,
            ``parity_weights``, ``num_init_states``, ``init_state``, ``num_bits``,
            ``select_swap_depth``, ``precision`` and ``phase_cmpr_op`` (see the class docstring).
        """
        return {k: getattr(self, k) for k in self.resource_keys}

    @classmethod
    def resource_rep(
        cls,
        qudit_dims,
        parity_weights,
        num_init_states: int,
        init_state: str,
        num_bits: int | None = None,
        select_swap_depth: int | None = None,
        precision: float | None = None,
        phase_cmpr_op: CompressedResourceOp | None = None,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        qudit_dims = _canonical_dims(qudit_dims)
        num_wires = cls._num_qubits(qudit_dims)
        _validate_state(num_wires, num_init_states, init_state, num_bits)
        params = {
            "qudit_dims": qudit_dims,
            "parity_weights": _validate_histogram(parity_weights, num_wires),
            "num_init_states": num_init_states,
            "init_state": init_state,
            "num_bits": num_bits,
            "select_swap_depth": select_swap_depth,
            "precision": precision,
            "phase_cmpr_op": phase_cmpr_op,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        qudit_dims,
        parity_weights,
        num_init_states: int,
        init_state: str,
        num_bits: int | None = None,
        select_swap_depth: int | None = None,
        precision: float | None = None,
        phase_cmpr_op: CompressedResourceOp | None = None,
    ):
        r"""Returns a list of ``GateCount`` objects for one shot.

        Args:
            See the class docstring.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: the decomposition
        """
        num_wires = cls._num_qubits(qudit_dims)
        gates = _input_state_decomp(
            num_wires, num_init_states, init_state, num_bits, select_swap_depth
        )
        for d, count in qudit_dims:
            ft = QuditFourierTransform.resource_rep(d)
            gates.append(GateCount(ft, count))
            gates.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": ft}), count))
        gates.append(GateCount(ParityPhaseLayer.resource_rep(num_wires, parity_weights, precision)))
        if phase_cmpr_op is not None:
            gates.append(GateCount(phase_cmpr_op))
        return gates


__all__ = [
    "ParityPhaseLayer",
    "QuditFourierTransform",
    "SumPhaseLayer",
    "IQPSamplingShot",
    "QuditSamplingShot",
    "iqp_parity_weights_all_gates",
    "iqp_parity_weights_arbitrary_gates",
    "iqp_parity_weights_from_gates",
    "qudit_site_spectrum",
    "qudit_parities_from_gates",
    "qudit_parity_weights_all_gates",
    "qudit_parity_weights_arbitrary_gates",
    "qudit_parity_weights_from_gates",
]
