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
"""Thumbnails for the state-preparation / THC / Trotter group of decomposition tables."""

from common import *  # noqa

S = 34  # wire spacing


def box(f, x, y0, y1=None, label="", fill=CY2, w=34, size=15):
    """A labelled gate box, in the style of Select's thumbnails."""
    return f.gate(x, y0, y1, label=label, w=w, fill=fill, pad=13, size=size)


def wires(f, ys, x0, x1):
    for y in ys:
        f.wire(y, x0, x1)


def ctrl_box(f, x, cys, ty0, ty1=None, label="", fill=CY2, w=34, on=None, size=15):
    """A box on ``ty0..ty1`` controlled by dots on ``cys``."""
    ty1 = ty0 if ty1 is None else ty1
    f.vline(x, min(cys + [ty0]), max(cys + [ty1]))
    for i, y in enumerate(cys):
        f.ctrl(x, y, on=True if on is None else on[i], r=4.6)
    box(f, x, ty0, ty1, label, fill, w, size)


def mux_box(f, x, cys, ty, label, fill=CY2, w=34):
    """A multiplexed (uniformly controlled) gate: square controls on ``cys``."""
    f.vline(x, min(cys + [ty]), max(cys + [ty]))
    for y in cys:
        f.sqctrl(x, y)
    box(f, x, ty, None, label, fill, w)


def phase_box(f, x, y0, y1, label, w=34, size=13):
    """A global phase. It acts on no wire, but it is drawn across all wires ``y0..y1``."""
    box(f, x, y0, y1, label, GY0, w, size)


# ------------------------------------------------------------------ MultiplexerStatePreparation


def _msp(f, complex_):
    ys = [0, S, 2 * S]
    x1 = 230 if complex_ else 175
    wires(f, ys, 0, x1)
    box(f, 35, ys[0], None, "$R_Y$", CY2, w=36)
    mux_box(f, 85, ys[:1], ys[1], "$R_Y$", MG2, w=36)
    mux_box(f, 135, ys[:2], ys[2], "$R_Y$", YL2, w=36)
    if complex_:
        box(f, 192, ys[0], ys[2], "$D$", PU2, w=34)
    wire_labels(f, -8, [(ys[0], "$w_0$"), (ys[1], "$w_1$"), (ys[2], "$w_2$")])


@drawing("multiplexer_state_prep_real")
def _(f):
    _msp(f, False)


@drawing("multiplexer_state_prep_complex")
def _(f):
    _msp(f, True)


# ------------------------------------------------------------------ MPSPrep


@drawing("mps_prep")
def _(f):
    w, q0, q1, q2 = 0, S + 6, 2 * S + 6, 3 * S + 6
    wires(f, [w, q0, q1, q2], 0, 215)
    fills = [CY2, MG2, YL2]
    for i, (x, q) in enumerate(zip((40, 105, 170), (q0, q1, q2))):
        f.vline(x, w, q)
        box(f, x, w, None, f"$U_{i}$", fills[i])
        box(f, x, q, None, f"$U_{i}$", fills[i])
    wire_labels(f, -8, [(w, "work"), (q0, "$w_0$"), (q1, "$w_1$"), (q2, "$w_2$")])


# ------------------------------------------------------------------ SumOfSlatersPrep


@drawing("sum_of_slaters_prep")
def _(f):
    enum, sys_, ident, cache = 0, S + 4, 2 * S + 8, 3 * S + 8
    x1 = 475
    wires(f, [enum, sys_, ident, cache], 0, x1)
    for y, lab in ((enum, "d"), (sys_, "n"), (ident, "m")):
        f.bundle(12, y, lab)
    box(f, 60, enum, None, "MSP", CY2, w=44, size=13)
    f.vline(115, enum, sys_)
    f.sqctrl(115, enum)
    box(f, 115, sys_, None, "QROM", MG2, w=50, size=13)
    box(f, 172, sys_, ident, "Enc", YL2, w=40, size=13)
    # one uncomputation round: X flips, AND ladder onto the cache, CNOT onto enum, AND† ladder
    box(f, 218, ident, None, "X", GY0, w=26)
    box(f, 262, ident, cache, "AND", PU2, w=42, size=13)
    f.vline(310, enum, cache)
    f.ctrl(310, cache, r=4.6)
    f.target(310, enum, r=8)
    box(f, 358, ident, cache, r"AND$^\dagger$", PU2, w=46, size=13)
    f.hdots(398, ident)
    f.hdots(398, enum)
    f.hdots(398, cache)
    box(f, 440, sys_, ident, "Enc", YL2, w=40, size=13)
    wire_labels(f, -8, [(enum, "enum"), (sys_, "system"), (ident, "ident"), (cache, "cache")])


# ------------------------------------------------------------------ PartialUnaryStatePreparation


@drawing("partial_unary_state_prep")
def _(f):
    sub, rest, work = 0, S + 6, 2 * S + 12
    x1 = 390
    wires(f, [sub, rest, work], 0, x1)
    f.bundle(12, sub, "m")
    f.bundle(12, rest, "n−m")
    box(f, 62, sub, None, "MSP", CY2, w=44, size=13)
    # partial unary iteration: shift, QROM, unshift
    box(f, 112, sub, None, "X", GY0, w=26)
    f.vline(158, sub, rest)
    f.sqctrl(158, sub)
    box(f, 158, rest, work, "QROM", MG2, w=50, size=13)
    box(f, 204, sub, None, "X", GY0, w=26)
    # fan-out: controlled MultiX
    f.vline(248, sub, rest)
    f.ctrl(248, rest, r=4.6)
    box(f, 248, sub, None, "X", YL2, w=26)
    # SWAP
    f.vline(288, sub, rest)
    _swap_marks(f, 288, [sub, rest])
    # Toffoli
    box(f, 330, sub, rest, "CCX", PU2, w=40, size=13)
    f.hdots(368, sub)
    f.hdots(368, rest)
    wire_labels(f, -8, [(sub, "subspace"), (rest, "rest"), (work, "work")])


def _swap_marks(f, x, ys):
    for y in ys:
        f.line(x - 6, y - 6, x + 6, y + 6, z=5, wobble=0)
        f.line(x - 6, y + 6, x + 6, y - 6, z=5, wobble=0)


# ------------------------------------------------------------------ UniformPrep


@drawing("uniform_prep_amplification")
def _(f):
    t0, t1, t2, flag = 0, S, 2 * S, 3 * S + 4
    ts = [t0, t1, t2]
    x1 = 400
    wires(f, ts + [flag], 0, x1)

    def hlayer(x):
        for y in ts:
            box(f, x, y, None, "H", CY2, w=28)

    hlayer(30)
    ctrl_box(f, 82, [], t0, t2, "", MG2, w=34)
    f.vline(82, t2, flag)
    f.target(82, flag, r=8)
    f.text(82, t1, "<5", size=13)
    box(f, 130, flag, None, "$R_Z$", YL2, w=36)
    ctrl_box(f, 178, [], t0, t2, "", MG2, w=34)
    f.vline(178, t2, flag)
    f.target(178, flag, r=8)
    f.text(178, t1, "<5", size=13)
    hlayer(225)
    # zero-controlled global phase
    yb = 3 * S + 4
    f.vline(270, t0, t2)
    for y in ts:
        f.ctrl(270, y, on=False, r=4.6)
    hlayer(315)
    phase_box(f, 365, t0, flag, r"$e^{-i\pi/2}$", w=44)
    f.text(270, -22, r"$e^{i\theta}$", size=12)
    wire_labels(f, -8, [(t0, "$t_0$"), (t1, "$t_1$"), (t2, "$t_2$"), (flag, "work")])
    del yb


# ------------------------------------------------------------------ AliasSampling


@drawing("alias_sampling_coherent")
def _(f):
    tgt, sig, alt, keep, flag = 0, S, 2 * S, 3 * S, 4 * S
    x1 = 270
    wires(f, [tgt, sig, alt, keep, flag], 0, x1)
    for y, lab in ((tgt, "2"), (sig, "4"), (alt, "2"), (keep, "4")):
        f.bundle(12, y, lab)
    box(f, 52, tgt, None, "Unif", CY2, w=42, size=13)
    box(f, 52, sig, None, "H", CY2, w=28)
    f.vline(110, tgt, keep)
    f.sqctrl(110, tgt)
    box(f, 110, alt, keep, "QROM", MG2, w=50, size=13)
    # comparator keep <= sigma -> flag
    f.vline(172, sig, flag)
    f.sqctrl(172, sig)
    f.sqctrl(172, keep)
    box(f, 172, flag, None, "≤", YL2, w=28)
    # controlled swap target <-> alt
    f.vline(230, tgt, flag)
    f.ctrl(230, flag, r=4.6)
    _swap_marks(f, 230, [tgt, alt])
    wire_labels(
        f, -8, [(tgt, "target"), (sig, r"$\sigma$"), (alt, "alt"), (keep, "keep"), (flag, "flag")]
    )


# ------------------------------------------------------------------ AliasSamplingTHC


@drawing("alias_sampling_thc_walker")
def _(f):
    mu, nu, edge, addr, alt, sig, flag, sym = [i * 30 for i in range(8)]
    x1 = 470
    wires(f, [mu, nu, edge, addr, alt, sig, flag, sym], 0, x1)
    # contiguous register: OutSquare + two SemiAdders
    f.vline(35, nu, addr)
    f.sqctrl(35, nu)
    box(f, 35, addr, None, "$x^2$", CY2, w=32)
    f.vline(78, nu, addr)
    f.sqctrl(78, nu)
    box(f, 78, addr, None, "+", CY2, w=28)
    f.vline(118, mu, addr)
    f.sqctrl(118, mu)
    box(f, 118, addr, None, "+", CY2, w=28)
    # QROM
    f.vline(168, addr, alt)
    f.sqctrl(168, addr)
    box(f, 168, alt, None, "QROM", MG2, w=50, size=13)
    # sample register + comparator
    box(f, 222, sig, None, "H", CY2, w=28)
    f.vline(262, alt, flag)
    box(f, 262, alt, sig, "≤", YL2, w=30)
    f.target(262, flag, r=8)
    # controlled swaps with the alternates
    f.vline(305, mu, flag)
    f.ctrl(305, flag, r=4.6)
    box(f, 305, mu, edge, "SWAP", GY0, w=46, size=12)
    box(f, 305, alt, None, "SWAP", GY0, w=46, size=12)
    # uncompute comparator
    f.vline(352, alt, flag)
    box(f, 352, alt, sig, r"≤$^\dagger$", YL2, w=34, size=13)
    f.target(352, flag, r=8)
    # symmetrisation
    box(f, 392, sym, None, "H", CY2, w=28)
    f.vline(435, mu, sym)
    f.ctrl(435, sym, r=4.6)
    f.ctrl(435, edge, on=False, r=4.6)
    _swap_marks(f, 435, [mu, nu])
    wire_labels(
        f,
        -8,
        [
            (mu, r"$\mu$"),
            (nu, r"$\nu$"),
            (edge, "edge"),
            (addr, "s"),
            (alt, "alt, keep"),
            (sig, r"$\sigma$"),
            (flag, "flag"),
            (sym, "sym"),
        ],
    )


# ------------------------------------------------------------------ SuperpositionTHC


@drawing("superposition_thc_amplification")
def _(f):
    mu, nu, aux, flags = 0, S, 2 * S, 3 * S + 4
    x1 = 620
    wires(f, [mu, nu, aux, flags], 0, x1)
    f.bundle(12, mu, "n")
    f.bundle(12, nu, "n")
    f.bundle(12, flags, "6")

    def ineq(x, label="cmp", fill=MG2):
        f.vline(x, mu, flags)
        f.sqctrl(x, mu)
        f.sqctrl(x, nu)
        box(f, x, flags, None, label, fill, w=40, size=13)

    def hh(x):
        box(f, x, mu, None, "H", CY2, w=28)
        box(f, x, nu, None, "H", CY2, w=28)

    hh(54)
    box(f, 99, aux, None, "$R_Y$", YL2, w=34)
    ineq(149)
    f.vline(194, aux, flags)
    f.ctrl(194, aux, r=4.6)
    box(f, 194, flags, None, "Z", PU2, w=28)
    ineq(242, r"cmp$^\dagger$")
    box(f, 292, aux, None, r"$R_Y^\dagger$", YL2, w=38)
    hh(337)
    box(f, 382, mu, aux, "$R_0$", PU2, w=36)
    hh(427)
    ineq(475)
    box(f, 524, flags, None, "X", PU2, w=28)
    f.hdots(524 + 26, flags)
    # a global phase commutes with every gate, so it is drawn last
    phase_box(f, 590, mu, flags, "GP")
    wire_labels(f, -8, [(mu, r"$\mu$"), (nu, r"$\nu$"), (aux, "amp"), (flags, "flags")])


# ------------------------------------------------------------------ SelectTHC


@drawing("select_thc_double_factorized")
def _(f):
    mu, nu, succ, edge, swp, sys_ = 0, 28, 56, 84, 112, 146
    x1 = 440
    wires(f, [mu, nu, succ, edge, swp, sys_], 0, x1)
    f.bundle(12, mu, "n")
    f.bundle(12, nu, "n")
    f.bundle(12, sys_, "N")

    def half(x, idx, label, fill, open_edge):
        f.vline(x, idx, sys_)
        f.sqctrl(x, idx)
        box(f, x, sys_, None, f"$U_{label}$", fill, w=34)
        xz = x + 46
        cys = [succ, edge] if open_edge else [succ]
        f.vline(xz, succ, sys_)
        f.ctrl(xz, succ, r=4.6)
        if open_edge:
            f.ctrl(xz, edge, on=False, r=4.6)
        box(f, xz, sys_, None, "Z", PU2, w=28)
        f.vline(xz + 46, idx, sys_)
        f.sqctrl(xz + 46, idx)
        box(f, xz + 46, sys_, None, f"$U_{label}^\\dagger$", fill, w=38)
        del cys

    half(40, mu, r"\mu", CY2, False)
    half(195, nu, r"\nu", MG2, True)
    f.vline(355, mu, edge)
    f.ctrl(355, edge, on=False, r=4.6)
    _swap_marks(f, 355, [mu, nu])
    box(f, 400, swp, None, "X", GY0, w=28)
    wire_labels(
        f,
        -8,
        [
            (mu, r"$\mu$"),
            (nu, r"$\nu$"),
            (succ, "succ"),
            (edge, "edge"),
            (swp, "swap"),
            (sys_, "system"),
        ],
    )


# ------------------------------------------------------------------ QubitizationTHC


@drawing("qubitization_thc_walk")
def _(f):
    idx, gar, spin, sys_ = 0, S, 2 * S, 3 * S + 4
    x1 = 540
    wires(f, [idx, gar, spin, sys_], 0, x1)
    f.bundle(12, idx, "2n")
    f.bundle(12, gar, "")
    f.bundle(12, spin, "2")
    f.bundle(12, sys_, "N")
    box(f, 61, idx, gar, "Sup", CY2, w=40, size=13)
    box(f, 116, idx, gar, "Alias", MG2, w=46, size=13)
    box(f, 166, spin, None, "H", CY2, w=28)
    box(f, 206, gar, None, "Z", PU2, w=28)
    box(f, 256, idx, sys_, "SELECT", YL2, w=56, size=13)
    box(f, 311, spin, None, "H", CY2, w=28)
    box(f, 358, idx, gar, r"Alias$^\dagger$", MG2, w=52, size=13)
    box(f, 416, idx, gar, r"Sup$^\dagger$", CY2, w=44, size=13)
    box(f, 468, idx, gar, "$R_0$", PU2, w=36)
    phase_box(f, 515, idx, sys_, "GP")
    wire_labels(f, -8, [(idx, "index"), (gar, "garbage"), (spin, "spin"), (sys_, "system")])


# ------------------------------------------------------------------ OneBodyBlockEncoding


@drawing("one_body_block_encoding_lcu")
def _(f):
    idx, gar, spin, a, b, work = 0, 30, 60, 98, 128, 162
    x1 = 600
    wires(f, [idx, spin, gar, a, b, work], 0, x1)
    f.bundle(12, idx, "")
    f.bundle(12, gar, "")
    f.bundle(12, a, "K")
    f.bundle(12, b, "K")
    box(f, 45, idx, gar, "Alias", MG2, w=46, size=13)
    box(f, 98, spin, None, "H", CY2, w=28)
    box(f, 140, a, None, "$V$", YL2, w=30)
    box(f, 140, b, None, "$V$", YL2, w=30)
    f.vline(185, idx, work)
    f.sqctrl(185, idx)
    box(f, 185, work, None, "<", GY0, w=28)
    box(f, 222, work, None, "Z", PU2, w=28)
    f.vline(259, idx, work)
    f.sqctrl(259, idx)
    box(f, 259, work, None, "<", GY0, w=28)
    box(f, 300, a, b, "H", CY2, w=28)
    f.vline(345, idx, a)
    f.sqctrl(345, idx)
    f.sqctrl(345, spin)
    box(f, 345, a, b, "QROM", MG2, w=50, size=13)
    box(f, 390, a, b, "H", CY2, w=28)
    box(f, 432, a, None, r"$V^\dagger$", YL2, w=34)
    box(f, 432, b, None, r"$V^\dagger$", YL2, w=34)
    box(f, 487, idx, gar, r"Alias$^\dagger$", MG2, w=52, size=13)
    box(f, 535, spin, None, "H", CY2, w=28)
    phase_box(f, 575, idx, work, "GP")
    wire_labels(
        f,
        -8,
        [
            (idx, "index"),
            (gar, "garbage"),
            (spin, "spin"),
            (a, r"$\alpha$"),
            (b, r"$\beta$"),
            (work, "work"),
        ],
    )


# ------------------------------------------------------------------ TrotterCDF / TrotterCGF


def _trotter(f, labels, rot):
    ys = [0, 40]
    x1 = 470
    wires(f, ys, 0, x1)
    for y, lab in zip(ys, labels):
        f.bundle(12, y, lab[1])

    def rots(x, label, fill):
        for y in ys:
            box(f, x, y, None, label, fill, w=38 if "dagger" not in label else 40)

    rots(45, f"${rot}_1$", CY2)
    box(f, 100, ys[0], ys[1], "ZZ", MG2, w=36, size=14)
    rots(155, f"${rot}_{{01}}$", CY2)
    box(f, 210, ys[0], ys[1], "$R_Z$", YL2, w=36)
    rots(265, f"${rot}_{{10}}$", CY2)
    box(f, 320, ys[0], ys[1], "ZZ", MG2, w=36, size=14)
    f.hdots(360, ys[0])
    f.hdots(360, ys[1])
    rots(405, f"${rot}_1^\\dagger$", CY2)
    phase_box(f, 450, ys[0], ys[-1], "GP")
    wire_labels(f, -8, [(y, lab[0]) for y, lab in zip(ys, labels)])


@drawing("trotter_cdf_second_order")
def _(f):
    _trotter(f, [(r"$\alpha$", "N"), (r"$\beta$", "N")], "U")


@drawing("trotter_cgf_second_order")
def _(f):
    _trotter(f, [("mode 0", "N"), ("mode 1", "N")], "U")


# ------------------------------------------------------------------ TrotterVibronic


@drawing("trotter_vibronic_phase_gradient")
def _(f):
    el, mode, cache, coef, pg = 0, S, 2 * S, 3 * S, 4 * S
    x1 = 560
    wires(f, [el, mode, cache, coef, pg], 0, x1)
    f.bundle(12, el, "n")
    f.bundle(12, mode, "k")
    f.bundle(12, cache, "2k")
    f.bundle(12, coef, "b")
    f.bundle(12, pg, "b")
    # one position fragment (diagonal in the example, so no basis change): load coefficients,
    # add / multiply into the gradient
    f.vline(85, el, coef)
    f.sqctrl(85, el)
    box(f, 85, coef, None, "QROM", MG2, w=50, size=13)
    f.vline(135, coef, pg)
    f.sqctrl(135, coef)
    box(f, 135, pg, None, "+", CY2, w=28)
    f.vline(180, el, coef)
    f.sqctrl(180, el)
    box(f, 180, coef, None, "QROM", MG2, w=50, size=13)
    f.vline(230, mode, pg)
    f.sqctrl(230, mode)
    f.sqctrl(230, coef)
    box(f, 230, pg, None, "×", YL2, w=28, size=24)
    f.hdots(268, coef)
    f.hdots(268, pg)
    # kinetic fragment: momentum space via AQFT
    box(f, 350, coef, None, "X", GY0, w=26)
    box(f, 390, mode, None, "QFT", PU2, w=40, size=13)
    f.vline(438, mode, cache)
    f.sqctrl(438, mode)
    box(f, 438, cache, None, "$x^2$", CY2, w=32)
    f.vline(480, cache, pg)
    f.sqctrl(480, cache)
    f.sqctrl(480, coef)
    box(f, 480, pg, None, "×", YL2, w=28, size=24)
    f.hdots(518, cache)
    f.hdots(518, mode)
    wire_labels(
        f,
        -8,
        [(el, "electronic"), (mode, "mode"), (cache, "cache"), (coef, "coeff"), (pg, "gradient")],
    )
