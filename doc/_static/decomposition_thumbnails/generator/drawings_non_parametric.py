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
"""Decomposition thumbnails for the fixed gates in ``non_parametric_ops.py``."""

from common import *  # noqa

W0, W1 = 0, 44
GP = GY0
KW = dict(pad=13, size=15)


def seq(f, wires, items, x0=36, gap=46, tail=30):
    """Draw single-wire gates ``(label, wire_y, fill, width)`` (or callables) in order."""
    x = x0
    for it in items:
        if callable(it):
            it(x)
        else:
            label, y, fill, w = it
            f.gate(x, y, label=label, w=w, fill=fill, **KW)
        x += gap
    end = x - gap + tail
    for y in wires:
        f.wire(y, 0, end)
    return end


def one(f, items, gap=50, x0=34, tail=30):
    seq(f, [W0], items, x0=x0, gap=gap, tail=tail)
    wire_labels(f, -8, [(W0, "$w_0$")])


def two_labels(f, extra=()):
    wire_labels(f, -8, [(W0, "$w_0$"), (W1, "$w_1$"), *extra])


def ppr(f, x, label, word, w=52, fill=PU2):
    f.gate(x, W0, W1, lines=[label, f"${word}$"], w=w, fill=fill, **KW)


def cnot(f, x, c, t):
    f.vline(x, min(c, t), max(c, t))
    f.ctrl(x, c, r=4.6)
    f.target(x, t, r=8)


@drawing("hadamard_to_rz_rx")
def _(f):
    one(
        f,
        [
            (r"$R_Z(\frac{\pi}{2})$", W0, CY2, 62),
            (r"$R_X(\frac{\pi}{2})$", W0, MG2, 62),
            (r"$R_Z(\frac{\pi}{2})$", W0, CY2, 62),
            (r"$e^{i\frac{\pi}{2}}$", W0, GP, 50),
        ],
        gap=72,
        x0=40,
    )


@drawing("hadamard_to_rz_ry")
def _(f):
    one(
        f,
        [
            (r"$R_Z(\pi)$", W0, CY2, 56),
            (r"$R_Y(\frac{\pi}{2})$", W0, YL2, 62),
            (r"$e^{i\frac{\pi}{2}}$", W0, GP, 50),
        ],
        gap=72,
        x0=40,
    )


@drawing("hadamard_ppm")
def _(f):
    A = W1
    xs = [30, 82, 148, 214, 266, 320]
    for y in (W0, A):
        f.wire(y, 0, 350)
    f.gate(xs[0], W0, label="Z", w=30, fill=CY2, **KW)
    f.gate(xs[1], W0, A, lines=["PPM", "$YY$"], w=50, fill=GY0, **KW)
    f.gate(xs[2], A, label="PPM $X$", w=66, fill=GY0, **KW)
    f.gate(xs[3], W0, label="Y", w=30, fill=YL2, **KW)
    f.gate(xs[4], W0, label=r"$e^{i\frac{\pi}{2}}$", w=46, fill=GP, **KW)
    f.gate(xs[5], A, label="Z", w=30, fill=CY2, **KW)
    f.text((xs[3] + xs[4]) / 2, W0 - 27, r"if $m_0 = m_1$", size=12)
    f.text(xs[5], A + 27, r"if $m_1$", size=12)
    f.text(xs[1], A + 27, "$m_0$", size=12)
    f.text(xs[2], A + 27, "$m_1$", size=12)
    wire_labels(f, -8, [(W0, "$w_0$"), (A, r"$|0\rangle$")])


@drawing("hadamard_ppr")
def _(f):
    f.wire(W0, 0, 236)
    f.gate(46, W0, label=r"$\mathrm{PPR}_2(Z)$", w=76, fill=MG2, **KW)
    f.gate(134, W0, label=r"$\mathrm{PPR}_4(Y)$", w=76, fill=PU2, **KW)
    f.gate(206, W0, label=r"$e^{i\frac{\pi}{2}}$", w=50, fill=GP, **KW)
    wire_labels(f, -8, [(W0, "$w_0$")])


@drawing("paulix_to_rx")
def _(f):
    one(f, [(r"$R_X(\pi)$", W0, MG2, 58), (r"$e^{i\frac{\pi}{2}}$", W0, GP, 50)], gap=74, x0=42)


@drawing("pauliy_to_ry")
def _(f):
    one(f, [(r"$R_Y(\pi)$", W0, YL2, 58), (r"$e^{i\frac{\pi}{2}}$", W0, GP, 50)], gap=74, x0=42)


@drawing("pauliz_to_phase_shift")
def _(f):
    one(f, [(r"$P(\pi)$", W0, CY2, 52)], x0=90, tail=90)


@drawing("s_to_phase_shift")
def _(f):
    one(f, [(r"$P(\frac{\pi}{2})$", W0, CY2, 56)], x0=90, tail=90)


@drawing("t_to_phase_shift")
def _(f):
    one(f, [(r"$P(\frac{\pi}{4})$", W0, CY2, 56)], x0=90, tail=90)


@drawing("sx_to_rx")
def _(f):
    one(
        f,
        [(r"$R_X(\frac{\pi}{2})$", W0, MG2, 62), (r"$e^{i\frac{\pi}{4}}$", W0, GP, 50)],
        gap=74,
        x0=42,
    )


@drawing("swap_to_cnot")
def _(f):
    for y in (W0, W1):
        f.wire(y, 0, 200)
    cnot(f, 55, W0, W1)
    cnot(f, 100, W1, W0)
    cnot(f, 145, W0, W1)
    two_labels(f)


@drawing("swap_to_ppr")
def _(f):
    for y in (W0, W1):
        f.wire(y, 0, 295)
    ppr(f, 40, r"$\mathrm{PPR}_4$", "YY", fill=YL2)
    ppr(f, 110, r"$\mathrm{PPR}_4$", "XX", fill=MG2)
    ppr(f, 180, r"$\mathrm{PPR}_4$", "ZZ", fill=CY2)
    f.gate(254, W0, W1, label=r"$e^{i\frac{\pi}{4}}$", w=50, fill=GP, **KW)
    two_labels(f)


@drawing("ecr_decomp")
def _(f):
    for y in (W0, W1):
        f.wire(y, 0, 350)
    f.gate(30, W0, label="Z", w=30, fill=CY2, **KW)
    cnot(f, 72, W0, W1)
    f.gate(116, W1, label="SX", w=36, fill=PU2, **KW)
    f.gate(150, W0, label=r"$R_X(\frac{\pi}{2})$", w=62, fill=MG2, **KW)
    f.gate(222, W0, label=r"$R_Y(\frac{\pi}{2})$", w=62, fill=YL2, **KW)
    f.gate(294, W0, label=r"$R_X(\frac{\pi}{2})$", w=62, fill=MG2, **KW)
    two_labels(f)


@drawing("iswap_decomp")
def _(f):
    for y in (W0, W1):
        f.wire(y, 0, 210)
    f.gate(32, W0, label="S", w=30, fill=CY2, **KW)
    f.gate(32, W1, label="S", w=30, fill=CY2, **KW)
    f.gate(76, W0, label="H", w=30, fill=YL2, **KW)
    cnot(f, 114, W0, W1)
    cnot(f, 148, W1, W0)
    f.gate(186, W1, label="H", w=30, fill=YL2, **KW)
    two_labels(f)


@drawing("iswap_to_ppr")
def _(f):
    for y in (W0, W1):
        f.wire(y, 0, 230)
    ppr(f, 70, r"$\mathrm{PPR}_4^\dagger$", "YY", fill=YL2)
    ppr(f, 160, r"$\mathrm{PPR}_4^\dagger$", "XX", fill=MG2)
    two_labels(f)


@drawing("siswap_decomp")
def _(f):
    for y in (W0, W1):
        f.wire(y, 0, 410)
    g = dict(w=34, fill=PU2, **KW)
    r = dict(w=46, fill=CY2, **KW)
    f.gate(28, W0, label="SX", **g)
    f.gate(72, W0, label=r"$R_Z$", **r)
    cnot(f, 110, W0, W1)
    f.gate(146, W0, label="SX", **g)
    f.gate(190, W0, label=r"$R_Z$", **r)
    f.gate(234, W0, label="SX", **g)
    f.gate(278, W0, label=r"$R_Z$", **r)
    f.gate(146, W1, label="SX", **g)
    f.gate(190, W1, label=r"$R_Z$", **r)
    cnot(f, 318, W0, W1)
    f.gate(356, W0, label="SX", **g)
    f.gate(356, W1, label="SX", **g)
    two_labels(f)


@drawing("siswap_to_ppr")
def _(f):
    for y in (W0, W1):
        f.wire(y, 0, 230)
    ppr(f, 70, r"$\mathrm{PPR}_8^\dagger$", "YY", fill=YL2)
    ppr(f, 160, r"$\mathrm{PPR}_8^\dagger$", "XX", fill=MG2)
    two_labels(f)


def ppr_and_phase(f, label, fill, phase):
    """A single-wire Pauli product rotation followed by a global phase."""
    f.wire(W0, 0, 190)
    f.gate(58, W0, label=label, w=76, fill=fill, **KW)
    f.gate(150, W0, label=phase, w=50, fill=GP, **KW)
    wire_labels(f, -8, [(W0, "$w_0$")])


@drawing("paulix_to_ppr")
def _(f):
    ppr_and_phase(f, r"$\mathrm{PPR}_2(X)$", MG2, r"$e^{i\frac{\pi}{2}}$")


@drawing("pauliy_to_ppr")
def _(f):
    ppr_and_phase(f, r"$\mathrm{PPR}_2(Y)$", MG2, r"$e^{i\frac{\pi}{2}}$")


@drawing("pauliz_to_ppr")
def _(f):
    ppr_and_phase(f, r"$\mathrm{PPR}_2(Z)$", MG2, r"$e^{i\frac{\pi}{2}}$")


@drawing("s_to_ppr")
def _(f):
    ppr_and_phase(f, r"$\mathrm{PPR}_4(Z)$", PU2, r"$e^{i\frac{\pi}{4}}$")


@drawing("t_to_ppr")
def _(f):
    ppr_and_phase(f, r"$\mathrm{PPR}_8(Z)$", YL2, r"$e^{i\frac{\pi}{8}}$")


W2 = 88


def three_labels(f):
    wire_labels(f, -8, [(W0, "$w_0$"), (W1, "$w_1$"), (W2, "$w_2$")])


@drawing("ppr2_to_paulis")
def _(f):
    for y in (W0, W1, W2):
        f.wire(y, 0, 170)
    f.gate(40, W0, W2, label=r"$e^{-i\frac{\pi}{2}}$", w=50, fill=GP, **KW)
    f.gate(115, W0, label="X", w=30, fill=MG2, **KW)
    f.gate(115, W1, label="Y", w=30, fill=YL2, **KW)
    f.gate(115, W2, label="Z", w=30, fill=CY2, **KW)
    three_labels(f)


def ppr_via_z_gate(f, central, phase):
    """``PPR_k(1, "XYZ")``: change basis to Z on the first wire, apply ``central``, change back."""
    for y in (W0, W1, W2):
        f.wire(y, 0, 420)
    s = dict(w=30, fill=CY2, **KW)
    h = dict(w=30, fill=YL2, **KW)
    f.gate(36, W0, W2, label=phase, w=50, fill=GP, **KW)
    f.gate(86, W0, label="H", **h)
    f.gate(86, W1, label=r"$S^\dagger$", **s)
    f.gate(124, W1, label="H", **h)
    cnot(f, 162, W2, W1)
    cnot(f, 196, W1, W0)
    f.gate(234, W0, label=central, w=30, fill=PU2, **KW)
    cnot(f, 272, W1, W0)
    cnot(f, 306, W2, W1)
    f.gate(344, W0, label="H", **h)
    f.gate(344, W1, label="H", **h)
    f.gate(382, W1, label="S", **s)
    three_labels(f)


@drawing("ppr4_to_cliffords")
def _(f):
    ppr_via_z_gate(f, "S", r"$e^{-i\frac{\pi}{4}}$")


@drawing("ppr8_to_clifford_t")
def _(f):
    ppr_via_z_gate(f, "T", r"$e^{-i\frac{\pi}{8}}$")
