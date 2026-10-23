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
"""Decomposition thumbnails for the arithmetic templates (Incrementer, SemiAdder, TemporaryAND,
OutMultiplier, SignedOutMultiplier, LeftClassicalComparator, LeftQuantumComparator)."""

from common import *  # noqa

KW = dict(pad=13, size=15)
XF, HF, TF, SF = YL2, CY2, MG2, PU2


def cnot(f, x, c, t):
    f.vline(x, min(c, t), max(c, t))
    f.ctrl(x, c, r=4.6)
    f.target(x, t, r=8)


def fan(f, x, c, ts):
    """Several CNOTs sharing a control, drawn in one column."""
    f.vline(x, min([c] + ts), max([c] + ts))
    f.ctrl(x, c, r=4.6)
    for t in ts:
        f.target(x, t, r=8)


def mcx(f, x, cs, t, bits=None):
    f.vline(x, min(cs + [t]), max(cs + [t]))
    for i, c in enumerate(cs):
        f.ctrl(x, c, on=True if bits is None else bool(bits[i]), r=4.6)
    f.target(x, t, r=8)


def elb(f, x, cs, t, bits=None):
    """TemporaryAND (or its adjoint): the target wire starts (ends) here, drawn by the caller."""
    f.vline(x, min(cs + [t]), max(cs + [t]))
    for i, c in enumerate(cs):
        f.ctrl(x, c, on=True if bits is None else bool(bits[i]), r=4.6)


def xg(f, x, y):
    f.gate(x, y, label="X", w=30, fill=XF, **KW)


def g(f, x, y, label, fill, w=30, y1=None):
    f.gate(x, y, y1, label=label, w=w, fill=fill, **KW)


def reg_ctrl(f, x, ys, t):
    """Register-valued inputs (square controls) feeding a block on ``t``."""
    f.vline(x, min(ys + [t]), max(ys + [t]))
    for y in ys:
        f.sqctrl(x, y)


# ---------------- Incrementer (n = 4) ----------------
Q = [0, 28, 56, 84]


@drawing("incrementer_elbow_ladder")
def _(f):
    A0, A1 = 116, 144
    xs = [30, 72, 114, 156, 198, 240, 282, 322]
    for y in Q:
        f.wire(y, 0, 350)
    f.wire(A0, xs[0], xs[5])
    f.wire(A1, xs[1], xs[3])
    elb(f, xs[0], [Q[3], Q[2]], A0)
    elb(f, xs[1], [A0, Q[1]], A1)
    cnot(f, xs[2], A1, Q[0])
    elb(f, xs[3], [A0, Q[1]], A1)
    cnot(f, xs[4], A0, Q[1])
    elb(f, xs[5], [Q[3], Q[2]], A0)
    cnot(f, xs[6], Q[3], Q[2])
    xg(f, xs[7], Q[3])
    wire_labels(f, -8, [(y, f"$q_{i}$") for i, y in enumerate(Q)] + [(A0, "$w_0$"), (A1, "$w_1$")])


@drawing("incrementer_multi_controlled_x")
def _(f):
    xs = [40, 100, 160, 220]
    for y in Q:
        f.wire(y, 0, 250)
    mcx(f, xs[0], [Q[1], Q[2], Q[3]], Q[0])
    mcx(f, xs[1], [Q[2], Q[3]], Q[1])
    mcx(f, xs[2], [Q[3]], Q[2])
    xg(f, xs[3], Q[3])
    wire_labels(f, -8, [(y, f"$q_{i}$") for i, y in enumerate(Q)])


# ---------------- SemiAdder (m = n = 3) ----------------
@drawing("semi_adder_elbow_ladder")
def _(f):
    X0, Y0, W0, X1, Y1, W1, X2, Y2 = [i * 24 for i in range(8)]
    xs = [26 + 31 * i for i in range(13)]
    for y in (X0, Y0, X1, Y1, X2, Y2):
        f.wire(y, 0, xs[-1] + 26)
    f.wire(W1, xs[0], xs[11])
    f.wire(W0, xs[3], xs[8])
    elb(f, xs[0], [X2, Y2], W1)
    cnot(f, xs[1], W1, X1)
    cnot(f, xs[2], W1, Y1)
    elb(f, xs[3], [X1, Y1], W0)
    cnot(f, xs[4], W1, W0)
    cnot(f, xs[5], W0, Y0)
    cnot(f, xs[6], X0, Y0)
    cnot(f, xs[7], W1, W0)
    elb(f, xs[8], [X1, Y1], W0)
    cnot(f, xs[9], W1, X1)
    cnot(f, xs[10], X1, Y1)
    elb(f, xs[11], [X2, Y2], W1)
    cnot(f, xs[12], X2, Y2)
    wire_labels(
        f,
        -8,
        [
            (X0, "$x_0$"),
            (Y0, "$y_0$"),
            (W0, "$w_0$"),
            (X1, "$x_1$"),
            (Y1, "$y_1$"),
            (W1, "$w_1$"),
            (X2, "$x_2$"),
            (Y2, "$y_2$"),
        ],
    )


# ---------------- TemporaryAND (control_values = (1, 0)) ----------------
C0, C1, T = 0, 30, 60


@drawing("temporary_and_clifford_t")
def _(f):
    xs = [26 + 40 * i for i in range(12)]
    for y in (C0, C1, T):
        f.wire(y, 0, xs[-1] + 24)
    xg(f, xs[0], C1)
    g(f, xs[1], T, "H", HF)
    g(f, xs[2], T, "T", TF)
    cnot(f, xs[3], C1, T)
    g(f, xs[4], T, r"$T^\dagger$", TF)
    cnot(f, xs[5], C0, T)
    g(f, xs[6], T, "T", TF)
    cnot(f, xs[7], C1, T)
    g(f, xs[8], T, r"$T^\dagger$", TF)
    g(f, xs[9], T, "H", HF)
    g(f, xs[10], T, r"$S^\dagger$", SF)
    xg(f, xs[11], C1)
    wire_labels(f, -8, [(C0, "$c_0$"), (C1, "$c_1$"), (T, r"$|0\rangle$")])


@drawing("temporary_and_toffoli")
def _(f):
    xs = [50, 125, 200]
    for y in (C0, C1, T):
        f.wire(y, 0, 240)
    xg(f, xs[0], C1)
    mcx(f, xs[1], [C0, C1], T)
    xg(f, xs[2], C1)
    wire_labels(f, -8, [(C0, "$c_0$"), (C1, "$c_1$"), (T, r"$|0\rangle$")])


def ppr(f, x, paulis, angle, fill=PU2, contiguous=True):
    """A Pauli product rotation: one Pauli letter per wire (a single box over adjacent wires,
    separate boxes joined by a line across skipped wires) and its angle above the circuit."""
    ys = sorted(paulis)
    if contiguous:
        f.gate(x, ys[0], ys[-1], w=30, fill=fill, **KW)
        for y in ys:
            f.text(x, y, f"${paulis[y]}$", size=15)
    else:
        f.vline(x, ys[0], ys[-1])
        for y in ys:
            g(f, x, y, f"${paulis[y]}$", fill)
    f.text(x, C0 - 30, angle, size=12)


@drawing("temporary_and_ppr")
def _(f):
    xs = [30, 80, 128, 176, 224, 274]
    for y in (C0, C1, T):
        f.wire(y, 0, 300)
    xg(f, xs[0], C1)
    ppr(f, xs[1], {C0: "Z", C1: "Z", T: "Y"}, r"$\frac{\pi}{8}$")
    ppr(f, xs[2], {C1: "Z", T: "Y"}, r"$-\frac{\pi}{8}$", fill=MG2)
    ppr(f, xs[3], {C0: "Z", T: "Y"}, r"$-\frac{\pi}{8}$", fill=MG2, contiguous=False)
    ppr(f, xs[4], {T: "Y"}, r"$\frac{\pi}{8}$", fill=CY2)
    xg(f, xs[5], C1)
    wire_labels(f, -8, [(C0, "$c_0$"), (C1, "$c_1$"), (T, r"$|0\rangle$")])


# ---------------- LeftQuantumComparator (n = 2, "<=") ----------------
@drawing("left_quantum_comparator_elbow_ladder")
def _(f):
    X1, Y1, W0, X0, Y0, TG = [i * 26 for i in range(6)]
    xs = [26 + 34 * i for i in range(10)]
    for y in (X1, Y1, X0, Y0):
        f.wire(y, 0, xs[-1] + 26)
    f.wire(W0, xs[1], xs[-1] + 26)
    f.wire(TG, xs[5], xs[-1] + 26)
    cnot(f, xs[0], X1, Y1)
    elb(f, xs[1], [X1, Y1], W0)
    cnot(f, xs[2], X1, Y1)
    cnot(f, xs[3], X0, Y0)
    cnot(f, xs[4], X0, W0)
    elb(f, xs[5], [W0, Y0], TG)
    cnot(f, xs[6], X0, W0)
    cnot(f, xs[7], W0, TG)
    cnot(f, xs[8], X0, Y0)
    xg(f, xs[9], TG)
    wire_labels(
        f,
        -8,
        [(X1, "$x_1$"), (Y1, "$y_1$"), (W0, "work"), (X0, "$x_0$"), (Y0, "$y_0$"), (TG, "target")],
    )


# ---------------- LeftClassicalComparator (n = 3, L = 6, "<") ----------------
@drawing("left_classical_comparator_elbow_ladder")
def _(f):
    X2, W0, X1, W1, X0, TG = [i * 34 for i in range(6)]
    xs = [28 + 40 * i for i in range(8)]
    end = xs[-1] + 28
    for y in (X2, W0, X1, X0):
        f.wire(y, 0, end)
    f.wire(W1, xs[1], end)
    f.wire(TG, xs[5], end)
    xg(f, xs[0], X1)
    xg(f, xs[0], W0)
    elb(f, xs[1], [W0, X1], W1)
    xg(f, xs[2], W0)
    xg(f, xs[2], X1)
    cnot(f, xs[3], W0, W1)
    xg(f, xs[4], X0)
    xg(f, xs[4], W1)
    elb(f, xs[5], [W1, X0], TG)
    xg(f, xs[6], W1)
    xg(f, xs[6], X0)
    cnot(f, xs[7], W1, TG)
    wire_labels(
        f,
        -8,
        [(X2, "$x_2$"), (W0, "$w_0$"), (X1, "$x_1$"), (W1, "$w_1$"), (X0, "$x_0$"), (TG, "target")],
    )


# ---------------- OutMultiplier (n = m = 2, k = 4) ----------------
def bundles(f, x, items):
    for y, lab in items:
        f.bundle(x, y, lab)


@drawing("out_multiplier_qft")
def _(f):
    XR, YR, OR = 0, 34, 72
    for y in (XR, YR, OR):
        f.wire(y, 0, 300)
    bundles(f, 22, [(XR, "$n$"), (YR, "$m$"), (OR, "$k$")])
    g(f, 80, OR, "QFT", HF, w=50)
    reg_ctrl(f, 160, [XR, YR], OR)
    g(f, 160, OR, r"$\Phi_+$", MG2, w=44)
    g(f, 240, OR, r"QFT$^\dagger$", HF, w=60)
    wire_labels(f, -8, [(XR, "$x$"), (YR, "$y$"), (OR, "output")])


@drawing("out_multiplier_adder")
def _(f):
    X0, X1, YR, OR, WR = 0, 28, 58, 92, 130
    for y in (X0, X1, YR, OR, WR):
        f.wire(y, 0, 280)
    bundles(f, 24, [(YR, "$m$"), (OR, "$k$"), (WR, "$w$")])
    for x, c, lab, fill in ((110, X1, "$+y$", CY2), (200, X0, "$+2y$", MG2)):
        f.vline(x, c, OR)
        f.ctrl(x, c, r=4.6)
        f.sqctrl(x, YR)
        g(f, x, OR, lab, fill, w=46, y1=WR)
    wire_labels(f, -8, [(X0, "$x_0$"), (X1, "$x_1$"), (YR, "$y$"), (OR, "output"), (WR, "work")])


@drawing("out_multiplier_caddsub")
def _(f):
    X0, X1, YR, OR, AX = 0, 28, 58, 92, 122
    xs = [70, 132, 196, 252, 304, 356, 408, 462]
    end = 500
    for y in (X0, X1, YR, OR, AX):
        f.wire(y, 0, end)
    bundles(f, 24, [(YR, "$m$"), (OR, "$k$")])
    for x, c, fill in ((xs[0], X1, CY2), (xs[1], X0, MG2)):
        f.vline(x, c, OR)
        f.ctrl(x, c, r=4.6)
        f.sqctrl(x, YR)
        g(f, x, OR, r"$\pm y$", fill, w=44, y1=AX)
    reg_ctrl(f, xs[2], [X0, X1], OR)
    g(f, xs[2], OR, "$+x$", YL2, w=44, y1=AX)
    g(f, xs[3], OR, "X", XF, w=30, y1=AX)
    f.sqctrl(xs[4], YR)
    f.vline(xs[4], YR, OR)
    g(f, xs[4], OR, "$+y$", CY2, w=40, y1=AX)
    g(f, xs[5], OR, "$+1$", PU2, w=40)
    g(f, xs[6], OR, "X", XF, w=30, y1=AX)
    f.sqctrl(xs[7], YR)
    f.vline(xs[7], YR, OR)
    g(f, xs[7], OR, "$+y$", MG2, w=40)
    wire_labels(
        f, -8, [(X0, "$x_0$"), (X1, "$x_1$"), (YR, "$y$"), (OR, "output"), (AX, r"$|0\rangle$")]
    )


@drawing("out_multiplier_cache")
def _(f):
    XR, YR, OR, CR = 0, 32, 64, 100
    for y in (XR, YR, OR, CR):
        f.wire(y, 0, 290)
    bundles(f, 22, [(XR, "$n$"), (YR, "$m$"), (OR, "$k$"), (CR, "$k$")])
    reg_ctrl(f, 80, [XR, YR], CR)
    g(f, 80, CR, r"$\times$", CY2, w=36)
    f.vline(160, OR, CR)
    f.sqctrl(160, CR)
    g(f, 160, OR, "$+$", MG2, w=36)
    reg_ctrl(f, 240, [XR, YR], CR)
    g(f, 240, CR, r"$\times^\dagger$", CY2, w=40)
    wire_labels(f, -8, [(XR, "$x$"), (YR, "$y$"), (OR, "output"), (CR, r"$|0\rangle$")])


# ---------------- SignedOutMultiplier (n = m = 2, k = 4) ----------------
@drawing("signed_out_multiplier_zeroed")
def _(f):
    X0, X1, Y0, Y1, O0, OR, W0, W1 = [i * 28 for i in range(8)]
    x = [46 + 34 * i for i in range(17)]
    end = x[-1] + 22
    for y in (X0, X1, Y0, Y1, O0, OR):
        f.wire(y, 0, end)
    f.wire(W0, 0, end)
    f.wire(W1, 0, end)
    f.bundle(18, OR, "$k-1$")

    def neg_inc(i, c, ys, fill):
        fan(f, x[i], c, ys)
        f.vline(x[i + 1], min(ys + [c]), max(ys + [c]))
        f.ctrl(x[i + 1], c, r=4.6)
        g(f, x[i + 1], ys[0], "$+1$", fill, w=34, y1=ys[-1])

    cnot(f, x[0], X0, W0)
    cnot(f, x[1], Y0, W1)
    neg_inc(2, W0, [X0, X1], CY2)
    neg_inc(4, W1, [Y0, Y1], CY2)
    reg_ctrl(f, x[6], [X0, X1, Y0, Y1], OR)
    g(f, x[6], OR, r"$\times$", MG2, w=30)
    cnot(f, x[7], W0, O0)
    cnot(f, x[8], W1, O0)
    neg_inc(9, O0, [OR], YL2)
    neg_inc(11, W0, [X0, X1], CY2)
    neg_inc(13, W1, [Y0, Y1], CY2)
    cnot(f, x[15], X0, W0)
    cnot(f, x[16], Y0, W1)
    wire_labels(
        f,
        -8,
        [
            (X0, "$x_0$"),
            (X1, "$x_1$"),
            (Y0, "$y_0$"),
            (Y1, "$y_1$"),
            (O0, "$z_0$"),
            (OR, "$z$"),
            (W0, r"$|0\rangle$"),
            (W1, r"$|0\rangle$"),
        ],
    )


@drawing("signed_out_multiplier_not_zeroed")
def _(f):
    XR, YR, OR, TR = 0, 32, 64, 100
    for y in (XR, YR, OR, TR):
        f.wire(y, 0, 230)
    bundles(f, 22, [(XR, "$n$"), (YR, "$m$"), (OR, "$k$"), (TR, "$k$")])
    reg_ctrl(f, 90, [XR, YR], TR)
    g(f, 90, TR, r"$\times$", CY2, w=36)
    f.vline(170, OR, TR)
    f.sqctrl(170, TR)
    g(f, 170, OR, "$+$", MG2, w=36)
    wire_labels(f, -8, [(XR, "$x$"), (YR, "$y$"), (OR, "output"), (TR, r"$|0\rangle$")])
