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
"""Decomposition thumbnails for the templates group (QFT, AQFT, FlipSign, MultiX, IQP,
IQPEmbedding, GQSP, QSVT, QuantumPhaseEstimation, SelectPauliRot, QROM, BasisRotation)."""

from common import *  # noqa

KW = dict(pad=13, size=15)
HF, PF, XF, ZF, UF, RF = CY2, MG2, YL2, PU2, TQ2, LM2


def g(f, x, y, label, fill, w=30, y1=None):
    f.gate(x, y, y1, label=label, w=w, fill=fill, **KW)


def cnot(f, x, c, t, on=True):
    f.vline(x, min(c, t), max(c, t))
    f.ctrl(x, c, on=on, r=4.6)
    f.target(x, t, r=8)


def cg(f, x, cs, t, label, fill, w=30, bits=None, t1=None):
    """Gate on ``t`` (to ``t1``) controlled on the wires ``cs``."""
    ys = cs + [t] + ([t1] if t1 is not None else [])
    f.vline(x, min(ys), max(ys))
    for i, c in enumerate(cs):
        f.ctrl(x, c, on=True if bits is None else bool(bits[i]), r=4.6)
    g(f, x, t, label, fill, w=w, y1=t1)


def swap(f, x, y0, y1, ctrl_y=None, s=6):
    ys = [y0, y1] + ([ctrl_y] if ctrl_y is not None else [])
    f.vline(x, min(ys), max(ys))
    for y in (y0, y1):
        f.line(x - s, y - s, x + s, y + s, z=5, wobble=0)
        f.line(x - s, y + s, x + s, y - s, z=5, wobble=0)
    if ctrl_y is not None:
        f.ctrl(x, ctrl_y, r=4.6)


def wires(f, ys, x1, x0=0):
    for y in ys:
        f.wire(y, x0, x1)


def wlabels(f, ys, name="w"):
    wire_labels(f, -8, [(y, f"${name}_{i}$") for i, y in enumerate(ys)])


# ---------------- QFT / AQFT ----------------
@drawing("qft_cphase_swap")
def _(f):
    W = [0, 34, 68]
    wires(f, W, 300)
    g(f, 26, W[0], "H", HF)
    cg(f, 72, [W[1]], W[0], r"$R_\phi$", PF, w=36)
    cg(f, 118, [W[2]], W[0], r"$R_\phi$", PF, w=36)
    g(f, 164, W[1], "H", HF)
    cg(f, 210, [W[2]], W[1], r"$R_\phi$", PF, w=36)
    g(f, 254, W[2], "H", HF)
    swap(f, 284, W[0], W[2])
    wlabels(f, W)


@drawing("aqft_cphase_swap")
def _(f):
    W = [0, 30, 60, 90]
    wires(f, W, 360)
    x = 26
    for i in range(4):
        g(f, x, W[i], "H", HF)
        if i < 3:
            cg(f, x + 44, [W[i + 1]], W[i], r"$R_\phi$", PF, w=36)
        x += 88
    swap(f, 318, W[0], W[3])
    swap(f, 344, W[1], W[2])
    wlabels(f, W)


# ---------------- FlipSign / MultiX ----------------
@drawing("flip_sign_controlled_z")
def _(f):
    W = [0, 34, 68]
    wires(f, W, 230)
    g(f, 60, W[2], "X", XF)
    cg(f, 115, [W[0], W[1]], W[2], "Z", ZF, bits=[1, 0])
    g(f, 170, W[2], "X", XF)
    wlabels(f, W)


@drawing("multix_conditional_x")
def _(f):
    W = [0, 34, 68]
    wires(f, W, 210)
    g(f, 105, W[0], "X", XF)
    g(f, 105, W[2], "X", XF)
    wire_labels(f, -8, [(W[0], "$b_0=1$"), (W[1], "$b_1=0$"), (W[2], "$b_2=1$")])


# ---------------- IQP / IQPEmbedding ----------------
@drawing("iqp_multirz")
def _(f):
    W = [0, 40]
    wires(f, W, 330)
    g(f, 40, W[0], r"$e^{-i\pi YX/4}$", PU2, w=74, y1=W[1])
    for y in W:
        g(f, 104, y, "H", HF)
        g(f, 296, y, "H", HF)
    g(f, 150, W[0], r"$R_Z$", RF, w=36)
    g(f, 150, W[1], r"$R_Z$", RF, w=36)
    g(f, 210, W[0], r"$R_{ZZ}$", MG2, w=44, y1=W[1])
    wlabels(f, W)


@drawing("iqp_embedding_multirz")
def _(f):
    W = [0, 34, 68]
    wires(f, W, 290)
    for y in W:
        g(f, 26, y, "H", HF)
        g(f, 72, y, r"$R_Z$", RF, w=36)

    def zz(x, a, b):
        f.vline(x, a, b)
        g(f, x, a, r"$R_{ZZ}$", MG2, w=44)
        g(f, x, b, r"$R_{ZZ}$", MG2, w=44)

    zz(130, W[0], W[1])
    zz(190, W[0], W[2])
    zz(250, W[1], W[2])
    wlabels(f, W)


# ---------------- GQSP ----------------
@drawing("gqsp_signal_processing")
def _(f):
    C, T = 0, 40
    wires(f, [C, T], 330)

    def block(x):
        for dx, lab, fill, w in (
            (0, "X", XF, 28),
            (36, "U3", HF, 36),
            (72, "X", XF, 28),
            (106, "Z", ZF, 28),
        ):
            g(f, x + dx, C, lab, fill, w=w)
        return x + 106

    x = block(22)
    cg(f, x + 40, [C], T, "$U$", FILLS[1], w=36, bits=[0])
    block(x + 80)
    wire_labels(f, -8, [(C, "control"), (T, "target")])


# ---------------- QSVT ----------------
@drawing("qsvt_alternating_projectors")
def _(f):
    W = [0, 34]
    wires(f, W, 300)
    xs = [30, 90, 150, 210, 270]
    labs = [
        (r"$\Pi_{\phi_0}$", MG2),
        ("$U$", CY2),
        (r"$\Pi_{\phi_1}$", MG2),
        (r"$U^\dagger$", CY2),
        (r"$\Pi_{\phi_2}$", MG2),
    ]
    for x, (lab, fill) in zip(xs, labs):
        g(f, x, W[0], lab, fill, w=40, y1=W[1])
    wire_labels(f, -8, [(W[0], "anc"), (W[1], "sys")])


# ---------------- QPE ----------------
@drawing("quantum_phase_estimation_qft")
def _(f):
    E = [0, 30, 60]
    T = 104
    wires(f, E + [T], 330)
    for y in E:
        g(f, 26, y, "H", HF)
    for i, (x, p) in enumerate(((84, "4"), (146, "2"), (208, ""))):
        f.vline(x, E[i], T)
        f.ctrl(x, E[i], r=4.6)
        f.gate(x, T, label=f"$U^{p}$" if p else "$U$", w=38, fill=FILLS[i], pad=13, size=15)
    g(f, 284, E[0], r"QFT$^\dagger$", PU2, w=56, y1=E[2])
    wire_labels(f, -8, [(E[0], "$e_0$"), (E[1], "$e_1$"), (E[2], "$e_2$"), (T, "target")])


# ---------------- SelectPauliRot ----------------
@drawing("select_pauli_rot_cnot_ladder")
def _(f):
    C0_, T = 0, 40
    wires(f, [C0_, T], 330)
    g(f, 26, T, r"$S^\dagger$", PU2, w=32)
    g(f, 66, T, "H", HF)
    x = 108
    for _ in range(2):
        g(f, x, T, r"$R_Z$", RF, w=36)
        cnot(f, x + 40, C0_, T)
        x += 78
    g(f, x + 4, T, "H", HF)
    g(f, x + 44, T, "S", PU2)
    wire_labels(f, -8, [(C0_, "$c_0$"), (T, "target")])


# ---------------- QROM ----------------
@drawing("qrom_select_swap")
def _(f):
    S1, S0, T0, T1, A0, A1 = 0, 30, 64, 94, 124, 154
    wires(f, [S1, S0, T0, T1, A0, A1], 500)
    x = 24
    for rep in range(2):
        g(f, x, T0, "H", HF)
        g(f, x, T1, "H", HF)
        swap(f, x + 40, T1, A1, ctrl_y=S1)
        swap(f, x + 70, T0, A0, ctrl_y=S1)
        g(f, x + 122, S0, "QROM", YL2, w=60, y1=A1)
        swap(f, x + 174, T0, A0, ctrl_y=S1)
        swap(f, x + 204, T1, A1, ctrl_y=S1)
        x += 240
    wire_labels(
        f,
        -8,
        [(S1, "$c_1$"), (S0, "$c_0$"), (T0, "$t_0$"), (T1, "$t_1$"), (A0, "$w_0$"), (A1, "$w_1$")],
    )


def mx(f, x, t0, t1, j, ctrl_y=None, on=True, fill=None):
    if ctrl_y is not None:
        f.vline(x, ctrl_y, t1)
        f.ctrl(x, ctrl_y, on=on, r=4.6)
    f.gate(x, t0, t1, label=f"$X^{{b_{j}}}$", w=38, fill=fill or FILLS[j], pad=13, size=15)


@drawing("qrom_unary_iteration")
def _(f):
    C0_, C1_, AUX_, T0, T1 = 0, 30, 62, 98, 128
    wires(f, [C0_, C1_, T0, T1], 470)
    xs = [26, 72, 114, 150, 186, 228, 270, 306, 348, 390, 434]
    f.wire(AUX_, xs[0], xs[-1])
    elbow(f, xs[0], [0, 0])
    mx(f, xs[1], T0, T1, 0, ctrl_y=AUX_)
    g(f, xs[2], C0_, "X", XF)
    cnot(f, xs[3], C0_, AUX_)
    g(f, xs[4], C0_, "X", XF)
    mx(f, xs[5], T0, T1, 1, ctrl_y=AUX_)
    cnot(f, xs[6], C0_, AUX_)
    cnot(f, xs[7], C1_, AUX_)
    mx(f, xs[8], T0, T1, 2, ctrl_y=AUX_)
    cnot(f, xs[9] - 4, C0_, AUX_)
    mx(f, xs[10] - 8, T0, T1, 3, ctrl_y=AUX_)
    elbow(f, xs[10] + 26, [1, 1])
    f.wire(AUX_, xs[-1], xs[10] + 26)
    wire_labels(
        f, -8, [(C0_, "$c_0$"), (C1_, "$c_1$"), (AUX_, "work"), (T0, "$t_0$"), (T1, "$t_1$")]
    )


@drawing("qrom_measurement")
def _(f):
    C0_, C1_, AUX_, T0, T1 = 0, 30, 62, 96, 126
    wires(f, [C0_, C1_, T0, T1], 530)
    x_and, x_end = 70, 420
    f.wire(AUX_, x_and, 486)
    mx(f, 28, T0, T1, 0)
    elbow(f, x_and, [0, 0])
    cnot(f, 106, C0_, AUX_, on=False)
    cnot(f, 140, AUX_, T1)
    cnot(f, 174, C0_, AUX_)
    cnot(f, 208, C1_, AUX_)
    cnot(f, 242, AUX_, T0)
    cnot(f, 276, C0_, AUX_)
    g(f, 320, AUX_, r"$M_{XXX}$", GY0, w=50, y1=T1)
    # CZ conditioned on the X-type measurement
    f.panel(352, C0_ - 16, 388, C1_ + 16, fill=GY0, dash=True, stroke=GY2)
    f.vline(370, C0_, C1_)
    f.ctrl(370, C0_, r=4.6)
    f.ctrl(370, C1_, r=4.6)
    f.meter(x_end, AUX_, s=13)
    # corrections conditioned on the Z measurement
    f.panel(446, AUX_ - 18, 522, T1 + 18, fill=GY0, dash=True, stroke=GY2)
    g(f, 466, AUX_, "X", XF, w=26)
    mx(f, 500, T0, T1, 3, fill=FILLS[3])
    wire_labels(
        f, -8, [(C0_, "$c_0$"), (C1_, "$c_1$"), (AUX_, "work"), (T0, "$t_0$"), (T1, "$t_1$")]
    )


# ---------------- BasisRotation ----------------
@drawing("basis_rotation_real_givens")
def _(f):
    W = [0, 34, 68]
    wires(f, W, 250)
    g(f, 34, W[0], r"$R_\phi$", PF, w=36)
    for x, (a, b) in zip((90, 150, 210), ((0, 1), (1, 2), (0, 1))):
        g(f, x, W[a], "G", CY2, w=34, y1=W[b])
    wlabels(f, W)


@drawing("basis_rotation_complex_givens")
def _(f):
    W = [0, 34, 68]
    wires(f, W, 300)
    for y in W:
        g(f, 30, y, r"$R_\phi$", PF, w=36)
    x = 84
    for a, b in ((0, 1), (1, 2), (0, 1)):
        g(f, x, W[a], "G", CY2, w=34, y1=W[b])
        g(f, x + 44, W[a], r"$R_\phi$", PF, w=36)
        x += 76
    wlabels(f, W)
