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
"""Decomposition-rule diagrams for QubitUnitary and DiagonalQubitUnitary."""

from common import *  # noqa

RXF, RYF, RZF, ROTF, UF, GPF, DF = CY2, MG2, PU2, YL2, CY2, PK2, YL2
GAP = 16
W0, W1, W2 = 0, 34, 68


def _w(label):
    tw = TextPath((0, 0), label, size=15, prop=FONT).get_extents().width
    return max(32, tw + 16)


def chain(f, gates, gphase=r"$e^{i\alpha}$", length=270):
    """Single-qubit gates in order on one wire, then a free-floating global phase."""
    ws = [_w(lab) for lab, _ in gates]
    gw = _w(gphase)
    content = sum(ws) + GAP * len(ws) + gw
    x = (length - content) / 2
    f.wire(W0, 0, length)
    for (lab, fill), w in zip(gates, ws):
        f.gate(x + w / 2, W0, label=lab, w=w, fill=fill, pad=13, size=15)
        x += w + GAP
    f.gate(x + gw / 2, 34, label=gphase, w=gw, fill=GPF, pad=13, size=15)
    wire_labels(f, -8, [(W0, "$w_0$")])


def g(f, x, y, label, fill, y1=None):
    f.gate(x, y, y1, label=label, w=_w(label), fill=fill, pad=13, size=15)


def cnot(f, x, c, t):
    f.vline(x, min(c, t), max(c, t))
    f.ctrl(x, c, r=4.6)
    f.target(x, t, r=8)


def mux(f, x, ctrl_ys, t, label, fill):
    """SelectPauliRot: square controls (one rotation per control state) on the target."""
    f.vline(x, min(ctrl_ys + [t]), max(ctrl_ys + [t]))
    for y in ctrl_ys:
        f.sqctrl(x, y)
    g(f, x, t, label, fill)


@drawing("qubit_unitary_zyz")
def _(f):
    chain(f, [(r"$R_Z(\phi)$", RZF), (r"$R_Y(\theta)$", RYF), (r"$R_Z(\omega)$", RZF)])


@drawing("qubit_unitary_zxz")
def _(f):
    chain(f, [(r"$R_Z(\phi)$", RZF), (r"$R_X(\theta)$", RXF), (r"$R_Z(\omega)$", RZF)])


@drawing("qubit_unitary_xzx")
def _(f):
    chain(f, [(r"$R_X(\phi)$", RXF), (r"$R_Z(\theta)$", RZF), (r"$R_X(\omega)$", RXF)])


@drawing("qubit_unitary_xyx")
def _(f):
    chain(f, [(r"$R_X(\phi)$", RXF), (r"$R_Y(\theta)$", RYF), (r"$R_X(\omega)$", RXF)])


@drawing("qubit_unitary_rot")
def _(f):
    chain(f, [(r"$R(\phi,\theta,\omega)$", ROTF)])


@drawing("qubit_unitary_two_qubit")
def _(f):
    for y in (W0, W1):
        f.wire(y, 0, 370)
    fills = [CY2, MG2, PU2, YL2]
    g(f, 22, W0, "$A_1$", fills[0])
    g(f, 22, W1, "$B_1$", fills[1])
    cnot(f, 64, W1, W0)
    g(f, 106, W0, r"$R_Z$", RZF)
    g(f, 106, W1, r"$R_Y$", RYF)
    cnot(f, 148, W0, W1)
    g(f, 190, W1, r"$R_Y$", RYF)
    cnot(f, 232, W1, W0)
    g(f, 274, W0, "$A_2$", fills[2])
    g(f, 274, W1, "$B_2$", fills[3])
    g(f, 330, W1 + 34, r"$e^{i\alpha}$", GPF)
    wire_labels(f, -8, [(W0, "$w_0$"), (W1, "$w_1$")])


@drawing("qubit_unitary_multi_qubit")
def _(f):
    for y in (W0, W1, W2):
        f.wire(y, 0, 400)
    x, step = 26, 54
    labels = ["$V_1$", "$V_2$", "$V_3$", "$V_4$"]
    rots = [(r"$R_Z$", RZF), (r"$R_Y$", RYF), (r"$R_Z$", RZF)]
    fills = [CY2, MG2, YL2, PU2]
    for j in range(4):
        g(f, x, W1, labels[j], fills[j], y1=W2)
        x += step
        if j < 3:
            mux(f, x, [W1, W2], W0, *rots[j])
            x += step
    wire_labels(f, -8, [(W0, "$w_0$"), (W1, "$w_1$"), (W2, "$w_2$")])


@drawing("diagonal_qubit_unitary_recursive")
def _(f):
    for y in (W0, W1):
        f.wire(y, 0, 280)
    g(f, 95, W0, "$D'$", DF)
    mux(f, 185, [W0], W1, r"$R_Z$", RZF)
    wire_labels(f, -8, [(W0, "$w_0$"), (W1, "$w_1$")])


@drawing("diagonal_qubit_unitary_mux_on_aux")
def _(f):
    for y in (W0, W1, W2):
        f.wire(y, 0, 280)
    mux(f, 140, [W0, W1], W2, r"$R_Z$", RZF)
    wire_labels(f, -8, [(W0, "$w_0$"), (W1, "$w_1$"), (W2, r"$|0\rangle$")])
