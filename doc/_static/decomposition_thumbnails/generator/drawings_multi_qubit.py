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
"""Decomposition-rule diagrams for the multi-qubit parametric gates
(MultiRZ, PauliRot, PCPhase, IsingXX, IsingYY, IsingZZ, IsingXY, SingleExcitation)."""

from common import *  # noqa

RXF, RYF, RZF, PSF, CLF, PPRF, GPF, MRZF = CY2, MG2, PU2, TQ2, GY0, YL2, PK2, PU2
KW = dict(pad=13, size=15)
DY = 38  # wire spacing
W = [i * DY for i in range(4)]


L = 330


def wires(f, n, length=L, labels=True):
    for y in W[:n]:
        f.wire(y, 0, length)
    if labels:
        wire_labels(f, -8, [(W[i], f"${i}$") for i in range(n)])


def box(f, x, y, label, fill, w=34, y1=None):
    f.gate(x, y, y1, label=label, w=w, fill=fill, **KW)


def cnot(f, x, c, t):
    f.vline(x, min(c, t), max(c, t))
    f.ctrl(x, c, r=4.6)
    f.target(x, t, r=8)


def cy(f, x, c, t):
    f.vline(x, min(c, t), max(c, t))
    f.ctrl(x, c, r=4.6)
    box(f, x, t, "Y", CLF, w=30)


def cps(f, x, ctrls, t, label, w=58):
    """Controlled PhaseShift: ``ctrls`` = [(y, on), ...]."""
    f.vline(x, min([y for y, _ in ctrls] + [t]), t)
    for y, on in ctrls:
        f.ctrl(x, y, on=on, r=4.6)
    box(f, x, t, label, PSF, w=w)


# ---- MultiRZ (n=3) ----
@drawing("multirz_cnot_ladder")
def _(f):
    wires(f, 3)
    xs = [55, 107, 165, 223, 275]
    cnot(f, xs[0], W[2], W[1])
    cnot(f, xs[1], W[1], W[0])
    box(f, xs[2], W[0], r"$R_Z(\theta)$", RZF, w=58)
    cnot(f, xs[3], W[1], W[0])
    cnot(f, xs[4], W[2], W[1])


@drawing("multirz_to_ppr")
def _(f):
    wires(f, 3)
    box(f, L / 2, W[0], r"$e^{-i\theta ZZZ/2}$", PPRF, w=100, y1=W[2])


# ---- PauliRot (P = XYZ) ----
@drawing("paulirot_basis_change")
def _(f):
    wires(f, 3)
    box(f, 52, W[0], "H", CLF, w=30)
    box(f, 52, W[1], r"$\mathrm{PPR}_4(X)$", PPRF, w=76)
    box(f, 165, W[0], r"$\mathrm{MultiRZ}(\theta)$", MRZF, w=100, y1=W[2])
    box(f, 278, W[0], "H", CLF, w=30)
    box(f, 278, W[1], r"$\mathrm{PPR}_4^\dagger(X)$", PPRF, w=76)


# ---- PCPhase (n=4, dim=3) ----
@drawing("pcphase_ctrl_phase_shifts")
def _(f):
    wires(f, 4)
    box(f, 32, W[1], "X", CLF, w=30)
    cps(f, 92, [(W[0], False)], W[1], r"$R_\phi(2\phi)$", w=64)
    box(f, 152, W[1], "X", CLF, w=30)
    cps(f, 222, [(W[0], False), (W[1], False), (W[2], True)], W[3], r"$R_\phi(-2\phi)$", w=72)
    box(f, 298, W[0], r"$e^{-i\phi}$", GPF, w=40, y1=W[3])


def conj(f, gate, mid):
    """``gate`` - ``mid`` - ``gate`` on two wires, as for the Ising CNOT/CY rules."""
    wires(f, 2)
    gate(f, 85, W[0], W[1])
    mid()
    gate(f, 245, W[0], W[1])


def ppr2(f, label):
    wires(f, 2)
    box(f, 165, W[0], label, PPRF, w=90, y1=W[1])


# ---- IsingXX ----
@drawing("isingxx_cnot_rx")
def _(f):
    conj(f, cnot, lambda: box(f, 165, W[0], r"$R_X(\phi)$", RXF, w=58))


@drawing("isingxx_ppr")
def _(f):
    ppr2(f, r"$e^{-i\phi XX/2}$")


# ---- IsingYY ----
@drawing("isingyy_cy_ry")
def _(f):
    conj(f, cy, lambda: box(f, 165, W[0], r"$R_Y(\phi)$", RYF, w=58))


@drawing("isingyy_ppr")
def _(f):
    ppr2(f, r"$e^{-i\phi YY/2}$")


# ---- IsingZZ ----
@drawing("isingzz_cnot_rz")
def _(f):
    conj(f, cnot, lambda: box(f, 165, W[1], r"$R_Z(\phi)$", RZF, w=58))


@drawing("isingzz_ppr")
def _(f):
    ppr2(f, r"$e^{-i\phi ZZ/2}$")


def h_conj(f, gate, rot0, rot1):
    """H, ``gate``, two parallel rotations, ``gate``, H (IsingXY, SingleExcitation)."""
    wires(f, 2)
    box(f, 35, W[0], "H", CLF, w=30)
    gate(f, 87, W[0], W[1])
    box(f, 165, W[0], rot0[0], rot0[1], w=78)
    box(f, 165, W[1], rot1[0], rot1[1], w=78)
    gate(f, 243, W[0], W[1])
    box(f, 295, W[0], "H", CLF, w=30)


# ---- IsingXY ----
@drawing("isingxy_h_cy")
def _(f):
    h_conj(f, cy, (r"$R_Y(\phi/2)$", RYF), (r"$R_X(-\phi/2)$", RXF))


# ---- SingleExcitation ----
@drawing("singleexcitation_h_cnot_ry")
def _(f):
    h_conj(f, cnot, (r"$R_Y(-\phi/2)$", RYF), (r"$R_Y(-\phi/2)$", RYF))


@drawing("singleexcitation_ppr")
def _(f):
    wires(f, 2)
    box(f, 107, W[0], r"$e^{-i\phi YX/4}$", PPRF, w=90, y1=W[1])
    box(f, 223, W[0], r"$e^{i\phi XY/4}$", PPRF, w=90, y1=W[1])


@drawing("singleexcitation_ppr_rz")
def _(f):
    wires(f, 2)
    box(f, 55, W[0], r"$\mathrm{PPR}_4(XX)$", PPRF, w=84, y1=W[1])
    box(f, 165, W[0], r"$R_Z(\phi/2)$", RZF, w=78)
    box(f, 165, W[1], r"$R_Z(-\phi/2)$", RZF, w=78)
    box(f, 275, W[0], r"$\mathrm{PPR}_4^\dagger(XX)$", PPRF, w=84, y1=W[1])
