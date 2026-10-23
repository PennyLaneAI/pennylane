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
"""Decomposition-rule diagrams for the single-qubit parametric gates
(RX, RY, RZ, PhaseShift, Rot, U1, U2, U3)."""

from common import *  # noqa

RXF, RYF, RZF, ROTF, PSF, CLF, PPRF, GPF = CY2, MG2, PU2, YL2, TQ2, GY0, YL2, PK2
MINW = 270  # common wire length so all thumbnails share one text scale
GAP = 16


def _w(f, label, w):
    if w:
        return w
    tw = TextPath((0, 0), label, size=15, prop=FONT).get_extents().width
    return max(32, tw + 16)


def chain(f, gates, gphase=None):
    """Draw ``gates`` = [(label, fill), ...] in order on one wire; ``gphase`` is an optional
    global-phase label drawn as a free-floating box after the gates (it acts on no wire)."""
    ws = [_w(f, lab, None) for lab, _ in gates]
    gw = _w(f, gphase, None) if gphase else 0
    content = sum(ws) + GAP * (len(ws) - 1) + (GAP + gw if gphase else 0)
    length = max(MINW, content + 60)
    x = (length - content) / 2
    f.wire(0, 0, length)
    for (lab, fill), w in zip(gates, ws):
        f.gate(x + w / 2, 0, label=lab, w=w, fill=fill, pad=13, size=15)
        x += w + GAP
    if gphase:
        f.gate(x + gw / 2, 34, label=gphase, w=gw, fill=GPF, pad=13, size=15)


S, SD, H = ("S", CLF), (r"$S^\dagger$", CLF), ("H", CLF)
PS_PHI = (r"$R_\phi(\phi)$", PSF)


# ---- RX ----
@drawing("rx_to_rot")
def _(f):
    chain(f, [(r"$R(\pi/2,\ \phi,\ 7\pi/2)$", ROTF)])


@drawing("rx_to_rz_ry")
def _(f):
    chain(f, [(r"$R_Z(\pi/2)$", RZF), (r"$R_Y(\phi)$", RYF), (r"$R_Z(-\pi/2)$", RZF)])


@drawing("rx_to_ppr")
def _(f):
    chain(f, [(r"$e^{-i\phi X/2}$", PPRF)])


@drawing("rx_to_ry_cliff")
def _(f):
    chain(f, [S, (r"$R_Y(\phi)$", RYF), SD])


@drawing("rx_to_rz_cliff")
def _(f):
    chain(f, [H, (r"$R_Z(\phi)$", RZF), H])


# ---- RY ----
@drawing("ry_to_rot")
def _(f):
    chain(f, [(r"$R(0,\ \phi,\ 0)$", ROTF)])


@drawing("ry_to_rz_rx")
def _(f):
    chain(f, [(r"$R_Z(-\pi/2)$", RZF), (r"$R_X(\phi)$", RXF), (r"$R_Z(\pi/2)$", RZF)])


@drawing("ry_to_ppr")
def _(f):
    chain(f, [(r"$e^{-i\phi Y/2}$", PPRF)])


@drawing("ry_to_rx_cliff")
def _(f):
    chain(f, [SD, (r"$R_X(\phi)$", RXF), S])


@drawing("ry_to_rz_cliff")
def _(f):
    chain(f, [SD, H, (r"$R_Z(\phi)$", RZF), H, S])


# ---- RZ ----
@drawing("rz_to_phaseshift")
def _(f):
    chain(f, [PS_PHI], gphase=r"$e^{-i\phi/2}$")


@drawing("rz_to_rot")
def _(f):
    chain(f, [(r"$R(0,\ 0,\ \phi)$", ROTF)])


@drawing("rz_to_ry_rx")
def _(f):
    chain(f, [(r"$R_Y(\pi/2)$", RYF), (r"$R_X(\phi)$", RXF), (r"$R_Y(-\pi/2)$", RYF)])


@drawing("rz_to_ppr")
def _(f):
    chain(f, [(r"$e^{-i\phi Z/2}$", PPRF)])


@drawing("rz_to_rx_cliff")
def _(f):
    chain(f, [H, (r"$R_X(\phi)$", RXF), H])


@drawing("rz_to_ry_cliff")
def _(f):
    chain(f, [H, S, (r"$R_Y(\phi)$", RYF), SD, H])


# ---- PhaseShift, Rot, U1, U2, U3 ----
@drawing("phaseshift_to_rz_gp")
def _(f):
    chain(f, [(r"$R_Z(\phi)$", RZF)], gphase=r"$e^{i\phi/2}$")


@drawing("rot_to_rz_ry_rz")
def _(f):
    chain(f, [(r"$R_Z(\phi)$", RZF), (r"$R_Y(\theta)$", RYF), (r"$R_Z(\omega)$", RZF)])


@drawing("u1_to_phaseshift")
def _(f):
    chain(f, [PS_PHI])


@drawing("u2_to_rot_phaseshift")
def _(f):
    chain(f, [(r"$R(\delta,\ \pi/2,\ {-\delta})$", ROTF), (r"$R_\phi(\delta)$", PSF), PS_PHI])


@drawing("u3_to_rot_phaseshift")
def _(f):
    chain(f, [(r"$R(\delta,\ \theta,\ {-\delta})$", ROTF), (r"$R_\phi(\delta)$", PSF), PS_PHI])
