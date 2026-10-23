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
"""Shared helpers for the decomposition-rule diagrams: the drawing registry, common circuit
elements, and rendering to SVG in a 21:9 box."""

import os

import sketch
from sketch import *  # noqa

sketch.ASPECT = 21 / 9
sketch.DISPLAY_W = 300

D = {}
FILLS = [CY2, MG2, YL2, PU2]
C0, C1, AUX, TGT = 0, 30, 62, 104


def drawing(name):
    def deco(fn):
        D[name] = fn
        return fn

    return deco


def cu(f, x, j, ctrl_ys, bits, w=34, t=TGT):
    """A controlled U_j on the target wire."""
    f.vline(x, min(ctrl_ys), t)
    for y, b in zip(ctrl_ys, bits):
        f.ctrl(x, y, on=bool(b), r=4.6)
    f.gate(x, t, label=f"$U_{j}$", w=w, fill=FILLS[j], pad=13, size=15)


def cx(f, x, ctrl_ys, bits, t=AUX):
    f.vline(x, min(ctrl_ys + [t]), max(ctrl_ys + [t]))
    for y, b in zip(ctrl_ys, bits):
        f.ctrl(x, y, on=bool(b), r=4.6)
    f.target(x, t, r=8)


def elbow(f, x, bits):
    """Gidney's TemporaryAND: the work wire starts (or, for its adjoint, ends) at the elbow."""
    f.vline(x, C0, AUX)
    for y, b in zip((C0, C1), bits):
        f.ctrl(x, y, on=bool(b), r=4.6)


def wire_labels(f, x, labels):
    for y, s in labels:
        f.text(x, y, s, size=13, anchor="r")


def render(names, out):
    """Write ``<name>.svg`` into ``out`` for each registered drawing in ``names``."""
    os.makedirs(out, exist_ok=True)
    for name in names:
        f = Fig(seed=sum(map(ord, name)))
        D[name](f)
        with open(os.path.join(out, name + ".svg"), "w") as fh:
            fh.write(f.svg())
