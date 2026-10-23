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
"""Diagrams for the decomposition rules of ``Select``."""

from common import *  # noqa


@drawing("select_multi_control")
def _(f):
    t = 74
    for y in (C0, C1, t):
        f.wire(y, 0, 250)
    for j, x in enumerate((40, 95, 150, 205)):
        cu(f, x, j, [C0, C1], [(j >> 1) & 1, j & 1], t=t)
    wire_labels(f, -8, [(C0, "$c_0$"), (C1, "$c_1$"), (t, "target")])


@drawing("select_unary")
def _(f):
    xs = dict(and_=30, u0=72, x1=110, u1=148, n1=186, n2=218, u2=256, n3=294, u3=332, andd=372)
    for y in (C0, C1, TGT):
        f.wire(y, 0, 405)
    f.wire(AUX, xs["and_"], xs["andd"])
    elbow(f, xs["and_"], [0, 0])
    cu(f, xs["u0"], 0, [AUX], [1])
    cx(f, xs["x1"], [C0], [0])
    cu(f, xs["u1"], 1, [AUX], [1])
    cx(f, xs["n1"], [C0], [1])
    cx(f, xs["n2"], [C1], [1])
    cu(f, xs["u2"], 2, [AUX], [1])
    cx(f, xs["n3"], [C0], [1])
    cu(f, xs["u3"], 3, [AUX], [1])
    elbow(f, xs["andd"], [1, 1])
    wire_labels(f, -8, [(C0, "$c_0$"), (C1, "$c_1$"), (AUX, "work"), (TGT, "target")])


@drawing("select_multi_control_work_wire")
def _(f):
    for y in (C0, C1, AUX, TGT):
        f.wire(y, 0, 380)
    x = 26
    for j in range(4):
        bits = [(j >> 1) & 1, j & 1]
        cx(f, x, [C0, C1], bits)
        cu(f, x + 29, j, [AUX], [1], w=30)
        cx(f, x + 58, [C0, C1], bits)
        x += 88
    wire_labels(f, -8, [(C0, "$c_0$"), (C1, "$c_1$"), (AUX, r"$|0\rangle$"), (TGT, "target")])
