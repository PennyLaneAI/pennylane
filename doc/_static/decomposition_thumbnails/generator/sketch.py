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
"""Tiny hand-drawn-style SVG circuit drawing kit following the PennyLane illustration guide:
black monoline strokes with round caps/joins, organic (wobbly) lines, solid palette fills,
Quicksand text, transparent background."""

import math
import random

from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

# the font file itself is registered by generate.py, before this module is imported
FONT = FontProperties(family="Quicksand")

INK = "#222222"
# primary
CY1, CY2, CY3 = "#C8ECFD", "#A3E0FF", "#65C3FF"
MG1, MG2, MG3, MG4 = "#FFD4F7", "#FFB5F1", "#FF87EB", "#C756B2"
YL0, YL1, YL2, YL3 = "#FFF8E5", "#FFEEBF", "#FFE096", "#FFD36A"
GY0, GY1, GY2 = "#F0EEEF", "#BCBABA", "#ABA9A9"
TEAL0 = "#E3F5F3"
# secondary
PU1, PU2, PU3 = "#EEE0FE", "#E2C9FE", "#D7A2F6"
GN1, GN2 = "#E8FEEF", "#C8FDD9"
LM1, LM2 = "#EDF3CC", "#E1EBA8"
TQ1, TQ2 = "#D8F8F6", "#B5F2ED"
PK1, PK2 = "#FEE1ED", "#FDC9DE"
PE1, PE2 = "#FEE0CF", "#FDD3BB"

DISPLAY_W = 228  # px the drawing is shown at (thumbnail box minus padding)
ASPECT = 16 / 9


def _fmt(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _smooth(pts, closed=False):
    """Catmull-Rom spline through pts as an SVG path string."""
    if closed:
        pts = pts + pts[:3]
    d = [f"M{_fmt(pts[0][0])},{_fmt(pts[0][1])}"]
    n = len(pts)
    for i in range(n - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[i + 2] if i + 2 < n else pts[i + 1]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d.append(
            f"C{_fmt(c1[0])},{_fmt(c1[1])} {_fmt(c2[0])},{_fmt(c2[1])} {_fmt(p2[0])},{_fmt(p2[1])}"
        )
    if closed:
        # drop the duplicated lead-in and start at pts[1]
        d = [f"M{_fmt(pts[1][0])},{_fmt(pts[1][1])}"] + d[2:-1] + ["Z"]
    return "".join(d)


class Noise:
    """Smooth 1D noise built from a few random sinusoids."""

    def __init__(self, rng, period=None):
        self.terms = []
        for wl, a in ((90, 0.55), (45, 0.3), (23, 0.15)):
            if period:
                k = max(1, round(period / wl))
                wl = period / k
            self.terms.append((2 * math.pi / wl, a, rng.uniform(0, 2 * math.pi)))

    def __call__(self, s):
        return sum(a * math.sin(w * s + p) for w, a, p in self.terms)


class Fig:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.items = []  # (z, render(sw, amp))
        self.bbox = [math.inf, math.inf, -math.inf, -math.inf]

    def _grow(self, x0, y0, x1, y1, pad=0):
        b = self.bbox
        b[0], b[1] = min(b[0], x0 - pad), min(b[1], y0 - pad)
        b[2], b[3] = max(b[2], x1 + pad), max(b[3], y1 + pad)

    def _add(self, z, fn):
        self.items.append((z, len(self.items), fn))

    # ---------- primitives ----------
    def polyline(self, pts, z=1, color=INK, width=1.0, dash=None, wobble=1.0):
        """Wobbly open path through the given corner points."""
        noise = Noise(self.rng)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self._grow(min(xs), min(ys), max(xs), max(ys), 2)

        def render(sw, amp):
            out, s = [], 0.0
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                L = math.hypot(x1 - x0, y1 - y0)
                if L == 0:
                    continue
                nx, ny = -(y1 - y0) / L, (x1 - x0) / L
                n = max(2, int(L / 14) + 1)
                for i in range(n + 1):
                    if out and i == 0:
                        continue
                    t = i / n
                    o = noise(s + t * L) * amp * wobble
                    out.append((x0 + t * (x1 - x0) + nx * o, y0 + t * (y1 - y0) + ny * o))
                s += L
            da = f' stroke-dasharray="{_fmt(sw * 2.6)} {_fmt(sw * 2.6)}"' if dash else ""
            return (
                f'<path d="{_smooth(out)}" fill="none" stroke="{color}" '
                f'stroke-width="{_fmt(sw * width)}"{da}/>'
            )

        self._add(z, render)

    def line(self, x0, y0, x1, y1, **kw):
        self.polyline([(x0, y0), (x1, y1)], **kw)

    def rrect(self, x0, y0, x1, y1, fill="none", stroke=INK, r=4, z=3, width=1.0, dash=False):
        """Wobbly rounded rectangle."""
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
        self._grow(x0, y0, x1, y1, 2)
        perim = 2 * (x1 - x0 + y1 - y0)
        noise = Noise(self.rng, period=perim)

        def render(sw, amp):
            pts = []
            # walk the outline: (corner centre, start angle)
            corners = [
                ((x1 - r, y0 + r), -90),
                ((x1 - r, y1 - r), 0),
                ((x0 + r, y1 - r), 90),
                ((x0 + r, y0 + r), 180),
            ]
            raw = []
            for (cx, cy), a0 in corners:
                for k in range(4):
                    a = math.radians(a0 + 90 * k / 3)
                    raw.append(
                        (cx + r * math.cos(a), cy + r * math.sin(a), math.cos(a), math.sin(a))
                    )
            # densify straight edges
            dense = []
            for i, p in enumerate(raw):
                q = raw[(i + 1) % len(raw)]
                dense.append(p)
                L = math.hypot(q[0] - p[0], q[1] - p[1])
                n = int(L / 14)
                for j in range(1, n):
                    t = j / n
                    dense.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
            # outward normals for edge points: use direction from rect centre axis
            s = 0.0
            prev = dense[0]
            cxm, cym = (x0 + x1) / 2, (y0 + y1) / 2
            for p in dense:
                s += math.hypot(p[0] - prev[0], p[1] - prev[1])
                prev = p
                # normal: nearest side
                dx, dy = p[0] - cxm, p[1] - cym
                hw, hh = (x1 - x0) / 2 - r, (y1 - y0) / 2 - r
                ex, ey = max(abs(dx) - hw, 0) * (1 if dx > 0 else -1), max(abs(dy) - hh, 0) * (
                    1 if dy > 0 else -1
                )
                m = math.hypot(ex, ey) or 1
                o = noise(s) * amp
                pts.append((p[0] + ex / m * o, p[1] + ey / m * o))
            da = f' stroke-dasharray="{_fmt(sw * 2.6)} {_fmt(sw * 2.6)}"' if dash else ""
            return (
                f'<path d="{_smooth(pts, closed=True)}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{_fmt(sw * width)}"{da}/>'
            )

        self._add(z, render)

    def circle(self, cx, cy, r, fill="none", stroke=INK, z=4, width=1.0):
        self._grow(cx - r, cy - r, cx + r, cy + r, 2)
        noise = Noise(self.rng, period=2 * math.pi * 30)

        def render(sw, amp):
            pts = []
            n = max(10, int(2 * math.pi * r / 5))
            for i in range(n):
                a = 2 * math.pi * i / n
                rr = r + noise(a * 30) * amp * 0.5 * min(1, r / 8)
                pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
            return (
                f'<path d="{_smooth(pts, closed=True)}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{_fmt(sw * width)}"/>'
            )

        self._add(z, render)

    def text(self, x, y, s, size=16, anchor="c", color=INK, z=6, rotate=0):
        """Text as filled outlines; anchor in l/c/r, vertically centred on y."""
        tp = TextPath((0, 0), s, size=size, prop=FONT)
        ext = tp.get_extents()
        tr = Affine2D().translate(-ext.x0 - ext.width / 2, -ext.y0 - ext.height / 2).scale(1, -1)
        w, h = ext.width, ext.height
        if rotate:
            tr = tr.rotate_deg(-rotate)
            if rotate % 180:
                w, h = h, w
        dx = {"l": w / 2, "c": 0, "r": -w / 2}[anchor]
        tr = tr.translate(x + dx, y)
        path = tr.transform_path(tp)
        self._grow(x + dx - w / 2, y - h / 2, x + dx + w / 2, y + h / 2, 1)
        d = []
        for verts, code in path.iter_segments():
            v = [_fmt(c) for c in verts]
            if code == Path.MOVETO:
                d.append(f"M{v[0]},{v[1]}")
            elif code == Path.LINETO:
                d.append(f"L{v[0]},{v[1]}")
            elif code == Path.CURVE3:
                d.append(f"Q{v[0]},{v[1]} {v[2]},{v[3]}")
            elif code == Path.CURVE4:
                d.append(f"C{v[0]},{v[1]} {v[2]},{v[3]} {v[4]},{v[5]}")
            elif code == Path.CLOSEPOLY:
                d.append("Z")
        self._add(z, lambda sw, amp: f'<path d="{"".join(d)}" fill="{color}" stroke="none"/>')
        return w

    def raw_bezier(self, p0, c1, c2, p1, z=1, color=INK):
        """A smooth cubic (already organic), e.g. for crossing wires."""
        self._grow(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]), 2)
        jit = [self.rng.uniform(-1, 1) for _ in range(4)]

        def render(sw, amp):
            q1 = (c1[0], c1[1] + jit[0] * amp)
            q2 = (c2[0], c2[1] + jit[1] * amp)
            return (
                f'<path d="M{_fmt(p0[0])},{_fmt(p0[1])} C{_fmt(q1[0])},{_fmt(q1[1])} '
                f'{_fmt(q2[0])},{_fmt(q2[1])} {_fmt(p1[0])},{_fmt(p1[1])}" fill="none" '
                f'stroke="{color}" stroke-width="{_fmt(sw)}"/>'
            )

        self._add(z, render)

    # ---------- circuit vocabulary ----------
    def wire(self, y, x0, x1, **kw):
        self.line(x0, y, x1, y, z=1, wobble=0, **kw)

    def dwire(self, y, x0, x1, gap=2.2):
        self.line(x0, y - gap, x1, y - gap, z=1, width=0.7, wobble=0)
        self.line(x0, y + gap, x1, y + gap, z=1, width=0.7, wobble=0)

    def bundle(self, x, y, label=None, size=11, centered=False):
        self.line(x - 5, y + 7, x + 5, y - 7, z=2)
        if label:
            if centered:
                self.text(x, y - 16, label, size=size)
            else:
                self.text(x + 5, y - 13, label, size=size, anchor="l")

    def gate(self, x, y0, y1=None, label=None, w=None, fill=CY2, size=16, pad=15, lines=None):
        y1 = y0 if y1 is None else y1
        if w is None:
            if lines:
                tw = max(
                    TextPath((0, 0), t, size=size, prop=FONT).get_extents().width for t in lines
                )
            else:
                tw = (
                    TextPath((0, 0), label, size=size, prop=FONT).get_extents().width
                    if label
                    else 0
                )
            w = max(30, tw + 16)
        self.rrect(x - w / 2, y0 - pad, x + w / 2, y1 + pad, fill=fill)
        ym = (y0 + y1) / 2
        if lines:
            n = len(lines)
            for i, t in enumerate(lines):
                self.text(x, ym + (i - (n - 1) / 2) * size * 1.35, t, size=size)
        elif label:
            self.text(x, ym, label, size=size)
        return w

    def ctrl(self, x, y, on=True, r=4.2):
        self.circle(x, y, r, fill=INK if on else "#FFFFFF", z=5)

    def sqctrl(self, x, y, s=5):
        self.rrect(x - s, y - s, x + s, y + s, fill="#FFFFFF", r=1.5, z=5)

    def vline(self, x, y0, y1):
        self.line(x, y0, x, y1, z=2, wobble=0)

    def target(self, x, y, r=9):
        self.circle(x, y, r, fill="#FFFFFF", z=4)
        self.line(x - r, y, x + r, y, z=5, wobble=0)
        self.line(x, y - r, x, y + r, z=5, wobble=0)

    def meter(self, x, y, s=13, fill=GY0):
        self.rrect(x - s, y - s, x + s, y + s, fill=fill)
        pts = [(x - 8 + 16 * i / 10, y + 4 - 9 * math.sin(math.pi * i / 10)) for i in range(11)]
        self.polyline(pts, z=5, wobble=0.3)
        self.line(x, y + 5, x + 7, y - 8, z=5, wobble=0.3)

    def hdots(self, x, y, n=3, gap=6, r=1.5):
        for i in range(n):
            self.circle(x + (i - (n - 1) / 2) * gap, y, r, fill=INK, z=5)

    def vdots(self, x, y, n=3, gap=6, r=1.5):
        for i in range(n):
            self.circle(x, y + (i - (n - 1) / 2) * gap, r, fill=INK, z=5)

    def panel(self, x0, y0, x1, y1, fill, stroke=None, dash=False, r=10):
        self.rrect(
            x0, y0, x1, y1, fill=fill, stroke=stroke or "none", r=r, z=0, width=0.8, dash=dash
        )

    # ---------- output ----------
    def svg(self, margin=10):
        x0, y0, x1, y1 = self.bbox
        x0, y0, x1, y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
        w, h = x1 - x0, y1 - y0
        if w / h < ASPECT:
            nw = h * ASPECT
            x0 -= (nw - w) / 2
            w = nw
        else:
            nh = w / ASPECT
            y0 -= (nh - h) / 2
            h = nh
        scale = DISPLAY_W / w  # px per unit when shown as a thumbnail
        sw = min(max(1.35 / scale, 1.8), 3.0)
        amp = min(max(0.55 / scale, 0.6), 1.3)
        body = "\n".join(fn(sw, amp) for _, _, fn in sorted(self.items, key=lambda t: (t[0], t[1])))
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{_fmt(x0)} {_fmt(y0)} {_fmt(w)} {_fmt(h)}" '
            f'width="{_fmt(w)}" height="{_fmt(h)}">\n'
            f'<g stroke-linecap="round" stroke-linejoin="round">\n{body}\n</g>\n</svg>\n'
        )
