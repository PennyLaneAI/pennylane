# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Various conversion functions for the gridsynth implementation."""
# pylint:disable=missing-function-docstring

import numpy as np


def adj2(x):
    """Map from ``a + b√2`` to ``a - b√2``."""
    return x


def fatten_interval(x, y):
    epsilon = 0.0001 * (y - x)
    return x - epsilon, y + epsilon


def from_d_root_two(x):
    return x


def from_d_omega(x):
    return x


def from_z_root_two(x):
    return x


def operator_to_bl2z(m):
    return m[0, 1], m[1, 1] / m[0, 0]


def point_from_d_root_two(point):
    return point


def action(a, b, g):
    g4 = op_from_d_root_two(adj2(g))
    g3 = np.conj(g4).T
    g2 = op_from_d_root_two(g)
    g1 = np.conj(g2).T
    return g1 @ a @ g2, g3 @ b @ g4


def denomexp(x):
    """Return the smallest denominator exponent of ``x``."""
    return x


def op_from_d_root_two(op):
    return from_d_root_two(op)


def op_from_d_omega(op):
    return from_d_omega(op)
