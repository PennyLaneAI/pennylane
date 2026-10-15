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

# Portions of this file (the rotation sequences below) are derived from
# pygridsynth: https://github.com/quantum-programming/pygridsynth
#
# MIT License
#
# Copyright (c) 2024-2025 Shun Yamamoto and Nobuyuki Yoshioka
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Hardcoded PPR decomposition data for :class:`~.PhaseGradientStatePrep`.

Wires 0 to 2 are based on the exact gates Z, S^dagger and T^dagger. For j >= 3, the PPRs are
based on Clifford+T approximations of RZ(-pi / 2**j) obtained with pygridsynth v2.0.0
The gate sequence for wire j is obtained by
    pygridsynth.gridsynth_gates(-mpmath.pi / 2**j, mpmath.mpf(epsilon), up_to_phase=True)
with the following tolerances epsilon (wire: epsilon):
    3: 2.5e-12, 4: 4e-13, 5: 5e-13, 6: 3e-13, 7: 6e-12, 8: 3e-13, 9: 1e-12, 10: 2e-13,
    11: 6e-13, 12: 2e-12, 13: 4e-13, 14: 2.1e-12, 15: 3e-13, 16: 8e-14, 17: 1.3e-12,
    18: 7e-13, 19: 3e-13, 20: 3e-12, 21: 2e-12, 22: 2e-13, 23: 6e-13, 24: 2e-13, 25: 3e-13,
    26: 5e-13, 27: 8e-14, 28: 2.66e-13, 29: 5.3e-13
The sequence, and the preceding Hadamard, is processed into a pure π/8 PPR sequence, up to a
leading Clifford PPR.
"""

# fmt: off
INITIAL_PPRS: tuple[tuple[int, str]] = (
    (-4, "Y"), (4, "X"), (4, "X"), (2, "X"), (4, "X"), (4, "Y"), (4, "X"), (-4, "X"), (2, "X"),
    (-4, "X"), (-4, "X"), (4, "Y"), (4, "X"), (-4, "Y"), (-4, "X"), None, (2, "X"), (-4, "X"),
    (2, "X"), (4, "Y"), (-4, "Y"), (4, "X"), (4, "Y"), (4, "X"), (2, "X"), (4, "X"), None,
    (4, "Y"), (4, "X"), (2, "X"),
)

PI_OVER_8_PPRS: tuple[str] = (
    "",
    "",
    "Z",
    "yZyXyXZyZyXyzyzXyzXyXZXZyXZXZXYZYxZxZxyZxyZyZxyxzyzyzyzyxyZyZyZyZyXyXyXZyXZyXyXyzyxyxzyzyxzyzyxyZxyxyxzyxzxYzxYxYxZYZXY",
    "xyxzyzXzXyzXyXZyXyXZyXyzXyzXzYzxzxzyxzxzxYxZxZYxZxyxzxYxYxZYZYxYzxYzxYzYzxYxYxZYxYxZYxYzxYzxYxZxyxzyzXyzXyzXzXzYXYXYZXZXYXY",
    "ZyZyXyzyxyZxyxzxYxYxYzYXYXzXzXzYzYXYXYXzXyXyzXzXyXZyZxyZxZxZYxZxZYxZYZYZYZYZYxZxZYxZxyxyxzxzxzyxzxYzYXzXyzyzXzXzXyXyXZXZXYZX",
    "xZYxYzYXYXYXYZXZXZyXyzXyXyzXzYzxYxZxZYZYxZxZYZYxZYxZYxYzYzYzYXzYXzYzYzxYxZxyxzyxyZyZyZxZYxYzYzYXzXzXzYXYZXZXYXYXYZXZXZXYZXZXYZX",
    "zXzXzYzxYzxzyzyxzyzXyzyzXyzXzYzxYxZYxZYZXYZYZYxZxZxZxyZyZxZYZYZXZXZyZyZxZxyxzyzyxzxYzxzxzxYxZYZYxYxZxyZyXZXZyZyZyX",
    "YxZYZYZXYXYZXZXZyXZXZXYZXZyXyXyzXzYzYzYzxYzxYzYzxzxzxYxYzxzxYzYzYXYXzYzYXzXyXZXYXzXzXzXzXyXyXZXYXzXzYzYXYZYxZYZYZXYZYZXZXYXY",
    "ZYxZxyZyXyXyXZXYXzYzxzxzyzyzXyXyXZXYZXZyXyXyzyzXyzXyzXzXzYzxYxZYxYzxzxYzYXzYzYXzXyXyzyzyxzyzyzyzXzXzYXYXYZYZYxZxZxZYxZYZY",
    "XZyXyzyzyxzyxyZxZxyxzxYxZxyZxZxZxyZxZxZYxZxZYxZYZXZXZyXZyZxyxzyzyzyxyZxZxyxzyzXzXyXZyZyZyZyXyzXyzyzXzYXYZXYXYXYZXYXYZYxZxyxzxYxZYZY",
    "YXYZYZYZXYXYXzYzYzxYxZxZYxYxYxYzxzyxzyxyxzyxyZxZYxYxYzxzyzyzXzYzxzxYzYXYZXYZYxYzYXzYXzYXzXyzXzYXzYXYZXYZYZYZXYXzXyzXzXyXZyZyX",
    "ZXZyZxZYxZYxZxZYZXZyZyXZyXZXZXYZXZyZyXZyXZXZXYZXZyZyZxZxZxyxyZyZyXZyZyZxyZxZYxYxZYZYZXYZXZyXyXyXZyXyXyzXzXyzyxyZyXyXyX",
    "zXzYXzYXzXzXyXZyZxyxzxzxzxYzYzYXzYzxYxZxZxyZyZxZxZxyxzxYxZxyZxyZxZxZYxYxZYxZxZYZXZyXyXyXyzXyXyzXyXyzXyXyzXyXZyZxyZyXZXZXYXYZY",
    "zyxyZxyxzyxzxzxYxYzxzxYxZxZYZXZyXyzyxzxYxYxZYZXZXZXZyXyzXzXzYzxzxzxYxZxZYxYzYXzXzYzYzYXzYXzYzYzxYxZYZXYXYXzXyzXyXZXZyZyX",
    "YzxYxYxZxyZxyxzxzyzyxzyxyZxyZxyxyZxyZxZYZYxZYZXYXYZXZyZyZyZyZyZxZYxYzxYzYXYZXZXZXYZXYZXZyZxZxZYxZYZXYXzXzXyzyxzxYxZYxZYxYzYXY",
    "xyxyxzyxzyzyxzyzXyzyxyxzxzyxzyxyZyZyXZXZXYZXYZXYXYXzYXzXyXyzXzYzYXzXzYzYXYZYxYzYzxYzxYxZYxZxZYZXYXYZYxYxZxyxzyzXyXyzyxzyzXyXZXZX",
    "XYZYxZYxZxyxyxyZxZYxYxZYxZxZxyZxZxZxyxyxyxyZxyZyZyXyzyzXzXyzyxyZyXyzyxyxzxYzYzxYxYxZxZxZYxYzxYzxYxZxyxzxYzYXzXzYXYXzXyXZ",
    "XyzXzYXzXzYXzXzYXYZYZXZXYZXYXYZXYXYXYXzXyzXzYXzYzYzYXzXyXyzXzXzYzYXYXYZYxZYZYZXYZXZyXZXYZYxYxZxZxZxyZxyZyZyXZXYXzYzxYzYzYXY",
    "yxzyzXyXyzyxzxzyzyxzxYxYxYzxzxYxYzYzxYzxzyzXzXyzyzyzyzyxzyxzxYxZYxZxZxZYZXZyXZyXZyXZXYXYZXYXYXYZYZYZXZyZyZyZyZxZxZYxZYZXZXZXY",
    "YZXYXYXYZYZXZXZyZyZyXZyZyZxyxzyxyZyZxZYZXZyXZXZyZxZYZYxYzxYzxYzxYxYzYXzXzYXzYXzXzXzYXzYXzXzXyzXzXzYzxzxzyxzxzyxzxYxZxZYZY",
    "xyxyZyXyzyzyzXzXyXyzXzXyzXyzyzyxzyxzxzxYzYXzYXzXyXZyZxZxZYxZYZYZXYZYZYZYxZYZYZYZXZyZyXZXZXZXZyZxyxyZxyZxyxzxYzxYzYXYXY",
    "yxzxzyzXyXZyXyzXyXZXYXzYXYZXZyXyzyzyzyzyxzxzxzyxyZyZxZYxZxyxyxzxYzxzxzyxzxzxYzYzxzyxzxzxYzYXYZYxZYxYxYzYXYXYZYxYxYxYxZYZXZyX",
    "zxzyzXyzXyzXzYXzXzXyXZXYXzYzxYxYzYzxYxYxZxZYxYxZYZYxYxYzxYxZYZXZXZXYZXZyXyXyzXyXZyXZyZxyxzyxzxzyzXyXyXZXZyXZXYZYZXYXYXYXYXY",
    "YzYXzYzxzxYxZYZXYZXYZXZyZyZyZyZyZxyZyZyZxyxzxYzYXzXzXyzXyXyzyzXzYXYXYXYZXYZYxYxYzYzxzyzyzXzYzYXYZXYXYXzXzXyXyXZyZyXyXZXYXYZXZyX",
    "ZxyxyxzxzxYzxzyxyZyXZyXyzXyXZXZXYZXZyXZXYXYZYxYzYXzXzYXzYzYzxzyzXzYzxzxYzxYxYzYzxzyxzyxzyzXyXyzyxzyxyZxyZyXZXZyZxyxyZxZYxZYZY",
    "xYzYzYXYXzYXzYzYzxzyzXyXyXyzyzyxzyzyzyxyxzyzyzyzyxyxyZyXyXZXZyZxyxyxyxyxzyzXzXzYzxYzxzyzXzXyXyXyzXyzXyzXzYXzYXYXzXzXzXzYXYXY",
    "YxYzYzxYxZxyZyZyXZyZyXZyXZyZxyZxyZyZyXZyZxZYxYxYzYXYZYZYZYZXYZYxZxZYxYxYxYxZxyxyxzyxyxyZxyxyZxZxyxzyzyxzyxyZyZxZYZXZXYXzYXYXzYXY",
    "xYzxYzxYzYXYXYXYZYZYZYZXYXzYXzYXYZXYXYZYZYZYxZYZXZyXyXyXyzXzXyzyzyxzxzxzyzXyXyzyzyzyzXyXZXYZYZXYXzXyzXyXZyZyXZyZyZxyZxyZxZxZYZYZY",
    "xYxZxZxyxzyzXyXyzXyzyzyxzyzXyXyzXyzXzXzXyzXyzyxyxyxyxyxyZxZYZYZXZXZXYXYXzXyzXzYzxzyzXzYzxYzYzxzyzyzyzXzYzxzyzXyzyxzxzyzXzYzYXzXyX",
)

GLOBAL_PHASES: tuple[float] = (
    0.0, 0.0, -0.39269908169872414, 0.19634954085053827, 2.4543692606172054, -1.9144080232811331,
    1.5953400194010237, 0.7976700097031784, -0.7792622402458426, -3.138524692014499,
    -0.7838641826095573, -3.1408256631960407, 1.5711798219926387, -0.785206415798946,
    -1.1780013712959445, -2.7488456349914205, -1.5707723583450672, 2.356206474417838,
    0.7854041555095971, 0.3927020777549259, 1.5707978248243835, 0.7853989124122822,
    -1.178096870589138, -2.356194302938653, -2.748893478264377, -1.9634953616802364,
    -2.748893548484341, -2.7488935601876974, 2.7488935777426624, -1.5707963238689968,
)
# fmt: on
