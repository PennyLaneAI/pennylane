# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains 1-D and 2-D grid problem solving utilities."""


# pylint: disable=too-few-public-methods
class Ellipse:
    """A class representing an ellipse in a 2D grid problem.

    Args:
        center (tuple[float, float]): The center of the ellipse (x, y).
        axes (tuple[float, float]): The lengths of the semi-major and semi-minor axes (a, b).
        angle (float): The rotation angle of the ellipse in radians.
    """

    def __init__(self, center, axes, angle):
        self.center = center
        self.axes = axes
        self.angle = angle
