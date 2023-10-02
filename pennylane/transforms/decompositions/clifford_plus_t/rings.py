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
"""Various ring definitions for the gridsynth implementation."""
# pylint:disable=too-few-public-methods

import numpy as np

SQRT2 = np.sqrt(2)


class RootTwo:
    """Any object that can be represented as a + b√2"""

    def __init__(self, a, b):
        """Default constructor."""
        self.a = a
        self.b = b

    def float(self):
        """Convert from ring to float."""
        return self.a + self.b * SQRT2


class DRootTwo(RootTwo):
    """Dyadic Root-Two ring."""
