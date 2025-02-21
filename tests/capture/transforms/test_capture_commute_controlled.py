# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the ``CommuteControlledInterpreter`` class"""

# pylint:disable=wrong-import-position, unused-argument
import numpy as np
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]

from pennylane.transforms.optimization.commute_controlled import (
    CommuteControlledInterpreter,
    commute_controlled_plxpr_to_plxpr,
)
