# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTests for the AutoGraph source-to-source transformation feature."""

# pylint: disable=wrong-import-order, wrong-import-position, ungrouped-imports

import numpy as np
import pytest

import pennylane as qml
from pennylane import cond, measure

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")

# must be below jax importorskip
from pennylane.capture.autograph.transformer import TRANSFORMER

check_cache = TRANSFORMER.has_cache


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


if __name__ == "__main__":
    pytest.main(["-x", __file__])
