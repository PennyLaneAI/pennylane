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
"""Unit tests for the xDSL universe."""

import pytest

pytestmark = pytest.mark.external
pytest.importorskip("xdsl")

# pylint: disable=wrong-import-position
from xdsl.passes import ModulePass
from xdsl.universe import Universe as xUniverse

from pennylane.compiler.python_compiler import dialects, transforms
from pennylane.compiler.python_compiler.universe import XDSL_UNIVERSE, shared_dialects

all_dialects = tuple(getattr(dialects, name) for name in dialects.__all__)
all_transforms = tuple(
    transform
    for name in transforms.__all__
    if isinstance((transform := getattr(transforms, name)), type)
    and issubclass(transform, ModulePass)
)


def test_correct_universe():
    """Test that all the available dialects and transforms are available in the universe."""
    for d in all_dialects:
        if d.name not in shared_dialects:
            assert d.name in XDSL_UNIVERSE.all_dialects
            assert XDSL_UNIVERSE.all_dialects[d.name] == d

    for t in all_transforms:
        assert t.name in XDSL_UNIVERSE.all_passes
        assert XDSL_UNIVERSE.all_passes[t.name] == t


def test_correct_multiverse():
    """Test that all the available dialects and transforms are available in the multiverse."""
    multiverse = xUniverse.get_multiverse()

    for d in all_dialects:
        assert d.name in multiverse.all_dialects
        if d.name not in shared_dialects:
            assert multiverse.all_dialects[d.name] == d

    for t in all_transforms:
        assert t.name in multiverse.all_passes
        assert multiverse.all_passes[t.name] == t
