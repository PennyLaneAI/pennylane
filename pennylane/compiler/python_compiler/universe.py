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
"""xDSL universe for containing all dialects and passes."""

from xdsl.passes import ModulePass
from xdsl.universe import Universe

from . import dialects, transforms

shared_dialects = ("stablehlo", "transform")
dialects_all = (getattr(dialects, name) for name in dialects.__all__)
transforms_all = (getattr(transforms, name) for name in transforms.__all__)

# Dialects that are already provided by xDSL cannot be loaded into the multiverse, so we
# don't add them to our universe
names_to_dialects = {
    dialect.name: dialect for dialect in dialects_all if dialect.name not in shared_dialects
}
# The transforms module contains PassDispatcher instances as well as ModulePasses. We only
# want to collect the ModulePasses. We cannot use issubclass with instances, which is why
# we first check if isinstance(transform, type).
names_to_passes = {
    transform.name: transform
    for transform in transforms_all
    if isinstance(transform, type) and issubclass(transform, ModulePass)
}

# The Universe is used to expose custom dialects and transforms to xDSL. It is specified
# as an entry point in PennyLane's pyproject.toml file, which makes it available to look
# up by xDSL for tools such as xdsl-opt, xdsl-gui, etc.
XDSL_UNIVERSE = Universe(all_dialects=names_to_dialects, all_passes=names_to_passes)
