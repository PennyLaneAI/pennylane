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

"""This submodule contains xDSL dialects for the Python compiler."""

from .catalyst import Catalyst
from .mbqc import MBQC
from .quantum import Quantum
from .qec import QEC
from .stablehlo import StableHLO
from .transform import Transform


def get_universe_dialects():
    """Get a mapping between all available dialect names to dialect classes. This
    is used to initialize the PennyLane-xDSL universe, which is needed to make the
    dialects readily available to xDSL command-line tools."""
    return {
        "catalyst": Catalyst,
        "mbqc": MBQC,
        "quantum": Quantum,
        "qec": QEC,
        # Dialects available in xDSL already cannot be added to the universe, since
        # the "multiverse" cannot have duplicate dialects
        # "stablehlo": StableHLO,
        # "transform": Transform,
    }


__all__ = ["Catalyst", "MBQC", "Quantum", "QEC", "StableHLO", "Transform"]
