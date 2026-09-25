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
"""
This subpackage contains the emission of traced gadgets into xDSL IR and their lowering to
Catalyst's ``qecl`` dialect.

It requires `xDSL <https://xdsl.dev>`__ and Catalyst's dialect definitions, which are
imported from an installed Catalyst or loaded from a source checkout (see
:func:`load_dialect_module`). The rest of :mod:`pennylane.gadget` does not depend on it.
"""

from .catalyst_dialects import CatalystNotFound, load_dialect_module
from .dialect import (
    DeformOp,
    DetachOp,
    DetectorsOp,
    FrameUpdateOp,
    GadgetDialect,
    ObservableOp,
    PhaseOp,
    RecordsType,
    RoundsOp,
)
from .emit import QECL_SOURCE, Emission, codeblock_type, context, emit
from .passes import LoweringGap, LoweringResult, lower_gadget_to_qecl

__all__ = [
    "CatalystNotFound",
    "load_dialect_module",
    "GadgetDialect",
    "RecordsType",
    "PhaseOp",
    "DeformOp",
    "RoundsOp",
    "DetachOp",
    "ObservableOp",
    "FrameUpdateOp",
    "DetectorsOp",
    "Emission",
    "emit",
    "context",
    "codeblock_type",
    "QECL_SOURCE",
    "LoweringGap",
    "LoweringResult",
    "lower_gadget_to_qecl",
]
