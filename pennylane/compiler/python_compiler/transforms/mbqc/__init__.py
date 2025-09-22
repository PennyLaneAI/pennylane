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
"""PennyLane-xDSL transformations API specifically for the MBQC transform."""

from .convert_to_mbqc_formalism import ConvertToMBQCFormalismPass, convert_to_mbqc_formalism_pass
from .decompose_graph_state import (
    DecomposeGraphStatePass,
    NullDecomposeGraphStatePass,
    decompose_graph_state_pass,
    null_decompose_graph_state_pass,
)

from .graph_state_utils import (
    get_num_aux_wires,
    get_graph_state_edges,
    n_vertices_from_packed_adj_matrix,
    edge_iter,
    generate_adj_matrix,
)


__all__ = [
    # Passes
    "ConvertToMBQCFormalismPass",
    "DecomposeGraphStatePass",
    "NullDecomposeGraphStatePass",
    "null_decompose_graph_state_pass",
    # Utils
    "get_num_aux_wires",
    "decompose_graph_state_pass",
    "get_graph_state_edges",
    "n_vertices_from_packed_adj_matrix",
    "edge_iter",
    "generate_adj_matrix",
]
