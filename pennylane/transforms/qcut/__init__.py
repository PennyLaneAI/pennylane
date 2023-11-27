# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This subpackage contains quantum function transforms for cutting quantum circuits.
"""

from .utils import (
    replace_wire_cut_node,
    replace_wire_cut_nodes,
    fragment_graph,
    find_and_place_cuts,
    place_wire_cuts,
    _remove_existing_cuts,
    _get_optim_cut,
    _is_valid_cut,
)
from .tapes import (
    graph_to_tape,
    tape_to_graph,
    expand_fragment_tape,
    _qcut_expand_fn,
    _get_measurements,
    _find_new_wire,
    _add_operator_node,
)
from .cutcircuit import cut_circuit, _cut_circuit_expand
from .montecarlo import (
    cut_circuit_mc,
    _cut_circuit_mc_expand,
    expand_fragment_tapes_mc,
    MC_MEASUREMENTS,
    MC_STATES,
    _identity,
    _pauliX,
    _pauliY,
    _pauliZ,
)
from .processing import (
    qcut_processing_fn,
    qcut_processing_fn_sample,
    qcut_processing_fn_mc,
    contract_tensors,
    _process_tensor,
    _to_tensors,
    _reshape_results,
    _get_symbol,
)
from .kahypar import kahypar_cut, _graph_to_hmetis
from .cutstrategy import CutStrategy
from .qcut import (
    MeasureNode,
    PrepareNode,
    _prep_one_state,
    _prep_zero_state,
    _prep_plus_state,
    _prep_minus_state,
    _prep_iplus_state,
    _prep_iminus_state,
)
