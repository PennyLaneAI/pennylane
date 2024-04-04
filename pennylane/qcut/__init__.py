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
This module contains quantum function transforms for cutting quantum circuits.

.. currentmodule:: pennylane

Overview
--------

This module defines transform functions for circuit cutting. This allows
for 'cutting' (or splitting) of large circuits into smaller circuits, to allow
them to be executed on devices that have a restricted number of qubits.

Transforms for circuit cutting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~cut_circuit
    ~cut_circuit_mc


Utility functions
~~~~~~~~~~~~~~~~~

There are also low-level functions that can be used to build up the circuit cutting functionalities:

.. autosummary::
    :toctree: api

    ~qcut.tape_to_graph
    ~qcut.replace_wire_cut_nodes
    ~qcut.fragment_graph
    ~qcut.graph_to_tape
    ~qcut.expand_fragment_tape
    ~qcut.expand_fragment_tapes_mc
    ~qcut.qcut_processing_fn
    ~qcut.qcut_processing_fn_sample
    ~qcut.qcut_processing_fn_mc
    ~qcut.CutStrategy
    ~qcut.kahypar_cut
    ~qcut.place_wire_cuts
    ~qcut.find_and_place_cuts

Cutting Circuits
----------------

Circuit cutting can allow you to replace a circuit with ``N`` wires by a set
of circuits with less than ``N`` wires (see also
`Peng et. al <https://arxiv.org/abs/1904.00102>`_). This comes
with a cost: the smaller circuits require a greater number of device
executions to be evaluated.

In PennyLane, circuit cutting for circuits that terminate in expectation values
can be activated by positioning :class:`~.pennylane.WireCut` operators at the
desired cut locations, and by decorating the QNode with
the :func:`~.pennylane.cut_circuit` transform.

Cut circuits remain fully differentiable, and the resulting circuits can be
executed on parallel devices if available. Please see the
:func:`~.pennylane.cut_circuit` documentation for more details.

.. note::

    Simulated quantum circuits that produce samples can be cut using
    the :func:`~.pennylane.cut_circuit_mc`
    transform, which is based on the Monte Carlo method.

Automatic cutting
-----------------

PennyLane also has experimental support for automatic cutting of circuits ---
that is, the ability to determine optimum cut location without explicitly
placing :class:`~.pennylane.WireCut` operators. This can be enabled by using the
``auto_cutter`` keyword argument of :func:`~.pennylane.cut_circuit`; refer to the
function documentation for more details.
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
    MeasureNode,
    PrepareNode,
    _prep_one_state,
    _prep_zero_state,
    _prep_plus_state,
    _prep_minus_state,
    _prep_iplus_state,
    _prep_iminus_state,
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
from .cutcircuit_mc import (
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
