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

from .cutcircuit import _cut_circuit_expand, cut_circuit
from .cutcircuit_mc import (
    MC_MEASUREMENTS,
    MC_STATES,
    _cut_circuit_mc_expand,
    _identity,
    _pauliX,
    _pauliY,
    _pauliZ,
    cut_circuit_mc,
    expand_fragment_tapes_mc,
)
from .cutstrategy import CutStrategy
from .kahypar import _graph_to_hmetis, kahypar_cut
from .processing import (
    _get_symbol,
    _process_tensor,
    _reshape_results,
    _to_tensors,
    contract_tensors,
    qcut_processing_fn,
    qcut_processing_fn_mc,
    qcut_processing_fn_sample,
)
from .tapes import (
    _add_operator_node,
    _find_new_wire,
    _get_measurements,
    _qcut_expand_fn,
    expand_fragment_tape,
    graph_to_tape,
    tape_to_graph,
)
from .utils import (
    MeasureNode,
    PrepareNode,
    _get_optim_cut,
    _is_valid_cut,
    _prep_iminus_state,
    _prep_iplus_state,
    _prep_minus_state,
    _prep_one_state,
    _prep_plus_state,
    _prep_zero_state,
    _remove_existing_cuts,
    find_and_place_cuts,
    fragment_graph,
    place_wire_cuts,
    replace_wire_cut_node,
    replace_wire_cut_nodes,
)
