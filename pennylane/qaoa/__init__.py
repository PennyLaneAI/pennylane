# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Overview
--------

This module provides a collection of methods that help in the construction of
QAOA workflows.

.. currentmodule:: pennylane.qaoa

Mixer Hamiltonians
~~~~~~~~~~~~~~~~~~

Methods for constructing QAOA mixer Hamiltonians.

.. autosummary::
    :toctree: api

    mixers.bit_flip_mixer
    mixers.x_mixer
    mixers.xy_mixer

Cost Hamiltonians
~~~~~~~~~~~~~~~~~

Methods for generating QAOA cost Hamiltonians corresponding to
different optimization problems.

.. autosummary::
    :toctree: api

    cost.bit_driver
    cost.edge_driver
    cost.max_clique
    cost.max_independent_set
    cost.max_weight_cycle
    cost.maxcut
    cost.min_vertex_cover

QAOA Layers
~~~~~~~~~~~

Methods that define cost and mixer layers for use in QAOA workflows.

.. autosummary::
    :toctree: api

    layers.cost_layer
    layers.mixer_layer

Cycle Optimization
~~~~~~~~~~~~~~~~~~

Functionality for finding the maximum weighted cycle of directed graphs.

.. autosummary::
    :toctree: api

    cycle.cycle_mixer
    cycle.edges_to_wires
    cycle.loss_hamiltonian
    cycle.net_flow_constraint
    cycle.out_flow_constraint
    cycle.wires_to_edges

"""

from .mixers import x_mixer, xy_mixer, bit_flip_mixer
from .cycle import (
    edges_to_wires,
    wires_to_edges,
    cycle_mixer,
    loss_hamiltonian,
    out_flow_constraint,
    net_flow_constraint,
)
from .cost import (
    bit_driver,
    edge_driver,
    maxcut,
    max_independent_set,
    min_vertex_cover,
    max_clique,
    max_weight_cycle,
)
from .layers import cost_layer, mixer_layer

__all__ = [
    "x_mixer",
    "xy_mixer",
    "bit_flip_mixer",
    "edges_to_wires",
    "wires_to_edges",
    "cycle_mixer",
    "loss_hamiltonian",
    "out_flow_constraint",
    "net_flow_constraint",
    "bit_driver",
    "edge_driver",
    "maxcut",
    "max_independent_set",
    "min_vertex_cover",
    "max_clique",
    "max_weight_cycle",
    "cost_layer",
    "mixer_layer",
]
