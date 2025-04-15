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
.. currentmodule:: pennylane.qaoa

Mixer Hamiltonians
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    mixers

Cost Hamiltonians
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    cost

QAOA Layers
~~~~~~~~~~~

.. autosummary::
    :toctree: api

    layers

Cycle Optimization
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    cycle

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
