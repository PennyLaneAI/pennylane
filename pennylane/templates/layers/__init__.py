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
r"""
Layers are trainable templates that are typically repeated, using different adjustable parameters in each repetition.
They implement a transformation from a quantum state to another quantum state.
"""

from .strongly_entangling import StronglyEntanglingLayers
from .random import RandomLayers
from .cv_neural_net import CVNeuralNetLayers
from .simplified_two_design import SimplifiedTwoDesign
from .basic_entangler import BasicEntanglerLayers
from .particle_conserving_u2 import ParticleConservingU2
from .particle_conserving_u1 import ParticleConservingU1
from .gate_fabric import GateFabric
