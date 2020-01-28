# Copyright 2018-2019 Xanadu Quantum Technologies Inc.

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
Embeddings are templates encode features (i.e., classical data) into a quantum state.
They can optionally be repeated, and may contain trainable parameters. Embeddings are typically
used at the beginning of a circuit.
"""

from .angle_embedding import AngleEmbedding
from .amplitude_embedding import AmplitudeEmbedding
from .basis_embedding import BasisEmbedding
from .displacement_embedding import DisplacementEmbedding
from .qaoa_embedding import QAOAEmbedding
from .squeezing_embedding import SqueezingEmbedding

__all__ = ["AngleEmbedding",
           "AmplitudeEmbedding",
           "BasisEmbedding",
           "DisplacementEmbedding",
           "QAOAEmbedding",
           "SqueezingEmbedding"]
