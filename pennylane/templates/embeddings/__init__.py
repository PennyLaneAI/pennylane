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
Embeddings are templates encode features (i.e., classical data) into a quantum state.
They can optionally be repeated, and may contain trainable parameters. Embeddings are typically
used at the beginning of a circuit.
"""

from .angle import AngleEmbedding
from .amplitude import AmplitudeEmbedding
from .basis import BasisEmbedding
from .displacement import DisplacementEmbedding
from .iqp import IQPEmbedding
from .qaoaembedding import QAOAEmbedding
from .squeezing import SqueezingEmbedding
