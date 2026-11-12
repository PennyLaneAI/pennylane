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

"""Built-in decomposition rules for templates.embeddings.amplitude."""

# pylint: disable=too-many-arguments

from pennylane.decomposition import add_decomps
from pennylane.decomposition_rules.qubit.state_preparation import _state_prep_decomp
from pennylane.templates.embeddings.amplitude import AmplitudeEmbedding

add_decomps(AmplitudeEmbedding, _state_prep_decomp)
