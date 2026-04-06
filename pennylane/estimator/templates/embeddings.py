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
r"""Resource operators for embedding templates."""

from pennylane.estimator.templates.stateprep import BasisState


class BasisEmbedding(BasisState):
    r"""Resource class for preparing a single basis state, as an embedding.
    Mirrors :class:`~.BasisEmbedding`, which inherits from :class:`~.BasisState`
    but is otherwise identical to it.

    Args:
        num_wires (int): number of wires the operator acts on
        wires (WiresLike, Optional): the wire(s) the operation acts on
    """
