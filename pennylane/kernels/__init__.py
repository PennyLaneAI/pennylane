# Copyright

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
This subpackage defines functions that relate to quantum kernel methods.

.. currentmodule:: pennylane.kernels
.. autosummary::
    :toctree: api
"""
from .embedding_kernel import EmbeddingKernel
from .cost_functions import kernel_polarization, kernel_target_alignment, kernel_matrix
