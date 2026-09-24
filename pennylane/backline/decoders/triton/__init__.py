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

"""Triton-backed decoder builders for :mod:`~pennylane.backline`.

This package contains Triton kernels and build helpers for compiling syndrome
decoders into shared libraries that can be loaded by backline devices.

.. currentmodule:: pennylane.backline.decoders.triton
.. autosummary::
   :toctree: api

   decoder_frontend
   algorithms
   bp_iters
   persistent_kernel
   triton_so_builder
"""

__all__: list[str] = []
