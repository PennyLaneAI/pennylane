# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
The data subpackage provides functionality to access, store and manipulate quantum datasets within the PennyLane framework.

Datasets are generally stored and accessed using the :class:`~pennylane.data.Dataset` class.
Pre-computed datasets are available for download and can be accessed using the :func:`~pennylane.data.load` function.
Alternatively, users can easily create, write to disk, and read custom datasets using functions within the
:class:`~pennylane.data.Dataset` class.

.. currentmodule:: pennylane.data
.. autosummary::
   :toctree: api

   qml.data.dataset
   qml.data.data_manager
"""

from .dataset import Dataset
from .data_manager import load, list_datasets
