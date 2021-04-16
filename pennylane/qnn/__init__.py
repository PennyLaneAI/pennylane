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
"""This module contains classes and functions for constructing quantum neural networks from
QNodes."""
import importlib

import pennylane.qnn.cost

class_map = {"KerasLayer": "keras", "TorchLayer": "torch"}


def __getattr__(name):
    """Allow for lazy-loading of KerasLayer and TorchLayer so that TensorFlow and PyTorch are not
    automatically loaded with PennyLane"""
    if name in class_map:
        mod = importlib.import_module("." + class_map[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
