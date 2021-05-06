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
This module contains functionality for converting PennyLane QNodes into layers that are compatible
with Keras and PyTorch.

.. note::

    Check out our :doc:`Keras <demos/tutorial_qnn_module_tf>` and
    :doc:`Torch <demos/tutorial_qnn_module_torch>` tutorials for further details.


.. rubric:: Classes

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: autosummary/class_no_inherited.rst

    ~KerasLayer
    ~TorchLayer
"""
import importlib

from . import cost

class_map = {"KerasLayer": "keras", "TorchLayer": "torch"}
mods = ("keras", "torch")


def __getattr__(name):
    """Allow for lazy-loading of KerasLayer and TorchLayer so that TensorFlow and PyTorch are not
    automatically loaded with PennyLane"""
    if name in class_map:
        mod = importlib.import_module("." + class_map[name], __name__)
        return getattr(mod, name)
    if name in mods:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
