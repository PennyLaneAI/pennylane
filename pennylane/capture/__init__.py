# Copyright 2024 Xanadu Quantum Technologies Inc.

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
.. currentmodule:: pennylane

This module implements PennyLane's capturing mechanism for hybrid
quantum-classical programs.

.. warning::

    This module is experimental and will change significantly in the future.


To activate and deactivate the new PennyLane program capturing mechanism, use
the switches ``qml.capture.enable_plxpr`` and ``qml.capture.disable_plxpr``.
Whether or not the capturing mechanism is currently being used can be
queried with ``qml.capture.plxpr_enabled``.
By default, the mechanism is disabled:

.. code-block:: pycon

    >>> import pennylane as qml
    >>> qml.capture.plxpr_enabled()
    False
    >>> qml.capture.enable_plxpr()
    >>> qml.capture.plxpr_enabled()
    True
    >>> qml.capture.disable_plxpr()
    >>> qml.capture.plxpr_enabled()
    False

"""
from .switches import enable_plxpr, disable_plxpr, plxpr_enabled
