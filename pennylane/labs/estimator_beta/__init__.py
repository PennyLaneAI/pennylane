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
r"""
This module contains experimental features for
resource estimation.

.. warning::

    This module is experimental. Frequent changes will occur,
    with no guarantees of stability or backwards compatibility.

.. currentmodule:: pennylane.labs.estimator_beta

"""
import pennylane as qml
# from pennylane.estimator.estimate import estimate
from .estimate import estimate as estimate_new


# class _EnableExpQRE:
#     default_estimate = estimate
#     new_estimate = estimate_new

#     def __init__(self):
#         return


# def enable_experimental_qre():
#     # cls = _EnableExpQRE()
#     default_estimate = estimate
#     qml.estimator.estimate = estimate_new
#     return default_estimate


# def disable_experimental_qre(default_estimate):
#     # cls = _EnableExpQRE()
#     qml.estimator.estimate = default_estimate
