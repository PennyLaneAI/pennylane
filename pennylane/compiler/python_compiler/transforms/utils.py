# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Registering transforms with plxpr to catalyst map"""

import pennylane as qml
from pennylane.devices.preprocess import null_postprocessing

from .apply_transform_sequence import register_pass


def xdsl_transform(_klass):
    """Register the xdsl transform into the plxpr to catalyst map"""

    # avoid dependency on catalyst
    # pylint: disable-next=import-outside-toplevel
    import catalyst

    def identity_transform(tape):
        """Stub, we only need the name to be unique"""
        return tape, null_postprocessing

    identity_transform.__name__ = "xdsl_transform" + _klass.__name__
    transform = qml.transform(identity_transform)

    # Map from plxpr to register transform
    catalyst.from_plxpr.register_transform(transform, _klass.name, False)

    # Register this pass as available in the apply-transform-sequence
    # interpreter
    def get_pass_instance():
        return _klass()

    register_pass(_klass.name, get_pass_instance)

    return transform
