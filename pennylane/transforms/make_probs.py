# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transform to strip return_types and return probabilities instead."""
# pylint: disable=import-outside-toplevel

import pennylane as qml

from .batch_transform import batch_transform


@batch_transform
def _make_probs(tape, wires=None, post_processing_fn=None):
    """Ignores the return types of any qnode and creates a new one that outputs probabilities"""
    if wires == None:
        wires = tape.wires

    with qml.tape.QuantumTape() as new_tape:
        for op in tape.operations:
            qml.apply(op)
        qml.probs(wires=wires)

    if post_processing_fn == None:
        post_processing_fn = lambda x: qml.math.squeeze(qml.math.stack(x))

    return [new_tape], post_processing_fn
