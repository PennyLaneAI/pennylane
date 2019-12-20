# Copyright 2018-2019 Xanadu Quantum Technologies Inc.

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
This module contains experimental, contributed, and beta code.
"""
import pennylane as qml
from pennylane.operation import All, Observable, Operation, Probability
from pennylane.ops import Identity

import pennylane.beta.vqe


def prob(wires):
    r"""Probability of each computational basis state.

    This measurement function accepts no observables, and instead
    instructs the QNode to return a flat array containing the
    probabilities of each quantum state.

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    # pylint: disable=protected-access
    op = Identity(wires=wires, do_queue=False)
    op.return_type = Probability

    if qml._current_context is not None:
        # add observable to QNode observable queue
        qml._current_context._append_op(op)

    return op
