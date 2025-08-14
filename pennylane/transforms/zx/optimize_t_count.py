# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
This module contains a transform to apply the
`full_optimize <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.full_optimize>`__
pass (available through the external `pyzx <https://pyzx.readthedocs.io/en/latest/index.html>`__ package)
to a PennyLane Clifford + T circuit.
"""

from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn

from .converter import from_zx, to_zx
from .helper import _needs_pyzx


@_needs_pyzx
@transform
def optimize_t_count(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """TODO"""
    # pylint: disable=import-outside-toplevel
    import pyzx

    pyzx_graph = to_zx(tape)
    pyzx_circ = pyzx.Circuit.from_graph(pyzx_graph)

    try:
        pyzx_circ = pyzx.full_optimize(pyzx_circ)
    except TypeError:
        raise TypeError("The input quantum circuit must be a Clifford + T circuit.") from None

    qscript = from_zx(pyzx_circ.to_graph())
    new_tape = tape.copy(operations=qscript.operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumScript``.
        """
        return results[0]

    return (new_tape,), null_postprocessing
