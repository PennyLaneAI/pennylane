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
This subpackage contains the quantum tape, which tracks, queues, and
validates quantum operations and measurements.
"""

from .operation_recorder import OperationRecorder
from .qscript import QuantumScript, QuantumScriptBatch, QuantumScriptOrBatch, make_qscript
from .tape import QuantumTape, QuantumTapeBatch, TapeError, expand_tape_state_prep

try:
    from .plxpr_conversion import plxpr_to_tape
except ImportError:  # pragma: no cover

    # pragma: no cover
    def plxpr_to_tape(jaxpr: "jax.core.Jaxpr", consts, *args, shots=None):  # pragma: no cover
        """A dummy version of ``plxpr_to_tape`` when jax is not installed on the system."""
        raise ImportError("plxpr_to_tape requires jax to be installed")  # pragma: no cover
