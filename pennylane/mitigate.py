# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO"""
from functools import wraps

from pennylane.transforms import single_tape_transform
from pennylane.tape import QuantumTape


def to_mitiq(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        args = list(args)

        circuit_kwarg = kwargs.get("circuit", None)
        qp_kwarg = kwargs.get("qp", None)

        if circuit_kwarg is not None:
            tape_key = "circuit"
        elif qp_kwarg is not None:
            tape_key = "qp"
        else:
            tape_key = None

        tape = circuit_kwarg or qp_kwarg or args[0]
        dev = kwargs.get("executor", None) or args[1]

        tape_no_measurements = _remove_measurements(tape)

        def new_executor(updated_tape):
            with QuantumTape as updated_tape_with_measurements:
                for op in updated_tape.operations:
                    op.queue()

                for meas in tape.measurements:
                    meas.queue()

            return dev.execute(updated_tape_with_measurements)[0]

        if tape_key is not None:
            kwargs[tape_key] = tape_no_measurements
        else:
            args[0] = tape_no_measurements

        if kwargs.get("executor", None) is not None:
            kwargs["executor"] = new_executor
        else:
            args[1] = new_executor()

        return fn(*args, **kwargs)

    return wrapper


@single_tape_transform
def _remove_measurements(tape):
    """Removes the measurements of a given tape

    Args:
        tape (QuantumTape): input quantum tape which may include measurements

    Returns:
        QuantumTape: the input tape with the measurements removed
    """
    for op in tape.operations:
        op.queue()
