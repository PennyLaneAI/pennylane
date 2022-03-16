# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pennylane as qml


class Control(qml.operation.Operator):
    """Arithmetic operator subclass representing the controlled version of a unitary operator."""

    def __init__(self, op, control_wires, control_values, do_queue=True, id=None):
        if not isinstance(op, qml.operation.Operation):
            raise ValueError("Can only control unitary operations on wires.")

        self.hyperparameters["base_op"] = op
        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["control_values"] = control_values
        combined_wires = qml.wires.Wires.all_wires([qml.wires.Wires(control_wires), op.wires])
        super().__init__(*op.parameters, wires=combined_wires, do_queue=do_queue, id=id)
        self._name = f"Controlled({op}, {control_wires})"

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"c-{self.hyperparameters['base_op']}"

    @property
    def num_wires(self):
        return len(self.wires)

    @staticmethod
    def compute_decomposition(*params, wires=None, base_op=None, control_wires=None, control_values=None, **hyperparameters):
        return [control(o, control_wires, control_values) for o in base_op.decomposition()]


    @staticmethod
    def compute_matrix(*args, base_op=None, control_wires=None, control_values=None, work_wires=None, **kwargs):

        base_matrix = base_op.compute_matrix(*args, **kwargs)

        base_matrix_size = qml.math.shape(base_matrix)[0]
        num_control_states = 2**len(control_wires)
        total_matrix_size = num_control_states * base_matrix_size

        if control_values is None:
            control_int = 0
        else:
            control_int = sum(2**i * v for i, v in enumerate(reversed(control_values)))

        padding_left = control_int * base_matrix_size
        padding_right = total_matrix_size - base_matrix_size - padding_left

        interface = qml.math.get_interface(base_matrix)
        left_pad = qml.math.cast_like(qml.math.eye(padding_left, like=interface), 1j)
        right_pad = qml.math.cast_like(qml.math.eye(padding_right, like=interface), 1j)

        return qml.math.block_diag([left_pad, base_matrix, right_pad])


def control(
    op, control_wires, control_values=None
):
    if control_values is None:
        control_values = [1]*len(control_wires)

    try:
        # there is a custom version defined
        return op.control(control_wires, control_values)
    except AttributeError:
        # default to an abstract arithmetic class
        return Control(op, control_wires, control_values)
