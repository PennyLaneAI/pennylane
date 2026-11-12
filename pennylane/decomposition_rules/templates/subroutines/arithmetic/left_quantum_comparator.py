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

"""Built-in decomposition rules for templates.subroutines.arithmetic.left_quantum_comparator."""

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, X
from pennylane.templates.subroutines.arithmetic.left_quantum_comparator import LeftQuantumComparator
from pennylane.templates.subroutines.arithmetic.temporary_and import Elbow
from pennylane.wires import Wires


def _left_quantum_comparator_resources(x_wires, y_wires, target_wire, work_wires, comparator):
    # pylint: disable=unused-argument
    num_y_wires = len(y_wires)
    resources = {
        Elbow: num_y_wires,
        CNOT: 2 + 5 * (num_y_wires - 1),
    }

    if comparator in [">=", "<="]:
        resources[X] = 1

    return resources


@register_resources(_left_quantum_comparator_resources, exact=True)
def _left_quantum_comparator(
    x_wires, y_wires, target_wire, work_wires, comparator, **_
):  # pylint: disable=too-many-arguments

    # revert to follow PL convention
    x_wires = x_wires[::-1]
    y_wires = y_wires[::-1]

    if comparator in ("<", ">="):
        x_wires, y_wires = y_wires, x_wires

    used_work_wires = Wires.all_wires([work_wires[: len(x_wires) - 1], target_wire])

    CNOT(wires=[x_wires[0], y_wires[0]])
    Elbow(wires=[x_wires[0], y_wires[0], used_work_wires[0]])
    CNOT(wires=[x_wires[0], y_wires[0]])

    if compiler.active() or capture.enabled():
        x_wires = math.array(x_wires, like="jax")
        y_wires = math.array(y_wires, like="jax")
        used_work_wires = math.array(used_work_wires, like="jax")

    # pylint: disable=no-value-for-parameter
    @for_loop(1, len(x_wires))
    def _loop(i):
        CNOT(wires=[x_wires[i], y_wires[i]])
        CNOT(wires=[x_wires[i], used_work_wires[i - 1]])
        Elbow(wires=[used_work_wires[i - 1], y_wires[i], used_work_wires[i]])
        CNOT(wires=[x_wires[i], used_work_wires[i - 1]])
        CNOT(wires=[used_work_wires[i - 1], used_work_wires[i]])
        CNOT(wires=[x_wires[i], y_wires[i]])

    _loop()

    if comparator in ("<=", ">="):
        X(target_wire)


add_decomps(LeftQuantumComparator, _left_quantum_comparator)
