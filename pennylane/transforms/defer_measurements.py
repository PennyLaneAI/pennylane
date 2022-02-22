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
"""Code for the tape transform implementing the deferred measurement principle."""
from pennylane.wires import Wires

import pennylane as qml
from pennylane.transforms import qfunc_transform, ctrl
from pennylane.queuing import apply
from pennylane.tape import QuantumTape, get_active_tape


@qfunc_transform
def defer_measurements(tape):
    """Quantum function transform that changes operations controlled on
    measurement outcomes in the middle of the circuit to controlled unitaries.

    This transform uses the `deferred measurement principle
    <https://en.wikipedia.org/wiki/Deferred_Measurement_Principle>`_.

    .. note::

        This transform does not extend the terminal measurements returned by
        the quantum function with the mid-circuit measurements.

    Args:
        qfunc (function): a quantum function

    **Example**

    Suppose we have a quantum function with mid-circuit measurements and
    conditional operations:

    .. code-block:: python3

        def func():
            qml.RY(0.123, wires=0)
            qml.Hadamard(wires=1)
            m_0 = qml.mid_measure(1)
            qml.if_then(m_0, qml.RY)(math.pi, wires=0)
            return qml.expval(qml.PauliZ(0))

    The ``defer_measurements`` transform enables us to run such a quantum
    function on any device.

    TODO:
    =======================

    The original circuit is:

    >>> dev = qml.device('default.qubit', wires=1)
    >>> qnode = qml.QNode(qfunc, dev)
    >>> print(qml.draw(qnode)())
     0: ──U0──┤ ⟨Z⟩
    U0 =
    [[-0.17111489+0.58564875j -0.69352236-0.38309524j]
     [ 0.25053735+0.75164238j  0.60700543-0.06171855j]]

    We can use the transform to decompose the gate:

    >>> transformed_qfunc = unitary_to_rot(qfunc)
    >>> transformed_qnode = qml.QNode(transformed_qfunc, dev)
    >>> print(qml.draw(transformed_qnode)())
     0: ──Rot(-1.35, 1.83, -0.606)──┤ ⟨Z⟩


    .. UsageDetails::

        This decomposition is not fully differentiable. We **can** differentiate
        with respect to input QNode parameters when they are not used to
        explicitly construct a :math:`4 \times 4` unitary matrix being
        decomposed. So for example, the following will work:

        .. code-block:: python3

            U = scipy.stats.unitary_group.rvs(4)

            def circuit(angles):
                qml.QubitUnitary(U, wires=["a", "b"])
                qml.RX(angles[0], wires="a")
                qml.RY(angles[1], wires="b")
                qml.CNOT(wires=["b", "a"])
                return qml.expval(qml.PauliZ(wires="a"))

            dev = qml.device('default.qubit', wires=["a", "b"])
            transformed_qfunc = qml.transforms.unitary_to_rot(circuit)
            transformed_qnode = qml.QNode(transformed_qfunc, dev)

        >>> g = qml.grad(transformed_qnode)
        >>> params = np.array([0.2, 0.3], requires_grad=True)
        >>> g(params)
        array([ 0.00296633, -0.29392145])

        However, the following example will **not** be differentiable:

        .. code-block:: python3

            def circuit(angles):
                z = angles[0]
                x = angles[1]

                Z_mat = np.array([[np.exp(-1j * z / 2), 0.0], [0.0, np.exp(1j * z / 2)]])

                c = np.cos(x / 2)
                s = np.sin(x / 2) * 1j
                X_mat = np.array([[c, -s], [-s, c]])

                U = np.kron(Z_mat, X_mat)

                qml.Hadamard(wires="a")

                # U depends on the input parameters
                qml.QubitUnitary(U, wires=["a", "b"])

                qml.CNOT(wires=["b", "a"])
                return qml.expval(qml.PauliX(wires="a"))
    """
    # TODO: do we need a map or can we just have a list?
    measured_wires = {}
    for op in tape.queue:
        if any([wire in measured_wires.values() for wire in op.wires]):
            raise ValueError("Cannot apply operations on {op.wires} as some has been measured already.")

        if isinstance(op, qml.ops.mid_circuit_measure._MidCircuitMeasure):
            measured_wires[op.measurement_id] = op.wires[0]

        elif op.__class__.__name__ == "_IfOp":
            # TODO: Why does op.dependant_measurements store the wire ids instead of labels?
            control = [measured_wires[m_id] for m_id in op.dependant_measurements]
            for branch, value in op.branches.items():
                if value:
                    ctrl(lambda: apply(op.then_op), control=Wires(control))()
        else:
            apply(op)
