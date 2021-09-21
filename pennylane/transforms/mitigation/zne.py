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

"""Tools for zero-noise extrapolation."""
from functools import partial

from numpy.polynomial.polynomial import Polynomial

from pennylane import math
from pennylane.tape import get_active_tape
from pennylane.transforms import batch_transform


def _fit_zne(x_values, energies, degree=1):
    """Fit a polynomial to the energy values to extrapolate down to the
    zero-noise limit.

    Args:
        v_values (np.array): The x-axis values.
        energies (np.array): The set of energies for increasing number of
            CNOT pair insertions.
        degree (int): The degree of the polynomial to use for the fit.

    Returns:
        A polynomial of the specified degree that is the best fit to the data.
    """

    unwrapped_energies = math.stack(energies).reshape(len(energies))

    return Polynomial.fit(x_values, unwrapped_energies, degree, full=True)[0](0)


def _generate_transformed_tapes(tape, transform, max_arg_val):
    """Given a tape, transform, and max value of the transform argument,
    construct and return the set of tapes that need to be executed.

    """
    current_tape = get_active_tape()

    if current_tape is not None:
        with current_tape.stop_recording():
            tapes = [transform.tape_fn(tape, arg) for arg in range(1, max_arg_val + 1)]
    else:
        tapes = [transform.tape_fn(tape, arg) for arg in range(1, max_arg_val + 1)]

    return tapes


@batch_transform
def zne(tape, mitigation_transform, max_arg_val):
    """Given a tape and a mitigation transform, return the zero-extrapolated
    value computed according to the functionality of the provided transform.

    Args:
        qnode (pennylane.QNode or .QuantumTape): quantum tape or QNode
        mitigation_transform (function): A quantum function transform
            with which to perform multiple experiments to extrapolate over.
        max_arg_val (int): The maximum value of the argument accepted by the
            transform.

    Returns:
        func: Function which accepts the same arguments as the QNode. When called, this
        function will return the extrapolated expectation value.

    **Example**

    Suppose we have a noisy device (we'll use here PennyLane-Qiskit):

    .. code-block:: python3

        from qiskit.test.mock import FakeVigo
        from qiskit.providers.aer import QasmSimulator
        from qiskit.providers.aer.noise import NoiseModel

        device = QasmSimulator.from_backend(FakeVigo())
        noise_model = NoiseModel.from_backend(device)

        dev = qml.device(
            "qiskit.aer", backend='qasm_simulator', wires=3, shots=10000, noise_model=noise_model
        )

        # To ensure the noise doesn't get transpiled away
        dev.set_transpile_args(**{"optimization_level" : 0})

    Now we would like to estimate the result of a QNode in the absence of noise:

    .. code-block:: python3

        @zne(qml.transforms.cnot_pair_insertion, 8)
        @qml.qnode(dev)
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=1)
            qml.CNOT(wires=[1, 2])
            qml.RZ(z, wires=2)
            qml.CNOT(wires=[2, 0])
            return qml.expval(qml.PauliZ(0))

    We can estimate the noiseless value by performing CNOT pair insertion,
    for an increasing number of pairs, and then linearly extrapolate back
    down to 0 pairs. The ``zne_func`` transforms returns a function that
    will perform such extrapolation automatically. Note that the transform
    returns the full polynomial of best fit.

    >>> circuit(0.3, 0.4, 0.5)
    0.7760857142857142

    We can also apply this directly to tapes, which will help us see more clearly
    what is happening in the transform internally. For example,

    .. code-block:: python3

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.3, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[1, 2])
            qml.expval(qml.PauliZ(0))

    >>> tapes, processing_fn = zne(tape, qml.transforms.cnot_pair_insertion, 10)

    We can inspect each tape and see how the CNOT pair insertion has changed
    the list of operations.

    >>> tapes[0].operations
    [RX(0.3, wires=[0]),
     CNOT(wires=[0, 1]),
     CNOT(wires=[0, 1]),
     CNOT(wires=[0, 1]),
     RY(0.4, wires=[1]),
     CNOT(wires=[1, 2]),
     CNOT(wires=[1, 2]),
     CNOT(wires=[1, 2]),
     RZ(0.5, wires=[2]),
     CNOT(wires=[2, 0]),
     CNOT(wires=[2, 0]),
     CNOT(wires=[2, 0])]

    We can then execute all the tapes to see the list of expectation values
    obtained when using an increasing number of CNOTs.

    >>> res = qml.execute(tapes, dev, gradient_fn=qml.gradients.param_shift)
    >>> res
    [tensor([0.7608], requires_grad=True),
     tensor([0.7442], requires_grad=True),
     tensor([0.7632], requires_grad=True),
     tensor([0.731], requires_grad=True),
     tensor([0.7464], requires_grad=True),
     tensor([0.728], requires_grad=True),
     tensor([0.7288], requires_grad=True),
     tensor([0.7056], requires_grad=True)]

    Finally, we can retrive the extrapolated value using the processing function:

    >>> processing_fn(res)
    0.7681571428571428
    """

    # Generate all the transformed tapes
    transformed_tapes = _generate_transformed_tapes(tape, mitigation_transform, max_arg_val)

    # The processing function should only accept the results of the executed
    # tapes. Since we also need a set of "x values" for the fit, we use a
    # partial function with those values already populated.
    arg_range = math.arange(1, max_arg_val + 1)
    processing_fn = partial(_fit_zne, arg_range)

    return transformed_tapes, processing_fn
