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

from numpy.polynomial.polynomial import Polynomial

from pennylane.tape import get_active_tape
from pennylane.math import stack, arange


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
    return Polynomial.fit(x_values, energies, degree, full=True)


def _generate_transformed_tapes(tape, transform, max_arg_val):
    """ Given a tape, transform, and max value of the transform argument,
    construct and return the set of tapes that need to be executed.

    """
    current_tape = get_active_tape()

    if current_tape is not None:
        with current_tape.stop_recording():
            tapes = [transform.tape_fn(tape, arg) for arg in range(1, max_arg_val + 1)]
    else:
        tapes = [transform.tape_fn(tape, arg) for arg in range(1, max_arg_val + 1)]

    return tapes


def zne(qnode, mitigation_transform, max_arg_val):
    """Given a tape and a mitigation transform, return the zero-extrapolated
    value computed according to the functionality of the provided transform.

    Args:
        qnode (.QNode): a QNode
        mitigation_transform (function): A quantum function transform
            with which to perform multiple experiments to extrapolate over.
        max_arg_val (int): The maximum value of the argument accepted by the
            transform.

    Returns:
        float, array: The linearly extrapolated value of the function at 0,
            and the data produced by evaluating the QNode for each point.

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
    will perform such extrapolation automatically.

    >>> qnode = qml.QNode(circuit, dev)
    >>> zne_func = qml.transforms.zne(qnode, qml.transforms.cnot_pair_insertion, 6)
    >>> extrap_val, data = zne_func(0.5, 0.1, -0.2)
    >>> extrap_val
    0.8390400000000006
    >>> data
    tensor([0.8302, 0.8176, 0.8144, 0.7962, 0.7942, 0.781], requires_grad=True)
    """

    def _zne_function(*args, **kwargs):
        qnode.construct(args, kwargs)
        original_tape = qnode.qtape

        transformed_tapes = _generate_transformed_tapes(
            original_tape, mitigation_transform, max_arg_val
        )

        res = stack([t.execute(device=qnode.device) for t in transformed_tapes]).reshape(
            len(transformed_tapes)
        )

        poly_results = _fit_zne(arange(1, max_arg_val + 1), res)

        # Return the value of the extrapolated function at 0
        return poly_results[0](0), res

    return _zne_function
