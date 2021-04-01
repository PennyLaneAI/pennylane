# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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


def simplify_tape(tape, original_inputs):

    def stop_at(obj):
        """Accepts a queue object and returns ``True``
           if this object should *not* be expanded."""

        # if the object in the queue is a tape, expand it,
        # so we do not have nested tape structures
        if isinstance(obj, qml.tape.QuantumTape):
            return False

        takes_input = any(p in original_inputs for p in obj.parameters)
        is_one_param_gate = hasattr(obj, "generator")
        if takes_input and not is_one_param_gate:

            # check if the new trainable parameters are
            # all identical to some original trainable
            # parameters - this makes sure no new parameters
            # were created that depend on the inputs
            expanded = obj.expand()
            new_inputs = expanded.get_parameters(trainable_only=True)
            if any(i not in original_inputs for i in new_inputs):
                raise ValueError(f"{obj} performs non-trivial manipulation of the inputs. "
                                 f"Aborted the expansion of the tape.")

            return True

    return tape.expand(depth=5, stop_at=stop_at)


def spectrum(qnode):
    r"""Compute the frequency spectrum of a qnode that returns the expectation value of
    a single quantum observable.

    The circuit represented by the tape can be interpreted as a function
    :math:`f: \mathbb{R}^N \rightarrow \mathbb{R}`. This function can always be
    expressed by a Fourier series

    .. math::

        \sum \limits_{n_1\in \Omega_1} \dots \sum \limits_{n_N \in \Omega_N}
        c_{n_1,\dots, n_N} e^{-i x_1 n_1} \dots e^{-i x_N n_N}

    summing over the *frequency spectra* :math:`\Omega_i \subseteq \mathbb{Z},`
    :math:`i=1,\dots,N`, where :math:`\mathbb{Z}` are the integers. Each
    spectrum has the property that :math:`0 \in \Omega_i`, and the spectrum is
    symmetric (for every :math:`n \in \Omega_i` we have that :math:`-n \in
    \Omega_i`).

    As shown in `Vidal and Theis (2019)
    <https://arxiv.org/abs/1901.11434>`_ and `Schuld, Sweke and Meyer (2020)
    <https://arxiv.org/abs/2008.08605>`_, the frequency spectra only
    depend on the eigenvalues of the generator :math:`G` of the gates
    :math:`e^{-ix_i G}` that the inputs enter into.  In many situations, the
    spectra are limited to a few frequencies only, which in turn limits the
    function class that the circuit can express.

    This function extracts the frequency spectra for all inputs and a circuit
    represented by a :class:`~.pennylane.QNode`. To mark quantum circuit arguments as
    inputs, they have to be differentiable.

    .. note::
        Instead of using differentiability to compute
        gradients, here it is used to conveniently track inputs through classical pre-processing and
        circuit decompositions steps.

    Args:
        qnode (pennylane.QNode): a quantum node that has been called with some inputs

    Returns:
        (Dict[pennylane.numpy.tensor, list[float]]): Dictionary of scalar inputs as keys, and
        their frequency spectra as values.

    **Example**

    .. code-block:: python

        import pennylane as qml
        from pennylane import numpy as anp
        from pennylane.beta.transforms import fourier

        x = anp.array([0.1, 0.3], requires_grad=True)
        z = anp.array([0.5, 0.2])

        dev = qml.device('default.qubit', wires=['a'])

        @qml.qnode(dev)
        def circuit(x, z):
            qml.RX(inpt[0], wires='a')
            qml.Rot(-4.1, inpt[1], 1.3, wires='a')
            qml.RX(inpt[0], wires='a')
            qml.Hadamard(wires='a')
            qml.RX(weights[0], wires='a')
            qml.T(wires='a')
            return qml.expval(qml.PauliZ(wires='a'))

        circuit(x, z) # circuit needs to be executed at least once

        frequencies = fourier.spectrum(circuit)

        for inp, freqs in frequencies.items():
            print(f"{inp}: {freqs}")

    """
    try:
        tape = qnode.qtape
    except:
        raise ValueError("Cannot extract circuit tape from qnode. Please make sure the qnode "
                         "has been executed once.")

    original_params = tape.get_parameters(trainable_only=True)

    try:
        simplify_tape(tape, original_params)
    except ValueError:
        raise ValueError("Circuit does not allow for spectrum extraction.")
