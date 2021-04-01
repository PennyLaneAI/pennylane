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
import numpy as np
import pennylane as qml
from functools import wraps


def _simplify_tape(tape, original_inputs):

    def stop_at(obj):
        """Accepts a queue object and returns ``True``
           if this object should *not* be expanded."""

        if isinstance(obj, qml.operation.Operation):

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

        return False

    return tape.expand(depth=5, stop_at=stop_at)


def _get_spectrum(op):

    g, coeff = op.generator

    if isinstance(g, qml.Operation):
        g = g.matrix

    if g is None:
        raise ValueError(f"no generator defined for operator {op}")

    evals = np.linalg.evals(g)
    sums = [evals[i]+evals[j] for i in range(len(evals)) for j in range(len(evals)) if i <= j]
    return sorted(list(set(sums)))


def _join_spectra(spec1, spec2):
    sums = [s1+s2 for s1 in spec1 for s2 in spec2]
    return sorted(list(set(sums)))


def spectrum(qnode):
    r"""Create a function that computes the frequency spectra of the qnode
    with respect to all differentiable inputs.

    If the circuit represented by the qnode returns the expectation value of an
    observable, it can be interpreted as a function
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
    <https://arxiv.org/abs/2008.08605>`_, if an input :math:`x_j, j = 1 \dots N`
    only enters into single-parameter gates of the form :math:`e^{-i x_j G}`, the
    frequency spectrum :math:`\Omega_j` is fully determined by the eigenvalues
    of the generators :math:`G`. In many situations, the spectra are limited
    to a few frequencies only, which in turn limits the function class that the circuit
    can express.

    Arguments to the qnode are marked as inputs :math:`x_1, \dots, x_N` by making them
    differentiable.

    .. note::
        Differentiability of the inputs is not used here to compute
        gradients, but to conveniently track inputs through classical
        pre-processing and circuit compilation procedures.

    Args:
        qnode (pennylane.QNode): a quantum node that has been called with some inputs

    Returns:
        (Dict[pennylane.numpy.tensor, list[float]]): Dictionary of scalar inputs as keys, and
        their frequency spectra as values.

    **Example**

    .. code-block:: python

        import pennylane as qml
        from pennylane import numpy as pnp
        from pennylane.beta.transforms import fourier

        x = anp.array([0.1, 0.3], requires_grad=True)
        z = anp.array([0.5, 0.2])

        dev = qml.device('default.qubit', wires=['a'])

        @qml.qnode(dev)
        def circuit(x, z):
            qml.RX(x[0], wires='a')
            qml.Rot(-4.1, x[1], z[0], wires='a')
            qml.RX(x[2], wires='a')
            qml.Hadamard(wires='a')
            qml.RX(z[1], wires='a')
            qml.T(wires='a')
            return qml.expval(qml.PauliZ(wires='a'))

        x = pnp.array([0.1, 0.2, 0.3])
        z = pnp.array([-0.1, 1.8])

        frequencies = fourier.spectrum(circuit)(x, z)

        for inp, freqs in frequencies.items():
            print(f"{inp}: {freqs}")

    """

    @wraps(qnode)
    def wrapper(*args, **kwargs):

        # extract the tape
        qnode.construct(args, kwargs)
        tape = qnode.qtape

        inpts = tape.get_parameters(trainable_only=True)

        try:
            simple_tape = _simplify_tape(tape, inpts)
        except ValueError:
            raise ValueError("Circuit does not allow for spectrum extraction.")

        freqs = {}
        for op in simple_tape.operations:

            # make sure this is a one-parameter operation
            assert len(op.parameters) == 1

            inpt = op.parameters[0]
            if inpt in inpts:
                spec = _get_spectrum(op)

                if inpt in freqs:
                    spec = _join_spectra(freqs[inpt], spec)

                freqs[inpt] = spec

    return wrapper
