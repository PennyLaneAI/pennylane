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
from itertools import product


def _simplify_tape(tape, original_inputs):
    """Expand the tape until every operation that an original input enters
    is a single-parameter gate of the form exp(-i x_i G) where G is a Hermitian generator.

    If the expansion involves any processing on the parameters, the circuit is not of a form that
    can be handled by the spectrum function, and an error is thrown.

    Args:
        tape (~.tape.QuantumTape): tape to simplify
        original_inputs (list): list of inputs, to check that they are conserved throughout expansion

    Returns:
        ~.tape.QuantumTape: expanded tape
    """

    def stop_at(obj):
        """Accepts a queue object and returns ``True``
           if this object should *not* be expanded."""

        if isinstance(obj, qml.operation.Operation):
            takes_input = any(p in original_inputs for p in obj.parameters)
            is_one_param_gate = obj.generator[0] is not None and len(obj.parameters) == 1
            if takes_input and not is_one_param_gate:

                # check if the trainable parameters after expansion are
                # all found in the original inputs
                # parameters - this makes sure no new parameters
                # were created that depend on the inputs
                try:
                    expanded = obj.expand()
                except NotImplementedError:
                    raise ValueError(f"Cannot expand {obj}. Aborting the expansion of the tape.")

                new_inputs = expanded.get_parameters(trainable_only=True)
                if any(i not in original_inputs for i in new_inputs):
                    raise ValueError(f"{obj} transforms the inputs. "
                                     f"Aborting the expansion of the tape.")

                return False

            else:

                return True

        # if object is not an op, it is probably a tape, so expand it
        return False

    return tape.expand(depth=5, stop_at=stop_at)


def _get_spectrum(op):

    g, coeff = op.generator

    if not isinstance(g, np.ndarray) and g is not None:
        g = g.matrix

    if g is None:
        raise ValueError(f"no generator defined for operator {op}")

    g = coeff*g
    evals = np.linalg.eigvals(g)
    # eigenvalues of hermitian ops are guaranteed to be real
    evals = np.real(evals)

    # append negative to cater for the complex conjugate part which subtracts eigenvalues
    evals = [evals, -evals]

    frequencies = [np.round(sum(comb), decimals=8) for comb in product(*evals)]
    unique_frequencies = list(set(frequencies))
    return sorted(unique_frequencies)


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

            if len(op.parameters) != 1:
                # inputs can only enter one-parameter gates
                continue
            else:
                inpt = op.parameters[0]
                if inpt in inpts:
                    spec = _get_spectrum(op)

                    if inpt in freqs:
                        spec = _join_spectra(freqs[inpt], spec)

                    freqs[inpt] = spec

        return freqs

    return wrapper
