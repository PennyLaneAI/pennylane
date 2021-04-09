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
"""Contains qnode transform that computes the fourier spectrum of a QNode."""
from functools import wraps
from itertools import product
from copy import copy
import numpy as np
import pennylane as qml


def _simplify_tape(tape, original_inputs):
    r"""Expand the tape until every operation that takes an original input
    is a single-parameter gate.

    A single-parameter gate is of the form :math:`\exp(-i x_i G)` where :math:`x_i` is a scalar input
    and :math:`G` is a Hermitian operator called a "generator".

    If the expansion involves any processing on the parameters (such as scalar multiplication)
    the circuit is not of a form that can be handled by the spectrum function, and an error is thrown.

    Args:
        tape (~.tape.QuantumTape): tape to simplify
        original_inputs (list): list of inputs, to check that they are conserved throughout expansion

    Returns:
        ~.tape.QuantumTape: expanded tape
    """

    def stop_at(obj):
        r"""Accepts a queue object and returns ``False`` if this object should be expanded.

        An object should be expanded if it is not an operation, and if it is not a single-parameter gate but
        takes inputs.

        Before returning ``False`` it is checked that an expansion is possible, and that it will not perform processing
        on the inputs.
        """

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
                    raise ValueError(
                        f"{obj} transforms the inputs. " f"Aborting the expansion of the tape."
                    )

                return False

            return True

        # if object is not an op, it is probably a tape, so expand it
        return False

    return tape.expand(depth=5, stop_at=stop_at)


def _get_spectrum(op):
    r"""Get the spectrum of the input x of a single-parameter operation :math:`\exp(-i x G)`.

    The spectrum is the set of sums of :math:`G`'s eigenvalues.
    """

    g, coeff = op.generator

    # some generators are operations
    if not isinstance(g, np.ndarray) and g is not None:
        g = g.matrix

    # the matrix or generator could be undefined
    if g is None:
        raise ValueError(f"no generator defined for operator {op}")

    g = coeff * g
    evals = np.linalg.eigvals(g)
    # eigenvalues of hermitian ops are guaranteed to be real
    evals = np.real(evals)

    # append negative to cater for the complex conjugate part which subtracts eigenvalues
    evals = [evals, -evals]

    frequencies = [np.round(sum(comb), decimals=8) for comb in product(*evals)]
    unique_frequencies = list(set(frequencies))
    return sorted(unique_frequencies)


def _join_spectra(spec1, spec2):
    r"""Join two spectra that belong to the same input.

    Since :math:`\exp(i a x)\exp(i b x) = \exp(i (a+b) x)`, spectra are
    joined by taking the set of sums of their elements.

    Args:
        spec1 (list[float]): first spectrum
        spec2 (list[float]): second spectrum
    Returns:
        list[float]: joined spectrum
    """
    sums = [s1 + s2 for s1 in spec1 for s2 in spec2]
    return sorted(list(set(sums)))


def spectrum(qnode):
    r"""Computes the frequency spectra of the qnode with respect to all
    differentiable inputs.

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
        gradients, but to track inputs through classical
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
        from pennylane.fourier import spectrum

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(x):
            qml.templates.AngleEmbedding(x[0:3], wires=[0, 1, 2])
            qml.RX(x[0], wires=1)
            qml.Rot(x[0], x[1], x[3], wires=1)
            qml.CNOT(wires=[1, 2])
            qml.RX(x[3], wires=2)
            return qml.expval(qml.PauliZ(wires=2))

        x = pnp.array([0.1, 0.2, 0.3, 0.4, 0.5], requires_grad=True)


        res = spectrum(circuit)(x)

        for inp, freqs in res.items():
            print(f"{inp}: {freqs}")

        >>> 0.1: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        >>> 0.2: [-2.0, -1.0, 0.0, 1.0, 2.0]
        >>> 0.3: [-1.0, 0.0, 1.0]
        >>> 0.4: [-2.0, -1.0, 0.0, 1.0, 2.0]
        >>> 0.5: [-1.0, 0.0, 1.0]

    """

    @wraps(qnode)
    def wrapper(*args, **kwargs):

        qnode_copy = copy(qnode)

        # hack: currently the tape only differentiates trainable/non-trainable params
        # if the qnode uses non-backprop diff rules.
        qnode_copy.diff_options["method"] = "parameter-shift"

        # extract the tape
        qnode_copy.construct(args, kwargs)
        tape = qnode_copy.qtape

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

            inpt = op.parameters[0]
            if inpt in inpts:
                spec = _get_spectrum(op)

                if inpt in freqs:
                    spec = _join_spectra(freqs[inpt], spec)

                freqs[inpt] = spec

        return freqs

    return wrapper
