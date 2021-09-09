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
"""Contains a transform that computes the frequency spectrum of a quantum
circuit."""
from itertools import product, combinations
from functools import wraps
import warnings
import numpy as np
import pennylane as qml


def _get_spectrum(op, decimals=8):
    r"""Extract the frequencies contributed by an input-encoding gate to the
    overall Fourier representation of a quantum circuit.

    If :math:`G` is the generator of the input-encoding gate :math:`\exp(-i x G)`,
    the frequencies are the differences between any two of :math:`G`'s eigenvalues.
    We will only compute non-negative frequencies in this subroutine.

    Args:
        op (~pennylane.operation.Operation): an instance of the `Operation` class
        decimals (int): Number of decimal places to round the frequencies to

    Returns:
        set[float]: non-negative frequencies contributed by this input-encoding gate
    """
    no_generator = False
    if hasattr(op, "generator"):
        g, coeff = op.generator

        if isinstance(g, np.ndarray):
            matrix = g
        elif hasattr(g, "matrix"):
            matrix = g.matrix
        else:
            no_generator = True
    else:
        no_generator = True

    if no_generator:
        raise ValueError(f"Generator of operation {op} is not defined.")

    matrix = coeff * matrix
    # todo: use qml.math.linalg once it is tested properly
    evals = np.linalg.eigvalsh(matrix)

    # compute all unique positive differences of eigenvalues, and add 0
    _spectrum = set(
        np.round(np.abs([x[1] - x[0] for x in combinations(evals, 2)]), decimals=decimals)
    )
    _spectrum |= {0}

    return _spectrum


def _join_spectra(spec1, spec2):
    r"""Join two sets of frequencies that belong to the same input.

    Since :math:`\exp(i a x)\exp(i b x) = \exp(i (a+b) x)`, frequency sets of two gates
    encoding the same :math:`x` are joined by computing the set of sums and absolute
    values of differences of their elements.
    We will only compute non-negative frequencies in this subroutine and assume the inputs
    to be non-negative frequencies as well.

    Args:
        spec1 (set[float]): first spectrum
        spec2 (set[float]): second spectrum
    Returns:
        set[float]: joined spectrum
    """
    if spec1 == {0}:
        return spec2
    if spec2 == {0}:
        return spec1

    sums = {s1 + s2 for s1 in spec1 for s2 in spec2}
    diffs = {np.abs(s1 - s2) for s1 in spec1 for s2 in spec2}

    return sums.union(diffs)


def _get_and_validate_classical_jacobians(qnode, argnums, args, kwargs):
    r"""Check classical preprocessing to be linear and return its Jacobian.

    The output of the `~pennylane.QNode` is only a Fourier series in the encoded :math:`x_i`
    if the processing of the QNode parameters into the gate parameters is linear.
    This method asserts this by computing the Jacobian of the processing at four positions
    and checking that it is constant.

    Args:
        qnode (pennylane.QNode): a quantum node of which to validate the preprocessing
        argnums (list[int]): the indices of the arguments with respect to which the Jacobian
            is computed; passed to `~pennylane.transforms.classical_jacobian`
        args (tuple): QNode arguments; the input parameters are one of four positions at which
            the Jacobian is computed, and the QNode arguments are left at these values
        kwargs (dict): QNode keyword arguments

    Returns:
        (tuple[array]): Jacobian of the classical preprocessing (at QNode arguments args).
    """
    if qnode.interface == "tf":
        import tensorflow as like_module
    elif qnode.interface == "torch":
        import torch as like_module
    else:
        like_module = np
    try:
        # Evaluate the classical Jacobian at (shape-adapted) 0, 1, 0.315, and the input args.
        zeros_args = (like_module.zeros_like(arg) for arg in args)
        ones_args = (like_module.ones_like(arg) for arg in args)
        frac_args = (like_module.ones_like(arg) * 0.315 for arg in args)
        jacs = [
            qml.transforms.classical_jacobian(qnode, argnums=argnums)(*_args, **kwargs)
            for _args in [zeros_args, ones_args, frac_args, args]
        ]
    except Exception as e:
        raise ValueError("Unable to compute Jacobian of the classical preprocessing.") from e

    # Check that the Jacobian is constant
    if not all(
        (all((np.allclose(jacs[0][i], jac[i]) for jac in jacs[1:])) for i in range(len(jacs[0])))
    ):
        raise ValueError(
            "The Jacobian of the classical preprocessing in the provided QNode is not constant; "
            "only linear classical preprocessing is supported."
        )

    # Note that jacs is a list of tuples of arrays
    return jacs[0]


def spectrum(qnode, encoding_args=None, encoding_gates=None, decimals=5):
    r"""Compute the frequency spectrum of the Fourier representation of quantum circuits.

    The circuit must only use single-parameter gates of the form :math:`e^{-i x_j G}` as
    input-encoding gates, which allows the computation of the spectrum by inspecting the gates'
    generators :math:`G`. The most important example of such gates are Pauli rotations.

    .. note::

        More precisely, the spectrum function relies on the gate to define a ``generator``,
        and will fail if gates marked as inputs do not have this attribute.

    The argument ``encoding_args`` controls which QNode arguments are considered as encoded
    inputs and the spectrum is computed only for those arguments.
    The input-encoding *gates* are those that are controlled by input-encoding QNode arguments.
    If no ``encoding_args`` are given, all QNode arguments are considered to be input-encoding
    arguments.

    .. note::

        Arguments or parameters in an argument that do not contribute to the Fourier series
        of the QNode with a frequency are considered as contributing with a constant term.
        That is, a parameter that does not control any gate has the spectrum ``[0]``.

    The returned spectrum is a dictionary with the following structure:

      #. The *keys* are the QNode argument indices in ``encoding_args``
      #. For scalar QNode arguments, the value is the spectrum of that parameter
      #. For iterable QNode arguments, the value is another dictionary whose
         #. keys are tuples that index the parameters within the iterable
         #. values are the spectra of the respective parameter
      #. If there is only a single QNode argument in the spectrum, the outer dictionary is
         unpacked, i.e, the only QNode argument index is skipped and only its value is returned.

    .. warning::

        The argument ``encoding_gates`` is deprecated and will be removed in a future version.
        In the current version, an attempt will be made to interpret the gate ``id``s as
        indices for QNode arguments, hence interpreting them like the argument ``encoding_args``.

    Args:
        qnode (pennylane.QNode): a quantum node representing a circuit in which
            input-encoding gates are marked by their ``id`` attribute
        encoding_args (list[int]): list of QNode argument indices, describing for which QNode
            arguments the spectrum is computed
        encoding_gates (list[str]): list of input-encoding gate ``id`` strings
            for which to compute the frequency spectra
        decimals (int): number of decimals to which to round frequencies.

        .. deprecated:: 0.18
            Use ``encoding_args`` instead.

    Returns:
        (dict or list[float]): Dictionary with the spectra. Typically -- but not always --
            the keys are the QNode argument indices and the values are frequency lists or
            dictionaries. In the latter case, keys are index tuples and values are the
            frequency spectra. See the description for details.


    **Details**

    A circuit that returns an expectation value of a Hermitian observable which depends on
    :math:`N` scalar inputs :math:`x_j` can be interpreted as a function
    :math:`f: \mathbb{R}^N \rightarrow \mathbb{R}` (as the observable is Hermitian,
    the expectation value is real-valued).
    This function can always be expressed by a Fourier-type sum

    .. math::

        \sum \limits_{\omega_1\in \Omega_1} \dots \sum \limits_{\omega_N \in \Omega_N}
        c_{\omega_1,\dots, \omega_N} e^{-i x_1 \omega_1} \dots e^{-i x_N \omega_N}

    over the *frequency spectra* :math:`\Omega_j \subseteq \mathbb{R},`
    :math:`j=1,\dots,N`. Each spectrum has the property that
    :math:`0 \in \Omega_j`, and the spectrum is symmetric
    (i.e., for every :math:`\omega \in \Omega_j` we have that :math:`-\omega \in\Omega_j`).
    If all frequencies are integer-valued, the Fourier sum becomes a *Fourier series*.

    As shown in `Vidal and Theis (2019) <https://arxiv.org/abs/1901.11434>`_ and
    `Schuld, Sweke and Meyer (2020) <https://arxiv.org/abs/2008.08605>`_,
    if an input :math:`x_j, j = 1 \dots N`,
    only enters into single-parameter gates of the form :math:`e^{-i x_j G}`
    (where :math:`G` is a Hermitian generator),
    the frequency spectrum :math:`\Omega_j` is fully determined by the eigenvalues
    of the generators :math:`G`. In many situations, the spectra are limited
    to a few frequencies only, which in turn limits the function class that the circuit
    can express.

    The ``spectrum`` function computes all frequencies that will potentially appear in the
    sets :math:`\Omega_1` to :math:`\Omega_N`.

    .. note::

        In more detail, the ``spectrum`` function also allows for preprocessing of the
        QNode arguments before they are fed into the gates, as long as this processing
        is *linear*. In particular, constant prefactors of the encoding arguments are
        allowed.

    **Example**

    Consider the following example, which uses non-trainable inputs ``x``, ``y`` and ``z``
    as well as trainable parameters ``w`` as arguments to the QNode.

    .. code-block:: python

        import pennylane as qml
        import numpy as np
        from spectrum import spectrum

        n_qubits = 3
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x, y, z, w):
            for i in range(n_qubits):
                qml.RX(0.5*x[i], wires=i)
                qml.Rot(w[0,i,0], w[0,i,1], w[0,i,2], wires=i)
                qml.RY(2.3*y[i], wires=i)
                qml.Rot(w[1,i,0], w[1,i,1], w[1,i,2], wires=i)
                qml.RX(z, wires=i)
            return qml.expval(qml.PauliZ(wires=0))

        x = np.array([1., 2., 3.])
        y = np.array([0.1, 0.3, 0.5])
        z = -1.8
        w = np.random.random((2, n_qubits, 3))
        res = spectrum(circuit, encoding_args=[0, 1, 2])(x, y, z, w)

    >>> print(qml.draw(circuit)(x, y, z, w))
    0: ──RX(0.5)──Rot(0.598, 0.949, 0.346)───RY(0.23)──Rot(0.693, 0.0738, 0.246)──RX(-1.8)──┤ ⟨Z⟩
    1: ──RX(1)────Rot(0.0711, 0.701, 0.445)──RY(0.69)──Rot(0.32, 0.0482, 0.437)───RX(-1.8)──┤
    2: ──RX(1.5)──Rot(0.401, 0.0795, 0.731)──RY(1.15)──Rot(0.756, 0.38, 0.38)─────RX(-1.8)──┤

    >>> for inp, freqs in res.items():
    >>>     print(f"{inp}: {freqs}")
    0: {(0,): [-0.5, 0.0, 0.5], (1,): [-0.5, 0.0, 0.5], (2,): [-0.5, 0.0, 0.5]}
    1: {(0,): [-2.3, 0.0, 2.3], (1,): [-2.3, 0.0, 2.3], (2,): [-2.3, 0.0, 2.3]}
    2: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

    .. note::
        While the Fourier spectrum usually does not depend
        on trainable circuit parameters or the actual values of the inputs,
        it may still change based on inputs to the QNode that alter the architecture
        of the circuit.

    Above, we selected all input-encoding parameters for the spectrum computation, using
    the ``encoding_args`` keyword argument. Note that the spectrum of the scalar argument
    ``z`` was unpacked from a single-entry dictionary to the spectrum in that entry.
    We may also restrict the full analysis to a single QNode argument, again using
    ``encoding_args``:

    >>> res = spectrum(circuit, encoding_args=[0])(x, y, z, w)
    >>> for inp, freqs in res.items():
    >>>     print(f"{inp}: {freqs}")
    (0,): [-0.5, 0.0, 0.5]
    (1,): [-0.5, 0.0, 0.5]
    (2,): [-0.5, 0.0, 0.5]

    As you can see, the output takes a slightly different form if we specify a single
    input-encoding QNode argument: The outer dictionary is unpacked and we simply obtain
    the inner dictionary, which previously was the value stored in ``res[0]``.
    As you may have guessed, if we request the spectrum for a single scalar variable, it
    comes in an even simpler structure:

    >>> res = spectrum(circuit, encoding_args=[2])(x, y, z, w)
    >>> print(res)
    [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

    >>> res = spectrum(circuit, encoding_args=[2])(x, y, z, w)
    >>> for inp, freqs in res.items():
    >>>     print(f"{inp}: {freqs}")

    .. warning::
        The ``spectrum`` function does not check if the result of the
        circuit is an expectation.
        It checks whether the classical preprocessing between
        QNode and gate arguments is linear by computing the Jacobian of the processing
        at four points. This makes it very unlikely -- *but not impossible* -- that
        non-linear functions go undetected.
        Furthermore, the QNode arguments *not* marked in ``encoding_args`` will not be
        considered in this analysis and if they resemble encoded inputs, the entire
        spectrum might be incorrect or the expectation value might not even admit one.


    The ``spectrum`` function works in all interfaces:

    .. code-block:: python

        import tensorflow as tf

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface='tf')
        def circuit(x):
            qml.RX(0.4*x[0], wires=0)
            qml.PhaseShift(x[1]*np.pi, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        x = tf.constant([1., 2.])
        res = spectrum(circuit)(x)

    >>> for inp, freqs in res.items():
    >>>     print(f"{inp}: {freqs}")
    (0,): [-0.4000000059604645, 0.0, 0.4000000059604645]
    (1,): [-3.141590118408203, 0.0, 3.141590118408203]
    """

    if np.isscalar(encoding_args):
        encoding_args = [encoding_args]

    if encoding_gates is not None:
        if encoding_args is not None:
            warnings.warn(
                "The argument encoding_gates is no longer valid and will be removed in"
                f" future versions. Ignoring encoding_gates={encoding_gates}..."
            )
        else:
            warnings.warn(
                "The argument encoding_gates is no longer valid and will be removed in"
                f" future versions. Trying to call spectrum with encoding_args={encoding_gates}..."
            )
            try:
                encoding_args = list(set(map(int, encoding_gates)))
            except ValueError as e:
                failing_id = " ".join(str(e).split(" ")[7:])
                raise ValueError(
                    "The provided encoding_gates could not be used as encoding_args."
                    f" Conversion to integers failed on {failing_id}."
                )

    atol = 10 ** (-decimals) if decimals is not None else 1e-10

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        nonlocal encoding_args
        # If no encoding_args are given, all QNode arguments are considered
        if encoding_args is None:
            encoding_args = list(range(len(args)))
        # Compute classical Jacobian and assert preprocessing is linear
        class_jacs = _get_and_validate_classical_jacobians(qnode, encoding_args, args, kwargs)
        # A map between Jacobians (contiguous) and arg indices (may be discontiguous)
        arg_idx_map = {i: arg_idx for i, arg_idx in enumerate(encoding_args)}
        # Initialize spectra for all requested parameters
        spectra = {arg_idx: {} for arg_idx in encoding_args}

        tape = qnode.qtape

        for jac_idx, class_jac in enumerate(class_jacs):
            _spectra = {
                arr_idx: {0} for arr_idx in product(*(range(sh) for sh in class_jac.shape[1:]))
            }
            unpack_inner_dict = len(class_jac.shape) == 1
            for op_idx, jac_of_op in enumerate(np.round(class_jac, decimals=decimals)):
                # Find the operation that belongs to the current op_idx
                op = tape._par_info[op_idx]["op"]
                # Find parameters feeding into the operation and if there are none, continue
                arr_ids = np.where(jac_of_op)
                if len(arr_ids[0]) == 0:
                    continue
                arr_ids = tuple((tuple(map(int, _ids)) for _ids in arr_ids))
                # Multi-parameter gates are not supported
                if len(op.parameters) != 1:
                    raise ValueError(
                        "Can only consider one-parameter gates as data-encoding gates; "
                        f"got {op.name}."
                    )
                # Get the spectrum of the current operation
                spec = _get_spectrum(op, decimals=decimals)
                if unpack_inner_dict:
                    # The curreny argument is a scalar
                    scale = float(jac_of_op)
                    # Rescale the operation spectrum
                    scaled_spec = [scale * f for f in spec]
                    # Join the new spectrum with the previously known spectrum for the parameter
                    _spectra[tuple()] = _join_spectra(_spectra[tuple()], scaled_spec)
                else:
                    # The curreny argument is an iterable
                    for arr_idx in zip(*arr_ids):
                        scale = float(jac_of_op[arr_idx])
                        # Rescale the operation spectrum
                        scaled_spec = [scale * f for f in spec]
                        # Join the new spectrum with the previously known spectrum for the parameter
                        _spectra[arr_idx] = _join_spectra(_spectra[arr_idx], scaled_spec)

            # Construct the full spectrum also containing negative frequencies and sort them
            _full_spectra = {}
            for arr_idx, spec in _spectra.items():
                spec = sorted(spec)
                _full_spectra[arr_idx] = [-freq for freq in spec[:0:-1]] + spec

            # Unpack the inner dictionary if the argument is a scalar
            if unpack_inner_dict:
                _full_spectra = _full_spectra[tuple()]

            spectra[arg_idx_map[jac_idx]] = _full_spectra

        # Unpack the outer dictionary if there only is one entry
        if len(encoding_args) == 1:
            spectra = spectra[encoding_args[0]]

        return spectra

    return wrapper
