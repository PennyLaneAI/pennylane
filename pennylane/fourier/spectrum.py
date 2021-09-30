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
from collections import OrderedDict
from inspect import signature
import numpy as np
import pennylane as qml


def _get_random_args(args, interface, num, seed):
    r"""Generate random arguments of the same shapes as provided args.
    Args:
        args (tuple): Original input arguments
        interface (str): Interface of the QNode into which the arguments will be fed
        num (int): Number of random argument sets to generate
    Returns:
        list[tuple]: List of length ``num`` with each entry being a random instance
        of arguments like ``args``.
    """
    if interface == "tf":
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        tf.random.set_seed(seed)
        rnd_args = []
        for _ in range(num):
            _args = (tf.random.uniform(tf.shape(_arg)) * 2 * np.pi - np.pi for _arg in args)
            _args = tuple(
                tf.Variable(_arg) if isinstance(arg, tf.Variable) else _arg
                for _arg, arg in zip(_args, args)
            )
            rnd_args.append(_args)
    elif interface == "torch":
        import torch  # pylint: disable=import-outside-toplevel

        torch.random.manual_seed(seed)
        rnd_args = [
            tuple(torch.rand(np.shape(arg)) * 2 * np.pi - np.pi for arg in args) for _ in range(num)
        ]
    else:
        np.random.seed(seed)
        rnd_args = [
            tuple(np.random.random(np.shape(arg)) * 2 * np.pi - np.pi for arg in args)
            for _ in range(num)
        ]

    return rnd_args

def _get_spectrum(op, decimals=8):
    r"""Extract the frequencies contributed by an input-encoding gate to the
    overall Fourier representation of a quantum circuit.

    If :math:`G` is the generator of the input-encoding gate :math:`\exp(-i x G)`,
    the frequencies are the differences between any two of :math:`G`'s eigenvalues.
    We only compute non-negative frequencies in this subroutine.

    Args:
        op (~pennylane.operation.Operation): :class:`~.pennylane.Operation` to extract
            the frequencies for
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

    # compute all unique positive differences of eigenvalues, then add 0
    _spectrum = set(
        np.round(np.abs([x[1] - x[0] for x in combinations(evals, 2)]), decimals=decimals)
    )
    _spectrum |= {0}

    return _spectrum


def _join_spectra(spec1, spec2):
    r"""Join two sets of frequencies that belong to the same input.

    Since :math:`\exp(i a x)\exp(i b x) = \exp(i (a+b) x)`, the spectra of two gates
    encoding the same :math:`x` are joined by computing the set of sums and absolute
    values of differences of their elements.
    We only compute non-negative frequencies in this subroutine and assume the inputs
    to be non-negative frequencies as well.

    Args:
        spec1 (set[float]): first spectrum
        spec2 (set[float]): second spectrum
    Returns:
        set[float]: joined spectrum
    """
    if spec1 in ({0}, {}):
        return spec2
    if spec2 in ({0}, {}):
        return spec1

    sums = {s1 + s2 for s1 in spec1 for s2 in spec2}
    diffs = {np.abs(s1 - s2) for s1 in spec1 for s2 in spec2}

    return sums.union(diffs)


def _get_and_validate_classical_jacobian(qnode, argnum, args, kwargs, num_pos):
    r"""Check classical preprocessing of a QNode to be linear and return its Jacobian.

    Args:
        qnode (pennylane.QNode): a quantum node of which to validate the preprocessing
        argnum (list[int]): the indices of the arguments with respect to which the Jacobian
            is computed; passed to `~pennylane.transforms.classical_jacobian`
        args (tuple): QNode arguments; the input parameters are one of four positions at which
            the Jacobian is computed, and the QNode arguments are left at these values
        kwargs (dict): QNode keyword arguments
        num_pos (int): Number of additional random positions at which to evaluate the
            Jacobian and test that it is constant

    Returns:
        (tuple[array]): Jacobian of the classical preprocessing (at QNode arguments args).

    The output of the `~pennylane.QNode` is only a Fourier series in the encoded :math:`x_i`
    if the processing of the QNode parameters into gate parameters is linear.
    This method asserts this linearity by computing the Jacobian of the processing at
    multiple positions and checking that it is constant.
    """
    try:
        # Get random input arguments
        all_args = _get_random_args(args, qnode.interface, num_pos, seed=291)
        all_args.append(args)
        # Evaluate the classical Jacobian at multiple input args.
        jac_fns = tuple(qml.transforms.classical_jacobian(qnode, argnum=num) for num in argnum)
        jacs = [tuple(_fn(*_args, **kwargs) for _fn in jac_fns) for _args in all_args]
    except Exception as e:
        raise ValueError("Could not compute Jacobian of the classical preprocessing.") from e

    # Check that the Jacobian is constant
    if not all(
        all(np.allclose(jacs[0][i], jac[i], atol=1e-6, rtol=0) for jac in jacs[1:])
        for i in range(len(jacs[0]))
    ):
        raise ValueError(
            "The Jacobian of the classical preprocessing in the provided QNode is not constant; "
            "only linear classical preprocessing is supported."
        )

    # Note that jacs is a list of tuples of arrays
    return jacs[0]


def _process_ids(encoding_args, argnum, qnode):
    r"""Process the passed ``encoding_args`` and ``argnum`` or infer them from
    the QNode signature.

    Args:
        encoding_args (dict[str, list[tuple]] or set): Parameter index dictionary;
            keys are argument names, values are index tuples for that argument
            or an ``Ellipsis``. If a ``set``, all values are set to ``Ellipsis``
        argnum (list[int]): Numerical indices for arguments
        qnode (QNode): QNode to infer the ``encoding_args`` and ``argnum`` from
            if both are ``None``
    Returns:
        OrderedDict[str, list[tuple]]: Ordered parameter index dictionary;
            keys are argument names, values are index tuples for that argument
            or an ``Ellipsis``
        list[int]: Numerical indices for arguments

    If ``encoding_args`` are passed, they take precedence over ``argnum``.
    Passing a set with ``keys`` is an alias for ``{key: ... for key in keys}``.
    If both, ``encoding_args`` and ``argnum`` are ``None``, all QNode arguments
    of ``qnode`` that do not have a default value defined are included.
    Of these arguments, all elements are included in ``encoding_args``
    """
    sig_pars = signature(qnode.func).parameters
    arg_names = list(sig_pars.keys())
    arg_names_no_def = [name for name, par in sig_pars.items() if par.default is par.empty]

    if encoding_args is None:
        if argnum is None:
            encoding_args = OrderedDict((name, ...) for name in arg_names_no_def)
            argnum = list(range(len(arg_names_no_def)))
        elif np.isscalar(argnum):
            encoding_args = OrderedDict({arg_names[argnum]: ...})
            argnum = [argnum]
        else:
            encoding_args = OrderedDict((arg_names[num], ...) for num in argnum)
    else:
        requested_names = set(encoding_args)
        if not all(name in arg_names for name in requested_names):
            raise ValueError(
                f"Not all names in {requested_names} are known. " f"Known arguments: {arg_names}"
            )
        # Selection of requested argument names from sorted names
        if isinstance(encoding_args, set):
            encoding_args = OrderedDict(
                (name, ...) for name in arg_names if name in requested_names
            )
        else:
            encoding_args = OrderedDict(
                (name, encoding_args[name]) for name in arg_names if name in requested_names
            )
        argnum = [arg_names.index(name) for name in encoding_args]

    return encoding_args, argnum


def spectrum(qnode, encoding_args=None, argnum=None, decimals=5, validation_kwargs=None):
    r"""Compute the frequency spectrum of the Fourier representation of quantum circuits.

    The circuit must only use single-parameter gates of the form :math:`e^{-i x_j G}` as
    input-encoding gates, which allows the computation of the spectrum by inspecting the gates'
    generators :math:`G`. The most important example of such gates are Pauli rotations.

    .. note::

        More precisely, the spectrum function relies on the gate to define a ``generator``,
        and will fail if gates marked controlled by marked parameters do not have this attribute.

    The argument ``argnum`` controls which QNode arguments are considered as encoded
    inputs and the spectrum is computed only for these arguments.
    The input-encoding *gates* are those that are controlled by input-encoding QNode arguments.
    If no ``argnum`` are given, all QNode arguments are considered to be input-encoding
    arguments.

    .. note::

        Arguments or parameters in an argument that do not contribute to the Fourier series
        of the QNode with a frequency are considered as contributing with a constant term.
        That is, a parameter that does not control any gate has the spectrum ``[0]``.

    The returned spectrum is an ``~.pennylane.argmap.ArgMap`` with the frequency spectra as
    values. See the
    `ArgMap documentation <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.argmap.ArgMap.html>`_
    for details on its structure.

    Args:
        qnode (pennylane.QNode): :class:`~.pennylane.QNode` to compute the spectrum for
        encoding_args (dict[str, list[tuple]], set): Parameter index dictionary;
            keys are argument names, values are index tuples for that argument
            or an ``Ellipsis``. If a ``set``, all values are set to ``Ellipsis``.
            The contained argument and parameter indices indicate the scalar variables
            for which the spectrum is computed
        argnum (list[int]): Numerical indices for arguments with respect to which
            to compute the spectrum
        decimals (int): number of decimals to which to round frequencies.
        num_pos (int): Number of additional random positions at which to evaluate the
            Jacobian of the preprocessing and test that it is constant.
            Setting ``num_pos=0`` will deactivate the test.

    Returns:
        function: Function which accepts the same arguments as the QNode.
            When called, this function will return a dictionary of dictionaries
            containing the frequency spectra per QNode parameter.

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
        is *linear*. In particular, constant prefactors for the encoding arguments are
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
        res = spectrum(circuit, argnum=[0, 1, 2])(x, y, z, w)

    This circuit looks as follows:

    >>> print(qml.draw(circuit)(x, y, z, w))
    0: ──RX(0.5)──Rot(0.598, 0.949, 0.346)───RY(0.23)──Rot(0.693, 0.0738, 0.246)──RX(-1.8)──┤ ⟨Z⟩
    1: ──RX(1)────Rot(0.0711, 0.701, 0.445)──RY(0.69)──Rot(0.32, 0.0482, 0.437)───RX(-1.8)──┤
    2: ──RX(1.5)──Rot(0.401, 0.0795, 0.731)──RY(1.15)──Rot(0.756, 0.38, 0.38)─────RX(-1.8)──┤

    Applying the ``spectrum`` function to the circuit for the non-trainable parameters, we obtain:

    >>> for inp, freqs in res.items():
    >>>     print(f"{inp}: {freqs}")
    "x": {(0,): [-0.5, 0.0, 0.5], (1,): [-0.5, 0.0, 0.5], (2,): [-0.5, 0.0, 0.5]}
    "y": {(0,): [-2.3, 0.0, 2.3], (1,): [-2.3, 0.0, 2.3], (2,): [-2.3, 0.0, 2.3]}
    "z": {(): [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]}

    .. note::
        While the Fourier spectrum usually does not depend
        on trainable circuit parameters or the actual values of the inputs,
        it may still change based on inputs to the QNode that alter the architecture
        of the circuit.

    Above, we selected all input-encoding parameters for the spectrum computation, using
    the ``argnum`` keyword argument. We may also restrict the full analysis to a single
    QNode argument, again using ``argnum``:

    >>> res = spectrum(circuit, argnum=[0])(x, y, z, w)
    >>> for inp, freqs in res.items():
    >>>     print(f"{inp}: {freqs}")
    "x": {(0,): [-0.5, 0.0, 0.5], (1,): [-0.5, 0.0, 0.5], (2,): [-0.5, 0.0, 0.5]}

    Selecting arguments by name instead of index is possible via the
    ``encoding_args`` argument:

    >>> res = spectrum(circuit, encoding_args={"y"})(x, y, z, w)
    >>> for inp, freqs in res.items():
    >>>     print(f"{inp}: {freqs}")
    "y": {(0,): [-2.3, 0.0, 2.3], (1,): [-2.3, 0.0, 2.3], (2,): [-2.3, 0.0, 2.3]}

    Note that for array-valued arguments the spectrum for each element of the array
    is computed. A more fine-grained control is available by passing index tuples
    for the respective argument name in ``encoding_args``:

    >>> encoding_args = {"y": [(0,),(2,)]}
    >>> res = spectrum(circuit, encoding_args=encoding_args)(x, y, z, w)
    >>> for inp, freqs in res.items():
    >>>     print(f"{inp}: {freqs}")
    "y": {(0,): [-2.3, 0.0, 2.3], (2,): [-2.3, 0.0, 2.3]}

    .. warning::
        The ``spectrum`` function does not check if the result of the
        circuit is an expectation value. It checks whether the classical preprocessing between
        QNode and gate arguments is linear by computing the Jacobian of the processing
        at multiple points. This makes it unlikely -- *but not impossible* -- that
        non-linear functions go undetected.
        The number of additional points at which the Jacobian is computed can be controlled
        via ``num_pos``, and the test is deactivated if ``num_pos=0`` (discouraged).
        Furthermore, the QNode arguments *not* marked in ``argnum`` will not be
        considered in this test and if they resemble encoded inputs, the entire
        spectrum might be incorrect or the circuit might not even admit one.

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

    >>> print(res)
    {"x": {(0,): [-0.4, 0.0, 0.4], (1,): [-3.14159, 0.0, 3.14159]}}

    """
    validation_kwargs = validation_kwargs or {}
    encoding_args, argnum = _process_ids(encoding_args, argnum, qnode)
    atol = 10 ** (-decimals) if decimals is not None else 1e-10
    # A map between Jacobians (contiguous) and arg names (may be discontiguous)
    arg_name_map = dict(enumerate(encoding_args))

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        # Compute classical Jacobian and assert preprocessing is linear
        jac_fn = qml.transforms.classical_jacobian(qnode, argnum=argnum)
        qml.math.is_independent(jac_fn, qnode.interface, args, kwargs, **validation_kwargs)
        class_jacs = jac_fn(*args, **kwargs)

        spectra = {}
        par_info = qnode.qtape._par_info  # pylint: disable=protected-access
        for jac_idx, class_jac in enumerate(class_jacs):
            arg_name = arg_name_map[jac_idx]
            if encoding_args[arg_name] is Ellipsis:
                requested_par_ids = set(product(*(range(sh) for sh in class_jac.shape[1:])))
            else:
                requested_par_ids = set(encoding_args[arg_name])
            _spectra = {par_idx: {0} for par_idx in requested_par_ids}

            for op_idx, jac_of_op in enumerate(np.round(class_jac, decimals=decimals)):
                op = par_info[op_idx]["op"]
                # Find parameters that where requested and feed into the operation
                if len(class_jac.shape) == 1:
                    # Scalar argument, only axis of Jacobian is for gates
                    if np.isclose(jac_of_op, 0.0, atol=atol, rtol=0):
                        continue
                    jac_of_op = {(): jac_of_op}
                    par_ids = {()}
                else:
                    par_ids = zip(*[map(int, _ids) for _ids in np.where(jac_of_op)])
                    par_ids = set(par_ids).intersection(requested_par_ids)
                    if len(par_ids) == 0:
                        continue
                # Multi-parameter gates are not supported
                if len(op.parameters) != 1:
                    raise ValueError(
                        "Can only consider one-parameter gates as data-encoding gates; "
                        f"got {op.name}."
                    )
                spec = _get_spectrum(op, decimals=decimals)

                for par_idx in par_ids:
                    scale = float(jac_of_op[par_idx])
                    scaled_spec = [scale * f for f in spec]
                    _spectra[par_idx] = _join_spectra(_spectra[par_idx], scaled_spec)

            # Construct the sorted spectrum also containing negative frequencies
            for idx, spec in _spectra.items():
                spec = sorted(spec)
                _spectra[idx] = [-freq for freq in spec[:0:-1]] + spec
            spectra[arg_name] = _spectra

        return spectra

    return wrapper
