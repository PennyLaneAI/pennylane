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
circuit including classical preprocessing within the QNode."""
from collections import OrderedDict
from functools import wraps
from inspect import signature
from itertools import product

import numpy as np

import pennylane as qml

from .utils import get_spectrum, join_spectra


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

    In ``qnode_spectrum`` both ``encoding_args`` and ``argnum`` are required.
    However, they can be inferred from one another and even from the QNode signature,
    which is done in this helper function, using the following rules/design choices:

      - If ``argnum`` is provided, the QNode arguments with the indices in ``argnum``
        are considered and added to ``encoding_args`` with an ``Ellipsis``, meaning
        that for array-valued arguments all parameters are considered in
        ``qnode_spectrum``.
      - If ``encoding_args`` is provided and is a dictionary, it is preserved
        up to arguments that do not appear in the QNode. Also, it is converted to
        an ``OrderedDict``, inferring the ordering from the QNode arguments.
        Passing a set with ``keys`` instead is an alias for
        ``{key: ... for key in keys}``.
        ``argnum`` will contain the indices of these arguments.
      - If both ``encoding_args`` and ``argnum`` are passed, ``encoding_args`` takes
        precedence over ``argnum``, in particular ``argnum`` is overwritten.
      - If neither is passed, all arguments of the passed QNode that do not have a
        default value defined are considered
        and their value is an ``Ellipsis``, so that all parameters of array-valued
        arguments will be considered in ``qnode_spectrum``.

    **Example**

    As an example, consider the qnode

    >>> @qml.qnode(dev)
    >>> def circuit(a, b, c, x=2):
    ...     return qml.expval(qml.X(0))

    which takes arguments:

    >>> a = np.array([2.4, 1.2, 3.1])
    >>> b = 0.2
    >>> c = np.arange(20, dtype=float).reshape((2, 5, 2))

    Then we may use the following inputs

    >>> encoding_args = {"a": [(1,), (2,)], "c": ..., "x": [()]}
    >>> argnum = [2, 0]

    in various combinations:

    >>> _process_ids(encoding_args, None, circuit)
    (OrderedDict([('a', [(1,), (2,)]), ('c', Ellipsis), ('x', [()])]), [0, 2, 3])

    The first output, ``encoding_args``, essentially is unchanged, it simply was ordered in
    the order of the QNode arguments. The second output, ``argnum``, contains all three
    argument indices because all of ``a``, ``b``, and ``c`` appear in ``encoding_args``.
    If we in addition pass ``argnum``, it is ignored:

    >>> _process_ids(encoding_args, argnum, circuit)
    (OrderedDict([('a', [(1,), (2,)]), ('c', Ellipsis), ('x', [()])]), [0, 2, 3])

    Only if we leave out ``encoding_args`` does it make a difference:

    >>> _process_ids(None, argnum, circuit)
    (OrderedDict([('a', Ellipsis), ('c', Ellipsis)]), [0, 2])

    Now only the arguments in ``argnum`` are considered, in particular the ``argnum`` input
    is simply sorted. In ``encoding_args``, all argument names are paired with an ``Ellipsis``.
    If we skip both inputs, all QNode arguments are extracted:

    >>> _process_ids(None, None, circuit)
    (OrderedDict([('a', Ellipsis), ('b', Ellipsis), ('c', Ellipsis)]), [0, 1, 2])

    Note that ``x`` does not appear here, because it has a default value defined and thus is
    considered a keyword argument.

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
            argnum = sorted(argnum)
            encoding_args = OrderedDict((arg_names[num], ...) for num in argnum)
    else:
        requested_names = set(encoding_args)
        if not all(name in arg_names for name in requested_names):
            raise ValueError(
                f"Not all names in {requested_names} are known. Known arguments: {arg_names}"
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


def qnode_spectrum(qnode, encoding_args=None, argnum=None, decimals=8, validation_kwargs=None):
    r"""Compute the frequency spectrum of the Fourier representation of quantum circuits,
    including classical preprocessing.

    The circuit must only use gates as input-encoding gates that can be decomposed
    into single-parameter gates of the form :math:`e^{-i x_j G}` , which allows the
    computation of the spectrum by inspecting the gates' generators :math:`G`.
    The most important example of such single-parameter gates are Pauli rotations.

    The argument ``argnum`` controls which QNode arguments are considered as encoded
    inputs and the spectrum is computed only for these arguments.
    The input-encoding *gates* are those that are controlled by input-encoding QNode arguments.
    If no ``argnum`` is given, all QNode arguments are considered to be input-encoding
    arguments.

    .. note::

        Arguments of the QNode or parameters within an array-valued QNode argument
        that do not contribute to the Fourier series of the QNode
        with any frequency are considered as contributing with a constant term.
        That is, a parameter that does not control any gate has the spectrum ``[0]``.

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
        validation_kwargs (dict): Keyword arguments passed to
            :func:`~.pennylane.math.is_independent` when testing for linearity of
            classical preprocessing in the QNode.

    Returns:
        function: Function which accepts the same arguments as the QNode.
        When called, this function will return a dictionary of dictionaries
        containing the frequency spectra per QNode parameter.

    **Details**

    A circuit that returns an expectation value of a Hermitian observable which depends on
    :math:`N` scalar inputs :math:`x_j` can be interpreted as a function
    :math:`f: \mathbb{R}^N \rightarrow \mathbb{R}`.
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

    The ``qnode_spectrum`` function computes all frequencies that will
    potentially appear in the sets :math:`\Omega_1` to :math:`\Omega_N`.

    .. note::

        The ``qnode_spectrum`` function also supports
        preprocessing of the QNode arguments before they are fed into the gates,
        as long as this processing is *linear*. In particular, constant
        prefactors for the encoding arguments are allowed.

    **Example**

    Consider the following example, which uses non-trainable inputs ``x``, ``y`` and ``z``
    as well as trainable parameters ``w`` as arguments to the QNode.

    .. code-block:: python

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
            return qml.expval(qml.Z(0))

    This circuit looks as follows:

    >>> x = np.array([1., 2., 3.])
    >>> y = np.array([0.1, 0.3, 0.5])
    >>> z = -1.8
    >>> w = np.random.random((2, n_qubits, 3))
    >>> print(qml.draw(circuit)(x, y, z, w))
    0: ──RX(0.50)──Rot(0.09,0.46,0.54)──RY(0.23)──Rot(0.59,0.22,0.05)──RX(-1.80)─┤  <Z>
    1: ──RX(1.00)──Rot(0.98,0.61,0.07)──RY(0.69)──Rot(0.62,0.00,0.28)──RX(-1.80)─┤
    2: ──RX(1.50)──Rot(0.65,0.07,0.36)──RY(1.15)──Rot(0.74,0.27,0.24)──RX(-1.80)─┤

    Applying the ``qnode_spectrum`` function to the circuit for
    the non-trainable parameters, we obtain:

    >>> res = qml.fourier.qnode_spectrum(circuit, argnum=[0, 1, 2])(x, y, z, w)
    >>> for inp, freqs in res.items():
    ...     print(f"{inp}: {freqs}")
    "x": {(0,): [-0.5, 0.0, 0.5], (1,): [-0.5, 0.0, 0.5], (2,): [-0.5, 0.0, 0.5]}
    "y": {(0,): [-2.3, 0.0, 2.3], (1,): [-2.3, 0.0, 2.3], (2,): [-2.3, 0.0, 2.3]}
    "z": {(): [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]}

    .. note::
        While the Fourier spectrum usually does not depend
        on trainable circuit parameters or the actual values of the inputs,
        it may still change based on inputs to the QNode that alter the architecture
        of the circuit.

    .. details::
        :title: Usage Details

        Above, we selected all input-encoding parameters for the spectrum computation, using
        the ``argnum`` keyword argument. We may also restrict the full analysis to a single
        QNode argument, again using ``argnum``:

        >>> res = qml.fourier.qnode_spectrum(circuit, argnum=[0])(x, y, z, w)
        >>> for inp, freqs in res.items():
        ...     print(f"{inp}: {freqs}")
        "x": {(0,): [-0.5, 0.0, 0.5], (1,): [-0.5, 0.0, 0.5], (2,): [-0.5, 0.0, 0.5]}

        Selecting arguments by name instead of index is possible via the
        ``encoding_args`` argument:

        >>> res = qml.fourier.qnode_spectrum(circuit, encoding_args={"y"})(x, y, z, w)
        >>> for inp, freqs in res.items():
        ...     print(f"{inp}: {freqs}")
        "y": {(0,): [-2.3, 0.0, 2.3], (1,): [-2.3, 0.0, 2.3], (2,): [-2.3, 0.0, 2.3]}

        Note that for array-valued arguments the spectrum for each element of the array
        is computed. A more fine-grained control is available by passing index tuples
        for the respective argument name in ``encoding_args``:

        >>> encoding_args = {"y": [(0,),(2,)]}
        >>> res = qml.fourier.qnode_spectrum(circuit, encoding_args=encoding_args)(x, y, z, w)
        >>> for inp, freqs in res.items():
        ...     print(f"{inp}: {freqs}")
        "y": {(0,): [-2.3, 0.0, 2.3], (2,): [-2.3, 0.0, 2.3]}

        .. warning::
            The ``qnode_spectrum`` function checks whether the classical preprocessing between
            QNode and gate arguments is linear by computing the Jacobian of the processing
            and applying :func:`~.pennylane.math.is_independent`. This makes it unlikely
            -- *but not impossible* -- that non-linear functions go undetected.
            The number of additional points at which the Jacobian is computed in the numerical
            test of ``is_independent`` as well as other options for this function
            can be controlled via ``validation_kwargs``.
            Furthermore, the QNode arguments *not* marked in ``argnum`` will not be
            considered in this test and if they resemble encoded inputs, the entire
            spectrum might be incorrect or the circuit might not even admit one.

        The ``qnode_spectrum`` function works in all interfaces:

        .. code-block:: python

            import tensorflow as tf

            dev = qml.device("default.qubit", wires=1)

            @qml.qnode(dev, interface='tf')
            def circuit(x):
                qml.RX(0.4*x[0], wires=0)
                qml.PhaseShift(x[1]*np.pi, wires=0)
                return qml.expval(qml.Z(0))

            x = tf.Variable([1., 2.])
            res = qml.fourier.qnode_spectrum(circuit)(x)

        >>> print(res)
        {"x": {(0,): [-0.4, 0.0, 0.4], (1,): [-3.14159, 0.0, 3.14159]}}

        Finally, compare ``qnode_spectrum`` with :func:`~.circuit_spectrum`, using
        the following circuit.

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def circuit(x, y, z):
                qml.RX(0.5*x**2, wires=0, id="x")
                qml.RY(2.3*y, wires=1, id="y0")
                qml.CNOT(wires=[1,0])
                qml.RY(z, wires=0, id="y1")
                return qml.expval(qml.Z(0))

        First, note that we assigned ``id`` labels to the gates for which we will use
        ``circuit_spectrum``. This allows us to choose these gates in the computation:

        >>> x, y, z = 0.1, 0.2, 0.3
        >>> circuit_spec_fn = qml.fourier.circuit_spectrum(circuit, encoding_gates=["x","y0","y1"])
        >>> circuit_spec = circuit_spec_fn(x, y, z)
        >>> for _id, spec in circuit_spec.items():
        ...     print(f"{_id}: {spec}")
        x: [-1.0, 0, 1.0]
        y0: [-1.0, 0, 1.0]
        y1: [-1.0, 0, 1.0]

        As we can see, the preprocessing in the QNode is not included in the simple spectrum.
        In contrast, the output of ``qnode_spectrum`` is:

        >>> adv_spec = qml.fourier.qnode_spectrum(circuit, encoding_args={"y", "z"})
        >>> for _id, spec in adv_spec.items():
        ...     print(f"{_id}: {spec}")
        y: {(): [-2.3, 0.0, 2.3]}
        z: {(): [-1.0, 0.0, 1.0]}

        Note that the values of the output are dictionaries instead of the spectrum lists, that
        they include the prefactors introduced by classical preprocessing, and
        that we would not be able to compute the advanced spectrum for ``x`` because it is
        preprocessed non-linearily in the gate ``qml.RX(0.5*x**2, wires=0, id="x")``.

    """
    # pylint: disable=too-many-branches,protected-access
    validation_kwargs = validation_kwargs or {}
    encoding_args, argnum = _process_ids(encoding_args, argnum, qnode)
    atol = 10 ** (-decimals) if decimals is not None else 1e-10
    # A map between Jacobian indices (contiguous) and arg names (may be discontiguous)
    arg_name_map = dict(enumerate(encoding_args))

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        old_interface = qnode.interface

        if old_interface == "auto":
            qnode.interface = qml.math.get_interface(*args, *list(kwargs.values()))

        jac_fn = qml.gradients.classical_jacobian(
            qnode, argnum=argnum, expand_fn=qml.transforms.expand_multipar
        )
        # Compute classical Jacobian and assert preprocessing is linear
        if not qml.math.is_independent(jac_fn, qnode.interface, args, kwargs, **validation_kwargs):
            raise ValueError(
                "The Jacobian of the classical preprocessing in the provided QNode "
                "is not constant; only linear classical preprocessing is supported."
            )
        # After construction, check whether invalid operations (for a spectrum)
        # are present in the QNode
        for m in qnode.qtape.measurements:
            if not isinstance(m, (qml.measurements.ExpectationMP, qml.measurements.ProbabilityMP)):
                raise ValueError(
                    f"The measurement {m.__class__.__name__} is not supported as it likely does "
                    "not admit a Fourier spectrum."
                )
        cjacs = jac_fn(*args, **kwargs)
        spectra = {}
        tape = qml.transforms.expand_multipar(qnode.qtape)
        par_info = tape._par_info

        # Iterate over jacobians per argument
        for jac_idx, cjac in enumerate(cjacs):
            # Obtain argument name for the jacobian index
            arg_name = arg_name_map[jac_idx]
            # Extract requested parameter indices for the current argument
            if encoding_args[arg_name] is Ellipsis:
                # If no index for this argument is specified, request all parameters within
                # the argument (Recall () is a valid index for scalar-valued arguments here)
                requested_par_ids = set(product(*(range(sh) for sh in cjac.shape[1:])))
            else:
                requested_par_ids = set(encoding_args[arg_name])
            # Each requested parameter at least "contributes" as a constant
            _spectra = {par_idx: {0} for par_idx in requested_par_ids}

            # Iterate over the axis of the current Jacobian that corresponds to the tape operations
            for op_idx, jac_of_op in enumerate(np.round(cjac, decimals=decimals)):
                op = par_info[op_idx]["op"]
                # Find parameters that both were requested and feed into the operation
                if len(cjac.shape) == 1:
                    # Scalar argument, only axis of Jacobian is for operations
                    if np.isclose(jac_of_op, 0.0, atol=atol, rtol=0):
                        continue
                    jac_of_op = {(): jac_of_op}
                    par_ids = {()}
                else:
                    # Array-valued argument
                    # Extract indices of parameters contributing to the current operation
                    par_ids = zip(*[map(int, _ids) for _ids in np.where(jac_of_op)])
                    # Exclude contributing parameters that were not requested
                    par_ids = set(par_ids).intersection(requested_par_ids)
                    if len(par_ids) == 0:
                        continue

                # Multi-parameter gates are not supported (we expanded the tape already)
                if len(op.parameters) != 1:
                    raise ValueError(
                        "Can only consider one-parameter gates as data-encoding gates; "
                        f"got {op.name}."
                    )
                spec = get_spectrum(op, decimals=decimals)

                # For each contributing parameter, rescale the operation's spectrum
                # and add it to the spectrum for that parameter
                for par_idx in par_ids:
                    scale = float(qml.math.abs(jac_of_op[par_idx]))
                    scaled_spec = [scale * f for f in spec]
                    _spectra[par_idx] = join_spectra(_spectra[par_idx], scaled_spec)

            # Construct the sorted spectrum also containing negative frequencies
            for idx, spec in _spectra.items():
                spec = sorted(spec)
                _spectra[idx] = [-freq for freq in spec[:0:-1]] + spec
            spectra[arg_name] = _spectra

            if old_interface == "auto":
                qnode.interface = "auto"
        return spectra

    return wrapper
