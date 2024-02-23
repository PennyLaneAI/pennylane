# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provides transforms for mitigating quantum circuits."""
from copy import copy

from typing import Any, Dict, Optional, Sequence, Callable

from pennylane import apply, adjoint
from pennylane.math import mean, shape, round
from pennylane.queuing import AnnotatedQueue
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms import transform

import pennylane as qml


@transform
def fold_global(tape: QuantumTape, scale_factor) -> (Sequence[QuantumTape], Callable):
    r"""Differentiable circuit folding of the global unitary ``circuit``.

    For a unitary circuit :math:`U = L_d .. L_1`, where :math:`L_i` can be either a gate or layer, ``fold_global`` constructs

    .. math:: \text{fold_global}(U) = U (U^\dagger U)^n (L^\dagger_d L^\dagger_{d-1} .. L^\dagger_s) (L_s .. L_d)

    where :math:`n = \lfloor (\lambda - 1)/2 \rfloor` and :math:`s = \lfloor \left(\lambda - 1 \right) (d/2) \rfloor` are determined via the ``scale_factor`` :math:`=\lambda`.
    The purpose of folding is to artificially increase the noise for zero noise extrapolation, see :func:`~.pennylane.transforms.mitigate_with_zne`.

    Args:
        tape (QNode or QuantumTape): the quantum circuit to be folded
        scale_factor (float): Scale factor :math:`\lambda` determining :math:`n` and :math:`s`

    Returns:
        qnode (QNode) or tuple[List[QuantumTape], function]: The folded circuit as described in :func:`qml.transform <pennylane.transform>`.

    .. seealso:: :func:`~.pennylane.transforms.mitigate_with_zne`; This function is analogous to the implementation in ``mitiq``  `mitiq.zne.scaling.fold_global <https://mitiq.readthedocs.io/en/v.0.1a2/apidoc.html?highlight=global_folding#mitiq.zne.scaling.fold_global>`_.

    **Example**

    Let us look at the following circuit.

    .. code-block:: python

        x = np.arange(6)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.RZ(x[2], wires=2)
            qml.CNOT(wires=(0,1))
            qml.CNOT(wires=(1,2))
            qml.RX(x[3], wires=0)
            qml.RY(x[4], wires=1)
            qml.RZ(x[5], wires=2)
            return qml.expval(qml.Z(0) @ qml.Z(1) @ qml.Z(2))


    Setting ``scale_factor=1`` does not affect the circuit:

    >>> folded = qml.transforms.fold_global(circuit, 1)
    >>> print(qml.draw(folded)(x))
    0: ──RX(0.0)─╭●──RX(3.0)──────────┤ ╭<Z@Z@Z>
    1: ──RY(1.0)─╰X─╭●────────RY(4.0)─┤ ├<Z@Z@Z>
    2: ──RZ(2.0)────╰X────────RZ(5.0)─┤ ╰<Z@Z@Z>

    Setting ``scale_factor=2`` results in the partially folded circuit :math:`U (L^\dagger_d L^\dagger_{d-1} .. L^\dagger_s) (L_s .. L_d)`
    with :math:`s = \lfloor \left(1 \mod 2 \right) d/2 \rfloor = 4` since the circuit is composed of :math:`d=8` gates.

    >>> folded = qml.transforms.fold_global(circuit, 2)
    >>> print(qml.draw(folded)(x))
    0: ──RX(0.0)─╭●──RX(3.0)──RX(3.0)†──RX(3.0)──────────────────┤ ╭<Z@Z@Z>
    1: ──RY(1.0)─╰X─╭●────────RY(4.0)───RY(4.0)†─╭●──╭●──RY(4.0)─┤ ├<Z@Z@Z>
    2: ──RZ(2.0)────╰X────────RZ(5.0)───RZ(5.0)†─╰X†─╰X──RZ(5.0)─┤ ╰<Z@Z@Z>

    Setting ``scale_factor=3`` results in the folded circuit :math:`U (U^\dagger U)`.

    >>> folded = qml.transforms.fold_global(circuit, 3)
    >>> print(qml.draw(folded)(x))
    0: ──RX(0.0)─╭●──RX(3.0)──RX(3.0)†───────────────╭●─────────RX(0.0)†──RX(0.0)─╭●──RX(3.0)──────────┤╭<Z@Z@Z>
    1: ──RY(1.0)─╰X─╭●────────RY(4.0)───RY(4.0)†─╭●──╰X†────────RY(1.0)†──RY(1.0)─╰X─╭●────────RY(4.0)─┤├<Z@Z@Z>
    2: ──RZ(2.0)────╰X────────RZ(5.0)───RZ(5.0)†─╰X†──RZ(2.0)†──RZ(2.0)──────────────╰X────────RZ(5.0)─┤╰<Z@Z@Z>

    .. note::

        Circuits are treated as lists of operations. Since the ordering of that list is ambiguous, so is its folding.
        This can be seen exemplarily for two equivalent unitaries :math:`U1 = X(0) Y(0) X(1) Y(1)` and :math:`U2 = X(0) X(1) Y(0) Y(1)`.
        The folded circuits according to ``scale_factor=2`` would be :math:`U1 (X(0) Y(0) Y(0) X(0))` and :math:`U2 (X(0) X(1) X(1) X(0))`, respectively.
        So even though :math:`U1` and :math:`U2` are describing the same quantum circuit, the ambiguity in their ordering as a list yields two differently folded circuits.

    .. details::

        The main purpose of folding is for zero noise extrapolation (ZNE). PennyLane provides a differentiable transform :func:`~.pennylane.transforms.mitigate_with_zne`
        that allows you to perform ZNE as a black box. If you want more control and `see` the extrapolation, you can follow the logic of the following example.

        We start by setting up a noisy device using the mixed state simulator and a noise channel.

        .. code-block:: python

            n_wires = 4

            # Describe noise
            noise_gate = qml.DepolarizingChannel
            noise_strength = 0.05

            # Load devices
            dev_ideal = qml.device("default.mixed", wires=n_wires)
            dev_noisy = qml.transforms.insert(noise_gate, noise_strength)(dev_ideal)

            x = np.arange(6)

            H = 1.*qml.X(0) @ qml.X(1) + 1.*qml.X(1) @ qml.X(2)

            def circuit(x):
                qml.RY(x[0], wires=0)
                qml.RY(x[1], wires=1)
                qml.RY(x[2], wires=2)
                qml.CNOT(wires=(0,1))
                qml.CNOT(wires=(1,2))
                qml.RY(x[3], wires=0)
                qml.RY(x[4], wires=1)
                qml.RY(x[5], wires=2)
                return qml.expval(H)

            qnode_ideal = qml.QNode(circuit, dev_ideal)
            qnode_noisy = qml.QNode(circuit, dev_noisy)

        We can then create folded versions of the noisy qnode and execute them for different scaling factors.

        >>> scale_factors = [1., 2., 3.]
        >>> folded_res = [qml.transforms.fold_global(qnode_noisy, lambda_)(x) for lambda_ in scale_factors]

        We want to later compare the ZNE with the ideal result.

        >>> ideal_res = qnode_ideal(x)

        ZNE is, as the name suggests, an extrapolation in the noise to zero. The underlyding assumption is that the level of noise is proportional to the scaling factor
        by artificially increasing the circuit depth. We can perform a polynomial fit using ``numpy`` functions. Note that internally in :func:`~.pennylane.transforms.mitigate_with_zne`
        a differentiable polynomial fit function :func:`~.pennylane.transforms.poly_extrapolate` is used.

        >>> # coefficients are ordered like coeffs[0] * x**2 + coeffs[1] * x + coeffs[0]
        >>> coeffs = np.polyfit(scale_factors, folded_res, 2)
        >>> zne_res = coeffs[-1]

        We used a polynomial fit of ``order=2``. Using ``order=len(scale_factors) -1`` is also referred to as Richardson extrapolation and implemented in :func:`~.pennylane.transforms.richardson_extrapolate`.
        We can now visualize our fit to see how close we get to the ideal result with this mitigation technique.

        .. code-block:: python

            x_fit = np.linspace(0, 3, 20)
            y_fit = np.poly1d(coeffs)(x_fit)

            plt.plot(scale_factors, folded_res, "x--", label="folded")
            plt.plot(0, ideal_res, "X", label="ideal res")
            plt.plot(0, zne_res, "X", label="ZNE res", color="tab:red")
            plt.plot(x_fit, y_fit, label="fit", color="tab:red", alpha=0.5)
            plt.legend()

        .. figure:: ../../_static/fold_global_zne_by-hand.png
            :align: center
            :width: 60%
            :target: javascript:void(0);


    """
    # The main intention for providing ``fold_global`` was for it to be used in combination with ``mitigate_with_zne``, which also works with mitiq functions.
    # To preserve the mitiq functionality, ``mitigate_with_zne`` should get a tape transform.
    # To make ``fold_global`` also user-facing and work with qnodes, this function is batch_transformed instead, and therefore applicable on qnodes.
    return [fold_global_tape(tape, scale_factor)], lambda x: x[0]


def _divmod(a, b):
    """Performs divmod but in an all-interface compatible manner"""
    out1 = qml.math.floor(a / b)
    out2 = a - out1 * b
    return int(out1), out2


def fold_global_tape(circuit, scale_factor):
    r"""
    This is the internal tape transform to be used with :func:`~.pennylane.transforms.mitigate_with_zne`.
    For the user-facing function see :func:`~.pennylane.transforms.fold_global`.

    Args:
        circuit (QuantumTape): the circuit to be folded
        scale_factor (float): Scale factor :math:`\lambda` determining :math:`n` and :math:`s`

    Returns:
        QuantumTape: Folded circuit

    """

    # TODO: simplify queing via qfunc(op) - currently just a workaround, to solve the problem of ownership when tape contains adjoint(op)
    # https://github.com/PennyLaneAI/pennylane/pull/2766 already touched on the issue, future work
    # in Q3 2022 should make it possible to substantially simplify this.
    def qfunc(op):
        copy(op).queue()

    # Generate base_circuit without measurements
    # Treat all circuits as lists of operations, build new tape in the end

    base_ops = circuit.expand().copy(copy_operations=True).operations

    num_global_folds, fraction_scale = _divmod(scale_factor - 1, 2)

    n_ops = len(base_ops)
    num_to_fold = int(round(fraction_scale * n_ops / 2))

    # Create new_circuit from folded list
    with AnnotatedQueue() as new_circuit_q:
        # Original U
        for op in base_ops:
            qfunc(op)

        # Folding U => U (U^H U)**n.
        for _ in range(int(num_global_folds)):
            for op in base_ops[::-1]:
                adjoint(qfunc)(op)

            for op in base_ops:
                qfunc(op)

        # Remainder folding U => U (U^H U)**n (L_d^H .. L_s^H) (L_s .. L_d)
        for i in range(n_ops - 1, n_ops - num_to_fold - 1, -1):
            adjoint(qfunc)(base_ops[i])

        for i in range(n_ops - num_to_fold, n_ops):
            qfunc(base_ops[i])

        # Append measurements
        for meas in circuit.measurements:
            apply(meas)

    return QuantumScript.from_queue(new_circuit_q)


# TODO: make this a pennylane.math function
def _polyfit(x, y, order):
    """Brute force implementation of polynomial fit"""
    x = qml.math.convert_like(x, y[0])
    x = qml.math.cast_like(x, y[0])
    X = qml.math.vander(x, order + 1)
    y = qml.math.stack(y)

    # scale X to improve condition number and solve
    scale = qml.math.sum(qml.math.sqrt((X * X)), axis=0)
    X = X / scale

    # Compute coeffs:
    # This part is typically done using a lstq solver, do it with the penrose inverse by hand:
    # i.e. coeffs = (X.T @ X)**-1 X.T @ y see https://en.wikipedia.org/wiki/Polynomial_regression
    c = qml.math.linalg.pinv(qml.math.transpose(X) @ X)
    c = c @ qml.math.transpose(X)
    c = qml.math.tensordot(c, y, axes=1)
    c = qml.math.transpose(qml.math.transpose(c) / scale)
    return c


def poly_extrapolate(x, y, order):
    r"""Extrapolator to :math:`f(0)` for polynomial fit.

    The polynomial is defined as ``f(x) = p[0] * x**deg + p[1] * x**(deg-1) + ... + p[deg]`` such that ``deg = order + 1``.
    This function is compatible with all interfaces supported by pennylane.

    Args:
        x (Array): Data in x
        y (Array): Data in y = f(x)
        order (int): Order of the polynomial fit

    Returns:
        float: Extrapolated value at f(0).

    .. seealso:: :func:`~.pennylane.transforms.richardson_extrapolate`, :func:`~.pennylane.transforms.mitigate_with_zne`

    **Example:**

    >>> x = np.linspace(1, 10, 5)
    >>> y = x**2 + x + 1 + 0.3 * np.random.rand(len(x))
    >>> qml.transforms.poly_extrapolate(x, y, 2)
    tensor(1.01717601, requires_grad=True)

    """
    coeff = _polyfit(x, y, order)
    return coeff[-1]


def richardson_extrapolate(x, y):
    r"""Polynomial fit where the degree of the polynomial is fixed to being equal to the length of ``x``.

    In a nutshell, this function is calling  :func:`~.pennylane.transforms.poly_extrapolate` with ``order = len(x)-1``.
    This function is compatible with all interfaces supported by pennylane.

    Args:
        x (Array): Data in x
        y (Array): Data in y = f(x)

    Returns:
        float: Extrapolated value at f(0).

    .. seealso:: :func:`~.pennylane.transforms.poly_extrapolate`, :func:`~.pennylane.transforms.mitigate_with_zne`

    **Example:**

    >>> x = np.linspace(1, 10, 5)
    >>> y = x**2 + x + 1 + 0.3 * np.random.rand(len(x))
    >>> qml.transforms.richardson_extrapolate(x, y)
    tensor(1.15105156, requires_grad=True)

    """
    return poly_extrapolate(x, y, len(x) - 1)


# pylint: disable=too-many-arguments, protected-access
@transform
def mitigate_with_zne(
    tape: QuantumTape,
    scale_factors: Sequence[float],
    folding: callable,
    extrapolate: callable,
    folding_kwargs: Optional[Dict[str, Any]] = None,
    extrapolate_kwargs: Optional[Dict[str, Any]] = None,
    reps_per_factor=1,
) -> (Sequence[QuantumTape], Callable):
    r"""Mitigate an input circuit using zero-noise extrapolation.

    Error mitigation is a precursor to error correction and is compatible with near-term quantum
    devices. It aims to lower the impact of noise when evaluating a circuit on a quantum device by
    evaluating multiple variations of the circuit and post-processing the results into a
    noise-reduced estimate. This transform implements the zero-noise extrapolation (ZNE) method
    originally introduced by
    `Temme et al. <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.180509>`__ and
    `Li et al. <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.7.021050>`__.

    Details on the functions passed to the ``folding`` and ``extrapolate`` arguments of this
    transform can be found in the usage details. This transform is compatible with functionality
    from the `Mitiq <https://mitiq.readthedocs.io/en/stable/>`__ package (version 0.11.0 and above),
    see the example and usage details for further information.

    Args:
        tape (QNode or QuantumTape): the quantum circuit to be error-mitigated
        scale_factors (Sequence[float]): the range of noise scale factors used
        folding (callable): a function that returns a folded circuit for a specified scale factor
        extrapolate (callable): a function that returns an extrapolated result when provided a
            range of scale factors and corresponding results
        folding_kwargs (dict): optional keyword arguments passed to the ``folding`` function
        extrapolate_kwargs (dict): optional keyword arguments passed to the ``extrapolate`` function
        reps_per_factor (int): Number of circuits generated for each scale factor. Useful when the
            folding function is stochastic.

    Returns:
        qnode (QNode) or tuple[List[.QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the mitigated results in the form of a tensor of a tensor, a tuple, or a nested tuple depending
        upon the nesting structure of measurements in the original circuit.

    **Example:**

    We first create a noisy device using ``default.mixed`` by adding :class:`~.AmplitudeDamping` to
    each gate of circuits executed on the device using the :func:`~.transforms.insert` transform:

    .. code-block:: python3

        import pennylane as qml

        noise_strength = 0.05

        dev = qml.device("default.mixed", wires=2)
        dev = qml.transforms.insert(qml.AmplitudeDamping, noise_strength)(dev)

    We can now set up a mitigated QNode by passing a ``folding`` and ``extrapolate`` function. PennyLane provides native
    functions :func:`~.pennylane.transforms.fold_global` and :func:`~.pennylane.transforms.poly_extrapolate` or :func:`~.pennylane.transforms.richardson_extrapolate` that
    allow for differentiating through them. Custom functions, as well as functionalities from the `Mitiq <https://mitiq.readthedocs.io/en/stable/>`__ package
    are supported as well (see usage details below).

    .. code-block:: python3

        from functools import partial
        from pennylane import numpy as np
        from pennylane import qnode

        from pennylane.transforms import fold_global, poly_extrapolate

        n_wires = 2
        n_layers = 2

        shapes = qml.SimplifiedTwoDesign.shape(n_wires, n_layers)
        np.random.seed(0)
        w1, w2 = [np.random.random(s) for s in shapes]

        @partial(qml.transforms.mitigate_with_zne, [1., 2., 3.], fold_global, poly_extrapolate, extrapolate_kwargs={'order': 2})
        @qnode(dev)
        def circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.Z(0))

    Executions of ``circuit`` will now be mitigated:

    >>> circuit(w1, w2)
    0.19113067083636542

    The unmitigated circuit result is ``0.33652776`` while the ideal circuit result is
    ``0.23688169`` and we can hence see that mitigation has helped reduce our estimation error.

    This mitigated qnode can be differentiated like any other qnode.

    >>> qml.grad(circuit)(w1, w2)
    (array([-0.89319941,  0.37949841]),
     array([[[-7.04121596e-01,  3.00073104e-01]],
            [[-6.41155176e-01,  8.32667268e-17]]]))

    .. details::
        :title: Usage Details

        **Theoretical details**

        A summary of ZNE can be found in `LaRose et al. <https://arxiv.org/abs/2009.04417>`__. The
        method works by assuming that the amount of noise present when a circuit is run on a
        noisy device is enumerated by a parameter :math:`\gamma`. Suppose we have an input circuit
        that experiences an amount of noise equal to :math:`\gamma = \gamma_{0}` when executed.
        Ideally, we would like to evaluate the result of the circuit in the :math:`\gamma = 0`
        noise-free setting.

        To do this, we create a family of equivalent circuits whose ideal noise-free value is the
        same as our input circuit. However, when run on a noisy device, each circuit experiences
        a noise equal to :math:`\gamma = s \gamma_{0}` for some scale factor :math:`s`. By
        evaluating the noisy outputs of each circuit, we can extrapolate to :math:`s=0` to estimate
        the result of running a noise-free circuit.

        A key element of ZNE is the ability to run equivalent circuits for a range of scale factors
        :math:`s`. When the noise present in a circuit scales with the number of gates, :math:`s`
        can be varied using `unitary folding <https://ieeexplore.ieee.org/document/9259940>`__.
        Unitary folding works by noticing that a unitary :math:`U` is equivalent to
        :math:`U U^{\dagger} U`. This type of transform can be applied to individual gates in the
        circuit or to the whole circuit. When no folding occurs, the scale factor is
        :math:`s=1` and we are running our input circuit. On the other hand, when each gate has been
        folded once, we have tripled the amount of noise in the circuit so that :math:`s=3`. For
        :math:`s \geq 3`, each gate in the circuit will be folded more than once. A typical choice
        of scale parameters is :math:`(1, 2, 3)`.

        **Unitary folding**

        This transform applies ZNE to an input circuit using the unitary folding approach. It
        requires a callable to be passed as the ``folding`` argument with signature

        .. code-block:: python

            fn(circuit, scale_factor, **folding_kwargs)

        where

        - ``circuit`` is a quantum tape,

        - ``scale_factor`` is a float, and

        - ``folding_kwargs`` are optional keyword arguments.

        The output of the function should be the folded circuit as a quantum tape.
        Folding functionality is available from the
        `Mitiq <https://mitiq.readthedocs.io/en/stable/>`__ package (version 0.11.0 and above)
        in the
        `zne.scaling.folding <https://mitiq.readthedocs.io/en/stable/apidoc.html#module-mitiq.zne.scaling.folding>`__
        module.

        .. warning::

            Calculating the gradient of mitigated circuits is not supported when using the Mitiq
            package as a backend for folding or extrapolation.

        **Extrapolation**

        This transform also requires a callable to be passed to the ``extrapolate`` argument that
        returns the extrapolated value(s). Its function should be

        .. code-block:: python

            fn(scale_factors, results, **extrapolate_kwargs)

        where

        - ``scale_factors`` are the ZNE scale factors,

        - ``results`` are the execution results of the circuit at the specified scale
          factors of shape ``(len(scale_factors), len(qnode_returns))``, and

        - ``extrapolate_kwargs`` are optional keyword arguments.

        The output of the extrapolate ``fn`` should be a flat array of
        length ``len(qnode_returns)``.

        Extrapolation functionality is available using ``extrapolate``
        methods of the factories in the
        `mitiq.zne.inference <https://mitiq.readthedocs.io/en/stable/apidoc.html#module-mitiq.zne.inference>`__
        module.
    """
    folding_kwargs = folding_kwargs or {}
    extrapolate_kwargs = extrapolate_kwargs or {}

    tape = tape.expand(stop_at=lambda op: not isinstance(op, QuantumScript))
    script_removed = QuantumScript(tape.operations[tape.num_preps :])

    tapes = [
        [folding(script_removed, s, **folding_kwargs) for _ in range(reps_per_factor)]
        for s in scale_factors
    ]

    tapes = [tape_ for tapes_ in tapes for tape_ in tapes_]  # flattens nested list

    # if folding was a batch transform, ignore the processing function
    if isinstance(tapes[0], tuple) and isinstance(tapes[0][0], list) and callable(tapes[0][1]):
        tapes = [t[0] for t, _ in tapes]

    prep_ops = tape.operations[: tape.num_preps]
    out_tapes = [QuantumScript(prep_ops + tape_.operations, tape.measurements) for tape_ in tapes]

    def processing_fn(results):
        """Maps from input tape executions to an error-mitigated estimate"""

        # content of `results` must be modified in this post-processing function
        results = list(results)

        for i, tape in enumerate(out_tapes):
            # stack the results if there are multiple measurements
            # this will not create ragged arrays since only expval measurements are allowed
            if len(tape.observables) > 1:
                results[i] = qml.math.stack(results[i])

        # Averaging over reps_per_factor repetitions
        results_flattened = []
        for i in range(0, len(results), reps_per_factor):
            # The stacking ensures the right interface is used
            # averaging over axis=0 is critical because the qnode may have multiple outputs
            results_flattened.append(mean(qml.math.stack(results[i : i + reps_per_factor]), axis=0))

        extrapolated = extrapolate(scale_factors, results_flattened, **extrapolate_kwargs)

        extrapolated = extrapolated[0] if shape(extrapolated) == (1,) else extrapolated

        # unstack the results in the case of multiple measurements
        return extrapolated if shape(extrapolated) == () else tuple(qml.math.unstack(extrapolated))

    return out_tapes, processing_fn
