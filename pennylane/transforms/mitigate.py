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

from typing import Any, Dict, Optional, Sequence, Union

from pennylane import QNode, apply, adjoint
from pennylane.math import mean, shape, round
from pennylane.tape import QuantumTape
from pennylane.transforms import batch_transform

import pennylane as qml


@batch_transform
def fold_global(circuit, scale_factor):
    r"""Diffable global circuit folding function as is done in `mitiq.zne.scaling.fold_global <https://mitiq.readthedocs.io/en/v.0.1a2/apidoc.html?highlight=global_folding#mitiq.zne.scaling.fold_global>`_

    For a unitary ``circuit`` :math:`U = L_d .. L_1`, where :math:`L_i` can be either a gate or layer, ``fold_global`` constructs

    .. math:: \text{fold_global}(U) = U (U^\dagger U)^n (L^\dagger_d L^\dagger_{d-1} .. L^\dagger_s) (L_s .. L_d)

    where :math:`n = \lfloor (\lambda - 1)/2 \rfloor` and :math:`s = \lfloor \left((\lambda -1) \mod 2 \right) (d/2) \rfloor` are determined via the ``scale_factor`` :math:`=\lambda`.
    The purpose of folding is to artificially increase the noise for zero noise extrapolation, see :func:`~.pennylane.transforms.mitigate_with_zne`.

    Args:
        circuit (callable or QuantumTape): the circuit to be folded
        scale_factor (float): Scale factor :math:`\lambda` determining :math:`n` and :math:`s`

    Returns:
        QuantumTape: Folded circuit

    **Example:**
    # TODO: change to qnode examples, provide manual ZNE with fitting example.
    .. code-block:: python

        x = np.arange(6)
        with qml.tape.QuantumTape() as tape:
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.RZ(x[2], wires=2)
            qml.CNOT(wires=(0,1))
            qml.CNOT(wires=(1,2))
            qml.RX(x[3], wires=0)
            qml.RY(x[4], wires=1)
            qml.RZ(x[5], wires=2)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

    Setting ``scale_factor = 1`` does not affect the circuit:

    >>> folded_tape = qml.transforms.fold_global(tape, 1)
    >>> print(qml.drawer.tape_text(folded_tape, decimals=1))
    0: ──RX(0.0)─╭●──RX(3.0)──────────┤ ╭<Z@Z@Z>
    1: ──RY(1.0)─╰X─╭●────────RY(4.0)─┤ ├<Z@Z@Z>
    2: ──RZ(2.0)────╰X────────RZ(5.0)─┤ ╰<Z@Z@Z>

    Setting ``scale_factor = 2`` results in the partially folded circuit :math:`U (L^\dagger_d L^\dagger_{d-1} .. L^\dagger_s) (L_s .. L_d)`
    with :math:`s = \lfloor \left(1 \mod 2 \right) d/2 \rfloor = 4` since the circuit is composed of :math:`d=8` gates.

    >>> folded_tape = qml.transforms.fold_global(tape, 2)
    >>> print(qml.drawer.tape_text(folded_tape, decimals=1))
    0: ──RX(0.0)─╭●──RX(3.0)──RX(3.0)†──RX(3.0)──────────────────┤ ╭<Z@Z@Z>
    1: ──RY(1.0)─╰X─╭●────────RY(4.0)───RY(4.0)†─╭●──╭●──RY(4.0)─┤ ├<Z@Z@Z>
    2: ──RZ(2.0)────╰X────────RZ(5.0)───RZ(5.0)†─╰X†─╰X──RZ(5.0)─┤ ╰<Z@Z@Z>

    Setting ``scale_factor = 3`` results in the folded circuit :math:`U (U^\dagger U)`.

    >>> folded_tape = qml.transforms.fold_global(tape, 3)
    >>> print(qml.drawer.tape_text(folded_tape, decimals=1))
    0: ──RX(0.0)─╭●──RX(3.0)──RX(3.0)†───────────────╭●─────────RX(0.0)†──RX(0.0)─╭●──RX(3.0)──────────┤╭<Z@Z@Z>
    1: ──RY(1.0)─╰X─╭●────────RY(4.0)───RY(4.0)†─╭●──╰X†────────RY(1.0)†──RY(1.0)─╰X─╭●────────RY(4.0)─┤├<Z@Z@Z>
    2: ──RZ(2.0)────╰X────────RZ(5.0)───RZ(5.0)†─╰X†──RZ(2.0)†──RZ(2.0)──────────────╰X────────RZ(5.0)─┤╰<Z@Z@Z>

    .. note::

        Circuits are treated as lists of operations. Since the ordering is ambiguous, as seen exemplarily
        for :math:`U = X(0) Y(0) X(1) Y(1) = X(0) X(1) Y(0) Y(1)`, also partially folded circuits are ambiguous.
    """
    # The main intention for providing ``fold_global`` was for it to be used in combination with ``mitigate_with_zne``, which also works with mitiq functions.
    # To preserve the mitiq functionality, ``mitigate_with_zne`` should get a tape transform.
    # To make ``fold_global`` also user-facing and work with qnodes, this function is batch_transformed instead, and therefore applicable on qnodes.
    return [fold_global_tape(circuit, scale_factor)], lambda x: x[0]


def fold_global_tape(circuit, scale_factor):
    """TODO doc-string linkling to fold_global"""

    if scale_factor < 1.0:
        raise AttributeError("scale_factor must be >= 1")
    assert scale_factor >= 1.0

    # TODO: simplify this - currently just a workaround, problem is when tape contains adjoint(op)
    def qfunc(op):
        copy(op).queue()

    # Generate base_circuit without measurements
    # Treat all circuits as lists of operations, build new tape in the end

    base_ops = circuit.expand().copy(copy_operations=True).operations

    def _divmod(a, b):
        """Performs divmod but in an all-interface compatible manner"""
        out1 = qml.math.floor(a / b)
        out2 = a - out1 * b
        return int(out1), int(out2)

    num_global_folds, fraction_scale = _divmod(scale_factor - 1, 2)

    n_ops = len(base_ops)
    num_to_fold = int(round(fraction_scale * n_ops / 2))

    # Create new_circuit from folded list
    with QuantumTape() as new_circuit:
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

    # return [new_circuit], lambda x: x
    return new_circuit


# TODO: make this a pennylane.math function
def _polyfit(x, y, order):
    """Brute force implementation of polynomial fit"""
    lhs = qml.math.vander(x, order + 1)
    rhs = qml.math.stack(y)  # [qml.math.stack(i) for i in y]

    # scale lhs to improve condition number and solve
    scale = qml.math.sum(qml.math.sqrt((lhs * lhs)), axis=0)
    lhs /= scale

    # Compute coeffs:
    # This part is typically done using a lstq solver, do it with the penrose inverse by hand:
    # i.e. coeffs = (X.T @ X)**-1 X.T @ y see https://en.wikipedia.org/wiki/Polynomial_regression
    c = qml.math.linalg.pinv(qml.math.transpose(lhs) @ lhs)
    c = c @ qml.math.transpose(lhs)
    c = qml.math.dot(c, rhs)

    c = qml.math.transpose(qml.math.transpose(c) / scale)
    return c


def poly_extrapolate(x, y, order):
    """Extrapolator to f(0) for polynomial fit.

    The polynomial is defined as ``f(x) = p[0] * x**deg + p[1] * x**(deg-1) + ... + p[deg]`` such that ``deg = order + 1``.

    Args:
        x (Array): Data in x
        y (Array): Data in y = f(x)
        order (int): Order of the polynomial fit

    Returns:
        float: Extrapolated value at f(0).

    .. seealso:: :func:`~.pennylane.transforms.Richardson_extrapolate`, :func:`~.pennylane.transforms.mitigate_with_zne`

    **Example:**

    >>> x = np.linspace(1, 10, 5)
    >>> y = x**2 + x + 1 + 0.3 * np.random.rand(len(x))
    >>> qml.transforms.poly_extrapolate(x, y, 2)
    tensor(1.01717601, requires_grad=True)

    """
    coeff = _polyfit(x, y, order)
    return coeff[-1]


def Richardson_extrapolate(x, y):
    """Polynomial fit :func:`~.pennylane.transforms.poly_extrapolate` with ``order = len(x)-1``.

    Args:
        x (Array): Data in x
        y (Array): Data in y = f(x)

    Returns:
        float: Extrapolated value at f(0).

    .. seealso:: :func:`~.pennylane.transforms.poly_extrapolate`, :func:`~.pennylane.transforms.mitigate_with_zne`

    **Example:**

    >>> x = np.linspace(1, 10, 5)
    >>> y = x**2 + x + 1 + 0.3 * np.random.rand(len(x))
    >>> qml.transforms.Richardson_extrapolate(x, y)
    tensor(1.15105156, requires_grad=True)

    """
    return poly_extrapolate(x, y, len(x) - 1)


# pylint: disable=too-many-arguments, protected-access, bad-continuation
@batch_transform
def mitigate_with_zne(
    circuit: Union[QNode, QuantumTape],
    scale_factors: Sequence[float],
    folding: callable,
    extrapolate: callable,
    folding_kwargs: Optional[Dict[str, Any]] = None,
    extrapolate_kwargs: Optional[Dict[str, Any]] = None,
    reps_per_factor=1,
) -> float:
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
        circuit (callable or QuantumTape): the circuit to be error-mitigated
        scale_factors (Sequence[float]): the range of noise scale factors used
        folding (callable): a function that returns a folded circuit for a specified scale factor
        extrapolate (callable): a function that returns an extrapolated result when provided a
            range of scale factors and corresponding results
        folding_kwargs (dict): optional keyword arguments passed to the ``folding`` function
        extrapolate_kwargs (dict): optional keyword arguments passed to the ``extrapolate`` function
        reps_per_factor (int): Number of circuits generated for each scale factor. Useful when the
            folding function is stochastic.

    Returns:
        float: the result of evaluating the circuit when mitigated using ZNE

    **Example:**

    We first create a noisy device using ``default.mixed`` by adding :class:`~.AmplitudeDamping` to
    each gate of circuits executed on the device using the :func:`~.transforms.insert` transform:

    .. code-block:: python3

        import pennylane as qml

        noise_strength = 0.05

        dev = qml.device("default.mixed", wires=2)
        dev = qml.transforms.insert(qml.AmplitudeDamping, noise_strength)(dev)

    We can now set up a mitigated QNode by passing a ``folding`` and ``extrapolate`` function. PennyLane provides propriertary
    functions :func:`~.pennylane.transforms.fold_global` and :func:`~.pennylane.transforms.poly_extrapolate` or :func:`~.pennylane.transforms.Richardson_extrapolate` that
    allow for differentiating through them. Custom functions, as well as functionalities from the `Mitiq <https://mitiq.readthedocs.io/en/stable/>`__ package
    are supported as well (see usage details below).

    .. code-block:: python3

        from pennylane import numpy as np
        from pennylane import qnode

        from pennylane.transforms import fold_global, poly_extrapolate

        n_wires = 2
        n_layers = 2

        shapes = qml.SimplifiedTwoDesign.shape(n_wires, n_layers)
        np.random.seed(0)
        w1, w2 = [np.random.random(s) for s in shapes]

        @qml.transforms.mitigate_with_zne([1., 2., 3.], fold_global, poly_extrapolate, extrapolate_kwargs = {'order': 2})
        @qnode(dev)
        def circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0))

    Executions of ``circuit`` will now be mitigated:

    >>> circuit(w1, w2)
    0.19113067083636542

    The unmitigated circuit result is ``0.33652776`` while the ideal circuit result is
    ``0.23688169`` and we can hence see that mitigation has helped reduce our estimation error.

    This mitigated qnode can be differentiated like any other qnode.

    >>> qml.grad(circuit)(w1, w2)

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


    if isinstance(folding, qml.batch_transform):
        folding = fold_global_tape

    tape = circuit.expand(stop_at=lambda op: not isinstance(op, QuantumTape))

    with QuantumTape() as tape_removed:
        for op in tape._ops:
            apply(op)
    # print("tape: ", tape)
    tapes = [
        [folding(tape_removed, s, **folding_kwargs) for _ in range(reps_per_factor)]
        for s in scale_factors
    ]
    # print("tapes: ", tapes)
    tapes = [tape_ for tapes_ in tapes for tape_ in tapes_]  # flattens nested list
    # print("tapes after un-nesting: ", tapes)

    out_tapes = []

    for tape_ in tapes:
        # pylint: disable=expression-not-assigned
        # print("tape_:", tape_)
        # tape_ = tape_[0][0]
        with QuantumTape() as t:
            [apply(p) for p in tape._prep]
            [apply(op) for op in tape_.operations]
            [apply(m) for m in tape.measurements]
        out_tapes.append(t)

    def processing_fn(results):
        """Maps from input tape executions to an error-mitigated estimate"""

        # Averaing over reps_per_factor repititons
        results_flattened = []
        for i in range(0, len(results), reps_per_factor):
            # The stacking ensures the right interface is used
            # averaging over axis=0 is critical because the qnode may have multiple outputs
            results_flattened.append(mean(qml.math.stack(results[i : i + reps_per_factor]), axis=0))

        extrapolated = extrapolate(scale_factors, results_flattened, **extrapolate_kwargs)

        return extrapolated[0] if shape(extrapolated) == (1,) else extrapolated

    return out_tapes, processing_fn
