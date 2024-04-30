# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains functions for computing the SPSA gradient
of a quantum tape.
"""
# pylint: disable=protected-access,too-many-arguments,too-many-branches,too-many-statements,unused-argument
from typing import Sequence, Callable
from functools import partial

import numpy as np

import pennylane as qml
from pennylane import transform
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from pennylane.transforms.tape_expand import expand_invalid_trainable

from .finite_difference import _processing_fn, finite_diff_coeffs
from .gradient_transform import (
    _all_zero_grad,
    assert_no_trainable_tape_batching,
    choose_trainable_params,
    find_and_validate_gradient_methods,
    _no_trainable_grad,
)
from .general_shift_rules import generate_multishifted_tapes


def _rademacher_sampler(indices, num_params, *args, rng):
    r"""Sample a random vector with (independent) entries from {+1, -1} with balanced probability.
    That is, each entry follows the
    `Rademacher distribution. <https://en.wikipedia.org/wiki/Rademacher_distribution>`_

    The signature corresponds to the one required for the input ``sampler`` to ``spsa_grad``:

    Args:
        indices (Sequence[int]): Indices of the trainable tape parameters that will be perturbed.
        num_params (int): Total number of trainable tape parameters.
        rng (np.random.Generator): A NumPy pseudo-random number generator.

    Returns:
        tensor_like: Vector of size ``num_params`` with non-zero entries at positions indicated
        by ``indices``, each entry sampled independently from the Rademacher distribution.
    """
    # pylint: disable=unused-argument
    direction = np.zeros(num_params)
    direction[indices] = rng.choice([-1, 1], size=len(indices))
    return direction


def _expand_transform_spsa(
    tape: qml.tape.QuantumTape,
    argnum=None,
    h=1e-5,
    approx_order=2,
    n=1,
    strategy="center",
    f0=None,
    validate_params=True,
    num_directions=1,
    sampler=_rademacher_sampler,
    sampler_rng=None,
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Expand function to be applied before spsa gradient."""
    expanded_tape = expand_invalid_trainable(tape)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [expanded_tape], null_postprocessing


@partial(
    transform,
    expand_transform=_expand_transform_spsa,
    classical_cotransform=_contract_qjac_with_cjac,
    final_transform=True,
)
def spsa_grad(
    tape: qml.tape.QuantumTape,
    argnum=None,
    h=1e-5,
    approx_order=2,
    n=1,
    strategy="center",
    f0=None,
    validate_params=True,
    num_directions=1,
    sampler=_rademacher_sampler,
    sampler_rng=None,
) -> (Sequence[qml.tape.QuantumTape], Callable):
    r"""Transform a circuit to compute the SPSA gradient of all gate
    parameters with respect to its inputs. This estimator shifts all parameters
    simultaneously and approximates the gradient based on these shifts and a
    finite-difference method.

    Args:
        tape (QNode or QuantumTape): quantum circuit to differentiate
        argnum (int or list[int] or None): Trainable parameter indices to differentiate
            with respect to. If not provided, the derivatives with respect to all
            trainable parameters are returned. Note that the indices are with respect to
            the list of trainable parameters.
        h (float or tensor_like[float]): Step size for the finite-difference method
            underlying the SPSA. Can be a tensor-like object
            with as many entries as differentiated *gate* parameters
        approx_order (int): The approximation order of the finite-difference method underlying
            the SPSA gradient.
        n (int): compute the :math:`n`-th derivative
        strategy (str): The strategy of the underlying finite difference method. Must be one of
            ``"forward"``, ``"center"``, or ``"backward"``.
            For the ``"forward"`` strategy, the finite-difference shifts occur at the points
            :math:`x_0, x_0+h, x_0+2h,\dots`, where :math:`h` is the stepsize ``h``.
            The ``"backwards"`` strategy is similar, but in
            reverse: :math:`x_0, x_0-h, x_0-2h, \dots`. Finally, the
            ``"center"`` strategy results in shifts symmetric around the
            unshifted point: :math:`\dots, x_0-2h, x_0-h, x_0, x_0+h, x_0+2h,\dots`.
        f0 (tensor_like[float] or None): Output of the evaluated input tape in ``tape``. If
            provided, and the gradient recipe contains an unshifted term, this value is used,
            saving a quantum evaluation.
        validate_params (bool): Whether to validate the tape parameters or not. If ``True``,
            the ``Operation.grad_method`` attribute and the circuit structure will be analyzed
            to determine if the trainable parameters support the finite-difference method,
            inferring that they support SPSA as well.
            If ``False``, the SPSA gradient method will be applied to all parameters without
            checking.
        num_directions (int): Number of sampled simultaneous perturbation vectors. An estimate for
            the gradient is computed for each vector using the underlying finite-difference
            method, and afterwards all estimates are averaged.
        sampler (callable): Sampling method to obtain the simultaneous perturbation directions.
            The sampler should take the following arguments:

            - A ``Sequence[int]`` that contains the indices of those trainable tape parameters
              that will be perturbed, i.e. have non-zero entries in the output vector.

            - An ``int`` that indicates the total number of trainable tape parameters. The
              size of the output vector has to match this input.

            - An ``int`` indicating the iteration counter during the gradient estimation.
              A valid sampling method can, but does not have to, take this counter into
              account. In any case, ``sampler`` has to accept this third argument.

            - The required keyword argument ``rng``, expected to be a NumPy pseudo-random
              number generator, which should be used to sample directions randomly.

            Note that the circuit evaluations in the various sampled directions are *averaged*,
            not simply summed up.

        sampler_rng (Union[np.random.Generator, int, None]): Either a NumPy pseudo-random number
            generator or an integer, which will be used as the PRNG seed. Default is None, which
            creates a NumPy PRNG without a seed. Note that calling ``spsa_grad`` multiple times
            with a seed (i.e., an integer) will result in the same directions being sampled in each
            call. In this case it is advisable to create a NumPy PRNG and pass it to
            ``spsa_grad`` in each call.

    Returns:
        qnode (QNode) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the Jacobian in the form of a tensor, a tuple, or a nested tuple depending upon the nesting
        structure of measurements in the original circuit.

    **Example**

    This gradient transform can be applied directly to :class:`QNode <pennylane.QNode>`
    objects:

    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     return qml.expval(qml.Z(0)), qml.var(qml.Z(0))
    >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    >>> qml.gradients.spsa_grad(circuit)(params)
    (tensor([ 0.18488771, -0.18488771, -0.18488771], requires_grad=True),
     tensor([-0.33357922,  0.33357922,  0.33357922], requires_grad=True))

    Note that the SPSA gradient is a statistical estimator that uses a given number of
    function evaluations that does not depend on the number of parameters. While this
    bounds the cost of the estimation, it also implies that the returned values are not
    exact (even with ``shots=None``) and that they will fluctuate.
    See the usage details below for more information.

    .. details::
        :title: Usage Details

        The number of directions in which the derivative is computed to estimate the gradient
        can be controlled with the keyword argument ``num_directions``. For the QNode above,
        a more precise gradient estimation from ``num_directions=20`` directions yields

        >>> qml.gradients.spsa_grad(circuit, num_directions=20)(params)
        (tensor([-0.53976776, -0.34385475, -0.46106048], requires_grad=True),
         tensor([0.97386303, 0.62039169, 0.83185731], requires_grad=True))

        We may compare this to the more precise values obtained from finite differences:

        >>> qml.gradients.finite_diff(circuit)(params)
        (tensor([-0.38751724, -0.18884792, -0.38355708], requires_grad=True),
         tensor([0.69916868, 0.34072432, 0.69202365], requires_grad=True))

        As we can see, the SPSA output is a rather coarse approximation to the true
        gradient, and this although the parameter-shift rule for three parameters uses
        just six circuit evaluations, much fewer than SPSA! Consequentially, SPSA is
        not necessarily useful for small circuits with few parameters, but will pay off
        for large circuits where other gradient estimators require unfeasibly many circuit
        executions.

        This quantum gradient transform can also be applied to low-level
        :class:`~.QuantumTape` objects. This will result in no implicit quantum
        device evaluation. Instead, the processed tapes, and post-processing
        function, which together define the gradient are directly returned:

        >>> ops = [qml.RX(params[0], 0), qml.RY(params[1], 0), qml.RX(params[2], 0)]
        >>> measurements = [qml.expval(qml.Z(0)), qml.var(qml.Z(0))]
        >>> tape = qml.tape.QuantumTape(ops, measurements)
        >>> gradient_tapes, fn = qml.gradients.spsa_grad(tape)
        >>> gradient_tapes
        [<QuantumScript: wires=[0], params=3>, <QuantumScript: wires=[0], params=3>]

        This can be useful if the underlying circuits representing the gradient
        computation need to be analyzed. Here we see that for ``num_directions=1``,
        the default, we obtain two tapes.

        Note that ``argnum`` refers to the index of a parameter within the list of trainable
        parameters. For example, if we have:

        >>> tape = qml.tape.QuantumScript(
        ...     [qml.RX(1.2, wires=0), qml.RY(2.3, wires=0), qml.RZ(3.4, wires=0)],
        ...     [qml.expval(qml.Z(0))],
        ...     trainable_params = [1, 2]
        ... )
        >>> qml.gradients.spsa_grad(tape, argnum=1)

        The code above will differentiate the third parameter rather than the second.

        The output tapes can then be evaluated and post-processed to retrieve
        the gradient:

        >>> dev = qml.device("default.qubit")
        >>> fn(qml.execute(gradient_tapes, dev, None))
        ((tensor(0.18488771, requires_grad=True),
          tensor(-0.18488771, requires_grad=True),
          tensor(-0.18488771, requires_grad=True)),
         (tensor(-0.33357922, requires_grad=True),
          tensor(0.33357922, requires_grad=True),
          tensor(0.33357922, requires_grad=True)))

        This gradient transform is compatible with devices that use shot vectors for execution.

        >>> shots = (10, 100, 1000)
        >>> dev = qml.device("default.qubit", shots=shots)
        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.Z(0)), qml.var(qml.Z(0))
        >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
        >>> qml.gradients.spsa_grad(circuit, h=1e-2)(params)
        ((array([ 10.,  10., -10.]), array([-18., -18.,  18.])),
         (array([-5., -5.,  5.]), array([ 8.9,  8.9, -8.9])),
         (array([ 1.5,  1.5, -1.5]), array([-2.667, -2.667,  2.667])))

        The outermost tuple contains results corresponding to each element of the shot vector,
        as is also visible by the increasing precision.
        Note that the stochastic approximation and the fluctuations from the shot noise
        of the device accumulate, leading to a very coarse-grained estimate for the gradient.
    """

    transform_name = "SPSA"
    assert_no_trainable_tape_batching(tape, transform_name)

    if argnum is None and not tape.trainable_params:
        return _no_trainable_grad(tape)

    trainable_params = choose_trainable_params(tape, argnum)
    diff_methods = (
        find_and_validate_gradient_methods(tape, "numeric", trainable_params)
        if validate_params
        else {idx: "F" for idx in trainable_params}
    )

    if all(g == "0" for g in diff_methods.values()):
        return _all_zero_grad(tape)

    gradient_tapes = []
    extract_r0 = False

    coeffs, shifts = finite_diff_coeffs(n=n, approx_order=approx_order, strategy=strategy)

    if 0 in shifts:
        # Finite difference formula includes a term with zero shift.

        if f0 is None:
            # Ensure that the unshifted tape is appended to the gradient tapes
            gradient_tapes.append(tape)
            extract_r0 = True

        # Skip the unshifted tape
        shifts = shifts[1:]

    if not isinstance(sampler_rng, (int, np.random.Generator, type(None))):
        raise ValueError(
            f"The argument sampler_rng is expected to be a NumPy PRNG, an integer or None, but is {sampler_rng}."
        )

    sampler_rng = np.random.default_rng(sampler_rng)

    num_trainable_params = len(tape.trainable_params)
    indices = [
        i for i in range(num_trainable_params) if (i in diff_methods and diff_methods[i] != "0")
    ]

    tapes_per_grad = len(shifts)
    all_coeffs = []
    for idx_rep in range(num_directions):
        direction = sampler(indices, num_trainable_params, idx_rep, rng=sampler_rng)
        # the `where` arg is being cast to list to avoid unexpected side effects from types that
        # override __array_ufunc__. See https://github.com/numpy/numpy/pull/23240 for more details
        inv_direction = qml.math.divide(
            1, direction, where=list(direction != 0), out=qml.math.zeros_like(direction)
        )
        # Use only the non-zero part of `direction` for the shifts, to skip redundant zero shifts
        _shifts = qml.math.tensordot(h * shifts, direction[indices], axes=0)
        all_coeffs.append(qml.math.tensordot(coeffs / h**n, inv_direction, axes=0))
        g_tapes = generate_multishifted_tapes(tape, indices, _shifts)
        gradient_tapes.extend(g_tapes)

    def _single_shot_batch_result(results):
        """Auxiliary function for post-processing one batch of results corresponding to finite
        shots or a single component of a shot vector"""

        r0, results = (results[0], results[1:]) if extract_r0 else (f0, results)
        num_measurements = len(tape.measurements)
        if num_measurements == 1:
            grads = 0
            for rep, _coeffs in enumerate(all_coeffs):
                res = list(results[rep * tapes_per_grad : (rep + 1) * tapes_per_grad])
                if r0 is not None:
                    res.insert(0, r0)
                res = qml.math.stack(res)
                grads = (
                    qml.math.tensordot(qml.math.convert_like(_coeffs, res), res, axes=[[0], [0]])
                    + grads
                )
            grads = grads * (1 / num_directions)
            if num_trainable_params == 1:
                return qml.math.convert_like(grads[0], res)
            return tuple(qml.math.convert_like(g, res) for g in grads)

        grads = []
        for i in range(num_measurements):
            grad = 0
            for rep, _coeffs in enumerate(all_coeffs):
                res = [r[i] for r in results[rep * tapes_per_grad : (rep + 1) * tapes_per_grad]]
                if r0 is not None:
                    res.insert(0, r0[i])
                res = qml.math.stack(res)
                grad = (
                    qml.math.tensordot(qml.math.convert_like(_coeffs, res), res, axes=[[0], [0]])
                    + grad
                )
            grad = grad / num_directions
            grads.append(tuple(qml.math.convert_like(g, grad) for g in grad))

        if num_trainable_params == 1:
            return tuple(g[0] for g in grads)
        return tuple(grads)

    processing_fn = partial(
        _processing_fn, shots=tape.shots, single_shot_batch_fn=_single_shot_batch_result
    )

    return gradient_tapes, processing_fn
