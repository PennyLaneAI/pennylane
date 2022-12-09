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
# pylint: disable=protected-access,too-many-arguments,too-many-branches,too-many-statements
import warnings
from collections.abc import Sequence

import numpy as np

import pennylane as qml
from pennylane._device import _get_num_copies

from .finite_difference import (
    _all_zero_grad_new,
    _no_trainable_grad_new,
    finite_diff_coeffs,
)
from .gradient_transform import (
    gradient_transform,
    grad_method_validation,
    choose_grad_methods,
    gradient_analysis,
)
from .general_shift_rules import generate_multishifted_tapes


def _rademacher_sampler(indices, num_params, *args, seed=None):
    r"""Sample a random vector with (independent) entries from {+1, -1} with balanced probability.
    That is, each entry follows the
    `Rademacher distribution. <https://en.wikipedia.org/wiki/Rademacher_distribution>`_

    The signature corresponds to the one required for the input ``sampler`` to ``spsa_grad``:

    Args:
        indices (Sequence[int]): Indices of the trainable tape parameters that will be perturbed.
        num_params (int): Total number of trainable tape parameters.

    Returns:
        tensor_like: Vector of size ``num_params`` with non-zero entries at positions indicated
        by ``indices``, each entry sampled independently from the Rademacher distribution.
    """
    # pylint: disable=unused-argument
    if seed is not None:
        np.random.seed(seed)
    direction = np.zeros(num_params)
    direction[indices] = np.random.choice([-1, 1], size=len(indices))
    return direction


@gradient_transform
def _spsa_grad_new(
    tape,
    argnum=None,
    h=1e-5,
    approx_order=2,
    n=1,
    strategy="center",
    f0=None,
    validate_params=True,
    shots=None,
    num_directions=1,
    sampler=_rademacher_sampler,
    sampler_seed=None,
):
    r"""Transform a QNode to compute the SPSA gradient of all gate
    parameters with respect to its inputs. This estimator shifts all parameters
    simultaneously and approximates the gradient based on these shifts and a
    finite-difference method. This function is adapted to the new return system.

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to differentiate
        argnum (int or list[int] or None): Trainable parameter indices to differentiate
            with respect to. If not provided, the derivatives with respect to all
            trainable parameters are returned.
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
        shots (None, int, list[int], list[ShotTuple]): The device shots that will be used to
            execute the tapes outputted by this transform. Note that this argument doesn't
            influence the shots used for tape execution, but provides information
            to the transform about the device shots and helps in determining if a shot
            sequence was used to define the device shots for the new return types output system.
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

            - The keyword argument ``seed``, expected to be ``None`` or an ``int``.
              This argument should be passed to some method that seeds any randomness used in
              the sampler.

            Note that the circuit evaluations in the various sampled directions are *averaged*,
            not simply summed up.

        sampler_seed (int or None): Seed passed to ``sampler``. The seed is passed in each
            call to the sampler, so that only one unique direction is sampled even if
            ``num_directions>1``.

    Returns:
        function or tuple[list[QuantumTape], function]:

        - If the input is a QNode, an object representing the Jacobian (function) of the QNode
          that can be executed to obtain the Jacobian matrix.
          The type of the matrix returned is either a tensor, a tuple or a
          nested tuple depending on the nesting structure of the original QNode output.

        - If the input is a tape, a tuple containing a
          list of generated tapes, together with a post-processing
          function to be applied to the results of the evaluated tapes
          in order to obtain the Jacobian matrix.

    **Example**

    This gradient transform can be applied directly to :class:`QNode <pennylane.QNode>`
    objects:

    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))
    >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    >>> qml.gradients.spsa_grad(circuit)(params)
    ((tensor(-0.19280803, requires_grad=True),
      tensor(-0.19280803, requires_grad=True),
      tensor(0.19280803, requires_grad=True)),
     (tensor(0.34786926, requires_grad=True),
      tensor(0.34786926, requires_grad=True),
      tensor(-0.34786926, requires_grad=True)))

    Note that the SPSA gradient is a statistical estimator that uses a given number of
    function evaluations that does not depend on the number of parameters. While this
    bounds the cost of the estimation, it also implies that the returned values are not
    exact (even for devices with ``shots=None``) and that they will fluctuate.
    See the usage details below for more information.

    .. details::
        :title: Usage Details

        The number of directions in which the derivative is computed to estimate the gradient
        can be controlled with the keyword argument ``num_directions``. For the QNode above,
        a more precise gradient estimation from ``num_directions=20`` directions yields

        >>> qml.gradients.spsa_grad(circuit, num_directions=20)(params)
        ((tensor(-0.27362235, requires_grad=True),
          tensor(-0.07219669, requires_grad=True),
          tensor(-0.36369011, requires_grad=True)),
         (tensor(0.49367656, requires_grad=True),
          tensor(0.13025915, requires_grad=True),
          tensor(0.65617915, requires_grad=True)))

        We may compare this to the more precise values obtained from finite differences:

        >>> qml.gradients.finite_diff(circuit)(params)
        ((tensor(-0.38751724, requires_grad=True),
          tensor(-0.18884792, requires_grad=True),
          tensor(-0.38355708, requires_grad=True)),
         (tensor(0.69916868, requires_grad=True),
          tensor(0.34072432, requires_grad=True),
          tensor(0.69202365, requires_grad=True)))

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

        >>> with qml.tape.QuantumTape() as tape:
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     qml.expval(qml.PauliZ(0))
        ...     qml.var(qml.PauliZ(0))
        >>> gradient_tapes, fn = qml.gradients.spsa_grad(tape)
        >>> gradient_tapes
        [<QuantumTape: wires=[0], params=3>, <QuantumTape: wires=[0], params=3>]

        This can be useful if the underlying circuits representing the gradient
        computation need to be analyzed. Here we see that for ``num_directions=1``,
        the default, we obtain two tapes.

        The output tapes can then be evaluated and post-processed to retrieve
        the gradient:

        >>> dev = qml.device("default.qubit", wires=2)
        >>> fn(qml.execute(gradient_tapes, dev, None))
        ((array(-0.58222637), array(0.58222637), array(-0.58222637)),
         (array(1.05046797), array(-1.05046797), array(1.05046797)))


        Devices that have a shot vector defined can also be used for execution, provided
        the ``shots`` argument was passed to the transform:

        >>> shots = (10, 100, 1000)
        >>> dev = qml.device("default.qubit", wires=2, shots=shots)
        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))
        >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
        >>> qml.gradients.spsa_grad(circuit, shots=shots, h=1e-2)(params)
        (((array(0.), array(0.), array(0.)), (array(0.), array(0.), array(0.))),
         ((array(-1.4), array(-1.4), array(-1.4)),
          (array(2.548), array(2.548), array(2.548))),
         ((array(-1.06), array(-1.06), array(-1.06)),
          (array(1.90588), array(1.90588), array(1.90588))))

        The outermost tuple contains results corresponding to each element of the shot vector,
        as is also visible by the increasing precision.
        Note that the stochastic approximation and the fluctuations from the shot noise
        of the device accumulate, leading to a very coarse-grained estimate for the gradient.
    """
    if argnum is None and not tape.trainable_params:
        return _no_trainable_grad_new(tape, shots)

    if validate_params:
        if "grad_method" not in tape._par_info[0]:
            gradient_analysis(tape, grad_fn=_spsa_grad_new)
        diff_methods = grad_method_validation("numeric", tape)
    else:
        diff_methods = ["F" for i in tape.trainable_params]

    if all(g == "0" for g in diff_methods):
        return _all_zero_grad_new(tape, shots)

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

    method_map = choose_grad_methods(diff_methods, argnum)

    num_trainable_params = len(tape.trainable_params)
    indices = [i for i in range(num_trainable_params) if (i in method_map and method_map[i] != "0")]

    tapes_per_grad = len(shifts)
    all_coeffs = []
    for idx_rep in range(num_directions):
        direction = sampler(indices, num_trainable_params, idx_rep, seed=sampler_seed)
        inv_direction = qml.math.divide(
            1, direction, where=(direction != 0), out=qml.math.zeros_like(direction)
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
                res = results[rep * tapes_per_grad : (rep + 1) * tapes_per_grad]
                if r0 is not None:
                    res.insert(0, r0)
                res = qml.math.stack(res)
                grads = (
                    qml.math.tensordot(qml.math.convert_like(_coeffs, res), res, axes=[[0], [0]])
                    + grads
                )
            grads = grads / num_directions
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
            return grads[0]
        return tuple(grads)

    def processing_fn(results):
        shot_vector = isinstance(shots, Sequence)

        if not shot_vector:
            grads_tuple = _single_shot_batch_result(results)
        else:
            grads_tuple = []
            len_shot_vec = _get_num_copies(shots)
            for idx in range(len_shot_vec):
                res = [tape_res[idx] for tape_res in results]
                g_tuple = _single_shot_batch_result(res)
                grads_tuple.append(g_tuple)
            grads_tuple = tuple(grads_tuple)

        return grads_tuple

    return gradient_tapes, processing_fn


@gradient_transform
def spsa_grad(
    tape,
    argnum=None,
    h=1e-5,
    approx_order=2,
    n=1,
    strategy="center",
    f0=None,
    validate_params=True,
    shots=None,
    num_directions=1,
    sampler=_rademacher_sampler,
    sampler_seed=None,
):
    r"""Transform a QNode to compute the SPSA gradient of all gate
    parameters with respect to its inputs. This estimator shifts all parameters
    simultaneously and approximates the gradient based on these shifts and a
    finite-difference method.

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to differentiate
        argnum (int or list[int] or None): Trainable parameter indices to differentiate
            with respect to. If not provided, the derivatives with respect to all
            trainable parameters are returned.
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
        shots (None, int, list[int], list[ShotTuple]): The device shots that will be used to
            execute the tapes outputted by this transform. Note that this argument doesn't
            influence the shots used for tape execution, but provides information
            to the transform about the device shots and helps in determining if a shot
            sequence was used to define the device shots for the new return types output system.
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

            - The keyword argument ``seed``, expected to be ``None`` or an ``int``.
              This argument should be passed to some method that seeds any randomness used in
              the sampler.

        sampler_seed (int or None): Seed passed to ``sampler``. The seed is passed in each
            call to the sampler, so that only one unique direction is sampled even if
            ``num_directions>1``.

    Returns:
        function or tuple[list[QuantumTape], function]:

        - If the input is a QNode, an object representing the Jacobian (function) of the QNode
          that can be executed to obtain the Jacobian matrix.
          The returned matrix is a tensor of size ``(number_outputs, number_gate_parameters)``

        - If the input is a tape, a tuple containing a
          list of generated tapes, together with a post-processing
          function to be applied to the results of the evaluated tapes
          in order to obtain the Jacobian matrix.

    **Example**

    This gradient transform can be applied directly to :class:`QNode <pennylane.QNode>`
    objects:

    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))
    >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    >>> qml.gradients.spsa_grad(circuit)(params)
    tensor([[-0.19280803, -0.19280803,  0.19280803],
            [ 0.34786926,  0.34786926, -0.34786926]], requires_grad=True)

    Note that the SPSA gradient is a statistical estimator that uses a given number of
    function evaluations that does not depend on the number of parameters. While this
    bounds the cost of the estimation, it also implies that the returned values are not
    exact (even for devices with ``shots=None``) and that they will fluctuate.
    See the usage details below for more information.

    .. details::
        :title: Usage Details

        The number of directions in which the derivative is computed to estimate the gradient
        can be controlled with the keyword argument ``num_directions``. For the QNode above,
        a more precise gradient estimation from ``num_directions=20`` directions yields

        >>> qml.gradients.spsa_grad(circuit, num_directions=20)(params)
        tensor([[-0.32969058, -0.18924389, -0.28716881],
                [ 0.59483632,  0.34143874,  0.51811744]], requires_grad=True)

        We may compare this to the exact value obtained from parameter-shift rules:

        >>> qml.gradients.param_shift(circuit)(params)
        tensor([[-0.3875172 , -0.18884787, -0.38355704],
                [ 0.69916862,  0.34072424,  0.69202359]], requires_grad=True)

        As we can see, the SPSA output is a rather coarse approximation to the true
        gradient. This means we used more circuit evaluations (namely 20) to
        obtain a rough approximation than for the more precise parameter-shift
        result (which only cost us six evaluations).
        For few parameters, this will usually be the case, so that SPSA is not
        very useful for few-parameter gradients. However, for circuits with
        considerably more parameters, the parameter-shift gradient will require
        many more evaluations, while the number of directions we need for a proper
        approximation of the gradient will not grow as much. This makes SPSA
        useful for such circuits.

        This quantum gradient transform can also be applied to low-level
        :class:`~.QuantumTape` objects. This will result in no implicit quantum
        device evaluation. Instead, the processed tapes, and post-processing
        function, which together define the gradient are directly returned:

        >>> with qml.tape.QuantumTape() as tape:
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     qml.expval(qml.PauliZ(0))
        ...     qml.var(qml.PauliZ(0))
        >>> gradient_tapes, fn = qml.gradients.spsa_grad(tape)
        >>> gradient_tapes
        [<QuantumTape: wires=[0], params=3>, <QuantumTape: wires=[0], params=3>]

        This can be useful if the underlying circuits representing the gradient
        computation need to be analyzed. Here we see that for ``num_directions=1``,
        the default, we obtain two tapes.

        The output tapes can then be evaluated and post-processed to retrieve
        the gradient:

        >>> dev = qml.device("default.qubit", wires=2)
        >>> fn(qml.execute(gradient_tapes, dev, None))
        array([[-0.95992212, -0.95992212, -0.95992212],
               [ 1.73191645,  1.73191645,  1.73191645]])
    """
    if qml.active_return():
        return _spsa_grad_new(
            tape,
            argnum=argnum,
            h=h,
            approx_order=approx_order,
            n=n,
            strategy=strategy,
            f0=f0,
            validate_params=validate_params,
            shots=shots,
            num_directions=num_directions,
            sampler=sampler,
            sampler_seed=sampler_seed,
        )

    if argnum is None and not tape.trainable_params:
        warnings.warn(
            "Attempted to compute the gradient of a tape with no trainable parameters. "
            "If this is unintended, please mark trainable parameters in accordance with the "
            "chosen auto differentiation framework, or via the 'tape.trainable_params' property."
        )
        return [], lambda _: qml.math.zeros([tape.output_dim, 0])

    if validate_params:
        if "grad_method" not in tape._par_info[0]:
            gradient_analysis(tape, grad_fn=spsa_grad)
        diff_methods = grad_method_validation("numeric", tape)
    else:
        diff_methods = ["F" for i in tape.trainable_params]

    num_trainable_params = len(tape.trainable_params)
    if all(g == "0" for g in diff_methods):
        return [], lambda _: np.zeros([tape.output_dim, num_trainable_params])

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

    method_map = choose_grad_methods(diff_methods, argnum)

    indices = [i for i in range(num_trainable_params) if (i in method_map and method_map[i] != "0")]

    tapes_per_grad = len(shifts)
    all_coeffs = []
    for idx_rep in range(num_directions):
        direction = sampler(indices, num_trainable_params, idx_rep, seed=sampler_seed)
        inv_direction = qml.math.divide(
            1, direction, where=(direction != 0), out=qml.math.zeros_like(direction)
        )
        # Use only the non-zero part of `direction` for the shifts, to skip redundant zero shifts
        _shifts = qml.math.tensordot(h * shifts, direction[indices], axes=0)
        all_coeffs.append(qml.math.tensordot(coeffs / h**n, inv_direction, axes=0))
        g_tapes = generate_multishifted_tapes(tape, indices, _shifts)
        gradient_tapes.extend(g_tapes)

    def processing_fn(results):
        # HOTFIX: Apply the same squeezing as in qml.QNode to make the transform output consistent.
        # pylint: disable=protected-access
        if tape._qfunc_output is not None and not isinstance(tape._qfunc_output, Sequence):
            results = [qml.math.squeeze(res) for res in results]

        r0, results = (results[0], results[1:]) if extract_r0 else (f0, results)

        grads = 0
        for rep, _coeffs in enumerate(all_coeffs):
            res = results[rep * tapes_per_grad : (rep + 1) * tapes_per_grad]
            if r0 is not None:
                res.insert(0, r0)
            res = qml.math.stack(res)
            grads = (
                qml.math.tensordot(res, qml.math.convert_like(_coeffs, res), axes=[[0], [0]])
                + grads
            )

        grads = grads / num_directions

        # The following is for backwards compatibility; currently,
        # the device stacks multiple measurement arrays, even if not the same
        # size, resulting in a ragged array. (-> new return type system)
        if hasattr(grads, "dtype") and grads.dtype is np.dtype("object"):
            grads = qml.math.moveaxis(
                qml.math.array([qml.math.hstack(gs) for gs in zip(*grads)]), 0, -1
            )

        return grads

    return gradient_tapes, processing_fn
