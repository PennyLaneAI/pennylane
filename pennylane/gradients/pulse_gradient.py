# Copyright 2023 Xanadu Quantum Technologies Inc.

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
This module contains functions for computing the stochastic parameter-shift gradient
of pulse sequences in a qubit-based quantum tape.
"""
from typing import Sequence, Callable
from functools import partial
import warnings
import numpy as np

import pennylane as qml
from pennylane.pulse import ParametrizedEvolution, HardwareHamiltonian
from pennylane import transform

from .parameter_shift import _make_zero_rep
from .general_shift_rules import eigvals_to_frequencies, generate_shift_rule
from .gradient_transform import (
    _all_zero_grad,
    assert_no_state_returns,
    assert_no_trainable_tape_batching,
    assert_no_variance,
    choose_trainable_params,
    find_and_validate_gradient_methods,
    _no_trainable_grad,
    reorder_grads,
)

has_jax = True
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    has_jax = False


def _assert_has_jax(transform_name):
    """Check that JAX is installed and imported correctly, otherwise raise an error.

    Args:
        transform_name (str): Name of the gradient transform that queries the return system
    """
    if not has_jax:  # pragma: no cover
        raise ImportError(
            f"Module jax is required for the {transform_name} gradient transform. "
            "You can install jax via: pip install jax jaxlib"
        )


def raise_pulse_diff_on_qnode(transform_name):
    """Raises an error as the gradient transform with the provided name does
    not support direct application to QNodes.
    """
    msg = (
        f"Applying the {transform_name} gradient transform to a QNode directly is currently "
        "not supported. Please use differentiation via a JAX entry point "
        "(jax.grad, jax.jacobian, ...) instead.",
        UserWarning,
    )
    raise NotImplementedError(msg)


def _split_evol_ops(op, ob, tau):
    r"""Randomly split a ``ParametrizedEvolution`` with respect to time into two operations and
    insert a Pauli rotation using a given Pauli word and rotation angles :math:`\pm\pi/2`.
    This yields two groups of three operations each.

    Args:
        op (ParametrizedEvolution): operation to split up.
        ob (`~.Operator`): generating Hamiltonian term to insert the parameter-shift rule for.
        tau (float or tensor_like): split-up time(s). If multiple times are passed, the split-up
            operations are set up to return intermediate time evolution results, leading to
            broadcasting effectively.

    Returns:
        tuple[list[`~.Operation`]]: The split-time evolution, expressed as three operations in the
            inner lists. The number of tuples is given by the number of shifted terms in the
            parameter-shift rule of the generating Hamiltonian term ``ob``.
        tensor_like: Coefficients of the parameter-shift rule of the provided generating Hamiltonian
            term ``ob``.
    """
    t0, *_, t1 = op.t
    # If there are multiple values for tau, use broadcasting
    if bcast := qml.math.ndim(tau) > 0:
        # With broadcasting, create a sorted array of [t_0, *sorted(taus), t_1]
        # Use this array for both, the pulse before and after the inserted operation.
        # The way we slice the resulting tape results later on accomodates for the additional
        # time points t_0 and t_1 in the array.
        tau = jnp.sort(tau)
        before_t = jnp.concatenate([jnp.array([t0]), tau, jnp.array([t1])])
        after_t = before_t.copy()
    else:
        # Create a time interval from start to split and one from split to end
        before_t = jax.numpy.array([t0, tau])
        after_t = jax.numpy.array([tau, t1])

    if qml.pauli.is_pauli_word(ob):
        prefactor = qml.pauli.pauli_word_prefactor(ob)
        word = qml.pauli.pauli_word_to_string(ob)
        insert_ops = [qml.PauliRot(shift, word, ob.wires) for shift in [np.pi / 2, -np.pi / 2]]
        coeffs = [prefactor, -prefactor]
    else:
        with warnings.catch_warnings():
            if len(ob.wires) <= 4:
                warnings.filterwarnings(
                    "ignore", ".*the eigenvalues will be computed numerically.*"
                )
            eigvals = qml.eigvals(ob)
        coeffs, shifts = zip(*generate_shift_rule(eigvals_to_frequencies(tuple(eigvals))))
        insert_ops = [qml.exp(qml.dot([-1j * shift], [ob])) for shift in shifts]

    # Create Pauli rotations to be inserted at tau
    ode_kwargs = op.odeint_kwargs
    # If we are broadcasting, make use of the `return_intermediate` and `complementary` features
    ops = tuple(
        [
            op(op.data, before_t, return_intermediate=bcast, **ode_kwargs),
            insert_op,
            op(op.data, after_t, return_intermediate=bcast, complementary=bcast, **ode_kwargs),
        ]
        for insert_op in insert_ops
    )
    return ops, jnp.array(coeffs)


def _split_evol_tape(tape, split_evolve_ops, op_idx):
    """Replace a marked ``ParametrizedEvolution`` in a given tape by provided operations, creating
    one tape per group of operations.

    Args:
        tape (QuantumTape): original tape
        split_evolve_ops (tuple[list[qml.Operation]]): The time-split evolution operations as
            created by ``_split_evol_ops``. For each group of operations, a new tape
            is created.
        op_idx (int): index of the operation to replace within the tape

    Returns:
        list[QuantumTape]: new tapes with replaced operation, one tape per group of operations in
        ``split_evolve_ops``.
    """
    ops_pre = tape.operations[:op_idx]
    ops_post = tape.operations[op_idx + 1 :]
    return [
        qml.tape.QuantumScript(ops_pre + split + ops_post, tape.measurements, shots=tape.shots)
        for split in split_evolve_ops
    ]


# pylint: disable=too-many-arguments
def _parshift_and_integrate(
    results,
    cjacs,
    int_prefactor,
    psr_coeffs,
    single_measure,
    has_partitioned_shots,
    use_broadcasting,
):
    """Apply the parameter-shift rule post-processing to tape results and contract
    with classical Jacobians, effectively evaluating the numerical integral of the stochastic
    parameter-shift rule.

    Args:
        results (list): Tape evaluation results, corresponding to the modified quantum
            circuit result when using the applicable parameter shifts and the sample splitting
            times. Results should be ordered such that the different shifted circuits for a given
            splitting time are grouped together
        cjacs (tensor_like): classical Jacobian evaluated at the splitting times
        int_prefactor (float): prefactor of the numerical integration, corresponding to the size
            of the time range divided by the number of splitting time samples
        psr_coeffs (tensor_like or tuple[tensor_like]): Coefficients of the parameter-shift
            rule to contract the results with before integrating numerically.
        single_measure (bool): Whether the results contain a single measurement per shot setting
        has_partitioned_shots (bool): Whether the results have a shot vector axis
        use_broadcasting (bool): Whether broadcasting was used in the tapes that returned the
            ``results``.
    Returns:
        tensor_like or tuple[tensor_like] or tuple[tuple[tensor_like]]: Gradient entry
    """

    def _contract(coeffs, res, cjac):
        """Contract three tensors, the first two like a standard matrix multiplication
        and the result with the third tensor along the first axes."""
        return jnp.tensordot(jnp.tensordot(coeffs, res, axes=1), cjac, axes=[[0], [0]])

    if isinstance(psr_coeffs, tuple):
        num_shifts = [len(c) for c in psr_coeffs]

        def _psr_and_contract(res_list, cjacs, int_prefactor):
            """Execute the parameter-shift rule and contract with classical Jacobians.
            This function assumes multiple generating terms for the pulse parameter
            of interest"""
            res = jnp.stack(res_list)
            idx = 0

            # Preprocess the results: Reshape, create slices for different generating terms
            if use_broadcasting:
                # Slice the results according to the different generating terms. Slice away the
                # first and last value for each term, which correspond to the initial condition
                # and the final value of the time evolution, but not to splitting times
                res = tuple(res[idx : (idx := idx + n), 1:-1] for n in num_shifts)
            else:
                shape = jnp.shape(res)
                num_taus = shape[0] // sum(num_shifts)
                # Reshape the slices of the results corresponding to different generating terms.
                # Afterwards the first axis corresponds to the splitting times and the second axis
                # corresponds to the different shifts of the respective term.
                # Finally move the shifts-axis to the first position of each term.
                res = tuple(
                    jnp.moveaxis(
                        jnp.reshape(
                            res[idx : (idx := idx + n * num_taus)], (num_taus, n) + shape[1:]
                        ),
                        1,
                        0,
                    )
                    for n in num_shifts
                )

            # Contract the results, parameter-shift rule coefficients and (classical) Jacobians,
            # and include the rescaling factor from the Monte Carlo integral and from global
            # prefactors of Pauli word generators.
            diff_per_term = jnp.array(
                [_contract(c, r, cjac) for c, r, cjac in zip(psr_coeffs, res, cjacs)]
            )
            return qml.math.sum(diff_per_term, axis=0) * int_prefactor

    else:
        num_shifts = len(psr_coeffs)

        def _psr_and_contract(res_list, cjacs, int_prefactor):
            """Execute the parameter-shift rule and contract with classical Jacobians.
            This function assumes a single generating term for the pulse parameter
            of interest"""
            res = jnp.stack(res_list)

            # Preprocess the results: Reshape, create slices for different generating terms
            if use_broadcasting:
                # Slice away the first and last values, corresponding to the initial condition
                # and the final value of the time evolution, but not to splitting times
                res = res[:, 1:-1]
            else:
                # Reshape the results such that the first axis corresponds to the splitting times
                # and the second axis corresponds to different shifts. All other axes are untouched.
                # Afterwards move the shifts-axis to the first position.
                shape = jnp.shape(res)
                new_shape = (shape[0] // num_shifts, num_shifts) + shape[1:]
                res = jnp.moveaxis(jnp.reshape(res, new_shape), 1, 0)

            # Contract the results, parameter-shift rule coefficients and (classical) Jacobians,
            # and include the rescaling factor from the Monte Carlo integral and from global
            # prefactors of Pauli word generators.
            return _contract(psr_coeffs, res, cjacs) * int_prefactor

    nesting_layers = (not single_measure) + has_partitioned_shots
    if nesting_layers == 1:
        return tuple(_psr_and_contract(r, cjacs, int_prefactor) for r in zip(*results))
    if nesting_layers == 0:
        # Single measurement without shot vector
        return _psr_and_contract(results, cjacs, int_prefactor)

    # Multiple measurements with shot vector. Not supported with broadcasting yet.
    if use_broadcasting:
        # TODO: Remove once #2690 is resolved
        raise NotImplementedError(
            "Broadcasting, multiple measurements and shot vectors are currently not "
            "supported all simultaneously by stoch_pulse_grad."
        )
    return tuple(
        tuple(_psr_and_contract(_r, cjacs, int_prefactor) for _r in zip(*r)) for r in zip(*results)
    )


# pylint: disable=too-many-arguments
@partial(transform, final_transform=True)
def stoch_pulse_grad(
    tape: qml.tape.QuantumTape,
    argnum=None,
    num_split_times=1,
    sampler_seed=None,
    use_broadcasting=False,
) -> (Sequence[qml.tape.QuantumTape], Callable):
    r"""Compute the gradient of a quantum circuit composed of pulse sequences by applying the
    stochastic parameter shift rule.

    For a pulse-based cost function :math:`C(\boldsymbol{v}, T)`
    with variational parameters :math:`\boldsymbol{v}` and evolution time :math:`T`, it is given by
    (c.f. Eqn. (6) in `Leng et al. (2022) <https://arxiv.org/abs/2210.15812>`__ with altered
    notation):

    .. math::

        \frac{\partial C}{\partial v_k}
        = \int_{0}^{T} \mathrm{d}\tau \sum_{j=1}^m
        \frac{\partial f_j}{\partial v_k}(\boldsymbol{v}, \tau)
        \left[C_j^{(+)}(\boldsymbol{v}, \tau) - C_j^{(-)}(\boldsymbol{v}, \tau)\right]

    Here, :math:`f_j` are the pulse envelopes that capture the time dependence of the pulse
    Hamiltonian:

    .. math::

        H(\boldsymbol{v}, t) = H_\text{drift} + \sum_j f_j(\boldsymbol{v}, t) H_j,

    and :math:`C_j^{(\pm)}` are modified cost functions:

    .. math::
            C_j^{(\pm)}(\boldsymbol{v}, \tau)&=
            \bra{\psi^{(\pm)}_{j}(\boldsymbol{v}, \tau)} B
            \ket{\psi^{(\pm)}_{j}(\boldsymbol{v}, \tau)} \\
            \ket{\psi^{(\pm)}_{j}(\boldsymbol{v}, \tau)}
            &= U_{\boldsymbol{v}}(T, \tau) e^{-i (\pm \frac{\pi}{4}) H_j}
            U_{\boldsymbol{v}}(\tau, 0)\ket{\psi_0}.

    That is, the :math:`j`\ th modified time evolution in these circuit interrupts the
    evolution generated by the pulse Hamiltonian by inserting a rotation gate generated by
    the corresponding Hamiltonian term :math:`H_j` with a rotation angle of
    :math:`\pm\frac{\pi}{4}`.

    See below for a more detailed description. The integral in the first equation above
    is estimated numerically in the stochastic parameter-shift rule. For this, it samples
    split times :math:`\tau` and averages the modified cost functions and the Jacobians
    of the envelopes :math:`\partial f_j / \partial v_k` at the sampled times suitably.

    Args:
        tape (QuantumTape): quantum circuit to differentiate
        argnum (int or list[int] or None): Trainable tape parameter indices to differentiate
            with respect to. If not provided, the derivatives with respect to all
            trainable parameters are returned. Note that the indices are with respect to
            the list of trainable parameters.
        num_split_times (int): number of time samples to use in the stochastic parameter-shift
            rule underlying the differentiation; also see details
        sample_seed (int): randomness seed to be used for the time samples in the stochastic
            parameter-shift rule
        use_broadcasting (bool): Whether to use broadcasting across the different sampled
            splitting times. If ``False`` (the default), one set of modified tapes per
            splitting time is created, if ``True`` only a single set of broadcasted, modified
            tapes is created, increasing performance on simulators.

    Returns:
        tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the Jacobian in the form of a tensor, a tuple, or a nested tuple depending upon the nesting
        structure of measurements in the original circuit.

    This transform realizes the stochastic parameter-shift rule for pulse sequences, as introduced
    in `Banchi and Crooks (2018) <https://quantum-journal.org/papers/q-2021-01-25-386/>`_ and
    `Leng et al. (2022) <https://arxiv.org/abs/2210.15812>`_.

    .. note::

        This function requires the JAX interface and does not work with other autodiff interfaces
        commonly encountered with PennyLane.
        Finally, this transform is not JIT-compatible yet.

    .. note::

        This function uses a basic sampling approach with a uniform distribution to estimate the
        integral appearing in the stochastic parameter-shift rule. In many cases, there are
        probability distributions that lead to smaller variances of the estimator.
        In addition, the sampling approach will not reduce trivially to simpler parameter-shift
        rules when used with simple pulses (see details and examples below), potentially leading
        to imprecise results and/or unnecessarily large computational efforts.

    .. warning::

        This transform may not be applied directly to QNodes. Use JAX entrypoints
        (``jax.grad``, ``jax.jacobian``, ...) instead or apply the transform on the tape level.
        Also see the examples below.

    **Examples**

    Consider a pulse program with a single two-qubit pulse, generated by a Hamiltonian
    with three terms: the non-trainable term :math:`\frac{1}{2}X_0`, the trainable
    constant (over time) term :math:`v_1 Z_0 Z_1` and the trainable sinoidal term
    :math:`\sin(v_2 t) (\frac{1}{5} Y_0 + \frac{7}{10} X_1)`.

    .. code-block:: python

        jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit.jax")

        def sin(p, t):
            return jax.numpy.sin(p * t)

        ZZ = qml.Z(0) @ qml.Z(1)
        Y_plus_X = qml.dot([1/5, 3/5], [qml.Y(0), qml.X(1)])
        H = 0.5 * qml.X(0) + qml.pulse.constant * ZZ + sin * Y_plus_X

        def ansatz(params):
            qml.evolve(H)(params, (0.2, 0.4))
            return qml.expval(qml.Y(1))

        qnode = qml.QNode(ansatz, dev, interface="jax", diff_method=qml.gradients.stoch_pulse_grad)

    The program takes the two parameters :math:`v_1, v_2` for the two trainable terms:

    >>> params = [jax.numpy.array(0.4), jax.numpy.array(1.3)]
    >>> qnode(params)
    Array(-0.0905377, dtype=float64)

    And as we registered the differentiation method :func:`~.stoch_pulse_grad`,
    we can compute its gradient in a hardware compatible manner:

    >>> jax.grad(qnode)(params)
    [Array(0.00109782, dtype=float64, weak_type=True),
     Array(-0.05833371, dtype=float64, weak_type=True)] # results may differ

    Note that the derivative is computed using a stochastic parameter-shift rule,
    which is based on a sampled approximation of an integral expression (see theoretical
    background below). This makes the computed derivative an approximate quantity subject
    to statistical fluctuations with notable variance. The number of samples used to
    approximate the integral can be chosen with ``num_split_times``, the seed for the
    sampling can be fixed with ``sampler_seed``:

    .. code-block:: python

        qnode = qml.QNode(
            ansatz,
            dev,
            interface="jax",
            diff_method=qml.gradients.stoch_pulse_grad,
            num_split_times=5, # Use 5 samples for the approximation
            sampler_seed=18, # Fix randomness seed
        )

    >>> jax.grad(qnode)(params)
    [Array(0.00207256, dtype=float64, weak_type=True),
     Array(-0.05989856, dtype=float64, weak_type=True)]

    We may activate the option ``use_broadcasting`` to improve the performance when running
    on classical simulators. Internally, it reuses intermediate results of the time evolution.
    We can compare the performance with a simple test:

    .. code-block:: python

        from time import process_time
        faster_grad_qnode = qml.QNode(
            ansatz,
            dev,
            interface="jax",
            diff_method=qml.gradients.stoch_pulse_grad,
            num_split_times=5, # Use 5 samples for the approximation
            sampler_seed=18, # Fix randomness seed
            use_broadcasting=True, # Activate broadcasting
        )
        times = []
        for node in [qnode, faster_grad_qnode]:
            start = process_time()
            jax.grad(node)(params)
            times.append(process_time() - start)

    >>> print(times) # Show the gradient computation times in seconds.
    [55.75785480000002, 12.297400500000009]

    .. warning::

        As the option ``use_broadcasting=True`` adds a broadcasting dimension to the modified
        circuits, it is not compatible with circuits that already are broadcasted.

    .. details::
        :title: Theoretical background
        :href: theory

        Consider a pulse generated by a time-dependent Hamiltonian

        .. math::

            H(\boldsymbol{v}, t) = H_\text{drift} + \sum_j f_j(v_j, t) H_j,

        where :math:`\boldsymbol{v}=\{v_j\}` are variational parameters and :math:`t` is the time.
        In addition, consider a cost function that is based on using this pulse for
        a duration :math:`T`
        in a pulse sequence and measuring the expectation value of an observable.
        For simplicity we absorb the parts of the sequence
        before and after the considered pulse into the initial state and the observable,
        respectively:

        .. math::

            C(\boldsymbol{v}, t) =
            \bra{\psi_0} U_{\boldsymbol{v}}(T, 0)^\dagger B U_{\boldsymbol{v}}(T, 0)\ket{\psi_0}.

        Here, we denoted the unitary evolution under :math:`H(\boldsymbol{v}, t)` from time
        :math:`t_1` to :math:`t_2` as :math:`U_{\boldsymbol{v}(t_2, t_1)}`.
        Then the derivative of :math:`C` with respect to a specific parameter :math:`v_k`
        is given by (see Eqn. (6) of `Leng et al. (2022) <https://arxiv.org/abs/2210.15812>`_)

        .. math::

            \frac{\partial C}{\partial v_k}
            = \int_{0}^{T} \mathrm{d}\tau \sum_{j=1}^m
            \frac{\partial f_j}{\partial v_k}(\boldsymbol{v}, \tau)
            \widetilde{C_j}(\boldsymbol{v}, \tau).

        Here, the integral ranges over the duration of the pulse, the partial derivatives of
        the coefficient functions, :math:`\partial f_j / \partial v_k`, are computed classically,
        and :math:`\widetilde{C_j}` is a linear combination of the results from modified pulse
        sequence executions based on generalized parameter-shift rules
        (see e.g. `Kyriienko and Elfving (2022) <https://arxiv.org/abs/2108.01218>`_ or
        `Wierichs et al. (2022) <https://doi.org/10.22331/q-2022-03-30-677>`_ for more details
        and :func:`~.param_shift` for an implementation of the non-stochastic generalized shift
        rules)
        Given the parameter shift rule with coefficients :math:`\{y_\ell\}` and shifts
        :math:`\{x_\ell\}` for the single-parameter pulse :math:`\exp(-i \theta H_j)`,
        the linear combination is given by

        .. math::

            \widetilde{C_j}(\boldsymbol{v}, \tau)&=\sum_{\ell=1} y_\ell
            \bra{\psi_{j}(\boldsymbol{v}, x_\ell, \tau)} B
            \ket{\psi_{j}(\boldsymbol{v}, x_\ell, \tau)} \\
            \ket{\psi_{j}(\boldsymbol{v}, x_\ell, \tau)}
            &= U_{\boldsymbol{v}}(T, \tau) e^{-i x_\ell H_j}
            U_{\boldsymbol{v}}(\tau, 0)\ket{\psi_0}.

        In practice, the time integral over :math:`\tau` is computed by sampling values for
        the time, evaluating the integrand, and averaging appropriately. The probability
        distribution used for the sampling may have a significant impact on the quality of the
        obtained estimates, in particular with regards to their variance.
        In this function, a uniform distribution over the interval :math:`[0, t]` is used,
        which often can be improved upon.

        **Examples**

        Consider the pulse generated by

        .. math::

            H(\boldsymbol{v}, t) = \frac{1}{2} X_0 + v_1 Z_0 Z_1 + \sin(v_2 t) X_1

        and the observable :math:`B=Y_1`. There are two variational parameters, :math:`v_1`
        and :math:`v_2`, for which we may compute the derivative of the cost function:

        .. math::

            \frac{\partial C}{\partial v_1}
            &= \int_{0}^{T} \mathrm{d}\tau \ \widetilde{C_1}((v_1, v_2), \tau)\\
            \frac{\partial C}{\partial v_2}
            &= \int_{0}^{T} \mathrm{d}\tau \cos(v_2 \tau) \tau \ \widetilde{C_2}((v_1, v_2), \tau)\\
            \widetilde{C_j}((v_1, v_2), \tau)&=
            \bra{\psi_{j}((v_1, v_2), \pi/4, \tau)} B
            \ket{\psi_{j}((v_1, v_2), \pi/4, \tau)}\\
            &-\bra{\psi_{j}((v_1, v_2), -\pi/4, \tau)} B
            \ket{\psi_{j}((v_1, v_2), -\pi/4, \tau)} \\
            \ket{\psi_{j}((v_1, v_2), x, \tau)}
            &= U_{(v_1, v_2)}(T, \tau) e^{-i x H_j}U_{(v_1, v_2)}(\tau, 0)\ket{0}.

        Here we used the partial derivatives

        .. math::

            \frac{\partial f_1}{\partial v_1}&= 1\\
            \frac{\partial f_2}{\partial v_2}&= \cos(v_2 t) t \\
            \frac{\partial f_1}{\partial v_2}=
            \frac{\partial f_2}{\partial v_1}&= 0

        and the fact that both :math:`H_1=Z_0 Z_1` and :math:`H_2=X_1`
        have two unique eigenvalues and therefore admit a two-term parameter-shift rule
        (see e.g. `Schuld et al. (2018) <https://arxiv.org/abs/1811.11184>`_).

        As a second scenario, consider the single-qubit pulse generated by

        .. math::

            H((v_1, v_2), t) = v_1 \sin(v_2 t) X

        together with the observable :math:`B=Z`.
        You may already notice that this pulse can be rewritten as a :class:`~.RX` rotation,
        because we have a single Hamiltonian term and the spectrum of :math:`H` consequently
        will be constant up to rescaling.
        In particular, the unitary time evolution under the Schrödinger equation is given by

        .. math::

            U_{(v_1, v_2)}(t_2, t_1) &=
            \exp\left(-i\int_{t_1}^{t_2} \mathrm{d}\tau v_1 \sin(v_2 \tau) X\right)\\
            &=\exp(-i\theta(v_1, v_2) X)\\
            \theta(v_1, v_2) &= \int_{t_1}^{t_2} \mathrm{d}\tau v_1 \sin(v_2 \tau)\\
            &=-\frac{v_1}{v_2}(\cos(v_2 t_2) - \cos(v_2 t_1)).

        As the ``RX`` rotation satisfies a (non-stochastic) two-term parameter-shift rule,
        we could compute the derivatives with respect to :math:`v_1` and :math:`v_2` by
        implementing :math:`\exp(-i\theta(v_1, v_2) X)`, applying the two-term shift rule
        and evaluating the classical Jacobian of the mapping :math:`\theta(v_1, v_2)`.

        Using the stochastic parameter-shift rule instead will lead to approximation errors.
        This is because the approximated integral not only includes the shifted circuit
        evaluations, which do not depend on :math:`\tau` in this example, but also on the
        classical Jacobian, which is *not* constant over :math:`\tau`.
        Therefore, it is important to implement pulses in the simplest way possible.
    """
    # pylint:disable=unused-argument
    transform_name = "stochastic pulse parameter-shift"
    _assert_has_jax(transform_name)
    assert_no_state_returns(tape.measurements, transform_name)
    assert_no_variance(tape.measurements, transform_name)
    assert_no_trainable_tape_batching(tape, transform_name)

    if num_split_times < 1:
        raise ValueError(
            "Expected a positive number of samples for the stochastic pulse "
            f"parameter-shift gradient, got {num_split_times}."
        )

    if argnum is None and not tape.trainable_params:
        return _no_trainable_grad(tape)

    if use_broadcasting and tape.batch_size is not None:
        raise ValueError("Broadcasting is not supported for tapes that already are broadcasted.")

    trainable_params = choose_trainable_params(tape, argnum)
    diff_methods = find_and_validate_gradient_methods(tape, "analytic", trainable_params)

    if all(g == "0" for g in diff_methods.values()):
        return _all_zero_grad(tape)

    argnum = [i for i, dm in diff_methods.items() if dm == "A"]

    sampler_seed = sampler_seed or np.random.randint(18421)
    key = jax.random.PRNGKey(sampler_seed)

    return _expval_stoch_pulse_grad(tape, argnum, num_split_times, key, use_broadcasting)


def _generate_tapes_and_cjacs(
    tape, operation, key, num_split_times, use_broadcasting, par_idx=None
):
    """Generate the tapes and compute the classical Jacobians for one given
    generating Hamiltonian term of one pulse.

    Args:
        tape (QuantumScript): Tape for which to compute the stochastic pulse parameter-shift
            gradient tapes.
        operation (tuple[Operation, int, int]): Information about the pulse operation to be
            shifted. The first entry is the operation itself, the second entry is its position
            in the ``tape``, and the third entry is the index of the differentiated parameter
            (and generating term) within the ``HardwareHamiltonian`` of the operation.
        key (tuple[int]): Randomness key to create spliting times.
        num_split_times (int): Number of splitting times at which to create shifted tapes for
            the stochastic shift rule.
        use_broadcasting (bool): Whether to use broadcasting in the shift rule or not.

    Returns:
        list[QuantumScript]: Gradient tapes for the indicated operation and Hamiltonian term.
        list[tensor_like]: Classical Jacobian at the splitting times for the given parameter.
        float: Prefactor for the Monte Carlo estimate of the integral in the stochastic shift rule.
        tensor_like: Parameter-shift coefficients for the shift rule of the indicated term.
    """
    op, op_idx, term_idx = operation
    coeff, ob = op.H.coeffs_parametrized[term_idx], op.H.ops_parametrized[term_idx]
    if par_idx is None:
        cjac_fn = jax.jacobian(coeff, argnums=0)
    else:
        # For `par_idx is not None`, we need to extract the entry of the coefficient
        # Jacobian that belongs to the parameter of interest. This only happens when
        # more than one parameter effectively feeds into one coefficient (HardwareHamiltonian)

        def cjac_fn(params, t):
            return jax.jacobian(coeff, argnums=0)(params, t)[par_idx]

    t0, *_, t1 = op.t
    taus = jnp.sort(jax.random.uniform(key, shape=(num_split_times,)) * (t1 - t0) + t0)
    if isinstance(op.H, HardwareHamiltonian):
        op_data = op.H.reorder_fn(op.data, op.H.coeffs_parametrized)
    else:
        op_data = op.data
    cjacs = [cjac_fn(op_data[term_idx], tau) for tau in taus]
    if use_broadcasting:
        split_evolve_ops, psr_coeffs = _split_evol_ops(op, ob, taus)
        tapes = _split_evol_tape(tape, split_evolve_ops, op_idx)
    else:
        tapes = []
        for tau in taus:
            split_evolve_ops, psr_coeffs = _split_evol_ops(op, ob, tau)
            tapes.extend(_split_evol_tape(tape, split_evolve_ops, op_idx))
    int_prefactor = (t1 - t0) / num_split_times
    return tapes, cjacs, int_prefactor, psr_coeffs


def _tapes_data_hardware(tape, operation, key, num_split_times, use_broadcasting):
    """Create tapes and gradient data for a trainable parameter of a HardwareHamiltonian,
    taking into account its reordering function.

    Args:
        tape (QuantumScript): Tape for which to compute the stochastic pulse parameter-shift
            gradient tapes.
        operation (tuple[Operation, int, int]): Information about the pulse operation to be
            shifted. The first entry is the operation itself, the second entry is its position
            in the ``tape``, and the third entry is the index of the differentiated parameter
            within the ``HardwareHamiltonian`` of the operation.
        key (tuple[int]): Randomness key to create spliting times in ``_generate_tapes_and_cjacs``
        num_split_times (int): Number of splitting times at which to create shifted tapes for
            the stochastic shift rule.
        use_broadcasting (bool): Whether to use broadcasting in the shift rule or not.

    Returns:
        list[QuantumScript]: Gradient tapes for the indicated operation and Hamiltonian term.
        tuple: Gradient postprocessing data.
            See comment below.

    This function analyses the ``reorder_fn`` of the ``HardwareHamiltonian`` of the pulse
    that is being differentiated. Given a ``term_idx``, the index of the parameter
    in the Hamiltonian, stochastic parameter shift tapes are created for all terms in the
    Hamiltonian into which the parameter feeds. While this is a one-to-one relation for
    standard ``ParametrizedHamiltonian`` objects, the reordering function of
    the ``HardwareHamiltonian`` requires to create tapes for multiple Hamiltonian terms,
    and for each term ``_generate_tapes_and_cjacs`` is called.

    The returned gradient data has four entries:

      1. ``int``: Total number of tapes created for all the terms that depend on the indicated
         parameter.
      2. ``tuple[tensor_like]``: Classical Jacobians for all terms and splitting times
      3. ``float``: Prefactor for the Monte Carlo estimate of the integral in the stochastic
         shift rule.
      4. ``tuple[tensor_like]``: Parameter-shift coefficients for all terms.

    The tuple axes in the second and fourth entry correspond to the different terms in the
    Hamiltonian.
    """
    op, op_idx, term_idx = operation
    # Map a simple enumeration of numbers from HardwareHamiltonian input parameters to
    # ParametrizedHamiltonian parameters. This is typically a fan-out function.
    fake_params, allowed_outputs = np.arange(op.num_params), set(range(op.num_params))
    reordered = op.H.reorder_fn(fake_params, op.H.coeffs_parametrized)

    def _raise():
        raise ValueError(
            "Only permutations, fan-out or fan-in functions are allowed as reordering functions "
            "in HardwareHamiltonians treated by stoch_pulse_grad. The reordering function of "
            f"{op.H} mapped {fake_params} to {reordered}."
        )

    cjacs, tapes, psr_coeffs = [], [], []
    for coeff_idx, x in enumerate(reordered):
        # Find out whether the value term_idx, corresponding to the current parameter of interest,
        # has been mapped to x (for scalar x) or into x (for 1d x). If so, generate tapes and data
        # Also check that only allowed outputs have been produced by the reordering function.
        if not hasattr(x, "__len__"):
            if x not in allowed_outputs:
                _raise()
            if x != term_idx:
                continue
            cjac_idx = None
        else:
            if not all(_x in list(range(op.num_params)) for _x in x):
                _raise()
            if term_idx not in x:
                continue
            cjac_idx = np.argwhere([_x == term_idx for _x in x])[0][0]

        _operation = (op, op_idx, coeff_idx)
        # Overwriting int_prefactor does not matter, it is equal for all parameters in this op,
        # because it only consists of the duration `op.t[-1]-op.t[0]` and `num_split_times`
        _tapes, _cjacs, int_prefactor, _psr_coeffs = _generate_tapes_and_cjacs(
            tape, _operation, key, num_split_times, use_broadcasting, cjac_idx
        )
        cjacs.append(qml.math.stack(_cjacs))
        tapes.extend(_tapes)
        psr_coeffs.append(_psr_coeffs)

    # The fact that psr_coeffs are a tuple only for hardware Hamiltonian generators will be
    # used in `_parshift_and_integrate`.
    data = (len(tapes), tuple(cjacs), int_prefactor, tuple(psr_coeffs))
    return tapes, data


# pylint: disable=too-many-arguments
def _expval_stoch_pulse_grad(tape, argnum, num_split_times, key, use_broadcasting):
    r"""Compute the gradient of a quantum circuit composed of pulse sequences that measures
    an expectation value or probabilities, by applying the stochastic parameter shift rule.
    See the main function for the signature.
    """
    tapes = []
    gradient_data = []
    for idx in range(tape.num_params):
        if idx not in argnum:
            # Only the number of tapes is needed to indicate a zero gradient entry
            gradient_data.append((0, None, None, None))
            continue

        key, _key = jax.random.split(key)
        operation = tape.get_operation(idx)
        op, *_ = operation
        if not isinstance(op, ParametrizedEvolution):
            raise ValueError(
                "stoch_pulse_grad does not support differentiating parameters of "
                "other operations than pulses."
            )
        if isinstance(op.H, HardwareHamiltonian):
            # Treat HardwareHamiltonians separately because they have a reordering function
            _tapes, data = _tapes_data_hardware(
                tape, operation, key, num_split_times, use_broadcasting
            )
        else:
            _tapes, cjacs, int_prefactor, psr_coeffs = _generate_tapes_and_cjacs(
                tape, operation, _key, num_split_times, use_broadcasting
            )
            data = (len(_tapes), qml.math.stack(cjacs), int_prefactor, psr_coeffs)

        tapes.extend(_tapes)
        gradient_data.append(data)

    num_measurements = len(tape.measurements)
    single_measure = num_measurements == 1
    num_params = len(tape.trainable_params)
    has_partitioned_shots = tape.shots.has_partitioned_shots
    tape_specs = (single_measure, num_params, num_measurements, tape.shots)

    def processing_fn(results):
        start = 0
        grads = []
        for num_tapes, cjacs, int_prefactor, psr_coeffs in gradient_data:
            if num_tapes == 0:
                grads.append(None)
                continue
            res = results[start : start + num_tapes]
            start += num_tapes
            # Apply the postprocessing of the parameter-shift rule and contract
            # with classical Jacobian, effectively computing the integral approximation
            g = _parshift_and_integrate(
                res,
                cjacs,
                int_prefactor,
                psr_coeffs,
                single_measure,
                has_partitioned_shots,
                use_broadcasting,
            )
            grads.append(g)

        # g will have been defined at least once (because otherwise all gradients would have
        # been zero), providing a representative for a zero gradient to emulate its type/shape.
        zero_rep = _make_zero_rep(g, single_measure, has_partitioned_shots)

        # Fill in zero-valued gradients
        grads = [zero_rep if g is None else g for g in grads]

        return reorder_grads(grads, tape_specs)

    return tapes, processing_fn


@stoch_pulse_grad.custom_qnode_transform
def stoch_pulse_grad_qnode_wrapper(self, qnode, targs, tkwargs):
    """A custom QNode wrapper for the gradient transform :func:`~.stoch_pulse_grad`.
    It raises an error, so that applying ``stoch_pulse_grad`` to a ``QNode`` directly
    is not supported.
    """
    # pylint:disable=unused-argument
    transform_name = "stochastic pulse parameter-shift"
    raise_pulse_diff_on_qnode(transform_name)
