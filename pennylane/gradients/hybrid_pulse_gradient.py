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
This module contains functions for computing the hybrid parameter-shift gradient
of pulse sequences in a qubit-based quantum tape.
"""
from collections.abc import Sequence
import numpy as np

import pennylane as qml

from pennylane.pulse import ParametrizedEvolution
from pennylane.ops.qubit.special_unitary import pauli_basis_strings, _pauli_decompose

from .parameter_shift import _make_zero_rep
from .pulse_gradient import _assert_has_jax
from .gradient_transform import (
    _all_zero_grad,
    assert_active_return,
    assert_no_state_returns,
    assert_no_variance,
    choose_grad_methods,
    gradient_analysis_and_validation,
    gradient_transform,
    _no_trainable_grad,
    reorder_grads,
)

try:
    import jax
except ImportError:
    # Handling the case where JAX is not installed is done via _assert_has_jax
    pass


def _one_parameter_generators(op):
    r"""Compute the effective generators of one-parameter groups that reproduce
    the partial derivatives of a parameterized evolution.

    Args:
        op (`~.ParametrizedEvolution`): Parametrized evolution for which to compute the generator

    Returns:
        tuple[tensor_like]: The generators for one-parameter groups that reproduce the
        partial derivatives of the parametrized evolution.
        The ``k``\ th entry of the returned ``tuple`` has the shape ``(2**N, 2**N, *par_shape)``
        where ``N`` is the number of qubits the evolution acts on and ``par_shape`` is the
        shape of the ``k``\ th parameter of the evolution.

    See the documentation of hybrid_pulse_grad for more details and a mathematical derivation.
    """

    def _compute_matrix(*args):
        return op(*args, t=op.t, **op.odeint_kwargs).matrix()

    data = [(1.0 + 0.0j) * d for d in op.data]
    # These lines compute the Jacobian of compute_matrix every time -> to be optimized
    jac = jax.jacobian(_compute_matrix, argnums=0, holomorphic=True)(data)

    # Compute the Omegas from the Jacobian.
    U_dagger = qml.math.conj(_compute_matrix([qml.math.detach(d) for d in data]))
    # After contracting, move the parameter derivative axis to the first position
    return tuple(
        qml.math.moveaxis(qml.math.tensordot(U_dagger, j, axes=[[0], [0]]), (0, 1), (-2, -1))
        for j in jac
    )


def _one_parameter_paulirot_coeffs(generators, num_wires):
    r"""Compute the Pauli coefficients of the one-parameter group generators that reproduce
    the partial derivatives of a parametrized evolution. The coefficients correspond
    to the decomposition into (rescaled) Pauli word generators as used by ``PauliRot``
    gates.

    Args:
        generators (tuple[tensor_like]): Generators of the one-parameter groups that
            reproduce the partial derivatives of the parametrized evolution of interest.
        num_wires (int): Number of wires the parametrized evolution acts on.

    Returns:
        tuple[tensor_like]: Coefficients of the provided generators in the Pauli basis,
        modified by a factor of ``2j`` (see warning below).

    .. warning::

        This function includes a prefactor ``2j`` in its output compared to the "standard" Pauli
        basis coefficients. That is, for the effective generator :math:`\frac{1}{3}iX`, which has
        the Pauli basis coefficient :math:`\frac{1}{3}i`, this function will compute the
        number :math:`-\frac{2}{3}`. This is in order to accomodate the use of ``qml.PauliRot``
        in the hybrid pulse differentiation pipeline.

    .. note::

        Currently, this function work via tensor multiplication, costing
        :math:`\mathcal{O}(n16^N)` scalar multiplications, where :math:`N` is the
        number of qubits the evolution acts on and :math:`n` is the number of (scalar) parameters
        of the evolution. This can be improved to :math:`\mathcal{O}(nN4^N}` by using the
        fast Walsh-Hadamard transform.
    """
    return tuple(qml.math.real(2j * _pauli_decompose(g, num_wires)) for g in generators)


def _nonzero_coeffs_and_words(coefficients, num_wires, atol=1e-8):
    """Given a tuple of coefficient tensors with a common first axis size :math:`4^N-1`,
    where :math:`N` is the number of wires, filter out those indices for the first axis for which
    the sliced coefficients are not all zero. Return these coefficients, as well as the
    corresponding Pauli word strings on :math:`N` wires for these indices.

    Args:
        coefficients(tuple[tensor_like]): Coefficients to filter.
        num_wires (int): Number of qubits the generators act on.
        atol (float): absolute tolerance used to determine whether a tensor is zero.
    """
    words = pauli_basis_strings(num_wires)
    new_coefficients = [[] for _ in coefficients]
    new_words = []
    for word, *coeffs in zip(words, *coefficients):
        if all(qml.math.allclose(c, 0.0, atol=atol, rtol=0.0) for c in coeffs):
            continue
        new_words.append(word)
        for new_coeffs, c in zip(new_coefficients, coeffs):
            new_coeffs.append(c)
    return new_coefficients, new_words


def _insert_op(tape, ops, op_idx):
    """Create new tapes, each with an additional operation from a provided list inserted at
    the specified position. Creates as many new tapes as there are operations provided.

    Args:
        tape (`~.QuantumTape`): Original tape.
        ops (list[qml.Operation]): Operations to insert (individually) at ``op_idx``
            to produce a new tape each.
        op_idx (int): Index at which to insert the operations in ``ops``.

    Returns:
        list[`~.QuantumScript`]: new tapes with inserted operations, one tape per operation
        in ``ops``.
    """
    ops_pre = tape.operations[:op_idx]
    ops_post = tape.operations[op_idx:]
    return [
        qml.tape.QuantumScript(ops_pre + [insert] + ops_post, tape.measurements) for insert in ops
    ]


def _generate_tapes_and_coeffs(tape, idx, atol, cache):
    """Compute the modified tapes and coefficients required to compute the hybrid pulse
    derivative of a tape with respect to an indicated trainable parameter.

    Args:
        tape (`~.QuantumTape`): Tape to differentiate.
        idx (int): Index of the parameter with respect to which to differentiate
            into ``tape.trainable_parameters``.
        atol (float): absolute tolerance used to determine whether a coefficient is zero.
        cache (dict): Caching dictionary that allows to skip adding duplicate modified tapes.

    Returns:
        list[`~.QuantumScript`]: Modified tapes to be added to the hybrid pulse differentiation
            tapes. It is an empty list if modified tapes were already created for another
            parameter of the pulse of interest.
        tuple[int, int, tensor_like]: Gradient computation data, consisting of the start and end
            indices into the total list of tapes as well as the coefficients that need to be
            contracted with the corresponding results to obtain the partial derivative with respect
            to the indicated trainable parameter.
        dict: Input ``cache`` dictionary. If the cache lookup missed, the cache is extended by one
            entry and its entry ``"total_num_tapes"`` is increased by the number of created tapes.
    """
    op, op_idx, term_idx = tape.get_operation(idx)
    if op_idx in cache:
        # operation was analyzed and tapes were added before. Retrieve tape indices and coefficients.
        start, end, all_coeffs = cache[op_idx]
        return [], (start, end, all_coeffs[term_idx]), cache

    if not isinstance(op, ParametrizedEvolution):
        # only ParametrizedEvolution can be treated with this gradient transform
        raise ValueError(
            "hybrid_pulse_grad does not support differentiating parameters of "
            "other operations than pulses."
        )

    num_wires = len(op.wires)
    # Obtain one-parameter-group coefficients in Pauli basis and filter for non-zero coeffs
    generators = _one_parameter_generators(op)
    all_coeffs = _one_parameter_paulirot_coeffs(generators, num_wires)
    all_coeffs, pauli_words = _nonzero_coeffs_and_words(all_coeffs, num_wires, atol)
    # create PauliRot gates for each Pauli word with a non-zero coefficient and for both shifts
    pauli_rots = [
        qml.PauliRot(angle, word, wires=op.wires)
        for word in pauli_words
        for angle in [np.pi / 2, -np.pi / 2]
    ]
    # create tapes with the above PauliRot gates inserted, one per tape
    tapes = _insert_op(tape, pauli_rots, op_idx)
    # get the previous total number of tapes from the cache and determine start and end indices
    end = (start := cache["total_num_tapes"]) + (num_tapes := len(tapes))
    # store the tape indices and coefficients for this operation, to be retrieved for other
    # parameters of the same operation
    cache[op_idx] = (start, end, all_coeffs)
    # increment total number of tapes
    cache["total_num_tapes"] += num_tapes
    return tapes, (start, end, all_coeffs[term_idx]), cache


def _parshift_and_contract(results, coeffs, single_measure, shot_vector):
    """Compute parameter-shift tape derivatives and contract them with coefficients.

    Args:
        results (list[tensor_like]): Tape execution results to be processed.
        coeffs (list[tensor_like]): Coefficients to be contracted.
        single_measure (bool): whether the tape execution results contain single measurements.
        shot_vector (bool): whether the tape execution results were obtained with a shot vector.

    Returns:
        tensor_like or tuple[tensor_like] or tuple[tuple[tensor_like]]: contraction between the
        parameter-shift derivative computed from ``results`` and the ``coeffs``.
    """

    def _parshift_and_contract_single(res_list, coeffs):
        psr_deriv = ((res := qml.math.stack(res_list))[::2] - res[1::2]) / 2
        return qml.math.tensordot(psr_deriv, coeffs, axes=[[0], [0]])

    if single_measure and not shot_vector:
        # single measurement and single shot entry
        return _parshift_and_contract_single(results, qml.math.stack(coeffs))
    if single_measure or not shot_vector:
        # single measurement or single shot entry, but not both
        return tuple(
            _parshift_and_contract_single(r, qml.math.stack(coeffs)) for r in zip(*results)
        )

    return tuple(
        tuple(_parshift_and_contract_single(_r, qml.math.stack(coeffs)) for _r in zip(*r))
        for r in zip(*results)
    )


def _expval_hybrid_pulse_grad(tape, argnum, shots, atol):
    """Compute the hybrid pulse parameter-shift rule for a quantum circuit that returns expectation
    values of observables.

    Args:
        tape (`~.QuantumTape`): Quantum circuit to be differentiated with the hybrid rule.
        argnum (int or list[int] or None): Trainable tape parameter indices to differentiate
            with respect to. If not provided, the derivatives with respect to all
            trainable parameters are returned.
        shots (None, int, list[int]): The shots of the device used to execute the tapes which are
            returned by this transform. Note that this argument does not *influence* the shots
            used for execution, but *informs* the transform about the shots to ensure a compatible
            return value formatting.
        atol (float): absolute tolerance used to determine vanishing contributions.

    Returns:
        function or tuple[list[QuantumTape], function]:

        - If the input is a QNode, an object representing the Jacobian (function) of the QNode
          that can be executed to obtain the Jacobian.
          The type of the Jacobian returned is either a tensor, a tuple or a
          nested tuple depending on the nesting structure of the original QNode output.

        - If the input is a tape, a tuple containing a
          list of generated tapes, together with a post-processing
          function to be applied to the results of the evaluated tapes
          in order to obtain the Jacobian.

    """
    argnum = argnum or tape.trainable_params
    gradient_data = []
    gradient_tapes = []
    cache = {"total_num_tapes": 0}
    for idx, trainable_idx in enumerate(tape.trainable_params):
        if trainable_idx not in argnum:
            # Indicate that there are no tapes for this parameter by setting start==end
            gradient_data.append((0, 0, None))
            continue

        tapes, data, cache = _generate_tapes_and_coeffs(tape, idx, atol, cache)
        gradient_data.append(data)
        gradient_tapes.extend(tapes)

    num_measurements = len(tape.measurements)
    single_measure = num_measurements == 1
    num_params = len(tape.trainable_params)
    shot_vector = isinstance(shots, Sequence)
    tape_specs = (single_measure, num_params, num_measurements, shot_vector, shots)

    def processing_fn(results):
        grads = []
        for start, end, coeffs in gradient_data:
            if start == end:
                grads.append(None)
                continue
            res = results[start:end]
            # Apply the postprocessing of the parameter-shift rule and contract with the
            # one-parameter group coefficients, computing the partial circuit derivative
            g = _parshift_and_contract(res, coeffs, single_measure, shot_vector)
            grads.append(g)

        # g will have been defined at least once (because otherwise all gradients would have
        # been zero), providing a representative for a zero gradient to emulate its type/shape.
        zero_rep = _make_zero_rep(g, single_measure, shot_vector)

        # Fill in zero-valued gradients
        grads = [zero_rep if g is None else g for g in grads]
        print(f"{grads=}")
        x = reorder_grads(grads, tape_specs)

        return x

    return gradient_tapes, processing_fn


def _hybrid_pulse_grad(tape, argnum=None, shots=None, atol=1e-7):
    r"""Transform a QNode to compute the hybrid pulse parameter-shift gradient of pulses
    in a pulse program with respect to their inputs.
    This method combines automatic differentiation of few-qubit operations with
    hardware-compatible shift rules. Thus, it is limited to few-qubit pulses, but allows
    for the evaluation of parameter-shift gradients for many-qubit pulse programs.

    For this differentiation method, the unitary matrix corresponding to each pulse
    is computed and differentiated with an autodiff framework. From this derivative,
    the so-called effective generators :math:`\Omega_{k, r}` of the pulses
    (one for each variational parameter) are extracted.
    Afterwards, the generators are decomposed into the Pauli basis and the
    standard parameter-shift rule is used to evaluate the derivatives of the pulse program
    in this basis. To this end, shifted ``PauliRot`` operations are inserted in the program.
    Finally, the Pauli basis derivatives are recombined into the gradient
    of the pulse program with respect to its original parameters.
    See the theoretical background section below for more details.

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to differentiate
        argnum (int or list[int] or None): Trainable tape parameter indices to differentiate
            with respect to. If not provided, the derivatives with respect to all
            trainable parameters are returned.
        shots (None, int, list[int]): The device shots that will be used to execute the tapes
            outputted by this transform. Note that this argument doesn't influence the shots
            used for tape execution, but provides information about the shots.
        atol (float): Precision parameter used to truncate the Pauli basis coefficients
            of the effective generators. Coefficients ``x`` satisfying
            ``qml.math.isclose(x, 0., atol=atol, rtol=0)=True`` are neglected.

    Returns:
        function or tuple[list[QuantumTape], function]:

        - If the input is a QNode, an object representing the Jacobian (function) of the QNode
          that can be executed to obtain the Jacobian.
          The type of the Jacobian returned is either a tensor, a tuple or a
          nested tuple depending on the nesting structure of the original QNode output.

        - If the input is a tape, a tuple containing a
          list of generated tapes, together with a post-processing
          function to be applied to the results of the evaluated tapes
          in order to obtain the Jacobian.

    **Example**

    Consider the parameterized Hamiltonian
    :math:`\theta_0 Y^{(0)}+f(\boldsymbol{\theta_1}, t) Y^{(1)} + \theta_2 Z^{(0)}X^{(1)}`
    with parameters :math:`\theta_0 = \frac{1}{5}`,
    :math:`\boldsymbol{\theta_1}=\left(\frac{3}{5}, \frac{1}{5}\right)^{T}` and
    :math:`\frac{2}{5}` as well as a time interval :math:`t=[\frac{1}{10}, \frac{9}{10}]`.

    .. code-block:: python

        jax.config.update("jax_enable_x64", True)
        H = (
            qml.pulse.constant * qml.PauliY(0)
            + jnp.polyval * qml.PauliY(1)
            + qml.pulse.constant * (qml.PauliZ(0)@qml.PauliX(1))
        )
        params = [jnp.array(0.2), jnp.array([0.6, 0.2]), jnp.array(0.4)]
        t = [0.1, 0.9]

    For simplicity, consider a pulse program consisting of this single pulse and a
    measurement of the expectation value of :math:`X^{(0)}`.
    .. code-block:: python

        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev, interface="jax", diff_method=qml.gradients.hybrid_pulse_grad)
        def circuit(params):
            op = qml.evolve(H)(params, t)
            return qml.expval(qml.PauliX(0))

    We registered the ``QNode`` to be differentiated with the ``hybrid_pulse_grad`` method.
    This allows us to simply differentiate it with ``jax.jacobian``, for example, internally
    making use of the hybrid pulse parameter-shift method.

    >>> jax.jacobian(circuit)(params)
    [Array(1.4189798, dtype=float32, weak_type=True),
     Array([0.00164918, 0.00284802], dtype=float32),
     Array(-0.09984542, dtype=float32, weak_type=True)]

    Alternatively, we may apply the transform to the tape of the pulse program, obtaining
    the tapes with inserted ``PauliRot`` gates together with the post-processing function:

    >>> tapes, fun = qml.gradients.hybrid_pulse_grad(circuit.tape, argnums=[0, 1, 2])
    >>> len(tapes)
    12

    There are :math:`12` tapes because there are two shift parameters per Pauli word
    and six Pauli words in the basis of the *dynamical Lie algebra (DLA)* of the pulse:
    :math:`\{Y^{(1)}, X^{(0)}X^{(1)}, X^{(0)}Z^{(1)}, Y^{(0)}, Z^{(0)}X^{(1)}, Z^{(0)}Z^{(1)}\}`.
    We may inspect one of the tapes, which differs from the original tape by the inserted
    rotation gate ``"RIY"``, i.e. a ``PauliRot(np.pi/2, "IY", wires=[0, 1])`` gate.

    >>> print(qml.drawer.tape_text(tapes[0]))
    0: ─╭RIY─╭ParametrizedEvolution─┤  <X>
    1: ─╰RIY─╰ParametrizedEvolution─┤

    Executing the tapes and applying the post-processing function to the results then
    yields the gradient:

    >>> fun(qml.execute(tapes, dev))
    (Array(1.41897933, dtype=float64),
     Array([0.00164914, 0.00284789], dtype=float64),
     Array(-0.09984585, dtype=float64))

    .. note::

        This function requires the JAX interface and does not work with other autodiff interfaces
        commonly encountered with PennyLane.
        In addition, this transform is only JIT-compatible with scalar pulse parameters.

    .. details::
        :title: Theoretical background
        :href: theory

        The hybrid pulse parameter-shift gradient method makes use of the *effective generator*
        of a pulse for given parameters and duration. Consider the parametrized Hamiltonian

        .. math::

            H(\{\boldsymbol{\theta_k}\}, t) = \sum_{k=1}^K f_k(\boldsymbol{\theta_k}, t) H_k

        where the Hamiltonian terms :math:`\{H_k\}` are constant and the :math:`\{f_k\}` are
        parametrized time-dependent functions depending the parameters
        :math:`\boldsymbol{\theta_k}`.
        We may associate a matrix function with this Hamiltonian (and a time interval
        :math:`[t_0, t_1]`), which is given by the matrix of the unitary time evolution operator
        under :math:`H` that solves the Schrödinger equation:

        .. math::

            U(\{\boldsymbol{\theta_k}\}, [t_0, t_1]) =
            \mathcal{T} \exp\left[ -i\int_{t_0}^{t_1}
            H(\{\boldsymbol{\theta_k}\}, \tau) \mathrm{d}\tau \right].

        To compute the hybrid pulse parameter-shift gradient, we are interested in the partial
        derivatives of this function, usually with respect to the parameters
        :math:`\boldsymbol{\theta}`. Provided that :math:`H` does not act on too many qubits,
        or that we have an alternative sparse representation of
        :math:`U(\{\boldsymbol{\theta_k}\}, [t_0, t_1])`,
        we may compute these partial derivatives

        .. math::

            \frac{\partial U(\{\boldsymbol{\theta_k}\}, [t_0, t_1])}{\partial \theta_{k, r}}

        classically via automatic differentiation, where :math:`\theta_{k, r}` is
        the :math:`r`\ -th scalar parameter in :math:`\boldsymbol{\theta_k}`.

        Now, due to the compactness of the groups :math:`\mathrm{SU}(N)` , we know that
        for each :math:`\theta_{k, r}` there is an *effective generator* :math:`\Omega_{k, r}`
        such that

        .. math::

            \frac{\partial U(\{\boldsymbol{\theta_k}\}, [t_0, t_1])}{\partial \theta_{k,r}} =
            U(\{\boldsymbol{\theta_k}\}, [t_0, t_1])
            \frac{\mathrm{d}}{\mathrm{d} x} \exp[x\Omega_{k,r}]\large|_{x=0}.

        Given that we can compute the left-hand side expression as well as the matrix
        for :math:`U` itself, we can compute :math:`\Omega_{k,r}` for all parameters
        :math:`\theta_{k,r}`.
        In addition, we may decompose these generators into Pauli words:

        .. math::

            \Omega_{k,r} =\sum_{\ell=1}^{L} \omega_{k,r}^{(\ell)} \left(P_{\ell}\right)

        The coefficients :math:`\omega_{k,r}^{(\ell)}` of the generators can be computed
        by decomposing the anti-Hermitian matrix :math:`\Omega_{k,r}` into the Pauli
        basis and only keeping the non-vanishing terms. This is possible by a tensor
        contraction with the full Pauli basis (or alternative, more efficient methods):

        .. math::

            \omega_{k,r}^{(\ell)} = \frac{1}{2^N}\mathrm{Tr}\left[P_\ell \Omega_{k,r}\right]

        where :math:`N` is the number of qubits.
        We may use all of the above to rewrite the partial derivative of an expectation
        value-based objective function:

        .. math::

            C(\{\boldsymbol{\theta_k}\}, [t_0, t_1])&=
            \langle\psi_0|U(\{\boldsymbol{\theta_k}\}, [t_0, t_1])^\dagger B
            U(\{\boldsymbol{\theta_k}\}, [t_0, t_1]) |\psi_0\rangle\\
            \frac{\partial C}{\partial \theta_{k,r}} (\{\boldsymbol{\theta_k}\}, [t_0, t_1])&=
            \langle\psi_0|\left[U^\dagger B U, \Omega_{k,r}\right]|\psi_0\rangle\\
            &=\sum_{\ell=1}^L \omega_{k,r}^{(\ell)}
            \langle\psi_0|\left[U^\dagger B U, P_\ell \right]|\psi_0\rangle\\
            &=\sum_{\ell=1}^L \tilde\omega_{k,r}^{(\ell)}
            \langle\psi_0|\left[U^\dagger B U, -\frac{i}{2}P_\ell \right]|\psi_0\rangle
            &=\sum_{\ell=1}^L \tilde\omega_{k,r}^{(\ell)}
            \frac{\mathrm{d}}{\mathrm{d}x} C_\ell(x)\large|_{x=0}

        where we skipped the arguments of the unitary for readability, introduced the
        modified coefficients :math:`\tilde\omega_{k,r}^{(\ell)}=2i\omega_{k,r}^{(\ell)}`,
        and denoted by :math:`C_\ell(x)` the modified cost function that has a Pauli
        rotation about :math:`-\frac{x}{2}P_\ell` inserted before the parametrized time
        evolution.

    """
    print(argnum)
    transform_name = "hybrid pulse parameter-shift"
    _assert_has_jax(transform_name)
    assert_active_return(transform_name)
    assert_no_state_returns(tape.measurements, transform_name)
    assert_no_variance(tape.measurements, transform_name)
    print(argnum)

    if argnum is None and not tape.trainable_params:
        return _no_trainable_grad(tape, shots)

    diff_methods = gradient_analysis_and_validation(tape, "analytic", grad_fn=hybrid_pulse_grad)

    print(diff_methods)
    if all(g == "0" for g in diff_methods):
        return _all_zero_grad(tape, shots)

    method_map = choose_grad_methods(diff_methods, argnum)

    argnum = [i for i, dm in method_map.items() if dm == "A"]

    return _expval_hybrid_pulse_grad(tape, argnum, shots, atol)


def expand_invalid_trainable_hybrid_pulse_grad(x, *args, **kwargs):
    r"""Do not expand any operation. We expect the ``hybrid_pulse_grad`` to be used
    on pulse programs and we do not expect decomposition pipelines between pulses
    and gate-based circuits yet.
    """
    # pylint:disable=unused-argument
    return x


hybrid_pulse_grad = gradient_transform(
    _hybrid_pulse_grad, expand_fn=expand_invalid_trainable_hybrid_pulse_grad
)
