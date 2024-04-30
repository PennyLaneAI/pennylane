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
This module contains functions for computing the pulse generator
parameter-shift gradient of pulse sequences in a qubit-based quantum tape.
"""
from typing import Callable, Sequence
from functools import partial
import numpy as np

import pennylane as qml

from pennylane.pulse import ParametrizedEvolution
from pennylane.ops.qubit.special_unitary import pauli_basis_strings, _pauli_decompose
from pennylane import transform

from .parameter_shift import _make_zero_rep
from .pulse_gradient import _assert_has_jax, raise_pulse_diff_on_qnode
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

try:
    import jax
except ImportError:
    # Handling the case where JAX is not installed is done via _assert_has_jax
    pass


def _one_parameter_generators(op):
    r"""Compute the effective generators :math:`\{\Omega_k\}` of one-parameter groups that
    reproduce the partial derivatives of a parameterized evolution.
    In particular, compute :math:`U` and :math:`\partial U / \partial \theta_k`
    and recombine them into :math:`\Omega_k = U^\dagger \partial U / \partial \theta_k`

    Args:
        op (`~.ParametrizedEvolution`): Parametrized evolution for which to compute the generator

    Returns:
        tuple[tensor_like]: The generators for one-parameter groups that reproduce the
        partial derivatives of the parametrized evolution.
        The ``k``\ th entry of the returned ``tuple`` has the shape ``(*par_shape, 2**N, 2**N)``
        where ``N`` is the number of qubits the evolution acts on and ``par_shape`` is the
        shape of the ``k``\ th parameter of the evolution.

    The effective generator can be computed from the derivative of the unitary
    matrix corresponding to the full time evolution of a pulse:

    .. math::

        \Omega_k = U(\theta)^\dagger \frac{\partial}{\partial \theta_k} U(\theta)

    Here :math:`U(\theta)` is the unitary matrix of the time evolution due to the pulse
    and :math:`\theta` are the variational parameters of the pulse.

    See the documentation of pulse_odegen for more details and a mathematical derivation.
    """

    def _compute_matrix(op_data):
        """Parametrized computation of the matrix for the given pulse ``op``."""
        return op(op_data, t=op.t, **op.odeint_kwargs).matrix()

    def _compute_matrix_split(op_data):
        """Parametrized computation of the matrix for the given pulse ``op``.
        Return the real and imaginary parts separately."""
        mat = _compute_matrix(op_data)
        return mat.real, mat.imag

    # Compute the Jacobian of _compute_matrix, giving the Jacobian of the real and imag parts
    # The output is a tuple, with one entry per parameter, each of which has the axes
    # (mat_dim, mat_dim, *parameter_shape)
    jac_real, jac_imag = jax.jacobian(_compute_matrix_split)(op.data)

    # Compute the matrix of the pulse itself and conjugate it. Skip the transposition of the adjoint
    # The output has the shape (mat_dim, mat_dim)
    U_dagger = qml.math.conj(_compute_matrix([qml.math.detach(d) for d in op.data]))

    # Compute U^\dagger @ \partial U / \partial \theta_k
    # For each entry ``j`` in the tuple ``jac``,
    # 1. Contract U_dagger with j along the axes [0], [0]. This effectively transposes
    #    U_dagger, which we skipped above.
    # 2. After contraction, the axes are (mat_dim, mat_dim, *parameter_shape).
    # 3. Move the first two axis to the last two positions.
    moveax = partial(qml.math.moveaxis, source=(0, 1), destination=(-2, -1))
    return tuple(
        moveax(qml.math.tensordot(U_dagger, j_r + 1j * j_i, axes=[[0], [0]]))
        for j_r, j_i in zip(jac_real, jac_imag)
    )


def _one_parameter_paulirot_coeffs(generators, num_wires):
    r"""Compute the Pauli coefficients of effective generators. The coefficients correspond
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
        in the pulse generator differentiation pipeline.

    """
    # The generators are skew-Hermitian. Therefore _pauli_decompose will return a purely
    # imaginary tensor. Multiplying with 2i results in a real-valued tensor, which
    # we prefer over a complex-valued tensor with vanishing imaginary part
    return tuple(qml.math.real(2j * _pauli_decompose(g, num_wires)) for g in generators)


def _nonzero_coeffs_and_words(coefficients, num_wires, atol=1e-8):
    """Given a tuple of coefficient tensors with a common first axis size :math:`4^N-1`,
    where :math:`N` is the number of wires, filter out those indices for the first axis for which
    the sliced coefficients are not all zero. Return these coefficients, as well as the
    corresponding Pauli word strings on :math:`N` wires.

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
        qml.tape.QuantumScript(ops_pre + [insert] + ops_post, tape.measurements, shots=tape.shots)
        for insert in ops
    ]


def _generate_tapes_and_coeffs(tape, idx, atol, cache):
    """Compute the modified tapes and coefficients required to compute the pulse generator
    derivative of a tape with respect to an indicated trainable parameter.

    Args:
        tape (`~.QuantumTape`): Tape to differentiate.
        idx (int): Index (referring to ``tape.trainable_parameters``) of the parameter
            with respect to which to differentiate.
        atol (float): absolute tolerance used to determine whether a coefficient is zero.
        cache (dict): Caching dictionary that allows to skip adding duplicate modified tapes.

    Returns:
        list[`~.QuantumScript`]: Modified tapes to be added to the pulse generator differentiation
            tapes. It is an empty list if modified tapes were already created for another
            parameter of the pulse of interest.
        tuple[int, int, tensor_like]: Gradient computation data, consisting of the start and end
            indices into the total list of tapes as well as the coefficients that need to be
            contracted with the corresponding results to obtain the partial derivative with respect
            to the indicated trainable parameter.
        dict: Input ``cache`` dictionary. If the cache lookup missed, the cache is extended by one
            entry and its entry ``"total_num_tapes"`` is increased by the number of created tapes.
    """
    # ``op`` is the operation into which the trainable tape parameter with index ``idx`` feeds.
    # ``op_idx`` is the index of ``op`` within all tape operations. ``term_idx`` is the index
    # of the trainable tape parameter within ``op``.
    op, op_idx, term_idx = tape.get_operation(idx)
    if op_idx in cache:
        # operation was analyzed and tapes were added before. Get tape indices and coefficients.
        start, end, all_coeffs = cache[op_idx]
        return [], (start, end, all_coeffs[term_idx]), cache

    if not isinstance(op, ParametrizedEvolution):
        # only ParametrizedEvolution can be treated with this gradient transform
        raise ValueError(
            "pulse_odegen does not support differentiating parameters of "
            f"other operations than pulses, but received operation {op}."
        )

    num_wires = len(op.wires)
    # Compute the effective generators of the given pulse _for all parameters_.
    generators = _one_parameter_generators(op)
    # Obtain one-parameter-group coefficients in Pauli basis and filter for non-zero coeffs
    all_coeffs = _one_parameter_paulirot_coeffs(generators, num_wires)
    all_coeffs, pauli_words = _nonzero_coeffs_and_words(all_coeffs, num_wires, atol)
    # create PauliRot gates for each Pauli word (with a non-zero coefficient) and for both shifts
    pauli_rots = [
        qml.PauliRot(angle, word, wires=op.wires)
        for word in pauli_words
        for angle in [np.pi / 2, -np.pi / 2]
    ]
    # create tapes with the above PauliRot gates inserted, one per tape
    tapes = _insert_op(tape, pauli_rots, op_idx)
    # get the previous total number of tapes from the cache and determine start and end indices
    end = (start := cache["total_num_tapes"]) + (num_tapes := len(tapes))
    # store the tape indices and coefficients for this operation, to be retrieved from the cache
    # for other parameters of the same operation, see first if clause above.
    cache[op_idx] = (start, end, all_coeffs)
    # increment total number of tapes
    cache["total_num_tapes"] += num_tapes
    # Return the start and end pointer for the created tapes/the corresponding results.
    # Also return the coefficients for the current trainable parameter of the pulse
    return tapes, (start, end, all_coeffs[term_idx]), cache


def _parshift_and_contract(results, coeffs, single_measure, single_shot_entry):
    """Compute parameter-shift tape derivatives and contract them with coefficients.

    Args:
        results (list[tensor_like]): Tape execution results to be processed.
        coeffs (list[tensor_like]): Coefficients to be contracted.
        single_measure (bool): whether the tape execution results contain single measurements.
        single_shot_entry (bool): whether the tape execution results were obtained with a single
            shots setting.

    Returns:
        tensor_like or tuple[tensor_like] or tuple[tuple[tensor_like]]: contraction between the
        parameter-shift derivative computed from ``results`` and the ``coeffs``.
    """

    def _parshift_and_contract_single(res_list, coeffs):
        """Execute the standard parameter-shift rule on a list of results
        and contract with Pauli basis coefficients."""
        psr_deriv = ((res := qml.math.stack(res_list))[::2] - res[1::2]) / 2
        return qml.math.tensordot(psr_deriv, coeffs, axes=[[0], [0]])

    if single_measure and single_shot_entry:
        # single measurement and single shot entry
        return _parshift_and_contract_single(results, qml.math.stack(coeffs))
    if single_measure or single_shot_entry:
        # single measurement or single shot entry, but not both
        return tuple(
            _parshift_and_contract_single(r, qml.math.stack(coeffs)) for r in zip(*results)
        )

    return tuple(
        tuple(_parshift_and_contract_single(_r, qml.math.stack(coeffs)) for _r in zip(*r))
        for r in zip(*results)
    )


def _expval_pulse_odegen(tape, argnum, atol):
    """Compute the pulse generator parameter-shift rule for a quantum circuit that returns expectation
    values of observables.

    Args:
        tape (`~.QuantumTape`): Quantum circuit to be differentiated with the pulse generator
            parameter-shift rule.
        argnum (int or list[int] or None): Trainable tape parameter indices to differentiate
            with respect to. If not provided, the derivatives with respect to all
            trainable parameters are returned. Note that the indices are with respect to
            the list of trainable parameters.
        atol (float): absolute tolerance used to determine vanishing contributions.

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing a
          list of generated tapes, together with a post-processing
          function to be applied to the results of the evaluated tapes
          in order to obtain the Jacobian.

    """
    argnum = argnum or tape.trainable_params
    # Initialize a cache that will store the following:
    # 1. a single entry ``str: int`` that memorizes the number of tapes that
    #    were created overall thus far. It is initialized in the following line
    # 2. one entry per ``ParametrizedEvolution`` that contains at least one trainable
    #    parameter of the tape that also is marked as trainable as per ``argnum``.
    #    These entries have the format ``int: (int, int, tuple[tensor_like])``. The key is the
    #    index for the pulse within the tape operations. The first two entries of the value
    #    are a "start" and an "end" pointer, referencing which tapes in ``gradient_tapes``
    #    have been created for the pulse.
    #    Correspondingly, these pointers define the sub-list of results that should be used
    #    for the given pulse in the postprocessing function below.
    #    The third entry in the value tuple contains the coefficients of the effective
    #    generators of the given pulse. These are given in a tuple, with one entry per
    #    parameter of the pulse.
    #    For details on these coefficients, see _one_parameter_paulirot_coeffs.
    cache = {"total_num_tapes": 0}
    # ``gradient_data`` will contain tuples ``(int, int, tensor_like)``, with the first two
    # entries being the start/end pointers explained above, and the third entry being the
    # coefficients tensor for one particular parameter of a pulse (one entry of the coefficients
    # tuple in the corresponding ``cache`` entry).
    gradient_data = []
    gradient_tapes = []
    tape_params = tape.get_parameters()
    for idx, param in enumerate(tape_params):
        shape = qml.math.shape(param)

        if idx not in argnum:
            # Trainable parameters that are de-selected by ``argnum`` receive a vanishing
            # gradient entry.
            # Indicate that there are no tapes for this parameter by setting start==end
            gradient_data.append((0, 0, None, shape))
            continue

        # Access the pulse in the tape into which the current trainable parameter
        # feeds. If this pulse is analyzed for the first time, compute its effective
        # generators and their Pauli basis coefficients for _all_ parameters of the pulse,
        # and create the tapes of the modified cost function for all occurring Pauli words.
        # If the pulse has been analyzed before, retrieve the tape/results pointers of
        # the pulse and the coefficients belonging to the current parameter from the cache,
        # but do not add create any additional tapes.
        tapes, data, cache = _generate_tapes_and_coeffs(tape, idx, atol, cache)

        gradient_data.append((*data, shape))
        gradient_tapes.extend(tapes)

    # Extract some specifications about the original tape, required for output formatting
    num_measurements = len(tape.measurements)
    single_measure = num_measurements == 1
    num_params = len(tape.trainable_params)
    partitioned_shots = tape.shots.has_partitioned_shots
    tape_specs = (single_measure, num_params, num_measurements, tape.shots)

    def processing_fn(results):
        """Post-process the results of the parameter-shifted tapes for
        ``pulse_odegen`` into the gradient."""
        grads = []
        zero_parshapes = []
        # Iterate over gradient_data, which contains one entry for each of the trainable parameters in argnum
        for start, end, coeffs, par_shape in gradient_data:
            # If start and end pointer are equal, no tapes contribute and we get a vanishing
            # gradient for this parameter. For this, add an entry ``None``, which will
            # be replaced by the appropriately formatted zero later on. Memorize the parameter
            # shape for said formatting.
            if start == end:
                grads.append(None)
                zero_parshapes.append(par_shape)
                continue

            # Extract the results corresponding to the start and end pointer of the pulse
            # to which the current parameter belongs.
            res = results[start:end]
            # Apply the parameter-shift rule (respecting the tape output formatting)
            # and contract the result with the coefficients of the effective generators
            # in the Pauli basis. This computes the partial derivative.
            g = _parshift_and_contract(res, coeffs, single_measure, not partitioned_shots)
            grads.append(g)

            # Memorize the parameter shape for the nonzero gradient entry
            nonzero_parshape = par_shape

        # g will have been defined at least once (because otherwise all gradients would have
        # been zero), providing a representative for a zero gradient to emulate its type/shape.
        # This shape needs to be modified because pulse parameters may differ in shape.
        zero_parshapes = iter(zero_parshapes)
        for i, _g in enumerate(grads):
            if _g is None:
                par_shapes = (nonzero_parshape, next(zero_parshapes))
                # Make zero representative gradient entry, adapting the shape
                zero_rep = _make_zero_rep(g, single_measure, partitioned_shots, par_shapes)
                # Fill in zero-valued gradient entry
                grads[i] = zero_rep

        # Reformat the flat list of gradients into an output that fits the original tape specs.
        return reorder_grads(grads, tape_specs)

    return gradient_tapes, processing_fn


@partial(transform, final_transform=True)
def pulse_odegen(
    tape: qml.tape.QuantumTape, argnum=None, atol=1e-7
) -> (Sequence[qml.tape.QuantumTape], Callable):
    r"""Transform a circuit to compute the pulse generator parameter-shift gradient of pulses
    in a pulse program with respect to their inputs.
    This method combines automatic differentiation of few-qubit operations with
    hardware-compatible shift rules.
    It allows for the evaluation of parameter-shift gradients for many-qubit pulse programs
    on hardware, with the limitation that the individual pulses must be acting on a
    sufficiently small number of qubits.

    For this differentiation method, the unitary matrix :math:`U` of a pulse gate and its derivative
    :math:`\partial_k U` are computed classically with an autodiff framework.
    From :math:`\partial_k U` and :math:`U` we can deduce the so-called effective generators
    :math:`\Omega_{k}` assuming the form

    .. math:: \partial_k U = U \Omega_k.

    These effective generators are then decomposed into the Pauli basis and the
    standard parameter-shift rule is used to evaluate the derivatives of the pulse program
    in this basis.

    To this end, shifted ``PauliRot`` operations are inserted in the program.
    Finally, the Pauli basis derivatives are recombined into the gradient
    of the pulse program with respect to its original parameters.
    See the theoretical background section below for more details.

    Args:
        tape (QuantumTape): quantum circuit to differentiate
        argnum (int or list[int] or None): Trainable tape parameter indices to differentiate
            with respect to. If not provided, the derivatives with respect to all
            trainable parameters are returned. Note that the indices are with respect to
            the list of trainable parameters.
        atol (float): Precision parameter used to truncate the Pauli basis coefficients
            of the effective generators. Coefficients ``x`` satisfying
            ``qml.math.isclose(x, 0., atol=atol, rtol=0) == True`` are neglected.

    Returns:
        tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the Jacobian in the form of a tensor, a tuple, or a nested tuple depending upon the nesting
        structure of measurements in the original circuit.

    .. note::

        This function requires the JAX interface and does not work with other autodiff interfaces
        commonly encountered with PennyLane.
        In addition, this transform is only JIT-compatible with pulses that only have scalar
        parameters.

    .. warning::

        This transform may not be applied directly to QNodes. Use JAX entrypoints
        (``jax.grad``, ``jax.jacobian``, ...) instead or apply the transform on the tape
        level. Also see the examples below.

    **Example**

    Consider the parameterized Hamiltonian
    :math:`\theta_0 Y_{0}+f(\boldsymbol{\theta_1}, t) Y_{1} + \theta_2 Z_{0}X_{1}`
    with parameters :math:`\theta_0 = \frac{1}{5}`,
    :math:`\boldsymbol{\theta_1}=\left(\frac{3}{5}, \frac{1}{5}\right)^{T}` and
    :math:`\theta_2 = \frac{2}{5}`, the time-dependent function
    :math:`f(\boldsymbol{\theta_1}, t) = \theta_{1,0} t + \theta_{1,1}`
    as well as a time interval :math:`t=\left[\frac{1}{10}, \frac{9}{10}\right]`.

    .. code-block:: python

        from jax import numpy as jnp
        jax.config.update("jax_enable_x64", True)
        H = (
            qml.pulse.constant * qml.Y(0)
            + jnp.polyval * qml.Y(1)
            + qml.pulse.constant * (qml.Z(0) @ qml.X(1))
        )
        params = [jnp.array(0.2), jnp.array([0.6, 0.2]), jnp.array(0.4)]
        t = [0.1, 0.9]

    For simplicity, consider a pulse program consisting of this single pulse and a
    measurement of the expectation value of :math:`X_{0}`.

    .. code-block:: python

        dev = qml.device("default.qubit.jax")

        @qml.qnode(dev, interface="jax", diff_method=qml.gradients.pulse_odegen)
        def circuit(params):
            op = qml.evolve(H)(params, t)
            return qml.expval(qml.X(0))

    We registered the ``QNode`` to be differentiated with the ``pulse_odegen`` method.
    This allows us to simply differentiate it with ``jax.grad``, which internally
    makes use of the pulse generator parameter-shift method.

    >>> jax.grad(circuit)(params)
    [Array(1.41897932, dtype=float64, weak_type=True),
     Array([0.00164913, 0.00284788], dtype=float64),
     Array(-0.09984584, dtype=float64, weak_type=True)]

    Alternatively, we may apply the transform to the tape of the pulse program, obtaining
    the tapes with inserted ``PauliRot`` gates together with the post-processing function:

    >>> circuit.construct((params,), {}) # Build the tape of the circuit.
    >>> circuit.tape.trainable_params = [0, 1, 2]
    >>> tapes, fun = qml.gradients.pulse_odegen(circuit.tape, argnum=[0, 1, 2])
    >>> len(tapes)
    12

    Why are there :math:`12` tapes?
    Consider the terms in the time-dependent pulse Hamiltonian: :math:`\{Y_0, Y_1, Z_0X_1\}`.
    Via the Lie bracket, which is just the standard matrix commutator, they
    generate an algebra, the so-called *dynamical Lie algebra (DLA)* of the pulse.
    In order to find all Pauli words that occur in the DLA, we need to (recursively)
    calculate all possible commutators between the three words above and their
    commutators. For the three words above, we obtain three additional words:

    .. math::

        [Y_0, Z_0X_1] &\propto X_0X_1 \\
        [Y_1, Z_0X_1] &\propto Z_0Z_1 \\
        [Y_0, Z_0Z_1] &\propto X_0Z_1

    All other commutators result in expressions proportional to one of the six Pauli words.
    For each of these six words, we need to compute the standard parameter-shift rule
    requiring two shifted circuits, which yields :math:`12` tapes.

    We may inspect one of the tapes, which differs from the original tape by the inserted
    rotation gate ``"RIY"``, i.e. a ``PauliRot(np.pi/2, "IY", wires=[0, 1])`` gate.
    Note that the order of the tapes follows lexicographical ordering of the inserted
    Pauli rotations, so that :math:`Y_1` is the first of the six words.

    >>> print(qml.drawer.tape_text(tapes[0]))
    0: ─╭RIY─╭ParametrizedEvolution─┤  <X>
    1: ─╰RIY─╰ParametrizedEvolution─┤

    Executing the tapes and applying the post-processing function to the results then
    yields the gradient:

    >>> fun(qml.execute(tapes, dev))
    (Array(1.41897932, dtype=float64),
     Array([0.00164913, 0.00284788], dtype=float64),
     Array(-0.09984584, dtype=float64))

    .. note::

        For pulse Hamiltonians with complex generating terms and few parameters,
        the decomposition approach taken in this method may incur more
        (quantum and classical) computational cost than strictly necessary.

    .. details::
        :title: Theoretical background
        :href: theory

        The pulse generator parameter-shift gradient method makes use of the *effective generator*
        of a pulse for given parameters and duration. Consider the parametrized Hamiltonian

        .. math::

            H(\boldsymbol{\theta}, t) = \sum_{k=1}^K f_k(\boldsymbol{\theta}, t) H_k

        where the Hamiltonian terms :math:`\{H_k\}` are constant and the :math:`\{f_k\}` are
        parametrized time-dependent functions depending on the parameters
        :math:`\boldsymbol{\theta}`.
        The unitary time evolution operator associated with :math:`H` is the solution to the
        Schrödinger equation

        .. math::

            \frac{\mathrm{d} U}{\mathrm{d} t}(t) =
            -i H(\boldsymbol{\theta}, t) U(t), \quad U(0) = \mathbb{I}

        For a fixed time interval :math:`[t_0, t_1]`, we associate a matrix function
        :math:`U(\boldsymbol{\theta})` with the unitary evolution.
        To compute the pulse generator parameter-shift gradient, we are interested in the partial
        derivatives of this matrix function, usually with respect to the parameters
        :math:`\boldsymbol{\theta}`. Provided that :math:`H` does not act on too many qubits,
        or that we have an alternative sparse representation of
        :math:`U(\boldsymbol{\theta})`, we may compute these partial derivatives

        .. math::

            \frac{\partial U(\boldsymbol{\theta})}{\partial \theta_{k}}

        classically via automatic differentiation, where :math:`\theta_{k}` is
        the :math:`k`\ -th (scalar) parameter in :math:`\boldsymbol{\theta}`.

        Now, due to the compactness of the groups :math:`\mathrm{SU}(N)`\ , we know that
        for each :math:`\theta_{k}` there is an *effective generator* :math:`\Omega_{k}`
        such that

        .. math::

            \frac{\partial U(\boldsymbol{\theta})}{\partial \theta_{k}} =
            U(\boldsymbol{\theta})\Omega_{k}.

        Given that we can compute the left-hand side expression as well as the matrix
        for :math:`U` itself, we can compute :math:`\Omega_{k}` for all parameters
        :math:`\theta_{k}`.
        In addition, we may decompose these generators into Pauli words:

        .. math::

            \Omega_{k} = \sum_{\ell=1}^{L} \omega_{k}^{(\ell)} P_{\ell}

        The coefficients :math:`\omega_{k}^{(\ell)}` of the generators can be computed
        by decomposing the anti-Hermitian matrix :math:`\Omega_{k}` into the Pauli
        basis and only keeping the non-vanishing terms. This is possible via a tensor
        contraction with the full Pauli basis (or alternative, more efficient methods):

        .. math::

            \omega_{k}^{(\ell)} = \frac{1}{2^N}\mathrm{Tr}\left[P_\ell \Omega_{k}\right]

        where :math:`N` is the number of qubits and :math:`\ell = 1, .. , L` the Pauli word index.
        The number of non-zero Pauli words :math:`L` is typically equal to the dimension of the dynamical Lie algebra
        (can be lower if coefficients happen to be zero)
        and at most :math:`4^N-1`.

        Thus far, we discussed the derivative of the time evolution, or pulse.
        Now, consider an objective function that is based on measuring an expectation
        value after executing a pulse program:

        .. math::
            C(\boldsymbol{\theta})=
            \langle\psi_0|U(\boldsymbol{\theta})^\dagger B
            U(\boldsymbol{\theta}) |\psi_0\rangle

        Using the derivative of :math:`U` and the decomposition of the effective
        generator :math:`\Omega_k` above, we calculate the partial derivative of
        :math:`C`:

        .. math::

            \frac{\partial C}{\partial \theta_{k}} (\boldsymbol{\theta})&=
            \langle\psi_0|\left[U^\dagger B U, \Omega_{k}\right]|\psi_0\rangle\\
            &=\sum_{\ell=1}^L \omega_{k}^{(\ell)}
            \langle\psi_0|\left[U^\dagger B U, P_\ell \right]|\psi_0\rangle\\
            &=\sum_{\ell=1}^L \tilde\omega_{k}^{(\ell)}
            \langle\psi_0|\left[U^\dagger B U, -\frac{i}{2}P_\ell \right]|\psi_0\rangle\\
            &=\sum_{\ell=1}^L \tilde\omega_{k}^{(\ell)}
            \frac{\mathrm{d}}{\mathrm{d}x}
            \langle\psi_0|\exp\left(i\frac{x}{2}P_\ell \right)U^\dagger B
            U\exp\left(-i\frac{x}{2}P_\ell \right)|\psi_0\rangle\large|_{x=0}\\
            &=\sum_{\ell=1}^L \tilde\omega_{k}^{(\ell)}
            \frac{\mathrm{d}}{\mathrm{d}x} C_\ell(x)\large|_{x=0}

        where we skipped the argument :math:`\boldsymbol{\theta}` of :math:`U` for readability
        and introduced the modified coefficients
        :math:`\tilde\omega_{k}^{(\ell)}=2i\omega_{k}^{(\ell)}`.
        In the second to last step, we rewrote the commutator of :math:`U^\dagger BU` and
        :math:`\frac{i}{2}P_\ell` as the derivative (at zero) of a modified cost function
        :math:`C_\ell(x)` that executes a Pauli rotation about :math:`-i\frac{x}{2}P_\ell`
        before the parametrized time evolution. Here, the variable :math:`x` is just a
        convenient way to write the modified cost function. Note that its derivative with
        respect to :math:`x` can be computed with the standard two-term parameter-shift
        rule for Pauli rotation gates, i.e.

        .. math::

            \frac{\mathrm{d}}{\mathrm{d}x} C_\ell(x) {\large|}_{x=0} = \frac{1}{2} \left(C_\ell(\pi/2) - C_\ell(-\pi/2)\right)

        with :math:`C_\ell(x) = \langle\psi_0|e^{i\frac{x}{2}P_\ell} U^\dagger B U e^{-i\frac{x}{2}P_\ell} |\psi_0\rangle`.

        **Caching**

        Considering the derivation above, we notice that the same modified cost function
        :math:`C_\ell(x)` may appear in the derivatives of distinct parameters
        :math:`\theta_k` and :math:`\theta_m`, because they are shared by two terms in the pulse Hamiltonian.
        In order to not evaluate the same
        modified quantum circuit derivatives multiple times, we use an internal
        cache that avoids repeated creation of the same parameter-shifted circuits.
        In addition, all modified cost functions :math:`C_\ell` that would be multiplied
        with a vanishing coefficient :math:`\tilde\omega_{k}^{(\ell)}` *for all* :math:`k`
        are skipped altogether.
        This approach requires a few additional classical coprocessing steps but allows
        us to save quantum resources in many relevant pulse programs.

    """
    transform_name = "pulse generator parameter-shift"
    _assert_has_jax(transform_name)
    assert_no_state_returns(tape.measurements, transform_name)
    assert_no_variance(tape.measurements, transform_name)
    assert_no_trainable_tape_batching(tape, transform_name)

    if argnum is None and not tape.trainable_params:
        return _no_trainable_grad(tape)

    trainable_params = choose_trainable_params(tape, argnum)
    diff_methods = find_and_validate_gradient_methods(tape, "analytic", trainable_params)

    if all(g == "0" for g in diff_methods.values()):
        return _all_zero_grad(tape)

    argnum = [i for i, dm in diff_methods.items() if dm == "A"]

    return _expval_pulse_odegen(tape, argnum, atol)


@pulse_odegen.custom_qnode_transform
def pulse_odegen_qnode_wrapper(self, qnode, targs, tkwargs):
    """A custom QNode wrapper for the gradient transform :func:`~.pulse_odegen`.
    It raises an error, so that applying ``pulse_odegen`` to a ``QNode`` directly
    is not supported.
    """
    # pylint:disable=unused-argument
    transform_name = "pulse generator parameter-shift"
    raise_pulse_diff_on_qnode(transform_name)
