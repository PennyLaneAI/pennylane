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
from collections.abc import Sequence
import numpy as np

import pennylane as qml

from pennylane._device import _get_num_copies
from pennylane.pulse import ParametrizedEvolution
from pennylane.ops.qubit.special_unitary import pauli_basis_matrices, pauli_basis_strings

from .finite_difference import _all_zero_grad_new, _no_trainable_grad_new
from .parameter_shift import (
    _make_zero_rep,
    _reorder_grads,
)
from .stoch_pulse_gradient import _assert_has_jax, _split_evol_tapes
from .gradient_transform import (
    assert_active_return,
    assert_no_state_returns,
    assert_no_variance,
    choose_grad_methods,
    gradient_analysis_and_validation,
    gradient_transform,
)

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    # Handling the case where JAX is not installed is done via _assert_has_jax
    pass


def _get_one_parameter_generators(op):
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
    # theta = self.data[0]
    # if len(qml.math.shape(theta)) > 1:
    # raise ValueError("Broadcasting is not supported.")

    num_wires = len(op.wires)

    def _compute_matrix(*args, **kwargs):
        return op(*args, **kwargs).matrix()

    data = [1.0 + 0.0j * d for d in op.data]
    # These lines compute the Jacobian of compute_matrix every time -> to be optimized
    jac = jax.jacobian(_compute_matrix, argnums=0, holomorphic=True)(data, t=op.t)

    # Compute the Omegas from the Jacobian.
    U_dagger = qml.math.conj(_compute_matrix([qml.math.detach(d) for d in data], t=op.t))
    # After contracting, move the parameter derivative axis to the first position
    return tuple(
        [
            qml.math.moveaxis(qml.math.tensordot(U_dagger, j, axes=[[0], [0]]), (0, 1), (-2, -1))
            for j in jac
        ]
    )


def _get_one_parameter_paulirot_coeffs(op):
    r"""Compute the Pauli coefficients of the one-parameter group generators that reproduce
    the partial derivatives of a parametrized evolution. The coefficients correspond
    to the decomposition into (rescaled) Pauli word generators as used by ``PauliRot``
    gates.

    Args:
        op (`~.ParametrizedEvolution`): Parametrized evolution for which to compute the coefficients

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
    generators = _get_one_parameter_generators(op)
    num_wires = len(op.wires)
    basis = pauli_basis_matrices(num_wires)
    return tuple(
        [
            qml.math.real(2j * qml.math.tensordot(basis, g, axes=[[1, 2], [-1, -2]]))
            / 2**num_wires
            for g in generators
        ]
    )


def _get_nonzero_coeffs_and_words(coefficients, num_wires, atol=1e-8, rtol=0.0):
    """Given a tuple of coefficient tensors with a common first axis size :math:`4^N-1`,
    where :math:`N` is the number of wires, filter out those indices for the first axis for which
    the sliced coefficients are not all zero. Return these coefficients, as well as the
    corresponding Pauli word strings on :math:`N` wires for these indices.

    Args:
        coefficients(tuple[tensor_like]): Coefficients to filter.
        num_wires (int): Number of qubits the generators act on.
        atol (float): absolute tolerance used to determine whether a tensor is zero.
        rtol (float): relative tolerance used to determine whether a tensor is zero.
    """
    words = pauli_basis_strings(num_wires)
    new_coefficients = [[] for _ in coefficients]
    new_words = []
    for word, *coeffs in zip(words, *coefficients):
        if all(qml.math.allclose(c, 0.0, atol=atol, rtol=rtol) for c in coeffs):
            continue
        new_words.append(word)
        for new_coeffs, c in zip(new_coefficients, coeffs):
            new_coeffs.append(c)
    return new_coefficients, new_words


def _insert_op(tape, insert_ops, op_idx):
    """Create new tapes, each with an additional operation from a provided list inserted at
    the specified position. Creates as many new tapes as there are operations provided.

    Args:
        tape (`~.QuantumTape`): Original tape.
        insert_ops (list[qml.Operation]): Operations to insert (individually) at ``op_idx``
            to produce a new tape each.
        op_idx (int): Index at which to insert the operations in ``insert_ops``.

    Returns:
        list[`~.QuantumScript`]: new tapes with inserted operations, one tape per operation
        in ``insert_ops``.
    """
    ops_pre = tape.operations[:op_idx]
    ops_post = tape.operations[op_idx:]
    return [
        qml.tape.QuantumScript(ops_pre + [insert] + ops_post, tape.measurements)
        for insert in insert_ops
    ]


def _generate_tapes_and_coeffs(tape, idx, atol, rtol, cache):
    """Compute the modified tapes and coefficients required to compute the hybrid pulse
    derivative of a tape with respect to an indicated trainable parameter.

    Args:
        tape (`~.QuantumTape`): Tape to differentiate.
        idx (int): Index of the parameter with respect to which to differentiate
            into ``tape.trainable_parameters``.
        atol (float): absolute tolerance used to determine whether a coefficient is zero.
        rtol (float): relative tolerance used to determine whether a coefficient is zero.
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

    # Obtain one-parameter-group coefficients in Pauli basis and filter for non-zero coeffs
    all_coeffs = _get_one_parameter_paulirot_coeffs(op)
    all_coeffs, pauli_words = _get_nonzero_coeffs_and_words(all_coeffs, len(op.wires), atol, rtol)
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


def _expval_hybrid_pulse_grad(tape, argnum, shots, atol, rtol):
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
        rtol (float): relative tolerance used to determine vanishing contributions.

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
    gradient_data = []
    gradient_tapes = []
    cache = {"total_num_tapes": 0}
    for idx, trainable_idx in enumerate(tape.trainable_params):
        if trainable_idx not in argnum:
            # Indicate that there are no tapes for this parameter by setting start==end
            gradient_data.append((0, 0, None))
            continue

        tapes, data, cache = _generate_tapes_and_coeffs(tape, idx, atol, rtol, cache)
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

        return _reorder_grads(grads, tape_specs)

    return gradient_tapes, processing_fn


def _hybrid_pulse_grad(tape, argnum=None, shots=None, tolerances=(1e-7, 0.0)):
    """Super amazing docstring."""
    transform_name = "hybrid pulse parameter-shift"
    _assert_has_jax(transform_name)
    assert_active_return(transform_name)
    assert_no_state_returns(tape.measurements)
    assert_no_variance(tape.measurements, transform_name)

    if argnum is None and not tape.trainable_params:
        return _no_trainable_grad_new(tape, shots)

    diff_methods = gradient_analysis_and_validation(tape, "analytic", grad_fn=hybrid_pulse_grad)

    if all(g == "0" for g in diff_methods):
        return _all_zero_grad_new(tape, shots)

    method_map = choose_grad_methods(diff_methods, argnum)

    argnum = [i for i, dm in method_map.items() if dm == "A"]

    return _expval_hybrid_pulse_grad(tape, argnum, shots, *tolerances)


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
