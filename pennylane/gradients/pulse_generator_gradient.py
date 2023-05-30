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
from functools import partial
import numpy as np

import pennylane as qml

from pennylane.pulse import ParametrizedEvolution
from pennylane.ops.qubit.special_unitary import pauli_basis_strings, _pauli_decompose

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
        The ``k``\ th entry of the returned ``tuple`` has the shape ``(2**N, 2**N, *par_shape)``
        where ``N`` is the number of qubits the evolution acts on and ``par_shape`` is the
        shape of the ``k``\ th parameter of the evolution.

    The effective generator can be computed from the derivative of the unitary
    matrix corresponding to the full time evolution of a pulse:

    .. math::

        \Omega_k = U(\theta)^\dagger \frac{\partial}{\partial \theta_k} U(\theta)

    Here :math:`U(\theta)` is the unitary matrix of the time evolution due to the pulse
    and :math:`\theta` are the variational parameters of the pulse.

    See the documentation of pulse_generator for more details and a mathematical derivation.
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
        qml.tape.QuantumScript(ops_pre + [insert] + ops_post, tape.measurements) for insert in ops
    ]


def _generate_tapes_and_coeffs(tape, idx, atol, cache):
    """Compute the modified tapes and coefficients required to compute the pulse generator
    derivative of a tape with respect to an indicated trainable parameter.

    Args:
        tape (`~.QuantumTape`): Tape to differentiate.
        idx (int): Index of the parameter with respect to which to differentiate
            into ``tape.trainable_parameters``.
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
            "pulse_generator does not support differentiating parameters of "
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
