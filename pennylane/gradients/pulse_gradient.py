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
from copy import copy
import numpy as np

import pennylane as qml
from pennylane._device import _get_num_copies
from pennylane.measurements import MutualInfoMP, StateMP, VarianceMP, VnEntropyMP
from pennylane.pulse import ParametrizedEvolution

from .finite_difference import _all_zero_grad_new, _no_trainable_grad_new
from .parameter_shift import (
    _reorder_grad_axes_multi_measure,
    _reorder_grad_axes_single_measure_shot_vector,
)
from .gradient_transform import (
    choose_grad_methods,
    grad_method_validation,
    gradient_analysis,
    gradient_transform,
)

has_jax = True
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    has_jax = False


def split_evol_ops(op, word, key):
    """Create a randomly split-time evolution with inserted Pauli rotations between
    the two split up parts.

    Args:
        op (ParametrizedEvolution): The operation to split up.
        word (str): The Pauli word about which to rotate between the split up parts.
        key (list[int]): The randomness seed to pass to the sampler for sampling the split-up point.

    Returns:
        float: The sampled split-up time tau
        tuple[list[`~.Operation`]]: The split-time evolution, expressed as three operations in the
            inner lists. The first tuple entry contains the operations with positive Pauli rotation
            angle, the second entry the operations with negative Pauli rotation angle.
    """
    before, after = copy(op), copy(op)
    # Extract time interval, split it, and set the intervals of the copies to the split intervals
    t0, t1 = op.t
    tau = jax.random.uniform(key) * (t1 - t0) + t0
    before.t = jnp.array([t0, tau])
    after.t = jnp.array([tau, t1])
    # Create Pauli rotations to be inserted at tau
    evolve_plus = qml.PauliRot(jnp.pi / 2, word, wires=op.wires)
    evolve_minus = qml.PauliRot(-jnp.pi / 2, word, wires=op.wires)
    # Construct gate sequences of split intervals and inserted Pauli rotations
    ops = ([before, evolve_plus, after], [before, evolve_minus, after])
    return tau, ops


def split_evol_tapes(tape, split_evolve_ops, op_idx):
    """Make two tapes out of one by replacing the operation indicated by op_idx by its split-time
    evolution, once using the positive Pauli rotation angle operations and once the negative
    angle operations.

    Args:
        tape (QuantumTape): The original tape
        split_evolve_ops (tuple[list[qml.Operation]]): The time-split evolution operations as created by
            ``split_evol_ops``.
        op_idx (int): The operation index within the tape.

    Returns:
        list[QuantumTape]: The two new tapes with split time evolutions and inserted Pauli rotations.
    """
    tapes = []
    for split in split_evolve_ops:
        # Replace the indicated operation by its split-up variant
        ops = tape.operations[:op_idx] + split + tape.operations[op_idx + 1 :]
        tapes.append(qml.tape.QuantumScript(ops, tape.measurements))
    return tapes


def _stoch_pulse_grad(tape, argnum=None, num_samples=1, sampler_seed=None, shots=None, **kwargs):
    r"""Compute the gradient of a quantum circuit composed of pulse sequences by applying the
    stochastic parameter shift rule.

    - references
    - example(s)
    - interface restriction: JAX
    - sampling stuff
    """
    # pylint:disable=unused-argument
    if not has_jax:
        raise ImportError(
            "Module jax is required for the ``stoch_pulse_grad`` gradient transform. "
            "You can install jax via: pip install jax jaxlib"
        )
    if any(isinstance(m, VarianceMP) for m in tape.measurements):
        raise ValueError(
            "Computing the gradient of variances with the stochastic pulse parameter-shift gradient is not implemented."
        )
    if any(isinstance(m, (StateMP, VnEntropyMP, MutualInfoMP)) for m in tape.measurements):
        raise ValueError(
            "Computing the gradient of circuits that return the state is not supported."
        )

    if num_samples < 1:
        raise ValueError(
            f"Expected a positive number of samples for the stochastic pulse parameter-shift gradient, got {num_samples}."
        )

    if argnum is None and not tape.trainable_params:
        return _no_trainable_grad_new(tape, shots)

    gradient_analysis(tape, grad_fn=stoch_pulse_grad)
    method = "analytic"
    diff_methods = grad_method_validation(method, tape)

    if all(g == "0" for g in diff_methods):
        return _all_zero_grad_new(tape, shots)

    method_map = choose_grad_methods(diff_methods, argnum)

    argnum = [i for i, dm in method_map.items() if dm == "A"]

    sampler_seed = sampler_seed or np.random.randint(18421)
    key = jax.random.PRNGKey(sampler_seed)

    return _expval_stoch_pulse_grad(tape, argnum, num_samples, key, shots)


def _expval_stoch_pulse_grad(tape, argnum, num_samples, key, shots):
    r"""Compute the gradient of a quantum circuit composed of pulse sequences that measures
    an expectation value or probabilities, by applying the stochastic parameter shift rule.
    This function is adapted to the new return type system.

    Args:

    Returns:


    """
    tapes = []
    gradient_data = []
    for idx, trainable_idx in enumerate(tape.trainable_params):
        if trainable_idx not in argnum:
            gradient_data.append((0, [], 0))  # To do
            continue

        op, op_idx, term_idx = tape.get_operation(idx, return_op_index=True)
        if not isinstance(op, ParametrizedEvolution):
            raise ValueError(
                "stoch_pulse_grad does not support differentiating parameters of other operations than pulses."
            )

        this_tapes = []
        coeff, ham = op.H.coeffs_parametrized[term_idx], op.H.ops_parametrized[term_idx]
        word = qml.pauli.pauli_word_to_string(ham)
        cjac_fn = jax.jacobian(coeff, argnums=0)
        this_cjacs = []
        for _ in range(num_samples):
            key, _key = jax.random.split(key)
            tau, split_evolve_ops = split_evol_ops(op, word, _key)
            this_cjacs.append(cjac_fn(op.data[term_idx], tau))
            this_tapes.extend(split_evol_tapes(tape, split_evolve_ops, op_idx))
        gradient_data.append((len(this_tapes), jnp.stack(this_cjacs)))
        tapes.extend(this_tapes)

    def processing_fn(results):
        # pylint: disable=protected-access
        scalar_qfunc_output = tape._qfunc_output is not None and not isinstance(
            tape._qfunc_output, Sequence
        )
        if scalar_qfunc_output:
            results = [jnp.squeeze(res) for res in results]
        start = 0
        grads = []
        for num_tapes, cjacs in gradient_data:
            if num_tapes == 0:
                raise NotImplementedError("not implemented.")
            res = results[start : start + num_tapes]
            start += num_tapes
            diff = jnp.array(res[::2]) - jnp.array(res[1::2])
            grads.append(jnp.tensordot(diff, cjacs, axes=[[0], [0]]) / num_samples)

        num_measurements = len(tape.measurements)
        single_measure = num_measurements == 1
        num_params = len(tape.trainable_params)
        if single_measure and num_params == 1:
            return grads[0]
        shot_vector = isinstance(shots, Sequence)
        len_shot_vec = _get_num_copies(shots) if shot_vector else None
        if single_measure and shot_vector:
            return _reorder_grad_axes_single_measure_shot_vector(grads, num_params, len_shot_vec)
        if not single_measure:
            return _reorder_grad_axes_multi_measure(
                grads,
                num_params,
                num_measurements,
                len_shot_vec,
                shot_vector,
            )
        return tuple(grads)

    return tapes, processing_fn


def expand_invalid_trainable_stoch_pulse_grad(x, *args, **kwargs):
    """Do not expand anything."""
    # pylint:disable=unused-argument
    return x


stoch_pulse_grad = gradient_transform(_stoch_pulse_grad, expand_fn=expand_invalid_trainable_stoch_pulse_grad)
