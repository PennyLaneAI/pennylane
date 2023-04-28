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
# from collections.abc import Sequence
import numpy as np

import pennylane as qml

# from pennylane._device import _get_num_copies
# from pennylane.pulse import ParametrizedEvolution

from .finite_difference import _all_zero_grad_new, _no_trainable_grad_new
# from .parameter_shift import (
# _reorder_grad_axes_multi_measure,
# _reorder_grad_axes_single_measure_shot_vector,
# _make_zero_rep,
# )
from .stoch_pulse_gradient import _assert_has_jax
from .gradient_transform import (
    assert_active_return,
    assert_no_state_returns,
    assert_no_variance,
    choose_grad_methods,
    gradient_analysis_and_validation,
    gradient_transform,
)

def _hybrid_pulse_grad(tape, argnum=None, shots=None):
    transform_name = "hybrid pulse parameter-shift"
    _assert_has_jax(transform_name)
    assert_active_return(transform_name)
    assert_no_state_returns(tape.measurements)
    assert_no_variance(tape.measurements, transform_name)

    if argnum is None and not tape.trainable_params:
        return _no_trainable_grad_new(tape, shots)

    #if use_broadcasting and tape.batch_size is not None:
        #raise ValueError("Broadcasting is not supported for tapes that already are broadcasted.")

    diff_methods = gradient_analysis_and_validation(tape, "analytic", grad_fn=hybrid_pulse_grad)

    if all(g == "0" for g in diff_methods):
        return _all_zero_grad_new(tape, shots)

    method_map = choose_grad_methods(diff_methods, argnum)

    argnum = [i for i, dm in method_map.items() if dm == "A"]

    return _expval_hybrid_pulse_grad(tape, argnum, shots)

def _expval_hybrid_pulse_grad(tape, argnum, shots):
    tapes = []
    gradient_data = []
    for idx, trainable_idx in enumerate(tape.trainable_params):
        if trainable_idx not in argnum:
            # Only the number of tapes is needed to indicate a zero gradient entry
            gradient_data.append((0, None, None))
            continue

        """
        key, _key = jax.random.split(key)
        cjacs, _tapes, avg_prefactor = _generate_tapes_and_cjacs(
            tape, idx, _key, num_split_times, use_broadcasting
        )

        gradient_data.append((len(_tapes), qml.math.stack(cjacs), avg_prefactor))
        tapes.extend(_tapes)

    num_measurements = len(tape.measurements)
    single_measure = num_measurements == 1
    num_params = len(tape.trainable_params)
    shot_vector = isinstance(shots, Sequence)
    tape_specs = (single_measure, num_params, num_measurements, shot_vector, shots)

    def processing_fn(results):
        start = 0
        grads = []
        for num_tapes, cjacs, avg_prefactor in gradient_data:
            if num_tapes == 0:
                grads.append(None)
                continue
            res = results[start : start + num_tapes]
            start += num_tapes
            # Apply the postprocessing of the parameter-shift rule and contract
            # with classical Jacobian, effectively computing the integral approximation
            g = _parshift_and_integrate(
                res, cjacs, avg_prefactor, single_measure, shot_vector, use_broadcasting
            )
            grads.append(g)

        # g will have been defined at least once (because otherwise all gradients would have
        # been zero), providing a representative for a zero gradient to emulate its type/shape.
        zero_rep = _make_zero_rep(g, single_measure, shot_vector)

        # Fill in zero-valued gradients
        grads = [zero_rep if g is None else g for g in grads]

        return _reorder_grads(grads, tape_specs)
    """

    return tapes, processing_fn

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
