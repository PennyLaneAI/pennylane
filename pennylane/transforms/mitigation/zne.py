# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for zero-noise extrapolation."""

from numpy.polynomial.polynomial import Polynomial

from pennylane.tape import get_active_tape
from pennylane.math import stack, arange

def _fit_zne(x_values, energies, degree=1):
    """Fit a polynomial to the energy values to extrapolate down to the
    zero-noise limit.

    Args:
        v_values (np.array): The x-axis values.
        energies (np.array): The set of energies for increasing number of
            CNOT pair insertions.
        degree (int): The degree of the polynomial to use for the fit.

    Returns:
        A polynomial of the specified degree that is the best fit to the data.
    """
    return Polynomial.fit(x_values, energies, degree, full=True)


def _generate_transformed_tapes(tape, transform, max_arg_val):
    """ Given a tape, transform, and max value of the transform argument,
    construct and return the set of tapes that need to be executed.

    """
    current_tape = get_active_tape()

    # TODO: figure out how to make it use the argument
    if current_tape is not None:
        with current_tape.stop_recording():
            tapes = [transform.tape_fn(tape) for arg in range(1, max_arg_val+1)]
    else:
        tapes = [transform.tape_fn(tape) for arg in range(1, max_arg_val+1)]

    return tapes


def zne(qnode, mitigation_transform, max_arg_val):
    """Given a tape and a mitigation transform, return the zero-extrapolated
    value computed according to the functionality of the provided transform.
    """

    def _zne_function(*args, **kwargs):
        qnode.construct(args, kwargs)
        original_tape = qnode.qtape

        transformed_tapes = _generate_transformed_tapes(
            original_tape, mitigation_transform, max_arg_val
        )

        res = stack(
            [t.execute(device=qnode.device) for t in transformed_tapes]
        ).reshape(len(transformed_tapes))

        poly_results = _fit_zne(arange(1, max_arg_val + 1), res)

        # Return the value of the extrapolated function at 0
        return poly_results[0](0)

    return _zne_function
