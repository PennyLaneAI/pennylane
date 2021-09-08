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
"""Contains a transform that computes the frequency spectrum of a quantum
circuit."""
from itertools import chain, combinations
from functools import wraps
import warnings
import numpy as np
import pennylane as qml


def _get_spectrum(op):
    no_generator = False
    if hasattr(op, "generator"):
        g, coeff = op.generator

        if isinstance(g, np.ndarray):
            matrix = g
        elif hasattr(g, "matrix"):
            matrix = g.matrix
        else:
            no_generator = True
    else:
        no_generator = True

    if no_generator:
        raise ValueError(f"Generator of operation {op} is not defined.")

    matrix = coeff * matrix
    # eigenvalues of hermitian ops are guaranteed to be real
    # todo: use qml.math.linalg once it is tested properly
    evals = qml.math.real(np.linalg.eigvals(matrix))

    # compute all differences of eigenvalues
    unique_frequencies = set(
        chain.from_iterable(
            np.round((x[1] - x[0], x[0] - x[1]), decimals=8) for x in combinations(evals, 2)
        )
    )
    unique_frequencies = unique_frequencies.union({0})
    return sorted(unique_frequencies)


def _join_spectra(spec1, spec2):
    if spec1 == []:
        return sorted(set(spec2))
    if spec2 == []:
        return sorted(set(spec1))

    sums = [s1 + s2 for s1 in spec1 for s2 in spec2]
    return sorted(set(sums))


def _get_and_validate_classical_jacobians(qnode, *args, **kwargs):
    try:
        zeros_args = (np.zeros_like(arg) for arg in args)
        ones_args = (np.ones_like(arg) for arg in args)
        frac_args = (np.ones_like(arg)*0.315 for arg in args)
        jacs = [
            qml.transforms.classical_jacobian(qnode)(*_args, **kwargs)
            for _args in [zeros_args, ones_args, frac_args, args]
        ]
    except Exception as e:
        raise ValueError("Unable to compute jacobian of the classical preprocessing.") from e

    if not all((
        all((np.allclose(jacs[0][i], jac[i]) for jac in jacs[1:])) 
        for i in range(len(jacs[0]))
    )):
        raise ValueError(
            "The classical preprocessing in the provided qnode is not constant; "
            "only linear classical preprocessing is supported."
        )

    return jacs[0]


def spectrum(qnode, encoding_idx=None, encoding_gates=None, decimals=5):

    if np.isscalar(encoding_idx):
        encoding_idx = [encoding_idx]

    if encoding_gates is not None:
        if encoding_idx is not None:
            warnings.warn(
                "The argument encoding_gates is no longer valid and will be removed in"
                f" future versions. Ignoring encoding_gates={encoding_gates}..."
            )
        else:
            warnings.warn(
                "The argument encoding_gates is no longer valid and will be removed in"
                f" future versions. Trying to call spectrum with encoding_idx={encoding_gates}..."
            )
            try:
                encoding_idx = list(set(map(int, encoding_gates)))
            except ValueError as e:
                failing_id = ' '.join(str(e).split(' ')[7:])
                raise ValueError(
                    "The provided encoding_gates could not be used as encoding_idx."
                    f" Conversion to integers failed on {failing_id}."
                )

    atol = 10**(-decimals) if decimals is not None else 1e-10

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        nonlocal encoding_idx
        # Compute classical jacobian and assert preprocessing is linear 
        class_jacs = _get_and_validate_classical_jacobians(qnode, *args, **kwargs)
        if encoding_idx is None:
            # If no encoding_idx are given, all qnode arguments are considered
            encoding_idx = list(range(len(args)))
        # A map between jacobians (contiguous) and arg indices (may be discontiguous)
        arg_idx_map = {i: arg_idx for i, arg_idx in enumerate(encoding_idx)}
        # Initialize spectra for all requested parameters
        spectra = {arg_idx: {} for arg_idx in encoding_idx}

        tape = qnode.qtape

        for jac_idx, class_jac in enumerate(class_jacs):
            _spectra = np.zeros(class_jac.shape+(1,))
            for i, jac_of_op in enumerate(class_jac):
                # Find the operation that belongs to the current jac_of_op in the jacobian
                op = tape._par_info[i]["op"]
                # Multi-parameter gates are not supported
                if len(op.parameters) != 1:
                    raise ValueError(
                        "Can only consider one-parameter gates as data-encoding gates; "
                        f"got {op.name}."
                    )
                # Find parameters feeding into the current operation and if there are none, continue
                arr_ids = np.where(jac_of_op)
                if len(arr_ids[0])==0:
                    continue
                # Get the spectrum of the current operation
                spec = _get_spectrum(op)
                for arr_idx in zip(arr_ids):
                    # Rescale the operation spectrum 
                    scaled_spec = [float(jac_of_op[arr_idx])*f for f in spec]
                    # Join the new spectrum with the previously known spectrum for the parameter
                    _spectra[arr_idx] = _join_spectra(_spectra[arr_idx], scaled_spec)

            # Round frequencies if decimals for rounding are given
            #if decimals is not None:
                #np.round(_spectra, decimals, out=_spectra)
                #_spectra = {
                    #col_idx: sorted(set(np.round(spec, decimals))) 
                    #for col_idx, spec in _spectra.items()
                #}

            spectra[arg_idx_map[jac_idx]] = _spectra

        return spectra

    return wrapper
