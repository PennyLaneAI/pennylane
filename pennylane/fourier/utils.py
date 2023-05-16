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

"""Contains utility functions for the Fourier module."""

from itertools import combinations
import numpy as np


import pennylane as qml


def format_nvec(nvec):
    """Nice strings representing tuples of integers."""

    if isinstance(nvec, int):
        return str(nvec)

    return " ".join(f"{n: }" for n in nvec)


def get_spectrum(op, decimals):
    r"""Extract the frequencies contributed by an input-encoding gate to the
    overall Fourier representation of a quantum circuit.

    If :math:`G` is the generator of the input-encoding gate :math:`\exp(-i x G)`,
    the frequencies are the differences between any two of :math:`G`'s eigenvalues.
    We only compute non-negative frequencies in this subroutine.

    Args:
        op (~pennylane.operation.Operation): Operation to extract
            the frequencies for
        decimals (int): Number of decimal places to round the frequencies to

    Returns:
        set[float]: non-negative frequencies contributed by this input-encoding gate
    """
    matrix = qml.matrix(qml.generator(op, format="observable"))

    # todo: use qml.math.linalg once it is tested properly
    evals = np.linalg.eigvalsh(matrix)

    # compute all unique positive differences of eigenvalues, then add 0
    # note that evals are sorted already
    _spectrum = set(np.round([x[1] - x[0] for x in combinations(evals, 2)], decimals=decimals))
    _spectrum |= {0}

    return _spectrum


def join_spectra(spec1, spec2):
    r"""Join two sets of frequencies that belong to the same input.

    Since :math:`\exp(i a x)\exp(i b x) = \exp(i (a+b) x)`, the spectra of two gates
    encoding the same :math:`x` are joined by computing the set of sums and absolute
    values of differences of their elements.
    We only compute non-negative frequencies in this subroutine and assume the inputs
    to be non-negative frequencies as well.

    Args:
        spec1 (set[float]): first spectrum
        spec2 (set[float]): second spectrum
    Returns:
        set[float]: joined spectrum
    """
    if spec1 == {0}:
        return spec2
    if spec2 == {0}:
        return spec1

    sums = set()
    diffs = set()

    for s1 in spec1:
        for s2 in spec2:
            sums.add(s1 + s2)
            diffs.add(np.abs(s1 - s2))

    return sums.union(diffs)
