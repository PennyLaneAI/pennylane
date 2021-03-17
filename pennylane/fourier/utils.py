# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from itertools import product


def to_dict(coeffs):

    # infer hyperparameters
    degree = coeffs.shape[0] // 2 - 1
    n_inputs = len(coeffs.shape)

    # create generator for indices nvec = (n1, ..., nN),
    # ranging from (-d,...,-d) to (d,...,d).
    n_range = np.array(range(-degree, degree+1))
    n_ranges = [n_range] * n_inputs
    nvecs = product(*n_ranges)

    return {nvec: coeffs[nvec] for nvec in nvecs}
