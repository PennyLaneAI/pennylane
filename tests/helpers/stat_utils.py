# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Helper functions for testing stochastic processes.
"""
import numpy as np
from scipy.stats import fisher_exact


def fisher_exact_test(actual, expected, outcomes=(0, 1), threshold=0.1):
    """Checks that a binary sample matches the expected distribution using the Fisher exact test."""

    actual, expected = np.asarray(actual), np.asarray(expected)
    contingency_table = np.array(
        [
            [np.sum(actual == outcomes[0]), np.sum(actual == outcomes[1])],
            [np.sum(expected == outcomes[0]), np.sum(expected == outcomes[1])],
        ]
    )
    _, p_value = fisher_exact(contingency_table)
    assert p_value > threshold, "The sample does not match the expected distribution."
