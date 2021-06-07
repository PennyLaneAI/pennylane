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
"""Unit tests for the basic functions in qml.math
"""
import numpy as np
import pytest

from pennylane import math as fn


class TestFrobeniusInnerProduct:
    @pytest.mark.parametrize(
        "A,B,normalize,expected",
        [
            (np.eye(2), np.eye(2), False, 2.0),
            (np.eye(2), np.zeros((2, 2)), False, 0.0),
            (
                np.array([[1.0, 2.3], [-1.3, 2.4]]),
                np.array([[0.7, -7.3], [-1.0, -2.9]]),
                False,
                -21.75,
            ),
            (np.eye(2), np.eye(2), True, 1.0),
            (
                np.array([[1.0, 2.3], [-1.3, 2.4]]),
                np.array([[0.7, -7.3], [-1.0, -2.9]]),
                True,
                -0.7381450594,
            ),
        ],
    )
    def test_frobenius_inner_product(self, A, B, normalize, expected):
        assert expected == pytest.approx(fn.frobenius_inner_product(A, B, normalize=normalize))
