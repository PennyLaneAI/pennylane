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
"""
Tests for the fourier qnode transforms.
"""
import pytest
from pennylane.beta.transforms.fourier import spectrum, _join_spectra, _get_spectrum, _simplify_tape


class UnitTests:

    @pytest.mark.parametrize("spectrum1, spectrum2, expected", [([-1, 0, 1], [-1, 0, 1], [-2, -1, 0, 1, 2]),
                                                                ([-3, 0, 3], [-5, 0, 5],
                                                                 [-8, -5, -3, -2, 0, 2, 3, 5, 8]),
                                                                ([-2, -1, 0, 1, 2], [-1, 0, 1],
                                                                 [-3, -2, -1, 0, 1, 2, 3]),
                                                                ([-0.5, 0, 0.5], [-1, 0, 1],
                                                                 [-1.5, -1, -0.5, 0, 0.5, 1., 1.5])
                                                                ])
    def test_join_spectra(self, spectrum1, spectrum2, expected):
        joined = _join_spectra(spectrum1, spectrum2)
        assert joined == expected
