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
"""Tests for the gradients.general_shift_rules module."""
import pytest

import numpy as np
import pennylane as qml
from pennylane import numpy as np
from pennylane.gradients.general_shift_rules import get_shift_rule


class TestGetShiftRule:
    """Tests of input validation and output correctness of function `get_shift_rule`."""

    def test_invalid_frequency_spectrum(self):
        """Tests ValueError is raised if input frequency spectrum is non positive or non unique."""

        non_positive_frequency_spectrum = (-1, 1)
        non_unique_frequency_spectrum = (1, 2, 2, 3)

        assert pytest.raises(ValueError, get_shift_rule, non_positive_frequency_spectrum)
        assert pytest.raises(ValueError, get_shift_rule, non_unique_frequency_spectrum)

    def test_invalid_shifts(self):
        """Tests ValueError is raised if specified shifts is not of the same length as
        `frequencies`, or if shifts are non-unique."""

        frequencies = (1, 4, 5, 6)
        invalid_shifts_num = (np.pi / 8, 3 * np.pi / 8, 5 * np.pi / 8)
        non_unique_shifts = (np.pi / 8, 3 * np.pi / 8, 5 * np.pi / 8, np.pi / 8)

        assert pytest.raises(ValueError, get_shift_rule, frequencies, invalid_shifts_num)
        assert pytest.raises(ValueError, get_shift_rule, frequencies, non_unique_shifts)

    def test_two_term_rule_default_shifts(self):
        """Tests the correct two term equidistant rule is generated using default shift pi/2.
        Frequency 1 corresponds to any generator of the form: 1/2*P, where P is a Pauli word."""

        frequencies = (1,)

        n_terms = 2
        correct_terms = [[0.5, 1.0, np.pi / 2], [-0.5, 1.0, -np.pi / 2]]

        generated_terms = get_shift_rule(frequencies)[0]

        assert all([all(np.isclose(generated_terms[i], correct_terms[i])) for i in range(n_terms)])

    def test_four_term_rule_default_shifts(self):
        """Tests the correct two term equidistant rule is generated using the default shifts [pi/4, 3*pi/4].
        The frequency [1,2] corresponds to a generator e.g. of the form 1/2*X0Y1 + 1/2*Y0X1."""

        frequencies = (1, 2)

        n_terms = 4
        correct_terms = [
            [0.8535533905932737, 1.0, np.pi / 4],
            [-0.14644660940672624, 1.0, 3 * np.pi / 4],
            [-0.8535533905932737, 1.0, -np.pi / 4],
            [0.14644660940672624, 1.0, -3 * np.pi / 4],
        ]

        generated_terms = get_shift_rule(frequencies)[0]

        assert all([all(np.isclose(generated_terms[i], correct_terms[i])) for i in range(n_terms)])

    def test_eight_term_rule_non_equidistant_default_shifts(self):
        """Tests the correct non-equidistant eight term shift rule is generated given the
        frequencies using the default shifts. The frequency [1,4,5,6] corresponds to e.g.
        a 2-qubit generator of the form: 1/2*X0Y1 + 5/2*Y0X1."""

        frequencies = (1, 4, 5, 6)

        n_terms = 8
        correct_terms = [
            [2.8111804455102014, 1.0, np.pi / 8],
            [0.31327576445128014, 1.0, 3 * np.pi / 8],
            [-0.8080445791083615, 1.0, 5 * np.pi / 8],
            [-0.3101398980494395, 1.0, 7 * np.pi / 8],
            [-2.8111804455102014, 1.0, -np.pi / 8],
            [-0.31327576445128014, 1.0, -3 * np.pi / 8],
            [0.8080445791083615, 1.0, -5 * np.pi / 8],
            [0.3101398980494395, 1.0, -7 * np.pi / 8],
        ]

        generated_terms = get_shift_rule(frequencies)[0]

        assert all([all(np.isclose(generated_terms[i], correct_terms[i])) for i in range(n_terms)])

    def test_eight_term_rule_non_equidistant_custom_shifts(self):
        """Tests the correct non-equidistant eight term shift rule is generated given the
        frequencies using non-default shifts. The frequency [1,4,5,6] corresponds to e.g.
        a 2-qubit generator of the form: 1/2*X0Y1 + 5/2*Y0X1."""

        frequencies = (1, 4, 5, 6)
        custom_shifts = (1 / 13 * np.pi, 3 / 7 * np.pi, 2 / 3 * np.pi, 3 / 4 * np.pi)

        n_terms = 8
        correct_terms = [
            [2.709571194594805, 1.0, np.pi / 13],
            [-0.12914139932030527, 1.0, 3 * np.pi / 7],
            [-0.3820906256032637, 1.0, 2 * np.pi / 3],
            [0.436088184940856, 1.0, 3 * np.pi / 4],
            [-2.709571194594805, 1.0, -np.pi / 13],
            [0.12914139932030527, 1.0, -3 * np.pi / 7],
            [0.3820906256032637, 1.0, -2 * np.pi / 3],
            [-0.436088184940856, 1.0, -3 * np.pi / 4],
        ]

        generated_terms = get_shift_rule(frequencies, custom_shifts)[0]

        assert all([all(np.isclose(generated_terms[i], correct_terms[i])) for i in range(n_terms)])

    def test_non_integer_frequency_default_shifts(self):
        """Tests the correct four term shift rule is generated given non-integer frequencies."""

        frequencies = (1 / 3, 2 / 3)
        n_terms = 4

        correct_terms = [
            [0.2845177968644246, 1, 3 * np.pi / 4],
            [-0.048815536468908745, 1, 9 * np.pi / 4],
            [-0.2845177968644246, 1, -3 * np.pi / 4],
            [0.048815536468908745, 1, -9 * np.pi / 4],
        ]

        generated_terms = get_shift_rule(frequencies)[0]

        assert all([all(np.isclose(generated_terms[i], correct_terms[i])) for i in range(n_terms)])

    def test_non_integer_frequency_custom_shifts(self):
        """Tests the correct four term shift rule is generated given non-integer frequencies using
        explicitly defined shifts."""

        frequencies = (1 / 3, 2 / 3, 4 / 3)
        custom_shifts = (
            np.pi / 4,
            np.pi / 3,
            2 * np.pi / 3,
        )
        n_terms = 6

        correct_terms = [
            [1.7548361197453346, 1, 0.7853981633974483],
            [-0.8720240894718643, 1, 1.0471975511965976],
            [0.016695190986336428, 1, 2.0943951023931953],
            [-1.7548361197453346, 1, -0.7853981633974483],
            [0.8720240894718643, 1, -1.0471975511965976],
            [-0.016695190986336428, 1, -2.0943951023931953],
        ]

        generated_terms = get_shift_rule(frequencies, custom_shifts)[0]

        assert all([all(np.isclose(generated_terms[i], correct_terms[i])) for i in range(n_terms)])

    def test_near_singular_warning(self):
        """Tests a warning is raised if the determinant of the matrix to be inverted is near zero
        for obtaining parameter shift rules for the non-equidistant frequency case."""

        frequencies = (1, 2, 3, 4, 5, 67)

        with pytest.warns(None) as warnings:
            get_shift_rule(frequencies)

        raised_warning = False
        for warning in warnings:
            if "Solving linear problem with near zero determinant" in str(warning):
                raised_warning = True

        assert raised_warning
