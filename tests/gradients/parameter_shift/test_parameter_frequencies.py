# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the gradients.parameter_frequencies single dispatch function and handlers."""

import numpy as np
import pytest

from pennylane import gradients, math
from pennylane.exceptions import ParameterFrequenciesUndefinedError
from pennylane.gradients import parameter_frequencies
from pennylane.operation2 import Operator2
from pennylane.ops import Exp, Hermitian, PauliZ
from pennylane.ops.functions import eigvals, generator
from pennylane.wires import Wires, WiresLike


class TestParameterFrequencies:
    """Tests for ``parameter_frequencies``."""

    def test_parameter_frequencies_raises_error_too_many_dynamic_args(self):
        """Test that parameter_frequencies raises an error if there are too many dynamic arguments."""

        class MultiArgOpWithGen(Operator2):
            num_params = 2
            num_wires = 1
            dynamic_argnames = ("phi", "theta")
            wire_argnames = ("wires",)

            def __init__(self, phi: float, theta: float, wires: WiresLike):
                super().__init__(phi, theta, wires=wires)

            def generator(self):
                return Hermitian(np.zeros((2, 2)), wires=self.wires)

        op = MultiArgOpWithGen(0.1, 0.2, 0)

        with pytest.raises(ParameterFrequenciesUndefinedError):
            _ = parameter_frequencies(op)

    def test_parameter_frequencies_raises_error_no_generator(self):
        """Test that parameter_frequencies raises an error if the op.generator() is undefined."""

        class SingleArgOpNoGen(Operator2):
            num_params = 1
            num_wires = 1
            dynamic_argnames = ("phi",)
            wire_argnames = ("wires",)

            def __init__(self, phi: float, wires: WiresLike):
                super().__init__(phi, wires=wires)

        op = SingleArgOpNoGen(0.1, 0)
        with pytest.raises(ParameterFrequenciesUndefinedError):
            _ = parameter_frequencies(op)

    @pytest.mark.parametrize(
        "freqs", [[(0.5, 1.0), (0.5, 1.0)], [(0.3, 4.0), (0.1, 2.0)], [(0.5, 1.0), (0.8, 0.2)]]
    )
    def test_param_freqs_no_generator(self, freqs):
        """Test that parameter_frequencies are accessible when provided explicitly, even if the op.generator() is undefined."""

        class MultiArgOpNoGenParamFreqs(Operator2):
            num_params = 2
            num_wires = 1
            dynamic_argnames = ("phi", "theta")
            wire_argnames = ("wires",)

            def __init__(self, phi: float, theta: float, wires: WiresLike):
                super().__init__(phi, theta, wires=wires)

        @parameter_frequencies.register
        def multi_arg_op_no_gen_param_freqs(op: MultiArgOpNoGenParamFreqs):
            return freqs

        op = MultiArgOpNoGenParamFreqs(0.4, 0.3, wires=[0, 1])
        assert parameter_frequencies(op) == freqs

    @pytest.mark.parametrize(
        "matrix",
        [
            np.zeros((2, 2)),
            np.array([[1, 0], [0, 1]]),
            np.array([[0, 1], [1, 0]]),
        ],
    )
    def test_param_freqs_with_generator(self, matrix):
        """Test that parameter_frequencies relate to the eigenvalues of the generator if the op.generator() is defined."""

        class OpWithGen(Operator2):
            num_params = 1
            num_wires = 1
            dynamic_argnames = ("phi",)
            wire_argnames = ("wires",)

            num_params = 1
            num_wires = 1

            def __init__(self, phi: float, wires: WiresLike):
                super().__init__(phi, wires=wires)

            def generator(self):
                return Hermitian(matrix, wires=self.wires)

        op = OpWithGen(0.1, 0)

        gen = generator(op, format="observable")
        gen_eigvals = eigvals(gen)
        freqs_from_eigvals = gradients.eigvals_to_frequencies(tuple(gen_eigvals))

        assert math.allclose(parameter_frequencies(op), freqs_from_eigvals)

    def test_parameter_frequencies_given_an_op(self):
        """Test that parameter_frequencies retrieves the parameter frequencies defined on an Operator."""
        op = Exp(PauliZ(1), 1j)
        assert parameter_frequencies(op) == [(2,)]

    def test_parameter_frequencies_raises(self):
        """Test that parameter_frequencies raises an error if given the wrong type of object."""
        wires = Wires([0, 1, 2])
        with pytest.raises(ParameterFrequenciesUndefinedError):
            _ = parameter_frequencies(wires)
