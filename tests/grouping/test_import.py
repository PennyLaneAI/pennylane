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
Test import of the now deprecated grouping module.
"""
import pytest
import warnings


def test_pl_imports_with_no_warning():
    """Assert that importing PennyLane does not raise deprecation warning"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import pennylane as qml  # pylint:import-outside-toplevel

        _ = qml.PauliX(wires=0)

        assert len(w) == 0  # no warnings raised


def test_grouping_imports_with_warning():
    """Assert that accessing grouping does raise deprecation warning"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import pennylane as qml  # pylint:import-outside-toplevel
        from pennylane import grouping  # pylint:import-outside-toplevel

        op = qml.PauliX(wires=0)
        is_pauli = qml.grouping.is_pauli_word(op)  # some function from grouping

        assert is_pauli
        assert len(w) == 1  # warnings raised
        assert issubclass(w[-1].category, DeprecationWarning)
        assert (
            str(w[-1].message)
            == "The qml.grouping module is deprecated, please use qml.pauli instead."
        )
