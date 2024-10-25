# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Module test files for validation of qml.labs import
"""
# pylint: disable=import-outside-toplevel
# pylint: disable=unused-import
import pytest


def test_module_access():
    "labs module should not be visible without explicit import"
    import pennylane as qml

    with pytest.raises(Exception):
        print(qml.labs)


def test_module_import(import_labs):
    "Validate that explicitly importing the module makes it available under the `qml` alias."
    import pennylane as qml

    print(qml.labs)
