# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for the conversion between PennyLane circuits and pyzx `Circuit`."""

import re

from pyzx import Circuit


def _remove_measurement_patterns(text):
    r"""
    Removes patterns of the form "measure q[n] -> c[m];" from the input text,
    where n and m are arbitrary integers.

    Args:
        text (str): Input text containing measurement patterns

    Returns:
        str: Text with measurement patterns removed
    """
    pattern = r"measure\s+q\[\d+\]\s*->\s*c\[\d+\]\s*;"

    # Remove all occurrences of the pattern
    result = re.sub(pattern, "", text)

    return result


def _tape2pyzx(tape):
    r"""
    A translation function going the qasm route

    Args:
        tape (qml.tape.QuantumScript): input tape

    Returns:
        pyzx.Graph: graph instance of the pyzx circuit

    """
    g = Circuit.from_qasm(_remove_measurement_patterns(tape.to_openqasm()))

    return g
