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
"""Convenient utility functions for testing optimization transforms."""

def _compare_operation_lists(ops_obtained, names_expected, wires_expected):
    """Compare two lists of operations."""
    assert len(ops_obtained) == len(names_expected)
    assert all([op.name == exp_name for (op, exp_name) in zip(ops_obtained, names_expected)])
    assert all([op.wires == exp_wires for (op, exp_wires) in zip(ops_obtained, wires_expected)])
