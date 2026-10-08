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
"""
This module contains utilities for testing code built with PennyLane, such as custom
operators, decomposition rules, and programs captured with :mod:`~pennylane.capture`.

The equality assertion :func:`~pennylane.assert_equal` is also available as
``qp.testing.assert_equal``.

.. seealso::

    :mod:`pennylane.devices.tests`, the integration test suite that can be run against
    any PennyLane device with the ``pl-device-test`` command.

Program capture
^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.testing

.. autosummary::
    :toctree: api

    ~plxpr_to_tape
    ~assert_eqn_matches_op
    ~extract_all_primitives
    ~single_operator_eqn

"""

from pennylane.ops.functions.equal import assert_equal

from .capture import (
    assert_eqn_matches_op,
    extract_all_primitives,
    plxpr_to_tape,
    single_operator_eqn,
)
