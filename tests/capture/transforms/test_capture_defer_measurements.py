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
"""Unit tests for the ``DeferMeasurementsInterpreter`` class"""

# pylint:disable=wrong-import-position, protected-access
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import (
    adjoint_transform_prim,
    cond_prim,
    ctrl_transform_prim,
    for_loop_prim,
    grad_prim,
    jacobian_prim,
    measure_prim,
    qnode_prim,
    while_loop_prim,
)
from pennylane.tape.plxpr_conversion import CollectOpsandMeas
from pennylane.transforms.defer_measurements import (
    DeferMeasurementsInterpreter,
    defer_measurements_plxpr_to_plxpr,
)

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


class TestDeferMeasurementsInterpreter:
    """Unit tests for DeferMeasurementsInterpreter."""

    @pytest.mark.parametrize("aux_wires", [1, (), [1, 2, 3], qml.wires.Wires([1, 2, 3])])
    def test_init(self, aux_wires):
        """Test that the interpreter is initialized correctly."""
        interpreter = DeferMeasurementsInterpreter(aux_wires)
        assert interpreter._aux_wires == qml.wires.Wires(aux_wires)

    def test_resolve_mcm_values(self):
        """Test that the resolve_mcm_values method correctly processes MeasurementValues."""

    @pytest.mark.parametrize("reset", [True, False])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    def test_single_mcm(self, reset, postselect):
        """Test that a function with a single MCM is transformed correctly."""

    def test_multiple_mcms(self):
        """Test that applying multiple MCMs is transformed correctly."""

    def test_too_many_mcms(self):
        """Test that an error is raised if more MCMs are present than the number of aux_wires."""

    def test_simple_cond(self):
        """Test that a qml.cond using a single MCM predicate is transformed correctly."""

    def test_non_trivial_cond_predicate(self):
        """Test that a qml.cond using processed MCMs as a predicate is transformed correctly."""

    def test_cond_elif_false_fn(self):
        """Test that a qml.cond with elif and false branches is transformed correctly."""

    def test_cond_non_mcm(self):
        """Test that a qml.cond that does not use MCM predicates is transformed correctly."""


class TestDeferMeasurementsHigherOrderPrimitives:
    """Unit tests for transforming higher-order primitives with DeferMeasurementsInterpreter."""

    def test_for_loop(self):
        """Test that a for_loop primitive is transformed correctly."""

    def test_while_loop(self):
        """Test that a while_loop primitive is transformed correctly."""

    def test_adjoint(self):
        """Test that the adjoint_transform primitive is transformed correctly."""

    def test_control(self):
        """Test that the ctrl_transform primitive is transformed correctly."""

    def test_qnode(self):
        """Test that a qnode primitive is transformed correctly."""

    @pytest.mark.parametrize("diff_fn", [qml.grad, qml.jacobian])
    def test_grad_jac(self, diff_fn):
        """Test that differentiation primitives are transformed correctly."""


def test_defer_measurements_plxpr_to_plxpr():
    """Test that transforming plxpr works."""
