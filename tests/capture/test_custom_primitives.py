# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Unit tests for PennyLane custom primitives.
"""

# pylint: disable=wrong-import-position
import pytest

jax = pytest.importorskip("jax")

import pennylane as qp
from pennylane.capture.custom_primitives import PrimitiveType, QpPrimitive

pytestmark = pytest.mark.jax


def test_qp_primitive_prim_type_default():
    """Test that the default prim_type of a QpPrimitive is set correctly."""
    prim = QpPrimitive("primitive")
    assert prim._prim_type == PrimitiveType("default")  # pylint: disable=protected-access
    assert prim.prim_type == "default"


@pytest.mark.parametrize("cast_in_enum", [True, False])
@pytest.mark.parametrize("prim_type", ["operator", "measurement", "transform", "higher_order"])
def test_qp_primitive_prim_type_setter(prim_type, cast_in_enum):
    """Test that the QpPrimitive.prim_type setter works correctly"""
    prim = QpPrimitive("primitive")
    prim.prim_type = PrimitiveType(prim_type) if cast_in_enum else prim_type
    assert prim._prim_type == PrimitiveType(prim_type)  # pylint: disable=protected-access
    assert prim.prim_type == prim_type


def test_qp_primitive_prim_type_setter_invalid():
    """Test that setting an invalid prim_type raises an error"""
    prim = QpPrimitive("primitive")
    with pytest.raises(ValueError, match="not a valid PrimitiveType"):
        prim.prim_type = "blah"


@pytest.mark.capture
def test_compile_time_constant_eval():
    """Test that operators and measurements are correctly added to the jaxpr when compile time
    constant evaluation is enabled."""

    def f():
        qp.RX(0.5, 0)
        return qp.expval(qp.Z(0))

    with jax._src.config.eager_constant_folding(True):
        jaxpr = jax.make_jaxpr(f)()

    assert jaxpr.eqns[0].primitive == qp.RX._primitive
    assert jaxpr.eqns[-1].primitive == qp.measurements.ExpectationMP._obs_primitive
