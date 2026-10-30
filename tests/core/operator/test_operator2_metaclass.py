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
"""Tests for Operator2's metaclass."""

import numpy as np
import pytest
from operator2_utils import CompilableOp, DynOp, StaticOp

import pennylane as qp
from pennylane.core.operator import Operator2, get_op_class_by_name
from pennylane.core.operator.meta import OperatorRegistry
from pennylane.core.operator.operator2 import operator_p
from pennylane.testing import find_eqns
from pennylane.typing import AbstractArray, Bool, Complex, Float, Int, Wire
from pennylane.wires import Wires


class TestOperatorRegistry:
    """Tests that operator classes are registered when they are defined."""

    @pytest.mark.parametrize("op_class", (qp.Hadamard, qp.PauliX, qp.GlobalPhase))
    def test_builtin_subclasses(self, op_class):
        """Built-in operators are registered under their class names."""
        assert issubclass(op_class, Operator2)
        assert OperatorRegistry()[op_class.__name__] is op_class
        assert get_op_class_by_name(op_class.__name__) is op_class

    def test_user_defined_subclass(self):
        """A custom operator is registered without constructing an instance."""

        class RegistryCustomOp(Operator2):
            """A user-defined operator with a wire argument."""

            def __init__(self, wires):
                raise AssertionError("Registration must not instantiate the operator")

        assert OperatorRegistry()["RegistryCustomOp"] is RegistryCustomOp
        assert get_op_class_by_name("RegistryCustomOp") is RegistryCustomOp

    def test_indirect_subclass(self):
        """An inherited custom operator has its own registry entry."""

        class RegistryParentOp(Operator2):
            """A user-defined parent operator."""

            def __init__(self, wires):
                raise AssertionError("Registration must not instantiate the operator")

        class RegistryChildOp(RegistryParentOp):
            """An operator inheriting from a user-defined operator."""

        registry = OperatorRegistry()
        assert registry["RegistryParentOp"] is RegistryParentOp
        assert registry["RegistryChildOp"] is RegistryChildOp
        assert get_op_class_by_name("RegistryChildOp") is RegistryChildOp

    def test_duplicate_name(self):
        """Duplicate names warn and preserve the first registered class."""

        class RegistryDuplicateOp(Operator2):
            """The first operator registered under this name."""

            def __init__(self, wires):
                raise AssertionError("Registration must not instantiate the operator")

        original_class = RegistryDuplicateOp
        with pytest.warns(
            UserWarning,
            match=r"More than one operator class by the same name \(RegistryDuplicateOp\)",
        ):

            class RegistryDuplicateOp(Operator2):
                """A different operator with the same name."""

                def __init__(self, wires):
                    raise AssertionError("Registration must not instantiate the operator")

        assert RegistryDuplicateOp is not original_class
        assert OperatorRegistry()["RegistryDuplicateOp"] is original_class
        assert get_op_class_by_name("RegistryDuplicateOp") is original_class

    @pytest.mark.parametrize(
        "lookup", (get_op_class_by_name, lambda name: OperatorRegistry()[name])
    )
    def test_unknown_name(self, lookup):
        """Both lookup interfaces raise KeyError for an unregistered name."""

        with pytest.raises(KeyError) as exc_info:
            lookup("RegistryUnknownOp")

        assert exc_info.value.args == ("RegistryUnknownOp",)


class DynCanonOp(Operator2):  # pylint: disable=too-few-public-methods
    """Operator with a dynamic parameter and wires that performs canonicalization."""

    dynamic_argnames = ("phi",)

    def __init__(self, phi, wires):
        new_phi = phi if isinstance(phi, AbstractArray) else 2 * phi
        super().__init__(new_phi, wires)


def test_child_constructor_runs_when_concrete():
    """Tests a concrete input will trigger the child's constructor."""

    op = DynCanonOp(phi=2.0, wires=0)
    # __init__ is hit so phi is doubled
    assert op.phi == 4.0
    assert op.wires == Wires(0)


@pytest.mark.capture
@pytest.mark.parametrize("op", (StaticOp, CompilableOp))
def test_static_compilable_arg_validation(op):
    """Tests that an error is raised if dynamic arguments are fed to static / compilable arguments."""

    import jax

    def f(a):
        op(a, 0)

    error_msg = (
        rf"Argument '.*' of operator '{op.__name__}' must be a concrete, compile-time constant\."
    )
    with pytest.raises(ValueError, match=error_msg):
        _ = jax.make_jaxpr(f)(0.5)


class TestArgSpecValidationAbstractInputs:
    """Tests arg_spec validation when abstract inputs are used to construct operators."""

    def test_weak_dtype_is_preserved(self):
        """Tests that canonicalization preserves strength of dtype."""

        class MixedArgOp(Operator2):  # pylint: disable=too-few-public-methods
            """Operator with static, dynamic and hybrid argnames."""

            dynamic_argnames = ("dynamic_arg",)

            arg_specs = {"dynamic_arg": float, "wires": Wire[3]}

            def __init__(self, dynamic_arg, wires):
                super().__init__(dynamic_arg, wires=wires)

        # Abstract inputs get canonicalized
        # NOTE: Can safely upcast an int to a float.
        op = MixedArgOp(AbstractArray((), np.int32), Wire[3])
        assert op.dynamic_arg == Float
        # pylint: disable=protected-access
        assert op.dynamic_arg._weak_type

    def test_arg_spec_with_unknown_shape_canonicalizes_only_dtype(self):
        """Tests that only the dtype is promoted."""

        class MixedArgOp(Operator2):  # pylint: disable=too-few-public-methods
            """Operator with static, dynamic and hybrid argnames."""

            dynamic_argnames = ("dynamic_arg",)

            arg_specs = {"dynamic_arg": Float[...], "wires": Wire[3]}

            def __init__(self, dynamic_arg, wires):
                super().__init__(dynamic_arg, wires=wires)

        # Abstract inputs get canonicalized
        # NOTE: Can safely upcast an int to a float.
        op = MixedArgOp(Int[2, 3], Wire[3])
        assert op.dynamic_arg == Float[2, 3]

    def test_arg_spec_canonicalizes_abstract_inputs(self):
        """Tests that abstract inputs are canonicalized when possible."""

        class MixedArgOp(Operator2):  # pylint: disable=too-few-public-methods
            """Operator with static, dynamic and hybrid argnames."""

            dynamic_argnames = ("dynamic_arg",)

            arg_specs = {"dynamic_arg": Float[2, 3], "wires": Wire[3]}

            def __init__(self, dynamic_arg, wires):
                super().__init__(dynamic_arg, wires=wires)

        # Abstract inputs get canonicalized
        # NOTE: Can safely upcast an int to a float.
        op = MixedArgOp(Int[2, 3], Wire[3])
        assert op.dynamic_arg == Float[2, 3]

        # Abstract inputs that are not compatible raise an error
        # NOTE: Cannot downcast complex to float
        expected_msg = r"Parameter \'dynamic_arg\' does not match the operator\'s expected \'arg_specs\' dtype. Expected float64 but received complex128."
        with pytest.raises(ValueError, match=expected_msg):
            _ = MixedArgOp(Complex[2, 3], Wire[3])

        # Concrete inputs go through as normal
        op = MixedArgOp(np.ones((2, 3), int), [0, 1, 2])
        assert np.allclose(op.dynamic_arg, np.ones((2, 3), int))

    def test_valid_arg_spec_with_unknown_shape(self):
        """Tests that using ... in your arg_specs works as expected."""

        class MixedArgOp(Operator2):  # pylint: disable=too-few-public-methods
            """Operator with static, dynamic and hybrid argnames."""

            dynamic_argnames = ("dynamic_arg",)

            arg_specs = {"dynamic_arg": Float[...], "wires": Wire[3]}

            def __init__(self, dynamic_arg, wires):
                super().__init__(dynamic_arg, wires=wires)

        # Arg spec is defined as unknown shape, any of these are valid.
        op = MixedArgOp(Float[3], Wire[3])
        assert op.dynamic_arg == Float[3]
        op = MixedArgOp(Float[2, 3], Wire[3])
        assert op.dynamic_arg == Float[2, 3]
        op = MixedArgOp(Float[...], Wire[3])
        assert op.dynamic_arg == Float[...]

    def test_valid_arg_spec_with_fixed_shape(self):
        """Tests a simple valid arg spec."""

        class MixedArgOp(Operator2):  # pylint: disable=too-few-public-methods
            """Operator with static, dynamic and hybrid argnames."""

            dynamic_argnames = ("dynamic_arg",)

            arg_specs = {"dynamic_arg": Float[3], "wires": Wire[3]}

            def __init__(self, dynamic_arg, wires):
                super().__init__(dynamic_arg, wires=wires)

        op = MixedArgOp(Float[3], Wire[3])
        assert op.dynamic_arg == Float[3]

    @pytest.mark.parametrize("bad_dynamic_arg", (Float, Float[4], Float[-1], Float[...]))
    def test_invalid_dynamic_arg_spec(self, bad_dynamic_arg):
        """Tests arg_spec validation against operators constructed with abstract inputs."""

        class MixedArgOp(Operator2):  # pylint: disable=too-few-public-methods
            """Operator with static, dynamic and hybrid argnames."""

            dynamic_argnames = ("dynamic_arg",)

            arg_specs = {"dynamic_arg": Float[3], "wires": Wire[3]}

            def __init__(self, dynamic_arg, wires):
                super().__init__(dynamic_arg, wires=wires)

        with pytest.raises(
            ValueError, match=r"expected 'arg_specs' shape\. Expected \(3,\) but received .*"
        ):
            _ = MixedArgOp(bad_dynamic_arg, Wire[3])


@pytest.mark.capture
def test_bind_primitive():
    """Tests that a construction under capture will bind the primitive."""
    import jax

    cjaxpr = jax.make_jaxpr(lambda x: DynOp(x, wires=0))(2.0)
    # Make sure the operator primitive is in the JAXPR
    assert len(find_eqns(cjaxpr, operator_p)) == 1


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("value", [Int, Float[2], Bool[3], Wire[2], (1, Int), {"data": [Float[2]]}])
def test_abstract_compilable_argument_rejected(value, monkeypatch):
    """Reject abstract leaves before entering operator-specific constructors."""

    def unexpected_constructor(*_args, **_kwargs):
        pytest.fail("The operator constructor must not run for abstract compilable inputs.")

    monkeypatch.setattr(CompilableOp, "__init__", unexpected_constructor)
    with pytest.raises(ValueError, match="Argument 'n' of operator 'CompilableOp'.*abstract input"):
        CompilableOp(value, wires=0)


@pytest.mark.usefixtures("enable_and_disable_capture")
def test_flipsign_abstract_state_rejected():
    """FlipSign reports the invalid argument instead of a later MLIR conversion failure."""
    with pytest.raises(ValueError, match="Argument 'state' of operator 'FlipSign'.*abstract input"):
        qp.FlipSign(Bool[3], wires=Wire[3], work_wires=Wire[1])


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("value", [3, (0, 1, 0), {"data": [1, 2]}])
def test_concrete_compilable_arguments_with_abstract_wires(value):
    """Concrete compilable arguments still allow abstract wire arguments."""
    op = CompilableOp(value, wires=Wire[2])
    assert op.n == value
    assert op.wires == Wire[2]
