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
"""Basic unit tests for ``Operator2``."""

# pylint: disable=redefined-outer-name,protected-access,too-few-public-methods
# TODO: [sc-120817] Add interface tests
# TODO: [sc-120982] Add integration tests

import copy

import numpy as np
import pytest
from operator2_utils import CompilableOp, DynOp, FullOp, HybridWireOp
from scipy.sparse import csr_matrix

import pennylane as qp
from pennylane.core.operator import Operator2, StatePrepBase2, abstractify
from pennylane.core.queuing import AnnotatedQueue, apply
from pennylane.exceptions import (
    AdjointUndefinedError,
    DecompositionUndefinedError,
    DiagGatesUndefinedError,
    EigvalsUndefinedError,
    GeneratorUndefinedError,
    MatrixUndefinedError,
    PowUndefinedError,
    SparseMatrixUndefinedError,
    TermsUndefinedError,
)
from pennylane.operation import _UNSET_BATCH_SIZE
from pennylane.pauli import PauliSentence, PauliWord
from pennylane.pytrees.pytrees import flatten_registrations, unflatten_registrations
from pennylane.typing import AbstractArray, AbstractWires, Float, Wire
from pennylane.wires import Wires


# pylint: disable=too-many-public-methods
class TestInitSubclass:
    """Tests for the validation performed in ``Operator2.__init_subclass__``."""

    def test_str_argnames_converted_to_tuple(self):
        """Test that string argnames are converted into one-tuples."""

        class Op(Operator2):
            dynamic_argnames = "phi"
            static_argnames = "pw"
            hybrid_argnames = "wires"

            def __init__(self, phi, pw, wires):
                super().__init__(phi, pw, wires=wires)

        assert Op.dynamic_argnames == ("phi",)
        assert Op.static_argnames == ("pw",)
        assert Op.hybrid_argnames == ("wires",)
        assert Op.wire_argnames == ("wires",)
        assert Op.compilable_argnames == ()

    @pytest.mark.parametrize(
        "other_argnames",
        [
            {"hybrid_argnames": ("y", "z")},
            {"static_argnames": ("y", "z")},
            {"hybrid_argnames": ("y",), "static_argnames": ("z",)},
        ],
    )
    def test_static_hybrid_and_compilable_both_set_error(self, other_argnames):
        """Test that declaring both ``static_argnames``/``hybrid_argnames`` and
        ``compilable_argnames`` is not allowed."""

        def __init__(self, x, y, z, wires):
            Operator2.__init__(self, x, y, z, wires=wires)

        attrs = {"__init__": __init__, "compilable_argnames": ("x",), **other_argnames}

        with pytest.raises(
            TypeError,
            match="contain 'static_argnames' and 'hybrid_argnames', or 'compilable_argnames'",
        ):
            # This creates a class while allowing us to parametrize the attributes
            # that we want to test.
            type("Op", (Operator2,), attrs)

    @pytest.mark.parametrize(
        "first, second",
        [
            ("dynamic_argnames", "wire_argnames"),
            ("dynamic_argnames", "static_argnames"),
            ("dynamic_argnames", "compilable_argnames"),
            ("static_argnames", "wire_argnames"),
            ("compilable_argnames", "wire_argnames"),
        ],
    )
    def test_pairwise_overlap_error(self, first, second):
        """Test that dynamic, wire, static, and compilable argnames must be disjoint."""

        def __init__(self, x):
            Operator2.__init__(self, x)

        attrs = {first: ("x",), second: ("x",), "__init__": __init__}
        # If neither overlapping group is ``wire_argnames``, blank out the
        # default ``("wires",)`` since this signature has no ``wires`` param.
        if "wire_argnames" not in (first, second):
            attrs["wire_argnames"] = ()

        with pytest.raises(TypeError, match="must not overlap"):
            # This creates a class while allowing us to parametrize the attributes
            # that we want to test.
            _ = type("Op", (Operator2,), attrs)

    def test_hybrid_may_overlap_with_wires(self):
        """Test that ``hybrid_argnames`` is allowed to overlap with ``wire_argnames``."""

        class Op(Operator2):
            wire_argnames = ("wires",)
            hybrid_argnames = ("wires",)

            def __init__(self, wires):
                super().__init__(wires=wires)

        assert Op.hybrid_argnames == ("wires",)

    @pytest.mark.parametrize("other_group", ["dynamic_argnames", "static_argnames"])
    def test_hybrid_overlap_with_non_wire_error(self, other_group):
        """Test that ``hybrid_argnames`` may not overlap with dynamic, static,
        or compilable argnames."""

        def __init__(self, x, wires):
            Operator2.__init__(self, x, wires=wires)

        attrs = {"hybrid_argnames": ("x",), other_group: ("x",), "__init__": __init__}

        with pytest.raises(TypeError, match="overlap with dynamic, static"):
            # This creates a class while allowing us to parametrize the attributes
            # that we want to test.
            type("Op", (Operator2,), attrs)

    def test_unclassified_signature_parameter_error(self):
        """Test that every parameter in the signature must appear in some ``**_argnames`` tuple."""

        with pytest.raises(TypeError, match="not classified in any argnames"):
            # pylint: disable=unused-variable
            class Op(Operator2):
                # ``phi`` is not in any argnames tuple
                def __init__(self, phi, wires):
                    super().__init__(phi, wires=wires)

    def test_signature_captured(self):
        """Test that ``cls._sig`` is set to the subclass's __init__ signature."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        assert list(Op._sig.parameters) == ["phi", "wires"]
        assert "self" not in Op._sig.parameters

    def test_class_registered_as_pytree(self):
        """Test that subclasses are automatically registered as pytrees."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        assert Op in flatten_registrations
        assert Op in unflatten_registrations

    def test_wire_sizes_defaults_to_none_per_wire_arg(self):
        """Test that when ``wire_sizes`` is not declared, it defaults to a tuple
        of ``None`` matching the length of ``wire_argnames``."""

        class Op(Operator2):
            wire_argnames = ("wires", "ctrl_wires")

            def __init__(self, wires, ctrl_wires):
                super().__init__(wires=wires, ctrl_wires=ctrl_wires)

        assert Op.wire_sizes == (None, None)

    def test_wire_sizes_scalar_converted_to_tuple(self):
        """Test that a non-sequence ``wire_sizes`` is converted to a 1-tuple."""

        class Op(Operator2):
            wire_sizes = 2

            def __init__(self, wires):
                super().__init__(wires=wires)

        assert Op.wire_sizes == (2,)

    def test_wire_sizes_length_mismatch_error(self):
        """Test that ``wire_sizes`` must have the same length as ``wire_argnames``."""

        with pytest.raises(
            TypeError, match="'wire_sizes' must have the same length as 'wire_argnames'"
        ):
            # pylint: disable=unused-variable
            class Op(Operator2):
                wire_argnames = ("wires", "ctrl_wires")
                wire_sizes = (1, 1, 1)

                def __init__(self, wires, ctrl_wires):
                    super().__init__(wires=wires, ctrl_wires=ctrl_wires)

    def test_hybrid_wire_size_must_be_none(self):
        """Test that a hybrid wire argument cannot declare a fixed wire size."""

        with pytest.raises(
            TypeError,
            match="Expected wire_size == None for 'pytree_wires' as it is a hybrid wire argument",
        ):
            # pylint: disable=unused-variable
            class Op(Operator2):
                wire_argnames = ("pytree_wires",)
                hybrid_argnames = ("pytree_wires",)
                wire_sizes = (2,)

                def __init__(self, pytree_wires):
                    super().__init__([Wires(w) for w in pytree_wires])

    @pytest.mark.parametrize("invalid_size", [0, -1, 1.5, "two"])
    def test_wire_sizes_invalid_value_error(self, invalid_size):
        """Test that ``wire_sizes`` entries must be positive integers or ``None``."""

        with pytest.raises(TypeError, match="'wire_sizes' must be a sequence of"):
            # pylint: disable=unused-variable
            class Op(Operator2):
                wire_sizes = (invalid_size,)

                def __init__(self, wires):
                    super().__init__(wires=wires)

    @pytest.mark.parametrize("attr", ["hybrid_argnames", "static_argnames", "compilable_argnames"])
    def test_arg_specs_incompatible_with_other_arg_groups(self, attr):
        """Test that ``arg_specs`` cannot name hybrid, static, or compilable args."""

        attrs = {
            "dynamic_argnames": ("phi",),
            "arg_specs": {
                "phi": AbstractArray((), float),
                "extra": AbstractArray((), float),
            },
            attr: ("extra",),
            "__init__": lambda self, phi, extra, wires: Operator2.__init__(
                self, phi, extra, wires=wires
            ),
        }

        with pytest.raises(
            TypeError,
            match=r"Op\.arg_specs can only contain dynamic and wire arguments",
        ):
            type("Op", (Operator2,), attrs)

    def test_wire_sizes_derived_from_arg_specs(self):
        """Test that ``wire_sizes`` is inferred from ``arg_specs`` when not declared."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            wire_argnames = ("wires", "ctrl_wires")
            arg_specs = {
                "phi": AbstractArray((), float),
                "wires": AbstractWires(2),
                "ctrl_wires": AbstractWires(1),
            }

            def __init__(self, phi, wires, ctrl_wires):
                super().__init__(phi, wires=wires, ctrl_wires=ctrl_wires)

        assert Op.wire_sizes == (2, 1)

    def test_arg_specs_wire_sizes_mismatch_error(self):
        """Test that ``arg_specs`` and ``wire_sizes`` must agree on wire counts."""

        with pytest.raises(
            TypeError,
            match="Number of wires specified for 'wires' does not match",
        ):
            # pylint: disable=unused-variable
            class Op(Operator2):
                dynamic_argnames = ("phi",)
                wire_sizes = (3,)
                arg_specs = {
                    "phi": AbstractArray((), float),
                    "wires": AbstractWires(2),
                }

                def __init__(self, phi, wires):
                    super().__init__(phi, wires=wires)

    def test_arg_specs_builtin_num_types_canonicalized(self):
        """Test that builtin Python number types are canonicalized to ``AbstractArrays``."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            arg_specs = {"phi": float}

            # pylint: disable=useless-parent-delegation
            def __init__(self, phi, wires):
                super().__init__(phi, wires)

        assert Op.arg_specs == {"phi": AbstractArray((), float)}

    def test_has_fixed_sig_false_with_argnames_without_arg_specs(self):
        """Test that ``has_fixed_sig`` is ``False`` when ``arg_specs`` is not declared and there
        are any arguments."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        assert Op.has_fixed_sig is False

    def test_has_fixed_sig_true_without_argnames_without_arg_specs(self):
        """Test that ``has_fixed_sig`` is ``True`` when ``arg_specs`` is not declared and there
        are no arguments."""

        class Op(Operator2):
            wire_argnames = ()

            def __init__(self):
                # pylint: disable=useless-parent-delegation
                super().__init__()

        assert Op.has_fixed_sig is True

    def test_has_fixed_sig_true_for_fully_specified_static_types(self):
        """Test that ``has_fixed_sig`` is ``True`` when only dynamic and wire args are fully typed."""

        class Op(Operator2):
            dynamic_argnames = ("phi", "theta")
            wire_argnames = ("wires", "ctrl_wires")
            arg_specs = {
                "phi": AbstractArray((), float),
                "theta": AbstractArray((2,), float),
                "wires": AbstractWires(2),
                "ctrl_wires": AbstractWires(1),
            }

            def __init__(self, phi, theta, wires, ctrl_wires):
                super().__init__(phi, theta, wires=wires, ctrl_wires=ctrl_wires)

        assert Op.has_fixed_sig is True

    def test_has_fixed_sig_false_for_partial_arg_specs(self):
        """Test that ``has_fixed_sig`` is ``False`` when ``arg_specs`` omits some arguments."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            arg_specs = {"phi": AbstractArray((), float)}

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        assert Op.has_fixed_sig is False

    @pytest.mark.parametrize("phi_spec", [AbstractArray(..., float), AbstractArray((-1,), float)])
    def test_has_fixed_sig_false_for_unknown_rank_or_axis(self, phi_spec):
        """Test that ``has_fixed_sig`` is ``False`` for dynamic shapes that are not fully fixed."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            arg_specs = {"phi": phi_spec, "wires": AbstractWires(1)}

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        assert Op.has_fixed_sig is False

    def test_has_fixed_sig_false_for_dynamic_wire_count(self):
        """Test that ``has_fixed_sig`` is ``False`` when a wire arg has unknown length."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            arg_specs = {
                "phi": AbstractArray((), float),
                "wires": AbstractWires(-1),
            }

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        assert Op.has_fixed_sig is False

    def test_has_fixed_sig_true_after_number_type_canonicalization(self):
        """Test that canonicalized builtin number types still yield a fixed signature."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            arg_specs = {"phi": float, "wires": AbstractWires(2)}

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        assert Op.has_fixed_sig is True
        assert Op.arg_specs["phi"].shape_fixed is True

    @pytest.mark.parametrize(
        "extra_argnames",
        [
            {"static_argnames": ("label",)},
            {"hybrid_argnames": ("ops",)},
            {"compilable_argnames": ("n",)},
        ],
    )
    def test_has_fixed_sig_false_with_non_dynamic_wire_args(self, extra_argnames):
        """Test that ``has_fixed_sig`` is ``False`` when hybrid, static, or compilable args exist."""

        attrs = {
            "dynamic_argnames": ("phi",),
            "arg_specs": {
                "phi": AbstractArray((), float),
                "wires": AbstractWires(2),
            },
            **extra_argnames,
        }

        if "static_argnames" in extra_argnames:
            attrs["__init__"] = lambda self, phi, label, wires: Operator2.__init__(
                self, phi, label, wires=wires
            )
        elif "hybrid_argnames" in extra_argnames:
            attrs["__init__"] = lambda self, phi, ops, wires: Operator2.__init__(
                self, phi, ops, wires=wires
            )
        else:
            attrs["__init__"] = lambda self, phi, n, wires: Operator2.__init__(
                self, phi, n, wires=wires
            )

        Op = type("Op", (Operator2,), attrs)
        assert Op.has_fixed_sig is False

    def test_has_fixed_sig_false_with_hybrid_wire_arg(self):
        """Test that a hybrid wire argument prevents a fixed signature."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            wire_argnames = ("wires", "pytree_wires")
            hybrid_argnames = ("pytree_wires",)
            arg_specs = {
                "phi": AbstractArray((), float),
                "wires": AbstractWires(2),
            }

            def __init__(self, phi, pytree_wires, wires):
                super().__init__(phi, [Wires(w) for w in pytree_wires], wires=wires)

        assert Op.has_fixed_sig is False


class TestOperatorInit:
    """Tests for ``Operator2.__init__``."""

    def test_arguments_bound(self):
        """Test that constructor positional/keyword arguments are bound into ``arguments``."""

        op = DynOp(0.5, wires=0)
        assert op.arguments == {"phi": 0.5, "wires": Wires([0])}

    def test_defaults_applied(self):
        """Test that missing kwargs receive their default values."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            static_argnames = ("method",)

            # pylint: disable=unused-argument
            def __init__(self, phi, wires, method="auto"):
                super().__init__(phi, Wires(wires))

        op = Op(0.5, wires=0)
        assert op.arguments["method"] == "auto"
        assert op.is_abstract is False

    def test_wires_collected_from_wire_argnames(self):
        """Test that the ``_wires`` attribute combines the contents of ``wire_argnames``."""

        op = DynOp(0.5, wires=[0, 2])
        assert op.wires == Wires([0, 2])

    def test_wires_collected_from_multiple_wire_args(self):
        """Test that multiple wire argnames are unioned, preserving order."""

        class TwoWiresArgsOp(Operator2):
            wire_argnames = ("wires", "ctrl_wires")

            def __init__(self, wires, ctrl_wires):
                super().__init__(Wires(wires), Wires(ctrl_wires))

        op = TwoWiresArgsOp(wires=[0, 1], ctrl_wires=2)
        assert op.wires == Wires([0, 1, 2])

    def test_work_wires_excluded_from_wires(self):
        """Test that the ``work_wires``/``work_wire`` argnames are not added to ``_wires``."""

        class WithWorkWires(Operator2):
            wire_argnames = ("wires", "work_wires", "work_wire")

            def __init__(self, wires, work_wires, work_wire):
                super().__init__(Wires(wires), Wires(work_wires), Wires(work_wire))

        op = WithWorkWires(wires=[0, 1], work_wires=[2, 3], work_wire=4)
        assert op.wires == Wires([0, 1])

    def test_hybrid_arg_operator_wires_collected(self):
        """Test that operators inside ``hybrid_argnames`` contribute their wires."""

        class Composite(Operator2):
            hybrid_argnames = ("ops", "pytree_wires")
            wire_argnames = ("wires", "pytree_wires")

            def __init__(self, ops, pytree_wires, wires):
                ptw = [Wires(w) for w in pytree_wires]
                super().__init__(ops, ptw, Wires(wires))

        inner = [DynOp(0.1, wires=7), DynOp(0.2, wires=8)]
        op = Composite(inner, [0, [1, 2], [3, 4]], [5, 6])
        # Wires are ordered using wire_argnames, so `wires` come before `pytree_wires`
        assert op.wires == Wires([5, 6, 0, 1, 2, 3, 4, 7, 8])

    def test_non_hybrid_wire_arg_auto_wrapped_in_constructor(self):
        """Test that non-hybrid wire arguments are wrapped in ``Wires`` by the
        ``Operator2`` constructor even if the subclass forwards a raw value."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                # No Wires() wrapping here — the constructor should do it.
                super().__init__(phi, wires=wires)

        op = Op(0.5, wires=0)
        assert isinstance(op.arguments["wires"], Wires)
        assert op.arguments["wires"] == Wires([0])

    @pytest.mark.parametrize(
        "raw_wires, expected",
        [
            (0, Wires([0])),
            ([0, 1, 2], Wires([0, 1, 2])),
            ((0, 1), Wires([0, 1])),
            (range(3), Wires([0, 1, 2])),
            (Wires([3, 4]), Wires([3, 4])),
        ],
    )
    def test_non_hybrid_wire_arg_auto_wrapped_various_inputs(self, raw_wires, expected):
        """Test that the constructor accepts a variety of raw inputs for wire
        arguments and canonicalizes them to a ``Wires`` instance."""

        class Op(Operator2):
            def __init__(self, wires):
                super().__init__(wires=wires)

        op = Op(wires=raw_wires)
        assert op.arguments["wires"] == expected
        assert op.wires == expected

    def test_hybrid_wire_arg_with_non_wires_leaf_error(self):
        """Test that a hybrid wire argument whose leaves are not ``Wires``
        instances raises a ``TypeError``."""

        class Op(Operator2):
            wire_argnames = ("wires",)
            hybrid_argnames = ("wires",)

            def __init__(self, wires):
                # Forwards the raw list without wrapping each leaf in Wires.
                super().__init__(wires=wires)

        with pytest.raises(ValueError, match="Hybrid wires argument 'wires' is invalid"):
            _ = Op(wires=[0, [1, 2]])

    def test_wire_size_match(self):
        """Test that a wire argument matching the declared ``wire_sizes`` is accepted."""

        class Op(Operator2):
            wire_sizes = (2,)

            def __init__(self, wires):
                super().__init__(wires=wires)

        op = Op(wires=[0, 1])
        assert op.arguments["wires"] == Wires([0, 1])

    def test_wire_size_mismatch_error(self):
        """Test that supplying a different number of wires than ``wire_sizes`` raises an error."""

        class Op(Operator2):
            wire_sizes = (2,)

            def __init__(self, wires):
                super().__init__(wires=wires)

        with pytest.raises(
            ValueError, match="Incorrect number of wires for 'Op.wires'. Expected 2 wires but got 3"
        ):
            Op(wires=[0, 1, 2])

    @pytest.mark.parametrize("num_wires", (1, 3, 7))
    def test_wire_size_none_accepts_any_count(self, num_wires):
        """Test that ``wire_sizes`` of ``None`` (default) accepts any number of wires."""

        class Op(Operator2):
            def __init__(self, wires):
                super().__init__(wires=wires)

        _ = Op(wires=list(range(num_wires)))

    def test_wire_sizes_mixed(self):
        """Test that fixed and ``None`` wire sizes are individually enforced."""

        class Op(Operator2):
            wire_argnames = ("wires", "ctrl_wires")
            wire_sizes = (2, None)

            def __init__(self, wires, ctrl_wires):
                super().__init__(wires=wires, ctrl_wires=ctrl_wires)

        _ = Op(wires=[0, 1], ctrl_wires=[2, 3, 4])

        with pytest.raises(
            ValueError, match="Incorrect number of wires for 'Op.wires'. Expected 2 wires but got 1"
        ):
            Op(wires=[0], ctrl_wires=[1])

    def test_wire_sizes_none_skips_check_for_hybrid_wire_arg(self):
        """Test that a hybrid wire argument is not size-checked."""

        class Op(Operator2):
            wire_argnames = ("pytree_wires",)
            hybrid_argnames = ("pytree_wires",)

            def __init__(self, pytree_wires):
                super().__init__([Wires(w) for w in pytree_wires])

        # Varying leaf counts and per-leaf sizes are all accepted.
        _ = Op([[0, 1, 2], [3], [4, 5]])

    def test_op_is_queued_on_init(self):
        """Test that instantiating an operator appends it to the active queue."""

        with AnnotatedQueue() as q:
            op = DynOp(0.5, wires=0)

        assert len(q) == 1
        assert list(q.keys())[0].obj is op


class TestInitExpectedArgtypesValidation:
    """Tests for runtime ``arg_specs`` argument validation."""

    def test_valid_arg_specs_accepts_matching_args(self):
        """Test that matching dynamic and wire arguments pass validation."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            arg_specs = {
                "phi": AbstractArray((), float),
                "wires": AbstractWires(2),
            }

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        op = Op(0.5, wires=[0, 1])
        assert op.arguments["phi"] == 0.5
        assert op.wires == Wires([0, 1])

        op1 = Op(np.array(0.5), wires=[0, 1])
        assert op1.arguments["phi"] == np.array(0.5)
        assert op1.wires == Wires([0, 1])

    def test_numpy_scalar_passes_validation(self):
        """Test that numpy scalar dynamic arguments pass ``arg_specs`` validation."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            arg_specs = {
                "phi": AbstractArray((), float),
                "wires": AbstractWires(2),
            }

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        op = Op(np.array(0.5), wires=[0, 1])
        assert op.arguments["phi"] == np.array(0.5)

    def test_no_validation_without_arg_specs(self):
        """Test that operators without ``arg_specs`` skip type validation."""

        op = DynOp(0.5, wires=[0, 1, 2])
        assert op.wires == Wires([0, 1, 2])

    def test_dynamic_arg_wrong_shape_error(self):
        """Test that a dynamic argument with the wrong shape raises an error."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            arg_specs = {
                "phi": AbstractArray((2,), float),
                "wires": AbstractWires(1),
            }

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        with pytest.raises(
            ValueError,
            match=r"Parameter 'phi' does not match the operator's expected 'arg_specs' shape. Expected \(2,\) but received \(1,\)",
        ):
            Op(np.array([0.5]), wires=0)

    def test_dynamic_arg_wrong_dtype_error(self):
        """Test that a dynamic argument with the wrong dtype raises an error."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            arg_specs = {
                "phi": AbstractArray((), int),
                "wires": AbstractWires(1),
            }

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        with pytest.raises(
            ValueError,
            match=r"Parameter 'phi' does not match the operator's expected 'arg_specs' dtype. Expected int64 but received float64",
        ):
            Op(0.5, wires=0)

    def test_weak_type_allows_python_scalar(self):
        """Test that ``arg_specs`` weak types accept Python scalar inputs."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            arg_specs = {
                "phi": AbstractArray((), float),
                "wires": AbstractWires(2),
            }

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        op = Op(0.5, wires=[0, 1])
        assert op.arguments["phi"] == 0.5

    def test_wire_arg_wrong_length_error(self):
        """Test that a wire argument with the wrong length raises an error."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            arg_specs = {
                "phi": AbstractArray((), float),
                "wires": AbstractWires(2),
            }

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        with pytest.raises(
            ValueError, match="Incorrect number of wires for 'Op.wires'. Expected 2 wires but got 1"
        ):
            Op(0.5, wires=[0])


class TestProperties:
    """Tests for public properties of ``Operator2``."""

    def test_arguments(self):
        """Test that ``arguments`` maps all arguments to their values."""
        op = FullOp(0.5, "info", [], wires=0)
        assert op.arguments == {
            "phi": 0.5,
            "static": "info",
            "hybrid": [],
            "wires": Wires([0]),
        }

    def test_dynamic_args(self):
        """Test that ``dynamic_args`` is set correctly."""
        op = FullOp(0.5, "info", [], wires=0)
        assert op.dynamic_args == {"phi": 0.5}

    def test_static_args(self):
        """Test that ``static_args`` is set correctly."""
        op = FullOp(0.5, "info", [], wires=0)
        assert op.static_args == {"static": "info"}

    def test_wire_args(self):
        """Test that ``wire_args`` is set correctly."""
        op = FullOp(0.5, "info", [], wires=0)
        assert op.wire_args == {"wires": Wires([0])}

    def test_hybrid_args(self):
        """Test that ``hybrid_args`` is set correctly."""
        op = FullOp(0.5, "info", [], wires=0)
        assert op.hybrid_args == {"hybrid": []}

    def test_compilable_args(self):
        """Test that ``compilable_args`` is set correctly."""

        class Op(Operator2):
            compilable_argnames = ("pw",)

            def __init__(self, pw, wires):
                super().__init__(pw, wires=wires)

        op = Op("XY", wires=[0, 1])
        assert op.compilable_args == {"pw": "XY"}

    def test_name(self):
        """Test that ``name`` is the same as the class name."""
        op = DynOp(0.5, wires=0)
        assert op.name == op.__class__.__name__

    def test_wires(self):
        """Test ``wires`` is set correctly."""

        class Op(Operator2):
            wire_argnames = ("wires1", "wires")

            def __init__(self, wires, wires1):
                super().__init__(Wires(wires), Wires(wires1))

        op = Op([0, 1], [2, 3, 4])
        assert op.wires == Wires([2, 3, 4, 0, 1])


class TestBroadcasting:
    """Tests for parameter broadcasting."""

    def test_broadcasted_scalar_passes_validation(self):
        """Test that broadcasted scalar parameters pass ``arg_specs`` validation."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            ndim_params = (0,)
            arg_specs = {
                "phi": AbstractArray((), float),
                "wires": AbstractWires(1),
            }

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        op = Op([0.5, 0.6, 0.7], wires=0)
        assert op.arguments["phi"] == [0.5, 0.6, 0.7]

    def test_broadcasted_array_shape_validation(self):
        """Test that broadcasted parameters validate the non-broadcasting dimensions
        against ``arg_specs``."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            ndim_params = (1,)
            arg_specs = {
                "phi": AbstractArray((2,), float),
                "wires": AbstractWires(1),
            }

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        _ = Op(np.ones((4, 2)), wires=0)

    def test_broadcasted_array_wrong_shape_error(self):
        """Test that an invalid broadcasted parameter shape raises an error."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            ndim_params = (1,)
            arg_specs = {
                "phi": AbstractArray((2,), float),
                "wires": AbstractWires(1),
            }

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        expected_msg = r"Parameter 'phi' does not match the operator's expected 'arg_specs' shape. Expected \(2,\) \(non-broadcasting dimensions\) but received \(4, 3\)."
        with pytest.raises(ValueError, match=expected_msg):
            _ = Op(np.ones((4, 3)), wires=0)

        expected_msg = r"Parameter 'phi' does not match the operator's expected 'arg_specs' shape. Expected \(2,\) \(non-broadcasting dimensions\) but received \(4, 1, 2\)."
        with pytest.raises(ValueError, match=expected_msg):
            _ = Op(np.ones((4, 1, 2)), wires=0)

    def test_broadcasted_array_wrong_dtype_error(self):
        """Test that broadcasted parameters still enforce dtype compatibility."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            ndim_params = (0,)
            arg_specs = {
                "phi": AbstractArray((), int),
                "wires": AbstractWires(1),
            }

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        expected_msg = "Parameter 'phi' does not match the operator's expected 'arg_specs' dtype. Expected int64 but received float64"
        with pytest.raises(ValueError, match=expected_msg):
            _ = Op(np.array([0.5, 0.6]), wires=0)

    def test_broadcasted_inferred_ndim_from_arg_specs(self):
        """Test broadcast validation when ``ndim_params`` is inferred from ``arg_specs``."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            arg_specs = {
                "phi": AbstractArray((2, 3), float),
                "wires": AbstractWires(1),
            }

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        _ = Op(np.ones((5, 2, 3)), wires=0)

    @pytest.mark.parametrize("data, exp_batch_size", [(1.1, None), ([1.1, 2.2], 2)])
    def test_batch_size(self, data, exp_batch_size):
        """Test that ``batch_size`` is correct when ndim_params are specified
        during class definition."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            ndim_params = (0,)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        op = Op(data, wires=0)
        assert op._batch_size is _UNSET_BATCH_SIZE
        assert op.batch_size == exp_batch_size

    @pytest.mark.parametrize(
        "data", [(np.empty((2, 2)), np.empty((4, 4, 4, 4))), (np.array(1.5), np.empty((4,)))]
    )
    def test_inferred_ndim_params(self, data):
        """Test that ``ndim_params`` is assumed to be the same as the number of dimensions of
        input dynamic parameters if not set as a class variable."""

        class Op(Operator2):
            dynamic_argnames = ("a", "b")

            def __init__(self, a, b, wires):
                super().__init__(a, b, Wires(wires))

        op = Op(*data, wires=0)
        expected_ndims = tuple(d.ndim for d in data)
        assert op._ndim_params is _UNSET_BATCH_SIZE
        assert op.ndim_params == expected_ndims

    def test_wrong_ndim_error(self):
        """Test that parameters with wrong dimensionality raise an error."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            ndim_params = (0,)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        op = Op([[[0.5]]], wires=0)
        with pytest.raises(ValueError, match=r"wrong number\(s\) of dimensions"):
            _ = op.batch_size

    def test_mismatched_broadcasting_error(self):
        """Test that different broadcast dimensions across parameters raise an error."""

        class Op(Operator2):
            dynamic_argnames = ("a", "b")
            ndim_params = (0, 0)

            def __init__(self, a, b, wires):
                super().__init__(a, b, wires=wires)

        op = Op([0.3] * 4, [0.4] * 3, wires=0)
        with pytest.raises(ValueError, match="Broadcasting was attempted"):
            _ = op.batch_size


class TestEquality:
    """Tests for operator equality. Since ``Operator2.__eq__`` uses ``qp.equal``
    under the hood, the bulk of testing for operator equality are elsewhere."""

    def test_eq_identical_instance(self):
        """Test that an operator is equal to itself."""
        op = DynOp(0.5, wires=0)
        assert op == op  # pylint: disable=comparison-with-itself

    def test_eq_equal_operators(self):
        """Test that distinct operators with the same arguments are equal."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.5, wires=0)
        assert op1 == op2
        assert op2 == op1

    def test_eq_different_dynamic_args(self):
        """Test that operators with different dynamic args are not equal."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.6, wires=0)
        assert op1 != op2
        assert op2 != op1

    def test_eq_different_wires(self):
        """Test that operators acting on different wires are not equal."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.5, wires=1)
        assert op1 != op2
        assert op2 != op1

    def test_eq_different_type(self):
        """Test that two ``Operator2`` subclasses are not equal."""

        class Other(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        op1 = DynOp(0.5, wires=0)
        op2 = Other(0.5, wires=0)
        assert op1 != op2
        assert op2 != op1

    @pytest.mark.parametrize("other", [42, "string", 0.5, None, [DynOp(0.5, wires=0)]])
    def test_eq_against_non_operator(self, other):
        """Test that comparing an operator with a non-``Operator2`` value returns False."""
        op = DynOp(0.5, wires=0)
        assert op != other


class TestHash:
    """Tests for ``Operator2`` hashing."""

    def test_op_is_hashable(self):
        """Test that ``Operator2`` is hashable."""
        op = DynOp(0.5, wires=0)
        assert isinstance(hash(op), int)

    def test_op_can_be_dict_key(self):
        """Test that an ``Operator2`` instance can be used as a dict key."""
        op = DynOp(0.5, wires=0)
        d = {op: "value"}
        assert d[DynOp(0.5, wires=0)] == "value"

    def test_set_deduplicates_equal_operators(self):
        """Test that equal operators do not make unique entries when collected into a set."""
        ops = {DynOp(0.5, wires=0) for _ in range(5)}
        assert len(ops) == 1

    def test_equal_ops_hash(self):
        """Test the hash-equality invariant: ``a == b`` implies ``hash(a) == hash(b)``."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.5, wires=0)
        assert op1 == op2
        assert hash(op1) == hash(op2)

    def test_abstract_op_hash_contract(self):
        """Tests that an abstract op and abstractified op have the same hash."""
        op1 = DynOp(Float, Wire[2])
        op2 = abstractify(DynOp(0.5, [0, 1]))
        assert op1 == op2
        assert hash(op1) == hash(op2)

    def test_close_floats_same_hash(self):
        """Test that float values that round to the same 10 decimal places hash equally."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.5 + 1e-12, wires=0)
        assert hash(op1) == hash(op2)

    def test_different_args_hash(self):
        """Test that differing arguments produces different hashes."""
        op1 = FullOp(phi=0.5, static="a", hybrid=[0, 1], wires=0)
        op2 = FullOp(phi=0.6, static="ab", hybrid=[1, 1], wires=1)
        assert hash(op1) != hash(op2)

    def test_different_types_different_hash(self):
        """Test that operators of different types produce different hashes."""

        class Other(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        op1 = DynOp(0.5, wires=0)
        op2 = Other(0.5, wires=0)
        assert hash(op1) != hash(op2)

    def test_different_hybrid_operator_args_different_hash(self):
        """Test that differing hybrid arguments that have operator leaves produces different
        hashes."""
        op1 = FullOp(0.5, "a", [DynOp(0.5, wires=0)], wires=0)
        op2 = FullOp(0.5, "a", [DynOp(0.6, wires=0)], wires=0)
        assert hash(op1) != hash(op2)

    def test_hybrid_with_wires_leaf_equal_hash(self):
        """Test that hybrid arguments with ``Wires`` leaves hash equally for the same values."""
        op1 = HybridWireOp([[0, 1], [2]])
        op2 = HybridWireOp([[0, 1], [2]])
        assert hash(op1) == hash(op2)

        op3 = HybridWireOp([[0, 3], [2]])
        assert hash(op1) != hash(op3)

    def test_hybrid_different_pytree_structure_different_hash(self):
        """Test that hybrid arguments with different pytree structures hash differently."""
        op1 = FullOp(0.5, "a", [0.1, 0.2], wires=0)
        op2 = FullOp(0.5, "a", [0.1, [0.2]], wires=0)
        assert hash(op1) != hash(op2)

    @pytest.mark.parametrize("name", ["RX", "RY", "RZ", "PhaseShift", "Rot"])
    def test_rotation_gate_modulo_2pi(self, name):
        """Test that operators with designated rotation gate names hash modulo 2 * pi."""

        Op = type(
            name,
            (Operator2,),
            {
                "dynamic_argnames": ("phi",),
                "__init__": lambda self, phi, wires: Operator2.__init__(self, phi, wires=wires),
            },
        )

        op1 = Op(0.5, wires=0)
        op2 = Op(0.5 + 2 * np.pi, wires=0)
        assert hash(op1) == hash(op2)

    def test_non_rotation_no_modulo(self):
        """Test that non-rotation operators do not apply modulo 2 * pi hashing."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.5 + 2 * np.pi, wires=0)
        assert hash(op1) != hash(op2)


class TestPytreeMethods:
    """Tests for ``_flatten`` and ``_unflatten``."""

    def test_static_arg_in_metadata(self):
        """Test that flattening and unflattening work correctly when there are static_args."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            static_argnames = ("static",)

            def __init__(self, phi, static, wires):
                super().__init__(phi, static, wires=wires)

        op = Op(0.5, "bar", wires=0)
        data, metadata = op._flatten()

        assert data == ([0.5], [Wires([0])], [])
        assert metadata == ("bar",)

        new_op = Op._unflatten(data, metadata)
        assert new_op.arguments == op.arguments
        assert new_op.wires == op.wires

    def test_compilable_arg_in_metadata(self):
        """Test that flattening and unflattening work correctly when there are compilable_args."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            compilable_argnames = ("static",)

            def __init__(self, phi, static, wires):
                super().__init__(phi, static, wires=wires)

        op = Op(1.5, "bar", wires=[0, 1])
        data, metadata = op._flatten()

        assert data == ([1.5], [Wires([0, 1])], [])
        assert metadata == ("bar",)

        new_op = Op._unflatten(data, metadata)
        assert new_op.arguments == op.arguments
        assert new_op.wires == op.wires

    def test_flatten_order_dynamic_wire_hybrid(self):
        """Test that data is ordered as dynamic, then wires, then non-wire hybrid."""

        op = FullOp(0.5, "static", [-1, -2, -3], wires=0)
        data, metadata = op._flatten()

        assert data == ([0.5], [Wires([0])], [[-1, -2, -3]])
        assert metadata == ("static",)

        new_op = FullOp._unflatten(data, metadata)
        assert new_op.arguments == op.arguments
        assert new_op.wires == op.wires

    def test_hybrid_overlapping_with_wire_not_duplicated(self):
        """Test that hybrid wire arguments are flattened and unflattened correctly."""

        class Op(Operator2):
            wire_argnames = ("wires",)
            hybrid_argnames = ("wires",)

            def __init__(self, wires):
                wires = [Wires(w) for w in wires]
                super().__init__(wires=wires)

        op = Op(wires=[0, [1, 2], (3,)])
        data, metadata = op._flatten()

        assert data == ([], [[Wires([0]), Wires([1, 2]), Wires([3])]], [])
        assert metadata == ()

        new_op = Op._unflatten(data, metadata)
        assert new_op.arguments == op.arguments
        assert new_op.wires == op.wires

    def test_unflatten_does_not_queue(self):
        """Test that reconstructing an operator via ``_unflatten`` does not queue it."""
        op = DynOp(0.5, wires=0)
        data, metadata = op._flatten()

        with AnnotatedQueue() as q:
            _ = DynOp._unflatten(data, metadata)

        assert len(q) == 0


class TestDynamicProperties:
    """Tests for the dynamic properties generated using operator parameters."""

    def test_non_existent_attr(self):
        """Test that an attribute error is raised if trying to access a property
        that doesn't exist."""
        op = DynOp(phi=1.5, wires=0)
        with pytest.raises(AttributeError, match="object has no attribute"):
            _ = op.invalid_attr

    def test_signature_parameter_property(self):
        """Test that operator parameters are exposed as properties."""
        op = FullOp(phi=0.5, static="static", hybrid=[], wires=0)
        assert op.phi == 0.5
        assert op.static == "static"
        assert op.hybrid == []

    def test_explicit_class_attribute_not_overridden(self):
        """Test that attributes present in the class are not overridden by dynamic properties."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            phi = -1

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        op = Op(phi=100, wires=0)
        assert op.phi == -1


class TestDunderMethods:
    """Tests for ``Operator2`` dunder methods."""

    def test_repr_with_abstract_args(self):
        """Tests that abstract wires properly render."""

        op = DynOp(AbstractArray((1, 2), float), AbstractWires(1))
        assert (
            repr(op)
            == "DynOp(phi=AbstractArray((1, 2), float64, weak_type=True), wires=AbstractWires(1))"
        )

    def test_repr_with_dynamic_args(self):
        """Test that __repr__ includes dynamic parameters if present."""
        op = DynOp(0.5, wires=[0, 1])
        assert repr(op) == "DynOp(phi=0.5, wires=[0, 1])"

    def test_repr_without_dynamic_args(self):
        """Test that __repr__ prints without dynamic parameters if there are none."""

        class Op(Operator2):
            def __init__(self, wires):
                super().__init__(wires=wires)

        op = Op(wires=0)
        assert repr(op) == "Op(0)"

        op = Op(wires="a")
        assert repr(op) == "Op('a')"

    @pytest.mark.parametrize("num_wires", [1, 2])
    def test_repr_without_dynamic_args_abstract_wires(self, num_wires):
        """Test that __repr__ prints without dynamic parameters if there are none."""

        class Op(Operator2):
            def __init__(self, wires):
                super().__init__(wires=wires)

        op = Op(wires=AbstractWires(num_wires))
        assert repr(op) == f"Op(wires={AbstractWires(num_wires)!r})"

    def test_repr_without_dynamic_args_multiwire(self):
        """Test that __repr__ prints without dynamic parameters if there are none."""

        class Op(Operator2):
            def __init__(self, wires):
                super().__init__(wires=wires)

        op = Op(wires=[0, 1, 2])
        assert repr(op) == "Op(wires=[0, 1, 2])"

    def test_repr_without_dynamic_args_different_wire_argname(self):
        """Test that __repr__ prints without dynamic parameters if there are none."""

        class Op(Operator2):
            wire_argnames = ("my_wires",)

            def __init__(self, my_wires):
                super().__init__(my_wires=my_wires)

        op = Op(my_wires=0)
        assert repr(op) == "Op(0)"

    def test_repr_with_hybrid_wires(self):
        """Test that __repr__ prints correctly if there are hybrid wire arguments."""

        class Op(Operator2):
            hybrid_argnames = ("wires",)

            def __init__(self, wires):
                super().__init__(wires=[Wires(w) for w in wires])

        op = Op(wires=[[0], 1, [2, 3]])
        assert repr(op) == "Op(wires=[[0], [1], [2, 3]])"

    def test_copy(self):
        """Test that shallow copies of operators are created correctly."""
        op = FullOp(0.5, static="static", hybrid=[], wires=0)
        op_copy = copy.copy(op)

        assert op_copy is not op
        assert type(op_copy) is type(op)

        for attr in ("phi", "static", "hybrid", "wires", "arguments"):
            assert getattr(op_copy, attr) == getattr(op, attr)
            # Shallow copy so stored attributes must be the same references
            assert getattr(op_copy, attr) is getattr(op, attr)

    def test_deepcopy(self):
        """Test that deep copies of operators are created correctly."""
        # Putting phi in a container because integer pointers have special handling in Python
        phi = {0.5}
        static = {"foo": 1.5}
        hybrid = [1, 2, 3]

        op = FullOp(phi=phi, static=static, hybrid=hybrid, wires=0)
        op_deep = copy.deepcopy(op)

        assert op_deep is not op

        for attr in ("phi", "static", "hybrid", "wires", "arguments"):
            assert getattr(op_deep, attr) == getattr(op, attr)
            # Deep copy so stored attributes must NOT be the same references
            assert getattr(op_deep, attr) is not getattr(op, attr)


class TestPublicProperties:
    """Tests for public properties of ``Operator2``."""

    def test_arithmetic_depth_default(self):
        """Test that the default ``arithmetic_depth`` is 0."""
        op = DynOp(0.5, wires=0)
        assert op.arithmetic_depth == 0

    def test_is_verified_hermitian_default(self):
        """Test that the default ``is_verified_hermitian`` is False."""
        op = DynOp(0.5, wires=0)
        assert op.is_verified_hermitian is False

    def test_pauli_rep_default(self):
        """Test that ``pauli_rep`` is ``None`` by default."""

        op = DynOp(0.5, wires=0)
        assert op.pauli_rep is None


class TestLabel:
    """Tests for the ``label`` method."""

    def test_no_dynamic_args(self):
        """Test that ``label`` returns just the class name when there
        are no dynamic args."""

        class Op(Operator2):
            def __init__(self, wires):
                super().__init__(wires=Wires(wires))

        assert Op(wires=0).label() == "Op"

    @pytest.mark.parametrize(
        "category", ["static_argnames", "compilable_argnames", "hybrid_argnames"]
    )
    def test_non_dynamic_not_included(self, category):
        """Test that arguments not in ``dynamic_argnames`` aren't included
        in the label."""

        def __init__(self, phi, arg, wires):
            Operator2.__init__(self, phi, arg, Wires(wires))

        attrs = {
            "dynamic_argnames": ("phi",),
            "wire_argnames": ("wires",),
            category: ("arg",),
            "__init__": __init__,
        }
        # This creates a class while allowing us to parametrize the attributes
        # that we want to test.
        OpClass = type("OpClass", (Operator2,), attrs)

        assert OpClass(phi=1.5, arg=[1, 2, 3], wires=0).label() == "OpClass"

    def test_no_dynamic_args_with_base_label(self):
        """Test that ``base_label`` overrides the class name even with
        no dynamic args."""

        class Op(Operator2):
            def __init__(self, wires):
                super().__init__(wires=Wires(wires))

        assert Op(wires=0).label(base_label="custom") == "custom"

    def test_scalar_decimals_none_excludes_param(self):
        """Test that scalar parameters are excluded from the label when
        ``decimals`` is ``None``."""
        op = DynOp(1.23456, wires=0)
        assert op.label() == "DynOp"

    def test_scalar_decimals_formats_param(self):
        """Test that scalar parameters are formatted to ``decimals`` places."""
        op = DynOp(1.23456, wires=0)
        assert op.label(decimals=2) == "DynOp\n(1.23)"

    def test_base_label_with_decimals(self):
        """Test that ``base_label`` and ``decimals`` can be combined."""
        op = DynOp(1.23456, wires=0)
        assert op.label(decimals=3, base_label="custom") == "custom\n(1.235)"

    def test_multiple_dynamic_args(self):
        """Test that multiple dynamic params are joined with ``,\\n``."""

        class TwoArgs(Operator2):
            dynamic_argnames = ("phi", "theta")

            def __init__(self, phi, theta, wires):
                super().__init__(phi, theta, wires=Wires(wires))

        op = TwoArgs(1.23, 4.567, wires=0)
        assert op.label(decimals=2) == "TwoArgs\n(1.23,\n4.57)"

    def test_matrix_param_no_cache_omitted(self):
        """Test that a matrix parameter is omitted from the label when no
        cache is provided."""

        class MatOp(Operator2):
            dynamic_argnames = ("U",)

            def __init__(self, U, wires):
                super().__init__(U, wires=Wires(wires))

        op = MatOp(np.eye(2), wires=0)
        # Without a cache, matrix-valued params produce empty strings.
        assert op.label(decimals=2) == "MatOp"

    def test_matrix_param_with_cache(self):
        """Test that matrix parameters are recorded in the cache and referenced
        as ``M{index}``."""

        class MatOp(Operator2):
            dynamic_argnames = ("U",)

            def __init__(self, U, wires):
                super().__init__(U, wires=Wires(wires))

        cache = {"matrices": []}
        op = MatOp(np.eye(2), wires=0)

        assert op.label(cache=cache) == "MatOp\n(M0)"
        assert len(cache["matrices"]) == 1
        assert np.allclose(cache["matrices"][0], np.eye(2))

    def test_matrix_param_cache_reuse(self):
        """Test that the same matrix is reused from the cache rather than appended again."""

        class MatOp(Operator2):
            dynamic_argnames = ("U",)

            def __init__(self, U, wires):
                super().__init__(U, wires=Wires(wires))

        cache = {"matrices": []}
        op1 = MatOp(np.eye(2), wires=0)
        op2 = MatOp(np.eye(2), wires=1)

        assert op1.label(cache=cache) == "MatOp\n(M0)"
        assert op2.label(cache=cache) == "MatOp\n(M0)"
        assert len(cache["matrices"]) == 1

    def test_matrix_param_cache_growth(self):
        """Test that distinct matrices are appended with incrementing indices."""

        class MatOp(Operator2):
            dynamic_argnames = ("U",)

            def __init__(self, U, wires):
                super().__init__(U, wires=Wires(wires))

        cache = {"matrices": []}
        op1 = MatOp(np.eye(2), wires=0)
        op2 = MatOp(np.eye(4), wires=[0, 1])

        assert op1.label(cache=cache) == "MatOp\n(M0)"
        assert op2.label(cache=cache) == "MatOp\n(M1)"
        assert len(cache["matrices"]) == 2

    def test_subclass_override(self):
        """Test that a subclass can override ``label``."""

        class Op(Operator2):
            def __init__(self, wires):
                super().__init__(wires=Wires(wires))

            # pylint: disable=unused-argument
            def label(self, decimals=None, base_label=None, cache=None):
                return "custom_label"

        assert Op(wires=0).label() == "custom_label"


class TestGeneralMethods:
    """Tests for various general methods in ``Operator2``."""

    def test_pow_zero_returns_empty(self):
        """Test that ``op.pow(0)`` returns an empty list."""
        op = DynOp(0.5, wires=0)
        assert op.pow(0) == []

    def test_pow_one_returns_single(self):
        """Test that ``op.pow(1)`` returns a single-element list with a copy of the operator."""
        op = DynOp(0.5, wires=0)
        result = op.pow(1)

        assert len(result) == 1
        assert result[0] == op
        assert result[0] is not op

    def test_pow_positive_integer(self):
        """Test that ``op.pow(z)`` for positive integer ``z`` returns ``z`` copies."""
        op = DynOp(0.5, wires=0)
        result = op.pow(3)

        assert len(result) == 3
        assert all(r == op for r in result)
        assert all(r is not op for r in result)

        # Each op must be a new copy
        assert len({id(r) for r in result}) == 3

    def test_pow_queuing(self):
        """Test that ``op.pow(z)`` inside a queuing context queues ``z`` additional copies."""
        with AnnotatedQueue() as q:
            op = DynOp(0.5, wires=0)
            result = op.pow(2)

        # 1 entry from original op + 2 entries from pow.
        assert len(q) == 3
        assert len(result) == 2

    @pytest.mark.parametrize("z", [-1, -2, 1.5, 0.5])
    def test_pow_invalid_error(self, z):
        """Test that ``op.pow`` raises ``PowUndefinedError`` for negative or non-integer exponents."""
        op = DynOp(0.5, wires=0)
        with pytest.raises(PowUndefinedError):
            _ = op.pow(z)

    def test_subclass_pow_override(self):
        """Test that a subclass can override ``pow``."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=Wires(wires))

            def pow(self, z):
                return [type(self)(self.phi * z, wires=self.wires)]

        op = Op(2.0, wires=0)
        result = op.pow(3)
        assert len(result) == 1
        assert result[0].phi == 6.0

    def test_map_wires_basic(self):
        """Test that ``map_wires`` maps the wires of the operator according
        to the wire map."""
        op = DynOp(0.5, wires=0)
        new_op = op.map_wires({0: "a"})

        # Check that the original op is unchanged
        assert op.wires == Wires([0])
        assert op.arguments["wires"] == Wires([0])
        # Check new op
        assert new_op == DynOp(0.5, wires="a")
        assert new_op.wires == Wires(["a"])
        assert new_op.arguments["wires"] == Wires(["a"])

    def test_map_wires_unmapped_labels_preserved(self):
        """Test that wire labels missing from the wire map are kept as-is."""
        op = DynOp(0.5, wires=[0, 1, 2])
        new_op = op.map_wires({0: "a", 2: "c"})

        assert op.wires == Wires([0, 1, 2])
        assert op.arguments["wires"] == Wires([0, 1, 2])
        assert new_op == DynOp(0.5, wires=["a", 1, "c"])
        assert new_op.wires == Wires(["a", 1, "c"])
        assert new_op.arguments["wires"] == Wires(["a", 1, "c"])

    def test_map_wires_multiple_wire_args(self):
        """Test that ``map_wires`` maps wires across multiple wire arguments."""

        class TwoWireOp(Operator2):
            wire_argnames = ("wires", "ctrl_wires")

            def __init__(self, wires, ctrl_wires):
                super().__init__(Wires(wires), Wires(ctrl_wires))

        op = TwoWireOp(wires=[0, 1], ctrl_wires=[2])
        new_op = op.map_wires({0: "a", 1: "b", 2: "c"})

        assert op.wires == Wires([0, 1, 2])
        assert new_op == TwoWireOp(wires=["a", "b"], ctrl_wires=["c"])
        assert new_op.wires == Wires(["a", "b", "c"])

    def test_map_wires_pytree_hybrid_wires(self):
        """Test that ``map_wires`` correctly maps pytree-structured wires inside a hybrid arg."""

        class PytreeWiresOp(Operator2):
            wire_argnames = ("wires",)
            hybrid_argnames = ("wires",)

            def __init__(self, wires):
                wires = [Wires(w) for w in wires]
                super().__init__(wires=wires)

        op = PytreeWiresOp(wires=[[0], [1, 2]])
        new_op = op.map_wires({0: "a", 1: "b"})

        assert op.wires == Wires([0, 1, 2])
        assert new_op == PytreeWiresOp(wires=[["a"], ["b", 2]])
        assert new_op.wires == Wires(["a", "b", 2])

    def test_map_wires_op_argument(self):
        """Test that ``map_wires`` correctly maps hybrid arguments with operator leaves."""
        op = FullOp(0.5, static="static", hybrid=[DynOp(1.5, wires=[2, 3, 4])], wires=[0, 1])
        new_op = op.map_wires({0: "a", 2: "b", 3: "c"})

        assert op.wires == Wires([0, 1, 2, 3, 4])
        assert new_op == FullOp(
            0.5, static="static", hybrid=[DynOp(1.5, wires=["b", "c", 4])], wires=["a", 1]
        )
        assert new_op.wires == Wires(["a", 1, "b", "c", 4])

    def test_map_wires_pauli_rep(self):
        """Test that ``Operator2.map_wires`` maps the ``pauli_rep`` correctly
        when the subclass computes it during construction."""

        class PauliRepOp(Operator2):
            wire_argnames = ("wires",)

            def __init__(self, wires):
                super().__init__(wires=Wires(wires))
                self._pauli_rep = PauliSentence(
                    {PauliWord({self.wires[0]: "X", self.wires[1]: "Y"}): 1.0}
                )

        op = PauliRepOp(wires=[0, 1])
        new_op = op.map_wires({0: "a", 1: "b"})
        assert new_op.pauli_rep == PauliSentence({PauliWord({"a": "X", "b": "Y"}): 1.0})

    def test_simplify_default(self):
        """Test that ``simplify`` returns the operator itself by default."""
        op = DynOp(0.5, wires=0)
        assert op.simplify() is op

    def test_subclass_simplify_override(self):
        """Test that a subclass can override ``simplify``."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=Wires(wires))

            def simplify(self):
                return type(self)(0.0, wires=self.wires)

        op = Op(0.5, wires=0)
        simplified = op.simplify()

        # Assert that the original op is left unchanged
        assert op.phi == 0.5
        assert simplified == Op(0.0, wires=0)


class TestHasRepresentations:
    """Tests for the ``has_**`` representation properties."""

    def test_has_adjoint_default(self):
        """Test that ``has_adjoint`` is False by default."""
        assert DynOp.has_adjoint is False
        assert Operator2.has_adjoint is False

    def test_has_adjoint_true_when_adjoint_overridden(self):
        """Test that ``has_adjoint`` is True when the subclass overrides ``adjoint``."""

        class SelfAdj(Operator2):
            def __init__(self, wires):
                super().__init__(wires=Wires(wires))

            def adjoint(self):
                return self

        assert SelfAdj.has_adjoint is True

    def test_has_matrix_default(self):
        """Test that ``has_matrix`` is False when ``compute_matrix`` is not overridden."""
        assert DynOp.has_matrix is False

    def test_has_matrix_true_when_compute_overridden(self):
        """Test that ``has_matrix`` is True when ``compute_matrix`` is overridden."""

        class WithMatrix(Operator2):
            @staticmethod
            def compute_matrix(*_, **__):
                return np.eye(2)

            def __init__(self, wires):
                super().__init__(wires=wires)

        assert WithMatrix.has_matrix is True

    def test_has_matrix_true_when_method_overridden(self):
        """Test that ``has_matrix`` is True when ``matrix`` is overridden."""

        class WithMatrix(Operator2):
            # pylint: disable=unused-argument
            def matrix(self, wire_order=None):
                return np.eye(2)

            def __init__(self, wires):
                super().__init__(wires=wires)

        assert WithMatrix.has_matrix is True

    def test_has_sparse_matrix_default(self):
        """Test that ``has_sparse_matrix`` is False by default."""
        assert DynOp.has_sparse_matrix is False

    def test_has_sparse_matrix_true_when_compute_overridden(self):
        """Test that ``has_sparse_matrix`` is True when ``compute_sparse_matrix`` is overridden."""

        class WithSparse(Operator2):
            @staticmethod
            def compute_sparse_matrix(*_, **__):
                return csr_matrix(np.eye(2))

            def __init__(self, wires):
                super().__init__(wires=wires)

        assert WithSparse.has_sparse_matrix is True

    def test_has_sparse_matrix_true_when_method_overridden(self):
        """Test that ``has_sparse_matrix`` is True when ``sparse_matrix`` is overridden."""

        class WithSparse(Operator2):
            # pylint: disable=unused-argument
            def sparse_matrix(self, wire_order=None, format="csr"):
                return csr_matrix(np.eye(2))

            def __init__(self, wires):
                super().__init__(wires=wires)

        assert WithSparse.has_sparse_matrix is True

    def test_has_decomposition_default(self):
        """Test that ``has_decomposition`` is False by default."""
        assert DynOp.has_decomposition is False

    def test_has_decomposition_true_when_compute_overridden(self):
        """Test that ``has_decomposition`` is True when ``compute_decomposition`` is overridden."""

        class WithDecomp(Operator2):
            @staticmethod
            def compute_decomposition(*_, **__):
                return []

            def __init__(self, wires):
                super().__init__(wires=wires)

        assert WithDecomp.has_decomposition is True

    def test_has_decomposition_true_when_method_overridden(self):
        """Test that ``has_decomposition`` is True when ``decomposition`` is overridden."""

        class WithDecomp(Operator2):
            def decomposition(self):
                return []

            def __init__(self, wires):
                super().__init__(wires=wires)

        assert WithDecomp.has_decomposition is True

    def test_has_diagonalizing_gates_default(self):
        """Test that ``has_diagonalizing_gates`` is False by default."""
        assert DynOp.has_diagonalizing_gates is False

    def test_has_diagonalizing_gates_true_when_compute_overridden(self):
        """Test that ``has_diagonalizing_gates`` is True when ``compute_diagonalizing_gates`` is overridden."""

        class WithDiag(Operator2):
            @staticmethod
            def compute_diagonalizing_gates(*_, **__):
                return []

            def __init__(self, wires):
                super().__init__(wires=wires)

        assert WithDiag.has_diagonalizing_gates is True

    def test_has_diagonalizing_gates_true_when_method_overridden(self):
        """Test that ``has_diagonalizing_gates`` is True when ``diagonalizing_gates`` is overridden."""

        class WithDiag(Operator2):
            def diagonalizing_gates(self):
                return []

            def __init__(self, wires):
                super().__init__(wires=wires)

        assert WithDiag.has_diagonalizing_gates is True

    def test_has_generator_default(self):
        """Test that ``has_generator`` is False by default."""
        assert DynOp.has_generator is False

    def test_has_generator_true_when_method_overridden(self):
        """Test that ``has_generator`` is True when ``generator`` is overridden."""

        class WithGen(Operator2):
            def generator(self):
                return DynOp(0.5, wires=self.wires)

            def __init__(self, wires):
                super().__init__(wires=wires)

        assert WithGen.has_generator is True


# pylint: disable=unused-argument,too-many-public-methods
class TestRepresentations:
    """Tests for the various operator representation methods
    (and their corresponding ``compute_**`` static methods)."""

    def test_adjoint_default_error(self):
        """Test that the default ``adjoint`` raises ``AdjointUndefinedError``."""
        op = DynOp(0.5, wires=0)
        with pytest.raises(AdjointUndefinedError):
            op.adjoint()

    def test_subclass_adjoint_override(self):
        """Test that a subclass can override ``adjoint``."""

        class SelfAdj(Operator2):
            def __init__(self, wires):
                super().__init__(wires=Wires(wires))

            def adjoint(self):
                return type(self)(wires=self.wires)

        op = SelfAdj(wires=0)
        adj = op.adjoint()
        assert adj == op

    def test_matrix_undefined_error(self):
        """Test that the default matrix representation raises ``MatrixUndefinedError``."""
        op = DynOp(0.5, wires=0)
        with pytest.raises(MatrixUndefinedError):
            DynOp.compute_matrix(0.5, wires=0)
        with pytest.raises(MatrixUndefinedError):
            op.matrix()

    def test_sparse_matrix_undefined_error(self):
        """Test that the default sparse matrix representation raises ``SparseMatrixUndefinedError``."""
        op = DynOp(0.5, wires=0)
        with pytest.raises(SparseMatrixUndefinedError):
            DynOp.compute_sparse_matrix(0.5, wires=0)
        with pytest.raises(SparseMatrixUndefinedError):
            op.sparse_matrix()

    def test_decomposition_undefined_error(self):
        """Test that the default decomposition raises ``DecompositionUndefinedError``."""
        op = DynOp(0.5, wires=0)
        with pytest.raises(DecompositionUndefinedError):
            DynOp.compute_decomposition(0.5, wires=0)
        with pytest.raises(DecompositionUndefinedError):
            op.decomposition()

    def test_eigvals_undefined_error(self):
        """Test that the default eigenvalue representation raises ``EigvalsUndefinedError``."""
        op = DynOp(0.5, wires=0)
        with pytest.raises(EigvalsUndefinedError):
            DynOp.compute_eigvals(0.5, wires=0)
        with pytest.raises(EigvalsUndefinedError):
            op.eigvals()

    def test_diagonalizing_gates_undefined_error(self):
        """Test that the default diagonalizing gates raise ``DiagGatesUndefinedError``."""
        op = DynOp(0.5, wires=0)
        with pytest.raises(DiagGatesUndefinedError):
            DynOp.compute_diagonalizing_gates(0.5, wires=0)
        with pytest.raises(DiagGatesUndefinedError):
            op.diagonalizing_gates()

    def test_terms_undefined_error(self):
        """Test that the default terms representation raises ``TermsUndefinedError``."""
        op = DynOp(0.5, wires=0)
        with pytest.raises(TermsUndefinedError):
            op.terms()

    def test_generator_undefined_error(self):
        """Test that the default generator raises ``GeneratorUndefinedError``."""
        op = DynOp(0.5, wires=0)
        with pytest.raises(GeneratorUndefinedError, match="does not have a generator"):
            op.generator()

    def test_compute_matrix_used_by_matrix(self):
        """Test that ``matrix`` dispatches to ``compute_matrix`` with bound arguments."""

        class WithMatrix(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

            @staticmethod
            def compute_matrix(phi, wires):
                return phi * np.eye(2)

        op = WithMatrix(0.5, wires=0)
        assert np.allclose(op.matrix(), 0.5 * np.eye(2))

    def test_matrix_expands_for_wire_order(self):
        """Test that ``matrix`` expands the canonical matrix when ``wire_order`` is given."""

        class WithMatrix(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

            @staticmethod
            def compute_matrix(phi, wires):
                return phi * np.eye(2)

        op = WithMatrix(0.5, wires=1)
        canonical = op.compute_matrix(0.5, wires=op.wires)
        expanded = op.matrix(wire_order=[0, 1])
        expected = qp.math.expand_matrix(canonical, wires=op.wires, wire_order=[0, 1])
        assert np.allclose(expanded, expected)

    def test_compute_sparse_matrix_used_by_sparse_matrix(self):
        """Test that ``sparse_matrix`` dispatches to ``compute_sparse_matrix``."""

        class WithSparse(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

            @staticmethod
            def compute_sparse_matrix(phi, wires, format="csr"):
                return csr_matrix(phi * np.eye(2))

        op = WithSparse(0.5, wires=0)
        mat = op.sparse_matrix()
        assert mat.format == "csr"
        assert np.allclose(mat.toarray(), 0.5 * np.eye(2))

    def test_compute_decomposition_used_by_decomposition(self):
        """Test that ``decomposition`` dispatches to ``compute_decomposition``."""

        class WithDecomp(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

            @staticmethod
            def compute_decomposition(phi, wires):
                return [DynOp(phi, wires=wires[0])]

        op = WithDecomp(0.7, wires=0)
        decomp = op.decomposition()
        assert len(decomp) == 1
        assert decomp[0] == DynOp(0.7, wires=0)

    def test_compute_eigvals_used_by_eigvals(self):
        """Test that ``eigvals`` dispatches to ``compute_eigvals`` when defined."""

        class WithEigvals(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

            @staticmethod
            def compute_eigvals(phi, wires):
                return np.array([phi, -phi])

        op = WithEigvals(0.5, wires=0)
        assert np.allclose(op.eigvals(), [0.5, -0.5])

    def test_eigvals_inferred_from_matrix(self):
        """Test that ``eigvals`` is computed from the matrix when ``compute_eigvals`` is undefined."""

        class WithMatrix(Operator2):
            dynamic_argnames = ("theta",)

            def __init__(self, theta, wires):
                super().__init__(theta, wires=wires)

            @staticmethod
            def compute_matrix(theta, wires):
                return theta * np.eye(2)

        op = WithMatrix(0.3, wires=0)
        assert np.allclose(op.eigvals(), [0.3, 0.3])

    def test_compute_diagonalizing_gates_used_by_diagonalizing_gates(self):
        """Test that ``diagonalizing_gates`` dispatches to ``compute_diagonalizing_gates``."""

        class WithDiag(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

            @staticmethod
            def compute_diagonalizing_gates(phi, wires):
                return [DynOp(phi, wires=wires[0])]

        op = WithDiag(0.5, wires=0)
        assert op.diagonalizing_gates() == [DynOp(0.5, wires=0)]

    def test_matrix_same_wire_order_returns_canonical(self):
        """Test that ``matrix`` returns the canonical matrix when ``wire_order`` matches
        ``op.wires`` exactly (no expansion)."""

        class WithMatrix(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

            @staticmethod
            def compute_matrix(phi, wires):
                return phi * np.eye(2)

        op = WithMatrix(0.5, wires=0)
        canonical = op.matrix()
        assert np.allclose(op.matrix(wire_order=[0]), canonical)

    def test_matrix_same_wire_order_no_expand(self, mocker):
        """Test that ``expand_matrix`` is not called when ``wire_order`` equals ``op.wires``."""

        class WithMatrix(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

            @staticmethod
            def compute_matrix(phi, wires):
                return phi * np.eye(2)

        op = WithMatrix(0.5, wires=0)
        mock_expand = mocker.patch.object(qp.math, "expand_matrix")
        result = op.matrix(wire_order=[0])
        mock_expand.assert_not_called()
        assert np.allclose(result, 0.5 * np.eye(2))

    def test_sparse_matrix_with_wire_order_and_format(self):
        """Test that ``sparse_matrix`` expands for ``wire_order`` and respects ``format``."""

        class WithSparse(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

            @staticmethod
            def compute_sparse_matrix(phi, wires, format="csr"):
                return csr_matrix(phi * np.eye(2))

        op = WithSparse(0.5, wires=1)
        canonical = csr_matrix(0.5 * np.eye(2))
        mat = op.sparse_matrix(wire_order=[0, 1], format="csc")
        expected = qp.math.expand_matrix(canonical, wires=op.wires, wire_order=[0, 1]).asformat(
            "csc"
        )
        assert mat.format == "csc"
        assert np.allclose(mat.toarray(), expected.toarray())

    def test_sparse_matrix_instance_override(self):
        """Test that overriding ``sparse_matrix`` alone provides a working representation."""

        class WithSparse(Operator2):
            def sparse_matrix(self, wire_order=None, format="csr"):
                return csr_matrix(np.eye(2)).asformat(format)

            def __init__(self, wires):
                super().__init__(wires=wires)

        op = WithSparse(wires=0)
        assert np.allclose(op.sparse_matrix().toarray(), np.eye(2))

    def test_decomposition_instance_override_only(self):
        """Test that overriding only ``decomposition`` (not ``compute_decomposition``) works."""

        class WithDecomp(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

            def decomposition(self):
                return [DynOp(self.arguments["phi"], wires=self.wires[0])]

        op = WithDecomp(0.7, wires=0)
        assert WithDecomp.has_decomposition is True
        decomp = op.decomposition()
        assert decomp == [DynOp(0.7, wires=0)]

    def test_terms_returns_coefficients_and_operators(self):
        """Test that a subclass may override ``terms`` to return a linear combination."""

        class WithTerms(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

            def terms(self):
                return (
                    [self.arguments["phi"]],
                    [DynOp(self.arguments["phi"], wires=self.wires[0])],
                )

        op = WithTerms(0.5, wires=0)
        coeffs, ops = op.terms()
        assert coeffs == [0.5]
        assert ops == [DynOp(0.5, wires=0)]

    def test_generator_returns_operator(self):
        """Test that a subclass may override ``generator`` to return an operator."""

        class WithGen(Operator2):
            def __init__(self, wires):
                super().__init__(wires=wires)

            def generator(self):
                return DynOp(0.5, wires=self.wires[0])

        op = WithGen(wires=0)
        assert op.generator() == DynOp(0.5, wires=0)


class TestGraphDecomposition:
    """Tests for the graph-based decomposition fallback in ``Operator2.decomposition``:
    when ``compute_decomposition`` is not overridden, ``decomposition`` falls
    back to registered graph decomposition rules instead of immediately raising."""

    def test_compute_decomposition_takes_precedence(self):
        """An overridden ``compute_decomposition`` is used over registered graph rules."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

            @staticmethod
            def compute_decomposition(phi, wires):
                return [qp.RZ(phi, wires=wires[0])]

        with qp.decomposition.local_decomps():

            @qp.register_resources({qp.RX: 1})
            def use_rx(phi, wires, **__):
                qp.RX(phi, wires=wires[0])

            qp.add_decomps(Op, use_rx)

            decomp = Op(0.5, wires=0).decomposition()
            assert len(decomp) == 1
            assert qp.equal(decomp[0], qp.RZ(0.5, wires=0))

    def test_registered_rule_used_as_fallback(self):
        """Without an overridden ``compute_decomposition``, a registered rule is used."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        with qp.decomposition.local_decomps():

            @qp.register_resources({qp.RX: 1})
            def use_rx(phi, wires, **__):
                qp.RX(phi, wires=wires[0])

            qp.add_decomps(Op, use_rx)

            decomp = Op(0.5, wires=0).decomposition()
            assert len(decomp) == 1
            assert qp.equal(decomp[0], qp.RX(0.5, wires=0))

    def test_rule_receives_full_argument_model(self):
        """The rule is invoked with ``**op.arguments`` (dynamic, static, and wires)."""
        captured = {}

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            static_argnames = ("label",)

            def __init__(self, phi, label, wires):
                super().__init__(phi, label, wires=wires)

        with qp.decomposition.local_decomps():

            @qp.register_resources({qp.RX: 1, qp.RY: 1})
            def rule(phi, label, wires, **__):
                captured["phi"] = phi
                captured["label"] = label
                captured["wires"] = wires
                qp.RX(phi, wires=wires[0])
                qp.RY(phi, wires=wires[0])

            qp.add_decomps(Op, rule)

            decomp = Op(0.6, "spin", wires=2).decomposition()

        assert captured["phi"] == 0.6
        assert captured["label"] == "spin"
        assert captured["wires"] == Wires([2])
        assert [type(o).__name__ for o in decomp] == ["RX", "RY"]
        assert qp.equal(decomp[0], qp.RX(0.6, wires=2))
        assert qp.equal(decomp[1], qp.RY(0.6, wires=2))

    def test_no_rule_raises(self):
        """Without any registered rule, ``decomposition`` raises ``DecompositionUndefinedError``."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        op = Op(0.5, wires=0)
        with pytest.raises(DecompositionUndefinedError):
            op.decomposition()

    def test_decomposition_queued_during_recording(self):
        """Produced ops are queued when decomposition happens during active recording."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        with qp.decomposition.local_decomps():

            @qp.register_resources({qp.RX: 1, qp.RY: 1})
            def rule(phi, wires, **__):
                qp.RX(phi, wires=wires[0])
                qp.RY(phi, wires=wires[0])

            qp.add_decomps(Op, rule)

            op = Op(0.5, wires=0)
            with AnnotatedQueue() as q:
                decomp = op.decomposition()

        assert q.queue == decomp
        assert [type(o).__name__ for o in q.queue] == ["RX", "RY"]

    def test_decomposition_not_queued_outside_recording(self):
        """Outside an active recording context, decomposition returns ops without queuing."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        with qp.decomposition.local_decomps():

            @qp.register_resources({qp.RX: 1})
            def rule(phi, wires, **__):
                qp.RX(phi, wires=wires[0])

            qp.add_decomps(Op, rule)

            op = Op(0.5, wires=0)
            decomp = op.decomposition()

        assert len(decomp) == 1
        assert qp.equal(decomp[0], qp.RX(0.5, wires=0))

    def test_has_decomposition_reflects_registered_rules(self):
        """``has_decomposition`` (a class-level check) reports ``True`` once decomposition rules
        are registered for the operator type, for both class and instance access."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        # No rules registered globally: both class and instance report False.
        assert Op.has_decomposition is False
        assert Op(0.5, wires=0).has_decomposition is False

        with qp.decomposition.local_decomps():

            @qp.register_resources({qp.RX: 1})
            def rule(phi, wires, **__):
                qp.RX(phi, wires=wires[0])

            qp.add_decomps(Op, rule)

            assert Op.has_decomposition is True
            assert Op(0.5, wires=0).has_decomposition is True


class TestStatePrepBase:
    """Unit tests for StatePrepBase2"""

    def test_state_prep_base_label(self):
        """Tests that the label is as expected."""

        class MyStatePrep(StatePrepBase2):
            wire_argnames = ("wires",)

            # pylint: disable=useless-parent-delegation
            def __init__(self, wires):
                super().__init__(wires)

            def state_vector(self, wire_order=None):
                return np.zeros(5)

        op = MyStatePrep(0)
        assert op.label() == "|Ψ⟩"
        assert op.label(base_label="|x⟩") == "|x⟩"

    def test_interface_not_implemented(self):
        """Tests that an error is raised if an interface isn't implemented."""

        # pylint: disable=useless-parent-delegation,abstract-method
        class BadStatePrep(StatePrepBase2):
            wire_argnames = ("wires",)

            def __init__(self, wires):
                super().__init__(wires)

            # state_vector is not implemented!

        with pytest.raises(TypeError, match="Can't instantiate abstract class BadStatePrep"):
            BadStatePrep(0)  # pylint: disable=abstract-class-instantiated


class NoParamOp(Operator2):
    """A simple operator with wires and no dynamic parameters."""

    def __init__(self, wires):
        super().__init__(wires=wires)


class TestLegacyCompatibilityViews:
    """Tests for selected legacy ``Operator`` compatibility views on ``Operator2``."""

    def test_no_param_op_legacy_views(self):
        """Test legacy views for an operator with no dynamic parameters."""
        op = NoParamOp(wires=0)

        assert op.data == ()
        assert op.parameters == []
        assert op.hyperparameters == {}

    def test_dynamic_op_data_preserves_order(self):
        """Test that ``data`` preserves dynamic argument order."""

        class TwoParamOp(Operator2):
            dynamic_argnames = ("alpha", "beta")

            def __init__(self, alpha, beta, wires):
                super().__init__(alpha, beta, wires=wires)

        op = TwoParamOp(1.1, 2.2, wires=0)

        assert op.data == (1.1, 2.2)
        assert op.parameters == [1.1, 2.2]
        assert op.parameters == list(op.data)

    def test_static_args_excluded_from_data_and_parameters(self):
        """Test that static args are not exposed through ``data`` or ``parameters``."""
        op = FullOp(0.5, static="XY", hybrid=[], wires=0)

        assert op.data == (0.5,)
        assert op.parameters == [0.5]

    def test_hyperparameters_include_static_and_hybrid_args(self):
        """Test that static and hybrid args appear in the legacy hyperparameter view."""
        nested = [DynOp(0.1, wires=0)]
        op = FullOp(0.5, static="XY", hybrid=nested, wires=0)

        assert op.hyperparameters == {"static": "XY", "hybrid": nested}

    def test_hyperparameters_exclude_dynamic_and_wire_args(self):
        """Test that dynamic and wire args are excluded from hyperparameters."""
        op = FullOp(0.5, static="XY", hybrid=[], wires=0)

        assert "phi" not in op.hyperparameters
        assert "wires" not in op.hyperparameters

    def test_compilable_args_in_hyperparameters(self):
        """Test that compilable args appear in the legacy hyperparameter view."""
        op = CompilableOp(n=3, wires=0)
        assert op.hyperparameters == {"n": 3}

    def test_nonstandard_wire_arg_excluded_from_hyperparameters(self):
        """Test that non-``wires`` wire argument names are excluded."""

        class AuxWiresOp(Operator2):
            dynamic_argnames = ("phi",)
            wire_argnames = ("wires", "aux_wires")
            wire_sizes = (1, None)

            def __init__(self, phi, wires, aux_wires):
                super().__init__(phi, wires=wires, aux_wires=aux_wires)

        op = AuxWiresOp(0.5, wires=0, aux_wires=[1, 2])

        assert "phi" not in op.hyperparameters
        assert "wires" not in op.hyperparameters
        assert "aux_wires" not in op.hyperparameters


class TestApply:
    @pytest.mark.parametrize("op2", [DynOp(1.0, wires=0), FullOp(0.3, "lbl", [1.0, 2.0], wires=0)])
    def test_apply(self, op2):
        """Tests that Operator2 can queue like Operator1 using ``qp.apply``."""

        with AnnotatedQueue() as q:
            apply(op2)

        assert len(q.queue) == 1
        assert q.queue[0] == op2

    def test_raises_outside_queueing_context(self):
        """Tests that outside a queuing context and without capture enabled, apply() raises when given an Operator2."""

        with pytest.raises(
            RuntimeError, match="No queuing context available to append operation to"
        ):
            apply(DynOp(1.0, wires=0))
