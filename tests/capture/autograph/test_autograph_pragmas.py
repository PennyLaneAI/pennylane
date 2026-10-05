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

"""PyTests for the AutoGraph source comment pragmas."""

# pylint: disable=wrong-import-order, wrong-import-position, ungrouped-imports

import ast
import warnings
from types import SimpleNamespace

import pytest
from malt.lang import directives
from malt.pyct import anno, origin_info

import pennylane as qp
from pennylane.capture.autograph.pragmas import GeneratedEdit, apply_edits, transform
from pennylane.exceptions import AutoGraphError, AutoGraphWarning

pytestmark = [pytest.mark.capture]

jax = pytest.importorskip("jax")

# must be below jax importorskip
from jax._src.core import ClosedJaxpr, Jaxpr
from jax.core import eval_jaxpr

from pennylane.capture.autograph.transformer import autograph_source, run_autograph


def run_pragmas(source):
    """Run the pragma pass over a snippet of source code, as AutoGraph itself would."""
    node = ast.parse(source)
    for statement in node.body:
        origin_info.resolve(statement, source, "<test>", statement.lineno, statement.col_offset)
    context = SimpleNamespace(info=SimpleNamespace(source_code=source))
    return transform(node, context)


def apply_pragmas(source):
    """Run the pragma pass over a snippet of source code and return the resulting tree."""
    return run_pragmas(source)[0]


def pragma_edits(source):
    """Run the pragma pass over a snippet of source code and return its deferred edits."""
    return run_pragmas(source)[1]


def edit_values(edit):
    """Read back the values an edit will pass to the generated call."""
    return {key: ast.literal_eval(value) for key, value in edit.values.items()}


def loop_options(node):
    """Read back the loop options a pragma recorded on a statement."""
    annotation = anno.getanno(node, anno.Basic.DIRECTIVES, {})
    options = annotation.get(directives.set_loop_options, {})
    return {key: ast.literal_eval(value) for key, value in options.items()}


def estimated_iterations(jaxpr):
    """Collect the iteration estimate of every loop in a jaxpr, including nested ones."""
    estimates = []

    for eqn in jaxpr.eqns:
        if "estimated_iterations" in eqn.params:
            estimates.append(eqn.params["estimated_iterations"])

        for value in eqn.params.values():
            values = value if isinstance(value, (list, tuple)) else [value]
            for item in values:
                if isinstance(item, Jaxpr):
                    estimates.extend(estimated_iterations(item))
                elif isinstance(item, ClosedJaxpr):
                    estimates.extend(estimated_iterations(item.jaxpr))

    return estimates


def trace(fn, *args):
    """Convert a function with AutoGraph and collect the iteration estimates of its loops."""
    return estimated_iterations(jax.make_jaxpr(run_autograph(fn))(*args).jaxpr)


class TestPragmaScanning:
    """Test which comments are recognized as pragmas and which statement they attach to."""

    def test_trailing_comment(self):
        """Test that a pragma at the end of a statement annotates that statement."""

        tree = apply_pragmas("for i in range(3):  # qphint: num-iters=4\n    x = i\n")
        assert loop_options(tree.body[0]) == {"num-iters": 4}

    def test_leading_comment(self):
        """Test that a pragma on its own line annotates the statement below it."""

        tree = apply_pragmas("# qphint: num-iters=4\nwhile x < 3:\n    x = x + 1\n")
        assert loop_options(tree.body[0]) == {"num-iters": 4}

    def test_several_leading_comments(self):
        """Test that a block of pragmas above a statement all annotate it."""

        source = "# qphint: num-iters=4\n# qphint: other=2\nfor i in range(3):\n    x = i\n"
        tree = apply_pragmas(source)
        assert loop_options(tree.body[0]) == {"num-iters": 4, "other": 2}

    def test_blank_lines_between_comment_and_statement(self):
        """Test that a pragma still attaches across blank lines and ordinary comments."""

        source = "# qphint: num-iters=4\n\n# just a comment\nfor i in range(3):\n    x = i\n"
        tree = apply_pragmas(source)
        assert loop_options(tree.body[0]) == {"num-iters": 4}

    @pytest.mark.parametrize(
        "comment",
        ["# TODO: num-iters=4", "# pylint: disable=no-member", "# type: ignore", "# qphint"],
    )
    def test_ordinary_comments_ignored(self, comment):
        """Test that comments which are not pragmas are left alone."""

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tree = apply_pragmas(f"for i in range(3):  {comment}\n    x = i\n")

        assert loop_options(tree.body[0]) == {}

    def test_one_line_compound_statement(self):
        """Test that the outermost statement on a line claims the pragma."""

        tree = apply_pragmas("for i in range(3): x = i  # qphint: num-iters=4\n")
        loop = tree.body[0]
        assert loop_options(loop) == {"num-iters": 4}
        assert loop_options(loop.body[0]) == {}

    def test_nested_function(self):
        """Test that a pragma inside a nested function annotates the nested statement."""

        source = "def outer():\n    def inner():\n        for i in range(3):  # qphint: num-iters=4\n            x = i\n"
        tree = apply_pragmas(source)
        loop = tree.body[0].body[0].body[0]
        assert loop_options(loop) == {"num-iters": 4}

    def test_pragma_without_a_statement_warns(self):
        """Test that a pragma on a line that starts no statement is reported."""

        source = "x = min(\n    1,  # qphint: num-iters=4\n    2,\n)\n"
        with pytest.warns(AutoGraphWarning, match="is not attached to a statement"):
            apply_pragmas(source)

    def test_pragma_at_end_of_source_warns(self):
        """Test that a trailing pragma with no statement below it is reported."""

        with pytest.warns(AutoGraphWarning, match="is not attached to a statement"):
            apply_pragmas("x = 1\n# qphint: num-iters=4\n")

    def test_unknown_pragma_warns(self):
        """Test that an unrecognized pragma name is reported rather than silently dropped."""

        with pytest.warns(AutoGraphWarning, match="Unknown AutoGraph pragma 'qpnope'"):
            tree = apply_pragmas("for i in range(3):  # qpnope: num-iters=4\n    x = i\n")

        assert loop_options(tree.body[0]) == {}


class TestQphintParsing:
    """Test how the body of a qphint pragma is parsed."""

    def test_several_pairs(self):
        """Test that comma separated pairs are all recorded."""

        tree = apply_pragmas("for i in range(3):  # qphint: a=1, b='two'\n    x = i\n")
        assert loop_options(tree.body[0]) == {"a": 1, "b": "two"}

    def test_commas_inside_a_value(self):
        """Test that a comma inside brackets does not split a pair."""

        tree = apply_pragmas("for i in range(3):  # qphint: a=[1, 2], b={'c': 3}\n    x = i\n")
        assert loop_options(tree.body[0]) == {"a": [1, 2], "b": {"c": 3}}

    def test_extra_whitespace(self):
        """Test that whitespace around the pragma and its pairs is ignored."""

        tree = apply_pragmas("for i in range(3):  #   qphint :  a = 1 ,  b = 2 \n    x = i\n")
        assert loop_options(tree.body[0]) == {"a": 1, "b": 2}

    def test_annotating_an_unsupported_statement(self):
        """Test that a qphint on a statement that cannot be annotated is an error."""

        with pytest.raises(AutoGraphError, match="can only annotate a 'for', 'while' or 'if'"):
            apply_pragmas("x = 1  # qphint: num-iters=4\n")

    def test_empty_body(self):
        """Test that a qphint without any pairs is an error."""

        with pytest.raises(AutoGraphError, match="is empty"):
            apply_pragmas("for i in range(3):  # qphint:\n    x = i\n")

    @pytest.mark.parametrize("body", ["num-iters", "=4", "num-iters 4"])
    def test_malformed_pair(self, body):
        """Test that a pair which is not 'key=value' is an error."""

        with pytest.raises(AutoGraphError, match="expected comma separated 'key=value' pairs"):
            apply_pragmas(f"for i in range(3):  # qphint: {body}\n    x = i\n")

    def test_non_literal_value(self):
        """Test that a value which is not a Python literal is an error."""

        with pytest.raises(AutoGraphError, match="must be a Python literal"):
            apply_pragmas("for i in range(3):  # qphint: num-iters=n\n    x = i\n")

    def test_repeated_key_in_one_pragma(self):
        """Test that setting the same key twice in one pragma is an error."""

        with pytest.raises(AutoGraphError, match="sets 'a' more than once"):
            apply_pragmas("for i in range(3):  # qphint: a=1, a=2\n    x = i\n")

    def test_repeated_key_across_pragmas(self):
        """Test that two pragmas on the same loop cannot set the same key."""

        source = "# qphint: a=1\n# qphint: a=2\nfor i in range(3):\n    x = i\n"
        with pytest.raises(AutoGraphError, match="already has it set"):
            apply_pragmas(source)

    def test_conflict_with_set_loop_options(self):
        """Test that a loop cannot carry both a pragma and a set_loop_options call.

        AutoGraph's own directives converter runs later and replaces the whole set of loop
        options, so the two cannot be combined.
        """

        source = (
            "for i in range(3):  # qphint: num-iters=4\n"
            "    malt.experimental.set_loop_options(maximum_iterations=3)\n"
            "    x = i\n"
        )
        with pytest.raises(AutoGraphError, match="also calls 'set_loop_options'"):
            apply_pragmas(source)


class TestQphintOnIfStatements:
    """Test annotating the branches of an if statement.

    AutoGraph offers no annotation for if statements the way it does for loops, so the hints
    are deferred and added to the generated ``if_stmt`` call instead.
    """

    def test_true_branch(self):
        """Test that a pragma on the if line annotates the true branch."""

        (edit,) = pragma_edits("if x > 5:  # qphint: a=1\n    y = 1\n")
        assert (edit.call, edit.keyword, edit.lineno) == ("if_stmt", "true_hints", 1)
        assert edit_values(edit) == {"a": 1}

    def test_false_branch(self):
        """Test that a pragma on the else line annotates the false branch."""

        source = "if x > 5:\n    y = 1\nelse:  # qphint: a=1\n    y = 2\n"
        (edit,) = pragma_edits(source)
        assert (edit.call, edit.keyword, edit.lineno) == ("if_stmt", "false_hints", 1)
        assert edit_values(edit) == {"a": 1}

    def test_both_branches(self):
        """Test that the two branches of one if statement are annotated separately."""

        source = "if x > 5:  # qphint: a=1\n    y = 1\nelse:  # qphint: a=2\n    y = 2\n"
        edits = {edit.keyword: edit_values(edit) for edit in pragma_edits(source)}
        assert edits == {"true_hints": {"a": 1}, "false_hints": {"a": 2}}

    def test_else_branch_with_a_multiline_body(self):
        """Test that the else line is found when the true branch spans several lines."""

        source = "if x > 5:\n    y = 1\n    for i in range(3):\n        y = y + i\nelse:  # qphint: a=1\n    y = 2\n"
        (edit,) = pragma_edits(source)
        assert (edit.keyword, edit_values(edit)) == ("false_hints", {"a": 1})

    def test_elif_branch(self):
        """Test that an elif is annotated as the true branch of its own if statement."""

        source = "if x > 5:\n    y = 1\nelif x > 3:  # qphint: a=1\n    y = 2\n"
        (edit,) = pragma_edits(source)
        assert (edit.keyword, edit.lineno, edit_values(edit)) == ("true_hints", 3, {"a": 1})

    def test_nested_if_statements(self):
        """Test that pragmas on nested if statements are kept apart."""

        source = "if x > 5:  # qphint: a=1\n    if x > 7:  # qphint: a=2\n        y = 1\n"
        edits = {edit.lineno: edit_values(edit) for edit in pragma_edits(source)}
        assert edits == {1: {"a": 1}, 2: {"a": 2}}

    def test_several_pragmas_on_one_branch_merge(self):
        """Test that a block of pragmas above an if statement contributes to one edit."""

        source = "# qphint: a=1\n# qphint: b=2\nif x > 5:\n    y = 1\n"
        (edit,) = pragma_edits(source)
        assert edit_values(edit) == {"a": 1, "b": 2}

    def test_repeated_key_on_one_branch(self):
        """Test that two pragmas cannot set the same key for the same branch."""

        source = "# qphint: a=1\n# qphint: a=2\nif x > 5:\n    y = 1\n"
        with pytest.raises(AutoGraphError, match="already has it set"):
            pragma_edits(source)

    def test_annotating_the_else_clause_of_a_loop(self):
        """Test that the else clause of a loop cannot be annotated, as it is not converted."""

        source = "for i in range(3):  # qphint: a=1\n    y = 1\nelse:  # qphint: a=2\n    y = 2\n"
        with pytest.raises(AutoGraphError, match="annotates the 'else' clause of a loop"):
            pragma_edits(source)

    def test_statement_without_a_source_location(self):
        """Test that a statement carrying no origin information is reported."""

        source = "if x > 5:  # qphint: a=1\n    y = 1\n"
        # Parsed without resolving origins, unlike run_pragmas.
        node = ast.parse(source)
        context = SimpleNamespace(info=SimpleNamespace(source_code=source))
        with pytest.raises(AutoGraphError, match="no source location"):
            transform(node, context)

    def test_edit_that_matches_no_generated_call(self):
        """Test that an edit which finds no call to attach to is reported."""

        node = ast.parse("x = 1\n")
        edit = GeneratedEdit(1, "if_stmt", "true_hints", {"a": ast.Constant(1)})
        with pytest.warns(AutoGraphWarning, match="could not be applied"):
            apply_edits(node, [edit])

    def test_generated_call(self):
        """Test that the hints reach the if_stmt call in the converted source."""

        def f(x):
            y = 0
            if x > 5:  # qphint: a=1
                y = 1
            else:  # qphint: a=2
                y = 2
            return y

        source = autograph_source(run_autograph(f))
        assert "true_hints={'a': 1}" in source
        assert "false_hints={'a': 2}" in source

    def test_conversion_is_unaffected(self):
        """Test that annotating a branch does not change what the function computes."""

        def f(x):
            y = 0
            if x > 5:  # qphint: a=1
                y = 1
            else:  # qphint: a=2
                y = 2
            return y

        jaxpr = jax.make_jaxpr(run_autograph(f))(0)
        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 8)[0] == 1
        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2)[0] == 2


class TestQphintIntegration:
    """Test that a qphint pragma reaches the loop AutoGraph generates."""

    def test_for_loop(self):
        """Test that a pragma on a Python for loop sets the iteration estimate."""

        def f(n):
            x = 0
            for i in range(n):  # qphint: num-iters=10
                x = x + i
            return x

        assert trace(f, 3) == [10]

    def test_while_loop(self):
        """Test that a pragma on a Python while loop sets the iteration estimate."""

        def f(n):
            i = 0
            # qphint: num-iters=7
            while i < n:
                i = i + 1
            return i

        assert trace(f, 3) == [7]

    def test_while_loop_with_a_break(self):
        """Test that a pragma survives the rewrite AutoGraph applies to a loop with a break."""

        def f(n):
            i = 0
            # qphint: num-iters=10
            while i < n:
                i = i + 1
                if i > 2:
                    break
            return i

        assert trace(f, 5) == [10]

    def test_loop_without_a_pragma(self):
        """Test that a loop with no pragma is left without an estimate."""

        def f(n):
            x = 0
            for i in range(n):
                x = x + i
            return x

        assert trace(f, 3) == [None]

    def test_only_the_annotated_loop_is_hinted(self):
        """Test that a pragma applies to its own loop and not to its neighbours."""

        def f(n):
            x = 0
            for i in range(n):  # qphint: num-iters=10
                x = x + i
            for j in range(n):
                x = x + j
            return x

        assert trace(f, 3) == [10, None]

    def test_loop_over_an_array(self):
        """Test that a pragma works on a loop over an iterable rather than a range."""

        def f(array):
            x = 0
            for value in array:  # qphint: num-iters=10
                x = x + value
            return x

        assert trace(f, jax.numpy.array([1, 2, 3])) == [10]

    def test_loop_over_an_enumeration(self):
        """Test that a pragma works on a loop over an enumeration."""

        def f(array):
            x = 0
            for i, value in enumerate(array):  # qphint: num-iters=10
                x = x + i * value
            return x

        assert trace(f, jax.numpy.array([1, 2, 3])) == [10]

    def test_nested_loops(self):
        """Test that pragmas on nested loops are kept apart."""

        def f(n):
            x = 0
            for i in range(n):  # qphint: num-iters=10
                for j in range(n):  # qphint: num-iters=5
                    x = x + i * j
            return x

        assert sorted(estimate for estimate in trace(f, 3)) == [5, 10]

    def test_nested_function(self):
        """Test that a pragma inside a nested function is applied."""

        def f(n):
            def inner(m):
                x = 0
                for i in range(m):  # qphint: num-iters=12
                    x = x + i
                return x

            return inner(n)

        assert trace(f, 3) == [12]

    def test_unrecognized_hint_key_is_dropped(self):
        """Test that a hint PennyLane does not know about is ignored at runtime."""

        def f(n):
            x = 0
            for i in range(n):  # qphint: num-iters=10, unsupported=True
                x = x + i
            return x

        assert trace(f, 3) == [10]

    def test_misspelled_hint_key(self):
        """Test that a close misspelling is still understood, as it is for qp.hint."""

        def f(n):
            x = 0
            for i in range(n):  # qphint: num_iters=10
                x = x + i
            return x

        assert trace(f, 3) == [10]

    def test_qnode(self):
        """Test that a pragma inside a QNode reaches the generated loop."""

        @qp.qnode(qp.device("default.qubit", wires=1))
        def circuit(n):
            for _ in range(n):  # qphint: num-iters=10
                qp.X(0)
            return qp.expval(qp.Z(0))

        assert trace(circuit, 3) == [10]
