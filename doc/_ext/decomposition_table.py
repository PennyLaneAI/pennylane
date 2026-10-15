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
Sphinx extension adding the ``decomposition-table`` directive: a table listing the decomposition
rules of an operator. Each rule is headed by the expression accessing it by name, followed by its
circuit diagram and its resources.

.. code-block:: rst

    .. decomposition-table:: qp.Select
        :symbols: :math:`K` operators, :math:`c` control wires

        .. decomposition-rule:: _select_decomp_unary
            :image: /_static/decomposition_thumbnails/select_unary.svg
            :requires: :math:`c - 1` work wires
            :where: :math:`N = c + K - 2`

            :class:`~.TemporaryAND` | :math:`N`
            :class:`~.CNOT` | :math:`K - 1`

Each content line of a rule is a gate and its count, both inline reStructuredText, separated by
`` | ``.
"""

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective


class decomposition_rule(nodes.General, nodes.Element):
    """Placeholder for one rule, turned into a table row by ``DecompositionTable``."""


class DecompositionRule(SphinxDirective):
    """One row of a ``decomposition-table``."""

    required_arguments = 1
    has_content = True
    option_spec = {
        "image": directives.unchanged_required,
        "requires": directives.unchanged,
        "where": directives.unchanged,
    }

    def _inline(self, text):
        children, messages = self.state.inline_text(text, self.lineno)
        return nodes.paragraph("", "", *children), messages

    def run(self):
        name = self.arguments[0]
        node = decomposition_rule(name=name)

        image = nodes.image(uri=self.options["image"], alt=name)
        node["image"] = image

        messages = []
        requires = None
        if "requires" in self.options:
            requires, msgs = self._inline("Requires " + self.options["requires"])
            requires["classes"].append("decomp-requires")
            messages += msgs
        node["requires"] = requires

        resources = nodes.container(classes=["decomp-resources"])
        for line in self.content:
            if not line.strip():
                continue
            gate, count = (part.strip() for part in line.split(" | ", maxsplit=1))
            gate_par, msgs = self._inline(gate)
            gate_par["classes"].append("decomp-gate")
            messages += msgs
            count_par, msgs = self._inline(count)
            count_par["classes"].append("decomp-count")
            messages += msgs
            resources += [gate_par, count_par]
        if "where" in self.options:
            par, msgs = self._inline("where " + self.options["where"])
            par["classes"].append("decomp-where")
            resources += par
            messages += msgs
        node["resources"] = resources

        return [node, *messages]


class DecompositionTable(SphinxDirective):
    """A table of ``decomposition-rule`` entries for one operator."""

    required_arguments = 1
    has_content = True
    option_spec = {"symbols": directives.unchanged}

    def run(self):
        op = self.arguments[0]
        content = nodes.Element()
        self.state.nested_parse(self.content, self.content_offset, content)
        rules = [child for child in content.children if isinstance(child, decomposition_rule)]
        others = [child for child in content.children if not isinstance(child, decomposition_rule)]

        name = op.rsplit(".", maxsplit=1)[-1]
        # the "<" is escaped, so that it is not read as the start of the link target
        access = f'qp.list_decomps({op})["\\<name>"]'
        intro = [
            f"The following decomposition rules for ``{name}`` can be accessed directly via "
            f":func:`{access} <pennylane.list_decomps>`."
        ]
        if "symbols" in self.options:
            intro.append("Resources are given for " + self.options["symbols"] + ".")
        intro.append(
            f"Use :func:`qp.decompose(gate_set=..., fixed_decomps={{{op}: {access}}}) "
            "<pennylane.decompose>` to specifically access a given rule when decomposing."
        )

        # inline_text only parses inline markup, so the list itself is built from nodes
        caption = nodes.bullet_list(classes=["decomp-table-caption"])
        messages = []
        for sentence in intro:
            children, msgs = self.state.inline_text(sentence, self.lineno)
            caption += nodes.list_item("", nodes.paragraph("", "", *children))
            messages += msgs

        table = nodes.table(classes=["decomposition-table", "colwidths-given"])
        group = nodes.tgroup(cols=2)
        table += group
        for width in (45, 55):
            group += nodes.colspec(colwidth=width)

        head = nodes.thead()
        head_row = nodes.row()
        for title in ("Decomposition rule", "Resources"):
            head_row += nodes.entry("", nodes.paragraph(text=title))
        head += head_row
        group += head

        body = nodes.tbody()
        for rule in rules:
            # the full expression accessing the rule, spanning both columns
            access = nodes.literal(
                "",
                "",
                nodes.inline(text=f"name=", classes=["decomp-access-op"]),
                nodes.inline(text=f'"{rule["name"]}"', classes=["decomp-access-name"]),
            )
            access_row = nodes.row(classes=["decomp-access-row"])
            access_row += nodes.entry("", nodes.paragraph("", "", access), morecols=1)
            body += access_row

            row = nodes.row(classes=["decomp-body-row"])
            row += nodes.entry("", rule["image"], classes=["decomp-circuit"])
            resources = nodes.entry("")
            if rule["requires"] is not None:
                resources += rule["requires"]
            resources += rule["resources"]
            row += resources
            body += row
        group += body

        return [caption, table, *others, *messages]


def setup(app):
    """Sphinx entry point for this extension."""
    app.add_node(decomposition_rule)
    app.add_directive("decomposition-rule", DecompositionRule)
    app.add_directive("decomposition-table", DecompositionTable)
    app.add_css_file("decomposition_table.css")
    return {"parallel_read_safe": True, "parallel_write_safe": True}
