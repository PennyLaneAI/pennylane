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
rules of an operator. Each rule is headed by its name, followed by an example circuit diagram and
its general resources.

.. code-block:: rst

    .. decomposition-table:: qp.Select
        :symbols: :math:`K` operators, :math:`c` control wires
        :example: :math:`K=4`, :math:`c=2`

        .. decomposition-rule:: _select_decomp_unary
            :image: /_static/decomposition_thumbnails/select_unary.svg
            :requires: :math:`c - 1` work wires
            :where: :math:`N = c + K - 2`

            :class:`~.TemporaryAND` | :math:`N` | 1
            :class:`~.CNOT` | :math:`K - 1` | 3

Each content line of a rule is a gate, its count, and optionally its count in the example circuit,
all inline reStructuredText, separated by `` | ``.
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

        rows = [
            [part.strip() for part in line.split(" | ")] for line in self.content if line.strip()
        ]
        # an optional third column gives the counts for the example circuit
        titles = ["Gate", "Count", "Example"][: max(len(row) for row in rows)]
        classes = ["decomp-gate", "decomp-count", "decomp-example-count"]
        resources = nodes.container(classes=["decomp-resources", f"decomp-cols-{len(titles)}"])
        for title in titles:
            resources += nodes.paragraph(text=title, classes=["decomp-resources-head"])
        for row in rows:
            for cell, cls in zip(row + [""] * (len(titles) - len(row)), classes):
                par, msgs = self._inline(cell)
                par["classes"].append(cls)
                resources += par
                messages += msgs
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
    option_spec = {"symbols": directives.unchanged, "example": directives.unchanged}

    def run(self):
        op = self.arguments[0]
        content = nodes.Element()
        self.state.nested_parse(self.content, self.content_offset, content)
        rules = [child for child in content.children if isinstance(child, decomposition_rule)]
        others = [child for child in content.children if not isinstance(child, decomposition_rule)]

        # the "<" is escaped, so that it is not read as the start of the link target
        access = f'qp.list_decomps({op})["\\<name>"]'
        intro = []
        if "symbols" in self.options:
            intro.append(
                "Resources are general, for " + self.options["symbols"] + ". "
                "Each circuit shows one example of the rule."
            )
        intro.append(
            f'Use a rule when decomposing with :func:`qp.decompose(gate_set=..., fixed_decomps={{{op}: qp.list_decomps({op})["\\<name>"]}}) '
            f"<pennylane.decompose>`."
        )

        children, messages = self.state.inline_text(" ".join(intro), self.lineno)
        caption = nodes.paragraph("", "", *children, classes=["decomp-table-caption"])

        # with only the gate and count columns, the resources need less room, so the circuits
        # are drawn 50% bigger
        two_cols = all("decomp-cols-2" in rule["resources"]["classes"] for rule in rules)
        table = nodes.table(classes=["decomposition-table", "colwidths-given"])
        if two_cols:
            table["classes"].append("decomp-two-cols")
        group = nodes.tgroup(cols=2)
        table += group
        for width in (60, 40) if two_cols else (40, 60):
            group += nodes.colspec(colwidth=width)

        body = nodes.tbody()
        for rule in rules:
            # the name of the rule, spanning both columns
            access = nodes.literal(rule["name"], rule["name"], classes=["decomp-access-name"])
            access_row = nodes.row(classes=["decomp-access-row"])
            access_row += nodes.entry("", nodes.paragraph("", "", access), morecols=1)
            body += access_row

            row = nodes.row(classes=["decomp-body-row"])
            circuit = nodes.entry("", rule["image"], classes=["decomp-circuit"])
            if "example" in self.options:
                children, msgs = self.state.inline_text(
                    "Example: " + self.options["example"], self.lineno
                )
                circuit += nodes.paragraph("", "", *children, classes=["decomp-example"])
                messages += msgs
            row += circuit
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
