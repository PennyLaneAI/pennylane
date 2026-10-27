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

    # the only required argument is the decomp rule name, the rest is passed as options
    required_arguments = 1
    has_content = True
    option_spec = {
        # a decomp circuit image is required, and is left as an image
        "image": directives.unchanged_required,
        # we are not required to specify anything, such as work wires, required by the circuit
        # if we do specify something as text, it is left as text
        "requires": directives.unchanged,
        # we are not specified to elaborate on the decomp with mathematical definitions,
        # but if we do, the text remains text
        "where": directives.unchanged,
    }

    def _parse_paragraph(self, text):
        """Parse inline reStructuredText into a paragraph and any diagnostic messages."""
        inline_nodes, messages = self.state.inline_text(text, self.lineno)
        return nodes.paragraph("", "", *inline_nodes), messages

    def run(self):
        """Collect the circuit image, requirements, and resources for one rule."""
        rule_name = self.arguments[0]
        rule_node = decomposition_rule(name=rule_name)
        rule_node["image"] = nodes.image(uri=self.options["image"], alt=rule_name)

        parsing_messages = []
        requirements = None
        if "requires" in self.options:
            requirements, messages = self._parse_paragraph("Requires " + self.options["requires"])
            requirements["classes"].append("decomp-requires")
            parsing_messages.extend(messages)
        rule_node["requires"] = requirements

        resource_rows = [
            [cell.strip() for cell in line.split(" | ")] for line in self.content if line.strip()
        ]
        column_titles = ["Gate", "Count", "Example"][: max(len(row) for row in resource_rows)]
        column_classes = ["decomp-gate", "decomp-count", "decomp-example-count"]
        column_count = len(column_titles)
        resources = nodes.container(classes=["decomp-resources", f"decomp-cols-{column_count}"])
        for title in column_titles:
            resources += nodes.paragraph(text=title, classes=["decomp-resources-head"])
        for resource_row in resource_rows:
            padded_cells = resource_row + [""] * (column_count - len(resource_row))
            for cell_text, column_class in zip(padded_cells, column_classes):
                paragraph, messages = self._parse_paragraph(cell_text)
                paragraph["classes"].append(column_class)
                resources += paragraph
                parsing_messages.extend(messages)
        if "where" in self.options:
            explanation, messages = self._parse_paragraph("where " + self.options["where"])
            explanation["classes"].append("decomp-where")
            resources += explanation
            parsing_messages.extend(messages)
        rule_node["resources"] = resources

        return [rule_node, *parsing_messages]


class DecompositionTable(SphinxDirective):
    """A table of ``decomposition-rule`` entries for one operator."""

    required_arguments = 1
    has_content = True
    option_spec = {"symbols": directives.unchanged, "example": directives.unchanged}

    def run(self):
        """Build a caption and a two-column table from the nested rule directives."""
        operator_name = self.arguments[0]
        parsed_content = nodes.Element()
        self.state.nested_parse(self.content, self.content_offset, parsed_content)
        rule_nodes = [
            child for child in parsed_content.children if isinstance(child, decomposition_rule)
        ]
        other_nodes = [
            child for child in parsed_content.children if not isinstance(child, decomposition_rule)
        ]

        caption_parts = []
        if "symbols" in self.options:
            caption_parts.append(
                "Resources are general, for " + self.options["symbols"] + ". "
                "Each circuit shows one example of the rule."
            )
        # Escape "<" so reStructuredText does not interpret it as a link target.
        rule_lookup = f'qp.list_decomps({operator_name})["\\<name>"]'
        caption_parts.append(
            "Use a rule when decomposing with "
            f":func:`qp.decompose(gate_set=..., fixed_decomps={{{operator_name}: {rule_lookup}}}) "
            "<pennylane.decompose>`."
        )

        inline_nodes, parsing_messages = self.state.inline_text(" ".join(caption_parts), self.lineno)
        caption = nodes.paragraph("", "", *inline_nodes, classes=["decomp-table-caption"])

        # with only the gate and count columns, the resources need less room, so the circuits
        # are drawn 50% bigger
        two_cols = all("decomp-cols-2" in rule["resources"]["classes"] for rule in rules)
        table = nodes.table(classes=["decomposition-table", "colwidths-given"])
        if two_cols:
            table["classes"].append("decomp-two-cols")
        column_group = nodes.tgroup(cols=2)
        table += column_group
        for column_width in (60, 40) if two_cols else (40, 60):
            column_group += nodes.colspec(colwidth=column_width)

        table_body = nodes.tbody()
        for rule_node in rule_nodes:
            rule_name = nodes.literal(
                rule_node["name"], rule_node["name"], classes=["decomp-access-name"]
            )
            name_row = nodes.row(classes=["decomp-access-row"])
            name_row += nodes.entry("", nodes.paragraph("", "", rule_name), morecols=1)
            table_body += name_row

            detail_row = nodes.row(classes=["decomp-body-row"])
            circuit_cell = nodes.entry("", rule_node["image"], classes=["decomp-circuit"])
            if "example" in self.options:
                inline_nodes, messages = self.state.inline_text(
                    "Example: " + self.options["example"], self.lineno
                )
                circuit_cell += nodes.paragraph("", "", *inline_nodes, classes=["decomp-example"])
                parsing_messages.extend(messages)
            detail_row += circuit_cell
            resources_cell = nodes.entry("")
            if rule_node["requires"] is not None:
                resources_cell += rule_node["requires"]
            resources_cell += rule_node["resources"]
            detail_row += resources_cell
            table_body += detail_row
        column_group += table_body

        return [caption, table, *other_nodes, *parsing_messages]


def setup(app):
    """Sphinx entry point for this extension."""
    app.add_node(decomposition_rule)
    app.add_directive("decomposition-rule", DecompositionRule)
    app.add_directive("decomposition-table", DecompositionTable)
    app.add_css_file("decomposition_table.css")
    return {"parallel_read_safe": True, "parallel_write_safe": True}
