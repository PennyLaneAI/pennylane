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

A rule registered for the operator inside an adjoint or controlled region (e.g.
``add_decomps("C(Hadamard)", ...)``) is marked with ``:region: adjoint`` or ``:region: controlled``.
Such rules are listed after the regular rules, below a divider explaining how to access them.
"""

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective

# symbolic operator name for each region, used in the fixed_decomps key, e.g. "Adjoint(Hadamard)"
REGIONS = {"adjoint": "Adjoint", "controlled": "Controlled"}


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
        # when dealing with rules registered for the operator inside an adjoint or controlled region,
        # we can specify what kind of region from the given tuple of values
        "region": lambda arg: directives.choice(arg, tuple(REGIONS)),
    }

    def _parse_paragraph(self, text):
        """Parse inline reStructuredText into a paragraph and any diagnostic messages."""
        inline_nodes, messages = self.state.inline_text(text, self.lineno)
        return nodes.paragraph("", "", *inline_nodes), messages

    def run(self):
        """Collect the circuit image, requirements, and resources for one rule."""
        rule_name = self.arguments[0]
        rule_node = decomposition_rule(name=rule_name, region=self.options.get("region"))
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


def _region_row(operator_name, region):
    """A full-width divider row explaining how to access the rules for ``region``."""
    region_key = f"{REGIONS[region]}({operator_name.rsplit('.', maxsplit=1)[-1]})"
    region_text = nodes.paragraph(
        "",
        "The following rules are for ",
        nodes.literal(region_key, region_key),
        nodes.Text(", and they can be accessed with:"),
    )
    usage_code = f'qp.decompose(gate_set=..., fixed_decomps={{"{region_key}": "<name>"}})'
    usage = nodes.literal_block(usage_code, usage_code, language="python")
    region_row = nodes.row(classes=["decomp-region-row"])
    region_row += nodes.entry("", region_text, usage, morecols=1)
    return region_row


class DecompositionTable(SphinxDirective):
    """A table of ``decomposition-rule`` entries for one operator."""

    # the only required argument is the operation name, the rest is passed as options
    required_arguments = 1
    has_content = True
    # symbol def'ns and example values for them are optional and left as text
    option_spec = {"symbols": directives.unchanged, "example": directives.unchanged}

    def run(self):
        """Build a caption and a two-column table from the nested rule directives."""
        operator_name = self.arguments[0]
        parsed_content = nodes.Element()
        self.state.nested_parse(self.content, self.content_offset, parsed_content)

        rule_nodes = []
        other_nodes = []
        for child in parsed_content.children:
            if isinstance(child, decomposition_rule):
                rule_nodes.append(child)
            else:
                other_nodes.append(child)

        caption_parts = []
        if "symbols" in self.options:
            caption_parts.append(
                "Resources are general, for " + self.options["symbols"] + ". "
                "Each circuit shows one example of the rule."
            )
        caption_parts.append(
            "Use a rule when decomposing with :func:`qp.decompose <pennylane.decompose>`:"
        )

        inline_nodes, parsing_messages = self.state.inline_text(
            " ".join(caption_parts), self.lineno
        )
        caption = nodes.paragraph("", "", *inline_nodes, classes=["decomp-table-caption"])
        usage_code = f'qp.decompose(gate_set=..., fixed_decomps={{{operator_name}: "<name>"}})'
        usage = nodes.literal_block(usage_code, usage_code, language="python")

        # with only the gate and count columns, the resources need less room, so the circuits
        # are drawn 50% bigger
        two_cols = all("decomp-cols-2" in rule["resources"]["classes"] for rule in rule_nodes)
        table = nodes.table(classes=["decomposition-table", "colwidths-given"])
        if two_cols:
            # marks when all rules in the table have two columns,
            # as opposed to decomp-cols-2 which marks that an individual rule has two
            table["classes"].append("decomp-two-cols")
        column_group = nodes.tgroup(cols=2)
        table += column_group
        for column_width in (60, 40) if two_cols else (40, 60):
            column_group += nodes.colspec(colwidth=column_width)

        # regular rules first, then each region's rules, keeping the written order within each group
        rule_nodes.sort(key=lambda rule: [None, *REGIONS].index(rule["region"]))

        table_body = nodes.tbody()
        current_region = None
        for rule_node in rule_nodes:
            if rule_node["region"] != current_region:
                current_region = rule_node["region"]
                table_body += _region_row(operator_name, current_region)

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

        return [caption, usage, table, *other_nodes, *parsing_messages]


def setup(app):
    """Sphinx entry point for this extension."""
    app.add_node(decomposition_rule)
    app.add_directive("decomposition-rule", DecompositionRule)
    app.add_directive("decomposition-table", DecompositionTable)
    app.add_css_file("decomposition_table.css")
    return {"parallel_read_safe": True, "parallel_write_safe": True}
