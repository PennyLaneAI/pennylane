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
Sphinx extension adding the ``subroutine-table`` directive: an ``autosummary`` table with an
additional thumbnail column. Each content line is an object name followed by an image path:

.. code-block:: rst

    .. subroutine-table::

        ~pennylane.QFT  /_static/templates/subroutines/qft.svg
"""

from docutils import nodes
from sphinx.ext.autosummary import Autosummary


class SubroutineTable(Autosummary):
    """An ``autosummary`` table with a thumbnail image in front of each entry."""

    def get_table(self, items):
        # Autosummary only reads the first token of each content line, so the image paths
        # following the names are only used here
        images = {}
        for line in self.content:
            parts = line.split()
            if len(parts) == 2:
                images[parts[0].lstrip("~")] = parts[1]

        table_spec, table = super().get_table(items)
        table_spec["spec"] = r"\X{1}{5}\X{1}{5}\X{3}{5}"

        real_table = table[0]
        real_table["classes"].append("subroutine-table")
        group = real_table[0]
        group["cols"] = 3
        group.insert(0, nodes.colspec("", colwidth=0))
        # relative widths of the thumbnail, name, and summary columns
        for colspec, width in zip(group.findall(nodes.colspec), (33, 35, 32)):
            colspec["colwidth"] = width

        # get_table creates one row per item, in order
        for (_, _, _, real_name), row in zip(items, group[-1].children):
            entry = nodes.entry("", classes=["subroutine-thumbnail"])
            if real_name in images:
                entry += nodes.image(
                    uri=images[real_name], alt=real_name.rsplit(".", maxsplit=1)[-1]
                )
            row.insert(0, entry)

        return [table_spec, table]


def setup(app):
    """Sphinx entry point for this extension."""
    app.setup_extension("sphinx.ext.autosummary")
    app.add_directive("subroutine-table", SubroutineTable)
    app.add_css_file("subroutine_table.css")
    return {"parallel_read_safe": True, "parallel_write_safe": True}
