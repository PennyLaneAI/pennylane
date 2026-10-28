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
"""Add alias links while Sphinx renders docstrings, without changing ``__doc__``."""

import pennylane as qp


def add_public_aliases(app, what, name, obj, options, lines):
    """Edit Sphinx's reStructuredText ``lines`` in place; no return value is needed."""
    # Only document PennyLane classes and functions.
    if what not in {"class", "function"}:
        return
    if not getattr(obj, "__module__", "").startswith("pennylane."):
        return

    # The documented name can differ from obj.__name__ on alias pages.
    documented_name = name.rsplit(".", 1)[-1]
    # Inspect loaded exports without triggering lazy imports; "is" identifies aliases.
    aliases = sorted(
        export_name
        for export_name, exported_obj in vars(qp).items()
        if not export_name.startswith("_")
        and export_name != documented_name
        and exported_obj is obj
    )
    if not aliases:
        return

    # Sphinx roles link to the full target; "~" displays only its short name.
    role = "class" if what == "class" else "func"
    references = ", ".join(f":{role}:`~pennylane.{alias}`" for alias in aliases)
    # Preserve the opening paragraph, which Sphinx also uses in summary tables.
    insertion_index = 0
    while insertion_index < len(lines) and not lines[insertion_index].strip():
        insertion_index += 1
    while insertion_index < len(lines) and lines[insertion_index].strip():
        insertion_index += 1
    # reStructuredText notes require blank lines and indented content.
    lines[insertion_index:insertion_index] = [
        "",
        ".. note::",
        "",
        f"   Also available as {references}.",
        "",
    ]


def setup(app):
    """Entry point called by Sphinx when this extension is loaded."""
    # Autodoc must register the event before we connect to it.
    app.setup_extension("sphinx.ext.autodoc")
    app.connect("autodoc-process-docstring", add_public_aliases)
    # No shared state is modified, so parallel builds are safe.
    return {"parallel_read_safe": True, "parallel_write_safe": True}
