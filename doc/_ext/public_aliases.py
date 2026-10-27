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
"""Add links to public aliases when Sphinx documents PennyLane objects."""

import pennylane as qp


def add_public_aliases(app, what, name, obj, options, lines):
    """Append links to other top-level exports of the documented object."""
    if what not in {"class", "function"}:
        return
    if not getattr(obj, "__module__", "").startswith("pennylane."):
        return

    documented_name = name.rsplit(".", 1)[-1]
    aliases = sorted(
        export_name
        for export_name, exported_obj in vars(qp).items()
        if not export_name.startswith("_")
        and export_name != documented_name
        and exported_obj is obj
    )
    if not aliases:
        return

    role = "class" if what == "class" else "func"
    references = ", ".join(f":{role}:`~pennylane.{alias}`" for alias in aliases)
    lines.extend(["", ".. note::", "", f"   Also available as {references}.", ""])


def setup(app):
    """Register the docstring hook."""
    app.setup_extension("sphinx.ext.autodoc")
    app.connect("autodoc-process-docstring", add_public_aliases)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
