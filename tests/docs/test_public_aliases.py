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
"""Tests for the Sphinx extension that discovers public Python aliases."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, call

import pytest

import pennylane as qp

EXTENSION_PATH = Path(__file__).resolve().parents[2] / "doc/_ext/public_aliases.py"
spec = importlib.util.spec_from_file_location("public_aliases", EXTENSION_PATH)
public_aliases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public_aliases)


@pytest.mark.parametrize(
    "name, aliases",
    [
        ("ControlledPhaseShift", ["CPhase"]),
        ("CPhase", ["ControlledPhaseShift"]),
        ("PauliX", ["X"]),
        ("X", ["PauliX"]),
        ("Hadamard", ["H"]),
        ("Identity", ["I"]),
        ("Select", ["Multiplexer", "Multiplexor"]),
    ],
)
def test_class_aliases(name, aliases):
    """Alias links are sorted, bidirectional, and omit the documented name."""
    lines = ["Original documentation."]
    public_aliases.add_public_aliases(
        None, "class", f"pennylane.{name}", getattr(qp, name), None, lines
    )
    references = ", ".join(f":class:`~pennylane.{alias}`" for alias in aliases)
    assert lines == [
        "Original documentation.",
        "",
        ".. note::",
        "",
        f"   Also available as {references}.",
        "",
    ]


def test_function_aliases():
    """Functions use the Sphinx function role."""
    lines = []
    public_aliases.add_public_aliases(
        None, "function", "pennylane.commutator", qp.commutator, None, lines
    )
    assert ":func:`~pennylane.comm`" in "\n".join(lines)


@pytest.mark.parametrize(
    "what, name, obj",
    [("class", "U1", qp.U1), ("method", "PauliX.matrix", qp.X.matrix), ("class", "dict", dict)],
)
def test_no_spurious_aliases(what, name, obj):
    """Equivalent operators, methods, and third-party objects are not aliases."""
    lines = ["Original documentation."]
    public_aliases.add_public_aliases(None, what, name, obj, None, lines)
    assert lines == ["Original documentation."]


def test_new_and_private_aliases(monkeypatch):
    """New public exports are discovered without a registry; private names are ignored."""
    monkeypatch.setattr(qp, "ExtraX", qp.X, raising=False)
    monkeypatch.setattr(qp, "_PrivateX", qp.X, raising=False)
    lines = []
    public_aliases.add_public_aliases(None, "class", "pennylane.PauliX", qp.X, None, lines)
    text = "\n".join(lines)
    assert ":class:`~pennylane.ExtraX`" in text
    assert "_PrivateX" not in text
    assert ":class:`~pennylane.PauliX`" not in text


def test_setup():
    """The extension registers the autodoc event."""
    app = Mock()
    public_aliases.setup(app)
    assert app.mock_calls == [
        call.setup_extension("sphinx.ext.autodoc"),
        call.connect("autodoc-process-docstring", public_aliases.add_public_aliases),
    ]


def test_extension_loads_before_autodoc(tmp_path):
    """The actual Sphinx application supports the extension order in doc/conf.py."""
    pytest.importorskip("sphinx")
    source = tmp_path / "source"
    source.mkdir()
    (source / "conf.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(EXTENSION_PATH.parent)!r})\n"
        "extensions = ['public_aliases', 'sphinx.ext.autodoc']\n"
        "project = 'Alias documentation'\n",
        encoding="utf-8",
    )
    (source / "index.rst").write_text(
        "Alias documentation\n===================\n", encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-n",
            "-W",
            "-b",
            "html",
            str(source),
            str(tmp_path / "html"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
