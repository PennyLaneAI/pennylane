# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane` configuration classe :class:`Configuration`.
"""
# pylint: disable=protected-access

import contextlib
import importlib
import io
import json
import re

import pytest

import pennylane as qml


@pytest.mark.slow
def test_about():
    """
    about: Tests if the about string prints correct.
    """
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        qml.about()
    out = f.getvalue().strip()

    assert "Version:" in out
    pl_version_match = re.search(r"Version:\s+([\S]+)\n", out).group(1)
    # 0.43.0-rc0 -> 0.43.0rc0
    # 0.43.0-dev0 -> 0.43.0.dev0
    is_rc_version = "rc" in pl_version_match
    assert qml.version().replace("-", "" if is_rc_version else ".") in pl_version_match
    assert "Numpy version" in out
    assert "Scipy version" in out
    assert "default.qubit" in out
    assert "default.gaussian" in out


def test_about_prints_core_fields(capsys):
    about = importlib.import_module("pennylane.about")
    about.about()
    cap = capsys.readouterr()

    assert cap.err == ""
    assert re.search(r"Name:\s*pennylane", cap.out, re.I)
    assert "Version:" in cap.out
    assert "Summary:" in cap.out
    assert "Location:" in cap.out


def test_about_shows_editable_location(monkeypatch, capsys):
    about = importlib.import_module("pennylane.about")

    class Dist:  # pylint: disable=too-few-public-methods
        metadata = {"Name": "PennyLane", "Version": "x"}

        @staticmethod
        def read_text(name):  # pylint: disable=unused-argument
            return json.dumps({"dir_info": {"editable": True}, "url": "file:///tmp/pl"})

    im = importlib.import_module("importlib.metadata")
    monkeypatch.setattr(about, "metadata", im)
    monkeypatch.setattr(im, "distribution", lambda _name: Dist())
    monkeypatch.setattr(about, "_pkg_location", lambda: "/site-packages/pennylane")

    about.about()
    out = capsys.readouterr().out
    assert "Editable project location: /tmp/pl" in out
    assert re.search(r"Location:\s*/site-packages/pennylane", out)


def test_catalyst_version(monkeypatch):
    """Tests the catalyst_version function."""
    about = importlib.import_module("pennylane.about")

    # Test when catalyst is not found
    monkeypatch.setattr(about, "find_spec", lambda name: None if name == "catalyst" else True)
    assert about.catalyst_version() is None

    # Test when catalyst is found but version is not available
    monkeypatch.setattr(about, "find_spec", lambda name: True)
    monkeypatch.setattr(
        about, "version", lambda name: "0.1.0" if name == "pennylane_catalyst" else None
    )
    assert about.catalyst_version() == "0.1.0"
