# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit test module for the merge rotations transform"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.external

pytest.importorskip("xdsl")
pytest.importorskip("catalyst")


# pylint: disable=wrong-import-position
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.visualization import generate_mlir_graph


@pytest.mark.usefixtures("enable_disable_plxpr")
class TestMLIRGraph:
    "Test the MLIR graph generation"

    def _collect_files(self, tmp_path: Path) -> set[str]:
        out_dir = tmp_path / "mlir_generated_graphs"
        return {f.name for f in out_dir.glob("*.svg")}

    def test_no_transforms(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        "Test the MLIR graph is not generated when no transforms are applied"

        monkeypatch.chdir(tmp_path)

        @generate_mlir_graph
        @qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def _():
            qml.RX(0.1, 0)
            qml.RX(2.0, 0)
            qml.CNOT([0, 2])
            qml.CNOT([0, 2])
            return qml.state()

        _()

        assert not self._collect_files(tmp_path)

    def test_transforms_no_args(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        "Test the MLIR graph generation with no arguments to the QNode"

        monkeypatch.chdir(tmp_path)

        @generate_mlir_graph
        @qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
        @qml.compiler.python_compiler.transforms.merge_rotations_pass
        @qml.compiler.python_compiler.transforms.iterative_cancel_inverses_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def _():
            qml.RX(0.1, 0)
            qml.RX(2.0, 0)
            qml.CNOT([0, 2])
            qml.CNOT([0, 2])
            return qml.state()

        _()

        files = self._collect_files(tmp_path)
        assert len(files) == 3
        assert files == {
            "QNode_level_0_no_transforms.svg",
            "QNode_level_1_after_xdsl-merge-rotations.svg",
            "QNode_level_2_after_xdsl-cancel-inverses.svg",
        }

    def test_transforms_args(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        "Test the MLIR graph generation with arguments to the QNode"

        monkeypatch.chdir(tmp_path)

        @qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
        @qml.compiler.python_compiler.transforms.merge_rotations_pass
        @qml.compiler.python_compiler.transforms.iterative_cancel_inverses_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def _(x, y):
            qml.RX(x, 0)
            qml.RX(y, 0)
            return qml.state()

        generate_mlir_graph(_)(0.1, 0.2)

        files = self._collect_files(tmp_path)
        assert len(files) == 3
        assert files == {
            "QNode_level_0_no_transforms.svg",
            "QNode_level_1_after_xdsl-merge-rotations.svg",
            "QNode_level_2_after_xdsl-cancel-inverses.svg",
        }


if __name__ == "__main__":
    pytest.main(["-x", __file__])
