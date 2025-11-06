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
"""Unit test module for the MLIR graph generation in the Unified Compiler visualization module."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.external

pytest.importorskip("xdsl")
pytest.importorskip("catalyst")


# pylint: disable=wrong-import-position
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import (
    iterative_cancel_inverses_pass,
    merge_rotations_pass,
)
from pennylane.compiler.python_compiler.visualization import generate_mlir_graph


@pytest.fixture(autouse=True)
def _chdir_tmp(monkeypatch, tmp_path: Path):
    """Ensure all tests run inside a temp directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def collect_files(tmp_path: Path) -> set[str]:
    """Return the set of generated SVG files."""
    out_dir = tmp_path / "mlir_generated_graphs"
    return {f.name for f in out_dir.glob("*.svg")}


def assert_files(tmp_path: Path, expected: set[str]):
    """Check that the generated files match the expected set."""
    files = collect_files(tmp_path)
    assert files == expected, f"Expected {expected}, got {files}"


@pytest.mark.usefixtures("enable_disable_plxpr")
class TestMLIRGraph:
    "Test the MLIR graph generation"

    @pytest.mark.parametrize("qjit", [True, False])
    def test_no_transforms(self, tmp_path: Path, qjit: bool):
        "Test the MLIR graph is still generated when no transforms are applied"

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def _():
            qml.RX(0.1, 0)
            qml.RX(2.0, 0)
            qml.CNOT([0, 2])
            qml.CNOT([0, 2])
            return qml.state()

        if qjit:
            _ = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(_)

        generate_mlir_graph(_)()
        assert collect_files(tmp_path) == {"QNode_level_0_no_transforms.svg"}

    @pytest.mark.parametrize("qjit", [True, False])
    def test_xdsl_transforms_no_args(self, tmp_path: Path, qjit: bool):
        "Test the MLIR graph generation with no arguments to the QNode with and without qjit"

        @merge_rotations_pass
        @iterative_cancel_inverses_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def _():
            qml.RX(0.1, 0)
            qml.RX(2.0, 0)
            qml.CNOT([0, 2])
            qml.CNOT([0, 2])
            return qml.state()

        if qjit:
            _ = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(_)

        generate_mlir_graph(_)()
        assert_files(
            tmp_path,
            {
                "QNode_level_0_no_transforms.svg",
                "QNode_level_1_after_xdsl-cancel-inverses.svg",
                "QNode_level_2_after_xdsl-merge-rotations.svg",
            },
        )

    @pytest.mark.parametrize("qjit", [True, False])
    def test_xdsl_transforms_args(self, tmp_path: Path, qjit: bool):
        "Test the MLIR graph generation with arguments to the QNode for xDSL transforms"

        @merge_rotations_pass
        @iterative_cancel_inverses_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def _(x, y, w1, w2):
            qml.RX(x, w1)
            qml.RX(y, w2)
            return qml.state()

        if qjit:
            _ = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(_)

        generate_mlir_graph(_)(0.1, 0.2, 0, 1)
        assert_files(
            tmp_path,
            {
                "QNode_level_0_no_transforms.svg",
                "QNode_level_1_after_xdsl-cancel-inverses.svg",
                "QNode_level_2_after_xdsl-merge-rotations.svg",
            },
        )

    @pytest.mark.parametrize("qjit", [True, False])
    def test_catalyst_transforms_args(self, tmp_path: Path, qjit: bool):
        "Test the MLIR graph generation with arguments to the QNode for catalyst transforms"

        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def _(x, y, w1, w2):
            qml.RX(x, w1)
            qml.RX(y, w2)
            return qml.state()

        if qjit:
            _ = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(_)

        generate_mlir_graph(_)(0.1, 0.2, 0, 1)
        assert_files(
            tmp_path,
            {
                "QNode_level_0_no_transforms.svg",
                "QNode_level_1_after_remove-chained-self-inverse.svg",
                "QNode_level_2_after_merge-rotations.svg",
            },
        )

    @pytest.mark.parametrize("qjit", [True, False])
    def test_catalyst_xdsl_transforms_args(self, tmp_path: Path, qjit: bool):
        "Test the MLIR graph generation with arguments to the QNode for catalyst and xDSL transforms"

        @qml.transforms.merge_rotations
        @iterative_cancel_inverses_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def _(x, y, w1, w2):
            qml.RX(x, w1)
            qml.RX(y, w2)
            return qml.state()

        if qjit:
            _ = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(_)

        generate_mlir_graph(_)(0.1, 0.2, 0, 1)
        assert_files(
            tmp_path,
            {
                "QNode_level_0_no_transforms.svg",
                "QNode_level_1_after_xdsl-cancel-inverses.svg",
                "QNode_level_2_after_merge-rotations.svg",
            },
        )

    def test_cond(self, tmp_path: Path):
        "Test the MLIR graph generation for a conditional"

        @merge_rotations_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def _(pred, arg1, arg2):
            """Quantum circuit with conditional branches."""

            qml.RX(0.10, wires=0)

            def true_fn(arg1, arg2):
                qml.RY(arg1, wires=0)
                qml.RX(arg2, wires=0)
                qml.RZ(arg1, wires=0)

            def false_fn(arg1, arg2):
                qml.RX(arg1, wires=0)
                qml.RX(arg2, wires=0)

            qml.cond(pred > 0, true_fn, false_fn)(arg1, arg2)
            qml.RX(0.10, wires=0)
            return qml.expval(qml.Z(wires=0))

        generate_mlir_graph(_)(0.5, 0.1, 0.2)
        assert_files(
            tmp_path,
            {
                "QNode_level_0_no_transforms.svg",
                "QNode_level_1_after_xdsl-merge-rotations.svg",
            },
        )

    def test_cond_with_mcm(self, tmp_path: Path):
        "Test the MLIR graph generation for a conditional with MCM"

        def true_fn(arg):
            qml.RX(arg, 0)

        def false_fn(arg):
            qml.RY(3 * arg, 0)

        @merge_rotations_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def _(x, y):
            """Quantum circuit with conditional branches."""

            qml.RX(x, 0)
            m = qml.measure(0)

            qml.cond(m, true_fn, false_fn)(y)
            return qml.expval(qml.Z(0))

        generate_mlir_graph(_)(0.5, 0.1)
        assert_files(
            tmp_path,
            {
                "QNode_level_0_no_transforms.svg",
                "QNode_level_1_after_xdsl-merge-rotations.svg",
            },
        )

    def test_for_loop(self, tmp_path: Path):
        "Test the MLIR graph generation for a for loop"

        @merge_rotations_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def _():
            @qml.for_loop(0, 100)
            def loop(_):
                qml.RX(0.1, 0)
                qml.RX(0.1, 0)

            # pylint: disable=no-value-for-parameter
            loop()
            return qml.state()

        generate_mlir_graph(_)()
        assert_files(
            tmp_path,
            {
                "QNode_level_0_no_transforms.svg",
                "QNode_level_1_after_xdsl-merge-rotations.svg",
            },
        )

    def test_while_loop(self, tmp_path: Path):
        "Test the MLIR graph generation for a while loop"

        @merge_rotations_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def _(x):
            def cond_fn(x):
                return x < 2

            @qml.while_loop(cond_fn)
            def loop(x):
                return x**2

            loop(x)
            return qml.expval(qml.PauliZ(0))

        generate_mlir_graph(_)(0.5)
        assert_files(
            tmp_path,
            {
                "QNode_level_0_no_transforms.svg",
                "QNode_level_1_after_xdsl-merge-rotations.svg",
            },
        )


if __name__ == "__main__":
    pytest.main(["-x", __file__])
