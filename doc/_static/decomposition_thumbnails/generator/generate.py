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
This file generates the circuit diagrams shown by the ``decomposition-table`` directive
(``doc/_ext/decomposition_table.py``), one SVG per decomposition rule, drawn with
:func:`~.draw_mpl` in the ``"pennylane"`` style.

It is not intended to be used in any Continuous Integration, but saves time and hassle for
developers adding or changing a decomposition rule.

Usage, from this folder::

    python generate.py                # regenerate every diagram
    python generate.py hadamard_ppm   # regenerate only the named diagrams

Each diagram is registered in an ``examples_*.py`` module with ``@example("<name>")`` on a
quantum function that applies one decomposition rule to the instance shown in its table. It is
written to ``../<name>.svg``.
"""

import importlib
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
# fixed element ids, so that regenerating an unchanged diagram leaves its file unchanged
matplotlib.rcParams["svg.hashsalt"] = "decomposition-thumbnails"

# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt

import pennylane as qp

folder = pathlib.Path(__file__).parent
EXAMPLES = {}

# composite operators that only group the rule's gates, and are drawn as those gates
_WRAPPERS = ("ChangeOpBasis", "Prod")


def example(name, wires=1, work=r"$|0\rangle$"):
    """Register the quantum function decorated by this as the diagram ``name``.

    The quantum function takes the wires to draw, as a list of their labels, and applies one
    decomposition rule to the instance shown in the rule's table.

    Args:
        name (str): the diagram's file name, without the ``.svg`` extension
        wires (int | list[str]): the number of wires, labelled ``w_0, w_1, ...``, or their labels
        work (str): the label of wires allocated by the rule
    """
    if isinstance(wires, int):
        wires = [f"$w_{{{i}}}$" for i in range(wires)]

    def deco(qfunc):
        EXAMPLES[name] = (qfunc, list(wires), work)
        return qfunc

    return deco


class _DrawnGlobalPhase(qp.operation.Operation):
    """Draws a global phase across the given wires.

    A conditional :class:`~.GlobalPhase` acts on no wires, which :func:`~.drawer.tape_mpl` cannot
    draw, so it is drawn with this stand-in instead.
    """

    num_params = 1

    def label(self, decimals=None, base_label=None, cache=None):
        return "GlobalPhase"


def _expand(op):
    """The gates of ``op``, with grouping operators replaced by the gates they group."""
    if op.name.startswith(_WRAPPERS):
        return [g for sub in op.decomposition() for g in _expand(sub)]
    return [op]


def _tape(qfunc, wires):
    with qp.queuing.AnnotatedQueue() as q:
        qfunc(wires)
    ops = [g for op in qp.tape.QuantumScript.from_queue(q).operations for g in _expand(op)]
    ops = [op for op in ops if op.name not in ("Allocate", "Deallocate")]
    all_wires = qp.wires.Wires.all_wires([wires] + [op.wires for op in ops])

    # a global phase acts on no wire, but it is drawn across all of them
    def on_all_wires(op):
        if op.name == "Conditional(GlobalPhase)" and not op.wires:
            phase = _DrawnGlobalPhase(*op.base.parameters, wires=all_wires)
            return qp.ops.Conditional(op.meas_val, phase)
        return op

    return qp.tape.QuantumScript([on_all_wires(op) for op in ops]), all_wires


def draw(name):
    """Draw the registered diagram ``name`` to ``../<name>.svg``."""
    qfunc, wires, work = EXAMPLES[name]
    tape, all_wires = _tape(qfunc, wires)
    fig, ax = qp.drawer.tape_mpl(tape, style="pennylane", wire_order=all_wires, show_all_wires=True)
    # wires allocated by the rule have no label of their own
    for text in ax.texts:
        if text.get_text() == "<DynamicWire>":
            text.set_text(work)
    fig.savefig(
        folder.parent / f"{name}.svg",
        bbox_inches="tight",
        transparent=True,
        metadata={"Date": None},
    )
    plt.close(fig)


if __name__ == "__main__":
    sys.path.insert(0, str(folder))
    # the examples register into the importable ``generate`` module, not into ``__main__``
    generate = importlib.import_module("generate")
    for module in sorted(folder.glob("examples_*.py")):
        importlib.import_module(module.stem)

    names = sys.argv[1:] or list(generate.EXAMPLES)
    unknown = set(names) - set(generate.EXAMPLES)
    if unknown:
        sys.exit(f"Unknown diagrams: {', '.join(sorted(unknown))}")
    for n in names:
        generate.draw(n)
    print(f"Wrote {len(names)} diagrams to {folder.parent}")
