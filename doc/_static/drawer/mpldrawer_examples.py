# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This file automatically generates and saves a series of example pictures for the
circuit drawer. This will be useful during early stages when the project is still
undergoing cosmetic changes.
"""

import pathlib
import matplotlib.pyplot as plt

from pennylane.drawer import MPLDrawer
from pennylane.drawer.style import _set_style

folder = pathlib.Path(__file__).parent


def labels(savefile="labels_test.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=1)
    drawer.label(["a", "b"])
    plt.savefig(folder / savefile)
    plt.close()


def labels_formatted(savefile="labels_formatted.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=1)
    drawer.label(["a", "b"], text_options={"color": "indigo", "fontsize": "xx-large"})
    plt.savefig(folder / savefile)
    plt.close()


def box_gates(savefile="box_gates.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=1)

    drawer.box_gate(layer=0, wires=(0, 1), text="CY")
    plt.savefig(folder / savefile)
    plt.close()


def box_gates_formatted(savefile="box_gates_formatted.png"):
    box_options = {"facecolor": "lightcoral", "edgecolor": "maroon", "linewidth": 5}
    text_options = {"fontsize": "xx-large", "color": "maroon"}

    drawer = MPLDrawer(n_wires=2, n_layers=1)

    drawer.box_gate(
        layer=0, wires=(0, 1), text="CY", box_options=box_options, text_options=text_options
    )
    plt.savefig(folder / savefile)
    plt.close()


def autosize(savefile="box_gates_autosized.png"):

    drawer = MPLDrawer(n_layers=4, n_wires=2)

    drawer.box_gate(layer=0, wires=0, text="A longer label")
    drawer.box_gate(layer=0, wires=1, text="Label")

    drawer.box_gate(layer=1, wires=(0, 1), text="long multigate label")

    drawer.box_gate(layer=3, wires=(0, 1), text="Not autosized label", autosize=False)

    plt.savefig(folder / savefile)
    plt.close()


def ctrl(savefile="ctrl.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=3)

    drawer.ctrl(layer=0, wires=0, wires_target=1)
    drawer.ctrl(layer=1, wires=(0, 1), control_values=[0, 1])

    options = {"color": "indigo", "linewidth": 4}
    drawer.ctrl(layer=2, wires=(0, 1), control_values=[1, 0], options=options)

    plt.savefig(folder / savefile)
    plt.close()


def cond(savefile="cond.png"):
    drawer = MPLDrawer(n_wires=3, n_layers=4)

    drawer.cond(layer=1, measured_layer=0, wires=[0], wires_target=[1])

    options = {'color': "indigo", 'linewidth': 1.5}
    drawer.cond(layer=3, measured_layer=2, wires=(1,), wires_target=(2,), options=options)

    plt.savefig(folder / savefile)
    plt.close()


def CNOT(savefile="cnot.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=2)

    drawer.CNOT(0, (0, 1))

    options = {"color": "indigo", "linewidth": 4}
    drawer.CNOT(1, (1, 0), options=options)
    plt.savefig(folder / savefile)
    plt.close()


def SWAP(savefile="SWAP.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=2)

    drawer.SWAP(0, (0, 1))

    swap_options = {"linewidth": 2, "color": "indigo"}
    drawer.SWAP(1, (0, 1), options=swap_options)

    plt.savefig(folder / savefile)
    plt.close()


def measure(savefile="measure.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=1)
    drawer.measure(layer=0, wires=0)

    measure_box = {"facecolor": "white", "edgecolor": "indigo"}
    measure_lines = {"edgecolor": "indigo", "facecolor": "plum", "linewidth": 2}
    drawer.measure(layer=0, wires=1, box_options=measure_box, lines_options=measure_lines)

    plt.savefig(folder / savefile)
    plt.close()


def integration(style="default", savefile="example_basic.png"):
    _set_style(style)
    drawer = MPLDrawer(n_wires=5, n_layers=6)

    drawer.label(["0", "a", r"$|\Psi\rangle$", r"$|\theta\rangle$", "aux"])

    drawer.box_gate(0, [0, 1, 2, 3, 4], "Entangling Layers")
    drawer.box_gate(1, [0, 2, 3], "U(θ)")

    drawer.box_gate(1, 4, "Z")

    drawer.SWAP(2, (3, 4))
    drawer.CNOT(2, (0, 2))

    drawer.ctrl(3, [1, 3], control_values=[True, False])
    drawer.box_gate(
        layer=3, wires=2, text="H", box_options={"zorder": 4}, text_options={"zorder": 5}
    )

    drawer.ctrl(4, [1, 2])

    drawer.measure(5, 0)

    drawer.fig.suptitle("My Circuit", fontsize="xx-large")
    plt.savefig(folder / savefile)
    _set_style("default")
    plt.close()


def integration_rcParams(savefile="example_rcParams.png"):
    plt.rcParams["patch.facecolor"] = "mistyrose"
    plt.rcParams["patch.edgecolor"] = "maroon"
    plt.rcParams["text.color"] = "maroon"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["patch.linewidth"] = 4
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["lines.color"] = "indigo"
    plt.rcParams["lines.linewidth"] = 5
    plt.rcParams["figure.facecolor"] = "ghostwhite"

    drawer = MPLDrawer(n_wires=5, n_layers=5)

    drawer = MPLDrawer(n_wires=5, n_layers=6)

    drawer.label(["0", "a", r"$|\Psi\rangle$", r"$|\theta\rangle$", "aux"])

    drawer.box_gate(0, [0, 1, 2, 3, 4], "Entangling Layers")
    drawer.box_gate(1, [0, 2, 3], "U(θ)")

    drawer.box_gate(1, 4, "Z")

    drawer.SWAP(2, (3, 4))
    drawer.CNOT(2, (0, 2))

    drawer.ctrl(3, [1, 3], control_values=[True, False])
    drawer.box_gate(
        layer=3, wires=2, text="H", box_options={"zorder": 4}, text_options={"zorder": 5}
    )

    drawer.ctrl(4, [1, 2])

    drawer.measure(5, 0)

    drawer.fig.suptitle("My Circuit", fontsize="xx-large")

    plt.savefig(folder / savefile)
    _set_style("default")
    plt.close()


def integration_formatted(savefile="example_formatted.png"):

    wire_options = {"color": "indigo", "linewidth": 4}
    drawer = MPLDrawer(n_wires=2, n_layers=4, wire_options=wire_options)

    label_options = {"fontsize": "x-large", "color": "indigo"}
    drawer.label(["0", "a"], text_options=label_options)

    box_options = {"facecolor": "lightcoral", "edgecolor": "maroon", "linewidth": 5}
    text_options = {"fontsize": "xx-large", "color": "maroon"}
    drawer.box_gate(layer=0, wires=0, text="Z", box_options=box_options, text_options=text_options)

    swap_options = {"linewidth": 4, "color": "darkgreen"}
    drawer.SWAP(layer=1, wires=(0, 1), options=swap_options)

    ctrl_options = {"linewidth": 4, "color": "teal"}
    drawer.CNOT(layer=2, wires=(0, 1), options=ctrl_options)
    drawer.ctrl(layer=3, wires=(0, 1), options=ctrl_options)

    measure_box = {"facecolor": "white", "edgecolor": "indigo"}
    measure_lines = {"edgecolor": "indigo", "facecolor": "plum", "linewidth": 2}
    for wire in range(2):
        drawer.measure(layer=4, wires=wire, box_options=measure_box, lines_options=measure_lines)

    drawer.fig.suptitle("My Circuit", fontsize="xx-large")

    plt.savefig(folder / savefile)
    plt.close()


def float_layer(savefile="float_layer.png"):

    drawer = MPLDrawer(2, 2)

    drawer.box_gate(layer=0.5, wires=0, text="Big Gate", extra_width=0.5)
    drawer.box_gate(layer=0, wires=1, text="X")
    drawer.box_gate(layer=1, wires=1, text="Y")

    plt.savefig(folder / savefile)
    plt.close()


if __name__ == "__main__":
    labels()
    labels_formatted()
    box_gates()
    box_gates_formatted()
    autosize()
    ctrl()
    CNOT()
    SWAP()
    measure()
    integration()
    float_layer()
    integration(style="black_white", savefile="black_white_style.png")
    integration_rcParams()
    integration_formatted()
    cond()
