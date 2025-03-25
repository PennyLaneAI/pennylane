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
This file generates the images used in docstrings for``qml.transforms.draw.draw_mpl``.
This makes it easier to keep docstrings up to date with the latest styling.

It is not intended to be used in any Continuous Integration, but save time and hassle
for developers when making any change that might impact the resulting figures.
"""

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import draw_mpl

folder = pathlib.Path(__file__).parent


def main_example(circuit):

    fig, ax = draw_mpl(circuit)(1.2345, 1.2345)

    plt.savefig(folder / "main_example.png")
    plt.close()


def decimals(dev):
    @qml.qnode(dev)
    def circuit2(x, y):
        qml.RX(x, wires=0)
        qml.Rot(*y, wires=0)
        return qml.expval(qml.PauliZ(0))

    fig, ax = draw_mpl(circuit2, decimals=2)(1.23456, [1.2345, 2.3456, 3.456])

    plt.savefig(folder / "decimals.png")


def wire_order(circuit):
    fig, ax = draw_mpl(circuit, wire_order=[3, 2, 1, 0])(1.2345, 1.2345)
    plt.savefig(folder / "wire_order.png")
    plt.close()


def show_all_wires(circuit):
    fig, ax = draw_mpl(circuit, wire_order=["aux"], show_all_wires=True)(1.2345, 1.2345)

    plt.savefig(folder / "show_all_wires.png")
    plt.close()


def postprocessing(circuit):
    fig, ax = draw_mpl(circuit)(1.2345, 1.2345)
    fig.suptitle("My Circuit", fontsize="xx-large")

    options = {"facecolor": "white", "edgecolor": "#f57e7e", "linewidth": 6, "zorder": -1}
    box1 = plt.Rectangle((-0.5, -0.5), width=3.0, height=4.0, **options)
    ax.add_patch(box1)

    ax.annotate(
        "CSWAP",
        xy=(3, 2.5),
        xycoords="data",
        xytext=(3.8, 1.5),
        textcoords="data",
        arrowprops={"facecolor": "black"},
        fontsize=14,
    )

    plt.savefig(folder / "postprocessing.png")
    plt.close()


def rcparams(circuit):
    plt.rcParams["patch.facecolor"] = "mistyrose"
    plt.rcParams["patch.edgecolor"] = "maroon"
    plt.rcParams["text.color"] = "maroon"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["patch.linewidth"] = 4
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["lines.color"] = "indigo"
    plt.rcParams["lines.linewidth"] = 5
    plt.rcParams["figure.facecolor"] = "ghostwhite"

    fig, ax = qml.draw_mpl(circuit, style="rcParams")(1.2345, 1.2345)

    plt.savefig(folder / "rcparams.png")
    plt.close()
    qml.drawer.use_style("black_white")


def use_style(circuit):

    fig, ax = qml.draw_mpl(circuit, style="sketch")(1.2345, 1.2345)

    plt.savefig(folder / "sketch_style.png")
    plt.close()
    qml.drawer.use_style("black_white")


def wires_labels(circuit):
    fig, ax = draw_mpl(
        circuit, wire_options={"color": "teal", "linewidth": 5}, label_options={"size": 20}
    )(1.2345, 1.2345)
    plt.savefig(folder / "wires_labels.png")
    plt.close()


def mid_measure():
    def circuit():
        m0 = qml.measure(0)
        qml.Hadamard(1)
        qml.cond(m0, qml.PauliZ)(1)

    _ = draw_mpl(circuit)()
    plt.savefig(folder / "mid_measure.png")
    plt.close()

def max_length():
    def circuit():
        for _ in range(10):
            qml.X(0)
        return qml.expval(qml.Z(0))

    figs_and_axes = draw_mpl(circuit, max_length=5)()
    figs_and_axes[0][0].savefig(folder / "max_length1.png")
    figs_and_axes[1][0].savefig(folder / "max_length2.png")

@qml.transforms.merge_rotations
@qml.transforms.cancel_inverses
@qml.qnode(qml.device("default.qubit"), diff_method="parameter-shift")
def _levels_circ():
    qml.RandomLayers([[1.0, 20]], wires=(0, 1))
    qml.Permute([2, 1, 0], wires=(0, 1, 2))
    qml.PauliX(0)
    qml.PauliX(0)
    qml.RX(0.1, wires=0)
    qml.RX(-0.1, wires=0)
    return qml.expval(qml.PauliX(0))


def levels():
    for level in ("top", "user", None, slice(1, 2)):
        draw_mpl(_levels_circ, level=level)()
        plt.savefig(folder / f"level_{str(level).split('(')[0].lower()}.png")
        plt.close


if __name__ == "__main__":

    dev = qml.device('lightning.qubit', wires=(0,1,2,3))

    @qml.qnode(dev)
    def circuit(x, z):
        qml.QFT(wires=(0,1,2,3))
        qml.IsingXX(1.234, wires=(0,2))
        qml.Toffoli(wires=(0,1,2))
        mcm = qml.measure(1)
        mcm_out = qml.measure(2)
        qml.CSWAP(wires=(0,2,3))
        qml.RX(x, wires=0)
        qml.cond(mcm, qml.RY)(np.pi / 4, wires=3)
        qml.CRZ(z, wires=(3,0))
        return qml.expval(qml.Z(0)), qml.probs(op=mcm_out)


    main_example(circuit)
    decimals(dev)
    wire_order(circuit)
    show_all_wires(circuit)
    postprocessing(circuit)
    use_style(circuit)
    rcparams(circuit)
    wires_labels(circuit)
    mid_measure()
    levels()
    max_length()
