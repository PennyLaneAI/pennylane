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
This file generates the images used in docstrings for``qml.circuit_drawer.tape_mpl``.
This makes it easier to keep docstrings up to date with the latest styling.

It is not intended to be used in any Continuous Integration, but save time and hassle
for developers when making any change that might impact the resulting figures.
"""

import pathlib
import matplotlib.pyplot as plt
import pennylane as qml


from pennylane.drawer import tape_mpl

folder = pathlib.Path(__file__).parent

def default(tape):

    fig, ax = tape_mpl(tape)

    plt.savefig(folder / "default.png")
    plt.close()

def decimals():

    with qml.tape.QuantumTape() as tape2:
        qml.RX(1.23456, wires=0)
        qml.Rot(1.2345,2.3456,3.456, wires=0)
        qml.expval(qml.PauliZ(0))

    fig, ax = tape_mpl(tape2, decimals=2)
    plt.savefig(folder / "decimals.png")
    plt.close()

def wire_order(tape):

    fig, ax = tape_mpl(tape, wire_order=[3,2,1,0])

    plt.savefig(folder / "wire_order.png")
    plt.close()

def show_all_wires(tape):

    fig, ax = tape_mpl(tape, wire_order=["aux"], show_all_wires=show_all_wires)
    plt.savefig(folder / "show_all_wires.png")
    plt.close()

def rcparams(tape):

    plt.rcParams['patch.facecolor'] = 'mistyrose'
    plt.rcParams['patch.edgecolor'] = 'maroon'
    plt.rcParams['text.color'] = 'maroon'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['patch.linewidth'] = 4
    plt.rcParams['patch.force_edgecolor'] = True
    plt.rcParams['lines.color'] = 'indigo'
    plt.rcParams['lines.linewidth'] = 5
    plt.rcParams['figure.facecolor'] = 'ghostwhite'

    fig, ax = tape_mpl(tape)

    plt.savefig(folder / "rcparams.png")
    plt.close()
    plt.style.use('default')

def use_style(tape):

    qml.drawer.use_style('black_white')

    fig, ax = tape_mpl(tape)

    plt.savefig(folder / "black_white_style.png")
    plt.close()
    plt.style.use('default')

def wires_and_labels(tape):

    fig, ax = tape_mpl(tape, wire_options={'color':'black', 'linewidth': 5},
                  label_options={'size': 20})

    plt.savefig(folder / "wires_labels.png")
    plt.close()

def postprocessing(tape):

    fig, ax = tape_mpl(tape)
    fig.suptitle("My Circuit", fontsize="xx-large")

    options = {'facecolor': "white", 'edgecolor': "#f57e7e", "linewidth": 6, "zorder": -1}
    box1 = plt.Rectangle((-0.5, -0.5), width=3.0, height=4.0, **options)
    ax.add_patch(box1)

    ax.annotate("CSWAP", xy=(3, 2.5), xycoords='data', xytext=(3.8,1.5), textcoords='data',
                arrowprops={'facecolor': 'black'}, fontsize=14)

    plt.savefig(folder / "postprocessing.png")
    plt.close()

if __name__ == "__main__":

    with qml.tape.QuantumTape() as tape:
        qml.QFT(wires=(0,1,2,3))
        qml.IsingXX(1.234, wires=(0,2))
        qml.Toffoli(wires=(0,1,2))
        qml.CSWAP(wires=(0,2,3))
        qml.RX(1.2345, wires=0)
        qml.CRZ(1.2345, wires=(3,0))
        qml.expval(qml.PauliZ(0))

    default(tape)
    decimals()
    wire_order(tape)
    show_all_wires(tape)
    use_style(tape)
    rcparams(tape)
    wires_and_labels(tape)
    postprocessing(tape)
