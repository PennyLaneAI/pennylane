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
circuit drawer.  This will be useful during early stages when the project is still
undergoing cosmetic changes.
"""

import matplotlib.pyplot as plt

from pennylane.circuit_drawer import MPLDrawer


def labels(savefile="labels.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=1)
    drawer.label(["a", "b"])
    plt.savefig(savefile)


def box_gates(savefile="box_gates.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=2)

    drawer.box_gate(layer=0, wires=0, text="Y")
    drawer.box_gate(layer=1, wires=(0, 1), text="CRy(0.1)", rotate_text=True)
    plt.savefig(savefile)


def ctrl(savefile="ctrl.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=2)

    drawer.ctrl(layer=0, wire_ctrl=0, wire_target=1)
    drawer.ctrl(layer=1, wire_ctrl=(0, 1))
    plt.savefig(savefile)


def CNOT(savefile="cnot.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=2)

    drawer.CNOT(0, (0, 1))
    drawer.CNOT(1, (1, 0))
    plt.savefig(savefile)


def SWAP(savefile="SWAP.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=1)

    drawer.SWAP(0, (0, 1))
    plt.savefig(savefile)


def measure(savefile="measure.png"):
    drawer = MPLDrawer(n_wires=1, n_layers=1)

    drawer.measure(0, 0)
    plt.savefig(savefile)

def integration(style='default', savefile="example_basic.png"):
    plt.style.use(style)
    drawer = MPLDrawer(n_wires=5,n_layers=5)

    drawer.label(["0","a",r"$|\Psi\rangle$",r"$|\theta\rangle$", "aux"])

    drawer.box_gate(0, [0,1,2,3,4], "Entangling Layers", rotate_text=True)
    drawer.box_gate(1, [0, 1], "U(θ)")
    drawer.box_gate(1, 4, "X", color='lightcoral')

    drawer.SWAP(1, (2, 3))
    drawer.CNOT(2, (0,2), color='forestgreen')

    drawer.ctrl(3, [1,3])
    drawer.box_gate(3, 2, "H", zorder_base=2)

    drawer.ctrl(4, [1,2])

    drawer.measure(5, 0)

    drawer.fig.suptitle('My Circuit', fontsize='xx-large')
    plt.savefig(savefile)
    plt.style.use('default')

def float_layer(savefile="float_layer.png"):
    drawer = MPLDrawer(2,2)

    drawer.box_gate(layer=0.5, wires=0, text="Big Gate", extra_width=0.5)
    drawer.box_gate(layer=0, wires=1, text="X")
    drawer.box_gate(layer=1, wires=1, text="Y")

    plt.savefig(savefile)


labels()
box_gates()
ctrl()
CNOT()
SWAP()
measure()
integration()
float_layer()
integration(style='Solarize_Light2', savefile="example_Solarize_Light2.png")

