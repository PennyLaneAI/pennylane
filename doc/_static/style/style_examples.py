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

import pennylane as qml
import matplotlib.pyplot as plt

folder = pathlib.Path(__file__).parent

def make_imag(circuit, style):
    qml.drawer.use_style(style)

    fig, ax = qml.draw_mpl(circuit)(1.2345,1.2345)
    fig.suptitle(style, fontsize='xx-large')

    plt.savefig(folder / (style + "_style.png"))
    plt.close()
    qml.drawer.use_style('default')

if __name__ == "__main__":

    dev = qml.device('lightning.qubit', wires=(0,1,2,3))
    @qml.qnode(dev)
    def circuit(x, z):
        qml.QFT(wires=(0,1,2,3))
        qml.Toffoli(wires=(0,1,2))
        qml.CSWAP(wires=(0,2,3))
        qml.RX(x, wires=0)
        qml.CRZ(z, wires=(3,0))
        return qml.expval(qml.PauliZ(0))

    for style in qml.drawer.available_styles():
        make_imag(circuit, style)

