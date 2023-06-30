# Copyright 2023 Xanadu Quantum Technologies Inc.

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
TODO: docstring
"""

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

from .drawable_layers import drawable_layers
from .tape_mpl import tape_mpl
from .state_animation import StateAnimation


def u_op(m, n):
    u = np.zeros((n, n))
    i = np.arange(n)
    j = np.mod(-m + i, n)
    u[i, j] = 1
    return u


def v_op(m, n):
    return np.diag(np.exp(2j * np.pi * m * np.arange(n) / n))


def traces(state):
    n = state.shape[-1]
    tr = np.zeros((state.shape[0], 2 * n, 2 * n), dtype=np.complex128)
    for k in range(2 * n):
        u = u_op(k, n)
        for l in range(2 * n):
            t = u @ v_op(l, n) * np.exp(1j * np.pi * k * l / n)
            tr[:, k, l] = qml.math.einsum('ab,ca,cb->c', t, np.conj(state), state, optimize='greedy')

    return tr


def wigner_fn(state):
    n = state.shape[-1]
    w = np.zeros((state.shape[0], 2 * n, 2 * n))
    tr = traces(state)
    k, l = np.arange(2 * n)[None, :], np.arange(2 * n)[:, None]
    for q in range(2 * n):
        for p in range(2 * n):
            phase = np.exp(-1j * np.pi * (k * p - l * q) / n)
            w[:, q, p] = np.real(np.sum(tr * phase, axis=(1, 2)))

    return w / ((2 * n) ** 2)


class WignerAnimation(StateAnimation):

    def __init__(self, states, tape, fontsize=22):
        super().__init__(qml.math.stack(states), tape)

        self.wigner_mesh = None
        self.indicator_line = None
        self.cbar_axes = None

        self.num_wires = int(np.floor(np.log2(self.states.shape[-1]) + 1 / 2))
        self.num_layers = len(drawable_layers(self.tape.operations))
        self.indicator_start = len(self.tape._prep) - 1 / 2
        self.indicator_end = self.num_layers - 1 / 2

        self.fontsize = fontsize

        self.fns = wigner_fn(self.states)

    def create_figure(self):
        width = self.num_layers + 3
        ax1_height = ((self.num_layers + 3) * 2) // 3
        ax2_height = self.num_wires + 1
        height = ax1_height + ax2_height

        fig, axs = plt.subplots(
            2,
            1,
            figsize=(width, height),
            height_ratios=[ax1_height / height, ax2_height / height]
        )
        fig.tight_layout()

        axs[0].get_xaxis().set_ticks([])
        axs[0].get_yaxis().set_ticks([])

        return fig, axs

    def initialize(self, fig, axs):
        # draw the circuit on the bottom plot
        tape_mpl(self.tape, fig_and_ax=(fig, axs[1]), fontsize=self.fontsize)

        # draw the wigner function on the top plot
        n = self.states.shape[-1]
        self.wigner_mesh = axs[0].pcolormesh(np.zeros((2 * n, 2 * n)), cmap='bwr', vmin=-1/n, vmax=1/n)

        # colorbar for wigner function
        cbar = fig.colorbar(self.wigner_mesh, cax=self.cbar_axes, location='right')
        cbar.set_ticks([-1/n, 0, 1/n], labels=[f"-1/{n}", 0, f"1/{n}"])
        self.cbar_axes = cbar.ax
        self.cbar_axes.tick_params(labelsize=self.fontsize)

        # indicator line to show current point in execution
        self.indicator_line = plt.Line2D((), ())
        axs[1].add_line(self.indicator_line)

        return self.wigner_mesh, self.indicator_line

    def update(self, idx):
        # position of the indicator line
        x_pos = self.indicator_start + (self.indicator_end - self.indicator_start) * (idx / (len(self.states) - 1))

        self.wigner_mesh.set_array(self.fns[idx])
        self.indicator_line.set_data((x_pos, x_pos), (-1, self.num_wires))

        return self.wigner_mesh, self.indicator_line
