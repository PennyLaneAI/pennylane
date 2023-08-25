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

import numpy as np
import pennylane as qml

from .core import VisualizationBase


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
            tr[:, k, l] = qml.math.einsum(
                "ab,ca,cb->c", t, np.conj(state), state, optimize="greedy"
            )

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


class WignerFunction(VisualizationBase):
    def __init__(self, tape):
        super().__init__(tape)

        self.cbar_axes = None
        self.num_wires = len(tape.wires)
        self.num_layers = len(qml.drawer.drawable_layers.drawable_layers(tape.operations))

    @property
    def figsize(self):
        w = self.num_layers + 3
        h = ((self.num_layers + 3) * 2) // 3

        return w, h

    def _initialize(self, fig, axs):
        axs.get_xaxis().set_ticks([])
        axs.get_yaxis().set_ticks([])

        n = 2**self.num_wires
        wigner_mesh = axs.pcolormesh(np.zeros((2 * n, 2 * n)), cmap="bwr", vmin=-1 / n, vmax=1 / n)

        # colorbar for wigner function
        cbar = fig.colorbar(wigner_mesh, cax=self.cbar_axes, location="right")
        cbar.set_ticks([-1 / n, 0, 1 / n], labels=[f"-1/{n}", 0, f"1/{n}"])
        self.cbar_axes = cbar.ax
        self.cbar_axes.tick_params()

        return (wigner_mesh,)

    def _update(self, artists, state):
        wigner_mesh = artists[0]
        fn = wigner_fn(qml.math.stack([state]))[0]
        wigner_mesh.set_array(fn)
