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

from abc import abstractmethod
from typing import Sequence, Callable

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pennylane as qml


class VisualizationBase:
    def __init__(self, tape):
        self.tape = tape
        self.artists = None

    @property
    def figsize(self):
        """
        Return the figsize of the plot

        Returns:
            tuple[float, float]
        """
        return 6.4, 4.8

    def initialize_figure(self, fig, axs):
        """
        Place artists on the figure and return them

        Returns:
            Sequence[plt.Artist]
        """
        self.artists = self._initialize(fig, axs)
        return self.artists

    @abstractmethod
    def _initialize(self, fig, axs):
        return []

    def update_figure(self, res):
        """
        Update the artists for the given result and return the updated artists

        Returns:
            Sequence[plt.Artist]
        """
        self._update(self.artists, res)
        return self.artists

    @abstractmethod
    def _update(self, artists, res):
        pass


@qml.transforms.core.transform
def plot_visualize(
    tape: qml.tape.QuantumTape, vis_class
) -> (Sequence[qml.tape.QuantumTape], Callable):
    def processing_fn(res):
        vis = vis_class(tape)

        fig, axs = plt.subplots(figsize=vis.figsize)
        vis.initialize_figure(fig, axs)
        vis.update_figure(res[0])

        return res[0]

    return [tape], processing_fn


@qml.transforms.core.transform
def plot_animate(
    tape: qml.tape.QuantumTape, vis_class, interval=1000
) -> (Sequence[qml.tape.QuantumTape], Callable):
    new_tapes = []
    for i in range(len(tape.operations) + 1):
        new_tapes.append(qml.tape.QuantumTape(tape.operations[:i], tape.measurements))

    def processing_fn(res):
        vis = vis_class(tape)

        num_wires = len(tape.wires)
        num_layers = len(qml.drawer.drawable_layers.drawable_layers(tape.operations))
        indicator_start = tape.num_preps - 1 / 2
        indicator_end = num_layers - 1 / 2

        w0, h0 = vis.figsize
        w1, h1 = num_layers + 3, num_wires + 1

        fig, axs = plt.subplots(
            2, 1, figsize=(max(w0, w1), h0 + h1), height_ratios=[h0 / (h0 + h1), h1 / (h0 + h1)]
        )

        # indicator line to show current point in execution
        indicator_line = plt.Line2D((), ())

        def init_fn():
            artists = vis.initialize_figure(fig, axs[0])

            qml.drawer.tape_mpl(tape, fig_and_ax=(fig, axs[1]))
            axs[1].add_line(indicator_line)

            return (*artists, indicator_line)

        def update_fn(index):
            artists = vis.update_figure(res[index])
            x_pos = indicator_start + (indicator_end - indicator_start) * index / num_layers
            indicator_line.set_data((x_pos, x_pos), (-1, num_wires))
            return (*artists, indicator_line)

        ani = FuncAnimation(
            fig,
            update_fn,
            len(res),
            init_fn,
            interval=interval,
            blit=True,
        )

        return ani

    return new_tapes, processing_fn
