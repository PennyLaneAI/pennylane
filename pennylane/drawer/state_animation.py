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
import functools

import numpy as np
import scipy as sp
from matplotlib.animation import FuncAnimation

import pennylane as qml


class StateAnimation:

    def __init__(self, states, tape):
        self.states = states
        self.tape = tape

    def create_figure(self):
        """
        Create a Figure and a (sequence of) Axes objects

        Returns:
            plt.Figure, plt.Axes
        """
        raise NotImplementedError

    def initialize(self, fig, axs):
        """
        Place artists on the first frame of the animation, and return them

        Returns:
            Sequence[plt.Artist]
        """
        raise NotImplementedError

    def update(self, idx):
        """
        Update the artists for the frame corresponding to the given index,
            and return the updated artists

        Returns:
            Sequence[plt.Artist]
        """
        raise NotImplementedError


def _split_operator(op, num=None):
    """
    Split the operator into num identical operators which simulates time-evolution.

    This function currently only works for a handful of ops
    """
    if num is None:
        return [op]

    if isinstance(op, qml.ControlledPhaseShift):
        return [qml.ControlledPhaseShift(op.data[0] / num, wires=op.wires) for _ in range(num)]

    if isinstance(op, qml.Hadamard):
        gen = np.array([[1 - 1 / np.sqrt(2), -1 / np.sqrt(2)], [-1 / np.sqrt(2), 1 + 1 / np.sqrt(2)]])
        mat = sp.linalg.expm(1j * np.pi / 2 * gen / num)
        return [qml.QubitUnitary(mat, wires=op.wires) for _ in range(num)]

    if isinstance(op, qml.CNOT):
        gen = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, -1], [0, 0, -1, 1]])
        mat = sp.linalg.expm(1j * np.pi / 2 * gen  / num)
        return [qml.QubitUnitary(mat, wires=op.wires) for _ in range(num)]

    raise NotImplementedError


def _frame_generator(num_frames, padding):
    if padding is None:
        return range(num_frames)

    for _ in range(padding):
        yield 0

    for i in range(num_frames):
        yield i

    for _ in range(padding):
        yield num_frames - 1


def _animate(ani_class, states, tape, interval=50, padding=None, save_path=None, **kwargs):

    state_animation = ani_class(states, tape, **kwargs)
    fig, axs = state_animation.create_figure()

    ani = FuncAnimation(
        fig,
        state_animation.update,
        _frame_generator(len(states), padding),
        functools.partial(state_animation.initialize, fig=fig, axs=axs),
        interval=interval,
        save_count=None if padding is None else len(states) + 2 * padding,
        blit=True,
    )

    if save_path:
        ani.save(save_path, bitrate=2**16)

    return ani


def animate(qnode, ani_class, num=None, **ani_kwargs):

    @functools.wraps(qnode)
    def wrapper(*args, **kwargs):
        qnode.construct(args, kwargs)

        # add snapshots for all states
        new_ops = [qml.Snapshot()]
        for op in qnode.tape._ops:
            split_ops = _split_operator(op, num=num)
            for split_op in split_ops:
                new_ops.extend([split_op, qml.Snapshot()])

        new_tape = qml.tape.QuantumScript(new_ops, qnode.tape.measurements, qnode.tape._prep)

        with qml.debugging._Debugger(qnode.device) as dbg:
            res = qml.execute([new_tape], device=qnode.device, gradient_fn=None)

        return _animate(ani_class, list(dbg.snapshots.values()), qnode.tape, **ani_kwargs)

    return wrapper
