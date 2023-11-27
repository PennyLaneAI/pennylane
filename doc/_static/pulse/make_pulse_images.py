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
pulse programming module.
"""

import pathlib
import matplotlib.pyplot as plt
from jax import numpy as jnp

import pennylane as qml

folder = pathlib.Path(__file__).parent


def parametrized_coefficients_example(savefile="parametrized_coefficients_example.png"):
    def f1(p, t):
        return jnp.sin(p[0] * t**2) + p[1]

    def f2(p, t):
        return p * jnp.cos(t)

    H = 2 * qml.PauliX(0) + f1 * qml.PauliY(0) + f2 * qml.PauliZ(0)

    times = jnp.linspace(0., 5., 1000)
    fs = H.coeffs_parametrized
    ops = H.ops_parametrized
    params = [[4.6, 2.3], 1.2]

    fig, axs = plt.subplots(nrows=len(ops))
    for n, f in enumerate(fs):
        ax = axs[n]
        ax.plot(times, f(params[n], times), label=f"p={params[n]}")
        ax.set_ylabel(f"f{n}")
        ax.legend(loc="upper left")

    ax.set_xlabel("Time")
    axs[0].set_title(f"H parametrized coefficients")
    plt.tight_layout()

    plt.savefig(folder / savefile)
    plt.close()


def pwc_example(savefile="pwc_example.png"):
    params = jnp.array([1, 2, 3, 4, 5])
    time = jnp.linspace(0, 10, 1000)
    y = qml.pulse.pwc(timespan=(2, 7))(params, time)

    plt.plot(time, y, label=f"params = {params}, \n timespan = (2, 7)")
    plt.legend(loc="upper left")

    plt.savefig(folder / savefile)
    plt.close()


def rect_example(savefile="rect_example.png"):
    def f(p, t):
        return jnp.polyval(p, t)

    p = jnp.array([1, 2, 3])
    time = jnp.linspace(0, 10, 1000)

    y1 = f(p, time)
    y2 = [qml.pulse.rect(f, windows=[(1, 7)])(p, t) for t in time]

    plt.plot(time, y1, label=f"polyval, P = {p}")
    plt.plot(time, y2, label="rect(polyval), windows = [(1, 7)]")
    plt.legend()

    plt.savefig(folder / savefile)
    plt.close()


if __name__ == "__main__":
    parametrized_coefficients_example()
    pwc_example()
    rect_example()

