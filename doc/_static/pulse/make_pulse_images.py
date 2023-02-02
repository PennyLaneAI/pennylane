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
from jax import numpy as jnp

import pennylane as qml

folder = pathlib.Path(__file__).parent


def parametrized_coefficients_example(savefile="parametrized_coefficients_example.png"):

    # defining the coefficients fj(v, t) for the two parametrized terms
    f1 = lambda p, t: p * jnp.sin(t) * (t - 1)
    f2 = lambda p, t: p[0] * jnp.cos(p[1] * t ** 2)

    # defining the operations for the three terms in the Hamiltonian
    XX = qml.PauliX(0) @ qml.PauliX(1)
    YY = qml.PauliY(0) @ qml.PauliY(1)
    ZZ = qml.PauliZ(0) @ qml.PauliZ(1)

    H1 = 2 * XX + f1 * YY + f2 * ZZ

    times = jnp.linspace(0., 5., 1000)
    fs = H1.coeffs_parametrized
    ops = H1.ops_parametrized
    params = [1.2, [2.3, 3.4]]

    fig, axs = plt.subplots(nrows=len(ops))
    for n, f in enumerate(fs):
        ax = axs[n]
        ax.plot(times, f(params[n], times), label=f"p={params[n]}")
        ax.set_ylabel(f"{ops[n].label()}")
        ax.legend(loc="upper left")

    ax.set_xlabel("time")
    axs[0].set_title(f"H1 parametrized coefficients")
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

