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
This file contains functions that visualize properties of certain kernels.
"""
import numpy as np


def two_dim_kernel(kernel, ax, evaluate_at=None, x=None):
    """Plot a real-valued kernel defined on one-dimensional data."""

    if x is None:
        x = np.linspace(-10, 10, 50)

    if evaluate_at is None:
        evaluate_at = np.array([0., 0.])

    xx, yy = np.meshgrid(x, x)
    Z = [kernel([[x, y], evaluate_at]) for x, y in zip(xx.flatten(), yy.flatten())]
    Z = np.reshape(Z, xx.shape)

    surf = ax.plot_surface(xx, yy, Z, cmap='viridis', linewidth=0, antialiased=False)

    ax.colorbar(surf, shrink=0.5, aspect=10)
    return ax


def plot_fourier_transform(kernel):

    # create fourier trafo
    # plot
    return 0
