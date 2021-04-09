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
"""Contains visualization functions for Fourier series and coefficients."""
from itertools import product
import numpy as np

# Matplotlib is not a hard requirement for PennyLane in general, but it *is*
# a hard requirement for everything in this module.
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from .utils import to_dict, format_nvec


def _adjust_spine_placement(ax):
    ax.xaxis.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def fourier_violin_plot(coeffs, n_inputs, ax, colour_dict=None, show_freqs=True):
    """Plots a list of sets of Fourier coefficients as a violin plot.

    Args:
        coeffs (array[complex]): A list of sets of Fourier coefficients.
        n_inputs (int): The number of input variables in the function.
        ax (list[matplotlib.axes._subplots.AxesSubplot]): Axis on which to plot. Must
            be a pair of axes from a subplot where ``sharex="row"`` and ``sharey="col"``.
        colour_dict (dict[str, str]): A dictionary of the form {"real" : colour_string,
            "imag" : other_colour_string} indicating which colours should be used in the plot.
        show_freqs (bool): Whether or not to print the frequency labels on the plot axis.

    Returns:
        ax: The axes on which the data is plotted.
    """
    # Check dimensionality; it's possible a user has provided only a single set
    # of coefficients to this function.
    if len(coeffs.shape) == n_inputs:
        coeffs = np.array([coeffs])

    # The axis received must be a pair of axes in a subplot.
    if colour_dict is None:
        colour_dict = {"real": "purple", "imag": "green"}

    # extract the x ticks
    nvecs = list(to_dict(coeffs[0]).keys())
    nvecs_formatted = [format_nvec(nvec) for nvec in nvecs]

    # make data
    data = {}
    data["real"] = np.array([[c[nvec].real for nvec in nvecs] for c in coeffs])
    data["imag"] = np.array([[c[nvec].imag for nvec in nvecs] for c in coeffs])

    for (data_type, axis) in zip(["real", "imag"], ax):
        violin = axis.violinplot(data[data_type], showextrema=False)
        for bd in violin["bodies"]:
            bd.set_color(colour_dict[data_type])
            bd.set_alpha(0.7)
        axis.set_ylabel(data_type)
        axis.xaxis.set_ticks(np.arange(1, len(data[data_type][0]) + 1))
        _adjust_spine_placement(axis)

    # Format axes
    ax[0].tick_params(axis="x", colors="white")  # hack to get rid of ticks but keep grid

    if show_freqs:
        ax[1].tick_params(axis="x", which="both", length=0)  # remove ticks without removing labels
        ax[1].xaxis.set_ticklabels(nvecs_formatted, fontsize=10, color="grey")
        ax[1].xaxis.set_ticks_position("top")
    else:
        ax[1].tick_params(axis="x", colors="white")  # hack to get rid of ticks but keep grid

    return ax


def fourier_box_plot(coeffs, n_inputs, ax, colour_dict=None, show_freqs=True, show_fliers=True):
    """Plots a set of Fourier coefficients as a box plot.

    Args:
        coeffs (array[complex]): A single set of Fourier coefficients.
        n_inputs (int): The number of input variables in the function.
        ax (list[matplotlib.axes._subplots.AxesSubplot]): Axis on which to plot. Must
            be a pair of axes from a subplot where ``sharex="row"`` and ``sharey="col"``.
        colour_dict (dict[str, str]): A dictionary of the form {"real" : colour_string,
            "imag" : other_colour_string} indicating which colours should be used in the plot.
        show_freqs (bool): Whether or not to print the frequency labels on the plot axis.
        show_fliers (bool): Whether to display the box plot outliers.

    Returns:
        ax: The axes on which the data is plotted.
    """
    # Check dimensionality; it's possible a user has provided only a single set
    # of coefficients to this function.
    if len(coeffs.shape) == n_inputs:
        coeffs = np.array([coeffs])

    # The axis received must be a pair of axes in a subplot.
    if colour_dict is None:
        colour_dict = {"real": "purple", "imag": "green"}

    # extract the x ticks
    nvecs = list(to_dict(coeffs[0]).keys())
    nvecs_formatted = [format_nvec(nvec) for nvec in nvecs]

    # Make data
    data = {}
    data["real"] = np.array([[c[nvec].real for nvec in nvecs] for c in coeffs])
    data["imag"] = np.array([[c[nvec].imag for nvec in nvecs] for c in coeffs])

    for (data_type, axis) in zip(["real", "imag"], ax):
        data_colour = colour_dict[data_type]
        axis.boxplot(
            data[data_type],
            boxprops=dict(
                facecolor=to_rgb(data_colour) + (0.4,), color=data_colour, edgecolor=data_colour
            ),
            medianprops=dict(color=data_colour, linewidth=1.5),
            flierprops=dict(markeredgecolor=data_colour),
            whiskerprops=dict(color=data_colour),
            capprops=dict(color=data_colour),
            patch_artist=True,
            showfliers=show_fliers,
        )

        _adjust_spine_placement(axis)
        axis.set_ylabel(data_type)
        axis.xaxis.set_ticks(np.arange(1, len(nvecs) + 1))

    ax[0].tick_params(axis="x", colors="white")  # hack to get rid of ticks but keep grid

    if show_freqs:
        ax[1].tick_params(axis="x", which="both", length=0)  # remove ticks without removing labels
        ax[1].xaxis.set_ticklabels(nvecs_formatted, fontsize=10, color="grey")
        ax[1].xaxis.set_ticks_position("top")
    else:
        ax[1].tick_params(axis="x", colors="white")  # hack to get rid of ticks but keep grid

    return ax


def fourier_bar_plot(coeffs, n_inputs, ax, colour_dict=None, show_freqs=True):
    """Plots a set of Fourier coefficients as a bar plot.

    Args:
        coeffs (array[complex]): A single set of Fourier coefficients.
        n_inputs (int): The number of input variables in the function.
        ax (list[matplotlib.axes._subplots.AxesSubplot]): Axis on which to plot. Must
            be a pair of axes from a subplot where ``sharex="row"`` and ``sharey="col"``.
        colour_dict (dict[str, str]): A dictionary of the form {"real" : colour_string,
            "imag" : other_colour_string} indicating which colours should be used in the plot.
        show_freqs (bool): Whether or not to print the frequency labels on the plot axis.

    Returns:
        ax: The axes on which the data is plotted.
    """
    # This function plots only a single set of coefficients, so dimensions must match
    if len(coeffs.shape) != n_inputs:
        raise ValueError("Function fourier_bar_plot accepts only one set of Fourier coefficients.")

    # The axis received must be a pair of axes in a subplot.
    if colour_dict is None:
        colour_dict = {"real": "purple", "imag": "green"}

    # extract the x ticks
    nvecs = list(to_dict(coeffs).keys())
    nvecs_formatted = [format_nvec(nvec) for nvec in nvecs]

    # make data
    data = {}
    data["real"] = np.array([coeffs[nvec].real for nvec in nvecs])
    data["imag"] = np.array([coeffs[nvec].imag for nvec in nvecs])
    data_len = len(data["real"])

    for (data_type, axis) in zip(["real", "imag"], ax):
        axis.bar(np.arange(data_len), data[data_type], color=colour_dict[data_type], alpha=0.7)
        axis.set_ylabel(data_type)
        axis.xaxis.set_ticks(np.arange(data_len))
        _adjust_spine_placement(axis)

    ax[0].tick_params(axis="x", colors="white")  # hack to get rid of ticklabels but keep grid

    if show_freqs:
        ax[1].tick_params(axis="x", which="both", length=0)  # remove ticks without removing labels
        ax[1].xaxis.set_ticklabels(nvecs_formatted, fontsize=10, color="grey")
        ax[1].xaxis.set_ticks_position("top")
    else:
        ax[1].tick_params(axis="x", colors="white")  # hack to get rid of ticklabels but keep grid

    return ax


def fourier_panel_plot(coeffs, n_inputs, ax, colour=None):
    """Plot list of sets of coefficients in the complex plane for a 1- or 2-dimensional function.

    Args:
        coeffs (array[complex]): A list set of Fourier coefficients. Must be
            1- or 2-dimensional.
        n_inputs (int): The number of variables in the function.
        ax (list[matplotlib.axes._subplots.AxesSubplot]): Axis on which to plot. For
            1-dimensional data, length must be the number of frequencies. For 2-dimensional
            data, must be a grid that matches the dimensions of a single set of coefficients.
        colour (str): The outline colour of the points on the plot.

    Returns:
        ax: The axes on which the data is plotted.
    """
    if colour is None:
        colour = "tab:blue"

    # In case a single set of coefficients is sent
    if len(coeffs.shape) == n_inputs:
        coeffs = np.array([coeffs])

    if n_inputs > 2:
        raise ValueError("Plotting not implemented for > 2-dimensional FFT outputs.")

    if ax.shape != coeffs[0].shape:
        raise ValueError("Shape of subplot axes must match the shape of the coefficient data.")

    # This could probably be more efficient.
    # Plot 1D case
    if n_inputs == 1:
        # Range is (0, ..., degree) for rfft, (0, ... degree, -degree, ..., -1) for fft
        n_freqs = coeffs.shape[1] // 2 + (coeffs.shape[1] % 2)
        frequency_range = list(range(n_freqs)) + list(range(-n_freqs + 1, 0))

        for coeff in range(coeffs.shape[1]):
            ax[coeff].scatter(
                coeffs[:, coeff].real, coeffs[:, coeff].imag, facecolor="white", edgecolor=colour
            )
            ax[coeff].set_title(f"{frequency_range[coeff]}", fontsize=14)
            ax[coeff].grid(True)
            ax[coeff].set_aspect("equal")

    # Plot 2D case
    else:
        n_freqs = coeffs.shape[1] // 2 + (coeffs.shape[1] % 2)

        frequency_range = list(range(n_freqs)) + list(range(-n_freqs + 1, 0))

        for coeff_1, coeff_2 in product(list(range(coeffs.shape[1])), list(range(coeffs.shape[2]))):
            ax[coeff_1, coeff_2].scatter(
                coeffs[:, coeff_1, coeff_2].real,
                coeffs[:, coeff_1, coeff_2].imag,
                facecolor="white",
                edgecolor=colour,
            )
            ax[coeff_1, coeff_2].set_title(
                f"{frequency_range[coeff_1]}, {frequency_range[coeff_2]}", fontsize=14
            )
            ax[coeff_1, coeff_2].grid(True)
            ax[coeff_1, coeff_2].set_aspect("equal")

    return ax


def fourier_reconstruct_function_1D_plot(coeffs, ax=None):
    """Visualize a 1D periodic function given by a set of Fourier coefficients.

    Args:
        coeffs (np.ndarray): Fourier coefficients of a 1-dimensional function.
        ax (matplotlib.axes._subplots.AxesSubplot): Axis on which to plot. If None, the
            current axis from ``plt.gca()`` will be used.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: The axis after plotting is complete.
    """

    ax = ax or plt.gca()

    if len(coeffs.shape) != 1:
        raise ValueError("Function fourier_reconstruct_function_1D_plot takes 1-dimensional input.")

    def reconstructed_function(x):
        # At each point x, the value of the function is given by
        # c_0 + c_1 e^{ix} + c_2 e^{2ix} + ... + c_{-n} e^{-inx} + ... c_{-1} e^{-ix}

        n_freqs = len(coeffs) // 2 + (len(coeffs) % 2)

        function_value = 0

        for freq in range(n_freqs):
            # Ignore tiny coefficients
            if np.isclose(coeffs[freq], 0):
                continue

            if freq == 0:
                function_value += coeffs[freq]
            else:
                function_value += coeffs[freq] * np.exp(freq * 1j * x)
                function_value += np.conj(coeffs[freq]) * np.exp(-freq * 1j * x)

        return function_value

    n_points = 500
    grid_range = np.linspace(-np.pi, np.pi, n_points)

    ax.plot(grid_range, reconstructed_function(grid_range))

    return ax


def fourier_reconstruct_function_2D_plot(coeffs, ax=None):
    """Visualize a 2D periodic function given by a set of Fourier coefficients.

    Args:
        coeffs (np.ndarray): Fourier coefficients of a 2-dimensional function.
        ax (matplotlib.axes._subplots.AxesSubplot): Axis on which to plot. If None, the
            current axis from ``plt.gca()`` will be used.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: The axis after plotting is complete.
    """
    ax = ax or plt.gca()

    if len(coeffs.shape) != 2:
        raise ValueError("Function fourier_reconstruct_function_2D_plot takes 2-dimensional input.")

    def reconstructed_function(x):
        # At each point x, the value of the function is given by
        # c_0 + c_1 e^{ix} + c_2 e^{2ix} + ... + c_{-n} e^{-inx} + ... c_{-1} e^{-ix}

        n_freqs = coeffs.shape[1] // 2 + (coeffs.shape[1] % 2)

        function_value = 0j

        # Loop through only positive frequencies because the negative ones we get through complex conj.
        for freq_1, freq_2 in product(list(range(n_freqs)), list(range(n_freqs))):
            # Ignore tiny coefficients
            if np.isclose(coeffs[freq_1, freq_2], 0):
                continue

            if freq_1 == 0 and freq_2 == 0:
                function_value += coeffs[freq_1, freq_2]
            else:
                function_value += (
                    coeffs[freq_1, freq_2] * np.exp(freq_1 * 1j * x[0]) * np.exp(freq_2 * 1j * x[1])
                )
                function_value += (
                    np.conj(coeffs[freq_1, freq_2])
                    * np.exp(-freq_1 * 1j * x[0])
                    * np.exp(-freq_2 * 1j * x[1])
                )

        return function_value

    n_points = 50
    grid_range = np.linspace(-np.pi, np.pi, n_points)

    function_values = np.zeros((n_points, n_points))

    # Fill each point in the grid with the value of model evaluated at that point
    # Take only the real part because our function is real-valued
    for index in np.ndindex(*((n_points,) * 2)):
        coordinate_vector = [grid_range[i] for i in index]
        function_values[index] = reconstructed_function(coordinate_vector).real

    ax.imshow(function_values)

    return ax


def fourier_radial_box_plot(
    coeffs, n_inputs, ax, show_freqs=True, colour_dict=None, show_fliers=True
):
    """Plot distributions of Fourier coefficients on a radial plot as box plots.

    Produces either 2-panel plot in which the left panel represents the real
    parts of Fourier coefficients, and the right the imaginary parts, or a
    1-panel plot in which half the panel shows the real part and the other half
    shows the complex. This method accepts multiple sets of coefficients, and
    plots the distribution of each coefficient as a boxplot.

    Args:
        coeffs (np.ndarray): A set of Fourier coefficients. Assumed to be
            from a full fft transform.
        n_inputs (int): Dimension of the transformed function.
        show_freqs (bool): Whether or not to label the frequencies on
            the radial axis. Turn off for large plots.
        colour_dict (str : str): Specify a colour mapping for positive and negative
            real/imaginary components. If none specified, will default to:
            {"real" : "red", "imag" : "black"}
        showfliers (bool): Whether or not to plot outlying "fliers" on the boxplots.
        merge_plots (bool): Whether to plot real/complex values on the same panel, or
            on separate panels. Default is to plot real/complex values on separate panels.
    """
    # Take care of single-input case
    if len(coeffs.shape) == n_inputs:
        coeffs = np.array([coeffs])

    if colour_dict is None:
        colour_dict = {"real": "red", "imag": "black"}

    # Number, width, and placement of pie slices
    N = coeffs[0].size
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles[-N // 2 + 1 :], angles[: -N // 2 + 1]))[::-1]
    width = (angles[1] - angles[0]) / 2

    # Extract the radial ticks
    nvecs = list(to_dict(coeffs[0]).keys())
    nvecs_formatted = [format_nvec(nvec) for nvec in nvecs]

    # Make data
    data = {}
    data["real"] = np.array([[c[nvec].real for nvec in nvecs] for c in coeffs])
    data["imag"] = np.array([[c[nvec].imag for nvec in nvecs] for c in coeffs])

    # Set up the violin plots
    for data_type, a in zip(["real", "imag"], ax):
        data_colour = colour_dict[data_type]

        a.boxplot(
            data[data_type],
            positions=angles,
            widths=width,
            boxprops=dict(
                facecolor=to_rgb(data_colour) + (0.4,), color=data_colour, edgecolor=data_colour
            ),
            medianprops=dict(color=data_colour, linewidth=1.5),
            flierprops=dict(markeredgecolor=data_colour),
            whiskerprops=dict(color=data_colour),
            capprops=dict(color=data_colour),
            patch_artist=True,
            showfliers=show_fliers,
        )

        # Rotate so that the 0 frequency is at the to
        a.set_thetagrids((180 / np.pi) * angles, labels=nvecs_formatted)
        a.set_theta_zero_location("N")
        a.set_rlabel_position(0)

    # Set and rotate the tickmarks; taken from SO
    # https://stackoverflow.com/questions/46719340/how-to-rotate-tick-labels-in-polar-matplotlib-plot
    for a in ax:
        if show_freqs:
            for label, angle in zip(a.get_xticklabels(), angles):
                x, y = label.get_position()
                lab = a.text(
                    x,
                    y,
                    label.get_text(),
                    transform=label.get_transform(),
                    ha=label.get_ha(),
                    va=label.get_va(),
                    fontsize=14,
                    color="grey",
                )
                if angle > np.pi:
                    lab.set_rotation((180 / np.pi) * angle + 90)
                else:
                    lab.set_rotation((180 / np.pi) * angle + 270)

            a.tick_params(pad=10)

        a.set_xticklabels([])

    return ax
