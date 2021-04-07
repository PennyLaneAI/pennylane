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
from .utils import to_dict, format_nvec

# Matplotlib is not a hard requirement for PennyLane in general, but it *is*
# a hard requirement for everything in this module.
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Module matplotlib is required for visualization in the Fourier module.")

from matplotlib.colors import to_rgb


def _adjust_spine_placement(ax):
    ax.xaxis.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def fourier_violin_plot(coeffs, ax, colour_dict=None, show_freqs=True):
    """Plots a list of sets of Fourier coefficients in a 1-dimensional violin plot.

    Args:
        coeffs (array[complex]): A list of sets of Fourier coefficients.
        ax (list[matplotlib.axes._subplots.AxesSubplot]): Axis on which to plot. Must
            be a pair of axes from a subplot where ``sharex="row"`` and ``sharey="col"``.
        colour_dict (dict[str, str]): A dictionary of the form {"real" : colour_string,
            "imag" : other_colour_string} indicating which colours should be used in the plot.
        show_freqs (bool): Whether or not to print the frequency labels on the plot axis.

    Returns:
        ax: The axes on which the data is plotted.

    **Example usage**

    ```
    fig, ax = plt.subplots(2, 1, figsize=(15, 4))
    fourier_violin_plot(coeffs, ax);
    ```
    """
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
    assert len(data["real"][0]) == len(nvecs)
    assert len(data["imag"][0]) == len(nvecs)

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


def fourier_bar_plot(coeffs, ax, colour_dict=None, show_freqs=True):
    """Plots a set of Fourier coefficients as a bar plot.

    Args:
        coeffs (array[complex]): A single set of Fourier coefficients.
        ax (list[matplotlib.axes._subplots.AxesSubplot]): Axis on which to plot. Must
            be a pair of axes from a subplot where ``sharex="row"`` and ``sharey="col"``.
        colour_dict (dict[str, str]): A dictionary of the form {"real" : colour_string,
            "imag" : other_colour_string} indicating which colours should be used in the plot.
        show_freqs (bool): Whether or not to print the frequency labels on the plot axis.

    Returns:
        ax: The axes on which the data is plotted.

    **Example usage**

    ```
    fig, ax = plt.subplots(2, 1, figsize=(15, 4))
    fourier_linear_violin_plot(coeffs, ax);
    ```
    """

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
        axis.bar(list(range(data_len)), data[data_type], color=colour_dict[data_type], alpha=0.7)
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


def fourier_panel_plot(coeffs, ax):
    """Plot list of sets of coefficients in the complex plane for a 1- or 2-dimensional FFT.

    Args:
        coeffs (array[complex]): A list set of Fourier coefficients. Must be
            1- or 2-dimensional.
        ax (list[matplotlib.axes._subplots.AxesSubplot]): Axis on which to plot. For
            1-dimensional data, length must be the number of frequencies. For 2-dimensional
            data, must be a grid that matches the dimensions of a single set of coefficients.

    Returns:
        ax: The axes on which the data is plotted.
    """
    if len(coeffs.shape) > 3:
        raise ValueError("Plotting not implemented for > 2-dimensional FFT outputs.")

    if ax.shape != coeffs[0].shape:
        raise ValueError("Shape of subplot axes must match the shape of the coefficient data.")

    # This could probably be more efficient.
    # Plot 1D case
    if len(coeffs.shape) == 2:
        # Range is (0, ..., degree) for rfft, (0, ... degree, -degree, ..., -1) for fft
        n_freqs = coeffs.shape[1] // 2 + (coeffs.shape[1] % 2)
        frequency_range = list(range(n_freqs)) + list(range(-n_freqs + 1, 0))

        for coeff in range(coeffs.shape[1]):
            ax[coeff].scatter(coeffs[:, coeff].real, coeffs[:, coeff].imag)
            ax[coeff].set_title(f"{frequency_range[coeff]}", fontsize=14)
            ax[coeff].grid(True)

    # Plot 2D case
    else:
        n_freqs = coeffs.shape[1] // 2 + (coeffs.shape[1] % 2)

        frequency_range = list(range(n_freqs)) + list(range(-n_freqs + 1, 0))

        for coeff_1, coeff_2 in product(list(range(coeffs.shape[1])), list(range(coeffs.shape[2]))):
            ax[coeff_1, coeff_2].scatter(
                coeffs[:, coeff_1, coeff_2].real, coeffs[:, coeff_1, coeff_2].imag
            )
            ax[coeff_1, coeff_2].set_title(
                f"{frequency_range[coeff_1]}, {frequency_range[coeff_2]}", fontsize=14
            )
            ax[coeff_1, coeff_2].grid(True)
            ax[coeff_1, coeff_2].set(aspect="equal")

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


def fourier_radial_box_plots(
    coeffs,
    n_inputs,
    degree,
    show_freqs=True,
    colour_dict=None,
    showfliers=True,
    merge_plots=False,
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
        degree (int): The max Fourier frequency obtained.
        show_freqs (bool): Whether or not to label the frequencies on
            the radial axis. Turn off for large plots.
        colour_dict (str : str): Specify a colour mapping for positive and negative
            real/imaginary components. If none specified, will default to:
            {"real" : "red", "imag" : "black"}
        showfliers (bool): Whether or not to plot outlying "fliers" on the boxplots.
        merge_plots (bool): Whether to plot real/complex values on the same panel, or
            on separate panels. Default is to plot real/complex values on separate panels.
    """

    if colour_dict is None:
        colour_dict = {"real": "red", "imag": "black"}

    # How many pie slices - need to look at a single thing
    N = coeffs[0].size

    # Width and placement of pie slices
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    width = (angles[1] - angles[0]) / 2

    # Get labels; reearrange so that the 0 is in the middle
    # (Will have to change if the extra Nyquist frequency is there)
    if n_inputs == 1:
        frequencies = list(range(-degree, degree + 1))
    else:
        frequencies = list(product(list(range(-degree, degree + 1)), repeat=n_inputs))

    # We're going to rearrange all the coefficients, of each sample
    rearranged_coeffs = np.zeros((coeffs.shape[0], N), dtype=np.complex)

    split_point = None

    for idx, frequency in enumerate(frequencies):
        if n_inputs == 1:
            if frequency == 0:
                split_point = idx
        else:
            if frequency == (0,) * n_inputs:
                split_point = idx

        for coeff_sample_idx in range(coeffs.shape[0]):
            rearranged_coeffs[coeff_sample_idx][idx] = coeffs[coeff_sample_idx][frequency]

    real_radii_distributions = []
    imag_radii_distributions = []
    # For each frequency, make a list of this coefficient from each sample
    for idx, frequency in enumerate(frequencies):
        real_radii_distributions.append(
            [rearranged_coeffs.real[x, idx] for x in range(len(coeffs))]
        )
        imag_radii_distributions.append(
            [rearranged_coeffs.imag[x, idx] for x in range(len(coeffs))]
        )

    if merge_plots:
        full_radii_distributions = list(real_radii_distributions[split_point:]) + list(
            imag_radii_distributions[split_point + 1 :][::-1]
        )

        plot_real_radii_distributions = full_radii_distributions[: split_point + 1]
        plot_imag_radii_distributions = full_radii_distributions[split_point + 1 :]

        plot_real_angles = angles[: split_point + 1]
        plot_imag_angles = angles[split_point + 1 :]
    else:
        plot_real_radii_distributions = real_radii_distributions
        plot_imag_radii_distributions = imag_radii_distributions

        rearranged_angles = np.concatenate((angles[-split_point:], angles[: split_point + 1]))
        plot_real_angles = rearranged_angles
        plot_imag_angles = rearranged_angles

    # Set up the panels
    num_subplots = 1 if merge_plots else 2
    fig, ax = plt.subplots(
        1, num_subplots, figsize=(20, 8), sharex=True, sharey=True, subplot_kw=dict(polar=True)
    )

    if merge_plots:
        avail_axes = [ax, ax]
    else:
        avail_axes = [ax[0], ax[1]]

    # Set up the violin plots
    for idx, a in enumerate(avail_axes):
        if idx == 0:
            coeff_part = "real"
            plot_angles = plot_real_angles
            plot_radii = plot_real_radii_distributions
        else:
            coeff_part = "imag"
            plot_angles = plot_imag_angles
            plot_radii = plot_imag_radii_distributions

        a.boxplot(
            plot_radii,
            positions=plot_angles,
            widths=width,
            boxprops=dict(
                facecolor=to_rgb(colour_dict[coeff_part]) + (0.4,),
                color=colour_dict[coeff_part],
                edgecolor=colour_dict[coeff_part],
            ),
            medianprops=dict(color=colour_dict[coeff_part], linewidth=1.5),
            flierprops=dict(markeredgecolor=colour_dict[coeff_part]),
            whiskerprops=dict(color=colour_dict[coeff_part]),
            capprops=dict(color=colour_dict[coeff_part]),
            patch_artist=True,
            showfliers=showfliers,
        )

        # Rotate so that the 0 frequency is at the top
        a.set_theta_zero_location("N")

    frequency_labels = [" " + format_nvec(f) + " " for f in frequencies]

    full_labels = frequencies[split_point:] + frequencies[split_point + 1 :][::-1]
    full_labels = [format_nvec(x) for x in full_labels]

    # Set and rotate the tickmarks
    if merge_plots:
        avail_axes[0].set_thetagrids((180 / np.pi) * angles, labels=full_labels)
    else:
        for a in avail_axes:
            a.set_thetagrids((180 / np.pi) * rearranged_angles, labels=frequency_labels)

    # Set and rotate the tickmarks; taken from SO
    # https://stackoverflow.com/questions/46719340/how-to-rotate-tick-labels-in-polar-matplotlib-plot
    if show_freqs:
        for a in avail_axes[:num_subplots]:
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

            # Make sure the labels are well-spaced
            a.set_xticklabels([])
            a.tick_params(pad=10)
            a.set_rlabel_position(0)

    if num_subplots == 1:
        avail_axes[0].set_title(
            "Real (left) --- Imag (right)", fontsize=16, y=-0.025 * len(full_labels[0])
        )
    else:
        avail_axes[0].set_title("Real", fontsize=16, y=-0.025 * len(full_labels[0]))
        avail_axes[1].set_title("Imag", fontsize=16, y=-0.025 * len(full_labels[0]))

    plt.tight_layout()

