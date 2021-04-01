# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from .utils import to_dict, format_nvec


def violin_plot(list_of_coeffs, figsize=None):
    """Plots a list of sets of Fourier coefficients in a 1-dimensional plot."""

    # extract the x ticks
    nvecs = list(to_dict(list_of_coeffs[0]).keys())
    nvecs_formatted = [format_nvec(nvec) for nvec in nvecs]

    # make data
    data_real = np.array([[coeffs[nvec].real for nvec in nvecs] for coeffs in list_of_coeffs])
    data_imag = np.array([[coeffs[nvec].imag for nvec in nvecs] for coeffs in list_of_coeffs])
    assert len(data_real[0]) == len(nvecs)
    assert len(data_imag[0]) == len(nvecs)

    if figsize is None:
        f, (ax1, ax2) = plt.subplots(
            2, 1, sharex="row", sharey="col", figsize=(int(len(nvecs) / 10) + 4, 4)
        )
    else:
        f, (ax1, ax2) = plt.subplots(2, 1, sharex="row", sharey="col", figsize=figsize)

    violin1 = ax1.violinplot(data_real, showextrema=False)
    for bd in violin1["bodies"]:
        bd.set_color("purple")
        bd.set_alpha(0.7)
    ax1.set_ylabel("real")
    ax1.xaxis.set_ticks(np.arange(1, len(nvecs) + 1))
    ax1.xaxis.grid()
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="x", colors="white")  # hack to get rid of ticks but keep grid
    ax1.set_axisbelow(True)

    violin2 = ax2.violinplot(data_imag, showextrema=False)
    for bd in violin2["bodies"]:
        bd.set_color("green")
        bd.set_alpha(0.7)
    ax2.set_ylabel("imag")
    ax2.xaxis.grid()
    ax2.xaxis.set_ticks(np.arange(1, len(nvecs) + 1))
    ax2.xaxis.set_ticklabels(nvecs_formatted, fontsize=6, color="grey")
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="x", which="both", length=0)  # remove ticks without removing labels
    ax2.xaxis.set_ticks_position("top")

    plt.subplots_adjust(hspace=0.0)
    plt.xticks(rotation="vertical")
    plt.tight_layout()

    return plt


def bar_plot(coeffs, show_freq=True, figsize=None):
    """Plots a list of sets of Fourier coefficients in a 1-dimensional plot."""

    # extract the x ticks
    nvecs = list(to_dict(coeffs).keys())
    nvecs_formatted = [format_nvec(nvec) for nvec in nvecs]

    # make data
    coeffs_real = np.array([coeffs[nvec].real for nvec in nvecs])
    coeffs_imag = np.array([coeffs[nvec].imag for nvec in nvecs])

    if figsize is None:
        plt.figure(figsize=(int(len(nvecs) / 10) + 4, 4))
    else:
        plt.figure(figsize=figsize)

    ax1 = plt.subplot(2, 1, 1)
    ax1.bar(np.arange(len(coeffs_real)), coeffs_real, color="purple", alpha=0.7)
    ax1.set_ylabel("real")
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="x", colors="white")  # hack to get rid of ticklabels but keep grid
    ax1.xaxis.set_ticks(np.arange(len(nvecs)))
    ax1.xaxis.grid()
    ax1.set_axisbelow(True)

    ax2 = plt.subplot(2, 1, 2)
    ax2.bar(np.arange(len(coeffs_imag)), coeffs_imag, color="green", alpha=0.6)
    ax2.set_ylabel("imag")
    ax2.xaxis.set_ticklabels(nvecs_formatted, fontsize=6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="x", which="both", length=0)  # remove ticks without removing labels
    ax2.xaxis.set_ticks(np.arange(len(nvecs)))
    ax2.xaxis.grid()
    ax2.set_axisbelow(True)

    if not show_freq:
        ax2.tick_params(axis="x", colors="white")  # hack to get rid of ticklabels but keep grid

    plt.xticks(rotation="vertical")
    plt.tight_layout()
    return plt


def complex_panel_plot(coeffs, real_fft=False, savefig=False, title=None):
    """Plot coefficients in the complex plane for a 1- or 2-dimensional FFT.

    Args:
        coeffs (np.ndarray): Fourier coefficients.
        real_fft (bool): Whether the coefficients are the result of rfftn instead of fftn.
            Affects only plot size and axis labelling.
        savefig (bool): Whether or not to save the figure.
        title (str): Title for the file of saved figure.
    """
    if len(coeffs.shape) > 3:
        raise ValueError("Plotting not implemented for > 2-dimensional FFT outputs.")

    # This could probably be more efficient.
    # Plot 1D case
    if len(coeffs.shape) == 2:
        # Range is (0, ..., degree) for rfft, (0, ... degree, -degree, ..., -1) for fft
        if real_fft:
            n_freqs = coeffs.shape[1]
            frequency_range = list(range(n_freqs))
        else:
            n_freqs = coeffs.shape[1] // 2 + (coeffs.shape[1] % 2)
            frequency_range = list(range(n_freqs)) + list(range(-n_freqs + 1, 0))

        # Set up the grid
        fig, ax = plt.subplots(
            1, coeffs.shape[1], sharex=True, sharey=True, figsize=(3 * coeffs.shape[1], 3)
        )

        for coeff in range(coeffs.shape[1]):
            ax[coeff].scatter(coeffs[:, coeff].real, coeffs[:, coeff].imag)
            ax[coeff].set_title(f"{frequency_range[coeff]}", fontsize=14)
            ax[coeff].grid(True)

    # Plot 2D case
    else:
        if real_fft:
            n_freqs = coeffs.shape[2]
        else:
            n_freqs = coeffs.shape[1] // 2 + (coeffs.shape[1] % 2)

        frequency_range = list(range(n_freqs)) + list(range(-n_freqs + 1, 0))

        # Set up the grid
        fig, ax = plt.subplots(
            coeffs.shape[1],
            coeffs.shape[2],
            sharex=True,
            sharey=True,
            figsize=(3 * coeffs.shape[1], 3 * coeffs.shape[2]),
        )

        for coeff_1, coeff_2 in product(list(range(coeffs.shape[1])), list(range(coeffs.shape[2]))):
            ax[coeff_1, coeff_2].scatter(
                coeffs[:, coeff_1, coeff_2].real, coeffs[:, coeff_1, coeff_2].imag
            )
            ax[coeff_1, coeff_2].set_title(
                f"{frequency_range[coeff_1]}, {frequency_range[coeff_2]}", fontsize=14
            )
            ax[coeff_1, coeff_2].grid(True)
            ax[coeff_1, coeff_2].set(aspect="equal", adjustable="box")

    # Sort out title and saving; autogenerate a title if one not provided
    if title is None:
        title = f"Fourier coefficients, first {coeffs.shape[1] - 1} frequencies."

    fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if savefig:
        plt.savefig(title + ".pdf")
    else:
        plt.show()


def fourier_reconstruct_function_1D_plot(coeffs, real_fft=False, savefig=False, title=None):
    """Visualize the 1D periodic function given by the Fourier coefficients.

    Args:
        coeffs (np.ndarray): Fourier coefficients.
        real_fft (bool): Whether the coefficients are the result of rfftn instead of fftn.
        savefig (bool): Whether or not to save the figure.
        title (str): Title for the file of saved figure.
    """

    if len(coeffs.shape) != 1:
        raise ValueError("Function fourier_reconstruct_function_1D_plot takes 1-dimensional input.")

    def reconstructed_function(x):
        # At each point x, the value of the function is given by
        # c_0 + c_1 e^{ix} + c_2 e^{2ix} + ... + c_{-n} e^{-inx} + ... c_{-1} e^{-ix}

        if real_fft:
            n_freqs = len(coeffs)
        else:
            n_freqs = len(coeffs) // 2 + (len(coeffs) % 2)

        function_value = 0
        # Start with positive coefficients
        for freq in range(n_freqs):
            if freq == 0:
                function_value += coeffs[freq]
            else:
                if np.abs(coeffs[freq]) > 1e-8:
                    function_value += coeffs[freq] * np.exp(freq * 1j * x)
                    function_value += np.conj(coeffs[freq]) * np.exp(-freq * 1j * x)

        return function_value

    n_points = 500
    grid_range = np.linspace(-np.pi, np.pi, n_points)

    plt.plot(grid_range, reconstructed_function(grid_range))
    if title:
        plt.title(title)

    if savefig:
        plt.savefig(title + ".pdf")
    else:
        plt.show()


def fourier_reconstruct_function_2D_plot(coeffs, real_fft=False, savefig=False, title=None):
    """Visualize the 2D periodic function given by the Fourier coefficients.

    Args:
        coeffs (np.ndarray): 2-dimensional Fourier coefficients.
        real_fft (bool): Whether the coefficients are the result of rfftn instead of fftn.
        savefig (bool): Whether or not to save the figure.
        title (str): Title for the file of saved figure.
    """

    if len(coeffs.shape) != 2:
        raise ValueError("Function fourier_reconstruct_function_2D_plot takes 2-dimensional input.")

    def reconstructed_function(x):
        # At each point x, the value of the function is given by
        # c_0 + c_1 e^{ix} + c_2 e^{2ix} + ... + c_{-n} e^{-inx} + ... c_{-1} e^{-ix}

        if real_fft:
            n_freqs = coeffs.shape[1]
        else:
            n_freqs = coeffs.shape[1] // 2 + (coeffs.shape[1] % 2)

        function_value = 0
        # Start with positive coefficients
        for freq_1, freq_2 in product(list(range(n_freqs)), list(range(n_freqs))):
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
    for index in np.ndindex(*((n_points,) * 2)):
        coordinate_vector = [grid_range[i] for i in index]
        function_values[index] = reconstructed_function(coordinate_vector)

    plt.figure(figsize=(8, 8))
    plt.imshow(function_values)

    if title:
        plt.title(title)

    if savefig:
        plt.savefig(title + ".pdf")
    else:
        plt.show()


def radial_box_plots(
    coeffs,
    n_inputs,
    degree,
    print_freq_labels=True,
    colour_dict=None,
    savefig=False,
    title=None,
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
        print_freq_labels (bool): Whether or not to label the frequencies on
            the radial axis. Turn off for large plots.
        colour_dict (str : str): Specify a colour mapping for positive and negative
            real/imaginary components. If none specified, will default to:
            {"real" : "red", "imag" : "black"}
        savefig (bool): Whether or not to save the figure. The filename of the saved figure
            will be set to the figure title. If no title is given, the output file will
            be "FourierPlot.pdf".
        title (str): A title for the plot.
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
    if print_freq_labels:
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

    # Titles and labeling
    if title is not None:
        plt.suptitle(title, fontsize=20)

    if num_subplots == 1:
        avail_axes[0].set_title(
            "Real (left) --- Imag (right)", fontsize=16, y=-0.025 * len(full_labels[0])
        )
    else:
        avail_axes[0].set_title("Real", fontsize=16, y=-0.025 * len(full_labels[0]))
        avail_axes[1].set_title("Imag", fontsize=16, y=-0.025 * len(full_labels[0]))

    plt.tight_layout()

    if savefig:
        if title is None:
            title = "FourierPlot"
        plt.savefig(title + ".pdf")
    else:
        plt.show()
