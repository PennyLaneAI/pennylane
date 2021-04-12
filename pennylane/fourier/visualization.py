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


def _validate_coefficients(coeffs, n_inputs, can_be_list=True):
    """Helper function to validate input coefficients of plotting functions.

    Args:
        coeffs (array[complex]): A set (or list of sets) of Fourier coefficients of a
            n_inputs-dimensional function.
        n_inputs (int): The number of inputs (dimension) of the function the coefficients are for.
        can_be_list (bool): Whether or not the plotting function accepts a list of
            coefficients, or only a single set.

    Raises:
        TypeError: If the coefficients are not a list, or numpy array.
        ValueError: if the coefficients are not a suitable type for the plotting function.
    """
    # Make sure we have a list or numpy array
    if not isinstance(coeffs, list) and not isinstance(coeffs, np.ndarray):
        raise TypeError(
            "Input to coefficient plotting functions must be a list of numerical "
            f"Fourier coefficients. Received input of type {type(coeffs)}"
        )

    # In case we have a list, turn it into a numpy array
    if isinstance(coeffs, list):
        coeffs = np.array(coeffs)

    # Check if the user provided a single set of coefficients to a function that is
    # meant to accept multiple samples; add an extra dimension around it if needed
    if len(coeffs.shape) == n_inputs and can_be_list:
        coeffs = np.array([coeffs])

    # Check now that we have the right number of axes for the type of function
    required_shape_size = n_inputs + 1 if can_be_list else n_inputs
    if len(coeffs.shape) != required_shape_size:
        raise ValueError(
            f"Plotting function expected a list of {n_inputs}-dimensional inputs. "
            f"Received coefficients of {len(coeffs.shape)}-dimensional function."
        )

    # Shape in all dimensions of a single set of coefficients must be the same
    shape_set = set(coeffs.shape[1:]) if can_be_list else set(coeffs.shape)
    if len(shape_set) != 1:
        raise ValueError(
            "All dimensions of coefficient array must be the same. "
            f"Received array with dimensions {coeffs.shape}"
        )

    # Size of each sample dimension must be 2d + 1 where d is the degree
    shape_dim = coeffs.shape[1] if can_be_list else coeffs.shape[0]
    if (shape_dim - 1) % 2 != 0:
        raise ValueError(
            "Shape of input coefficients must be 2d + 1, where d is the largest frequency. "
            f"Coefficient array with shape {coeffs.shape} is invalid."
        )

    # Return the coefficients; we may have switched to a numpy array or added a needed extra dimension
    return coeffs


def _extract_data_and_labels(coeffs):
    """Helper function for creating frequency labels and partitionining data.

    Args:
        coeffs (array[complex]): A list of sets of Fourier coefficients.

    Returns:
        (list(str), dict[str, array[complex]): The set of frequency labels, and a data
            dictionary split into real and imaginary parts.
    """
    # extract the x ticks
    nvecs = list(to_dict(coeffs[0]).keys())
    nvecs_formatted = [format_nvec(nvec) for nvec in nvecs]

    # make data
    data = {}
    data["real"] = np.array([[c[nvec].real for nvec in nvecs] for c in coeffs])
    data["imag"] = np.array([[c[nvec].imag for nvec in nvecs] for c in coeffs])

    return nvecs_formatted, data


def _adjust_spine_placement(ax):
    """Helper function to set some common axis properties when plotting."""
    ax.xaxis.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def fourier_violin_plot(coeffs, n_inputs, ax, colour_dict=None, show_freqs=True):
    """Plots a list of sets of Fourier coefficients as a violin plot.

    Args:
        coeffs (array[complex]): A list of sets of Fourier coefficients. The shape of the array
            should resemble that of the output of numpy/scipy's ``fftn`` function, or
            :func:`~.pennylane.fourier.fourier_coefficients`.
        n_inputs (int): The number of input variables in the function.
        ax (list[matplotlib.axes._subplots.AxesSubplot]): Axis on which to plot. Must
            be a pair of axes from a subplot where ``sharex="row"`` and ``sharey="col"``.
        colour_dict (dict[str, str]): A dictionary of the form {"real" : colour_string,
            "imag" : other_colour_string} indicating which colours should be used in the plot.
        show_freqs (bool): Whether or not to print the frequency labels on the plot axis.

    Returns:
        ax: The axes on which the data is plotted.

    **Example**

    .. code-block:: python

        import matplotlib as plt
        from pennylane.fourier import fourier_coefficients, fourier_violin_plot

        f = ... # A function
        n_inputs = ... # Number of inputs to the function
        degree = ... # Degree to which coefficients should be calculated

        # Calculate the Fourier coefficients; may be a single set or a list of
        # multiple sets of coefficients
        coeffs = fourier_coefficients(f, n_inputs, degree)

        # Set up subplots and plot
        fig, ax = plt.subplots(2, 1, sharey=True, figsize=(15, 4))
        fourier_violin_plot(coeffs, n_inputs, ax, show_freqs=True);

    """
    coeffs = _validate_coefficients(coeffs, n_inputs, True)

    if colour_dict is None:
        colour_dict = {"real": "purple", "imag": "green"}

    # Get the labels and data
    nvecs_formatted, data = _extract_data_and_labels(coeffs)

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
        coeffs (array[complex]): A list of sets of Fourier coefficients. The shape of the array
            should resemble that of the output of numpy/scipy's ``fftn`` function, or
            :func:`~.pennylane.fourier.fourier_coefficients`.
        n_inputs (int): The number of input variables in the function.
        ax (list[matplotlib.axes._subplots.AxesSubplot]): Axis on which to plot. Must
            be a pair of axes from a subplot where ``sharex="row"`` and ``sharey="col"``.
        colour_dict (dict[str, str]): A dictionary of the form {"real" : colour_string,
            "imag" : other_colour_string} indicating which colours should be used in the plot.
        show_freqs (bool): Whether or not to print the frequency labels on the plot axis.
        show_fliers (bool): Whether to display the box plot outliers.

    Returns:
        ax: The axes on which the data is plotted.

    **Example**

    .. code-block:: python

        import matplotlib as plt
        from pennylane.fourier import fourier_coefficients, fourier_box_plot

        f = ... # A function
        n_inputs = ... # Number of inputs to the function
        degree = ... # Degree to which coefficients should be calculated

        # Calculate the Fourier coefficients; may be a single set or a list of
        # multiple sets of coefficients
        coeffs = fourier_coefficients(f, n_inputs, degree)

        # Set up subplots and plot
        fig, ax = plt.subplots(2, 1, sharey=True, figsize=(15, 4))
        fourier_box_plot(coeffs, n_inputs, ax, show_freqs=True);
    """
    _validate_coefficients(coeffs, n_inputs, True)

    # The axis received must be a pair of axes in a subplot.
    if colour_dict is None:
        colour_dict = {"real": "purple", "imag": "green"}

    # Get the labels and data
    nvecs_formatted, data = _extract_data_and_labels(coeffs)

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

        coeffs (array[complex]): A single set of Fourier coefficients. The dimensions of the
            array should be

        n_inputs (int): The number of input variables in the function.
        ax (list[matplotlib.axes._subplots.AxesSubplot]): Axis on which to plot. Must
            be a pair of axes from a subplot where ``sharex="row"`` and ``sharey="col"``.
        colour_dict (dict[str, str]): A dictionary of the form {"real" : colour_string,
            "imag" : other_colour_string} indicating which colours should be used in the plot.
        show_freqs (bool): Whether or not to print the frequency labels on the plot axis.

    Returns:
        ax: The axes on which the data is plotted.

    **Example**

    .. code-block:: python

        import matplotlib as plt
        from pennylane.fourier import fourier_coefficients, fourier_bar_plot

        f = ... # A function
        n_inputs = ... # Number of inputs to the function
        degree = ... # Degree to which coefficients should be calculated

        # A single set of Fourier coefficients only for the bar plot
        coeffs = fourier_coefficients(f, n_inputs, degree)

        # Set up subplots and plot
        fig, ax = plt.subplots(2, 1, sharey=True, figsize=(15, 4))
        fourier_box_plot(coeffs, n_inputs, ax, show_freqs=True);
    """
    coeffs = _validate_coefficients(coeffs, n_inputs, False)

    # The axis received must be a pair of axes in a subplot.
    if colour_dict is None:
        colour_dict = {"real": "purple", "imag": "green"}

    # Get the labels and data
    nvecs_formatted, data = _extract_data_and_labels(np.array([coeffs]))
    data_len = len(data["real"][0])

    for (data_type, axis) in zip(["real", "imag"], ax):
        axis.bar(np.arange(data_len), data[data_type][0], color=colour_dict[data_type], alpha=0.7)
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
        coeffs (array[complex]): A list set of Fourier coefficients. Must be 1-
            or 2-dimensional, i.e., the array should have shape ``(2d + 1,)``
            for 1-dimensional, or ``(2d + 1, 2d + 1)`` where ``d`` is the
            degree, i.e., the maximum frequency of present in the coefficients.
            Such an array may be the output of the numpy/scipy ``fft``/``fft2`` functions,
            or :func:`~.pennylane.fourier.fourier_coefficients`.
        n_inputs (int): The number of variables in the function.
        ax (list[matplotlib.axes._subplots.AxesSubplot]): Axis on which to plot. For
            1-dimensional data, length must be the number of frequencies. For 2-dimensional
            data, must be a grid that matches the dimensions of a single set of coefficients.
        colour (str): The outline colour of the points on the plot.

    Returns:
        matplotlib.AxesSubplot: The axes on which the data is plotted.

    **Example**

    .. code-block:: python

        import matplotlib as plt
        from pennylane.fourier import fourier_coefficients, fourier_panel_plot

        f = ... # A function in 1 or 2 variables
        n_inputs = ... # Number of inputs to the function
        degree = ... # Degree to which coefficients should be calculated

        # Calculate the Fourier coefficients; may be a single set or a list of
        # multiple sets of coefficients
        coeffs = fourier_coefficients(f, n_inputs, degree)

        # Set up subplots and plot; need as many plots as there are coefficients
        fig, ax = plt.subplots(
            2*degree+1, 2*degree+1, sharex=True, sharey=True, figsize=(15, 4)
        )
        fourier_panel_plot(coeffs, n_inputs, ax);

    """
    if n_inputs in [1, 2]:
        coeffs = _validate_coefficients(coeffs, n_inputs, True)
    else:
        raise ValueError(
            "Panel plot function accepts input coefficients for only 1- or 2-dimensional functions."
        )

    if colour is None:
        colour = "tab:blue"

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
        coeffs (array[complex]): Fourier coefficients of a 1-dimensional
            function. The array should have length ``2d + 1`` where ``d`` is the
            degree, i.e., the maximum frequency of present in the coefficients.
            Such an array may be the output of the numpy/scipy ``fft`` function, or
            :func:`~.pennylane.fourier.fourier_coefficients`.
        ax (matplotlib.axes._subplots.AxesSubplot): Axis on which to plot. If None, the
            current axis from ``plt.gca()`` will be used.

    Returns:
        matplotlib.AxesSubplot: The axis after plotting is complete.

    **Example**

    .. code-block:: python

        import matplotlib as plt
        from pennylane.fourier import fourier_coefficients, fourier_reconstruct_function_1D_plot

        f = ... # A function in 1 variable
        n_inputs = 1
        degree = ... # Degree to which coefficients should be calculated

        # Calculate the Fourier coefficients
        coeffs = fourier_coefficients(f, n_inputs, degree)

        # It is not necessary to create subplots; the current axis will be used here
        fourier_reconstruct_function_1D_plot(coeffs)

    """
    coeffs = _validate_coefficients(coeffs, 1, False)

    ax = ax or plt.gca()

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
        coeffs (array[complex]): Fourier coefficients of a 2-dimensional
            function. Shape should match the output of a 2-dimensional Fourier
            transform, ``(2d + 1, 2d + 1)``, where ``d`` is the degree, i.e., the
            maximum frequency of present in the coefficients. Such an array may
            be the output of the numpy/scipy ``fft2`` function, or
            :func:`~.pennylane.fourier.fourier_coefficients`.
        ax (matplotlib.axes._subplots.AxesSubplot): Axis on which to plot. If None, the
            current axis from ``plt.gca()`` will be used.

    Returns:
        matplotlib.AxesSubplot: The axis after plotting is complete.

    **Example**

    .. code-block:: python

        import matplotlib as plt
        from pennylane.fourier import fourier_coefficients, fourier_reconstruct_function_2D_plot

        f = ... # A function in 2 variables
        n_inputs = 2
        degree = ... # Degree to which coefficients should be calculated

        # Calculate the Fourier coefficients
        coeffs = fourier_coefficients(f, n_inputs, degree)

        # It is not necessary to create subplots; the current axis may be used here
        fourier_reconstruct_function_2D_plot(coeffs)

    """
    coeffs = _validate_coefficients(coeffs, 2, False)

    ax = ax or plt.gca()

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

    Produces a 2-panel plot in which the left panel represents the real parts of
    Fourier coefficients. This method accepts multiple sets of coefficients, and
    plots the distribution of each coefficient as a boxplot.

    Args:
        coeffs (array[complex]): A list of sets of Fourier coefficients. The shape of the array
            should resemble that of the output of numpy/scipy's ``fftn`` function, or
            :func:`~.pennylane.fourier.fourier_coefficients`.
        n_inputs (int): Dimension of the transformed function.
        show_freqs (bool): Whether or not to label the frequencies on
            the radial axis. Turn off for large plots.
        colour_dict (str : str): Specify a colour mapping for positive and negative
            real/imaginary components. If none specified, will default to:
            {"real" : "red", "imag" : "black"}
        showfliers (bool): Whether or not to plot outlying "fliers" on the boxplots.
        merge_plots (bool): Whether to plot real/complex values on the same panel, or
            on separate panels. Default is to plot real/complex values on separate panels.

    Returns:
        matplotlib.AxesSubplot: The axis after plotting is complete.

    **Example**

    .. code-block:: python

        import matplotlib as plt
        from pennylane.fourier import fourier_coefficients, fourier_radial_box_plot

        f = ... # A function
        n_inputs = ... # Number of inputs to the function
        degree = ... # Degree to which coefficients should be calculated

        # Calculate the Fourier coefficients; may be a single set or a list of
        # multiple sets of coefficients
        coeffs = fourier_coefficients(f, n_inputs, degree)

        fig, ax = plt.subplots(
            1, 2, sharex=True, sharey=True, subplot_kw=dict(polar=True), figsize=(15, 8)
        )
        fourier_radial_box_plot(coeffs, n_inputs, ax)

    """
    coeffs = _validate_coefficients(coeffs, n_inputs, True)

    if colour_dict is None:
        colour_dict = {"real": "red", "imag": "black"}

    # Number, width, and placement of pie slices
    N = coeffs[0].size
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles[-N // 2 + 1 :], angles[: -N // 2 + 1]))[::-1]
    width = (angles[1] - angles[0]) / 2

    # Get the labels and data
    nvecs_formatted, data = _extract_data_and_labels(coeffs)

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

            a.tick_params(pad=7 * n_inputs)

        a.set_xticklabels([])

    return ax
