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

# pylint:disable=too-many-arguments,blacklisted-name

# Matplotlib is not a hard requirement for PennyLane in general, but it *is*
# a hard requirement for everything in this module.
try:
    from matplotlib.colors import to_rgb
except (ModuleNotFoundError, ImportError) as e:  # pragma: no cover
    raise ImportError(
        "Module matplotlib is required for visualization in the Fourier module. "
        "You can install matplolib via \n\n   pip install matplotlib"
    ) from e

from .utils import format_nvec


def _validate_coefficients(coeffs, n_inputs, can_be_list=True):
    """Helper function to validate input coefficients of plotting functions.

    Args:
        coeffs (array[complex]): A set (or list of sets) of Fourier coefficients of a
            n_inputs-dimensional function.
        n_inputs (int): The number of inputs (dimension) of the function the coefficients are for.
        can_be_list (bool): Whether or not the plotting function accepts a list of
            coefficients, or only a single set.

    Raises:
        TypeError: If the coefficients are not a list or array.
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

    # Size of each sample dimension must be 2d_i + 1 where d_i is the i-th degree
    dims = coeffs.shape[1:] if can_be_list else coeffs.shape
    if any((dim - 1) % 2 for dim in dims):
        raise ValueError(
            "Shape of input coefficients must be 2d_i + 1, where d_i is the largest frequency "
            f"in the i-th input. Coefficient array with shape {coeffs.shape} is invalid."
        )

    # Return the coefficients; we may have switched to a numpy array or added a needed extra dimension
    return coeffs


def _extract_data_and_labels(coeffs):
    """Helper function for creating frequency labels and partitioning data.

    Args:
        coeffs (array[complex]): A list of sets of Fourier coefficients.

    Returns:
        (list(str), dict[str, array[complex]): The set of frequency labels, and a data
            dictionary split into real and imaginary parts.
    """
    # extract the x ticks: create generator for indices nvec = (n1, ..., nN),
    # ranging from (-d1,...,-dN) to (d1,...,dN).
    nvecs = list(product(*(np.array(range(-(d // 2), d // 2 + 1)) for d in coeffs[0].shape)))
    nvecs_formatted = [format_nvec(nvec) for nvec in nvecs]

    # extract flattened data by real part and imaginary part, and respecting negative indices
    data = {}
    data["real"] = np.array([[c[nvec] for nvec in nvecs] for c in coeffs.real])
    data["imag"] = np.array([[c[nvec] for nvec in nvecs] for c in coeffs.imag])

    return nvecs_formatted, data


def _adjust_spine_placement(ax):
    """Helper function to set some common axis properties when plotting."""
    ax.xaxis.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def violin(coeffs, n_inputs, ax, colour_dict=None, show_freqs=True):
    """Plots a list of sets of Fourier coefficients as a violin plot.

    Args:
        coeffs (list[array[complex]]): A list of sets of Fourier coefficients. The shape of the
            coefficient arrays should resemble that of the output of NumPy/SciPy's ``fftn`` function, or
            :func:`~.pennylane.fourier.coefficients`.
        n_inputs (int): The number of input variables in the function.
        ax (array[matplotlib.axes.Axes]): Axis on which to plot. Must
            be a pair of axes from a subplot where ``sharex="row"`` and ``sharey="col"``.
        colour_dict (dict[str, str]): A dictionary of the form ``{"real" : colour_string,
            "imag" : other_colour_string}`` indicating which colours should be used in the plot.
        show_freqs (bool): Whether or not to print the frequency labels on the plot axis.

    Returns:
        array[matplotlib.axes.Axes]: The axes on which the data is plotted.

    **Example**

    Suppose we have the following quantum function:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit_with_weights(w, x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[1, 0])

            qml.Rot(*w[0], wires=0)
            qml.Rot(*w[1], wires=1)
            qml.CNOT(wires=[1, 0])

            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[1, 0])

            return qml.expval(qml.Z(0))

    We would like to compute and plot the distribution of Fourier coefficients
    for many random values of the weights ``w``. First, we generate all the coefficients:

    .. code-block:: python

        from functools import partial

        coeffs = []

        n_inputs = 2
        degree = 2

        for _ in range(100):
            weights = np.random.normal(0, 1, size=(2, 3))
            c = coefficients(partial(circuit_with_weights, weights), n_inputs, degree)
            coeffs.append(c)

    We can now plot by setting up a pair of ``matplotlib`` axes and passing them
    to the plotting function:

    >>> import matplotlib.pyplot as plt
    >>> from pennylane.fourier.visualize import violin
    >>> fig, ax = plt.subplots(2, 1, sharey=True, figsize=(15, 4))
    >>> violin(coeffs, n_inputs, ax, show_freqs=True)

    .. image:: ../../_static/fourier_vis_violin.png
        :align: center
        :width: 800px
        :target: javascript:void(0);
    """
    coeffs = _validate_coefficients(coeffs, n_inputs, True)

    # Check axis shape
    if ax.size != 2:
        raise ValueError("Matplotlib axis should consist of two subplots.")

    if colour_dict is None:
        colour_dict = {"real": "purple", "imag": "green"}

    # Get the labels and data
    nvecs_formatted, data = _extract_data_and_labels(coeffs)

    for data_type, axis in zip(["real", "imag"], ax):
        violinplt = axis.violinplot(data[data_type], showextrema=False)
        for bd in violinplt["bodies"]:
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


def box(coeffs, n_inputs, ax, colour_dict=None, show_freqs=True, show_fliers=True):
    """Plot a list of sets of Fourier coefficients as a box plot.

    Args:
        coeffs (list[array[complex]]): A list of sets of Fourier coefficients. The shape of the
            coefficient arrays should resemble that of the output of numpy/scipy's ``fftn``
            function, or :func:`~.pennylane.fourier.coefficients`.
        n_inputs (int): The number of input variables in the function.
        ax (array[matplotlib.axes.Axes]): Axis on which to plot. Must
            be a pair of axes from a subplot where ``sharex="row"`` and ``sharey="col"``.
        colour_dict (dict[str, str]): A dictionary of the form {"real" : colour_string,
            "imag" : other_colour_string} indicating which colours should be used in the plot.
        show_freqs (bool): Whether or not to print the frequency labels on the plot axis.
        show_fliers (bool): Whether to display the box plot outliers.

    Returns:
        array[matplotlib.axes.Axes]: The axes after plotting is complete.

    **Example**

    Suppose we have the following quantum function:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit_with_weights(w, x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[1, 0])

            qml.Rot(*w[0], wires=0)
            qml.Rot(*w[1], wires=1)
            qml.CNOT(wires=[1, 0])

            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[1, 0])

            return qml.expval(qml.Z(0))

    We would like to compute and plot the distribution of Fourier coefficients
    for many random values of the weights ``w``. First, we generate all the coefficients:

    .. code-block:: python

        from functools import partial

        coeffs = []

        n_inputs = 2
        degree = 2

        for _ in range(100):
            weights = np.random.normal(0, 1, size=(2, 3))
            c = coefficients(partial(circuit_with_weights, weights), n_inputs, degree)
            coeffs.append(c)

    We can now plot by setting up a pair of ``matplotlib`` axes and passing them
    to the plotting function:

    >>> import matplotlib.pyplot as plt
    >>> from pennylane.fourier.visualize import box
    >>> fig, ax = plt.subplots(2, 1, sharey=True, figsize=(15, 4))
    >>> box(coeffs, n_inputs, ax, show_freqs=True)

    .. image:: ../../_static/fourier_vis_box.png
        :align: center
        :width: 800px
        :target: javascript:void(0);
    """
    coeffs = _validate_coefficients(coeffs, n_inputs, True)

    # Check axis shape
    if ax.size != 2:
        raise ValueError("Matplotlib axis should consist of two subplots.")

    # The axis received must be a pair of axes in a subplot.
    if colour_dict is None:
        colour_dict = {"real": "purple", "imag": "green"}

    # Get the labels and data
    nvecs_formatted, data = _extract_data_and_labels(coeffs)

    for data_type, axis in zip(["real", "imag"], ax):
        data_colour = colour_dict[data_type]
        axis.boxplot(
            data[data_type],
            boxprops={
                "facecolor": to_rgb(data_colour) + (0.4,),
                "color": data_colour,
                "edgecolor": data_colour,
            },
            medianprops={"color": data_colour, "linewidth": 1.5},
            flierprops={"markeredgecolor": data_colour},
            whiskerprops={"color": data_colour},
            capprops={"color": data_colour},
            patch_artist=True,
            showfliers=show_fliers,
        )

        _adjust_spine_placement(axis)
        axis.set_ylabel(data_type)
        axis.xaxis.set_ticks(np.arange(1, len(nvecs_formatted) + 1))

    ax[0].tick_params(axis="x", colors="white")  # hack to get rid of ticks but keep grid

    if show_freqs:
        ax[1].tick_params(axis="x", which="both", length=0)  # remove ticks without removing labels
        ax[1].xaxis.set_ticklabels(nvecs_formatted, fontsize=10, color="grey")
        ax[1].xaxis.set_ticks_position("top")
    else:
        ax[1].tick_params(axis="x", colors="white")  # hack to get rid of ticks but keep grid

    return ax


def bar(coeffs, n_inputs, ax, colour_dict=None, show_freqs=True):
    """Plot a set of Fourier coefficients as a bar plot.

    Args:

        coeffs (array[complex]): A single set of Fourier coefficients. The dimensions of the coefficient
            array should be ``(2d + 1, ) * n_inputs`` where ``d`` is the largest frequency.
        n_inputs (int): The number of input variables in the function.
        ax (list[matplotlib.axes.Axes]): Axis on which to plot. Must
            be a pair of axes from a subplot where ``sharex="row"`` and ``sharey="col"``.
        colour_dict (dict[str, str]): A dictionary of the form ``{"real" : colour_string,
            "imag" : other_colour_string}`` indicating which colours should be used in the plot.
        show_freqs (bool): Whether or not to print the frequency labels on the plot axis.

     Returns:
        array[matplotlib.axes.Axes]: The axes after plotting is complete.

    **Example**

    Suppose we have the following quantum function:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit_with_weights(w, x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[1, 0])

            qml.Rot(*w[0], wires=0)
            qml.Rot(*w[1], wires=1)
            qml.CNOT(wires=[1, 0])

            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[1, 0])

            return qml.expval(qml.Z(0))

    We would like to compute and plot a single set of Fourier coefficients. We will
    choose some values for ``w`` at random:

    .. code-block:: python

        from functools import partial

        n_inputs = 2
        degree = 2

        weights = np.random.normal(0, 1, size=(2, 3))
        coeffs = coefficients(partial(circuit_with_weights, weights), n_inputs, degree)

    We can now plot by setting up a pair of ``matplotlib`` axes and passing them
    to the plotting function:

    >>> import matplotlib.pyplot as plt
    >>> from pennylane.fourier.visualize import bar
    >>> fig, ax = plt.subplots(2, 1, sharey=True, figsize=(15, 4))
    >>> bar(coeffs, n_inputs, ax, colour_dict={"real" : "red", "imag" : "blue"})

    .. image:: ../../_static/fourier_vis_bar_plot_2.png
        :align: center
        :width: 800px
        :target: javascript:void(0);
    """
    coeffs = _validate_coefficients(coeffs, n_inputs, False)

    # Check axis shape
    if ax.size != 2:
        raise ValueError("Matplotlib axis should consist of two subplots.")

    # The axis received must be a pair of axes in a subplot.
    if colour_dict is None:
        colour_dict = {"real": "purple", "imag": "green"}

    # Get the labels and data
    nvecs_formatted, data = _extract_data_and_labels(np.array([coeffs]))
    data_len = len(data["real"][0])

    for data_type, axis in zip(["real", "imag"], ax):
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


def panel(coeffs, n_inputs, ax, colour=None):
    """Plot a list of sets of coefficients in the complex plane for a 1- or 2-dimensional function.

    Args:
        coeffs (list[array[complex]]): A list of sets of Fourier coefficients. The shape of the
            coefficient arrays must all be either 1- or 2-dimensional, i.e.,
            each array should have shape ``(2d + 1,)``
            for the 1-dimensional case, or ``(2d + 1, 2d + 1)`` where ``d`` is the
            degree, i.e., the maximum frequency of present in the coefficients.
            Such an array may be the output of the numpy/scipy ``fft``/``fft2`` functions,
            or :func:`~.pennylane.fourier.coefficients`.
        n_inputs (int): The number of variables in the function.
        ax (array[matplotlib.axes._subplots.AxesSubplot]): Axis on which to plot. For
            1-dimensional data, length must be the number of frequencies. For 2-dimensional
            data, must be a grid that matches the dimensions of a single set of coefficients.
        colour (str): The outline colour of the points on the plot.

    Returns:
        array[matplotlib.axes.Axes]: The axes after plotting is complete.

    **Example**

    Suppose we have the following quantum function:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit_with_weights(w, x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[1, 0])

            qml.Rot(*w[0], wires=0)
            qml.Rot(*w[1], wires=1)
            qml.CNOT(wires=[1, 0])

            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[1, 0])

            return qml.expval(qml.Z(0))

    We would like to compute and plot the distribution of Fourier coefficients
    for many random values of the weights ``w``. First, we generate all the coefficients:

    .. code-block:: python

        from functools import partial

        coeffs = []

        n_inputs = 2
        degree = 2

        for _ in range(100):
            weights = np.random.normal(0, 1, size=(2, 3))
            c = coefficients(partial(circuit_with_weights, weights), n_inputs, degree)
            coeffs.append(c)

    We can now plot by setting up a pair of ``matplotlib`` axes and passing them
    to the plotting function. The of axes must be large enough to represent all
    the available coefficients (in this case, since we have 2 variables and use
    degree 2, we need a 5x5 grid.

    >>> import matplotlib.pyplot as plt
    >>> from pennylane.fourier.visualize import panel
    >>> fig, ax = plt.subplots(5, 5, figsize=(12, 10), sharex=True, sharey=True)
    >>> panel(coeffs, n_inputs, ax)

    .. image:: ../../_static/fourier_vis_panel.png
        :align: center
        :width: 800px
        :target: javascript:void(0);

    """
    if n_inputs in [1, 2]:
        coeffs = _validate_coefficients(coeffs, n_inputs, True)
    else:
        raise ValueError(
            "Panel plot function accepts input coefficients for only 1- or 2-dimensional functions."
        )

    if ax.shape != coeffs[0].shape:
        raise ValueError("Shape of subplot axes must match the shape of the coefficient data.")

    if colour is None:
        colour = "tab:blue"

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


def radial_box(coeffs, n_inputs, ax, show_freqs=True, colour_dict=None, show_fliers=True):
    """Plot a list of sets of Fourier coefficients on a radial plot as box plots.

    Produces a 2-panel plot in which the left panel represents the real parts of
    Fourier coefficients. This method accepts multiple sets of coefficients, and
    plots the distribution of each coefficient as a boxplot.

    Args:
        coeffs (list[array[complex]]): A list of sets of Fourier coefficients. The shape of the
            coefficient arrays should resemble that of the output of numpy/scipy's ``fftn`` function, or
            :func:`~.pennylane.fourier.coefficients`.
        n_inputs (int): Dimension of the transformed function.
        ax (array[matplotlib.axes.Axes]): Axes to plot on. For this function, subplots
            must specify ``subplot_kw={"polar":True}`` upon construction.
        show_freqs (bool): Whether or not to label the frequencies on
            the radial axis. Turn off for large plots.
        colour_dict (dict[str, str]): Specify a colour mapping for positive and negative
            real/imaginary components. If none specified, will default to:
            ``{"real" : "red", "imag" : "black"}``
        showfliers (bool): Whether or not to plot outlying "fliers" on the boxplots.
        merge_plots (bool): Whether to plot real/complex values on the same panel, or
            on separate panels. Default is to plot real/complex values on separate panels.

    Returns:
        array[matplotlib.axes.Axes]: The axes after plotting is complete.

    **Example**

    Suppose we have the following quantum function:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit_with_weights(w, x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[1, 0])

            qml.Rot(*w[0], wires=0)
            qml.Rot(*w[1], wires=1)
            qml.CNOT(wires=[1, 0])

            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[1, 0])

            return qml.expval(qml.Z(0))

    We would like to compute and plot the distribution of Fourier coefficients
    for many random values of the weights ``w``. First, we generate all the coefficients:

    .. code-block:: python

        from functools import partial

        coeffs = []

        n_inputs = 2
        degree = 2

        for _ in range(100):
            weights = np.random.normal(0, 1, size=(2, 3))
            c = coefficients(partial(circuit_with_weights, weights), n_inputs, degree)
            coeffs.append(c)

    We can now plot by setting up a pair of ``matplotlib`` axes and passing them
    to the plotting function. Note that the axes passed must use polar coordinates.

    .. code-block:: python

        import matplotlib.pyplot as plt
        from pennylane.fourier.visualize import radial_box

        fig, ax = plt.subplots(
            1, 2, sharex=True, sharey=True,
            subplot_kw={"polar": True},
            figsize=(15, 8)
        )

        radial_box(coeffs, 2, ax, show_freqs=True, show_fliers=False)

    .. image:: ../../_static/fourier_vis_radial_box.png
        :align: center
        :width: 800px
        :target: javascript:void(0);

    """
    coeffs = _validate_coefficients(coeffs, n_inputs, True)

    # Check axis shape
    if ax.size != 2:
        raise ValueError("Matplotlib axis should consist of two subplots.")

    if ax[0].name != "polar" or ax[1].name != "polar":
        raise ValueError("Matplotlib axes for radial_box must be polar.")

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
            boxprops={
                "facecolor": to_rgb(data_colour) + (0.4,),
                "color": data_colour,
                "edgecolor": data_colour,
            },
            medianprops={"color": data_colour, "linewidth": 1.5},
            flierprops={"markeredgecolor": data_colour},
            whiskerprops={"color": data_colour},
            capprops={"color": data_colour},
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
