# Copyright 2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Contains the ``Visualize`` context manager.
"""
from matplotlib import pyplot as plt

try:
    from IPython import display
except ImportError:
    pass


def _j_nb():
    try:
        # pylint: disable=import-outside-toplevel
        from IPython import get_ipython

        # pylint: enable=import-outside-toplevel
        if "IPKernelApp" not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


class Visualize:
    """
    A context manager for storing data relating to visualizations and post-processing of
    variational quantum circuit optimization.

    When a PennyLane optimization loop is wrapped in the Visualize context manager, data from each optimization step
    is recorded and outputted as text or a graph.

    Keyword Args:
        step_size (int): The number of steps taken between each data recording instance.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, step_size=1):

        if not isinstance(step_size, int):
            raise ValueError("'step_size' must be of type int, got {}".format(type(step_size)))

        self.step_size = step_size

        self._step_log = 0
        self._x_log = []
        self._y_log = []
        self._param_log = []

        self._graph_bool = False

    @property
    def cost_data(self):
        """
        Returns a tuple of the recorded optimization steps, and the corresponding values of the cost function.

        Returns:
            tuple: a tuple containing lists of optimization steps and cost function values
        """

        return (self._x_log, self._y_log)

    @property
    def param_data(self):
        """
        Returns a tuple of the recorded optimization steps, and the corresponding variational parameters.

        Returns:
            tuple: a tuple containing lists of optimization steps and parameters values
        """

        return (self._x_log, self._param_log)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self._graph_bool:
            plt.show()

    def update(self, cost=None, params=None):
        """
        Updates the data in the Visualize context manager instance. This data can then be called by the user, or
        utilized by the active context manager to create text or graphical outputs.

        Keyword Args:

            cost (float): Value of the cost function corresponding to an optimization step
            params (Iterable): Variational parameters corresponding to an optimization step

        .. UsageDetails::

            Consider the simple variational quantum circuit and the cost function:

            .. code-block:: python3

                import pennylane as qml
                from pennylane.visualize import Visualize

                dev = qml.device('default.qubit', wires=3)

                @qml.qnode(dev)
                def circuit(gamma):

                    qml.RX(gamma, wires=0)
                    qml.RY(gamma, wires=1)
                    qml.RX(gamma, wires=2)
                    qml.CNOT(wires=[0, 2])
                    qml.RY(gamma, wires=1)

                    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

                def cost_fn(gamma):
                    return sum(circuit(gamma[0]))

            To utilize the PennyLane visualizations functionality, we wrap the optimization loop
            in the context manager, and call the ``update()`` method with each optimization step:

            .. code-block:: python3

                optimizer = qml.GradientDescentOptimizer()
                steps = 5
                params = np.array([1.])

                with Visualize() as viz:
                    for i in range(steps):
                        params, cost = optimizer.step_and_cost(cost_fn, params)
                        viz.update(cost=cost, params=params)

            We can then return values of data recorded during the optimization procedure. For example, we can call
            the values of the cost function after each optimization step:

            >>> steps, cost = viz.cost_data
            >>> print(cost)
            [0.29001330299410394, 0.16958978666464564, 0.0554891459073546, -0.05179578982297689, -0.151952780085496]
        """

        self._step_log += 1

        if self._step_log % self.step_size == 0:

            self._x_log.append(self._step_log)

            if params is not None:
                self._param_log.append(params)
            if cost is not None:
                self._y_log.append(cost)

    def text(self, step=True, cost=False, params=False, static=True):
        """
        Returns a text-based output of the data in the Visualize context manager instance.

        Keyword args:
            step (bool): Choice of whether the number of steps are outputted or not.
            cost (bool): Choice of whether the cost function values are outputted or not.
            params (bool): Choice of whether the parameter values are outputted or not.
            static (bool): Choice of whether the text output only displays the information at the current
                           step (static), or all steps.

        .. Warning::

            Usage of ``viz.text()`` and dynamic ``viz.update()`` simultaneously within Jupyter leads to a strange
            flickering of the output window. In this situation,
            it is recommended that the text output built into the graph is used instead of ``viz.text()``.

        .. UsageDetails::

            Usage of the ``text()`` method is similar to usage of the ``update()`` method. For instance, consider the
            following variational circuit and cost function:

            .. code-block:: python3

                import pennylane as qml
                from pennylane.visualize import Visualize

                dev = qml.device('default.qubit', wires=3)

                @qml.qnode(dev)
                def circuit(gamma):

                    qml.RX(gamma, wires=0)
                    qml.RY(gamma, wires=1)
                    qml.RX(gamma, wires=2)
                    qml.CNOT(wires=[0, 2])
                    qml.RY(gamma, wires=1)

                    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

                def cost_fn(gamma):
                    return sum(circuit(gamma[0]))

            We then call the ``text()`` and ``update()`` methods inside the optimization loop:

            .. code-block:: python3

                optimizer = qml.GradientDescentOptimizer()
                steps = 5
                params = np.array([1.])

                with Visualize() as viz:
                    for i in range(steps):
                        params, cost = optimizer.step_and_cost(cost_fn, params)
                        viz.update(cost=cost, params=params)
                        viz.text(cost=True, params=True, static=False)

            .. code-block:: none

                || Optimization Step 001 || Cost: 0.41608205 || Parameters: ['1.03569363'] ||
                || Optimization Step 002 || Cost: 0.29001330 || Parameters: ['1.07061477'] ||
                || Optimization Step 003 || Cost: 0.16958979 || Parameters: ['1.10463974'] ||
                || Optimization Step 004 || Cost: 0.05548915 || Parameters: ['1.13766278'] ||
                || Optimization Step 005 || Cost: -0.0517958 || Parameters: ['1.16959683'] ||

        """

        if not isinstance(step, bool):
            raise ValueError("'step' must be of type bool, got {}".format(type(step)))

        if not isinstance(cost, bool):
            raise ValueError("'cost' must be of type bool, got {}".format(type(cost)))

        if not isinstance(params, bool):
            raise ValueError("'params' must be of type bool, got {}".format(type(params)))

        if self._step_log % self.step_size == 0:

            output_str = "|"

            if step:
                if not static:
                    output_str += "| Optimization Step {} |".format(str(self._step_log).zfill(3))
                else:
                    output_str += "| Optimization Step {} |".format(self._step_log)
            if cost:
                if self._y_log[len(self._y_log) - 1] < 0:
                    output_str += "| Cost: {} |".format(
                        format(self._y_log[len(self._y_log) - 1], ".7f")
                    )
                else:
                    output_str += "| Cost: {} |".format(
                        format(self._y_log[len(self._y_log) - 1], ".8f")
                    )
            if params:
                display_par = [
                    format(o, ".7f") if o < 0 else format(o, ".8f")
                    for o in self._param_log[len(self._param_log) - 1]
                ]
                output_str += "| Parameters: {} |".format(display_par)

            output_str += "|"

            if static:
                print(output_str, end="\r")
            else:
                print(output_str)

    def graph(self, color=None, xlim=None, ylim=None):
        """
        Returns a graph of the value of the cost function after each optimization step.

        Keyword args:
            color (str): The color of the graph. See the `Matplotlib documentation
                         <https://matplotlib.org/tutorials/colors/colors.html>`_ for more information on
                         the formats of color that can be passed as an argument.
            xlim (tuple): The minimum and maximum values on the x-axis.
            ylim (tuple): The minimum and maximum values on the y-axis.

        .. Warning::

            Usage of ``viz.text()`` and dynamic ``viz.update()`` simultaneously within Jupyter leads to a strange
            flickering of the output window. In this situation,
            it is recommended that the text output built into the graph is used instead of ``viz.text()``.

        .. UsageDetails::

            Usage of the ``graph()`` method is similar to usage of the ``update()`` method. For instance, consider the
            following variational circuit and cost function:

            .. code-block:: python3

               import pennylane as qml
                from pennylane.visualize import Visualize

                dev = qml.device('default.qubit', wires=3)

                @qml.qnode(dev)
                def circuit(gamma):

                    qml.RX(gamma, wires=0)
                    qml.RY(gamma, wires=1)
                    qml.RX(gamma, wires=2)
                    qml.CNOT(wires=[0, 2])
                    qml.RY(gamma, wires=1)

                    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

                def cost_fn(gamma):
                    return sum(circuit(gamma[0]))

            We then call the ``graph()`` and ``update()`` methods inside the optimization loop:

            .. code-block:: python3

                optimizer = qml.GradientDescentOptimizer()
                steps = 5
                params = np.array([1.])

                with Visualize() as viz:
                    for i in range(steps):
                        params, cost = optimizer.step_and_cost(cost_fn, params)
                        viz.update(cost=cost, params=params)
                        viz.graph()

            Execution of this code block will result an a graph window, which will dynamically plot the value
            of the cost function after each optimization step.

            In addition, the ``graph()`` method can also be called outside the optimization loop (after the
            optimization has finished), to produce a static graph showing the values of the cost function for
            each optimization step:

            .. code-block:: python3

                optimizer = qml.GradientDescentOptimizer()
                steps = 5
                params = np.array([1.])

                with Visualize() as viz:
                    for i in range(steps):
                        params, cost = optimizer.step_and_cost(cost_fn, params)
                        viz.update(cost=cost, params=params)
                    viz.graph()
        """

        if color is None:
            color = "#1D9598"

        self._graph_bool = True

        if not isinstance(color, str):
            raise ValueError("'color' must be of type str, got {}".format(type(color)))

        if _j_nb():

            plt.clf()

            if xlim is not None:
                mn, mx = xlim
                plt.xlim(mn, mx)

            if ylim is not None:
                mn, mx = ylim
                plt.ylim(mn, mx)

            plt.ylabel("Cost")
            plt.xlabel("Steps")

            plt.plot(self._x_log, self._y_log, color=color)
            plt.scatter(self._x_log, self._y_log, color=color)

            plt.gcf().text(0.13, 1.02, "Step {}".format(len(self._x_log)), fontsize=12)
            plt.gcf().text(
                0.13, 0.95, "Cost = {}".format(self._y_log[len(self._y_log) - 1]), fontsize=12
            )

            display.display(plt.gcf())
            display.clear_output(wait=True)

        else:

            plt.clf()

            if xlim is not None:
                mn, mx = xlim
                plt.xlim(mn, mx)

            if ylim is not None:
                mn, mx = ylim
                plt.ylim(mn, mx)

            plt.ylabel("Cost")
            plt.xlabel("Steps")

            plt.plot(self._x_log, self._y_log, color=color)
            plt.scatter(self._x_log, self._y_log, color=color)

            plt.gcf().text(0.13, 0.94, "Step {}".format(len(self._x_log)), fontsize=11)
            plt.gcf().text(
                0.13, 0.9, "Cost = {}".format(self._y_log[len(self._y_log) - 1]), fontsize=11
            )

            plt.pause(0.025)
