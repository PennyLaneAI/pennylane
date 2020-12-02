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

from matplotlib import pyplot as plt
from IPython import display
import time

def _j_nb():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
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

    Args:
        steps (int): The number of steps for which the circuit is being optimized.

    Keyword Args:
        step_size (int): The number of steps taken between each data recording instance.
        cost_fn (func): The cost function that is being optimized.

    """

    def __init__(self, steps, step_size=1, cost_fn=None):

        if not isinstance(steps, int):
            raise ValueError("'steps' must be of type int, got {}".format(type(steps)))

        if not isinstance(step_size, int):
            raise ValueError("'step_size' must be of type int, got {}".format(type(step_size)))

        self.steps = steps
        self.step_size = step_size
        self.cost_fn = cost_fn

        self._step_log = 0
        self._x_log = []
        self._y_log = []
        self._param_log = []

        self._complete = False

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

        return (self.x_log, self._param_log)

    def __enter__(self):
        print("Beginning Optimization")
        print("--------------------------")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Optimization Complete")

    def update(self, params=None):
        """
        Updates the data in the Visualize context manager instance.
        """

        self._step_log += 1

        if self._step_log % self.step_size == 0:

            self._x_log.append(self._step_log)

            if params is not None:
                self._param_log.append(params)
                if self.cost_fn is not None:
                    val = self.cost_fn(params)
                    self._y_log.append(val)

    def text(self, step=True, cost=False, params=False):
        """
        Returns the data in the Visualize context manager instance, a text-based output.
        """

        if not isinstance(step, bool):
            raise ValueError("'step' must be of type bool, got {}".format(type(step)))

        if not isinstance(cost, int):
            raise ValueError("'cost' must be of type bool, got {}".format(type(cost)))

        if not isinstance(params, int):
            raise ValueError("'params' must be of type bool, got {}".format(type(params)))

        if self._step_log % self.step_size == 0:

            if step:
                print("Optimization Step {} / {}".format(self._step_log, self.steps))
            if cost:
                print("Cost: {}".format(self._y_log[len(self._y_log)-1]))
            if params:
                print("Parameters: {}".format(self._param_log[len(self._param_log)-1]))
            print("--------------------------")

    def graph(self):
        """
        Returns a graph of the value of the cost function for each optimization step.
        """

        if _j_nb():

            plt.clf()

            plt.ylabel("Cost")
            plt.xlabel("Steps")

            plt.plot(self._x_log, self._y_log, color="#1D9598")
            plt.scatter(self._x_log, self._y_log, color="#1D9598")

            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(0.05)
            if self.steps == self._step_log:
                plt.show()

        else:

            plt.ylabel("Cost")
            plt.xlabel("Steps")

            plt.plot(self._x_log, self._y_log, color="#1D9598")
            plt.scatter(self._x_log, self._y_log, color="#1D9598")
            plt.pause(0.025)
            if self.steps == self._step_log:
                plt.show()