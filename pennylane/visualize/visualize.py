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

import numpy as np
import pennylane as qml
from matplotlib import pyplot as plt

class Visualize:

    """
    A context manager for storing data relating to visualizations and post-processing of
    quantum circuit optimization.

    The main idea behind this method is that we will wrap the optimization loop inside of some
    context manager, which will recored things like the number of optimization steps, the parameters,
    and the value of the cost function for each step.

    """

    def __init__(self, steps, cost_fn=None):

        self.steps = steps
        self.step_size = step_size
        self.cost_fn = cost_fn

        self.step_log = 0
        self.x_log = []
        self.y_log = []
        self.param_log = []

        self.complete = False

    def __enter__(self):
        print("Beginning Optimization")
        print("--------------------------")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Optimization Complete")

    def update(self, params=None):

        self.step_log += 1
        self.x_log.append(self.step_log)

        if params is not None:
            self.param_log.append(params)
            if self.cost_fn is not None:
                val = self.cost_fn(params)
                self.y_log.append(val)

    def text(self, step=True, cost=False, params=False):

        if step:
            print("Optimization Step {} / {}".format(self.step_log, self.steps))
        if cost:
            print("Cost: {}".format(self.y_log[len(self.y_log)-1]))
        if params:
            print("Parameters: {}".format(self.param_log[len(self.param_log)-1]))
        print("--------------------------")

    def graph(self):

        plt.scatter(self.x_log, self.y_log)
        plt.pause(0.05)
        if self.steps == self.step_log:
            plt.show()