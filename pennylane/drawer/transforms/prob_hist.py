# Copyright 2023 Xanadu Quantum Technologies Inc.

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

from .core import VisualizationBase


class ProbHistogram(VisualizationBase):
    def _initialize(self, fig, axs):
        axs.set_ylim([0, 1])
        dims = 2 ** len(self.tape.wires)
        return axs.bar(np.arange(dims), np.zeros(dims), color="b").patches

    def _update(self, artists, probs):
        for rect, prob in zip(artists, probs):
            rect.set_height(prob)
