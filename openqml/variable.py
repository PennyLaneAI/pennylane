# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OpenQML parameterised variable class"""

class Variable:
    """Stores the parameter reference.

    Represents a free device parameter (with a non-fixed value).
    Each time the device is executed, it is given a vector of
    parameter values. Variable is essentially an index into that vector.

    Args:
      idx (int): parameter index >= 0
      val (int or complex or float): initial value of the variable (optional)
    """
    def __init__(self, idx, name=None, val=None):
        self.idx = idx  #: int: parameter index
        self.name = name
        self.val = val

    def __str__(self):
        return 'Variable: {}'.format(self.idx)

    # def __add__(self, other):
    #     if isinstance(other, Variable):
    #         return
    #     self.val += other

    # def __mul__(self, other):
    #     self.val *= other

    # def __div__(self, other):
    #     self.val /= other