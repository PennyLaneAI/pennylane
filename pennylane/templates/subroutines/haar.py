# Copyright 2018-2019 Xanadu Quantum Technologies Inc.

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
Contains the ``HaarCircuit`` template.
"""
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane.templates.decorator import template


@template
def HaarCircuit(wires):
    r"""Description of my new template.

    Args:
        wires (Sequence[int]): wires the interferometer should act on

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        Give some code examples of how to use the template, what to be aware of, which
        options are available etc.
    """

    #############
    # Input checks
    #
    # Use the functions in :mod:``pennylane.templates.utils`` to check that the input
    # has the correct format. For example, if the features need to be lying in a certain
    # interval, this should be asserted here.
    #
    ###############

    # Add your main template here.
    # Since a template is a sequence of operations, there is no return statement.
