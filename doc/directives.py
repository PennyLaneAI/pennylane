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
Custom sphinx directives
"""
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from docutils import nodes


USAGE_DETAILS_TEMPLATE = """
.. raw:: html

    <a class="usage-details-header collapse-header" data-toggle="collapse" href="#usageDetails" aria-expanded="false" aria-controls="usageDetails">
        <h2 style="font-size: 24px;">
            <i class="fas fa-angle-down rotate" style="float: right;"></i> Usage Details
        </h2>
    </a>
    <div class="collapse" id="usageDetails">

{content}

.. raw:: html

    </div>
"""


class UsageDetails(Directive):
    """Create a collapsed Usage Details section in the documentation."""

    # defines the parameter the directive expects
    # directives.unchanged means you get the raw value from RST
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = True

    def run(self):
        rst = USAGE_DETAILS_TEMPLATE.format(content="\n".join(self.content))
        string_list = StringList(rst.split('\n'))
        node = nodes.section()
        self.state.nested_parse(string_list, self.content_offset, node)
        return [node]
