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
import os
from sphinx_gallery import gen_rst

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


GALLERY_TEMPLATE = """
.. raw:: html

    <div class="sphx-glr-thumbcontainer">
    
.. only:: html

    .. figure:: /{thumbnail}
    
        {description}
        
.. raw:: html

    </div>
"""


class CustomGalleryItemDirective(Directive):
    """Create a sphinx gallery style thumbnail.
    tooltip and figure are self explanatory. Description could be a link to
    a document like in below example.
    Example usage:

    .. customgalleryitem::
        :figure: /_static/img/thumbnails/babel.jpg
        :description: :doc:`/beginner/deep_learning_nlp_tutorial`

    If figure is specified, a thumbnail will be made out of it and stored in
    _static/thumbs. Therefore, consider _static/thumbs as a 'built' directory.
    """

    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {'figure': directives.unchanged,
                   'description': directives.unchanged}

    has_content = False
    add_index = False

    def run(self):
        try:
            if 'figure' in self.options:
                env = self.state.document.settings.env
                rel_figname, figname = env.relfn2path(self.options['figure'])
                thumbnail = os.path.join('_static/thumbs/', os.path.basename(figname))

                try:
                    os.makedirs('_static/thumbs')
                except FileExistsError:
                    pass

                # x, y = (300, 500)
                # gen_rst.scale_image(figname, thumbnail, int(x), int(y))
            else:
                thumbnail = '_static/thumbs/code.png'

            if 'description' in self.options:
                description = self.options['description']
            else:
                raise ValueError('description not found')

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []

        thumbnail_rst = GALLERY_TEMPLATE.format(thumbnail=thumbnail,
                                                description=description)
        thumbnail = StringList(thumbnail_rst.split('\n'))
        thumb = nodes.paragraph()
        self.state.nested_parse(thumbnail, self.content_offset, thumb)
        return [thumb]
