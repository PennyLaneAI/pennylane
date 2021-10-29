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


GALLERY_TEMPLATE = """
.. raw:: html

    <div class="card" style="width: 13.5rem; float:left; margin: 10px;">
        <a href={link}>
            <img class="card-img-top" src={thumbnail} alt="image not found" style="width: 13.5rem; height: 6rem;">
            <div class="card-body">
                <p class="card-text"> {description} </p>
            </div>
        </a>
    </div>
"""


class CustomGalleryItemDirective(Directive):
    """Create a sphinx gallery style thumbnail.
    tooltip and figure are self explanatory. Description could be a link to
    a document like in below example.
    Example usage:

    .. customgalleryitem::
        :figure: /_static/img/thumbnails/babel.jpg
        :description: This is a tutorial
        :link: /beginner/deep_learning_nlp_tutorial

    If figure is specified, a thumbnail will be made out of it and stored in
    _static/thumbs. Therefore, consider _static/thumbs as a 'built' directory.

    """

    required_arguments = 0
    optional_arguments = 4
    final_argument_whitespace = True
    option_spec = {'figure': directives.unchanged,
                   'description': directives.unchanged,
                   'link': directives.unchanged}

    has_content = False
    add_index = False

    def run(self):
        try:
            if 'figure' in self.options:
                thumbnail = self.options['figure']
            else:
                thumbnail = '_static/thumbs/code.png'

            if 'description' in self.options:
                description = self.options['description']
            else:
                raise ValueError('description not found')

            if 'link' in self.options:
                link = self.options['link']
            else:
                link = "introduction/templates"

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []

        thumbnail_rst = GALLERY_TEMPLATE.format(thumbnail=thumbnail,
                                                description=description,
                                                link=link)
        thumbnail = StringList(thumbnail_rst.split('\n'))
        thumb = nodes.paragraph()
        self.state.nested_parse(thumbnail, self.content_offset, thumb)
        return [thumb]


TITLE_CARD_TEMPLATE = """
.. raw:: html

    <div class="card" style="width: 15rem; float:left; margin: 10px;">
        <a href={link}>
            <div class="card-header">
                <b>{name}</b>
            </div>
            <div class="card-body">
                <p class="card-text"> {description} </p>
            </div>
        </a>
    </div>
"""


class TitleCardDirective(Directive):
    """Create a sphinx gallery style thumbnail.
    tooltip and figure are self explanatory. Description could be a link to
    a document like in below example.
    Example usage:

    .. customgalleryitem::
        :name: Installation
        :description: Description of page
        :link: /path/to/page

    """

    required_arguments = 0
    optional_arguments = 4
    final_argument_whitespace = True
    option_spec = {'name': directives.unchanged,
                   'description': directives.unchanged,
                   'link': directives.unchanged}

    has_content = False
    add_index = False

    def run(self):
        try:
            if 'name' in self.options:
                name = self.options['name']

            if 'description' in self.options:
                description = self.options['description']
            else:
                raise ValueError('description not found')

            if 'link' in self.options:
                link = self.options['link']
            else:
                link = "code/qml_templates"

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []

        thumbnail_rst = TITLE_CARD_TEMPLATE.format(name=name,
                                                   description=description,
                                                   link=link)
        thumbnail = StringList(thumbnail_rst.split('\n'))
        thumb = nodes.paragraph()
        self.state.nested_parse(thumbnail, self.content_offset, thumb)
        return [thumb]
