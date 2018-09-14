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

#!/usr/bin/env python3
import sys
import os
from setuptools import setup

with open("openqml_sf/_version.py") as f:
	version = f.readlines()[-1].split()[-1].strip("\"'")


requirements = [
    "strawberryfields",
    "openqml"
]

info = {
    'name': 'OpenQML-SF',
    'version': version,
    'maintainer': 'Xanadu Inc.',
    'maintainer_email': 'nathan@xanadu.ai',
    'url': 'http://xanadu.ai',
    'license': 'Apache License 2.0',
    'packages': [
                    'openqml_sf'
                ],
    'entry_points': {
        'openqml.plugins': [
            'strawberryfields.fock = openqml_sf:StrawberryFieldsFock',
            'strawberryfields.gaussian = openqml_sf:StrawberryFieldsGaussian'
            ],
        },
    'description': 'Open source library for continuous-variable quantum computation',
    'long_description': open('README.rst').read(),
    'provides': ["openqml_sf"],
    'install_requires': requirements,
    # 'extras_require': extra_requirements,
    'command_options': {
        'build_sphinx': {
            'version': ('setup.py', version),
            'release': ('setup.py', version)}}
}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3 :: Only',
    "Topic :: Scientific/Engineering :: Physics"
]

setup(classifiers=classifiers, **(info))
