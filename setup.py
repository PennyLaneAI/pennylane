# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup file for package installation."""
# pylint: disable=unspecified-encoding, consider-using-with

from setuptools import find_packages, setup

with open("pennylane/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "numpy",
    "scipy",
    "networkx",
    "rustworkx>=0.14.0",
    "autograd",
    "tomlkit",
    "appdirs",
    "autoray>=0.6.11",
    "cachetools",
    "pennylane-lightning>=0.40",
    "requests",
    "typing_extensions",
    "packaging",
    "diastatic-malt",
]

info = {
    "name": "PennyLane",
    "version": version,
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "https://github.com/PennyLaneAI/pennylane",
    "license": "Apache License 2.0",
    "packages": find_packages(where="."),
    "entry_points": {
        # TODO: rename entry point 'pennylane.plugins' to 'pennylane.devices'.
        # This requires a rename in the setup file of all devices, and is best done during another refactor
        "pennylane.plugins": [
            "default.qubit = pennylane.devices:DefaultQubit",
            "default.gaussian = pennylane.devices:DefaultGaussian",
            "default.mixed = pennylane.devices.default_mixed:DefaultMixed",
            "reference.qubit = pennylane.devices.reference_qubit:ReferenceQubit",
            "null.qubit = pennylane.devices.null_qubit:NullQubit",
            "default.qutrit = pennylane.devices.default_qutrit:DefaultQutrit",
            "default.clifford = pennylane.devices.default_clifford:DefaultClifford",
            "default.qutrit.mixed = pennylane.devices.default_qutrit_mixed:DefaultQutritMixed",
            "default.tensor = pennylane.devices.default_tensor:DefaultTensor",
        ],
        "console_scripts": ["pl-device-test=pennylane.devices.tests:cli"],
    },
    "description": "PennyLane is a cross-platform Python library for quantum computing, quantum machine learning, and quantum chemistry. Train a quantum computer the same way as a neural network.",
    "long_description": open("README.md").read(),
    "long_description_content_type": "text/markdown",
    "provides": ["pennylane"],
    "install_requires": requirements,
    "extras_require": {"kernels": ["cvxpy", "cvxopt"]},
    "package_data": {"pennylane": ["devices/tests/pytest.ini", "drawer/plot.mplstyle"]},
    "include_package_data": True,
}

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
