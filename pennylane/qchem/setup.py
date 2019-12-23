# Copyright 2019 Xanadu Quantum Technologies Inc.
r"""
Setup
=====

Setup file for package.

"""
from setuptools import setup

with open("pennylane_qchem/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = ["pennylane", "openfermion", "openfermionpsi4", "openfermionpyscf"]

info = {
    "name": "PennyLane-Qchem",
    "version": version,
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "alaindelgado@xanadu.ai",
    "url": "http://xanadu.ai",
    "packages": ["pennylane_qchem"],
    "description": "Package for quantum chemistry applications",
    "long_description": open("README.rst").read(),
    "provides": ["pennylane_qchem"],
    "install_requires": requirements,
    # 'extras_require': extra_requirements,
    # 'command_options': {
    #     'build_sphinx': {
    #         'version': ('setup.py', version),
    #         'release': ('setup.py', version)}}
}

classifiers = [
    "Development Status :: 1 - Planning",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **info)
