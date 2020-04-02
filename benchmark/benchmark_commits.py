# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Benchmarking tool for different commits
"""
# pylint: disable=import-outside-toplevel,invalid-name
import argparse
import importlib
import subprocess
import sys
import os
import shutil
import pkg_resources

import numpy as np
import zipfile

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

class prepend_to_path:
    """Context manager for prepending a path to the system path"""
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, etype, value, traceback):
        sys.path.pop(0)

class temporary_directory:
    """Context manager for prepending a path to the system path"""
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        os.mkdir(self.path)

    def __exit__(self, etype, value, traceback):
        print("Deleting temporary installation")
        shutil.rmtree(self.path)

# benchmarking tool version
__version__ = "0.1.0"

def cli():
    """Parse the command line arguments, perform the requested action.
    """
    #TODO: Rename commit to revision which is the general git term
    # Use git rev-parse <revision> to get the SHA hash of the commit
    parser = argparse.ArgumentParser(description="PennyLane benchmarking tool for commits")
    parser.add_argument(
        "-c",
        "--commits",
        type=lambda x: x.split(","),
        help="comma-separated list of commits to run the benchmark on",
    )

    args, unknown_args = parser.parse_known_args()

    for commit in args.commits:
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), commit)
        with temporary_directory(directory):
            print(">>> Downloading {}".format(commit))
            subprocess.run([
                "pip",
                "download",
                "git+https://www.github.com/XanaduAI/pennylane@{}".format(commit),
                "-d",
                directory,
                "--no-deps",
                "-q",
            ])

            zip_path = os.path.join(directory, os.listdir(directory)[0])

            print(">>> Unpacking {}".format(zip_path))
            with zipfile.ZipFile(zip_path, 'r') as zip:
                zip.extractall(directory)

            print(">>> Setup {}".format(commit))
            with cd(os.path.join(directory, "PennyLane")):
                subprocess.run(["python", "setup.py", "-q", "bdist_wheel"])

            pl_path = os.path.join(directory, "PennyLane")
            with prepend_to_path(pl_path):
                print(">>> Reload modules")
                del_keys = []
                for key in sys.modules:
                    if "pennylane" in key:
                        del_keys.append(key)

                for key in del_keys:
                    del sys.modules[key]

                # TODO: there are still some old imports lingering around here...
                # Somehow they can surely be removed

                print(">>> Benchmark {}".format(commit))
                subprocess.run(["python", "benchmark.py"] + unknown_args)



if __name__ == "__main__":
    cli()
