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
# pylint: disable=subprocess-run-check
import argparse
import os
import stat
import shutil
import subprocess

# ANSI escape sequences for terminal colors
Colors = {
    "red": "\033[31m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[95m",
}
RESET = "\033[0m"


def col(text, color):
    """Wraps the given text in color ANSI sequences.

    Args:
        text  (str): text to print
        color (str): ANSI color code

    Returns:
        str: text wrapped with ANSI codes
    """
    return Colors[color] + text + RESET


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)
        self.savedPath = None

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


# benchmarking tool version
__version__ = "0.1.0"

def remove_readonly(func, path, excinfo):
    # pylint: disable=unused-argument
    os.chmod(path, stat.S_IWRITE)
    func(path)

def cli():
    """Parse the command line arguments, perform the requested action.
    """
    parser = argparse.ArgumentParser(description="PennyLane benchmarking tool for revisions")
    parser.add_argument(
        "-r",
        "--revisions",
        type=lambda x: x.split(","),
        help="comma-separated list of revisions to run the benchmark on",
    )

    # Only parse revisions, other args will go to the benchmarking script
    args, unknown_args = parser.parse_known_args()

    revisions_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "revisions")
    if not os.path.exists(revisions_directory):
        os.mkdir(revisions_directory)

    for revision in args.revisions:
        print(">>> Running benchmark for revision {}".format(col(revision, "red")))
        pl_directory = os.path.join(revisions_directory, revision)

        # We first make sure we get the latest version of the desired revision
        if os.path.exists(pl_directory):
            print(">>> Revision found locally, updating...")
            with cd(pl_directory):
                # Check if we're on a detached HEAD (i.e. for version revisions)
                res = subprocess.run(
                    "git rev-parse --abbrev-ref --symbolic-full-name HEAD",
                    capture_output=True,
                    check=True,
                )

                if "HEAD" not in str(res.stdout):
                    subprocess.run("git checkout {} -q".format(revision))
                    subprocess.run("git pull -q")
        else:
            print(">>> Revision not found locally, cloning...")
            os.mkdir(pl_directory)
            with cd(revisions_directory):
                subprocess.run("git clone https://www.github.com/xanaduai/pennylane {} -q".format(revision))
                with cd(pl_directory):
                    res = subprocess.run("git checkout {} -q".format(revision))

                # An error occured during checkout, so the revision does not exist
                if res.returncode != 0:
                    print(col(">>> Couldn't check out revision {}, deleting temporary data".format(revision), "red"))

                    # Regular rmtree hickups with read-only files. We thus use an errorhandler that tries to mark them
                    # as writeable and retries. See also
                    # https://stackoverflow.com/questions/1889597/deleting-directory-in-python/1889686
                    shutil.rmtree(pl_directory, onerror=remove_readonly)
                    continue


        benchmark_file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "benchmark.py"
        )
        benchmark_env = os.environ.copy()
        benchmark_env["PYTHONPATH"] = pl_directory + ";" + benchmark_env["PATH"]
        subprocess.run(
            ["python3", benchmark_file_path] + unknown_args + ["--noinfo"],
            env=benchmark_env,
            check=True,
        )


if __name__ == "__main__":
    cli()
