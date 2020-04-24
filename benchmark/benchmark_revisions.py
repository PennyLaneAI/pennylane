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
from pathlib import Path
import stat
import shutil
import subprocess

from benchmark import col


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = newPath
        self.savedPath = None

    def __enter__(self):
        self.savedPath = Path.cwd()
        os.chdir(str(self.newPath))

    def __exit__(self, etype, value, traceback):
        os.chdir(str(self.savedPath))


# benchmarking tool version
__version__ = "0.1.0"


def remove_readonly(func, path, excinfo):
    """Remove readonly flag from file and retry.

    This method is intended to be used with shutil.rmtree.
    """
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

    revisions_directory = Path.home() / "pennylane.benchmarks" / "revisions"

    if not revisions_directory.exists():
        revisions_directory.mkdir(parents=True)

    for revision in args.revisions:
        print(">>> Running benchmark for revision {}".format(col(revision, "red")))
        pl_directory = revisions_directory / revision

        # We first make sure we get the latest version of the desired revision
        if pl_directory.exists():
            print(">>> Revision found locally, updating...")
            with cd(pl_directory):
                # Check if we're on a detached HEAD (i.e. for version revisions)
                res = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "HEAD"],
                    stdout=subprocess.PIPE,
                    check=True,
                )

                if "HEAD" not in str(res.stdout):
                    subprocess.run(["git", "checkout", revision, "-q"])
                    subprocess.run(["git", "pull", "-q"])
        else:
            try:
                # If we already downloaded a revision we don't need to clone the whole
                # PL repository. Instead we just copy the first revision in the directory
                # checkout will then get the correct revision
                first_revision = next((x for x in revisions_directory.iterdir() if x.is_dir()))
                print(">>> Revision not found locally, copying...")
                print(str(first_revision), str(revisions_directory / revision))
                shutil.copytree(str(first_revision), str(revisions_directory / revision))

            except StopIteration:
                # Didn't find a local revision to copy, use git to clone
                print(">>> Revision not found locally, cloning...")

                pl_directory.mkdir()
                with cd(revisions_directory):
                    subprocess.run(
                        ["git", "clone", "https://www.github.com/xanaduai/pennylane", revision, "-q"],
                    )

            with cd(pl_directory):
                res = subprocess.run(["git", "checkout", revision, "-q"], check=True)

            # An error occured during checkout, so the revision does not exist
            if res.returncode != 0:
                print(
                    col(
                        ">>> Couldn't check out revision {}, deleting temporary data".format(
                            revision
                        ),
                        "red",
                    )
                )

                # Regular rmtree hickups with read-only files. We thus use an errorhandler that tries to mark them
                # as writeable and retries. See also
                # https://stackoverflow.com/questions/1889597/deleting-directory-in-python/1889686
                shutil.rmtree(str(pl_directory), onerror=remove_readonly)
                continue

        benchmark_file_path = Path(__file__).parent / "benchmark.py"
        print("benchmark_file_path = ", benchmark_file_path)
        benchmark_env = os.environ.copy()
        benchmark_env["PYTHONPATH"] = str(pl_directory) + ";" + benchmark_env["PATH"]
        subprocess.run(
            ["python3", str(benchmark_file_path)] + unknown_args + ["--noinfo"],
            env=benchmark_env,
            check=True,
        )


if __name__ == "__main__":
    cli()
