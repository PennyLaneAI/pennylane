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
import locale
import os
import subprocess
from pathlib import Path

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


def get_current_git_toplevel():
    """Finds the current git toplevel.

    Returns:
        Union[Path,NoneType]: The current git toplevel's path or None.
    """
    res = subprocess.run(["git", "rev-parse", "--show-toplevel", "-q"], stdout=subprocess.PIPE)

    if res.returncode == 0:
        return Path(res.stdout.decode(locale.getpreferredencoding()).strip())

    return None


def cli():
    """Parse the command line arguments, perform the requested action.
    """
    parser = argparse.ArgumentParser(description="PennyLane benchmarking tool for revisions")
    parser.add_argument(
        "-r",
        "--revisions",
        type=lambda x: x.split(","),
        help='Comma-separated list of revisions to run the benchmark on. Use "here" for the current git toplevel.',
    )

    # Only parse revisions, other args will go to the benchmarking script
    args, unknown_args = parser.parse_known_args()

    revisions_directory = Path.home() / ".pennylane" / "benchmarks" / "revisions"

    toplevel = get_current_git_toplevel()

    if revisions_directory.exists():
        with cd(revisions_directory):
            # Make really sure we don't reset the current git
            if toplevel == Path.cwd():
                raise Exception(
                    "Git accidently ended up in the current directory. Stopping to not cause any harm."
                )

            subprocess.run(["git", "fetch", "origin", "-q"], check=True)
            subprocess.run(["git", "reset", "--hard", "origin/master", "-q"], check=True)
    else:
        revisions_directory.mkdir(parents=True)

        subprocess.run(
            [
                "git",
                "clone",
                "https://www.github.com/xanaduai/pennylane",
                str(revisions_directory),
            ],
            check=True,
        )

    for revision in args.revisions:
        print(">>> Running benchmark for revision {}".format(col(revision, "red")))

        if revision == "here":
            toplevel = get_current_git_toplevel()

            if not toplevel:
                print(
                    col(">>> Wasn't able to determine the current git toplevel, skipping...", "red")
                )

                continue

            pl_directory = toplevel
        else:
            pl_directory = revisions_directory

            toplevel = get_current_git_toplevel()

            with cd(pl_directory):
                # Make really sure we don't reset the current git
                if toplevel == Path.cwd():
                    raise Exception(
                        "Git accidently ended up in the current directory. Stopping to not cause any harm."
                    )

                subprocess.run(["git", "fetch", "origin", "-q"], check=True)
                subprocess.run(["git", "reset", "--hard", revision, "-q"], check=True)

        benchmark_file_path = Path(__file__).parent / "benchmark.py"
        benchmark_env = os.environ.copy()
        benchmark_env["PYTHONPATH"] = str(pl_directory) + ";" + benchmark_env["PATH"]

        subprocess.run(
            ["python3", str(benchmark_file_path)] + unknown_args + ["--noinfo"],
            env=benchmark_env,
            check=True,
        )


if __name__ == "__main__":
    cli()
