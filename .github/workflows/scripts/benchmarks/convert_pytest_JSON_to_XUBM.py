"""
This module includes functionality to convert the provided pytest-benchmark JSON file to
JSON-XUBM (XanadU BenchMarks) format.
"""
import argparse, json, os, sys
########################################################################
# Parsing arguments
########################################################################
def parse_args():
    """Parse external arguments provided to the script."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filename",
        type=str,
        default="benchmarks.json",
        nargs="?",
        help="Name of the Pytest-benchmark JSON file to be imported.",
    )

    parser.add_argument(
        "--filename_XUBM",
        type=str,
        default="benchmarks_xubm.json",
        nargs="?",
        help="Name of the XUBM JSON file to be exported.",
    )

    parser.add_argument(
        "--author",
        type=str,
        nargs="?",
        help="User associated with the commit.",
    )

    parser.add_argument(
        "--github_reference",
        type=str,
        default="",
        nargs="?",
        help="The fully-formed ref of the branch or tag that triggered the workflow run.",
    )

    return parser.parse_args()

def create_benchmark_XUBM(stored_data, data, args):
    """This function converts the JSON file provided by pytest-benchmark to the JSON-XUBM (XanadU BenchMark) format.

    Args:
        stored_data: data we already have in-place or an empty dictionary.
        data: Pytest-benchmark JSON data
        args: customization arguments

    Returns:
        JSON-XUBM: JSON data
    """
    commit_reference = args.github_reference if (args.github_reference != "") else data["commit_info"]["branch"]
    if commit_reference not in stored_data:
        stored_data[commit_reference] = {}
    else:
        print("data already exist. doing nothing.")
        return stored_data

    hash_commit = hash(data["commit_info"]["time"]) + hash(commit_reference)
    for benchmark in data["benchmarks"]:
        benchmark_xubm = {}
        benchmark_xubm["uid"] = hash_commit + hash(benchmark["fullname"]) + hash(json.dumps(benchmark["params"]))
        benchmark_xubm["date"] = data["commit_info"]["time"]
        benchmark_xubm["fullname"] = benchmark["fullname"]
        benchmark_xubm["gitID"] = data["commit_info"]["id"]
        benchmark_xubm["runs"] = benchmark["stats"]["iterations"]
        benchmark_xubm["params"] = benchmark["params"]
        benchmark_xubm["runtime"] = benchmark["stats"]["mean"]
        benchmark_xubm["timeUnit"] = "seconds"
        benchmark_xubm["project"] = data["commit_info"]["project"]
        benchmark_xubm["machine"] = data["machine_info"]
        benchmark_xubm["users"] = args.author
        # Pytest-benchmark provides a reach variety of statistics. This will be stored as metadata.
        benchmark_xubm["metadata"] = {"stats": benchmark["stats"]}

        stored_data[commit_reference][benchmark["name"]] = benchmark_xubm
    return stored_data

if __name__ == "__main__":
    parsed_args = parse_args()

    if os.stat(parsed_args.filename).st_size == 0:
        print(parsed_args.filename+" is empty. Interrupting program.")
        sys.exit(0)

    with open(parsed_args.filename, 'r', encoding="utf-8") as file:
        pytest_data = json.load(file)

    # Check if the JSON-XUBM file already exist, or if we'll start with and empty dictionary.
    if os.path.isfile(parsed_args.filename_XUBM):
        with open(parsed_args.filename_XUBM, 'r', encoding="utf-8") as file:
            xubm_database = json.load(file)
    else:
        xubm_database = {}

    XUBM_data = create_benchmark_XUBM(xubm_database, pytest_data, parsed_args)

    with open(parsed_args.filename_XUBM, 'w', encoding="utf-8") as file:
        json.dump(XUBM_data, fp=file)
