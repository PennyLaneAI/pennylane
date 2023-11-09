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

    return parser.parse_args()

def create_benchmark_XUBM(data, args):
    """This function converts the JSON file provided by pytest-benchmark to the JSON-XUBM (XanadU BenchMark) format.

    Args:
        data: Pytest-benchmark JSON data
        args: customization arguments

    Returns:
        JSON-XUBM: XUBM data
    """
    new_data = {"xubm": []}

    hash_commit = hash(data["commit_info"]["time"]) + hash(data["commit_info"]["branch"])
    for benchmark in data["benchmarks"]:
        benchmark_xubm = {}
        #I propose an alternative way of calcuting the unique identifier with information provided by pytest:
        benchmark_xubm["uid"] = hash_commit + hash(benchmark["fullname"]) + hash(json.dumps(benchmark["params"]))
        benchmark_xubm["date"] = data["commit_info"]["time"]
        benchmark_xubm["name"] = benchmark["name"]
        # pytest provide more complete information than what we defined for "suite" ["Directory & filename"].
        # It will provide the full address of the test, relative to the tests folder.
        # Ex: "devices/test_default_qubit_jax.py::TestQNodeIntegration::test_qubit_circuit_with_jit"
        # Because of how Pennylane structure its test suite, the reference position will vary (we have more than one tests directory.)
        # I will follow pytest and call its fullname for now:
        benchmark_xubm["fullname"] = benchmark["fullname"]
        benchmark_xubm["gitID"] = data["commit_info"]["id"]
        benchmark_xubm["runs"] = benchmark["stats"]["iterations"]
        benchmark_xubm["params"] = benchmark["params"]
        # benchmark_xubm["runtime"] = benchmark["stats"]["mean"] * (1 + random.uniform(-0.25, 0.25))
        benchmark_xubm["runtime"] = benchmark["stats"]["mean"]
        #Results are always in second:
        benchmark_xubm["timeUnit"] = "seconds"
        benchmark_xubm["project"] = data["commit_info"]["project"]
        #Pytest provides a lot of machine info, we may consider filter it:
        benchmark_xubm["machine"] = data["machine_info"]

        benchmark_xubm["users"] = args.author
        # Pytest-benchmark provides a reach variety of statistics, I'm storing it as metadata.
        benchmark_xubm["metadata"] = {"stats": benchmark["stats"]}

        new_data["xubm"] += [benchmark_xubm, ]
    return new_data

if __name__ == "__main__":
    parsed_args = parse_args()

    if os.stat(parsed_args.filename).st_size == 0:
        print(parsed_args.filename+" is empty. Interrupting program.")
        sys.exit(0)

    with open(parsed_args.filename, 'r', encoding="utf-8") as file:
        pytest_data = json.load(file)

    XUBM_data = create_benchmark_XUBM(pytest_data, parsed_args)

    with open(parsed_args.filename_XUBM, 'w', encoding="utf-8") as file:
        json.dump(XUBM_data, fp=file)
