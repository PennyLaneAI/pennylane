"""
This module includes functionality to convert JSON-XUBM runtime data to a CSV file.
"""
import argparse, json, os, csv
########################################################################
# Parsing arguments
########################################################################
def parse_args():
    """Parse external arguments provided to the script."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filename_bm_references",
        type=str,
        default="benchmark_reference/benchmarks_xubm.json",
        nargs="?",
        help="Name of the JSON file with references.",
    )

    parser.add_argument(
        "--filename_bm_commit",
        type=str,
        default="benchmark_results/benchmarks_xubm.json",
        nargs="?",
        help="Name of the JSON file with local (commit) benchmarks.",
    )

    parser.add_argument(
        "--filename_bm_merged",
        type=str,
        default="benchmark_results/all_benchmarks_xubm.json",
        nargs="?",
        help="Name of the JSON file merging all benchmarks.",
    )

    parser.add_argument(
        "--filename_runtimes",
        type=str,
        default="benchmark_results/all_runtimes.csv",
        nargs="?",
        help="Name of the CSV file.",
    )

    return parser.parse_args()

def create_runtime_CSV_data_file(stored_data, args):
    """This function exports runtimes from the JSON data provided to a CSV file.

    Args:
        stored_data: benchmark JSON data.
        args: customization arguments

    """
    with open(args.filename_runtimes, 'w') as csv_filename:
        csv_writer = csv.writer(csv_filename)
        # We'll assume for now that all stored benchmarks for different branches/tags have the same set of benchmarks (or tests).
        first_line = ["branch/tag", ]
        for test_name in stored_data[next(iter(stored_data))]:
            first_line += [test_name,]
        csv_writer.writerow(first_line)

        for branch_or_tag in stored_data:
            next_line = [branch_or_tag, ]
            for test_name in stored_data[branch_or_tag]:
                next_line += [stored_data[branch_or_tag][test_name]["runtime"]]
            csv_writer.writerow(next_line)
    print("CSV file done!")

if __name__ == "__main__":
    parsed_args = parse_args()

    if os.stat(parsed_args.filename_bm_references).st_size != 0:
        with open(parsed_args.filename_bm_references, 'r', encoding="utf-8") as file:
            benchmark_data_reference = json.load(file)
    else:
        benchmark_data_reference = {}

    if os.stat(parsed_args.filename_bm_commit).st_size != 0:
        with open(parsed_args.filename_bm_commit, 'r', encoding="utf-8") as file:
            benchmark_data_commit = json.load(file)
    else:
        benchmark_data_commit = {}

    benchmark_data = benchmark_data_reference | benchmark_data_commit

    with open(parsed_args.filename_bm_merged, 'w', encoding="utf-8") as file:
        json.dump(benchmark_data, fp=file)

    create_runtime_CSV_data_file(benchmark_data, parsed_args)
