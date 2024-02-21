"""
This module includes functionality to plot the branch benchmarks normalized by the reference.
"""
import argparse, json, os, sys
import numpy as np

########################################################################
# Parsing arguments
########################################################################
def parse_args():
    """Parse external arguments provided to the script."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph_name",
        type=str,
        default="Benchmarks",
        nargs="?",
        help="Name of the set of benchmarks.",
    )

    parser.add_argument(
        "--filename_XUBM_ref",
        type=str,
        default="benchmark_reference/benchmarks_xubm.json",
        nargs="?",
        help="Name of the JSON-XUBM file with reference benchmarks.",
    )

    parser.add_argument(
        "--filename_XUBM",
        type=str,
        default="benchmark_results/benchmarks_xubm.json",
        nargs="?",
        help="Name of the JSON-XUBM file with most recent benchmarks.",
    )

    return parser.parse_args()

def format_plot_data(_ref_data, _data):
    """Here we format the data coming from JSON files in two arrays with graph data.

    Args:
        _ref_data (JSON-XUBM): reference benchmarks data
        _data (JSON-XUBM): local (or branch) benchmarks data

    Returns:
        tuple: data for x and y axis
    """

    ratios= []
    names=[]
    for test_name in _ref_data:
        names += [test_name, ]
        ratios += [_data[test_name]["runtime"] / _ref_data[test_name]["runtime"], ]

    return names, ratios

if __name__ == "__main__":
    args = parse_args()

    if ((os.stat(args.filename_XUBM_ref).st_size == 0) or (os.stat(args.filename_XUBM).st_size == 0)):
        print(args.filename_XUBM_ref + " or " + args.filename_XUBM + " is empty. Interrupting program.")
        sys.exit(0)

    with open(args.filename_XUBM_ref, 'r', encoding="utf-8") as file:
        ref_data = json.load(file)
    ref_commit, ref_data = next(reversed(ref_data.items())) # last stored reference benchmark

    with open(args.filename_XUBM, 'r', encoding="utf-8") as file:
        data = json.load(file)
    commit, data = next(reversed(data.items())) # last stored local benchmark

    benchmark_names, benchmark_ratios = format_plot_data(ref_data, data)

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    number_of_benchmarks = len(benchmark_names)

    fig, ax = plt.subplots(figsize=(6.4, 0.5*number_of_benchmarks))

    colormat=np.where(np.array(benchmark_ratios)>1.0, 'r','b')

    ax.barh(benchmark_names, benchmark_ratios, color=colormat)

    ax.axvline(x = 1.0, color = 'k', linestyle = '--', zorder=0)

    ax.set_xlabel('runtime / reference runtime')
    ax.set_title(args.graph_name+" (reference: "+ref_commit+")")

    regr_patch = mpatches.Patch(color='red', label='Regression')
    prog_patch = mpatches.Patch(color='blue', label='Improvement')

    plt.legend(title='Performance', handles=[regr_patch, prog_patch])

    plt.savefig('benchmark_results/'+args.graph_name+'.png', bbox_inches='tight')
