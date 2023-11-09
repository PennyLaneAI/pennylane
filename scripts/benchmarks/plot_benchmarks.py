"""
This module includes functionality to plot the branch benchmarks normalized by the reference.
"""
import argparse, json
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
        default="Benchmark set",
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
        default="benchmarks_xubm.json",
        nargs="?",
        help="Name of the JSON-XUBM file with most recent benchmarks.",
    )

    return parser.parse_args()

def format_plot_data(ref_data, data):
    """Here we format the data coming from JSON files in two arrays with graph data.

    Args:
        ref_data (JSON-XUBM): reference benchmarks data
        data (JSON-XUBM): local (or branch) benchmarks data

    Returns:
        tuple: data for x and y axis
    """

    benchmark_ratios= []
    benchmark_names=[]
    for ref_benchmark, benchmark in zip(ref_data["xubm"], data["xubm"]):
        if [ref_benchmark["name"] == benchmark["name"]]:
            benchmark_names += [benchmark["name"], ]
            benchmark_ratios += [benchmark["runtime"] / ref_benchmark["runtime"], ]

    return benchmark_names, benchmark_ratios

if __name__ == "__main__":
    args = parse_args()

    with open(args.filename_XUBM_ref, 'r', encoding="utf-8") as file:
        ref_data = json.load(file)

    with open(args.filename_XUBM, 'r', encoding="utf-8") as file:
        data = json.load(file)

    benchmark_names, benchmark_ratios = format_plot_data(ref_data, data)

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    colormat=np.where(np.array(benchmark_ratios)>1.0, 'r','b')

    ax.barh(benchmark_names, benchmark_ratios, color=colormat)

    ax.axvline(x = 1.0, color = 'k', linestyle = '--', zorder=0)

    ax.set_xlabel('runtime / reference runtime')
    ax.set_title(args.graph_name)

    regr_patch = mpatches.Patch(color='red', label='Regression')
    prog_patch = mpatches.Patch(color='blue', label='Improvement')

    plt.legend(title='Performance', handles=[regr_patch, prog_patch])

    plt.savefig(args.graph_name+'.png', bbox_inches='tight')
