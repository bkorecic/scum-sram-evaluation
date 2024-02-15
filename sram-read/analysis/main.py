import questionary
import matplotlib.pyplot as plt
import logging
from utils import get_files, read_results
from analysis import (bit_error_rate,
                      autocorrelation,
                      fractional_hamming_weight,
                      inter_chip_hamming_distance,
                      stability,
                      inter_chip_min_entropy,
                      intra_chip_min_entropy,
                      calculate_frequencies)


# Define a custom formatter
formatter = logging.Formatter(
    '[%(levelname)s][%(asctime)s] | %(message)s', datefmt='%H:%M:%S')

# Configure logging with the custom formatter and set level to INFO
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s][%(asctime)s] | %(message)s',
                    datefmt='%H:%M:%S')


class AnalysisFunction:
    def __init__(self, fun, inter=False):
        self.fun = fun
        self.inter = inter


ANALYSIS_FUNCTIONS = {
    "Autocorrelation (plot+data)": AnalysisFunction(autocorrelation),
    "Error rate analysis (plot+data)": AnalysisFunction(bit_error_rate),
    "Fractional Hamming Weight (plot+data)": AnalysisFunction(fractional_hamming_weight),
    "Inter-chip hamming distance (data)": AnalysisFunction(inter_chip_hamming_distance, inter=True),
    "Calculate frequencies (files)": AnalysisFunction(calculate_frequencies),
    "Cell stability (plot+data), requires frequencies": AnalysisFunction(stability),
    "Inter-Chip min. entropy (data)": AnalysisFunction(inter_chip_min_entropy, inter=True),
    "Intra-Chip min. entropy (data)": AnalysisFunction(intra_chip_min_entropy)
}


def main():
    """
    Entry point. Get the files, read the results and run the functions.
    """
    all_files = get_files()

    chips_to_evaluate = questionary.checkbox(
        "Select the chips to evaluate",
        choices=all_files.keys()
    ).ask()

    functions_to_run = questionary.checkbox(
        "Select the functions to run",
        choices=ANALYSIS_FUNCTIONS.keys()
    ).ask()

    all_results = {chip_id: read_results(all_files[chip_id])
                   for chip_id in chips_to_evaluate}

    for fun_name in functions_to_run:
        fun = ANALYSIS_FUNCTIONS[fun_name]
        fig, ax = plt.subplots()
        if fun.inter:
            fun.fun(all_results=all_results, ax=ax, fig=fig)
        else:
            for i, chip_id in enumerate(chips_to_evaluate):
                fun.fun(results=all_results[chip_id], chip_id=chip_id,
                        chip_index=i, ax=ax, fig=fig)
        plt.show()


if __name__ == "__main__":
    main()
