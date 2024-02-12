import questionary
import matplotlib.pyplot as plt
from utils import get_files, read_results
from analysis import (bit_error_rate,
                      autocorrelation,
                      fractional_hamming_weight,
                      inter_chip_hamming_weight)


class AnalysisFunction:
    def __init__(self, fun, inter=False):
        self.fun = fun
        self.inter = inter


ANALYSIS_FUNCTIONS = {
    "Autocorrelation (plot+data)": AnalysisFunction(autocorrelation),
    "Error rate analysis (plot+data)": AnalysisFunction(bit_error_rate),
    "Fractional Hamming Weight (plot+data)": AnalysisFunction(fractional_hamming_weight),
    "Inter-chip hamming distance": AnalysisFunction(inter_chip_hamming_distance, inter=True)
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
        if fun.inter:
            fun.fun(all_results)
        else:
            for i, chip_id in enumerate(chips_to_evaluate):
                fun.fun(all_results[chip_id], chip_id=chip_id, chip_index=i)
        plt.show()


if __name__ == "__main__":
    main()
