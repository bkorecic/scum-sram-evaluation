import questionary
import matplotlib.pyplot as plt
from utils import get_files, read_results
from analysis import bit_error_rate, autocorrelation

ANALYSIS_FUNCTIONS = {
    "Autocorrelation (plot)": autocorrelation,
    "Error rate analysis (plot+data)": bit_error_rate
}


def main():
    """
    Entry point. Get the files, read the results and run the functions.
    """
    all_results = get_files()

    chips_to_evaluate = questionary.checkbox(
        "Select the chips to evaluate",
        choices=all_results.keys()
    ).ask()

    functions_to_run = questionary.checkbox(
        "Select the functions to run",
        choices=ANALYSIS_FUNCTIONS.keys()
    ).ask()

    for fun in functions_to_run:
        fig, ax = plt.subplots()
        for chip_id in chips_to_evaluate:
            results = read_results(all_results[chip_id])
            ANALYSIS_FUNCTIONS[fun](results, chip_id=chip_id, fig=fig, ax=ax)
        plt.show()


if __name__ == "__main__":
    main()
