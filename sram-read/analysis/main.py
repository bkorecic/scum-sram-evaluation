import questionary
import logging
from utils import get_files, read_results, ResultList
from analysis import (BitErrorRate,
                      Autocorrelation,
                      FractionalHammingWeight,
                      Frequencies,
                      Stability,
                      InterChipHammingDistance,
                      InterChipMinEntropy,
                      IntraChipMinEntropy)


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


analysis_suite = [BitErrorRate, Autocorrelation, FractionalHammingWeight,
                  Frequencies, Stability, InterChipHammingDistance,
                  InterChipMinEntropy, IntraChipMinEntropy]


def main():
    """
    Entry point. Get the files, read the results and run the functions.
    """
    all_files = get_files()

    chips_to_evaluate = questionary.checkbox(
        "Select the chips to evaluate",
        choices=all_files.keys()
    ).ask()

    analyses_to_run = questionary.checkbox(
        "Select the functions to run",
        choices=[questionary.Choice(title=c.name, value=c)
                 for c in analysis_suite]
    ).ask()

    all_results: list[ResultList] = [read_results(all_files[chip_id])
                                     for chip_id in chips_to_evaluate]

    for Analysis in analyses_to_run:
        a = Analysis(all_results=all_results)
        a.run()


if __name__ == "__main__":
    main()
