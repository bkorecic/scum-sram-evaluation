from utils import get_files, read_results
from analysis import bit_error_rate

# Add the chips you want to evaluate here
CHIPS_TO_EVALUATE = ["M42", "M47", "L45"]
FUNCTIONS_TO_RUN = [bit_error_rate]  # Add the functions you want to run here


def main():
    """
    Entry point. Get the files, read the results and run the functions.
    """
    all_results = get_files()
    for chip_id in CHIPS_TO_EVALUATE:
        print("Evaluating chip", chip_id)
        results = read_results(all_results[chip_id])
        for fun in FUNCTIONS_TO_RUN:
            print(fun(results))


if __name__ == "__main__":
    main()
