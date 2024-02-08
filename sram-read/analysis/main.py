from utils import get_files, read_results
from analysis import ber

CHIPS_TO_EVALUATE = ["M42", "M47", "L45"]
FUNS_TO_RUN = [ber]


def main():
    all_results = get_files()
    for chip_id in CHIPS_TO_EVALUATE:
        print("Evaluating chip", chip_id)
        results = read_results(all_results[chip_id])
        for fun in FUNS_TO_RUN:
            print(fun(results))


if __name__ == "__main__":
    main()
