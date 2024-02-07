from utils import get_files, read_results
from analysis import ber

CHIPS_TO_EVALUATE = ["M47"]
FUNS_TO_RUN = [ber]


def main():
    all_results = get_files()
    for chip_id in CHIPS_TO_EVALUATE:
        results = read_results(all_results[chip_id])
        for fun in FUNS_TO_RUN:
            fun(results)


if __name__ == "__main__":
    main()
