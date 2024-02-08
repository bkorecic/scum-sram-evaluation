import pathlib
import pickle
import numpy as np


class Result:
    def __init__(self,
                 start_timestamp: float,
                 end_timestamp: float,
                 data: bytes):
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.data = np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def hamming_distance(bit_arr1: np.ndarray, bit_arr2: np.ndarray) -> int:
    return np.sum(bit_arr1 ^ bit_arr2)


def get_files() -> dict:
    """
    Gets the files in the sibling results/ directory.
    Puts them into a dictionary where the key is the chip ID
    """
    ret_files = dict()
    results_dir = pathlib.Path(__file__).parent.parent / 'results'
    for f in results_dir.iterdir():
        chip_id = f.parts[-1].split('-')[0]
        if chip_id in ret_files.keys():
            ret_files[chip_id].append(f)
        else:
            ret_files[chip_id] = [f]
    return ret_files


def read_results(files: list[pathlib.Path]) -> list[Result]:
    """
    Read, unpickle, merge and return a list of result files

    files -- list of paths to result files
    """
    files.sort()  # Sort to read chronologically
    results = []
    for fp in files:
        with open(fp, 'rb') as f:
            try:
                while True:
                    data = pickle.load(f)
                    results.append(Result(*data))
            except EOFError:
                pass
    return results

