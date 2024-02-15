import pathlib
import pickle
import numpy as np
import constants
import logging

numpy_data_dir = pathlib.Path(__file__).parent.absolute() / 'numpy_data'
numpy_data_dir.mkdir(exist_ok=True)


class Result:
    """
    Class to represent a single readout from a chip
    """

    def __init__(self,
                 start_timestamp: float,
                 end_timestamp: float,
                 data: bytes):
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.data = np.unpackbits(np.frombuffer(data, dtype=np.uint8))


class ResultList (list):
    """
    Class to represent a list of results from a chip. Has the same interface
    as a python list but adds a "chip_id" attribute
    """

    def __init__(self, chip_id: str, *args):
        super().__init__(*args)
        self.chip_id = chip_id


def hamming_distance(bit_arr1: np.ndarray, bit_arr2: np.ndarray) -> int:
    """
    Calculate the Hamming distance between two bit arrays
    """
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


def read_results(files: list[pathlib.Path]) -> ResultList:
    """
    Read, unpickle, merge, trim and return a list of result files

    files -- list of paths to result files
    """
    files.sort()  # Sort to read chronologically
    chip_id = files[0].parts[-1].split('-')[0]
    results = ResultList(chip_id)
    for fp in files:
        with open(fp, 'rb') as f:
            try:
                while len(results) < constants.READINGS_TO_ANALYZE:
                    data = pickle.load(f)
                    results.append(Result(*data))
            except EOFError:
                pass
    if len(results) != constants.READINGS_TO_ANALYZE:
        logging.warning(f"Expected at least {constants.READINGS_TO_ANALYZE} readings for chip {chip_id}, got {len(results)}.")
    return results
