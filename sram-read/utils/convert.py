import sys
import pathlib
import pickle

EXPECTED_DATA_LEN = 112641


def convert(csv_path):
    """
    Utility for converting a CSV results file (old format)
    into a pickle results file (new format). Receives the path
    to a CSV file and writes to a .pickle file.
    """
    print('Converting ', csv_path)
    pickle_path = csv_path.with_suffix('.pickle')
    with open(csv_path, 'r') as csv_file, \
            open(pickle_path, 'wb') as pickle_file:
        for i, line in enumerate(csv_file.readlines()):
            line = line.split(',')
            t0 = float(line[0])
            t1 = float(line[1])
            if len(line[2]) != EXPECTED_DATA_LEN:
                print(f'Row {i} has incomplete or corrupted data')
                sys.exit(1)
            data = bytearray.fromhex(line[2])
            # reverse each 8 byte chunk because of little endianess
            # previous format read uint64_t chunks and new format reads uint8_t
            data = b''.join([data[i:i+8][::-1]
                            for i in range(0, len(data), 8)])
            pickle.dump([t0, t1, data], pickle_file)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: ', sys.argv[0], '<csv-path>')
        sys.exit(1)

    path = pathlib.Path(sys.argv[1])
    if not path.is_file():
        print('invalid file path')
        sys.exit(1)
    convert(path)
