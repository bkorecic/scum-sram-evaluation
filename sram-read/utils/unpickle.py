import sys
import pickle
import pathlib


def unpickle(pickle_path):
    """
    Receives a path to a pickle results file.
    De-serializes it and prints the timestamps
    and the first 50 bytes of the data.
    """
    data = []
    with open(pickle_path, 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass

    for row in data:
        print(row[0], row[1], row[2][:50].hex())


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: ', sys.argv[0], '<pickle-file-path>')
        sys.exit(1)

    path = pathlib.Path(sys.argv[1])
    if not path.is_file():
        print('invalid file path')
        sys.exit(1)
    unpickle(path)
