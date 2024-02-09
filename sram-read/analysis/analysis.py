import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.signal import correlate
from utils import Result, hamming_distance


def bit_error_rate(results: list[Result], chip):
    """
    Given a list of results, calculate the bit error rate.

    The calculation is done as follows: the first result is chosen
    as the nominal data, then the hamming distance is calculated
    between all the other results and the nominal data. The output
    is the average, divided by the amount of bits.

    results -- list of results to analyse the BER. Must have at least 2
    """
    if len(results) < 2:
        logging.error("Need at least two results to calculate BER.")
        return
    avg = 0.0
    for i in range(1, len(results)):
        # Calculate hamming distance between
        # each vector and the nominal vector
        avg += hamming_distance(results[0].data, results[i].data)

    # Get average hamming distance
    avg /= len(results)-1
    # Divide by amount of bits to get BER percentage
    ber = avg / len(results[0].data)
    return ber


def autocorrelation(results: list[Result], chip):
    """
    Given a list of results, calculate the autocorrelation.

    The calculation is done with the scipy.signal.correlate method.

    results -- list of results to analyse the autocorrelation.
               Only the first one will be used.
    """
    data = results[0].data.astype(np.int8)
    data[data == 0] = -1  # Replace 0s with -1s
    data = data.astype(np.float64)
    autocorr = correlate(data, data, mode='full', method='fft')
    autocorr /= np.max(autocorr)

    time_axis = np.arange(-len(data)+1, len(data))
    plt.scatter(time_axis, autocorr, marker='.')
    plt.title(f'Autocorrelation for SCuM {chip}')

    plt.tight_layout()
    plt.show()

    return None
