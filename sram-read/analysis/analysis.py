import logging
from utils import Result, hamming_distance


def ber(results: list[Result]):
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
