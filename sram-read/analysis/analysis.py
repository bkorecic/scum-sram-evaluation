import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import binom
from utils import Result, hamming_distance
from matplotlib.ticker import PercentFormatter


def bit_error_rate(results: list[Result], **kwargs):
    """
    Given a list of results, calculate metrics related to the bit error rate.

    The calculation is done as follows: the first result is chosen
    as the nominal data, then the hamming distance is calculated
    between all the other results and the nominal data. The output
    is the average, divided by the amount of bits.

    results -- list of results to analyse the BER. Must have at least 2
    """
    if len(results) < 1001:
        logging.error("Need at least 1001 results to analyse error rates.")
        return

    chip_id = kwargs.get('chip_id', 'unknown')

    n_bits = results[0].data.size
    nominal_data = results[0].data
    results = results[1:1+1000]

    hds = np.empty(len(results))  # Hamming distances
    avg_hd = 0.0
    min_hd = np.inf
    max_hd = -np.inf
    for i in range(len(results)):
        # Calculate hamming distance between
        # each vector and the nominal vector
        hds[i] = hamming_distance(nominal_data, results[i].data)
        min_hd = min(min_hd, hds[i])
        max_hd = max(max_hd, hds[i])
        avg_hd += hds[i]

    # Get average hamming distance
    avg_hd /= len(results)

    # Divide by amount of bits to get BER percentage
    ber = avg_hd / n_bits
    error_rates = np.round(hds / n_bits, 4)
    print(f'Chip {chip_id}:')
    print(f'\tAverage BER (bit error rate): {ber:.2%}')
    print(f'\tMinimum BER (bit error rate): {min_hd / n_bits:.2%}')
    print(f'\tMaximum BER (bit error rate): {max_hd / n_bits:.2%}')

    unique, counts = np.unique(error_rates, return_counts=True)
    plt.plot(unique, counts, marker='.', label=chip_id)
    plt.title('Frequency of bit error rates')
    # Label the axes
    plt.xlabel('Error rate (%)')
    plt.ylabel('Count')
    plt.legend()
    # Add percentage sign to x axis
    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))


def autocorrelation(results: list[Result], **kwargs):
    """
    Given a list of results, calculate the autocorrelation.

    The calculation is done with the scipy.signal.correlate method.

    results -- list of results to analyse the autocorrelation.
               Only the first one will be used.
    """
    chip_id = kwargs.get('chip_id', 'unknown')
    data = results[0].data.astype(np.int8)
    data[data == 0] = -1  # Replace 0s with -1s
    data = data.astype(np.float64)
    autocorr = correlate(data, data, mode='full', method='fft')
    autocorr /= np.max(autocorr)

    time_axis = np.arange(-len(data)+1, len(data))
    plt.scatter(time_axis, autocorr, marker='.')
    plt.title(f'Autocorrelation for {chip_id}')

    plt.tight_layout()
    plt.show()

    return None


def fractional_hamming_weight(results: list[Result], **kwargs):
    """
    Given a list of results, calculate the fractional hamming weight.

    The calculation is done as follows: the fractional hamming weight is
    calculated for all the (first 1000) results as the average bit value,
    and then the average between all of them is calculated.
    """
    if len(results) < 1001:
        logging.error("Need at least 1001 results to analyse "
                      "fractional hamming weight.")
        return

    results = results[:1000]
    chip_id = kwargs.get('chip_id', 'unknown')

    avg_fhw = 0.0
    for result in results:
        avg_fhw += np.mean(result.data)
    avg_fhw /= len(results)

    print(f'Fractional hamming weight for chip {chip_id}: {avg_fhw:.6f}')

    # Parameters for the binomial distribution
    n = 1000
    p = 0.5

    chip_index = kwargs.get('chip_index')

    if chip_index == 0:  # Only plot the binomial distribution once
        x_values = np.arange(0, n)

        # Calculate the binomial PMF
        binomial_pmf = binom.pmf(x_values, n, p)

        # Plot the normalized binomial distribution
        plt.plot(x_values / n, binomial_pmf, label='Binomial Distribution')

    plt.title('Mean of start-up values of all SRAM cells')
    plt.xlabel('Mean')
    plt.ylabel('Probability')
    plt.xticks(np.arange(0.45, 0.6, 0.01))
    plt.xlim(0.45, 0.55)

    # Plot the average fractional hamming weight
    plt.scatter([avg_fhw], [binom.pmf(int(avg_fhw*1000), n, p)],
                label=chip_id)

    plt.legend()
