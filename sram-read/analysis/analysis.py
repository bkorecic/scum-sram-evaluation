import numpy as np
import logging
import itertools
import matplotlib.pyplot as plt
import pathlib
from scipy.signal import correlate
from scipy.stats import binom
from utils import Result, hamming_distance, numpy_data_dir
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
    logging.info(f'Chip {chip_id}:')
    logging.info(f'\tAverage BER (bit error rate): {ber:.2%}')
    logging.info(f'\tMinimum BER (bit error rate): {min_hd / n_bits:.2%}')
    logging.info(f'\tMaximum BER (bit error rate): {max_hd / n_bits:.2%}')

    unique, counts = np.unique(error_rates, return_counts=True)
    plt.plot(unique, counts, label=chip_id)
    # plt.title('Frequency of bit error rates')
    # Label the axes
    plt.xlabel('BER (%)')
    plt.ylabel('Frequency')
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
    n = len(data)
    data[data == 0] = -1  # Replace 0s with -1s
    data = data.astype(np.float64)
    autocorr = correlate(data, data, mode='full', method='fft')
    autocorr /= n

    # Spotting the max correlation value at lag = -512 and lag = 512
    print(f'Chip {chip_id}:')
    print(f'\tCorrelation at lag -511 : {autocorr[450046]}')
    print(f'\tCorrelation at lag -512 : {autocorr[450047]}')
    print(f'\tCorrelation at lag -513 : {autocorr[450048]}')
    print(f'\tCorrelation at lag 512 : {autocorr[451071]}')

    time_axis = np.arange(-len(data)+1, len(data))
    plt.scatter(time_axis, autocorr, marker='.')
    plt.title(f'Autocorrelation for {chip_id}')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.tight_layout()
    plt.show()

    # Derive the highest three autocorrelation values and their corresponding lag values
    # takes too much time
    # top3_ind = np.argpartition(autocorr, -3)[-3:]
    # print('test')
    # print(f'test: {top3_ind}')
    # top3_val = autocorr[top3_ind]
    # top3_lag = time_axis[top3_ind]
    # print(f'Maximum autocorrelation values: {top3_val}:')
    # print(f'Their indices: {top3_lag}')

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
    n_bits = results[0].data.size

    avg_fhw = 0.0
    for result in results:
        avg_fhw += np.mean(result.data)
    avg_fhw /= len(results)

    logging.info(f'Fractional hamming weight for chip {chip_id}: {avg_fhw:.6f}')

    # Parameters for the binomial distribution
    n = n_bits
    p = 0.5

    chip_index = kwargs.get('chip_index')

    if chip_index == 0:  # Only plot the binomial distribution once
        x_values = np.arange(0, n)

        # Calculate the binomial PMF
        binomial_pmf = binom.pmf(x_values, n, p)

        # Plot the normalized binomial distribution
        plt.plot(x_values / n, binomial_pmf, label='Binomial Distribution')

    # plt.title('Mean of Start-up Values of All SRAM Cells')
    plt.xlabel('Mean')
    plt.ylabel('Probability of Occurence')
    plt.xticks(np.arange(0.45, 0.6, 0.01))
    plt.xlim(0.48, 0.52)

    # Plot the average fractional hamming weight
    plt.scatter([avg_fhw], [binom.pmf(int(avg_fhw*n), n, p)],
                label=chip_id)

    plt.legend()


def calculate_frequencies(results: list[Result], **kwargs):
    """
    Calculate the frequency of 1 in each bit position for all the given chips
    and store it in numpy files.
    """
    chip_id = kwargs.get('chip_id', 'unknown')
    results = results[:1000]  # Use first 1000 results
    n_bits = results[0].data.size
    bit_freq_1 = np.zeros(n_bits)  # Frequency of 1 in each bit position
    for result in results:
        for i in range(n_bits):
            bit_freq_1[i] += result.data[i]
    filename = f'bit_freq_1_{chip_id}.npy'
    np.save(pathlib.Path(numpy_data_dir / filename), bit_freq_1)
    logging.info(f'Saved bit frequencies for chip {chip_id} to {filename}')


def stability(results: list[Result], **kwargs):
    if len(results) < 1000:
        logging.error("Need at least 1000 results to analyse stability.")
        return
    results = results[:1000]
    n_bits = results[0].data.size

    chip_id = kwargs.get('chip_id', 'unknown')
    # Load bit_freq_1 from numpy file
    try:
        bit_freq_1 = np.load(
            pathlib.Path(numpy_data_dir / f'bit_freq_1_{chip_id}.npy'))
    except OSError as e:
        logging.error(f'Could not load bit_freq_1_{chip_id}.npy: {e}')
        return

    stable_bits = 0
    stability = np.empty(n_bits, dtype=np.float64)
    for i in range(n_bits):
        freq_1 = bit_freq_1[i]
        freq_0 = len(results) - freq_1
        if min(freq_1, freq_0) == 0:
            stable_bits += 1
        stability[i] = freq_1 / len(results)

    logging.info(f'Stable bits for chip {chip_id}: {stable_bits}/{n_bits}')

    plot_height = 50
    plot_width = 100
    bits_to_plot = plot_height * plot_width

    heat_matrix = np.reshape(stability[:bits_to_plot],
                             (plot_height, plot_width))

    plt.imshow(heat_matrix, cmap='RdBu',
               interpolation='nearest',
               vmin=0, vmax=0.5)

    # Disable x and y axis ticks
    plt.xticks([])
    plt.yticks([])

    # Colorbar
    plt.colorbar(location='bottom', pad=0.05)


def inter_chip_hamming_distance(all_results):
    avg = 0.0
    min_fhd = np.inf
    max_fhd = -np.inf
    for results_1, results_2 in itertools.combinations(all_results.values(),
                                                       r=2):
        d1 = results_1[0].data
        d2 = results_2[0].data
        n_bits = d1.size
        # Calculate fractional hamming distance
        fhd = hamming_distance(d1, d2) / n_bits
        min_fhd = min(min_fhd, fhd)
        max_fhd = max(max_fhd, fhd)
        avg += fhd
    n = len(all_results)
    avg /= n * (n - 1) // 2
    logging.info('Inter-chip fractional hamming distance:')
    logging.info(f'\tAverage: {avg:.6f}')
    logging.info(f'\tMinimum: {min_fhd:.6f}')
    logging.info(f'\tMaximum: {max_fhd:.6f}')
