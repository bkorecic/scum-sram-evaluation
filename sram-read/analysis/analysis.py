import numpy as np
import logging
import itertools
import matplotlib.pyplot as plt
import pathlib
import constants
from scipy.signal import correlate
from scipy.stats import binom
from utils import ResultList, hamming_distance, numpy_data_dir
from matplotlib.ticker import PercentFormatter
from abc import ABC, abstractmethod


class Analysis(ABC):
    """
    Abstract base class for all types of analysis.
    """
    name = None

    def __init__(self, all_results: list[ResultList]):
        self.all_results = all_results

    @abstractmethod
    def run(self):
        pass


class IntraChipAnalysis(Analysis):
    @staticmethod
    @abstractmethod
    def process(results: ResultList, chip_id: str):
        pass

    def run(self):
        """
        An intra-chip analysis is run for each chip.
        """
        for results in self.all_results:
            self.process(results)


class InterChipAnalysis(Analysis):
    @staticmethod
    @abstractmethod
    def process(self, all_results: list[ResultList]):
        pass

    def run(self):
        """
        An inter-chip analysis is run once for all chips.
        """
        self.process(self.all_results)


class BitErrorRate(InterChipAnalysis):
    """
    Calculate metrics related to the bit error rate.

    The calculation is done as follows: the first result is chosen
    as the nominal data, then the hamming distance is calculated
    between all the other results and the nominal data. The output
    is the average, divided by the amount of bits.

    The bit error rate of all the chips is plotted in a single graph.
    """
    name = "Bit Error Rate (data+plot)"

    @staticmethod
    def process(all_results: list[ResultList]):
        for results in all_results:
            n_bits = results[0].data.size
            nominal_data = results[0].data
            results.pop(0)

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
            logging.info(f'Chip {results.chip_id}:')
            logging.info(f'\tAverage BER (bit error rate): {ber:.5%}')
            logging.info(
                f'\tMinimum BER (bit error rate): {min_hd / n_bits:.5%}')
            logging.info(
                f'\tMaximum BER (bit error rate): {max_hd / n_bits:.5%}')

            unique, counts = np.unique(error_rates, return_counts=True)
            plt.plot(unique, counts, label=results.chip_id)
            # plt.title('Frequency of bit error rates')
            # Label the axes
            plt.xlabel('BER (%)')
            plt.ylabel('Frequency')
            plt.legend()
        # Add percentage sign to x axis
        plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1.0,
                                                             decimals=2))
        plt.show()


class Autocorrelation(IntraChipAnalysis):
    """
    Calculate the autocorrelation.

    The calculation is done with the scipy.signal.correlate method.

    Only the first result in "results" will be used.
    """
    name = "Autocorrelation (data+plot)"

    @staticmethod
    def process(results: ResultList):
        data = results[0].data.astype(np.int8)
        n = len(data)
        data[data == 0] = -1  # Replace 0s with -1s
        data = data.astype(np.float64)
        autocorr = correlate(data, data, mode='full', method='fft')
        autocorr /= n

        sorted_autocorr = np.sort(np.abs(autocorr))
        logging.info(f'Autocorrelation for {results.chip_id}:')
        logging.info(
            f'\tSmallest autocorrelation value: {sorted_autocorr[0]:.5e}')
        logging.info(
            f'\tLargest autocorrelation value: {sorted_autocorr[-2]:.5f}')
        logging.info(f'\tAutocorrelation average: {np.mean(autocorr):.5e}')

        time_axis = np.arange(-len(data)+1, len(data))
        plt.scatter(time_axis, autocorr, marker='.')
        plt.title(f'Autocorrelation for {results.chip_id}')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.tight_layout()
        plt.show()


class FractionalHammingWeight(InterChipAnalysis):
    """
    Calculate the fractional hamming weight for each chip and plot
    their positions on a normalized binomial distribution.

    The calculation is done as follows: the fractional hamming weight is
    calculated for all the results as the average bit value,
    and then the average between all of them is calculated.
    """
    name = "Fractional Hamming Weight (data+plot)"

    @staticmethod
    def process(all_results: list[ResultList]):
        n_bits = all_results[0][0].data.size

        # Parameters for the binomial distribution
        n = n_bits
        p = 0.5
        x_values = np.arange(0, n)

        # Calculate the binomial PMF
        binomial_pmf = binom.pmf(x_values, n, p)

        # Plot the normalized binomial distribution
        plt.plot(x_values / n, binomial_pmf, label='Binomial Distribution')

        for results in all_results:
            # Calculate min/max/avg fractional hamming weight for each chip
            min_fhw = np.inf
            max_fhw = -np.inf
            avg_fhw = 0.0
            for result in results:
                val = np.mean(result.data)
                min_fhw = min(min_fhw, val)
                max_fhw = max(max_fhw, val)
                avg_fhw += val
            avg_fhw /= len(results)

            logging.info(
                f'Chip {results.chip_id}:')
            logging.info(f'\tMinimum FHW: {min_fhw:.5f}')
            logging.info(f'\tMaximum FHW: {max_fhw:.5f}')
            logging.info(f'\tAverage FHW: {avg_fhw:.5f}')

            # plt.title('Mean of Start-up Values of All SRAM Cells')
            plt.xlabel('Mean')
            plt.ylabel('Probability of Occurence')
            plt.xticks(np.arange(0.45, 0.6, 0.01))
            plt.xlim(0.48, 0.52)

            # Plot the average fractional hamming weight
            plt.scatter([avg_fhw], [binom.pmf(int(avg_fhw*n), n, p)],
                        label=results.chip_id)
            plt.legend()
        plt.show()


class Frequencies(IntraChipAnalysis):
    """
    Calculate the frequency of 1 in each bit position for the given chip
    and store it a numpy file for later use.
    """
    name = "Calculate Frequencies (files)"

    @staticmethod
    def process(results: ResultList):
        n_bits = results[0].data.size
        bit_freq_1 = np.zeros(n_bits)  # Frequency of 1 in each bit position
        for result in results:
            for i in range(n_bits):
                bit_freq_1[i] += result.data[i]
        filename = f'bit_freq_1_{results.chip_id}.npy'
        np.save(pathlib.Path(numpy_data_dir / filename), bit_freq_1)
        logging.info(
            f'Saved bit frequencies for chip {results.chip_id} to {filename}')


class Stability(IntraChipAnalysis):
    """
    Calculate the stability of each bit position for the given chip.

    Plot the stability of the first 50*100 bits in a heatmap.
    """
    name = "Cell stability (data+plot), requires frequencies"

    @staticmethod
    def process(results: ResultList):
        n_bits = results[0].data.size

        # Load bit_freq_1 from numpy file
        try:
            bit_freq_1 = np.load(
                pathlib.Path(
                    numpy_data_dir / f'bit_freq_1_{results.chip_id}.npy'))
        except OSError as e:
            logging.error(
                f'Could not load bit_freq_1_{results.chip_id}.npy: {e}')
            return

        stable_bits = 0
        stability = np.empty(n_bits, dtype=np.float64)
        for i in range(n_bits):
            freq_1 = bit_freq_1[i]
            freq_0 = len(results) - freq_1
            if min(freq_1, freq_0) == 0:
                stable_bits += 1
            stability[i] = freq_1 / len(results)

        logging.info(
            f'Stable bits for chip {results.chip_id}: '
            f'{stable_bits/n_bits:.5f}')

        plot_height = 50
        plot_width = 100
        bits_to_plot = plot_height * plot_width

        heat_matrix = np.reshape(stability[:bits_to_plot],
                                 (plot_height, plot_width))

        plt.imshow(heat_matrix, cmap='RdBu',
                   interpolation='nearest',
                   vmin=0, vmax=1.0)

        # Disable x and y axis ticks
        plt.xticks([])
        plt.yticks([])

        # Colorbar
        colorbar = plt.colorbar(location='bottom', pad=0.05)
        colorbar.set_ticks([0, 0.5, 1.0])
        plt.show()


class InterChipHammingDistance(InterChipAnalysis):
    """
    Calculate the inter-chip (fractional) hamming distance average between all
    pairs of chips and plot the results in a heatmap.
    """
    name = "Inter-chip hamming distance (data+plot)"

    @staticmethod
    def process(all_results: list[ResultList]):
        avg = 0.0
        min_fhd = np.inf
        max_fhd = -np.inf
        chip_ids = [results.chip_id for results in all_results]
        chip_ids.sort()
        # Dict to map chip_id to index
        chip_index = {chip_id: i for i, chip_id in enumerate(chip_ids)}
        hd_matrix = np.zeros((len(all_results), len(all_results)))
        for res1, res2 in itertools.combinations(all_results, r=2):
            d1 = res1[0].data
            d2 = res2[0].data
            n_bits = d1.size
            # Calculate fractional hamming distance
            fhd = hamming_distance(d1, d2) / n_bits
            min_fhd = min(min_fhd, fhd)
            max_fhd = max(max_fhd, fhd)
            avg += fhd
            hd_matrix[chip_index[res1.chip_id]][chip_index[res2.chip_id]] = fhd
            hd_matrix[chip_index[res2.chip_id]][chip_index[res1.chip_id]] = fhd
        n = len(all_results)
        avg /= n * (n - 1) // 2
        logging.info('Inter-chip fractional hamming distance:')
        logging.info(f'\tAverage: {avg:.5f}')
        logging.info(f'\tMinimum: {min_fhd:.5f}')
        logging.info(f'\tMaximum: {max_fhd:.5f}')

        fig, ax = plt.subplots()
        plt.imshow(hd_matrix, cmap='viridis', vmin=0, vmax=1.0)
        ax.set_xticks(np.arange(len(chip_ids)), labels=chip_ids)
        ax.set_yticks(np.arange(len(chip_ids)), labels=chip_ids)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Add text annotations
        for i in range(len(chip_ids)):
            for j in range(len(chip_ids)):
                ax.text(j, i, f'{hd_matrix[i, j]:.3f}',
                        ha="center", va="center", color="w")

        plt.title('Inter-chip fractional hamming distance')
        plt.colorbar()
        plt.show()


class InterChipMinEntropy(InterChipAnalysis):
    """
    Calculate the minimum entropy for all chips.
    """
    name = "Inter-Chip min. entropy (data)"

    @staticmethod
    def process(all_results: list[ResultList]):
        n_bits = all_results[0][0].data.size
        n_chips = len(all_results)

        # Frequency of occurence of 1 at different positions
        freq_1 = np.zeros(n_bits)
        for results in all_results:
            freq_1 = np.add(freq_1, results[0].data)
        prob_1 = freq_1 / n_chips

        avg = 0.0
        for bit_prob_1 in prob_1:
            avg += -np.log2(max(bit_prob_1, 1-bit_prob_1))
        avg /= n_bits
        logging.info(f'Inter-chip min. entropy: {avg:.5f}')


class IntraChipMinEntropy(IntraChipAnalysis):
    """
    Calculate the minimum entropy for a chip.
    """
    name = "Intra-Chip min. entropy (data), requires frequencies"

    @staticmethod
    def process(results: ResultList):
        chip_id = results.chip_id
        # Load bit_freq_1 from numpy file
        try:
            bit_freq_1 = np.load(
                pathlib.Path(numpy_data_dir / f'bit_freq_1_{chip_id}.npy'))
        except OSError as e:
            logging.error(f'Calculation of frequencies '
                          f'required for chip {chip_id}')
            logging.error(e)
            return

        n_bits = bit_freq_1.size
        avg_ent = 0.0
        min_ent = np.inf
        max_ent = -np.inf
        for i in range(n_bits):
            p = bit_freq_1[i] / constants.READINGS_TO_ANALYZE
            val = -np.log2(max(p, 1-p))
            min_ent = min(min_ent, val)
            max_ent = max(max_ent, val)
            avg_ent += val
        avg_ent /= n_bits

        logging.info(f'Intra-chip min. entropy for chip {chip_id}:')
        logging.info(f'\tMinimum: {min_ent:.5f}')
        logging.info(f'\tMaximum: {max_ent:.5f}')
        logging.info(f'\tAverage: {avg_ent:.5f}')
