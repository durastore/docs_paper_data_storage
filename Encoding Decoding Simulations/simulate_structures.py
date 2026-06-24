from collections import defaultdict
import configargparse
import gmpy2
import huffman
from itertools import combinations
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats
import string
import zlib
from decode_structures import fuse_structures

NUMERALS=string.digits + string.ascii_letters + string.punctuation
rng = np.random.default_rng()
logger = logging.getLogger(__name__)


def load_freqs(freqfile):
    # Letter frequency from https://norvig.com/mayzner.html
    freq_df = pd.read_csv(freqfile, sep=' ', index_col=0)
    freq_series = freq_df['PERCENT']/100
    # Normalize to 1
    freq_series = freq_series / (freq_series.sum())
    freq_series.index = freq_series.index.str.lower()
    freq_dict = freq_series.to_dict()

    freqs = list(freq_dict.items()) # HuffmanCode requires (symbol, freq) pairs.
    return freqs


def write_counts(counts_dict, outfile):
    with open(outfile, 'w') as handle:
        handle.write('BC code,Frequency\n')
        for structure, count in sorted(counts_dict.items(), key=lambda x: x[1], reverse=True):
            handle.write(f'{structure},{count}\n')


def plot_distributions(mean, sigma, outfile, max_mult=5):
    fig, ax = plt.subplots()
    max_count = mean * max_mult + 2 * np.sqrt(max_mult) * sigma
    x = np.arange(max_count)
    for i in range(6):
        m = i*mean
        s = np.sqrt(i) * sigma
        if i == 0:
            s = 500
        dist = scipy.stats.norm(loc=m, scale=s)
        res = dist.pdf(x)
        ax.plot(x, res, label=i)

    ax.set_xlabel('Structure count')
    ax.set_ylabel('likelihood')
    ax.legend()
    ax.set_ylim(0, 0.0002)
    fig.tight_layout()
    plt.savefig(outfile)


def generate_derivative_recursive(current_structure, current_mult, remainder, error_rates, cutoff=1e-15):
    # Recursive function that calculates the overall rate of errors for all derivatives formed from one structure
    if remainder == '':
        return {current_structure: current_mult}
    next_pos = remainder[0]
    error_rates_pos = error_rates[next_pos]
    derivative_rates = error_rates_pos * current_mult
    new_structures = {}
    for key, rate in derivative_rates.items():
        mult = current_mult * rate
        if mult < cutoff:
            continue
        structure = current_structure + key
        new_structures.update(generate_derivative_recursive(structure, mult, remainder[1:], error_rates, cutoff=cutoff))
    return new_structures


def add_errors_recursive(counts_dict, error_grid):
    # Slower implementation, but can handle any combination of error
    # rates between different colors
    error_fractions = defaultdict(float)
    for structure, count in counts_dict.items():
        derivatives = generate_derivative_recursive('', 1, structure, error_grid)
        # Normalize rates 
        # Not all derivatives are made, so the rates do not add up to one
        # --> divide the count we had before over the ones that are made 
        sum_rates = sum(derivatives.values())
        for derivative, rate in derivatives.items():
            derivative_fraction = rate / sum_rates * count
            error_fractions[derivative] += derivative_fraction
    return error_fractions


def add_errors_constant_error(counts_dict, error_rates, subsets,
                              n_color_combinations, norm_factor, 
                              sampling_grids):
    # Faster implementation, but only works for constant error rates
    final_structures = defaultdict(int)
    for structure, count in counts_dict.items():
        expected_count_no_errors = count * norm_factor
        # Now assign the errors
        # For each amount of errors, get derivatives to assign them to
        for n_errors, fraction_errors in enumerate(error_rates):
            sampled_n_structures = int(scipy.stats.poisson.rvs(expected_count_no_errors * fraction_errors))
            if n_errors == 0:
                final_structures[structure] += sampled_n_structures
                continue
            # Distribute the errors over all possible combinations of positions by drawing from a multinomial distribution
            pvals, structure_indices, gridsize = sampling_grids[n_errors]
            error_positions = rng.choice(gridsize, p=pvals, size=sampled_n_structures)
            # Also draw the colors they are changing into 
            color_indices = rng.integers(0, n_color_combinations-1, size=(sampled_n_structures, n_errors))
            # Now actually generate the structures
            for sampled_index, color_indices_one_derivative in zip(error_positions, color_indices):
                derivative = create_derivative(structure, structure_indices, sampled_index,
                                               subsets, color_indices_one_derivative)
                final_structures[derivative] += 1
    return final_structures


def create_derivative(structure, structure_indices, sampled_index,
                      subsets, color_indices_one_derivative):
    changed_structure_positions = structure_indices[sampled_index] # Set of positions
    derivative = ''
    color_index_pointer = 0
    for i, current_char in enumerate(structure):
        if i in changed_structure_positions:
            derivative += subsets[current_char][color_indices_one_derivative[color_index_pointer]]
            color_index_pointer += 1
        else:
            derivative += current_char
    return derivative


def add_errors_constant_error_old(counts_dict, error_rates, subsets,
                              kmer_length, norm_factor, cutoff=1e-15):
    # Faster implementation, but only works for constant error rates
    final_structures = defaultdict(int)
    for structure, count in counts_dict.items():
        expected_count_no_errors = count * norm_factor
        # Now assign the errors
        # For each amount of errors, get derivatives to assign them to
        for n_errors, (fraction_errors, total_n_of_derivatives) in enumerate(error_rates):
            if n_errors == 0:
                final_structures[structure] += int(scipy.stats.poisson.rvs(expected_count_no_errors * fraction_errors))
                continue
            # Instead of iterating over all derivatives and rolling Poisson for each,
            # instead estimate how many successes we get from it, then generate that many derivatives
            dist = scipy.stats.poisson(fraction_errors * expected_count_no_errors)
            n_observations = 1
            while True:
                p = dist.pmf(n_observations)
                n_derived_structures = scipy.stats.binom(total_n_of_derivatives, p).rvs()
                # Now get a derivative for each hit
                derivatives = get_derived_structures(structure, subsets, n_errors, 
                                                     n_derived_structures, kmer_length)
                for derivative in derivatives:
                    final_structures[derivative] += n_observations
                if p < cutoff and n_derived_structures == 0:
                    break
                n_observations += 1
    return final_structures


def get_derived_structures(structure, subsets, n_errors,
                           n_derivatives, kmer_length):
    # Can make this more efficient by generating all derivatives simultaneously
    # This might get slow if you need to generate all derivatives 
    derivatives = set()
    for _ in range(n_derivatives):
        while True:
            derivative = generate_one_derivative(structure, subsets, 
                                                 n_errors, kmer_length)
            if derivative not in derivatives:
                derivatives.add(derivative)
                break
    return derivatives


def generate_one_derivative(structure, subsets, n_errors,
                           kmer_length):
    altered_positions = rng.choice(kmer_length, n_errors, replace=False)
    derivative = ''
    prev_pos = -1
    for position in altered_positions:
        new_char = rng.choice(subsets[structure[position]])
        derivative += structure[prev_pos+1:position] + new_char
        prev_pos = position
    derivative += structure[prev_pos+1:]
    return derivative


def get_kmers(encoded_text, kmer_length, shift, fill_character='0'):
    # Split into structures
    structures = []
    for i in range(0, len(encoded_text) - kmer_length + shift, shift):
        structure = encoded_text[i:i+kmer_length]
        structures.append(structure)
    # for shifts > 1, we might need to fill up the last structure
    last_structure = structures[-1]
    missing = kmer_length - len(last_structure)
    if missing > 0:
        last_structure += fill_character * missing
        structures[-1] = last_structure
    return structures


def generate_base_structures(structures, powerlaw):
    # Assign read counts
    counts = powerlaw.rvs(size=len(structures))
    # Merge read counts
    counts_dict = defaultdict(int)
    for structure, count in zip(structures, counts):
        counts_dict[structure] += count
    return counts_dict


def count_clashes(structures, shift):
    left_flanks = set()
    right_flanks = set()
    for structure in structures:
        left_flanks.add(structure[shift:])
        right_flanks.add(structure[:-shift])
    # Total amount of left flanks and right flanks should each be equal to the amount of structures
    n_structures = len(structures)
    structure_clashes = n_structures - len(set(structures))
    left_flank_clashes = n_structures - len(left_flanks)
    right_flank_clashes = n_structures - len(right_flanks)
    return structure_clashes, left_flank_clashes, right_flank_clashes


def convert_fractions_to_counts(structure_fractions, reference_structures=None, min_structure_count=100):
    # Upscale the fractions so that the lowest reference structure is seen at least 
    # the required amount of times
    # Then run Poisson over it to get a randomized count
    if reference_structures is None:
        reference_structures = structure_fractions
    min_reference_count = min(structure_fractions[structure] for structure in reference_structures)
    multiplication_factor = min_structure_count / min_reference_count
    expected_counts = np.fromiter((fraction*multiplication_factor for fraction in structure_fractions.values()), dtype=float)
    sampled_counts = scipy.stats.poisson.rvs(expected_counts)
    counts_dict = {}
    for structure, sampled_count in zip(structure_fractions, sampled_counts):
        if sampled_count > 1:
            counts_dict[structure] = sampled_count
    return counts_dict

def clean_text(text):
    cleaned_text = ''
    for char in text.lower():
        if char in string.ascii_lowercase:
            cleaned_text += char
    return cleaned_text


def encode_strings_zlib(text, n_color_combinations=3, encoding_scheme='utf-8', 
                        compression_level=9):
    # Compress to remove the "English" structure (k-mer spikes)
    input_bytes = text.encode(encoding_scheme)
    compressed_bytes = zlib.compress(input_bytes, level=compression_level)

    # Treat the bytes as one giant integer
    big_integer = gmpy2.mpz.from_bytes(compressed_bytes, byteorder='big')

    # Convert that integer to Base-n
    if big_integer == 0:
        return "0"

    digits = []
    while big_integer:
        big_integer, rem = gmpy2.t_divmod(big_integer, n_color_combinations)
        digits.append(NUMERALS[rem])

    # The digits come out backwards, so reverse them
    return ''.join(digits[::-1])


def encode_strings_huffman(text, n_color_combinations, freq_file):
    freqs = load_freqs(freq_file)
    n_ary_huffman = huffman.HuffmanCode(freqs, n_color_combinations)
    return n_ary_huffman.encode(text)


def main_simulation(encoded_text, base_folder, n_replicates, powerlaw,
                    kmer_length, n_color_combinations, shift,
                    error_grid_uncorrected, error_rates_corrected,
                    sampling_grids, disable_uncorrected_errors, 
                    folder_names = ['no_errors', 'corrected_errors', 
                                    'uncorrected_errors']):
    available_characters = list(NUMERALS[:n_color_combinations])
    character_subsets = prework_subset(available_characters)

    logger.info('Splitting text into kmers')
    structures = get_kmers(encoded_text, kmer_length, shift)
    structure_clashes, left_clashes, right_clashes = count_clashes(structures, shift)
    os.makedirs(base_folder, exist_ok=True)
    logfile = os.path.join(base_folder, 'clashes.txt')
    with open(logfile, 'w') as handle:
        handle.write(f'structure clashes\t{structure_clashes}\n')
        handle.write(f'duplicated left flanks\t{left_clashes}\n')
        handle.write(f'duplicated right flanks\t{right_clashes}\n')

    for n in range(1, n_replicates+1):
        logger.info(f'{n=}')
        outfolder = os.path.join(base_folder, f'replicate_{n}')
        os.makedirs(outfolder, exist_ok=True)

        base_structure_fractions = generate_base_structures(structures, powerlaw)
        # convert fractions to counts
        logger.info('Generating base counts')
        base_structure_counts = convert_fractions_to_counts(base_structure_fractions)
        # Check if applying error correction on base counts removes any (this depends on structure counts)
        logger.info('Fusing structures')
        n_error_corr_clashes = len(base_structure_counts) - len(fuse_structures(base_structure_counts, character_subsets))
        logfile = os.path.join(outfolder, 'clashes.txt')
        with open(logfile, 'w') as handle:
            handle.write(f'error_corr_clashes\t{n_error_corr_clashes}\n')

        # add errors - first with constant error rate (using the error correction in DNA paint)
        logger.info('Adding corrected errors')
        norm_factor = precalc_norm_factor(base_structure_fractions, error_rates_corrected)

        structure_counts_low_errors = add_errors_constant_error(base_structure_fractions, error_rates_corrected,
                                                                character_subsets, n_color_combinations,
                                                                 norm_factor, sampling_grids)
        datasets = [base_structure_counts, structure_counts_low_errors]
        if disable_uncorrected_errors:
            folder_names = folder_names[:2]
        else:
            logger.info('Adding uncorrected errors')
            error_fractions = add_errors_recursive(base_structure_fractions, error_grid_uncorrected)
            datasets.append(convert_fractions_to_counts(error_fractions, reference_structures=base_structure_counts))

        for dataset, folder_name in zip(datasets, folder_names):
            subfolder = os.path.join(outfolder, folder_name)
            os.makedirs(subfolder, exist_ok=True)
            outfile = os.path.join(subfolder, 'counts.txt')
            write_counts(dataset, outfile)


def create_error_rate_grids(category_errors, n_mixed_colors, n_unmixed_colors):
    n_color_combinations = n_mixed_colors + n_unmixed_colors
    index = list(NUMERALS[:n_color_combinations])

    # First error grid - all four types
    # Header is start digit, Index is target digit
    # Here we set the mixed colors as the first ones, the unmixed as last.
    # This is because the errors are sorted by probabilities, meaning
    # the highest occurence are at the final digits
    # Since unmixed color combinations have lower error rates, they are assigned to the
    # more frequently occuring ones
    error_grid_1 = np.zeros(shape=(n_color_combinations, n_color_combinations))
    error_grid_1[:n_mixed_colors, :n_mixed_colors] = category_errors['mix2mix'] / (n_mixed_colors-1)
    error_grid_1[:n_mixed_colors, n_mixed_colors:] = category_errors['unmix2mix'] / n_unmixed_colors
    error_grid_1[n_mixed_colors:, :n_mixed_colors] = category_errors['mix2unmix'] / n_mixed_colors
    error_grid_1[n_mixed_colors:, n_mixed_colors:] = category_errors['unmix2unmix'] / (n_unmixed_colors-1)
    error_grid_1 = finish_error_grid(error_grid_1, index)

    # Second error grid - only one type
    error_grid_2 = np.ones(shape=(n_color_combinations, n_color_combinations)) * category_errors.loc['mix2mix'] / (n_color_combinations - 1)
    error_grid_2 = finish_error_grid(error_grid_2, index)

    return error_grid_1, error_grid_2


def precalculate_error_info(category_errors, kmer_length):
    # Only implemented for constant error rates - calculate the chance for 0 to k errors
    base_error_rate = category_errors.loc['mix2mix']
    correct_rate = 1 - base_error_rate
    error_info = []
    for n_errors in range(0, kmer_length+1):
        # Total fraction of structures with this amount of errors first
        error_rate_all_derivatives = (base_error_rate ** n_errors) * (correct_rate ** (kmer_length - n_errors)) * math.comb(kmer_length, n_errors)
        # Then per derivative (spread equally)
        # total_n_of_derivatives = (n_color_combinations - 1) ** n_errors * math.comb(kmer_length, n_errors)
        # error_fraction_per_derivative = error_rate_all_derivatives / total_n_of_derivatives
        error_info.append(error_rate_all_derivatives)
    return error_info


def prework_error_sampling_grids(kmer_length):
    sampling_grids = {}
    for n_of_errors in range(1, kmer_length+1):
        # Store pvals and position conversion to match sampled indices from the pval to an actual combination of indices in the structure
        comb_list = [set(i) for i in (combinations(range(kmer_length), n_of_errors))]
        gridsize = len(comb_list)
        pvals = np.ones(gridsize) / gridsize
        sampling_grids[n_of_errors] = pvals, comb_list, gridsize
    return sampling_grids


def prework_subset(available_characters):
    subset = {}
    s = set(available_characters)
    for char in s:
        remaining = s - {char, }
        subset[char] = np.fromiter(remaining, dtype=object)
    return subset


def precalc_norm_factor(counts_dict, error_rates, reference_structures=None, min_structure_count=100):
    # After adding errors, the fraction of structures will be slightly decreased first 
    # (as only ~98% of structures will be without errors)
    # this function will calculate what the normalization is after that correction
    if reference_structures is None:
        reference_structures = counts_dict
    min_reference_count = min(counts_dict[structure] for structure in reference_structures)
    return min_structure_count / (min_reference_count * error_rates[0])


def precalculate_error_rates_mixed(category_errors, kmer_length):
    # Basically, each structure has a combination of mixed and unmixed colors
    # Each resulting error structure also will have a combination of mixed and unmixed colors
    # Up to k errors are present, but these can come from any combination of mixed and unmixed as well
    # and result in any combination of mixed and unmixed errors

    raise NotImplementedError()
    # For each combination, precalculate the error rates
    for n_mixed in range(kmer_length + 1):
        for n_errors in range(kmer_length + 1):
            for n_mixed_errors in range(n_errors + 1):
                n_unmixed_errors = n_errors - n_mixed_errors
    pass


def finish_error_grid(grid, index):
    np.fill_diagonal(grid, 0)
    totals = grid.sum(axis=0)
    rem = 1 - totals
    np.fill_diagonal(grid, rem)
    return pd.DataFrame(data=grid, index=index, columns=index)


def plot_counts(normalized_counts, pl, outfile):
    fig, ax = plt.subplots()
    binned_counts, bins, _ = ax.hist(normalized_counts, bins=20, label='Binned counts')
    xx = np.linspace(pl.ppf(0.01),
                     pl.ppf(0.99), 101)
    yy = pl.pdf(xx)
    ax.plot(xx, yy, label='Powerlaw fit')
    ax.set_xlabel('Normalized counts')
    ax.set_ylabel('Frequency of counts')
    ax.legend()
    plt.savefig(outfile)
    plt.close()


def write_log(logfile, text, huffman_encoded_text, zlib_encoded_text):
    with open(logfile, 'w') as handle:
        handle.write(f'Orginal text: {len(text)} chars\n')
        handle.write(f'Huffman encoded text: {len(huffman_encoded_text)} chars\n')
        handle.write(f'Zlib encoded text: {len(zlib_encoded_text)} chars\n\n')
        handle.write(f'Original text (first 1000 chars):\n{text[:1000]}\n\n')
        handle.write(f'Huffman encoded text (first 1000 chars):\n{huffman_encoded_text[:1000]}\n\n')
        handle.write(f'Zlib encoded text (first 1000 chars):\n{zlib_encoded_text[:1000]}\n\n')


def setup_logger():
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(asctime)s: %(message)s")


def parse_args():
    parser = parser = configargparse.ArgParser()
    parser.add('--n-colors', type=int, default=3)
    parser.add('--kmer-length', type=int, default=5)
    parser.add('--n-replicates', type=int, default=3)
    parser.add('-t', '--text', type=str)
    parser.add('-tf', '--text-file', type=str)
    parser.add('--counts-file', type=str, required=True)
    parser.add('--error-file', type=str, required=True)
    parser.add('--freq-file', type=str, default='letter_freq.txt')
    parser.add('-o', '--outfolder', type=str, required=True)
    parser.add('--disable-uncorrected-errors', default=False, action='store_true')
    parser.add('--disable-huffman', default=False, action='store_true')
    parser.add('--shift', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    setup_logger()
    settings = parse_args()
    if not os.path.isdir(settings.outfolder):
        os.mkdir(settings.outfolder)

    # These error rates accumulated all errors 
    # They indicate the chance that a color might be misassigned to any other color (Excluding X)
    # Calculated from an experiment with 6 color combinations
    category_error_rates = pd.read_csv(settings.error_file, index_col=0)['0']

    all_counts = pd.read_csv(settings.counts_file, index_col=0, header=0)['0']
    normalized_counts = all_counts / all_counts.sum()
    a, loc, scale = scipy.stats.powerlaw.fit(normalized_counts)
    powerlaw = scipy.stats.powerlaw(a, loc=loc, scale=scale)
    # plotfile = os.path.join(settings.outfolder, 'normalized_counts.png')
    # plot_counts(normalized_counts, powerlaw, plotfile)

    n_unmixed_colors = settings.n_colors
    n_mixed_colors = math.comb(settings.n_colors, 2)
    n_color_combinations = n_unmixed_colors + n_mixed_colors

    if settings.text is None:
        if settings.text_file is None:
            raise ValueError('Need at least one of --text or --text-file')
        with open(settings.text_file) as handle:
            text = handle.read()
            text = text.rstrip('\n')
    else:
        text = settings.text

    # Cleanup input text (needed for Huffman)
    cleaned_text = clean_text(text)

    # Setup error grids - 1 without error correction, 2 with error correction
    error_grid_uncorrected, error_grid_corrected = create_error_rate_grids(category_error_rates, n_mixed_colors, n_unmixed_colors)
    error_rates_corrected = precalculate_error_info(category_error_rates, settings.kmer_length)
    sampling_grids = prework_error_sampling_grids(settings.kmer_length)

    # Things to simulate
    # Without errors, with full errors, with error correction errors
    # Both huffman and zlib
    logger.info('Encoding huffman')
    huffman_encoding = encode_strings_huffman(cleaned_text, n_color_combinations, settings.freq_file)
    with open(os.path.join(settings.outfolder, 'huffman_encoding.txt'), 'w') as handle:
        handle.write(huffman_encoding)

    logger.info('Encoding zlib')
    zlib_encoding = encode_strings_zlib(text, n_color_combinations)
    with open(os.path.join(settings.outfolder, 'zlib_encoding.txt'), 'w') as handle:
        handle.write(zlib_encoding)

    logfile = os.path.join(settings.outfolder, 'log.txt')
    write_log(logfile, text, huffman_encoding, zlib_encoding)

    huffman_folder = os.path.join(settings.outfolder, 'huffman')
    logger.info('Running huffman simulations')
    main_simulation(huffman_encoding, huffman_folder, settings.n_replicates, powerlaw,
                    settings.kmer_length, n_color_combinations, settings.shift,
                    error_grid_uncorrected, error_rates_corrected, 
                    sampling_grids, settings.disable_uncorrected_errors)

    zlib_folder = os.path.join(settings.outfolder, 'zlib')
    logger.info('Running zlib simulations')
    main_simulation(zlib_encoding, zlib_folder, settings.n_replicates, powerlaw,
                    settings.kmer_length, n_color_combinations, settings.shift,
                    error_grid_uncorrected, error_rates_corrected, 
                    sampling_grids, settings.disable_uncorrected_errors)

    exit()