from Bio import SeqIO
from collections import defaultdict
import os
import matplotlib.pyplot as plt

from analyze_barcodes import calc_average_from_counts, calc_std_from_counts, plot_counts_bins


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    try:
        infolder = args[0]
        outfolder = args[1]
    except:
        print("Usage: python extract_barcodes_minimap.py infolder outfolder\nInfolder should point to a folder with fastq files")

    q_counters = defaultdict(int)
    length_counters = defaultdict(int)

    for infile in os.listdir(infolder):
        handle = SeqIO.parse(os.path.join(infolder, infile), 'fastq')
        for record in handle:
            length_counters[len(record)] += 1
            for qual in record.letter_annotations['phred_quality']:
                q_counters[qual] += 1

    average_length = calc_average_from_counts(length_counters)
    std_length = calc_std_from_counts(length_counters, average_length)
    plot_counts_bins(length_counters, 'read lengths', 30, average_length, std_length, logy=False)
    plt.savefig(os.path.join(outfolder, 'read_lengths.png'))
    plt.close()

    filtered_lengths = {length:count for length, count in length_counters.items() if length <= 15000}

    average_length = calc_average_from_counts(filtered_lengths)
    std_length = calc_std_from_counts(filtered_lengths, average_length)
    plot_counts_bins(filtered_lengths, 'filtered read lengths', 30, average_length, std_length, logy=False)
    plt.savefig(os.path.join(outfolder, 'filtered_read_lengths.png'))
    plt.close()

    average_q = calc_average_from_counts(q_counters)
    std_q = calc_std_from_counts(q_counters, average_q)
    plot_counts_bins(q_counters, 'qualities', 30, average_q, std_q, logy=False)
    plt.savefig(os.path.join(outfolder, 'qualities.png'))
    plt.close()

    filtered_qs = {qual:count for qual, count in q_counters.items() if qual <= 50}

    average_q = calc_average_from_counts(filtered_qs)
    std_q = calc_std_from_counts(filtered_qs, average_q)
    plot_counts_bins(filtered_qs, 'filtered qualities', 30, average_q, std_q, logy=False)
    plt.savefig(os.path.join(outfolder, 'filtered_qualities.png'))
    plt.close()
