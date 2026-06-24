# Count the barcodes extracted
from collections import defaultdict
import os
import sys

if __name__ == '__main__':
    try:
        infolder = sys.argv[1]
        outfolder = sys.argv[2]
        tag = sys.argv[3]
    except:
        print('Usage: python count_barcodes.py infolder outfolder tag')
        exit()

    outfile = os.path.join(outfolder, f'counts_{tag}.txt')
    logfile = os.path.join(outfolder, f'counts_{tag}.log')

    umi_outfile = os.path.join(outfolder, f'umis_{tag}.txt')
    umis = defaultdict(int)
    barcode_combinations = defaultdict(int)
    reads_analyzed = 0

    for file_n, infile in enumerate(os.listdir(infolder)):
        if not infile.endswith('colors.txt'):
            continue
        with open(os.path.join(infolder, infile)) as handle:
            for reads_analyzed, line in enumerate(handle, start=reads_analyzed):
                tabs = line.strip('\n').split('\t')
                colors = tabs[1:6]
                umi = tabs[-1]
                if any(color == 'None' for color in colors) or umi == '':
                    continue
                umis[umi] += 1
                barcode_combinations[tuple(tabs[1:])] += 1

    total_extracted = sum(umis.values())
    perc_extracted = round(100 * total_extracted / reads_analyzed, 2)

    with open(logfile, 'w') as handle:
        handle.write(f'Found {reads_analyzed} aligned reads\n')
        handle.write(f'of which {total_extracted} ({perc_extracted}%) had all barcodes\n')

    with open(outfile, 'w') as handle:
        for combination, read_count in barcode_combinations.items():
            handle.write('\t'.join(combination) + f'\t{read_count}\n')

    # UMI counts (for shepherd)
    allowed_lengths = {24, 25, 26}
    with open(umi_outfile, 'w') as handle:
        for umi, count in umis.items():
            if len(umi) in allowed_lengths:
                handle.write(f'{umi}\t{count}\n')