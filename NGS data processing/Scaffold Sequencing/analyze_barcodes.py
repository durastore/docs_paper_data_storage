# Analyze extracted barcodes

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import os
import pandas as pd

from get_read_distribution import calc_average_from_counts, calc_std_from_counts


def plot_counts_bins(counts, keyword, n_bins, avg, std, logy=True):
    xs = np.fromiter(counts.keys(), dtype=int)
    ys = np.fromiter(counts.values(), dtype=int)
    fig, ax = plt.subplots()
    ax.bar(xs, ys)
    ax.set_xlabel(keyword.capitalize())
    ax.set_ylabel('Counts')
    ax.set_title(f'{keyword.capitalize()} counts')
    ax.set_xticks(xs)
    txt = f'Average: {avg:.2f} +- {std:.2f}'
    anchored_text = AnchoredText(txt, loc='upper right')
    ax.add_artist(anchored_text)
    if logy:
        ax.set_yscale('log')
    fig.tight_layout()
    return xs, ys


def plot_hist(counts, keyword, average, std, logy=True):
    fig, ax = plt.subplots()
    n_bins = 20
    res = ax.hist(counts, bins=n_bins)
    ax.set_xlabel(keyword.capitalize())
    ax.set_ylabel('Counts')
    ax.set_title(f'{keyword.capitalize()} counts')
    txt = f'Average: {average:.2f} +- {std:.2f}'
    anchored_text = AnchoredText(txt, loc='upper right')
    ax.add_artist(anchored_text)
    if logy:
        ax.set_yscale('log')
    fig.tight_layout()
    return res


def plot_bar(counts, keyword, average, std, logy=True):
    fig, ax = plt.subplots()
    ys = np.bincount(counts)[1:]
    xs = np.arange(1, ys.size + 1)
    res = ax.bar(xs, ys)
    
    ax.set_xlabel(keyword.capitalize())
    ax.set_ylabel('Counts')
    ax.set_title(f'{keyword.capitalize()} counts')
    txt = f'Average: {average:.2f} +- {std:.2f}'
    anchored_text = AnchoredText(txt, loc='upper right')
    ax.add_artist(anchored_text)
    if logy:
        ax.set_yscale('log')
    fig.tight_layout()
    return xs, ys


def get_color_distribution_per_pos(color_combinations):
    for key in color_combinations:
        break
    nr_pos = len(key)
    colors_index = {'P2': 0, 'P3': 1, 'P5': 2}
    counts_per_pos = np.zeros((nr_pos, len(colors_index)), dtype=np.int64)
    for combination, count in color_combinations.items():
        for pos, color in enumerate(combination):
            counts_per_pos[pos][colors_index[color]] += count
    return counts_per_pos


def plot_colors_per_pos(counts_per_pos, colors=['r', 'b', 'y'], names=['P2', 'P3', 'P5']):
    labels = np.arange(1, 6)
    fig, ax = plt.subplots()
    bottom = np.zeros_like(labels)
    for row, color, name in zip(counts_per_pos.T, colors, names):
        ax.bar(labels, row, 0.35, color=color, label=name, bottom=bottom)
        bottom += row
    ax.set_title('Color count per position')
    ax.set_xlabel('Position')
    ax.set_ylabel('Color count')
    plt.legend()


def read_seq_to_clust(infile):
    seq_to_clust = {}
    with open(infile) as handle:
        _ = handle.readline() # skip header
        for line in handle:
            tabs = line.strip().split(',')
            seq_to_clust[tabs[0]] = int(tabs[1])
    return seq_to_clust


def map_color_to_most_abundant_umi(umis_per_colors):
    # Return a dict with 
    # a) the most abundant umi per color tuple
    # b) the ratio of this UMIs read count to others
    most_ab_umi_per_colors = {}
    ratio_read_counts = {}
    for color_combination, umis_color in umis_per_colors.items():
        total_reads = 0
        max_reads = 0
        next_best = 0
        for umi, read_count in umis_color.items():
            total_reads += read_count
            if read_count > max_reads:
                next_best = max_reads
                max_reads = read_count
                best_umi = umi
            elif read_count > next_best:
                next_best = read_count
        most_ab_umi_per_colors[color_combination] = best_umi
        ratio_read_counts[color_combination] = max_reads / total_reads, next_best, max_reads/next_best
    return most_ab_umi_per_colors, ratio_read_counts


if __name__ == '__main__':
    image_filetype = 'svg'
    mapfile = ''
    try:
        base_folder = sys.argv[1]
        outfolder = sys.argv[2]
        tag = sys.argv[3]
    except:
        print('Usage: python count_barcodes.py base_folder outfolder tag')
        exit()

    infile = os.path.join(base_folder, f'counts_{tag}.txt')
    logfile = os.path.join(outfolder, f'analysis_{tag}.log')
    color_umi_logfile = os.path.join(outfolder, f'umi_colors_{tag}.log')
    plotfiles = {'umi': f'umi_read_counts_{tag}',
                 'all': f'all_read_counts_{tag}',
                 'umi lengths': f'umi_lengths_read_counts_{tag}',
                 'colors': f'colors__read_counts_{tag}',
                 'umis per colors': f'umis_per_colors_{tag}',
                 'colors per umi': f'colors_per_umi_{tag}',
                }
    for key, f in plotfiles.items():
        plotfiles[key] = os.path.join(outfolder, f)
    colors_per_pos_plotfile = os.path.join(outfolder, f'colors_per_pos_{tag}.svg')
    colors_per_pos_datafile = os.path.join(outfolder, f'colors_per_pos_{tag}.csv')

    if os.path.isfile(mapfile):
        seq_to_clust = read_seq_to_clust(mapfile)
        map_umi = lambda umi: seq_to_clust.get(umi)
    else:
        map_umi = lambda umi: umi

    if outfolder and not os.path.isdir(outfolder):
        os.mkdir(outfolder)

    allowed_lengths = {24, 25, 26}

    reads_analyzed = 0
    umi_lengths = defaultdict(int)
    color_combinations = defaultdict(int)
    umis_per_colors = defaultdict(lambda: defaultdict(int))
    umis = defaultdict(int)
    barcode_combinations = defaultdict(int)
    colors_per_umi = defaultdict(set)

    with open(infile) as handle:
        for line in handle:
            tabs = line.strip('\n').split('\t')
            org_umi = tabs[5]
            len_umi = len(org_umi)
            if len_umi not in allowed_lengths:
                continue
            umi = map_umi(org_umi)
            if umi is None:
                continue

            colors = tabs[:5]
            count = int(tabs[6])
            color_tuple = tuple(tabs[:-2])
            combinations_tuple = color_tuple + (umi, )
            umis[umi] += count
            barcode_combinations[combinations_tuple] += count
            color_combinations[color_tuple] += count
            umi_lengths[len_umi] += count
            umis_per_colors[color_tuple][umi] += count
            colors_per_umi[umi].add(color_tuple)

    n_umis_per_colors = {colors: len(umis) for colors, umis in umis_per_colors.items()}
    n_colors_per_umis = {umi: len(colors) for umi, colors in colors_per_umi.items()}
    total_extracted = sum(umis.values())

    data = {'umi': umis, 'all': barcode_combinations, 'colors': color_combinations, 'umis per colors': n_umis_per_colors,
            'colors per umi': n_colors_per_umis}
    data_compiled = {}
    for keyword, counts in data.items():
        counts_np = np.fromiter(counts.values(), dtype=np.int64)
        average = np.mean(counts_np)
        std = np.std(counts_np)
        min_count = counts_np.min()
        max_count = counts_np.max()
        n_unique = len(counts)
        if np.unique(counts_np).size >= 20:
            res = plot_hist(counts_np, keyword, average, std, logy=True)
            values = res[0]
            bins = [f'{res[1][i]} - {res[1][i+1]}' for i in range(len(res[1]) - 1) ]
        else:
            bins, values = plot_bar(counts_np, keyword, average, std, logy=True)

        data_tmp = pd.DataFrame((bins, values)).T
        data_tmp.columns = ['bins', 'values']
        filebase = plotfiles[keyword]
        plotfile = f'{filebase}.{image_filetype}'
        datafile = f'{filebase}.csv'
        plt.savefig(plotfile)
        plt.close()
        data_tmp.to_csv(datafile)
        data_compiled[keyword] = average, std, min_count, max_count, n_unique
        
    average_length = calc_average_from_counts(umi_lengths)
    std_length = calc_std_from_counts(umi_lengths, average_length)
    edges, counts_arr = plot_counts_bins(umi_lengths, 'umi lengths', 3, average_length, std_length, logy=False)
    
    filebase = plotfiles['umi lengths']
    plotfile = f'{filebase}.{image_filetype}'
    plt.savefig(plotfile)
    plt.close()
    data_compiled['umi lengths'] = average_length, std_length, min(umi_lengths), max(umi_lengths), len(umi_lengths)

    most_ab_umi_per_colors, ratio_read_counts = map_color_to_most_abundant_umi(umis_per_colors)
    with open(color_umi_logfile, 'w') as handle:
        handle.write('\t'.join( ('Color combination', 'Number of UMIs', 'Total reads',
                                 'Most abundant UMI (or UMI cluster ID)', 'Read counts most abundant UMI total', 'Read counts most abundant UMI this color', 'Total color combinations UMI',
                                 'Fraction of total read counts in most abundant umi', 
                                 'Read count next best UMI', 'Ratio most abundant to next best')) + '\n')
        for color_combination, (ratio, next_best_read_count, next_best_ratio) in sorted(ratio_read_counts.items(), key=lambda x: x[1][0], reverse=True):
            color_combination_str = '-'.join(color_combination)
            best_umi = most_ab_umi_per_colors[color_combination]
            handle.write('\t'.join( map(str, (color_combination_str, len(umis_per_colors[color_combination]),
                                              color_combinations[color_combination], best_umi, umis[best_umi], umis_per_colors[color_combination][best_umi], n_colors_per_umis[best_umi],
                                              round(ratio, 4), next_best_read_count, round(next_best_ratio, 4)) )) + '\n')

    with open(logfile, 'w') as handle:
        handle.write(f'Total UMIs or UMI clusters processed: {total_extracted}\n')
        for keyword, data in data_compiled.items():
            handle.write(f'\n{keyword.capitalize()} stats:\n')
            handle.write(f'Number of different types: {data[4]}\n')
            handle.write(f'Average: {data[0]:.2f} +- {data[1]:.2f}\n')
            handle.write(f'Min: {data[2]:.2f}\tMax: {data[3]:.2f}\n')


    # Color counts per position
    counts_per_pos = get_color_distribution_per_pos(color_combinations)
    counts_per_pos_df = pd.DataFrame(counts_per_pos)
    counts_per_pos_df.columns = ['P2', 'P3', 'P5']
    counts_per_pos_df.to_csv(colors_per_pos_datafile)
    plot_colors_per_pos(counts_per_pos)
    plt.savefig(colors_per_pos_plotfile)
    plt.close()
    
