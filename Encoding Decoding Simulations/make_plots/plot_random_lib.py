import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'


def read_paint_data(infile, filter_x=True,
              lfill=5):
    data = {}
    with open(infile) as handle:
        _ = handle.readline()
        for line in handle:
            tabs = line.strip().split(',')
            name = tabs[0].ljust(lfill, '0')
            if filter_x and 'X' in tabs[0]:
                continue
            try:
                weight = int(tabs[1])
            except ValueError:
                continue
            data[name] = weight
    return data


def read_seqdata(infile, conversion_dict):
    data = {}
    with open(infile) as handle:
        _ = handle.readline() # header
        for line in handle:
            tabs = line.strip().split('\t')
            color_comb = ''.join(conversion_dict[color] for color in tabs[0].split('-'))
            total_reads = int(tabs[2])
            data[color_comb] = total_reads
    return data


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    paintfile = args[0] # Joined_bc_list-2025-12-15_19_56_46.csv
    seqfile = args[1] # umi_colors_3C.log

    conversion_dict = {
        'P2': '0',
        'P3': '1',
        'P5': '2',
    }

    paint_data = pd.Series(read_paint_data(paintfile), name='PAINT')
    seq_data = pd.Series(read_seqdata(seqfile, conversion_dict), name='Sequencing')
    df = pd.concat((paint_data, seq_data), axis=1)
    # Seq data contains all, paint data does not
    df = df.loc[~df.isna().any(axis=1)]
    res = scipy.stats.linregress(df['PAINT'], df['Sequencing'])

    fig, ax = plt.subplots()
    ax.scatter(df['PAINT'], df['Sequencing'],)
    ax.plot(df['PAINT'], res.intercept + res.slope * df['PAINT'], 'r',)
    ax.annotate(f'R^2: {res.rvalue**2}\n', (100, 15000))
    ax.set_xlabel('PAINT observations')
    ax.set_ylabel('Sequencing read counts')
    plt.savefig('Sequencing_color_correlation.svg')
    plt.close()
    df.to_csv('Sequencing_color_correlation.csv')
