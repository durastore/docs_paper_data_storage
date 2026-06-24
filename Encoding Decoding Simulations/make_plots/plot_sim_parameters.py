import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'


if __name__ == '__main__':
    error_file = '../category_error_rates.csv'
    counts_file = '../structure_counts_no_errors.csv'

    category_error_rates = pd.read_csv(error_file, index_col=0)['0']
    index = category_error_rates.index.str.split('2')
    arr1, arr2 = zip(*index)
    category_error_rates.index = pd.MultiIndex.from_arrays((arr1, arr2))
    mapper = {'mix': 'mixed', 'unmix': 'single'}
    category_error_rates = category_error_rates.rename(mapper).unstack(0)
    fig, ax = plt.subplots()
    sns.heatmap(data=category_error_rates, ax=ax, annot=True,
                vmin=0, vmax=1.0, cmap=sns.color_palette('mako', as_cmap=True))
    ax.set_xlabel('Original color type')
    ax.set_ylabel('Mutated color type')
    ax.set_title('Categorical mutation rates')
    plt.savefig('category_error_rates.svg')
    plt.close()
    category_error_rates.to_csv('category_error_rates.csv')

    all_counts = pd.read_csv(counts_file, index_col=0, header=0)['0']
    normalized_counts = all_counts / all_counts.sum()
    a, loc, scale = scipy.stats.powerlaw.fit(normalized_counts)
    powerlaw = scipy.stats.powerlaw(a, loc=loc, scale=scale)
    fig, ax = plt.subplots()
    binned_counts, bins, _ = ax.hist(normalized_counts, bins=20, label='Binned counts')
    xx = np.linspace(powerlaw.ppf(0.01),
                     powerlaw.ppf(0.999), 101)
    yy = powerlaw.pdf(xx)
    xx = xx[1:]
    yy = yy[1:]
    ax.plot(xx, yy, label='Powerlaw fit')
    ax.set_xlabel('Normalized counts')
    ax.set_ylabel('Frequency of counts')
    ax.legend()
    plt.savefig('normalized_counts.svg')
    plt.close()
    df = pd.DataFrame(data=[xx, yy]).T
    df.columns=['x', 'y']
    df.to_csv('plot_fit_raw.csv')
