import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
sys.path.insert(1, os.path.abspath('..'))
plt.rcParams['svg.fonttype'] = 'none'


from simulate_structures import encode_strings_huffman, encode_strings_zlib, clean_text


if __name__ == '__main__':
    n_unmixed_colors = 3
    n_mixed_colors = math.comb(n_unmixed_colors, 2)
    n_color_combinations = n_unmixed_colors + n_mixed_colors
    plotfile = 'text_lengths.svg'

    infolder = '../texts'
    texts = {
        'stack': 'stack',
        'university': 'university',
        'ki': 'karolinskainstitutet',
             }
    tags = [
        'fox',
        'poem1',
        'poem2',
        'declaration',
        'rothemund',
        'alice',
        'frankenstein'
        ]
    data = []
    for tag in tags:
        print(f'{tag=}')
        infile = os.path.join(infolder, tag)
        with open(infile) as handle:
            text = handle.read()
        cleaned_text = clean_text(text)
        huffman_encoding = encode_strings_huffman(cleaned_text, n_color_combinations, '../letter_freq.txt')
        zlib_encoding = encode_strings_zlib(text, n_color_combinations)
        data.append((tag, len(cleaned_text), len(huffman_encoding), 'huffman'))
        data.append((tag, len(text), len(zlib_encoding), 'zlib'))
    for tag, text in texts.items():
        huffman_encoding = encode_strings_huffman(text, n_color_combinations, '../letter_freq.txt')
        zlib_encoding = encode_strings_zlib(text, n_color_combinations)
        data.append((tag, len(text), len(huffman_encoding), 'huffman'))
        data.append((tag, len(text), len(zlib_encoding), 'zlib'))

    df = pd.DataFrame(data)
    df.columns=['tag', 'length_original', 'length_encoding', 'encoding_scheme']

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    sns.lineplot(data=df, x='length_original', y='length_encoding', 
                 hue='encoding_scheme', style='encoding_scheme',
                 markers=True, ax=ax)
    ax.set_xlabel('Original length')
    ax.set_ylabel('Encoded length (6-ary)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()
    plt.savefig('encoding_lengths_log.svg')
    plt.close()

    df.to_csv('encoding_lengths.csv')