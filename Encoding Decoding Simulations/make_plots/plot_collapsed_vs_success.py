from collections import defaultdict
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys

sys.path.insert(1, os.path.abspath(os.path.join(__file__, '..', '..')))
plt.rcParams['svg.fonttype'] = 'none'

from analyse_simulations import get_header
from simulate_structures import clean_text

def count_edges(graph_file):
    with open(graph_file) as handle:
        return sum(1 for _ in handle) - 1


def count_edges_betw_junctions(graph_file):
    sources = defaultdict(int)
    targets = defaultdict(int)
    edges = defaultdict(int)
    with open(graph_file) as handle:
        _ = handle.readline()
        for line in handle:
            tabs = line.strip().split('\t')
            source = tabs[0]
            target = tabs[1]
            sources[source] += 1
            targets[target] += 1
            edges[source, target] += 1
    # Get junctions - nodes with at least one incoming and more than one outgoing or vice versa
    junctions = set()
    for node, target_count in targets.items():
        source_count = sources.get(node, 0)
        if target_count > 1 and source_count >= 1 or\
           target_count >= 1 and source_count > 1:
            junctions.add(node)
    # Return n. of edges between junctions
    return sum(edge_count for (source, target), edge_count in edges.items() if source in junctions and target in junctions) #, sources, targets, edges, junctions


def get_texts(infile):
    with open(infile) as handle:
        full_text = handle.read().rstrip('\n')
    return full_text, clean_text(full_text)


def analyse_run_folders(folder, decode_file, full_text, cleaned_text):
    analyzable_items = []
    items = os.listdir(folder)
    for f in items:
        full_name = os.path.join(folder, f)
        if os.path.isdir(full_name):
            analyzable_items.extend(analyse_run_folders(full_name, decode_file, full_text, cleaned_text))
    if decode_file in items:
        header = get_header(folder, n_levels, color_level)
        full_decode_file = os.path.join(folder, decode_file)
        ref_text = cleaned_text if 'huffman' in header else full_text
        decode_result, n_edges, n_edges_collapsed, n_connected_groups = analyse_result(full_decode_file, ref_text)
        analyzable_items.append(tuple(header) + (n_edges, n_edges_collapsed, n_connected_groups, decode_result))
    return analyzable_items


def analyse_result(decode_file, expected_result):
    # get probability and n. collapsed edges
    with open(decode_file) as handle:
        data = json.load(handle)
    n_edges_collapsed = data['data']['n_collapsed_edges_betw_junctions']
    n_edges = data['data']['n_edges']
    n_connected_groups = data['data']['n_connected_groups']
    decode_result = 0.0
    largest_group_id = data['data']['largest_group_id']
    for decoding_info in data['decodings']:
        if decoding_info['connected_group_id'] != largest_group_id:
            continue
        msg = decoding_info['decoding']
        if msg == expected_result:
            decode_result = decoding_info['relative_probability_global']
            break
    return decode_result, n_edges, n_edges_collapsed, n_connected_groups


if __name__ == '__main__':
    color_level = 1
    n_levels = 7
    reference_texts = {}
    dfs = []
    columns = ['text', 'color', 'kmer', 'shift', 'encoding', 'replicate', 'errors', 'error_correction', 'n_edges', 'n_edges_collapsed_junctions', 'n_conn_groups', 'decode_result']
    infolder = sys.argv[1]
    for subfolder in os.listdir(infolder):
        if subfolder not in reference_texts:
            ref_file = os.path.abspath(os.path.join(__file__, '..', '..', 'texts', subfolder))
            if os.path.isfile(ref_file):
                reference_texts[subfolder] = get_texts(ref_file)
            else:
                reference_texts[subfolder] = subfolder, clean_text(subfolder)

    subfolders_order = ['university', 'ki', 'fox', 'poem1', 'poem2', 'declaration', 'rothemund', 'alice', 'frankenstein']
    for subfolder in subfolders_order:
        full_folder = os.path.join(infolder, subfolder)
        if not os.path.isdir(full_folder):
            continue
        print(subfolder)
        res = analyse_run_folders(full_folder, 'decodes.json', *reference_texts[subfolder] )
        df = pd.DataFrame(res)
        df.columns = columns
        # Ignore larger shift data for now
        df = df.loc[df['shift'] == 'shift_1']
        dfs.append(df)

        # For this plot, we just want to show that zlib results in less tangles due to more even usage of nodes
        # So we do without errors or error corection
        sub_df = df[ (df['errors'] == 'no_errors') & (df['error_correction'] == 'normal') ]

        for encoding in 'huffman', 'zlib':
            subsubdf = sub_df[ sub_df['encoding'] == encoding ]
            subsubdf_grouped = subsubdf.groupby(by=['color', 'kmer'])['n_edges_collapsed_junctions'].mean()
            data = subsubdf_grouped.unstack()
            heatmap_file = f'{subfolder}_{encoding}_heatmap_collapsed_edges_betw_junctions.svg'
            sns.heatmap(data, annot=True)
            plt.savefig(heatmap_file)
            plt.close()
            subsubdf_grouped.to_csv(f'{subfolder}_{encoding}_heatmap_collapsed_edges_betw_junctions.csv')

    bigdf = pd.concat(dfs)
    # Now we want to show, overall, what causes things to fail

    # For error correction / no error correction, plot displots
    # one for collapsed edges
    # one for n. connected groups
    # Ignore uncorrected errors
    bigdf = bigdf[bigdf['errors'] != 'uncorrected_errors']
    # Add a column for pass fail depending on result
    bigdf.loc[:, 'decode_pass'] = (bigdf['decode_result'] >= 0.9).astype(int)

    # Plot how many edges are removed when applying error correction to the simulations
    # without errors
    bigdf_neg_edges = bigdf.copy()
    bigdf_neg_edges = bigdf_neg_edges.loc[bigdf_neg_edges["errors"] == "no_errors"]
    bigdf_neg_edges.loc[bigdf_neg_edges["error_correction"] == "corrected", "n_edges"] = bigdf_neg_edges.loc[bigdf_neg_edges["error_correction"] == "corrected", "n_edges"]*-1

    edges_removed = bigdf_neg_edges.groupby(["text", "color", "kmer", "encoding", "replicate"], as_index=False, sort=False)["n_edges"].sum()

    fig, ax = plt.subplots()
    sns.boxplot(data=edges_removed, x="text", y="n_edges", hue="encoding", ax=ax)
    ax.set_yscale('symlog')
    ax.set_ylim(-1)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel("N. edges removed by error correction")
    fig.tight_layout()
    plt.savefig("n_edges_removed_error_corr.svg")

    for error_correction in ('normal', 'corrected'):
        for property in ('n_edges_collapsed_junctions', 'n_conn_groups'):
            subdf = bigdf[bigdf['error_correction'] == error_correction]
            fig, ax = plt.subplots()
            sns.boxplot(data=subdf, x='text', y=property, hue='decode_pass',
                        ax=ax)
            ax.set_yscale('symlog')
            ax.set_ylim(-1)
            error_corr_text = '_error_corr' if error_correction == 'corrected' else ''
            plt.savefig(f'boxplot_succes_vs_{property}{error_corr_text}.svg')
            plt.close()
            subdf[['text', property, 'decode_pass']].to_csv(f'boxplot_succes_vs_{property}{error_corr_text}.csv')

    bigdf.to_csv('accumulated_simuation_data.csv')
