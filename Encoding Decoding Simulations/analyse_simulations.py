import configargparse
import json
import os
import pandas as pd
import string

def analyse_result(decode_file, expected_result):
    # 0, if not present, 1 if present but not untied best solution, 2 if present and untied best solution
    max_score = 0
    result_score = None
    n_best_results = 0
    with open(decode_file) as handle:
        text = handle.read().rstrip('\n')
    # format:
    # !!>>
    # score
    # prob (optional)
    # <<!!
    # message
    parts = text.split('!!>>')
    for part in parts[1:]:
        score, msg = part.split('<<!!\n')
        msg = msg.strip('\n')
        score = float(score.strip().split('\n')[0])
        if score > max_score:
            max_score = score
            n_best_results = 1
        elif score == max_score:
            n_best_results += 1
        if msg == expected_result:
            result_score = score
    if result_score is None:
        return 0
    if result_score == max_score and n_best_results == 1:
        return 2
    return 1


def analyse_result(decode_file, expected_result):
    # get probability
    with open(decode_file) as handle:
        data = json.load(handle)
    largest_group_id = data['data']['largest_group_id']
    for decoding_info in data['decodings']:
        if decoding_info['connected_group_id'] != largest_group_id:
            continue
        msg = decoding_info['decoding']
        if msg == expected_result:
            return decoding_info['relative_probability_global']
    return 0.0



def get_run_folders(folder, required_file, expected_result, cleaned_text):
    analyzable_items = {}
    for f in os.listdir(folder):
        full_name = os.path.join(folder, f)
        if os.path.isdir(full_name):
            analyzable_items.update(get_run_folders(full_name, required_file, expected_result, cleaned_text))
        else:
            if f == required_file:
                header = get_header(folder, settings.n_levels, settings.color_level)
                if 'huffman' in header:
                    res = analyse_result(full_name, cleaned_text)
                elif 'zlib' in header:
                    res = analyse_result(full_name, expected_result)
                analyzable_items[header] = res
    return analyzable_items


def get_header(folder, n_levels, color_level):
    # Last 4 parts contain header info
    folder_names = folder.split(os.sep)[-n_levels:]
    header = []
    header.extend(folder_names[:color_level])
    header.extend(folder_names[color_level].split('_'))
    header.extend(folder_names[color_level+1:])
    return tuple(header)

def parse_args():
    parser = parser = configargparse.ArgParser()
    parser.add('-i', '--infolder', type=str, required=True)
    parser.add('-t', '--expected-text', type=str)
    parser.add('-tf', '--expected-text-file', type=str)
    parser.add('--color-level', type=int, default=0)
    parser.add('--n-levels', type=int, default=6)

    return parser.parse_args()

if __name__ == '__main__':
    settings = parse_args()

    if settings.expected_text_file is not None:
        with open(settings.expected_text_file) as handle:
            expected_text = handle.read().rstrip('\n')
    else:
        expected_text = settings.expected_text
    
    # Cleanup input text (needed for Huffman)
    cleaned_text = ''
    for char in expected_text.lower():
        if char in string.ascii_lowercase:
            cleaned_text += char

    results = get_run_folders(settings.infolder, 'decodes.json', expected_text, cleaned_text)
    series = pd.Series(results)
    data = series.unstack([-2, -1])
    if settings.n_levels == 5:
        order = (2, 0, 1, 3)
    elif settings.n_levels == 6:
        order = (3, 2, 0, 1, 4)
    else:
        raise ValueError(f'Unsupported n. of levels {settings.n_levels}')

    data.index = data.index.reorder_levels(order)
    data = data.reindex(['huffman', 'zlib'], level=0)
    data = data.sort_index()
    outfile = os.path.join(settings.infolder, 'overview_decodes.csv')
    data.to_csv(outfile, sep=',')

    # Average over replicates
    non_repl_levels = list(range(settings.n_levels - 1))
    data_avg_repl = data.groubpy(by=non_repl_levels).mean()
    outfile = os.path.join(settings.infolder, 'overview_decodes_averaged.csv')
    data_avg_repl.to_csv(outfile)