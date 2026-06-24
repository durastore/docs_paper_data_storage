from collections import defaultdict
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import count_errors
import visualize_networks
import decode_structures

logger = logging.getLogger(__name__)
plt.rcParams['svg.fonttype'] = 'none'

def traverse_directory(folder, all_reference_structures, reference_overview,
                       error_count_args, start_string='Joined_bc_list'):
    error_results = {}
    error_results_no_x = {}
    if os.path.basename(folder).startswith('Fig.7'): # Just skip this, no path data
        return error_results, error_results_no_x
    for f in os.listdir(folder):
        full_name = os.path.join(folder, f)
        if os.path.isdir(full_name):
            res = traverse_directory(full_name, all_reference_structures, reference_overview,
                                                    error_count_args, start_string)
            error_results.update(res[0])
            error_results_no_x.update(res[1])
        else:
            if f.startswith(start_string) and f.endswith('.csv'):
                logger.info(f'Found file {full_name}')
                result_folder = os.path.splitext(full_name)[0]
                if not os.path.isdir(result_folder):
                    os.mkdir(result_folder)
                error_folder = os.path.join(result_folder, 'error_rates')
                error_folder_no_x = os.path.join(result_folder, 'error_rates_no_x')
                decode_folder = os.path.join(result_folder, 'decodes')
                decode_folder_fused_structures = os.path.join(result_folder, 'decodes_fused_structures')

                ref_name = get_reference_name(folder, reference_overview)
                reference_structures = all_reference_structures[ref_name]

                logger.info(f'Decoding')
                decode_structures.process_file(full_name, decode_folder, fuse_nearby_structures=False, output_json_detailed=True)
                decode_structures.process_file(full_name, decode_folder_fused_structures, fuse_nearby_structures=True, output_json_detailed=True)
                logger.info('Visualizing graphs')
                visualize_networks.process_file(decode_folder, reference_structures)
                visualize_networks.process_file(decode_folder_fused_structures, reference_structures, True)

                if ref_name == 'docs_stores_info' or ref_name == '':
                    logger.info(f'Skipping error correction on {ref_name}')
                    continue
                logger.info(f'Counting errors')
                error_results[full_name] = count_errors.process_file(full_name, error_folder, reference_structures, 
                                                               *error_count_args, filter_x=False)
                error_results_no_x[full_name] = count_errors.process_file(full_name, error_folder_no_x, reference_structures, 
                                                               *error_count_args, filter_x=True)

    return error_results, error_results_no_x


def read_references(ref_folder):
    reference_structures = defaultdict(set)
    for f in os.listdir(ref_folder):
        full_path = os.path.join(ref_folder, f)
        if f == 'reference_overview.csv':
            overview = read_ref_overview(full_path)
        else:
            with open(full_path) as handle:
                for line in handle:
                    reference_structures[f].add(line.strip())
    return reference_structures, overview


def read_ref_overview(infile):
    d = {}
    with open(infile) as handle:
        for line in handle:
            tabs = line.strip().split(',')
            fig_n = tabs[0]
            data_descr = tabs[1] if len(tabs) == 2 else None
            d[fig_n] = data_descr
    return d


def get_reference_name(infolder, reference_overview):
    path_parts = infolder.split(os.sep)
    for part in path_parts:
        if 'Fig.' in part:
            break
    return reference_overview[part.split(' ')[0][4:]]


def setup_logger(outfolder, name=__name__):
    logfile = os.path.join(outfolder, 'experimental_wrapper_log.txt')
    root_logger = logging.getLogger(name)
    root_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(logfile, mode='w')
    formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)



def plot_bars_from_df(data, title, plotfile):
    fig, ax = plt.subplots()
    data.T.plot.bar(ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(plotfile)
    plt.close(fig)


def process_error_results(error_results, filter_x=False):
    all_pos_data = {key: value[0] for key, value in error_results.items()}
    all_pos_data = pd.concat(all_pos_data, axis=1)
    suppl = '_no_x' if filter_x else ''
    title_suppl = ' (filtered X)' if filter_x else ''
    plot_bars_from_df(all_pos_data, f'Error rates per position{title_suppl}', os.path.join(infolder, f'all_error_rates{suppl}.svg'))

    all_hamming_data = {key: value[1] for key, value in error_results.items()}
    all_hamming_data = pd.concat(all_hamming_data, axis=1)
    plot_bars_from_df(all_hamming_data, f'Hamming distances{title_suppl}', os.path.join(infolder, f'all_hd_data{suppl}.svg'))

if __name__ == '__main__':
    import sys
    infolder = sys.argv[1]
    reference_folder = sys.argv[2]
    setup_logger(infolder)

    error_count_args = count_errors.setup()
    all_reference_structures, reference_overview = read_references(reference_folder)
    results, results_no_x = traverse_directory(infolder, all_reference_structures, reference_overview,
                       error_count_args)
    process_error_results(results, filter_x=False)
    process_error_results(results_no_x, filter_x=True)
