import json
import os
import py4cytoscape as p4c
import pandas as pd
import time


def set_style(network, node_column=None):
    style_name = 'DOCS'
    if node_column is not None:
        style_name = f'{style_name}_{node_column}'
    style_names = p4c.get_visual_style_names()
    if style_name in style_names:
        p4c.delete_visual_style(style_name) # Reset because edge width parameters are otherwise based on older style

    defaults = {'NODE_SHAPE': "circle", 'NODE_SIZE': 30, 
                'NODE_FILL_COLOR': "#0725CF",
                'EDGE_TARGET_ARROW_SHAPE': 'Arrow'}
    edgeWidth = p4c.map_visual_property('edge width', 'weight', 'c')
    mappers = [edgeWidth]
    if node_column is not None:
        mappers.append(p4c.map_visual_property('NODE_FILL_COLOR', node_column,
                                                'c', [0, 1], 
                                                ["#0725CF", "#D41515"])) # c for continuous. Should be discrete but it throws an error. Maybe because integer
    p4c.create_visual_style(style_name, defaults, mappers)
    p4c.set_visual_style(style_name, network=network)


def get_reference_nodes(reference_structures):
    nodes = set()
    for structure in reference_structures:
        nodes.add(structure[:-1])
        nodes.add(structure[1:])
    return nodes


def process_file(decode_folder, reference_structures, fused_structures=False):
    dtypes = {'Source': str,
            'Target': str,
            'weight': int,
            'info': str}

    decode_file = os.path.join(decode_folder, 'decodes.json')
    graph_file = os.path.join(decode_folder, 'graph.txt')

    edge_df = pd.read_csv(graph_file, sep='\t', dtype=dtypes)
    edge_df.columns = [c.lower() for c in edge_df.columns]

    all_nodes_set = set()
    for col in 'source', 'target':
        for node in edge_df[col]:
            all_nodes_set.add(node)
    all_nodes = list(all_nodes_set)

    with open(decode_file) as handle:
        decode_data = json.load(handle)
    # Set node property for top 3 decodes
    node_data = {}
    for decode_info in decode_data['decodings'][:3]:
        solution = decode_info['decoding']
        column_name = solution[:10]
        node_data[column_name] = {}
        for node in decode_info['graph_nodes']:
            node_data[column_name][node] = 1
    node_data['reference'] = {node: 1 for node in get_reference_nodes(reference_structures) if node in all_nodes_set}

    node_df = pd.DataFrame.from_dict(node_data)
    node_df = node_df.reindex(all_nodes)
    node_df = node_df.fillna(0).astype(int)
    node_df = node_df.reset_index()
    node_df = node_df.rename({'index': 'id'}, axis=1)

    title = decode_folder
    collection = 'fused_structures' if fused_structures else 'nonfused_structures'
    network = p4c.create_network_from_data_frames(nodes=node_df, edges=edge_df,
                                                    title=title,
                                                    collection=collection,
                                                    )

    set_style(network)
    time.sleep(2)
    outfile = os.path.join(decode_folder, 'graph.svg')
    if os.path.isfile(outfile):
        os.remove(outfile)
    p4c.network_views.export_image(outfile, type='SVG', network=network)
    for column in node_df.columns[1:]:
        set_style(network, column)
        outfile = os.path.join(decode_folder, f'graph_{column}.svg')
        if os.path.isfile(outfile):
            os.remove(outfile)
        time.sleep(2)
        p4c.network_views.export_image(outfile, type='SVG', network=network)


if __name__ == '__main__':

    infile = '/home/alexander/Documents/data_storage/Origami_barcode_project/data_analysis/experimental_data/250527/decoded/Joined_bc_list-2025-05-27_15_40_25_norm-filt.csv/graph.txt'
    decode_file = '/home/alexander/Documents/data_storage/Origami_barcode_project/data_analysis/experimental_data/250527/decoded/Joined_bc_list-2025-05-27_15_40_25_norm-filt.csv/decodes.json'

    infolder = 'experimental_data'

    dtypes = {'Source': str,
            'Target': str,
            'weight': int,
            'info': str}

    for date_folder in os.listdir(infolder):
        print(date_folder)
        for folder in ['decoded', 'decoded_error_corr']:
            decode_folder = os.path.join(infolder, date_folder, folder)
            for experiment_name in os.listdir(decode_folder):
                print(experiment_name)
                experiment_folder = os.path.join(decode_folder, experiment_name)
                decode_file = os.path.join(experiment_folder, 'decodes.json')
                graph_file = os.path.join(experiment_folder, 'graph.txt')

                edge_df = pd.read_csv(graph_file, sep='\t', dtype=dtypes)
                edge_df.columns = [c.lower() for c in edge_df.columns]

                all_nodes = set()
                for col in 'source', 'target':
                    for node in edge_df[col]:
                        all_nodes.add(node)
                all_nodes = list(all_nodes)

                with open(decode_file) as handle:
                    decode_data = json.load(handle)
                # Set node property for top 3 decodes
                node_data = {}
                solutions = []
                for i, decode_info in enumerate(decode_data['decodings'][:3], start=1):
                    solution = decode_info['decoding']
                    column_name = solution[:10]
                    node_data[column_name] = {}
                    for node in decode_info['graph_nodes']:
                        node_data[column_name][node] = 1

                node_df = pd.DataFrame.from_dict(node_data)
                node_df = node_df.reindex(all_nodes)
                node_df = node_df.fillna(0).astype(int)
                node_df = node_df.reset_index()
                node_df = node_df.rename({'index': 'id'}, axis=1)

                title = f'{experiment_name}_{folder}'
                network = p4c.create_network_from_data_frames(nodes=node_df, edges=edge_df,
                                                              title=title,
                                                              collection=date_folder,
                                                              )

                set_style()
                time.sleep(1)
                outfile = os.path.join(experiment_folder, 'graph.svg')
                if os.path.isfile(outfile):
                    os.remove(outfile)
                p4c.network_views.export_image(outfile, type='SVG')
                for column in node_df.columns[1:]:
                    set_style(column)
                    outfile = os.path.join(experiment_folder, f'graph_{column}.svg')
                    if os.path.isfile(outfile):
                        os.remove(outfile)
                    time.sleep(1)
                    p4c.network_views.export_image(outfile, type='SVG')
