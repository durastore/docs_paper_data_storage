from collections import defaultdict, deque
import configargparse
from functools import cache, partial
import huffman
import itertools
import json
import logging
import math
import numpy as np
import os
import scipy.stats
from scipy.special import logsumexp
import string
import sys
import time
import zlib
from hexary_encoding import load_freqs

NUMERALS=string.digits + string.ascii_letters + string.punctuation
NUMERALS_MAP = {digit: index for index, digit in enumerate(NUMERALS)}
sys.set_int_max_str_digits(7500)
FILL_CHARACHTER=NUMERALS[0]

def read_data(infile, filter_x=True, cutoff=100):
    data = {}
    with open(infile) as handle:
        _ = handle.readline()
        for line in handle:
            tabs = line.strip().split(',')
            if filter_x and 'X' in tabs[0]:
                continue
            try:
                weight = int(tabs[1])
            except ValueError:
                continue
            if weight < cutoff:
                continue
            data[tabs[0]] = weight
    return data


def form_connections(data, shift):
    connections = defaultdict(lambda: defaultdict(list))
    node_weights = defaultdict(int)
    for code, weight in data.items():
        left = code[:-shift]
        right = code[shift:]
        new_info = right[-shift:]
        new_connection = {'weight': weight, 'info': new_info}
        connections[left][right].append(new_connection)
        node_weights[left] += weight
        node_weights[right] += weight
    return connections, node_weights


def flip_connections(connections):
    # Does not remember edge weights!
    flipped_connections = defaultdict(lambda: defaultdict(list))
    for node1, conn_nodes in connections.items():
        for node2, edges in conn_nodes.items():
            for edge in edges:
                flipped_connections[node2][node1].append(edge)
    return flipped_connections


def find_starts_ends(connections):
    # Definite start nodes are those with no incoming edges, only outgoing
    # Definite end nodes are those with no outgoing edges, only incoming
    all_nodes_with_incoming = set()
    for node, connected_nodes in connections.items():
        all_nodes_with_incoming |= set(connected_nodes)
    starts = set(connections) - all_nodes_with_incoming
    ends = all_nodes_with_incoming - set(connections)
    return starts, ends


def find_junctions(connections, flipped_connections):
    # Get junctions, nodes with at least one incoming connection and at least two outgoing connections
    # or vice versa
    junctions = set()
    all_candidates = set(connections) | set(flipped_connections)
    # Count number of outgoing connections per node to find any nodes that represent junctions
    for node in all_candidates:
        conns_out = connections.get(node, {})
        conns_in = flipped_connections.get(node, {})
        if (len(conns_out) > 1 and len(conns_in) >= 1) or\
           (len(conns_in) > 1 and len(conns_out) >= 1):
            junctions.add(node)
    return junctions


def get_connected_components(connections):
    # Convert to unidirected
    adj = defaultdict(set)
    all_nodes = set()
    for src, targets in connections.items():
        all_nodes.add(src)
        for t in targets:
            all_nodes.add(t)
            adj[src].add(t)
            adj[t].add(src) # Make it undirected

    visited = set()
    node_to_group = {}
    group_id = 0

    for node in all_nodes:
        if node in visited:
            continue
        # Standard BFS
        queue = deque([node])
        visited.add(node)
        while queue:
            curr = queue.popleft()
            node_to_group[curr] = group_id
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        group_id += 1
    return node_to_group


def get_all_connected_nodes(node_to_group):
    groups = defaultdict(set)
    for node, group_id in node_to_group.items():
        groups[group_id].add(node)
    return groups



def get_edge_weights(connections):
    edge_weights = {}
    for node1, nodes_connected in connections.items():
        for node2, connection_datas in nodes_connected.items():
            edge_weights[node1, node2] = connection_datas[0]['weight']
    return edge_weights


def write_graph(connections, outfile):
    with open(outfile, 'w') as handle:
        written_header = False
        for left, rights in connections.items():
            for right, edges in rights.items():
                for data in edges:
                    if not written_header:
                        header = 'Source\tTarget\t'
                        keys = data.keys()
                        for key in keys:
                            header += f'{key}\t'
                        header = header[:-1] + '\n'
                        handle.write(header)
                        written_header = True
                    to_write = f'{left}\t{right}\t'
                    for key in keys:
                        to_write += f'{data[key]}\t'
                    to_write = to_write[:-1] + '\n'
                    handle.write(to_write)



def read_graph(infile):
    connections = defaultdict(lambda: defaultdict(list))
    with open(infile) as handle:
        header = handle.readline()
        data_keys = header.strip().split('\t')[2:]
        for line in handle:
            tabs = line.strip().split('\t')
            left, right = tabs[:2]
            edge_info = {key: eval(value) for key, value in zip(data_keys, tabs[2:])}
            connections[left][right].append(edge_info)
    return connections


def write_node_weights(node_weights, outfile):
    with open(outfile, 'w') as handle:
        handle.write('Node\tFrequency\n')
        for node, weight in node_weights.items():
            handle.write(f'{node}\t{weight}\n')


class PathFinder:


    def __init__(self, connections, edge_weights, scoring_data, 
                 max_cons_score_decreases, timeout, scoring_scheme):
        self.connections = connections
        self.edge_weights = edge_weights
        self.scoring_data = scoring_data
        self.max_cons_score_decreases = max_cons_score_decreases
        self.timeout = timeout
        self.start_time = None
        self.edges_seen = None
        self.timed_out = False
        self.hit_recursion_limit = 0
        self.slope = scoring_data[0]
        self.intercept = scoring_data[1]
        if scoring_scheme == 'simple':
            self.calculate_score = self.calculate_score_simple
        elif scoring_scheme == 'weighted':
            self.calculate_score = self.calculate_score_weighted
        else:
            raise ValueError('Invalid scoring scheme')


    def find_all_paths(self, starts):
        # Brute force all paths - assume no edge can be visited twice
        paths = {}
        self.start_time = time.time()
        for start_node in starts:
            # print(f'{start_node=}')
            path = (start_node, )
            edges_seen = defaultdict(int)
            try:
                paths[start_node] = self.find_path_recursive(path, edges_seen, start_node, 0)
            except RecursionError:
                self.hit_recursion_limit += 1
        return paths


    def find_path_recursive(self, current_path, edges_seen, current_encoded_string, prev_cons_score_decreases):
        # print(f'{current_path=}')
        current_time = time.time()
        self.timed_out = (current_time - self.start_time) > self.timeout
        if self.timed_out:
            return []
        current_score = self.calculate_score(edges_seen)
        # print(f'{current_score=}')
        current_node = current_path[-1]
        next_connections = self.connections.get(current_node, {})
        # print(f'{next_connections=}')
        paths = []
        for end_node, edges in next_connections.items():
            for connection_data in edges: # In case multiple edges exist between nodes (for collapsed paths)
                edges_seen_new = edges_seen.copy()
                n1 = current_node
                for n2 in connection_data['nodes']:
                    edges_seen_new[n1, n2] += 1
                    n1 = n2
                new_score = self.calculate_score(edges_seen_new)
                # print(f'{end_node=}')
                # print(f'{new_score=}')
                if new_score is None:
                    continue
                if new_score < current_score:
                    # Allow no more than max_cons_score_decreases consecutively
                    consecutive_score_decreases = prev_cons_score_decreases + 1
                    if consecutive_score_decreases > self.max_cons_score_decreases:
                        continue
                else:
                    consecutive_score_decreases = 0
                path = current_path + (end_node, )
                encoded_string = current_encoded_string + connection_data['info']
                paths.extend(self.find_path_recursive(path, edges_seen_new, encoded_string, consecutive_score_decreases))
        # If we dont get any new paths, that means our current path is the best one, unless we have prev_cons_score_decreases
        # This happens if next_connections is empty, but also allows us to end at a junction if we already used outgoing edges
        if not paths and prev_cons_score_decreases == 0:
            paths.append((current_encoded_string, current_path, current_score, edges_seen))
        return paths


    def calculate_score_simple(self, edge_selection_counts):
        # For each edge, see how many times it is selected in the current counts, then calculate the score
        # Each selected edge add 1 to the score
        # If an edge is selected multiple times, the score decreases, but the decrease is less if the structure is highly abundant
        score = 0
        for edge, n_times_seen in edge_selection_counts.items():
            if n_times_seen == 1:
                score += 1
            elif n_times_seen > 1:
                weight = self.edge_weights[edge]
                penalty = (n_times_seen - 1) * (np.log(weight) * self.slope + self.intercept)
                score -= penalty
        return score


    def calculate_score_weighted(self, edge_selection_counts):
        # For each edge, see how many times it is selected in the current counts, then calculate the score
        # Each selected edge adds it's weight to the score
        # If an edge is selected multiple times, the score decreases, but the decrease is less if the structure is highly abundant
        score = 0
        for edge, n_times_seen in edge_selection_counts.items():
            weight = self.edge_weights[edge]
            if n_times_seen == 1:
                factor = 1
            elif n_times_seen > 1:
                # the "1 - " here is to convert a penalty that would only reduce the score to one that lowers the score when multiplying
                factor = (1 - (np.log(weight) * self.slope + self.intercept)) ** (n_times_seen - 1)
            score += factor * weight
        return score


def precalculate_scoring_data(counts):
    # Calculate a linear slope between the log of the frequency and the score penalty
    # At highest possible frequency, the score penalty will be 0.1
    # At lowest possible frequency, it will be 0.9
    max_count = counts.max()
    min_count = counts.min()
    lowest_x = np.log(min_count)
    highest_x = np.log(max_count)
    slope = -0.8 / (highest_x - lowest_x) # decrease from 0.9 to 0.1 as dy
    intercept = 0.9 - lowest_x * slope
    return slope, intercept


def collapse_paths(starts, junctions, connections):
    # Collapse straight paths starting from any start of junction point,
    # ending at any end or junction point
    # accumulate edge weight and info
    new_connections = defaultdict(lambda: defaultdict(list))
    for node in starts | junctions:
        # print(f'start node {node}')
        connected_nodes = connections.get(node, {})
        for path_head in connected_nodes:
            accumulated_nodes = []
            accumulated_weights = []
            accumulated_info = ''
            new_connected_nodes = connected_nodes
            while True: # Keep adding nodes as long as there is no junction
                # print(f'{path_head=}')
                accumulated_nodes.append(path_head)
                data = new_connected_nodes[path_head][0]
                accumulated_info += data['info']
                accumulated_weights.append(data['weight'])
                new_connected_nodes = connections.get(path_head)
                if new_connected_nodes is None or path_head in junctions:
                    break
                for path_head in new_connected_nodes: # Single item for loop to set the path head
                    break
            # Set as a new connection
            new_connections[node][path_head].append({'nodes': accumulated_nodes, 
                                                     'weights': accumulated_weights,
                                                     'info': accumulated_info})
    return new_connections


def analyze_collapsed(collapsed_connections, junctions):
    n_edges = 0
    n_edges_between_junctions = 0
    edge_sizes = []
    for node1, connected in collapsed_connections.items():
        for node2, edges in connected.items():
            n_edges += len(edges)
            if node1 in junctions and node2 in junctions:
                n_edges_between_junctions += len(edges)
            for edge in edges:
                edge_sizes.append(len(edge['nodes']))
    return n_edges, n_edges_between_junctions, edge_sizes


def get_structures_seen(edges_seen):
    # Simply convert the tuple of the edge to the structure
    counts = defaultdict(int)
    for (node1, node2), count in edges_seen.items():
        counts[node1] += count
    counts[node2] += count # Add end node
    return counts


def count_nodes_seen(collapsed_connections, collapsed_path):
    nodes_seen_counter = defaultdict(int)
    for node1, node2 in itertools.pairwise(collapsed_path):
        for node in itertools.chain((node1, node2), collapsed_connections[node1][node2]['nodes']):
            nodes_seen_counter[node] += 1
    return nodes_seen_counter


def fuse_structures(structure_data, subsets):
    # Compare all structures pairwise
    # add structures if they are within a distance of 1
    groups = []
    seen = set()
    for structure in structure_data:
        if structure in seen:
            continue
        group = set()
        queue = deque()
        queue.append(structure)
        while queue:
            current_structure = queue.pop()
            if current_structure in seen:
                continue
            group.add(current_structure)
            seen.add(current_structure)
            current_count = structure_data[current_structure]
            for structure2 in generate_structure_derivatives(current_structure, subsets):
                count2 = structure_data.get(structure2)
                if count2 is None or structure2 in seen or count2 >= current_count:
                    continue
                queue.append(structure2)
        groups.append(group)
    return groups


def generate_structure_derivatives(structure, subsets):
    # Generate all derivatives within one distance
    for i, char in enumerate(structure):
        for replacement_char in subsets[char]:
            yield f"{structure[:i]}{replacement_char}{structure[i+1:]}"


def prework_subset(available_characters):
    subset = {}
    s = set(available_characters)
    for char in s:
        remaining = s - {char, }
        subset[char] = remaining
    return subset


def combine_groups(fused_groups, data):
    # Assign total counts per group to the most abundant structure
    cleaned_data = {}
    for group in fused_groups:
        total = 0
        max_count = 0
        for structure in group:
            count = data[structure]
            total += count
            if count > max_count:
                max_structure = structure
                max_count = count
        #     print(structure, count)
        # print(f'{total=}\n')
        cleaned_data[max_structure] = total
    return cleaned_data


class ProbabilityEstimator:


    def __init__(self, mean_count, std_count, min_count=1):
        dists = scipy.stats.Normal(mu=[min_count, mean_count], sigma=std_count)
        self.prob_func = dists.logpdf
        #self.dists = scipy.stats.poisson(np.array([min_count, mean_count]))
        #self.prob_func = dists.logpmf


    @cache
    def eval_edge_count(self, count):
        logps = self.prob_func(count)
        log_denom = logsumexp(logps)
        return logps - log_denom


    def calculate_relative_probabilities(self, edges_seen, connections, connected_group=None):
        # Evaluate per connected group, or all
        if connected_group is None:
            connected_group = connections
        sum_logp = 0
        for node1 in connected_group:
            connected_nodes = connections[node1]
            for node2, edges in connected_nodes.items():
                edge = edges[0] # Only one edge in uncollapsed paths
                weight = edge['weight']
                logp0, logp1 = self.eval_edge_count(weight)
                count = edges_seen.get((node1, node2), 0)
                if count > 0:
                    sum_logp += logp1
                else:
                    sum_logp += logp0
        return sum_logp


def decode_zlib(n_color_combinations, encoded_text, 
                encoding_scheme='utf-8'):
    total_int = 0
    current_mult = 1
    for i, digit in enumerate(encoded_text[::-1], start=0):
        digit_n = NUMERALS_MAP[digit]
        total_int += digit_n * current_mult
        current_mult *= n_color_combinations
    bytes_needed = int(math.ceil(total_int.bit_length() / 8))
    compressed_bytes = total_int.to_bytes(length=bytes_needed)
    decoded_message = zlib.decompress(compressed_bytes)
    return decoded_message.decode(encoding_scheme)


def write_decodes(path_infos, outfile):
    # format:
    # !!>>
    # score
    # relative_probability
    # group id
    # <<!!
    # message
    with open(outfile, 'w') as handle:
        for path_info in path_infos:
            decoded_message = path_info['decoding']
            score = path_info['score']
            rel_prob = path_info['relative_probability_component']
            rel_prob_glob = path_info['relative_probability_global']
            group_id = path_info['connected_group_id']
            handle.write(f'!!>>\n{score}\n{rel_prob}\n{rel_prob_glob}\n{group_id}\n<<!!\n{decoded_message}\n')


def setup_logger(outfolder):
    logfile = os.path.join(outfolder, 'log.txt')
    root_logger = logging.getLogger(__name__)
    # Clear wrappers
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(logfile, mode='w')
    formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    return root_logger


def parse_args():
    parser = parser = configargparse.ArgParser()
    parser.add('--n-colors', type=int, default=3)
    parser.add('-i', '--infile', type=str, required=True)
    parser.add('-o', '--outfolder', type=str, required=True)
    parser.add('--correct-errors', default=False, action='store_true')
    parser.add('-d', '--decoding-scheme', choices=['huffman', 'zlib'], required=True)
    parser.add('--read-cutoff', type=int, default=1)
    parser.add('--shift', type=int, default=1)
    parser.add('--timeout', type=float, default=np.inf)
    parser.add('--scoring-scheme', choices=['simple', 'weighted'], default='simple')
    parser.add('--output-detailed-json', default=False, action='store_true')
    return parser.parse_args()


def process_file(infile, outfolder, n_colors=3,
                 decoding_scheme='huffman',
                 fuse_nearby_structures=False,
                 shift=1, timeout=np.inf,
                 scoring_scheme='weighted',
                 output_json_detailed=False,
                 read_cutoff=1, logger=None):
    # Just a wrapper, mostly for experimental data
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)
    n_color_combinations = n_colors + math.comb(n_colors, 2)

    if logger is None:
        logger = setup_logger(outfolder)

    logger.info('Loading data')
    data = read_data(infile, cutoff=read_cutoff)
    counts = np.fromiter(data.values(), dtype=int)
    mean_count = counts.mean()
    std_count = counts.std()

    if fuse_nearby_structures:
        logger.info('Applying error correction (fusing structures)')
        available_characters = list(NUMERALS[:n_color_combinations])
        subsets = prework_subset(available_characters)
        fused_groups = fuse_structures(data, subsets)
        return fused_groups
        data = combine_groups(fused_groups, data)

    logger.info('Prepwork')
    connections, node_weights = form_connections(data, shift)
    edge_weights = get_edge_weights(connections)
    flipped_connections = flip_connections(connections)
    node_to_group = get_connected_components(connections)
    connected_groups = get_all_connected_nodes(node_to_group)
    largest_group_size = 0
    for group_id, group in connected_groups.items():
        size = len(group)
        if size > largest_group_size:
            largest_group_size = size
            largest_connected_group_id = group_id

    outfile = os.path.join(outfolder, 'graph.txt')
    write_graph(connections, outfile)
    write_node_weights(node_weights, os.path.join(outfolder, 'node_weights.txt'))

    junctions = find_junctions(connections, flipped_connections)

    # Find definite starts/end: nodes with only outgoing connections or only incoming connections
    starts, ends = find_starts_ends(connections)

    # Collapse, starting from a start or a junction (either type), until another junction or an end is reached
    # remember all edge weights / added characters in between
    collapsed_connections = collapse_paths(starts, junctions, connections)
    outfile = os.path.join(outfolder, 'collapsed_graph.txt')
    write_graph(collapsed_connections, outfile)

    scoring_data = precalculate_scoring_data(counts)
    # If no starting nodes, set all junctions to starts
    if len(starts) == 0:
        starts = junctions
    logger.info('Searching paths')
    pathfinder = PathFinder(collapsed_connections, edge_weights, scoring_data,
                            max_cons_score_decreases=1, timeout=timeout,
                            scoring_scheme=scoring_scheme)
    return pathfinder, starts
    paths = pathfinder.find_all_paths(starts)
    if pathfinder.hit_recursion_limit > 0:
        logger.info(f'Recursion limit reached for {pathfinder.hit_recursion_limit} out of {len(starts)} start nodes')
    if pathfinder.timed_out:
        logger.info('Timed out')

    if decoding_scheme == 'huffman':
        freqs = load_freqs()
        nary_huffman = huffman.HuffmanCode(freqs, n_color_combinations)
        decoding_function = nary_huffman.decode
    elif decoding_scheme == 'zlib':
        decoding_function = partial(decode_zlib, n_color_combinations)

    logger.info('Sorting paths')
    unpacked_paths = sorted((path_info for start_node_paths in paths.values() for path_info in start_node_paths),
                            key=lambda x: x[2], reverse=True)

    probscorer = ProbabilityEstimator(mean_count, std_count)

    logger.info('Decoding paths')
    all_path_info = defaultdict(list)
    grouped_logprob_scores = defaultdict(list)
    n_decoded = 0
    start_time = time.time()
    global_log_probs = []
    for message, collapsed_path, score, edges_seen in unpacked_paths:
        group_id = node_to_group[collapsed_path[0]]
        connected_group = connected_groups[group_id]

        prob_score_group = probscorer.calculate_relative_probabilities(edges_seen, connections, connected_group)
        prob_score_global = probscorer.calculate_relative_probabilities(edges_seen, connections)

        path_info = {
            'decoding': None,
            'score': score,
            'connected_group_id': group_id,
            'message': message,
            'global_prob_score': prob_score_global,
        }
        if output_json_detailed:
            path_info['graph_nodes'] = get_structures_seen(edges_seen)
            path_info['collapsed_path'] = collapsed_path

        grouped_logprob_scores[group_id].append(prob_score_group)
        global_log_probs.append(prob_score_global)

        while True:
            try:
                decoded_message = decoding_function(message)
                path_info['decoding'] = decoded_message
                n_decoded += 1
                # print(message, decoded_message, n_decoded)
            except:
                if shift > 1 and message.endswith(FILL_CHARACHTER):
                    message = message[:-1]
                    continue
            break
        all_path_info[group_id].append(path_info)
        time_now = time.time()
        if time_now - start_time > timeout:
            logger.info(f'Timeout decoding after {timeout} seconds')
            break

    logger.info('Calculating probabilities')
    global_log_probs = np.array(global_log_probs)
    logprob_global_denom = logsumexp(global_log_probs)
    print(f'{logprob_global_denom=}')
    # Normalize all log probability scores per group
    for group_id, logprob_scores in grouped_logprob_scores.items():
        logprob_scores = np.array(logprob_scores)
        logprob_denom = logsumexp(logprob_scores)
       
        rel_logprobscores = np.exp(logprob_scores - logprob_denom)
        for path_info, rel_prob in zip(all_path_info[group_id], rel_logprobscores):
            path_info['relative_probability_component'] = rel_prob
            path_info['relative_probability_global'] = np.exp(path_info.pop('global_prob_score') - logprob_global_denom)

    
    logger.info('Writing outputs')
    # Filter and flatten the paths (only successes)
    outfile = os.path.join(outfolder, 'decodes.txt')
    filtered_path_info = []
    for group_id, path_infos in all_path_info.items():
        grouped_info = []
        for path_info in path_infos:
            if path_info['decoding'] is not None:
                grouped_info.append(path_info)
        if grouped_info:
            filtered_path_info.extend(grouped_info)
    write_decodes(filtered_path_info, outfile)

    n_collapsed_edges, n_collapsed_edges_betw_junctions, edge_sizes = analyze_collapsed(collapsed_connections, junctions)
    edge_sizes = np.array(edge_sizes)
    json_data = {'data': {'timeout': pathfinder.timed_out,
                        'recursion_limit': pathfinder.hit_recursion_limit,
                        'n_connected_groups': len(connected_groups),
                        'largest_group_id': largest_connected_group_id,
                        'largest_group_size': largest_group_size,
                        'n_edges': sum(1 for edges in connections.values() for node in edges),
                        'n_junctions': len(junctions),
                        'n_collapsed_edges': n_collapsed_edges,
                        'n_collapsed_edges_betw_junctions': n_collapsed_edges_betw_junctions,
                        'collapsed_edge_size_mean': edge_sizes.mean(),
                        'collapsed_edge_size_std': edge_sizes.std(),
                        'n_start_nodes': len(starts),
                        'paths_found': len(unpacked_paths),
                        'paths_decoded': n_decoded,
                        },
                'decodings': filtered_path_info
    }
    with open(os.path.join(outfolder, 'decodes.json'), 'w') as handle:
        json.dump(json_data, handle, indent=4)
    logger.info(f'{n_decoded} decoded out of {len(unpacked_paths)}')
    return unpacked_paths, filtered_path_info


if __name__ == '__main__':
    settings = parse_args()
    fused_groups = process_file(settings.infile, settings.outfolder,
                                  settings.n_colors, settings.decoding_scheme,
                                  settings.correct_errors, settings.shift,
                                  settings.timeout, settings.scoring_scheme,
                                  settings.output_detailed_json)
