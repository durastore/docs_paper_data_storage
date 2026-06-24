from Bio import SeqIO
from Bio.Align import PairwiseAligner
import multiprocessing
import os
import re

class PafHit:

    typedict = {'i': int, 'Z': str, 'f': float, 'A': str}
    cs_pattern = re.compile(':[0-9]+|\*[a-z][a-z]|[=\+\-][A-Za-z]+')

    def __init__(self, line):
        tabs = line.strip().split('\t')
        self.query_name = tabs[0]
        self.cs = self.get_cs(tabs[12:])


    def __repr__(self):
        return(f'PafHit({self.query_name})')


    def get_cs(self, tabs):
        for tab in tabs:
            p1, _, rest = tab.partition(':')
            if p1 == 'cs':
                _, _, data = rest.partition(':')
                return str(data)

    
    def get_barcodes(self):
        # Note: assumes that the alignment program places insertion in barcodes at the start
        current_barcode = ''
        prev_hit = ''
        barcodes = []
        for hit in re.finditer(self.cs_pattern, self.cs):
            hit = hit.group()
            if hit.startswith('*n'):
                base = hit[2]
                if not current_barcode and prev_hit.startswith('+'):
                    current_barcode = prev_hit[1:]
                current_barcode += base
            elif current_barcode:
                barcodes.append(current_barcode.upper())
                current_barcode = ''
            prev_hit = hit
        # Add any remaining barcodes
        if current_barcode:
            barcodes.append(current_barcode.upper())
        return barcodes


def find_color(color_bc, reference_colors):
    # Returns None if ambiguous, no barcode or no clear matches
    if not color_bc:
        return
    scores = {}
    for name_ref, (seq_ref, cutoff_score) in reference_colors.items():
        score = aligner.score(seq_ref, color_bc)
        if score >= cutoff_score:
            scores[name_ref] = score
    if not scores:
        return
    score_values = list(scores.values())
    max_score = max(score_values)
    if score_values.count(max_score) > 1:
        return None
    for name_ref, score in scores.items():
        if score == max_score:
            return name_ref



def setup_aligner():
    # Set scores similar to blastn
    # Score N's as 0
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 2.0
    aligner.mismatch_score = -3.0
    aligner.gap_score = 0.0
    aligner.internal_open_gap_score = -5.0
    aligner.internal_extend_gap_score = -2.0
    return aligner


def process_file(args):
    infile, bc_file, color_file, reference_colors = args
    bc_handle = open(bc_file, 'w')
    color_handle = open(color_file, 'w')

    with open(infile) as handle:
        for line in handle:
            hit = PafHit(line)
            barcodes = hit.get_barcodes()
            if len(barcodes) < 6:
                continue
            bc_handle.write(f'{hit.query_name}\t' + '\t'.join(str(v) for v in barcodes) + '\n')
            color_matches = []
            for color_bc in barcodes[:-1]:
                match = find_color(color_bc, reference_colors)
                color_matches.append(match)
            color_handle.write(f'{hit.query_name}\t' + '\t'.join(map(str, color_matches)) + '\t' + str(barcodes[-1]) + '\n')

    bc_handle.close()
    color_handle.close()


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    try:
        infolder = args[0]
        outfolder = args[1]
    except:
        print("Usage: python extract_barcodes_minimap.py infolder outfolder\nInfolder should point to a folder with all fastq files aligned to the reference with minimap2")
    color_references_file = 'color_references.fasta'

    aligner = setup_aligner()

    reference_colors = {}
    for record in SeqIO.parse(color_references_file, 'fasta'):
        cutoff_score = aligner.score(record.seq, record.seq) * 0.3
        reference_colors[record.id] = record.seq, cutoff_score

    jobs = []
    for infile in os.listdir(infolder):
        filename = os.path.splitext(infile)[0]
        bc_file = os.path.join(outfolder, filename+'_barcodes.txt')
        color_file = os.path.join(outfolder, filename+'_colors.txt')
        jobs.append((os.path.join(infolder, infile), bc_file, color_file, reference_colors))
    with multiprocessing.Pool(8) as pool:
        for _ in pool.imap_unordered(process_file, jobs):
            pass
