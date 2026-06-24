# Encoding, decoding, and simulating data storage on the DOCS platform

These scripts and wrappers were used in encoding texts, simulating structures based on the encoding, generating graphs, extracting the message from the graph and decoding it again.

It also contains scripts to decode the experimental data.

The huffman implementation was adapted from the implementation here: https://github.com/rdbliss/huffman

## Overview of files

* **`huffman.py and tree.py`** - Huffman implementation
* **`simulate_structures.py`** - Encodes texts and simulates structures
* **`decode_structures.py`** - Generates directed graphs from structure observation data and attempts to decode them
* **`experimental_wrapper.py`** - Analyzes the experimental data
* **`visualize_networks.py`** - Helper script to generate Cytoscape visualizations
* **`category_error_rates.csv`** - The error rates used for simulation
* **`structure_counts_no_errors.csv`** - The structure observation counts used for simulation
* **`letter_freq.txt`** - The letter frequencies used for Huffman encoding
* **`wrapper_simulations`** - Convenience wrapper to run simulations and decode them
* **`texts`** - Reference texts used in simulation
* **`reference_structures`** - Reference structures for experimental data analysis
* **`make_plots`** - Scripts to generate plots made for the manuscript

## Encoding data and simulating structures

The main entrypoint to encode data is to run the simulate_structures.py script. This will encode a given text, split it into overlapping k-mers, generate read counts for those k-mers, and optionally generate error derivatives with a read count of the k-mers.

    usage: simulate_structures.py [-h] [--n-colors N_COLORS]
                                  [--kmer-length KMER_LENGTH]
                                  [--n-replicates N_REPLICATES] [-t TEXT]
                                  [-tf TEXT_FILE] --counts-file COUNTS_FILE
                                  --error-file ERROR_FILE [--freq-file FREQ_FILE]
                                  -o OUTFOLDER [--disable-uncorrected-errors]
                                  [--disable-huffman] [--shift SHIFT]

    options:
      -h, --help            show this help message and exit
      --n-colors N_COLORS
      --kmer-length KMER_LENGTH
      --n-replicates N_REPLICATES
      -t, --text TEXT
      -tf, --text-file TEXT_FILE
      --counts-file COUNTS_FILE
      --error-file ERROR_FILE
      --freq-file FREQ_FILE
      -o, --outfolder OUTFOLDER
      --disable-uncorrected-errors
      --disable-huffman
      --shift SHIFT


You can give a text either with the -t flag, directly into the terminal, or refer to a text file with the -tf flag (such as those given in the texts folder).

The --kmer-length, --n-colors and --shift flags determine how the data is split among different structures. Take note that --n-colors indicates the number of base colors. It is assumed that the structures have two colors per site, meaning that ncr(n, 2) color combinations are used in encoding.

The --error-file and --counts-file are used for the error rates and read count distribution. Examples used in the manuscript are given in this folder.

--freq-file refers to the letter frequency file used in the huffman encoding, obtained from https://norvig.com/mayzner.html

## Decoding data

The main script to decode structures is the decode_structures.py script.

    usage: decode_structures.py [-h] [--n-colors N_COLORS] -i INFILE -o OUTFOLDER [--correct-errors]
                                -d {huffman,zlib} [--read-cutoff READ_CUTOFF] [--shift SHIFT] [--timeout TIMEOUT]
                                [--scoring-scheme {simple,weighted}] [--output-detailed-json]

    options:
      -h, --help            show this help message and exit
      --n-colors N_COLORS
      -i, --infile INFILE
      -o, --outfolder OUTFOLDER
      --correct-errors
      -d, --decoding-scheme {huffman,zlib}
      --read-cutoff READ_CUTOFF
      --shift SHIFT
      --timeout TIMEOUT
      --scoring-scheme {simple,weighted}
      --output-detailed-json


The --infile should point to a structure abundance file. This is a .csv file with two columns: one for the colors observed (displayed as a case-sensitive alphanumeric string), and an observation count (integer).

The --n-colors and --shift flags determine how the data is split among different structures. Take note that --n-colors indicates the number of base colors. It is assumed that the structures have two colors per site, meaning that ncr(n, 2) color combinations are used in encoding.

Turn on algorithmic error correction with --correct-errors.

The script will take the allotted time given by the --timeout parameter (in seconds) twice, once to find viable paths, once to decode them.

The --scoring-scheme set to simple will disregard edge weights in the directed graphs. Set to --weighted, the weight of the edge (determined by the structure observation count) will be used. In the manuscript, --scoring-scheme was set to weighted.


## Make plots

This folder contains a collection of scripts used to generate the plots given in the manuscript. 

* plot_collapsed_vs_success.py - Given a folder with subfolders, where each subfolder is generated by the wrapper_simulations script, generate plots on their success rates
* plot_random_lib.py - Plot the correlation of observation counts between the barcodes of the random library as sequenced or as observed by PAINT
* plot_sim_parameters.py - Plot the error rates and observation counts used in the simulations
* plot_sizes.py - Plot the sizes of the texts used for simulations

## Experimental data analysis

The experimental data wrapper analyzes structure abundance files. These are a .csv file with two columns: one for the colors observed (displayed as a case-sensitive alphanumeric string), and an observation count (integer).

By default, it will recursively search for files starting with "Joined_bc_list" as these were generated by the PAINT processing pipeline.

It also requires a reference folder, which provides reference structure information for each folder analyzed.

By default, it will generate cytoscape graph images through py4cytoscape. It requires one to have Cytoscape open before starting the script.


## Dependencies
The scripts were run with python3.13 with the following packages installed. See also the env.yml file.

* scipy
* gmpy2
* numpy
* pandas
* configargparse
* matplotlib

Optionally, if you want to rerun the experimental wrapper with automated cytoscape visualization, you need to install Cytoscape (https://cytoscape.org/) and py4cytoscape (https://github.com/cytoscape/py4cytoscape)

