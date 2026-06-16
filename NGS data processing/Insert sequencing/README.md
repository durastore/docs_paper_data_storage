# ONT Read Processing and Consensus Pipeline

**Code used for processing insert NGS data presented in the DOCS paper.**

This pipeline extracts, orients, and trims ONT reads containing the starting and ending segments of a reference sequence (`ONT_extract`), and filters them based on user-specified length ranges (`ONT_filter`). It then leverages either CLUSTAL or MUSCLE (installed separately) to create a multiple sequence alignment (`ONT_MSA`), which is finally used to generate a polished consensus sequence (`ONT_cons`).

## Overview of Modules

* **`ONT_ref_seq.py`**: Saves a user-provided reference sequence to a FASTA file.
* **`ONT_extract.py`**: Extracts and orients reads based on the reference sequence flanking regions.
* **`ONT_filter.py`**: Provides an interactive histogram to filter reads by length.
* **`ONT_MSA.py`**: Executes multiple sequence alignment.
* **`ONT_cons.py`**: Generates the final consensus sequence.
* **`ONT_main.py`**: The main Graphical User Interface (GUI) to launch all modules.

## Dependencies

The scripts are written in Python 3 and require the following packages:
* `PySimpleGUI`
* `biopython`
* `pandas`
* `tables`
* `matplotlib`
* `scipy`
* `numpy`
* `tqdm`

**External Software Requirements:** The `ONT_MSA` module requires local installations of either ClustalW2 or MUSCLE. 
*Note: The current executable paths are set for Windows environments (`C:\ClustalW2\clustalw2.exe` and `C:\MUSCLE\muscle.exe`). Please update these paths in `ONT_MSA.py` if your installation locations differ.*

## Usage

To launch the pipeline, run the main script from your terminal:
`python ONT_main.py`
