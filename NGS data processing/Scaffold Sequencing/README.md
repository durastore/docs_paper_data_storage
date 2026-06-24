# ONT scaffold read processing scripts

**Code used for processing scaffold ONT NGS data presented in the DOCS paper.**

These scripts were used in the analysis of the scaffold NGS sequencing data. The reads are first aligned with minimap2, then barcodes and color sequences are extracted and analyzed.

## Outline on how to repeat the data analysis
### Align all reads with minimap2
Output format should be .paf format. In this example, each fastq file is aligned and handled independently, on a Linux system that has minimap2 installed.
Example code:

    for INFILE in fastq_pass/*; do
        CLEAN_NAME=$(basename $INFILE)
        minimap2 -x map-ont -o minimap2_results/$CLEAN_NAME.paf reference_insert.fa $INFILE 
    done

### Extract all barcodes and color sequences
Code:

    python extract_barcodes_minimap.py infolder outfolder

Where the infolder is the minimap2_results folder made in the previous step, and the outfolder is a new folder (e.g., barcodes_extracted).

### Compile information across all files
Code:

    python count_barcodes.py infolder outfolder tag

Where the infolder is the folder created in the previous step (barcodes_extracted), the outfolder is a new folder (e.g., barcodes_compiled_info) and tag is a name you can give to your run.

### Analyze all barcodes and color sequences
Code:

    python count_barcodes.py base_folder outfolder tag
    
Where the base_folder is the folder created in the previous step ( a name you can give to your run), the outfolder is a new folder (e.g., barcode_analysis) and tag is the name given in the previous call.

### Optional: Analyze reads
Code:

    python extract_barcodes_minimap.py infolder outfolder


Infolder should point to a folder with fastq files.

## Dependencies
Scripts are run in python3 with the following packages installed:

* biopython
* matplotlib
* numpy
* pandas

In addition, you need to install minimap2 (https://github.com/lh3/minimap2).
