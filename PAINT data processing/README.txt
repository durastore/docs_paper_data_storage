DOCS-PAINT: DNA-PAINT Data Processing Pipeline

Overview
DOCS-PAINT is a Python-based processing pipeline developed for the analysis of DNA-PAINT super-resolution data, specifically 
designed to support DNA Origami for Combinatorial data Storage (DOCS). This suite provides an intuitive graphical user interface (GUI) 
for cropping, segmenting, extracting, and statistically analyzing localization data to accurately decode DNA origami data carrier nanostructures.

Requirements
The pipeline is built in Python and requires the following dependencies:

numpy, pandas, scipy (Data manipulation, signal processing, and statistical analysis)

h5py (Handling HDF5 localization datasets)

opencv-python (cv2), shapely (Image processing, structural detection, and contour analysis)

matplotlib, seaborn (Data visualization)

PySimpleGUI (Graphical User Interface)

PyYAML (Metadata handling)

Pipeline Modules
The pipeline is executed via a central GUI, which provides access to four distinct processing workflows. 
Default values are provided for each adjustable parameter used during the respective processing steps.

1. Data Cropping (DOCS_PAINT_data_crop.py)
Input files: 
Undrifted, aligned Exchange PAINT channel localization files (.hdf5) and its associated metadata (.yaml) 
Output files:
Cropped, undrifted, aligned Exchange PAINT channel localization files (.hdf5) and its associated metadata (.yaml) 
Crop coordinate .csv file
Description and notes:
Removes non-overlapping areas from respective Exchange PAINT channel data sets.
Users can define spatial boundaries to crop the field of view, alternatively the code will pick the maximum possible area.

2. Data Segmentation (DOCS_PAINT_data_segmentation.py)
Input files:
Cropped, undrifted, aligned Exchange PAINT channel localization files (.hdf5) and its associated metadata (.yaml) 
Crop coordinate .csv file

Output files:
Segmentation parameters file (.csv)
Segmented localization file (.h5)

Description:
Identifies individual data carriers in the Exchange PAINT data based on user provided size thresholds and groups localizations from 
the different PAINT channels into ROIs centered around individual data carrier positions.

3. Site Localization Extraction (DOCS_PAINT_data_site_loc_extract.py)
Input files: 
Segmentation parameters file (.csv)
Segmented localization file (.h5)

Output files:
Processing parameters file (.csv)
Channel information file (.csv)
Processed localization file (.h5)
Exported, segmented data carrier images(.png)

Description:
Filters and groups localizations from individual DCs into positions on DCs. 

4. Result Visualization & Statistical Analysis (DOCS_PAINT_data_result_vis.py)
Input files: 
Channel information file (.csv)
Processed localization file (.h5)

Output files:
DC identity probability file (.csv)
DC code count file (.file)

Description:
Calculates the base probabilities required for decoding based on the data and determines DC identity based on a Bayessian probabilistic approach 
and counts the occurance of detected unique DC codes in the data.


Usage
To launch the pipeline, run the main script from your terminal:

Bash
python DOCS_PAINT_main.py

This will open the primary menu, allowing you to sequentially route your data through the cropping, segmentation, extraction, and processing workflows.