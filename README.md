# Intonational Unit TextGrid Generator

This project provides a Python script in a Jupyter Notebook to process a CSV file containing predicted intonational units (IUs) and generate a TextGrid file, which is commonly used in phonetics and linguistics to represent time-aligned annotations of audio data.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)

## Features

- Reads a CSV file with predicted intonational units.
- Generates a TextGrid file with intervals representing the start and end times of each intonational unit along with associated text.
- Handles gaps between intervals and creates empty intervals when necessary.
- Automatically creates a directory for output files if it doesn't exist.

## Requirements

- Python 3.x
- Standard library modules: `csv`, `os`

## Usage

1. **Prepare your CSV file:** Ensure your CSV file (`spice_segmentation_predictedIUs.csv`) is formatted correctly with the following columns:
   - Column 1: Index (not used)
   - Column 2: Filename
   - Column 3: Start time (xmin)
   - Column 4: End time (xmax)
   - Column 5: Text (word)
   - Column 6: IU start prediction (TRUE/FALSE)

2. **Run the notebook:**
   - Open the Jupyter Notebook (`iu_csv-to-textgrid.ipynb`) in Jupyter Notebook or JupyterLab.
   - Execute the cells in the notebook to process the CSV file and generate the TextGrid files.

3. **Output:** The script will generate a TextGrid file for each unique filename in the CSV file, saved in a folder named `textgrids`. The final output file will be named `output.TextGrid`.
