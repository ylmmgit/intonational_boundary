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

## CSV2Praat Auto Tool Interface

You can also use the [CSV2Praat Auto Tool](https://huggingface.co/spaces/zyshan-ds/CSV2Praat_Auto_Tool) interface for a more user-friendly experience.

### How to Use the Interface

1. **Upload a properly formatted CSV file:** Click to upload your CSV file or drag and drop it into the upload area.
2. **Enter Tier Name:** Provide a tier name according to your preference or as deemed appropriate for the data.
3. **Submit:** Click the submit button to process the file.
4. **Download:** After processing, you will receive a .zip file containing the generated Praat TextGrid files.

### Expected CSV Format

Please ensure that the CSV file adheres to the following format:

- The first row must contain headers: `, file_name, xmin, xmax, text, is_unit_start_pred`.
- Each subsequent row should contain the following columns for every word or segment in the audio file:
  - `file_name`: Identifier for the audio file, used to group intervals.
  - `xmin`: Start time of the segment (in seconds).
  - `xmax`: End time of the segment (in seconds).
  - `text`: The actual spoken word or phrase.
  - `is_unit_start_pred`: Marks the beginning of a new unit (TRUE/FALSE).

**Note:** 
- The interface currently only accepts CSVs with an index.
- **The last column name should be `is_unit_start_pred`, as this interface is targeted to any unit, not just intonational units.**

### Example CSV:

| file_name | xmin   | xmax   | text  | is_unit_start_pred |
|-----------|--------|--------|-------|---------------------|
| 0         | example1 | 20.42  | 20.74 | mhmm  | TRUE                |
| 1         | example1 | 20.74  | 20.81 | hello | TRUE                |
| 2         | example1 | 20.81  | 20.92 | world | FALSE               |
