# Parameterized large image pre-processing with _*pychunklbl.toolkit*_
The package module _*pychunklbl.toolkit*_ provides eight parameterized functions designed to work with large image files and provide pre-processing for machine learning with tensorflow 2.0.

The package module is intended developers to create machine learning datasets leveraging the [openslide](https://openslide.org/) library for usage with [tensorflow 2.0](https://www.tensorflow.org/).

****
## Installation
```
pip install -i https://test.pypi.org/simple/ pychunklbl==0.0.5

# requires python 3.5 or later
pip3 install -r requirements.txt
```

****
## Package usage:
Example *.yml* files are in the *DigiPath_MLTK/data/run_files* directory and may be used as templates to run with your data.

Each parameters (*.yml*) file is a template for running one of the eight methods. 

When edited with valid data they run from the command line using the example main function in this repository: <br>
`src/python/digipath_tk_run.py`

### Command line examples
Find the patches in a wsi file and write to a directory
```
python3 -m pychunklbl.cli -m wsi_to_patches -w data/CMU-1-Small-Region.svs -o results_cli
```

****
## Documentation:
[package modules usage:](https://ncsa.github.io/DigiPath_MLTK/) <br>

## High Level functions Reference:
Givin a WSI and a label export patches to a directory: <br> [image_file_to_patches_directory_for_image_level(run_parameters)](https://ncsa.github.io/DigiPath_MLTK/image_file_to_patches_directory_for_image_level.html) <br>

Givin a WSI and a label export patches to a TFRecords file: <br> 
[run_imfile_to_tfrecord(run_parameters)](https://ncsa.github.io/DigiPath_MLTK/image_file_to_tfrecord_and_view_tfrecord.html) <br>


