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
Example *.yml* files are in the DigiPath_MLTK/data/run_files directory and may be used as templates to run with your data.

Each parameters (*.yml*) file is a template for running one of the eight methods. 

When edited with valid data they run from the command line using the example main function in this repository: <br>
`src/python/digipath_tk_run.py`

### Command line example:
```
python3 ../DigiPath_MLTK/src/digipath_tk_run.py -run_directory . -run_file annotations_to_dir.yml
```

****
## Installation Test Notes:
The *DigiPath_MLTK/test* directory contains the Makefile that may be used for either development or post-installation testing.

The README.md details how to run the integration test suite after installing the package as a module.

By changing the first line of the Makefile `SCRIPT = ../src/python/digipath_tk_run.py` to `SCRIPT = ../src/python/digipath_mltk.py` the code in this (cloned) repository can be run for development purposes.

Note: the run_parameters from the *.yml* files are just regular python type dict when passed to the toolkit module.

****
## Documentation:
[Developer Documentation Notebooks: functions usage](https://ncsa.github.io/DigiPath_MLTK/) <br>

## High Level functions Reference:
Givin a WSI and a label export patches to a directory: [image_file_to_patches_directory_for_image_level()](https://ncsa.github.io/DigiPath_MLTK/image_file_to_patches_directory_for_image_level.html) <br>

Givin a WSI and a label export patches to a TFRecords file: [run_imfile_to_tfrecord()](https://ncsa.github.io/DigiPath_MLTK/image_file_to_tfrecord_and_view_tfrecord.html) <br>


