# Parameterized large image pre-processing with _*pychunklbl.toolkit*_
The package module _*pychunklbl.toolkit*_ provides eight parameterized functions designed to work with large image files and provide pre-processing for machine learning with tensorflow 2.0.

****
## Installation
```
pip install -i https://test.pypi.org/simple/ pychunklbl==0.0.5

# requires python 3.5 or later
pip3 install requirements.txt
```

****
## Package usage:
Example *.yml* files are in the DigiPath_MLTK/data/run_files directory and may be used as templates to run with your data.

Each *.yml* parameters file is a template for running one of the eight methods and (when edited with valid data files) may run from the command line with the example multi-main function in this repository: <br>
`src/python/digipath_tk_run.py`

### Call with the example main file:
```
python3 ../DigiPath_MLTK/src/digipath_tk_run.py -run_directory . -run_file annotations_to_dir.yml
```
