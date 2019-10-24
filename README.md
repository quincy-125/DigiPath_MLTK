# DigiPath_MLTK
### _Minimal operating instructions_ <br>
_Note: code works as described below, the new module will use the same command line calls._ <br>
****
## Requirements:
```
# python 3.5 or later
pip3 install requirements.txt
```
****
## Usage examples
1 Clone or unzip this repository to your computer. <br>
2 Create a run-directory and a results directory. <br>
3 Copy the yaml files to the run-directory. <br>
4 Edit the yaml files to point to you run and results directories. <br>
5 Edit any other yaml file run parameters. <br>
6 Run the code with the edited yaml file in the run directory.  <br>
```
python ../DigiPath_MLTK/src/python/digipath_mltk.py -run_directory . -run_file image_files_to_tfrecord.yml
```

## The "method" parameter in the yaml files:
Two methods are currently implemented in DigiPath_MLTK/src/python/digipath_mltk.py. <br>
```
SELECT = {"image_2_tfrecord": image_2_tfrecord,
          "tfrecord_2_masked_thumb": tfrecord_2_masked_thumb}
```
Setting the "method" parameter sends the run parameters to the execution function in <br>
  _DigiPath_MLTK/src/python/openslide_2_tfrecord.py_ <br>

## Importing _digipath_toolkit.py_ module:
Set the __python__ sys path to include the file location before importing a file as a python module. <br>
```
import sys
sys.path.insert(0, 'relative path from your code')
from digipath_toolkit import *
```
_note that PYTHONPATH env variable and os.path are virtually unknown to python import_
