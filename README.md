# DigiPath_MLTK package named *pychunklbl*
The _*py.toolkit*_ module doda
****
## Installation
```
pip install -i https://test.pypi.org/simple/ pychunklbl==0.0.5

# requires python 3.5 or later
pip3 install requirements.txt
```
****
## Package usage:
The toolkit module in the pychunklbl package takes .yml files as input to the  
```
python ../DigiPath_MLTK/src/python/digipath_mltk.py -run_directory . -run_file image_files_to_tfrecord.yml
```

## The "method" parameter in the yaml files:
Four methods are currently implemented in DigiPath_MLTK/src/python/digipath_mltk.py. <br>
```
SELECT = {"image_2_tfrecord": image_2_tfrecord,
          "tfrecord_2_masked_thumb": tfrecord_2_masked_thumb,
          "wsi_2_patches_dir": wsi_2_patches_dir,
          "write_mask_preview_set": write_mask_preview}
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
