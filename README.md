# Parameterized large image pre-processing with _*pychunklbl.toolkit*_
The package module _*pychunklbl.toolkit*_ provides eight parameterized functions designed to work with large image files and provide pre-processing for machine learning.

The package module is intended developers to create machine learning datasets leveraging the [openslide](https://openslide.org/) library for usage with [tensorflow 2.0](https://www.tensorflow.org/).

****
## Installation
```
pip install -i https://test.pypi.org/simple/ pychunklbl

# requires python 3.5 or later
pip3 install -r requirements.txt
```

****
### Command line examples

#### Find the patches in a wsi file and write to an image file for preview.
```
python3 -m pychunklbl.cli -m write_mask_preview_set -i DigiPath_MLTK/data/images/CMU-1-Small-Region.svs -o results
```

#### Find the patches in a wsi file and write to a directory.
```
python3 -m pychunklbl.cli -m wsi_to_patches_dir -i DigiPath_MLTK/data/images/CMU-1-Small-Region.svs -o results
```

#### Find the patches in a wsi file and write to a .tfrecords file.
```
python3 -m pychunklbl.cli -m wsi_to_patches -i DigiPath_MLTK//data/images/CMU-1-Small-Region.svs -o results
```

#### View the patch locations in a .tfrecoreds file.
```
python3 -m pychunklbl.cli -m tfrecord_2_masked_thumb -i DigiPath_MLTK/data/images/CMU-1-Small-Region.svs -r DigiPath_MLTK/data/tfrecords/CMU-1-Small-Region.tfrecords -o results
```
`data not available for the following examples`
#### Find pairs of patches with registration offset in two wsi files and write to a directory.
```
python3 -m pychunklbl.cli -m registration_to_dir -i fixed.tiff -f float.tiff -d wsi_pair_sample.csv -o results
```

#### Find pairs of patches with registration offset in two wsi files and write to a tfrecords file.
```
python3 -m pychunklbl.cli -m registration_to_dir -i  fixed.tiff -f float.tiff -d wsi_pair_sample.csv -o results
```

#### Find the patches in a wsi file defined in an annotations file with a priority file and write to a directory.
```
python3 -m pychunklbl.cli -m annotations_to_dir -i fixed.tff -p class_label_id_test.csv -a fixed.xml -o results
```

#### Find the patches in a wsi file defined in an annotations file with a priority file and write to a tfrecords file.
```
python3 -m pychunklbl.cli -m annotations_to_dir -i fixed.tff -p class_label_id_test.csv -a fixed.xml -o results
```

****
## Detailed module usage - documentation:
[package modules usage:](https://ncsa.github.io/DigiPath_MLTK/) <br>

### High level functions usage details:
Givin a WSI and a label export patches to a directory: <br> [image_file_to_patches_directory_for_image_level(run_parameters)](https://ncsa.github.io/DigiPath_MLTK/image_file_to_patches_directory_for_image_level.html) <br>

Givin a WSI and a label export patches to a TFRecords file: <br> 
[run_imfile_to_tfrecord(run_parameters)](https://ncsa.github.io/DigiPath_MLTK/image_file_to_tfrecord_and_view_tfrecord.html) <br>


