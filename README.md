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
python3 -m pychunklbl.cli -m write_mask_preview_set -w data/images/CMU-1-Small-Region.svs -o ../run_dir/results_cli
```

#### Find the patches in a wsi file and write to a directory.
```
python3 -m pychunklbl.cli -m wsi_to_patches_dir -w data/images/CMU-1-Small-Region.svs -o ../run_dir/results_cli
```

#### Find the patches in a wsi file and write to a .tfrecords file.
```
python3 -m pychunklbl.cli -m wsi_to_patches -w data/images/CMU-1-Small-Region.svs -o ../run_dir/results_cli
```

#### View the patch locations in a .tfrecoreds file.
```
python3 -m pychunklbl.cli -m tfrecord_2_masked_thumb -w data/images/CMU-1-Small-Region.svs -r ../data/tfrecords/CMU-1-Small-Region.tfrecords -o ../run_dir/results_cli
```

#### Find pairs of patches with registration offset in two wsi files and write to a directory.
```
python3 -m pychunklbl.cli -m registration_to_dir -w 54742d6c5d704efa8f0814456453573a.tiff -f e39a8d60a56844d695e9579bce8f0335.tiff -d wsi_pair_sample.csv -o ../run_dir/results_cli
```

#### Find pairs of patches with registration offset in two wsi files and write to a tfrecords file.
```
python3 -m pychunklbl.cli -m registration_to_dir -w 54742d6c5d704efa8f0814456453573a.tiff -f e39a8d60a56844d695e9579bce8f0335.tiff -d wsi_pair_sample.csv -o ../run_dir/results_cli
```

#### Find the patches in a wsi file defined in an annotations file with a priority file and write to a directory.
```
python3 -m pychunklbl.cli -m annotations_to_dir -w e39a8d60a56844d695e9579bce8f0335.tff -p class_label_id_test.csv -a e39a8d60a56844d695e9579bce8f0335.xml -o ../run_dir/results_cli
```

#### Find the patches in a wsi file defined in an annotations file with a priority file and write to a tfrecords file.
```
python3 -m pychunklbl.cli -m annotations_to_dir -w e39a8d60a56844d695e9579bce8f0335.tff -p class_label_id_test.csv -a e39a8d60a56844d695e9579bce8f0335.xml -o ../run_dir/results_cli
```

****
## Detailed module usage - documentation:
[package modules usage:](https://ncsa.github.io/DigiPath_MLTK/) <br>

### High level functions usage details:
Givin a WSI and a label export patches to a directory: <br> [image_file_to_patches_directory_for_image_level(run_parameters)](https://ncsa.github.io/DigiPath_MLTK/image_file_to_patches_directory_for_image_level.html) <br>

Givin a WSI and a label export patches to a TFRecords file: <br> 
[run_imfile_to_tfrecord(run_parameters)](https://ncsa.github.io/DigiPath_MLTK/image_file_to_tfrecord_and_view_tfrecord.html) <br>


