# Test using Makefile.
Clone repository, cd to (test) directory. <br>
_Note: there are two make targets for each test, python3 or python_ <br>
View in test directory:
```
cat Makefile
```
****
# Unit Tests:
```
make unit_tests
```
## Or if python is python3:
```
make unit_tests_py3
```

****
# Test from _DigiPath_MLTK/test_ with yaml file in run_dir:
## setup command:
```
make env_setup
```
- creates ../../run_dir/results 
- moves yaml files into ../../run_dir/

## preview mask images & patch locations:
Edit _write_mask_preview.yml_ that was copied with _make env_setup_
run with make in _DigiPath_MLTK/test_ directory:
```
make test_write_mask_preview
```
`mask preview set saved:` <br>
`	../../run_dir/CMU-1-Small-Regionmarked_thumb.jpg` <br>
`	../../run_dir/CMU-1-Small-Regionmask.jpg` <br>
`	../../run_dir/CMU-1-Small-Regionpatch_locations.tsv` <br>
(successfull run will print the file names as shown)

## write wsi image file with label to folder:
Edit _wsi_file_to_patches_dir.yml_ that was copied with _make env_setup_
run with make from _DigiPath_MLTK/test_ directory:
```
make test_wsi_2_patches_dir
```
`77 images found` <br>
will report the number of images found with the yaml file parameters

## still working - depreciated methods (refactoring):
****
Test that the .svs file in ../data/images/ is converted to a TFRecord:
```
time make test_im_2_tfr
```
Should produce output something like:
```
TFRecord file written:
../../run_dir/results/CMU-1-Small-Region.tfrecords

real	0m3.255s
user	0m3.512s
sys	0m0.467s
```

****
Test TFRecord file produces proper thumbnail with location of patch images (after running all the above)
- First Edit the *../run_dir/tfrecord_to_masked_thumb.yml* file to set the path of the file produced above.

- Change:
```
tfrecord_filename: ../data/tfrecords/CMU-1-Small-Region.tfrecords
``` 
to
```
tfrecord_filename: ../../run_dir/results/CMU-1-Small-Region.tfrecords
```

Then run the code with make
```
time make test_tfr_2_mthumb
````
Should produce output like:
```
Thumbnail Image File Written:
../../run_dir/results/test_image.jpg

real	0m3.287s
user	0m3.557s
sys	0m0.457s
```
View the output file to confirm

