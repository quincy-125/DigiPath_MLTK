# Test using Makefile.
Clone repository, cd to (test) directory.

****
# Unit Tests:
```
make unit_tests
```

****
# Run with parameters set in yaml file:
Setup a test environment at the same level as the cloned repo.
(Edit the yaml file copied to the run directory).
```
make env_setup
```
- creates ../../run_dir/results 
- moves yaml files into ../../run_dir/

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

