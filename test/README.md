# Test using Makefile.
Clone the repository, cd to this (test) directory.
Run make to setup a test environment at the same level as the cloned repo.
```
make env_setup
```
- creates ../../run_dir/results 
- move yaml files into ../../run_dir

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
