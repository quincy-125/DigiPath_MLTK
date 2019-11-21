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
****
# Ingegration Tests
### Run test from _DigiPath_MLTK/test_ with yaml files copied int run_dir and edited if desired:
## setup command:
```
make env_setup
```
- creates ../../run_dir/results 
- moves yaml files into ../../run_dir/
## test all command 
```
make integration_test
```
examine command line output and jpg files for correctness
