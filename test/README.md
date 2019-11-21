# Test using Makefile.
Clone repository, cd to (test) directory. <br>
Change to the _test_ directory 
```
cd DigiPath_MLTK/test
```
View the Makefile in test directory:
```
cat Makefile
```
## _Note:_
If your _python3_ command is _python_ you must edit the _PYTHON_NAME_ variable at the top of the Makefile
****
# Unit Tests:
```
make unit_tests
```
****
# Ingegration Tests

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
