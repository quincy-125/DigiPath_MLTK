## Makefile tests: for development debug or maintainance
Clone repository, cd to (test) directory. <br>
Change to the _test_ directory 
#### _Note:_
If your __*python3*__ command is __*python*__ you must edit the __*PYTHON_NAME*__ variable at the top of the Makefile
****
### Unit Tests:
```
make unit_tests
```
****
## Integration and package installation tests:
### run setup with make to create test directory and copy run files:
```
make env_setup
```
- creates ../../run_dir/results 
- moves yaml files into ../../run_dir/

### Ingegration Tests
```
make integration_test
```

### Instalation Tests
```
make installation_test
```

`examine command line output and jpg files for correctness`
