import unittest
from unittest import TestCase

import sys
sys.path.insert(0, '../src/python/')
from digipath_toolkit import dict_to_patch_name, patch_name_to_dict

class Test_patch_name_encoding(TestCase):
    def test_dict_to_patch_name(self):
        print('Here Here')

if __name__ == '__main__':
    unittest.main()

