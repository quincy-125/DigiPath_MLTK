"""
# Create dist files:
pip3 install --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel
"""
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as fh:
    readme_text = fh.read()

setup(name='pypatches',
	version='0.0.1',
	long_description=readme_text,
	long_description_content_type='text/markdown',
	author='Mayo-NCSA DigiPath_MLTK development team',
	url='https://https://github.com/ncsa/DigiPath_MLTK',
	classifiers=['License :: OSI Approved :: MIT License', 
	'Programming Language :: Python :: 3.5', 
	'Programming Language :: Python :: 3.6',
	'Programming Language :: Python :: 3.7',
	'Programming Language :: Python :: 3.8',
	"Operating System :: OS Independent"],
	python_requires='>=3.5',
	package_dir={'': 'pypatches'},
	packages=find_packages(where='pypatches'),
	project_urls={'Source': 'https://github.com/ncsa/DigiPath_MLTK'})

"""
# Upload to test pypi:
python3 -m pip install --user --upgrade twine

# with __token__ (the one from your test pypi account)
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

"""


