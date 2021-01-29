"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
from setuptools import setup
# codes.open is for support of python 2.x
from codecs import open
from os import path

import re

here = path.abspath(path.dirname(__file__))
re_ver = re.compile(r"__version__\s+=\s+'(.*)'")
with open(path.join(here, 'linearlr.py'), encoding='utf-8') as f:
    version = re_ver.search(f.read()).group(1)

setup(
    name='linearlr',
    version=version,
    description='Tuning-free learning rate schedule for training deep neural networks',
    long_description='See project page: https://github.com/mtli/linearlr',
    url='https://github.com/mtli/linearlr',
    author='Mengtian (Martin) Li',
    author_email='martinli.work@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    keywords='deep learning linear rate schedule',
    py_modules=['linearlr'],
    include_package_data = True,
)
