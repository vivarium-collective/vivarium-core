import os
import glob
import setuptools
from distutils.core import setup

with open("README.md", 'r') as readme:
    long_description = readme.read()

setup(
    name='vivarium-core',
    version='0.0.14',
    packages=[
        'vivarium',
        'vivarium.core',
        'vivarium.processes',
        'vivarium.library'
    ],
    author='Eran Agmon, Ryan Spangler',
    author_email='eagmon@stanford.edu, ryan.spangler@gmail.com',
    url='https://github.com/vivarium-collective/vivarium-core',
    license='MIT',
    entry_points={
        'console_scripts': []},
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_data={},
    include_package_data=True,
    install_requires=[
        'confluent-kafka',
        'matplotlib',
        'networkx',
        'numpy',
        'Pint',
        'pymongo',
        'pytest',
        'scipy'])
