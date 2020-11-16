import setuptools
from distutils.core import setup

_ = setuptools  # don't warn about this unused import; it might have side effects

with open("README.md", 'r') as readme:
    long_description = readme.read()

setup(
    name='vivarium-core',
    version='0.0.38',
    packages=[
        'vivarium',
        'vivarium.core',
        'vivarium.processes',
        'vivarium.composites',
        'vivarium.experiments',
        'vivarium.library',
        'vivarium.plots'
    ],
    author='Eran Agmon, Ryan Spangler',
    author_email='eagmon@stanford.edu, ryan.spangler@gmail.com',
    url='https://github.com/vivarium-collective/vivarium-core',
    license='MIT',
    entry_points={
        'console_scripts': []},
    description=(
        'Engine for composing and simulating computational biology '
        'models with the Vivarium interface.'
    ),
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
        'scipy',
    ])
