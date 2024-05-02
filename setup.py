import re
from setuptools import setup


VERSION = '1.6.4'


if __name__ == '__main__':
    with open("README.md", 'r') as readme:
        description = readme.read()
        # Patch the relative links to absolute URLs that will work on PyPI.
        description2 = re.sub(
            r']\(([\w/.-]+\.png)\)',
            r'](https://github.com/vivarium-collective/vivarium-core/raw/master/\1)',
            description)
        long_description = re.sub(
            r']\(([\w/.-]+)\)',
            r'](https://github.com/vivarium-collective/vivarium-core/blob/master/\1)',
            description2)

    setup(
        name='vivarium-core',
        version=VERSION,
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
        project_urls={
            'Source': 'https://github.com/vivarium-collective/vivarium-core',
            'Documentation': 'https://vivarium-core.readthedocs.io/en/latest/',
            'Changelog': 'https://github.com/vivarium-collective/vivarium-core/blob/master/CHANGELOG.md',
        },
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
        python_requires='>=3.9, <3.12',
        install_requires=[
            'matplotlib>=3.5.1',
            'networkx>=2.6.3',
            'numpy>=1.22.1',
            'Pint>=0.23',
            'pymongo>=4.0.1',
            'scipy>=1.7.3',
            'pytest>=6.2.5',
            'orjson>=3.8.0'
        ],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Topic :: Scientific/Engineering',
        ],
        keywords='vivarium multi-scale computational-biology biology simulation framework',
    )
