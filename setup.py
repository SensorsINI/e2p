"""Setup script for e2p."""

import setuptools
from setuptools import setup, find_packages

classifiers = """
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Topic :: Utilities
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Python Modules
License :: OSI Approved :: MIT License
"""

version="0.0.1"
package_name="e2p"

setup(
    name=package_name,
    version=version,
    description='Reconstructs polarization from PDAVIS camera',

    author="Tobi Delbruck, Haiyang Mei, Zuowen Wang",
    author_email="tobi@ini.uzh.ch",

    python_requires=">={}".format("3.8"),

    #  packages=find_packages(include=['v2ecore', 'v2e.*']),
    packages=find_packages(),
    url='https://github.com/SensorsINI/v2e',
    install_requires=[
        'matplotlib',
        'opencv-python == 4.5.5.64',
        'tqdm',
        'pandas',
        'h5py',
        'numpy == 1.24.2',
        'scikit-image',
        'pytorch_msssim',
        'timm',
        'engineering_notation',
        'psutil',
        'pyzmq',
        'thop',
        'IPython',
        'easygui',
        'pypref'
    ],

    scripts=['pdavis_demo.py'],

    entry_points={
        'console_scripts': ['pdavis_demo=pdavis_demo:main']
    },

    classifiers=list(filter(None, classifiers.split('\n'))),
)