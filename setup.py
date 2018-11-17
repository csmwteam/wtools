"""wtools: a Python package for W-Team research needs
"""

import setuptools

__version__ = '0.0.4'

with open("README.rst", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="wtools",
    version=__version__,
    author="CSM W-Team",
    author_email="info@pvgeo.org",
    description="W-Team reasearch tools",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/csmwteam",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.15',
        'scipy>=1.1',
        'pandas>=0.23',
        'pillow>=5.2.0',
        'xmltodict>=0.11.0',
        'vectormath>=0.2.0',
        'properties>=0.5.2',
    ],
    classifiers=(
        "Programming Language :: Python",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Visualization',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
    ),
)
