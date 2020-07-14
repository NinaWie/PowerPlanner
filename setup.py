"""Package installer."""
import os
from setuptools import setup
from setuptools import find_packages

LONG_DESCRIPTION = ''
if os.path.exists('README.md'):
    with open('README.md') as fp:
        LONG_DESCRIPTION = fp.read()

scripts = []

setup(
    name='power_planner',
    version='0.0.1',
    description='Optimizing power infrastructure layout',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author='Nina Wiedemann',
    author_email=('wnina@ethz.ch'),
    url='https://gitlab.ethz.ch/wnina/power-planner',
    license='MIT',
    install_requires=[
        'numpy', 'Kivy', 'numba', 'Pillow', 'rasterio', 'pandas', 'networkx',
        'matplotlib', 'scipy'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=find_packages('.'),
    scripts=scripts
)
