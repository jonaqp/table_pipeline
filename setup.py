from pathlib import Path

from setuptools import setup, find_packages

root = Path(__file__).parent
LICENSE = 'MIT'
NAME = 'table_pipeline'
with open(root / 'README.md') as f:
    LONG_DESCRIPTION = f.read()

with open(root / 'requirements.txt') as f:
    INSTALL_REQUIRED = f.read().split("\n")

setup(
    name=NAME,
    version="0.0.0",
    packages=find_packages(),
    license=LICENSE,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    install_requires=INSTALL_REQUIRED,
    tests_require=["pytest"]
)
