#!/usr/bin/env python3
"""
LlamaFarms setup script.
"""

import os

from setuptools import find_packages, setup

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="llamafarms",
    version="0.1.0",
    author="Nik Jois" "Nik Jois" "Nik Jois" "Nik Jois" "Nik Jois",
    author_email="nikjois@llamasearch.ai"
    "nikjois@llamasearch.ai"
    "nikjois@llamasearch.ai"
    "nikjois@llamasearch.ai"
    "nikjois@llamasearch.ai",
    description="Advanced precision agriculture platform with MLX acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llamafarms",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/llamafarms/issues",
        "Documentation": "https://github.com/yourusername/llamafarms/docs",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "llamafarms=llamafarms.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "llamafarms": ["data/*", "models/*"],
    },
)
