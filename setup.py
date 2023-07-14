import os
from pathlib import Path
from setuptools import setup, find_packages

parent_dir = Path(__file__).resolve().parent
presets_dir = os.path.abspath(os.path.join(os.path.pardir, "presets"))

setup(
    name="prettymaps",
    version="v1.0.0",
    description="A simple python library to draw pretty maps from OpenStreetMap data",
    long_description=parent_dir.joinpath("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/marceloprates/prettymaps",
    author="Marcelo Prates",
    author_email="marceloorp@gmail.com",
    license="MIT License",
    python_requires='>=3.10',
    packages=find_packages(exclude=("assets", "notebooks", "prints", "script")),
    install_requires=["osmnx==1.2.2", "Shapely>=1.8,<2.0", "pyyaml==6.0.0"],
    extras_require={"test": ["pytest", "pytest-regressions[image]", "pre-commit"]},
    classifiers=[
        "Intended Audience :: Science/Research",
    ],
    package_dir={"prettymaps": "prettymaps"},
    package_data={"prettymaps": ["presets/*.json"]},
)
