from setuptools import find_packages, setup

setup(
    name="mesa_frames",
    packages=find_packages(where="mesa_frames"),
    package_dir={"": "mesa_frames"},
    version="0.1.0-alpha1",
    description="An extension to the Mesa framework which uses Pandas DataFrames for enhanced performance",
    author="Adam Amer",
    license="MIT License",
)
