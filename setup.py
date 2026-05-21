from setuptools import setup, find_packages
setup(
    name="molbuilder",
    version="2.0.0",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "rdkit"],
    entry_points={"console_scripts": ["molbuilder=molbuilder.cli:main"]},
)
