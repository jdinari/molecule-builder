from setuptools import setup, find_packages
setup(
    name="molbuilder",
    version="1.1.0",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "rdkit"],
    entry_points={
        "console_scripts": [
            "molbuilder=molbuilder.cli:main",
            "molbuilder-ext=molbuilder.cli_extended:main_extended",
        ]
    },
)
