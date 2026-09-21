from setuptools import setup, find_packages

setup(
    name="d-caf",
    version="0.1.0",
    description="Dynamic Cluster Assembly Framework (D-CAF)",
    author="Juan Farias",
    packages=find_packages(include=["dcaf","dcaf.*"]),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pyyaml"
    ],
    entry_points={
        "console_scripts": [
            "dcaf-plummer=dcaf.cli:parametric_plummer",
        ],
    },
    scripts=["scripts/dcaf_analysis"],
    python_requires=">=3.11",
)
