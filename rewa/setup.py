"""
REWA - Reasoning & Validation Layer for AI Agents

A geometric, local-world, constraint-satisfaction system that decides
whether language should even be trusted.
"""

from setuptools import setup, find_packages

setup(
    name="rewa",
    version="0.1.0",
    description="Reasoning & Validation Layer for AI Agents",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="REWA Team",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.20.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    package_data={
        "rewa": ["../domain_rules/*.yaml"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "rewa-experiments=rewa.experiments.runner:run_all_experiments",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
