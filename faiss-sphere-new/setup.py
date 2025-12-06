from setuptools import setup, find_packages

setup(
    name="faiss-sphere",
    version="0.1.0",
    description="FAISS-Sphere: Exploiting K=1 Spherical Geometry for Vector Search",
    author="FAISS-Sphere Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "faiss-cpu>=1.7.3",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
        "wikipedia": [
            "datasets>=2.0.0",
            "transformers>=4.20.0",
            "torch>=1.10.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
