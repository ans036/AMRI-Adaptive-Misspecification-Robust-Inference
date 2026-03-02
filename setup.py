from setuptools import setup, find_packages

setup(
    name="amri",
    version="0.1.0",
    description=(
        "Adaptive Misspecification-Robust Inference: "
        "Confidence intervals that adapt to model misspecification"
    ),
    long_description=open("README.md", encoding="utf-8").read()
    if __import__("os").path.exists("README.md")
    else "",
    long_description_content_type="text/markdown",
    author="Anish",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "scikit-learn>=1.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
