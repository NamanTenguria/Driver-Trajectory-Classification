"""
Setup script for ADM Assignment 3: Trajectory Analysis
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adm-a3-trajectory-analysis",
    version="1.0.0",
    author="Naman Tenguria",
    author_email="naman.tenguria@example.com",
    description="Driver classification using trajectory data with RNN and LSTM models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/namantenguria/adm-a3-trajectory-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "trajectory-train=src.main:train_models",
            "trajectory-predict=src.prediction:run_prediction",
        ],
    },
)
