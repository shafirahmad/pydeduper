"""Setup script for PyDeduper."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="pydeduper",
    version="1.0.0",
    author="Shafir Ahmad",
    author_email="",
    description="A duplicate file finder with folder analysis capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shafirahmad/pydeduper",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: System Administrators",
        "Topic :: System :: Filesystems",
        "Topic :: Utilities",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Recommended for enhanced features
        "tqdm>=4.62.0",  # For progress bars and ETA calculations
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "pylint>=2.12.0",
            "mypy>=0.930",
        ],
        "enhanced": [
            "tqdm>=4.62.0",
            "colorama>=0.4.4",
            "pandas>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pydeduper=cli:main",
            "pydeduper-parallel=parallel_cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)