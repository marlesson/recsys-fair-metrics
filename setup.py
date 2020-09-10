#https://packaging.python.org/tutorials/packaging-projects/
import setuptools
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="recsys-fair-metrics", # Replace with your own username
    version="0.0.1",
    author="Marlesson Rodrigues Oliveira de Santana",
    author_email="marlessonsa@gmail.com",
    description="Tools for fairness evaluation in recommendation systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marlesson/recsys-fair-metrics",
   # packages=find_packages("src"),
   # package_dir={"": "src"},    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "tqdm>=4.*", "scipy>=1.4", "scikit-learn>=0.23.*", "psutil>=5.6",
        "requests>=2.*", "kaleido==0.0.3", "plotly>=4.*", "pandas>=1.1.*"
    ],    
)