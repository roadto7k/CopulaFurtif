from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="CopulaFurtif",
    version="0.0.1",
    description="Project for fitting copulas and performing trading analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="=Jérémy Loustau-laguide, Théo Reymermier",
    url="https://github.com/roadto7k/CopulaFurtif",
    license="MIT",
    
    packages=find_packages(),  

    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "yfinance",
        "sympy",
        "hypothesis",
        "pytest",
    ],
    python_requires=">=3.9",

    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
