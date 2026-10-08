from setuptools import setup, find_packages
import os

def parse_requirements(filename):
    if not os.path.exists(filename):
        return []
    with open(filename, "r", encoding="utf-8") as f:

        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="CMAB-Pruning", 
    version="0.1.0",
    author="start-ship-it", 
    description="MLLM-Pruning Method", 
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/start-ship-it/CMAB-Pruning", 
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"), 
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
    ],
)
