import os
import re
from setuptools import setup, find_packages

# Read version from __init__.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "src", "__init__.py")
    with open(version_file, "r") as f:
        match = re.search(r'__version__ = "(.*?)"', f.read())
        if match:
            return match.group(1)
        raise RuntimeError("Version not found in src/__init__.py")

setup(
    name='deepheal',
    version=get_version(),
    description='DeepHeal: Self-supervised representations of drug-response proteomics',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    # Manually specify packages to map 'src' to 'deepheal'
    packages=["deepheal", "deepheal.model", "deepheal.utils"],
    package_dir={"deepheal": "src"},
    author='Disheng Feng',
    author_email='fengds@fjtcm.edu.cn',
    url='https://github.com/DeepHeal/DeepHeal',
    license="MIT",
    install_requires=[
        "numpy",
        "tqdm",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
