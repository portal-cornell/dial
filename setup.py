from setuptools import find_packages, setup
from io import open


def read_requirements_file(filename):
    with open(filename) as f:
        return [line.strip() for line in f]


setup(
    name="dial",
    version="0.0.1",
    author="David Wu",
    author_email="david9dragon9@gmail.com",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="",
    package_dir={'': 'src'},
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=read_requirements_file("requirements.txt"),
    entry_points={},
    include_package_data=True,
    python_requires=">=3.10",
    tests_require=["pytest"],
)