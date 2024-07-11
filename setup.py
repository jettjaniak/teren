from setuptools import find_packages, setup

setup(
    name="teren",
    packages=find_packages(where="."),
    package_dir={"": "."},
    include_package_data=True,
)
