from setuptools import find_packages, setup

setup(
    name="myccl",
    version="0.0",
    description="Moving CCL codebase to Python.",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.7",
)
