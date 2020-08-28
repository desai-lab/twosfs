import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="twosfs",
    version="0.0.1",
    author="Daniel P. Rice",
    author_email="daniel.paul.rice@gmail.com",
    description="Simulating and manipulating the 2-SFS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dp-rice/twosfs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
