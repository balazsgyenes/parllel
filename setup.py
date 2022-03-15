import setuptools

CORE_REQUIRES = [
    "numpy", "nptyping",
    "gym",
    "numba",
    "tqdm",
]
TORCH_REQUIRES = [
    "torch",
]
TEST_REQUIRES = [
    # testing and coverage
    'pytest', 'coverage', 'pytest-cov',
    # unmandatory dependencies of the package itself
    'opencv-python', 'psutil', 'pyprind',  # TODO: update
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="parllel",
    version="0.0.1",
    author="Balazs Gyenes",
    author_email="balazs.gyenes@kit.edu",
    description="An RL accelerator framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",  #TODO: update
    # project_urls={
    #     "Bug Tracker": "https://github.com/pypa/sampleproject/issues",   #TODO: update
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "parllel"},
    # packages=setuptools.find_packages(where="parllel"),
    python_requires=">=3.9",
    install_requires=CORE_REQUIRES + TORCH_REQUIRES,
)
