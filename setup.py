import setuptools
from Cython.Build import cythonize

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pynoza",
    version="0.0a0",
    author="Elias Le Boudec",
    author_email="elias.leboudec@epfl.ch",
    description="Computation of time-domain solutions of Maxwell's equations using the cartesian multipole expansion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://c4science.ch/source/pynoza",
    project_urls={
        "Bug Tracker": "https://c4science.ch/source/pynoza",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
    ext_modules=cythonize("src/pynoza/solution.py",
                          compiler_directives={'language_level': "3"}),
)