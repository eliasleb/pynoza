from abc import ABC

import setuptools
from Cython.Build import cythonize
import os
from setuptools.command.build_ext import build_ext as build_ext_orig


class BuildExt(build_ext_orig):

    def run(self):
        os.system("cd speenoza; maturin develop --release; cd ..;")
        super().run()


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pynoza",
    version="0.0a1",
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
        "Programming Language :: Rust",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
    ext_modules=cythonize("src/pynoza/solution.py",
                          compiler_directives={'language_level': "3"}),
    cmdclass={
        'build_ext': BuildExt,
    },
)
