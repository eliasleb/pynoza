# Copyright (C) 2022  Elias Le Boudec
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


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
    version="0.3",
    author="Elias Le Boudec",
    author_email="elias.leboudec@epfl.ch",
    description="Computation of time-domain solutions of Maxwell's equations using the cartesian multipole expansion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eliasleb/pynoza",
    project_urls={
        "Bug Tracker": "https://github.com/eliasleb/pynoza",
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
