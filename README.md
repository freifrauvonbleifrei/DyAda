# `DyAda`: A Code for Dyadic Adaptivity in Optimization, Simulation, and Machine Learning

[![PyPI version](https://img.shields.io/pypi/v/dyada)](https://pypi.org/project/dyada/)
[![Supported Python version](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](https://github.com/freifrauvonbleifrei/DyAda/blob/main/pyproject.toml)
[![Python package CI](https://github.com/freifrauvonbleifrei/DyAda/actions/workflows/python-package.yml/badge.svg)](https://github.com/freifrauvonbleifrei/DyAda/actions/workflows/python-package.yml/)
![Coverage](coverage.svg)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/edde9abcb3a249979e71f08e23a14112)](https://app.codacy.com/gh/freifrauvonbleifrei/DyAda/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Installation

It's as simple as

```bash
pip install dyada[drawing,matplotlib,opengl]
```

Or, if you would like to change the source code, do

```bash
git clone https://github.com/freifrauvonbleifrei/DyAda.git
# ... git checkout the required version ...
pip install -e DyAda[drawing,matplotlib,opengl]
```

## Dyadic Adaptivity

Dyadic adaptivity means: A given hypercube of 2 or more dimensions may or may not
be subdivided into two parts in any number of dimensions.
Of the resulting sub-boxes, each may again be subdivided into two in any dimension,
and so forth.

### Why Dyadic Adaptivity?

Currently, the most common approach to adaptivity are octrees, which are a
special type of dyadic adaptivity: Each box is either refined in *every* dimension
or not at all.
For a three-d domain, the tree and the resulting partitioning could look like this:

<!-- 
images generated like this:
```bash
for f in *.tex ; do latexmk -pdf $f ; done
for d in *.pdf ; do inkscape --without-gui --file=$d --export-plain-svg=${d%.*}.svg ; done
rsvg-convert tikz_cubes_solid.svg -w 268 -h 252 -f svg -o tikz_cubes_solid.svg #etc.
``` -->
![The octree tree](docs/gfx/octree_tree.svg)

![The octree partitioning](docs/gfx/octree_solid.svg)

But maybe you didn't need all this resolution?

Maybe, in the finely-resolved areas, you only needed only *some* of the dimensions
resolved finely:

![The dyadic partitioning](docs/gfx/tikz_cubes_solid.svg)

This is what DyAda provides.

The tree will then look like this:

![The omnitree tree](docs/gfx/omnitree.svg)

And you will only have to use 14 degrees of freedom instead of 29!
This reduction will be even stronger if you go to higher dimensions.

## Contributing

Feel free to request features or voice your intent to work on/with DyAda as an 
[issue](https://github.com/freifrauvonbleifrei/DyAda/issues).
Depending on what you are looking for, exciting features may be in preparation,
or they may just be waiting for you to implement them!
