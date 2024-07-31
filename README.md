# `DyAda`: A Code for Memory-Saving Dyadic Adaptivity in Optimization and Simulation
[![Python package CI](https://github.com/freifrauvonbleifrei/DyAda/actions/workflows/python-package.yml/badge.svg)](https://github.com/freifrauvonbleifrei/DyAda/actions/workflows/python-package.yml/)  ![Coverage](coverage.svg)  [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

* it's adaptive, but not octree, semicoarsening allowed! -> benefits esp.
for higher-dimensional problems
* key components: your data + a refinement descriptor + a linearization
operator + (optional) an explicit tree for faster lookup (not yet implemented)
* linearization is flexible: can be Morton Z order, or a tree order (not yet implemented), or your own ordering

