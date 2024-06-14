# `DyAda`: A Code for Memory-Saving Dyadic Adaptivity in Optimization and Simulation

* it's adaptive, but not octree, semicoarsening allowed! -> benefits esp.
for higher-dimensional problems
* key components: your data + a refinement descriptor + a linearization
operator + (optional) an explicit tree for faster lookup
* linearization is flexible: can be Morton Z order, or a tree, or your own implementation

