#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

# %% [markdown]
# You can start with a regular `RefinementDescriptor`:

# %%
import bitarray as ba
import dyada
from random import randint

# %%
descriptor = dyada.RefinementDescriptor(2, [2, 1])
# dyada.plot_tree_tikz(descriptor, filename="simple_tree")
num_dimensions = descriptor.get_num_dimensions()
print(descriptor)

# %% [markdown]
# ```python
# RefinementDescriptor('11 01 00 00 ...0 00 01 00 00')
# ```


# %% [markdown]
# This one has four rectangles in the first dimension and two on the second, because
# the level `[2, 1]` is passed as base-2 exponents.
# If you uncomment the line with `plot_tree_tikz` and you have `latexmk` and some
# LaTeX tikz packages installed, the script will generate a `simple_tree.pdf` in the
# same folder.

# %% [markdown]
# You can use the descriptor and `MortonOrderLinearization` to build a `Discretization`:

# %%
discretization = dyada.Discretization(dyada.MortonOrderLinearization(), descriptor)
print("initial discretization:")
print(discretization)

# %% [markdown]
# ```python
# initial discretization:
# _________
# |_|_|_|_|
# |_|_|_|_|
# ```

# %% [markdown]
# If you want to refine a single rectangle at once, you can use `apply_single_refinement`:

# %%
new_discretization, index_mapping = dyada.apply_single_refinement(
    discretization, 0, track_mapping="boxes"
)
print("after refining box 0:")
print(new_discretization)

# %% [markdown]
# ```python
# after refining box 0:
# _________________
# |   |   |   |   |
# |___|___|___|___|
# |_|_|   |   |   |
# |_|_|___|___|___|
# ```


# %% [markdown]
# Of course, you can also refine only in a subset of the dimensions:

# %%
# select random index and refinement
random_index = randint(0, new_discretization.descriptor.get_num_boxes() - 1)
random_refinement = ba.bitarray("00")
while random_refinement.count() == 0:
    random_refinement = ba.bitarray(
        "".join(str(randint(0, 1)) for _ in range(num_dimensions))
    )
new_discretization, index_mapping = dyada.apply_single_refinement(
    new_discretization, random_index, random_refinement, track_mapping="boxes"
)
print("after refining random box:")
print(new_discretization)

# %% [markdown]
# ```python
# after refining random box:
# _________________________________
# |       |       |       |       |
# |_______|_______|_______|_______|
# |___|_|_|   |   |       |       |
# |___|___|___|___|_______|_______|
# ```


# %% [markdown]
# You can keep running the above and watch your discretization become finer and finer!

# %% [markdown]
# To refine many rectangles at once, you can collect the refinements
# as `PlannedAdaptiveRefinement` object:

# %%
refining = dyada.PlannedAdaptiveRefinement(discretization)
refining.plan_refinement(0, ba.bitarray("11"))
refining.plan_refinement(1, ba.bitarray("01"))
new_discretization, index_mapping = refining.apply_refinements(track_mapping="boxes")
# dyada.plot_all_boxes_2d(new_discretization, backend="matplotlib", labels="boxes")
print("after applying planned refinements:")
print(new_discretization)

# %% [markdown]
# ```python
# after applying planned refinements:
# _________________
# |   |   |   |   |
# |___|___|___|___|
# |_|_| | |   |   |
# |_|_|_|_|___|___|
# ```


# %% [markdown]
# If you uncomment the `plot_all_boxes_2d`, it will show you the discretization
# as matplotlib. Other backends are `tikz`, `ascii`(only 2d), and `opengl` (only 3d).

# %%
# TODO when changing this file, update permalinks in readme
