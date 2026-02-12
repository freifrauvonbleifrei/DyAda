#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later
import bitarray as ba
import dyada
from random import randint

descriptor = dyada.RefinementDescriptor(2, [1, 2])
# dyada.plot_tree_tikz(descriptor, filename="simple_tree")
num_dimensions = descriptor.get_num_dimensions()
print(descriptor)

discretization = dyada.Discretization(dyada.MortonOrderLinearization(), descriptor)
print("initial discretization:")
print(discretization)

new_discretization, index_mapping = dyada.apply_single_refinement(
    discretization, 0, track_mapping="boxes"
)
print("after refining box 0:")
print(new_discretization)

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

refining = dyada.PlannedAdaptiveRefinement(discretization)
refining.plan_refinement(0, ba.bitarray("11"))
refining.plan_refinement(1, ba.bitarray("10"))
new_discretization, index_mapping = refining.apply_refinements(track_mapping="boxes")
dyada.plot_all_boxes_2d(new_discretization, labels="boxes")
print("after applying planned refinements:")
print(new_discretization)