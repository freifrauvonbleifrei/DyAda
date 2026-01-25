# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later
from dyada.discretization import Discretization
from dyada.linearization import location_code_from_branch
from dyada.markers import MarkersType, MarkersMapProxyType, filter_markers_by_min_index


def merge_mappings(
    first_mapping: list[set[int]],
    second_mapping: list[set[int]],
) -> list[set[int]]:
    # if either mapping is empty, return the other one
    if not first_mapping:
        return second_mapping
    if not second_mapping:
        return first_mapping
    # merge the mappings
    merged_mapping: list[set[int]] = [set() for _ in range(len(first_mapping))]
    for k, v in enumerate(first_mapping):
        for v_i in v:
            merged_mapping[k] |= second_mapping[v_i]
    return merged_mapping


def correct_index_mapping(
    index_mapping: list[set[int]],
    old_discretization: Discretization,
    new_discretization: Discretization,
    markers: MarkersType | MarkersMapProxyType,
) -> None:
    # in case there were refinements at non-leaves, it might be that some mappings are not accurate
    # -> we check out all the location codes in the part of the old descriptor that were changed
    # and see if there's a better match in the new descriptor
    old_descriptor = old_discretization.descriptor
    one_after_last_considered_index = 0
    while one_after_last_considered_index < len(old_descriptor):
        marked_ancestor_index = min(
            filter_markers_by_min_index(
                markers, one_after_last_considered_index
            ).keys(),
            default=-1,
        )
        if marked_ancestor_index == -1:
            break

        unmodified_branch, _ = old_descriptor.get_branch(
            marked_ancestor_index, is_box_index=False
        )
        initial_branch_depth = len(unmodified_branch)
        old_ancestry = old_descriptor.get_ancestry(unmodified_branch)
        assert len(old_ancestry) == initial_branch_depth - 1
        old_ancestry.append(marked_ancestor_index)
        old_ancestry_exact = list(True for _ in old_ancestry)
        index_reverse_replace_map: dict[int, dict[int, set[int]]] = {}
        index_to_consider = marked_ancestor_index
        leaves_to_forget_except: dict[int, int] = {}
        while True:
            # immediately move to next index to consider (first will be OK)
            if old_descriptor.is_box(index_to_consider):
                # at the leaf / tip of a branch -> apply replacements
                for to_modify_depth in range(
                    initial_branch_depth - 1, len(unmodified_branch)
                ):
                    assert len(old_ancestry) == len(unmodified_branch)
                    to_modify_index = old_ancestry[to_modify_depth]
                    if old_ancestry_exact[to_modify_depth]:
                        youngest_that_influences = to_modify_depth
                        oldest_that_influences = to_modify_depth
                    else:
                        oldest_that_influences = initial_branch_depth - 1
                        for depth in range(to_modify_depth, -1, -1):
                            if old_ancestry_exact[depth]:
                                oldest_that_influences = depth
                                break
                        youngest_that_influences = len(unmodified_branch) - 1
                        for depth in range(to_modify_depth + 1, len(unmodified_branch)):
                            if old_ancestry_exact[depth]:
                                youngest_that_influences = depth - 1
                                break

                    influencing = filter(
                        lambda k: k >= oldest_that_influences
                        and k <= youngest_that_influences,
                        index_reverse_replace_map.keys(),
                    )
                    for depth in influencing:
                        assert initial_branch_depth - 1 <= depth
                        assert depth < len(unmodified_branch)
                        for (
                            replace_index,
                            to_replace_set,
                        ) in index_reverse_replace_map[depth].items():
                            if to_replace_set.intersection(
                                index_mapping[to_modify_index]
                            ):
                                index_mapping[to_modify_index].add(replace_index)
                                index_mapping[to_modify_index].difference_update(
                                    to_replace_set
                                )
                try:
                    unmodified_branch.advance_branch(initial_branch_depth)
                    # also prune the replace map if we went up
                    depths_to_remove = [
                        depth
                        for depth in index_reverse_replace_map.keys()
                        if depth >= len(unmodified_branch)
                    ]
                    for depth in depths_to_remove:
                        index_reverse_replace_map.pop(depth)
                    old_ancestry = old_ancestry[: len(unmodified_branch) - 1]
                    old_ancestry_exact = old_ancestry_exact[
                        : len(unmodified_branch) - 1
                    ]
                except IndexError:
                    one_after_last_considered_index = index_to_consider + 1
                    break
            else:
                unmodified_branch.grow_branch(old_descriptor[index_to_consider])
            index_to_consider += 1

            old_index_location_code = location_code_from_branch(
                unmodified_branch, old_discretization._linearization
            )
            try:
                matching_index = new_discretization.get_index_from_location_code(
                    old_index_location_code, get_box=False
                )
                exact_match = True
            except Discretization.NoExactMatchError as e:
                matching_index = e.closest_match
                exact_match = False
            assert isinstance(matching_index, int)
            old_ancestry.append(index_to_consider)
            old_ancestry_exact.append(exact_match)

            should_be_replaced = set(
                index
                for index in index_mapping[index_to_consider]
                if index < matching_index
            )
            if exact_match:
                index_reverse_replace_map[len(unmodified_branch) - 1] = {
                    matching_index: should_be_replaced
                }
                if old_descriptor.is_box(
                    index_to_consider
                ) and new_discretization.descriptor.is_box(matching_index):
                    leaves_to_forget_except[matching_index] = index_to_consider
            else:
                # replace only here
                index_mapping[index_to_consider].add(matching_index)
                index_mapping[index_to_consider].difference_update(should_be_replaced)
        # forget the leaves that can remember themselves
        for index in range(marked_ancestor_index, one_after_last_considered_index):
            for new_leaf_index, old_leaf_index in leaves_to_forget_except.items():
                if index != old_leaf_index:
                    index_mapping[index].discard(new_leaf_index)
