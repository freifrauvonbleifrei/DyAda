# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import dataclass, field

from dyada.descriptor import RefinementDescriptor, Branch
from dyada.discretization import Discretization
from dyada.linearization import location_code_from_branch
from dyada.markers import (
    MarkersType,
    MarkersMapProxyType,
    get_next_largest_markered_index,
)


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


@dataclass
class _BranchCorrectionState:
    index_mapping: list[set[int]]

    unmodified_branch: Branch
    initial_branch_depth: int

    old_ancestry: list[int]
    old_ancestry_exact: list[bool]

    index_reverse_replace_map: dict[int, dict[int, set[int]]] = field(
        default_factory=dict
    )  # TODO easier type
    leaves_to_forget_except: list[tuple[int, int]] = field(default_factory=list)

    index_to_consider: int = 0


def correct_index_mapping(
    index_mapping: list[set[int]],
    old_discretization: Discretization,
    new_discretization: Discretization,
    markers: MarkersType | MarkersMapProxyType,
) -> None:
    old_descriptor = old_discretization.descriptor
    one_after_last_considered_index = 0

    while one_after_last_considered_index < len(old_descriptor):
        marked_ancestor_index = get_next_largest_markered_index(
            markers, one_after_last_considered_index
        )
        if marked_ancestor_index == -1:
            break

        state = _initialize_branch_state_with_dataclass(
            index_mapping=index_mapping,
            old_descriptor=old_descriptor,
            marked_ancestor_index=marked_ancestor_index,
        )

        one_after_last_considered_index = _walk_branch(
            state=state,
            old_discretization=old_discretization,
            new_discretization=new_discretization,
        )

        _forget_self_remembering_leaves(
            index_mapping=index_mapping,
            marked_ancestor_index=marked_ancestor_index,
            one_after_last_considered_index=one_after_last_considered_index,
            leaves_to_forget_except=state.leaves_to_forget_except,
        )


def _initialize_branch_state_with_dataclass(
    index_mapping: list[set[int]],
    old_descriptor: RefinementDescriptor,
    marked_ancestor_index: int,
) -> _BranchCorrectionState:
    unmodified_branch, _ = old_descriptor.get_branch(
        marked_ancestor_index, is_box_index=False
    )
    initial_branch_depth = len(unmodified_branch)

    old_ancestry = old_descriptor.get_ancestry(unmodified_branch)
    assert len(old_ancestry) == initial_branch_depth - 1

    old_ancestry.append(marked_ancestor_index)
    old_ancestry_exact = [True] * len(old_ancestry)

    return _BranchCorrectionState(
        index_mapping=index_mapping,
        unmodified_branch=unmodified_branch,
        initial_branch_depth=initial_branch_depth,
        old_ancestry=old_ancestry,
        old_ancestry_exact=old_ancestry_exact,
        index_to_consider=marked_ancestor_index,
    )


def _walk_branch(
    state: _BranchCorrectionState,
    old_discretization: Discretization,
    new_discretization: Discretization,
) -> int:
    old_descriptor = old_discretization.descriptor
    while True:
        if old_descriptor.is_box(state.index_to_consider):
            _apply_reverse_replacements(state)

            try:
                _advance_branch_state(state)
            except IndexError:
                return state.index_to_consider + 1
        else:
            state.unmodified_branch.grow_branch(old_descriptor[state.index_to_consider])

        state.index_to_consider += 1

        matching_index, exact_match = _find_matching_new_index(
            state.unmodified_branch,
            old_discretization,
            new_discretization,
        )

        state.old_ancestry.append(state.index_to_consider)
        state.old_ancestry_exact.append(exact_match)

        _handle_index_replacement(
            state=state,
            old_descriptor=old_descriptor,
            new_descriptor=new_discretization.descriptor,
            matching_index=matching_index,
            exact_match=exact_match,
        )


def _handle_index_replacement(
    state: _BranchCorrectionState,
    old_descriptor: RefinementDescriptor,
    new_descriptor: RefinementDescriptor,
    matching_index: int,
    exact_match: bool,
):
    idx = state.index_to_consider

    should_be_replaced = {i for i in state.index_mapping[idx] if i < matching_index}

    if exact_match:
        depth = len(state.unmodified_branch) - 1
        state.index_reverse_replace_map[depth] = {matching_index: should_be_replaced}

        if old_descriptor.is_box(idx) and new_descriptor.is_box(matching_index):
            state.leaves_to_forget_except.append((idx, matching_index))
    else:
        state.index_mapping[idx].add(matching_index)
        state.index_mapping[idx].difference_update(should_be_replaced)


def _apply_reverse_replacements(state: _BranchCorrectionState) -> None:
    for to_modify_depth in range(
        state.initial_branch_depth - 1,
        len(state.unmodified_branch),
    ):
        to_modify_index = state.old_ancestry[to_modify_depth]

        youngest, oldest = _compute_influence_range(
            to_modify_depth,
            state.initial_branch_depth,
            state.old_ancestry_exact,
            len(state.unmodified_branch),
        )

        for depth, replacements in state.index_reverse_replace_map.items():
            if oldest <= depth <= youngest:
                for replace_index, to_replace in replacements.items():
                    if to_replace & state.index_mapping[to_modify_index]:
                        state.index_mapping[to_modify_index].add(replace_index)
                        state.index_mapping[to_modify_index].difference_update(
                            to_replace
                        )


def _advance_branch_state(state: _BranchCorrectionState) -> None:
    state.unmodified_branch.advance_branch(state.initial_branch_depth)

    depths_to_remove = [
        d for d in state.index_reverse_replace_map if d >= len(state.unmodified_branch)
    ]
    for d in depths_to_remove:
        state.index_reverse_replace_map.pop(d)

    del state.old_ancestry[len(state.unmodified_branch) - 1 :]
    del state.old_ancestry_exact[len(state.unmodified_branch) - 1 :]


def _compute_influence_range(
    to_modify_depth: int,
    initial_branch_depth: int,
    old_ancestry_exact: list[bool],
    branch_length: int,
) -> tuple[int, int]:
    if old_ancestry_exact[to_modify_depth]:
        return to_modify_depth, to_modify_depth

    oldest = initial_branch_depth - 1
    for d in range(to_modify_depth, -1, -1):
        if old_ancestry_exact[d]:
            oldest = d
            break

    youngest = branch_length - 1
    for d in range(to_modify_depth + 1, branch_length):
        if old_ancestry_exact[d]:
            youngest = d - 1
            break

    return youngest, oldest


def _forget_self_remembering_leaves(
    index_mapping: list[set[int]],
    marked_ancestor_index: int,
    one_after_last_considered_index: int,
    leaves_to_forget_except: list[tuple[int, int]],
) -> None:
    for index in range(marked_ancestor_index, one_after_last_considered_index):
        for old_leaf, new_leaf in leaves_to_forget_except:
            if index != old_leaf:
                index_mapping[index].discard(new_leaf)


def _find_matching_new_index(
    unmodified_branch: Branch,
    old_discretization: Discretization,
    new_discretization: Discretization,
) -> tuple[int, bool]:
    old_location_code = location_code_from_branch(
        unmodified_branch, old_discretization._linearization
    )
    try:
        return (
            new_discretization.get_index_from_location_code(
                old_location_code, get_box=False  # type: ignore
            ),
            True,
        )
    except Discretization.NoExactMatchError as e:
        return e.closest_match, False
