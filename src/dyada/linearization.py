# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

from abc import ABC, abstractmethod
from itertools import product
import bitarray as ba
from dataclasses import dataclass
from typing import Sequence, TypeAlias

from dyada.structure import copying_lru_cache


def single_bit_set_gen(num_dimensions: int):
    for i in range(num_dimensions):
        bit_array = ba.bitarray(num_dimensions)
        bit_array[i] = 1
        yield bit_array


class Linearization(ABC):
    @staticmethod
    @abstractmethod
    def get_binary_position_from_index(
        history_of_indices: Sequence[int],
        history_of_level_increments: Sequence[ba.bitarray],
    ) -> ba.bitarray: ...

    @staticmethod
    @abstractmethod
    def get_index_from_binary_position(
        binary_position: ba.bitarray,
        history_of_indices: Sequence[int],
        history_of_level_increments: Sequence[ba.bitarray],
    ) -> int: ...

    def __eq__(self, other):
        if not isinstance(other, Linearization):
            return False
        # for now, equality is just type equality
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))


class MortonOrderLinearization(Linearization):
    @staticmethod
    def get_binary_position_from_index(
        history_of_indices: Sequence[int],
        history_of_level_increments: Sequence[ba.bitarray],
    ) -> ba.bitarray:
        assert len(history_of_indices) == len(history_of_level_increments)

        this_level_increment = history_of_level_increments[-1]
        assert this_level_increment.count() > 0
        index_in_box = history_of_indices[-1]
        if not index_in_box < 2 ** this_level_increment.count() or index_in_box < 0:
            raise IndexError("Index " + str(index_in_box) + " out of bounds")

        number_of_dimensions = len(this_level_increment)
        binary_position = ba.bitarray(number_of_dimensions)
        # first dimension is the most contiguous
        for dim_index in range(number_of_dimensions):
            if this_level_increment[dim_index]:
                binary_position[dim_index] = index_in_box & 1
                index_in_box >>= 1
        return binary_position

    @staticmethod
    def get_index_from_binary_position(
        binary_position: ba.bitarray,
        history_of_indices: Sequence[int],
        history_of_level_increments: Sequence[ba.bitarray],
    ) -> int:
        assert len(history_of_indices) == len(history_of_level_increments) - 1

        this_level_increment = history_of_level_increments[-1]
        assert this_level_increment.count() > 0
        assert len(binary_position) == len(this_level_increment)
        for i in range(len(binary_position)):
            if binary_position[i]:
                assert this_level_increment[i]
        number_of_dimensions = len(this_level_increment)
        index_in_box = 0

        # first dimension is the most contiguous
        for dim_index in reversed(range(number_of_dimensions)):
            if this_level_increment[dim_index]:
                index_in_box <<= 1
                if binary_position[dim_index]:
                    index_in_box += 1

        return index_in_box


LocationCode: TypeAlias = tuple[ba.frozenbitarray, ...]


def location_code_from_history(
    history_of_binary_positions: Sequence[ba.bitarray],
    history_of_level_increments: Sequence[ba.bitarray],
) -> LocationCode:
    num_dimensions = len(history_of_binary_positions[0])
    depth = len(history_of_binary_positions)
    assert depth > 0
    assert len(history_of_level_increments) == depth
    transposed_positions = [
        ba.bitarray([position[d] for position in history_of_binary_positions])
        for d in range(num_dimensions)
    ]
    transposed_level_increments = [
        ba.bitarray([increment[d] for increment in history_of_level_increments])
        for d in range(num_dimensions)
    ]
    return tuple(
        ba.frozenbitarray(transposed_positions[d][transposed_level_increments[d]])
        for d in range(num_dimensions)
    )


def location_code_from_branch(branch, linearization: Linearization) -> LocationCode:
    history_of_indices, history_of_level_increments = branch.to_history()
    depth = len(history_of_indices)
    assert len(history_of_level_increments) == depth
    history_of_binary_positions = []
    for i in range(depth):
        history_of_binary_positions.append(
            linearization.get_binary_position_from_index(
                history_of_indices[: i + 1],
                history_of_level_increments[: i + 1],
            )
        )
    return location_code_from_history(
        history_of_binary_positions, history_of_level_increments
    )


def binary_or_none_generator(indices: Sequence[int], N: int):
    """Generate all tuples of length N with 0/1 at given indices, None elsewhere.
    None also repeats.

    Args:
        indices (Sequence[int]): indices for which to yield 0/1
        N (int): total number of positions

    Yields:
        tuple[Union[int, None]]: tuples of length N with 0/1 at given indices, None elsewhere
    """
    # value indices reversed so lower index flips faster
    value_indices = list(reversed(sorted(indices)))
    none_indices = [i for i in range(N) if i not in indices]

    pools = [[0, 1]] * len(value_indices) + [[None, None]] * len(none_indices)

    for vals in product(*pools):
        out: list[int | None] = [None] * N

        for i, v in zip(value_indices, vals[: len(value_indices)]):
            out[i] = v

        yield tuple(out)


@dataclass(frozen=True, order=True)
class TrackToken:
    t: int


class ChildGroupingTracker:
    """Groups children of a parent node by a bitmask partition.

    Children are sorted by `sort_mask` bits, so members of the same group
    (sharing the same `sort_mask` bits) are adjacent in pop order.
    Groups are distinguished by the complementary `group_mask` bits.

    Used by coarsening (sort_mask = dims being removed, group by remaining dims)
    and downsplit (sort_mask = dims kept at parent, group by pushed-down dims).
    """

    def __init__(
        self,
        sort_mask: ba.frozenbitarray,
        unresolved_mask: ba.frozenbitarray | None,
        positions: list[ba.frozenbitarray],
    ):
        self.sort_mask = sort_mask
        self.unresolved_mask = unresolved_mask
        self._group_mask = ba.frozenbitarray(
            ~sort_mask & ~unresolved_mask if unresolved_mask else ~sort_mask
        )
        self._group_pointers: dict[ba.frozenbitarray, TrackToken] = {}
        self._positions = positions  # reversed, so pop() gives correct order

    # Keep old attribute names accessible for existing code
    @property
    def separated_mask(self) -> ba.frozenbitarray:
        return self.sort_mask

    def __len__(self) -> int:
        return len(self._positions)

    def pop(self) -> tuple[ba.frozenbitarray, TrackToken | None]:
        """Pop next position. Returns (local_position, same_group_as)."""
        pos = self._positions.pop()
        group_key = ba.frozenbitarray(pos[self._group_mask])
        return pos, self._group_pointers.get(group_key)

    def register_group(
        self, local_position: ba.frozenbitarray, token: TrackToken
    ) -> None:
        """Register that a group maps to the given token."""
        group_key = ba.frozenbitarray(local_position[self._group_mask])
        self._group_pointers[group_key] = token


def indices_to_bitmask(
    indices: Sequence[int], num_dimensions: int
) -> ba.frozenbitarray:
    return ba.frozenbitarray(1 if i in indices else 0 for i in range(num_dimensions))


def bitmask_to_indices(
    bitmask: ba.bitarray | ba.frozenbitarray,
) -> tuple[int, ...]:
    return tuple(i for i in range(len(bitmask)) if bitmask[i])


@copying_lru_cache()
def get_initial_child_grouping(
    current_parent_refinement: ba.frozenbitarray,
    sort_dimensions: ba.frozenbitarray,
    unresolved_dimensions: ba.frozenbitarray | None = None,
    linearization: Linearization = MortonOrderLinearization(),
) -> ChildGroupingTracker:
    """Returns a ChildGroupingTracker for all current children of a parent patch.

    Children are sorted by `sort_dimensions` bits, so entries sharing the same
    sort-dimension values are adjacent. Groups are distinguished by the
    complementary bits.

    For coarsening: sort_dimensions = dims being removed (children to merge
    are adjacent). For downsplit: sort_dimensions = dims kept at parent
    (children in the same target-child group are adjacent).

    Args:
        current_parent_refinement: current parent's refinement, frozen to be hashable
        sort_dimensions: bitmask of dimensions that determine the sort order
        unresolved_dimensions: bitmask of dimensions that cannot be resolved
        linearization: Linearization. Defaults to MortonOrderLinearization().
    Returns:
        ChildGroupingTracker with positions reversed so pop() gives correct order.
    """
    if not isinstance(linearization, MortonOrderLinearization):
        raise NotImplementedError(
            "Initial child grouping only implemented for "
            "Morton order linearization, other linearizations may need "
            "different signature."
        )
    assert len(sort_dimensions) == len(current_parent_refinement)
    assert (sort_dimensions & ~current_parent_refinement).count() == 0
    if unresolved_dimensions is not None:
        current_parent_refinement = ba.frozenbitarray(
            current_parent_refinement | unresolved_dimensions
        )
    positions: list[ba.frozenbitarray] = []
    num_current_children = 2 ** current_parent_refinement.count()
    for child_index in range(num_current_children):
        binary_position = linearization.get_binary_position_from_index(
            (child_index,),
            (current_parent_refinement,),
        )
        positions.append(ba.frozenbitarray(binary_position))

    def _sort_key(pos: ba.frozenbitarray) -> str:
        combined_mask = (
            sort_dimensions | unresolved_dimensions
            if unresolved_dimensions
            else sort_dimensions
        )
        return pos[combined_mask].to01()[::-1]

    positions.sort(key=_sort_key)
    # reverse so we can pop() from the back
    positions.reverse()
    return ChildGroupingTracker(sort_dimensions, unresolved_dimensions, positions)


# Backwards-compatible aliases
CoarseningTracker = ChildGroupingTracker


@copying_lru_cache()
def get_initial_coarsening_stack(
    current_parent_refinement: ba.frozenbitarray,
    dimensions_to_coarsen: ba.frozenbitarray,
    dimensions_cannot_coarsen: ba.frozenbitarray | None = None,
    linearization: Linearization = MortonOrderLinearization(),
) -> ChildGroupingTracker:
    """Backwards-compatible wrapper: dimensions_to_coarsen = sort_dimensions."""
    return get_initial_child_grouping(
        current_parent_refinement,
        dimensions_to_coarsen,
        dimensions_cannot_coarsen,
        linearization,
    )


def get_initial_coarsen_refine_stack(
    current_parent_refinement: ba.frozenbitarray,
    dimensions_to_coarsen: ba.frozenbitarray,
    dimensions_to_refine: ba.frozenbitarray,
    dimensions_cannot_coarsen: ba.frozenbitarray | None = None,
    linearization: Linearization = MortonOrderLinearization(),
) -> CoarseningTracker:
    assert (dimensions_to_coarsen & ~current_parent_refinement).count() == 0
    if dimensions_cannot_coarsen is not None:
        assert dimensions_cannot_coarsen.count() > 0
        assert (dimensions_cannot_coarsen & dimensions_to_refine).count() == 0
        assert (dimensions_cannot_coarsen & dimensions_to_coarsen).count() == 0
        assert (dimensions_cannot_coarsen & current_parent_refinement).count() == 0
    assert (dimensions_to_refine & current_parent_refinement).count() == 0

    later_iterated_dimensions = current_parent_refinement | dimensions_to_refine
    return get_initial_child_grouping(
        later_iterated_dimensions,
        dimensions_to_coarsen,
        dimensions_cannot_coarsen,
        linearization,
    )
