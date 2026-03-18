# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

from abc import ABC, abstractmethod
from itertools import product
import bitarray as ba
from bitarray.util import ba2int, int2ba
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


class CoarseningTracker:
    """Tracks coarsening group mappings for children of a coarsening parent.

    Uses a counter instead of an explicit positions list.  The counter bits
    are arranged as remaining (LSBs) | separated | unresolved (MSBs), all in
    Morton order within each group.  Position bitarrays are computed on demand
    by scattering counter bits to dimension positions via bitarray masking.
    """

    def __init__(
        self,
        separated_mask: ba.frozenbitarray,
        unresolved_mask: ba.frozenbitarray | None,
        remaining_mask: ba.frozenbitarray,
        num_children: int,
    ):
        self.separated_mask = separated_mask
        self.unresolved_mask = unresolved_mask
        self._remaining_mask = remaining_mask
        self._num_children = num_children
        self._counter = 0
        self._num_remaining = remaining_mask.count()
        self._num_separated = separated_mask.count()
        self._num_unresolved = unresolved_mask.count() if unresolved_mask else 0
        self._remaining_key_mask = (1 << self._num_remaining) - 1
        self._separated_key_mask = (1 << self._num_separated) - 1
        self._ndim = len(separated_mask)
        self._group_pointers: dict[int, TrackToken] = {}

    def __len__(self) -> int:
        return self._num_children - self._counter

    @staticmethod
    def _int_to_ba_le(n: int, length: int) -> ba.bitarray:
        """Convert int to bitarray with LSB first (little-endian bit order)."""
        bits = int2ba(n, length=length)
        bits.reverse()
        return bits

    def _to_position(self, k: int) -> ba.frozenbitarray:
        """Convert counter value to position bitarray by scattering bits."""
        pos = ba.bitarray(self._ndim)
        if self._num_remaining > 0:
            pos[self._remaining_mask] = self._int_to_ba_le(
                k & self._remaining_key_mask, self._num_remaining
            )
        if self._num_separated > 0:
            pos[self.separated_mask] = self._int_to_ba_le(
                (k >> self._num_remaining) & self._separated_key_mask,
                self._num_separated,
            )
        if self._num_unresolved > 0:
            pos[self.unresolved_mask] = self._int_to_ba_le(
                k >> (self._num_remaining + self._num_separated),
                self._num_unresolved,
            )
        return ba.frozenbitarray(pos)

    def _remaining_key(self, local_position: ba.frozenbitarray) -> int:
        """Extract remaining-dimensions key from a position bitarray."""
        if self._num_remaining == 0:
            return 0
        return ba2int(local_position[self._remaining_mask][::-1])

    def pop(self) -> tuple[ba.frozenbitarray, TrackToken | None]:
        """Pop next position. Returns (local_position, same_index_as)."""
        k = self._counter
        self._counter += 1
        pos = self._to_position(k)
        return pos, self._group_pointers.get(k & self._remaining_key_mask)

    def register_group(
        self, local_position: ba.frozenbitarray, token: TrackToken
    ) -> None:
        """Register that a remaining-position group maps to the given token."""
        self._group_pointers[self._remaining_key(local_position)] = token


def indices_to_bitmask(
    indices: Sequence[int], num_dimensions: int
) -> ba.frozenbitarray:
    return ba.frozenbitarray(1 if i in indices else 0 for i in range(num_dimensions))


def bitmask_to_indices(
    bitmask: ba.bitarray | ba.frozenbitarray,
) -> tuple[int, ...]:
    return tuple(i for i in range(len(bitmask)) if bitmask[i])


@copying_lru_cache()
def get_initial_coarsening_stack(
    current_parent_refinement: ba.frozenbitarray,
    dimensions_to_coarsen: ba.frozenbitarray,
    dimensions_cannot_coarsen: ba.frozenbitarray | None = None,
    linearization: Linearization = MortonOrderLinearization(),
) -> CoarseningTracker:
    """Returns a CoarseningTracker for all current children of a parent patch.

    Args:
        current_parent_refinement: current parent's refinement, frozen to be hashable
        dimensions_to_coarsen: a frozenbitarray mask of dimensions to coarsen
        dimensions_cannot_coarsen: a frozenbitarray mask of dimensions that cannot be coarsened
        linearization: Linearization. Defaults to MortonOrderLinearization().
    Returns:
        CoarseningTracker whose counter iterates children in the correct order.
    """
    if not isinstance(linearization, MortonOrderLinearization):
        raise NotImplementedError(
            "Initial coarsening stack generation only implemented for"
            + "Morton order linearization, other linearizations may need"
            + "different signature."
        )
    assert len(dimensions_to_coarsen) == len(current_parent_refinement)
    assert (dimensions_to_coarsen & ~current_parent_refinement).count() == 0
    separated_mask = dimensions_to_coarsen
    effective_refinement = current_parent_refinement
    if dimensions_cannot_coarsen is not None:
        effective_refinement = ba.frozenbitarray(
            current_parent_refinement | dimensions_cannot_coarsen
        )

    # Remaining = refined and neither separated nor unresolved
    if dimensions_cannot_coarsen is not None:
        remaining_mask = ba.frozenbitarray(
            effective_refinement & ~separated_mask & ~dimensions_cannot_coarsen
        )
    else:
        remaining_mask = ba.frozenbitarray(effective_refinement & ~separated_mask)

    num_children = 2 ** effective_refinement.count()
    return CoarseningTracker(
        separated_mask,
        dimensions_cannot_coarsen,
        remaining_mask,
        num_children,
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
    return get_initial_coarsening_stack(
        later_iterated_dimensions,
        dimensions_to_coarsen,
        dimensions_cannot_coarsen,
        linearization,
    )
