from abc import ABC, abstractmethod
import bitarray as ba
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, TypeAlias

from dyada.structure import copying_lru_cache


def single_bit_set_gen(num_dimensions: int):
    for i in range(num_dimensions):
        bit_array = ba.bitarray(num_dimensions)
        bit_array[i] = 1
        yield bit_array


LocationCode: TypeAlias = tuple[ba.frozenbitarray, ...]


def location_code_from_strings(s: Sequence[str]) -> LocationCode:
    return tuple(ba.frozenbitarray(bit_string) for bit_string in s)


def location_codes_from_history(
    history_of_binary_positions: Sequence[ba.bitarray],
    history_of_level_increments: Sequence[ba.bitarray],
) -> LocationCode:
    if len(history_of_binary_positions) == 0:
        return ()
    num_dimensions = len(history_of_binary_positions[0])
    depth = len(history_of_binary_positions)
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


def location_codes_from_branch(branch, linearization):
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
    return location_codes_from_history(
        history_of_binary_positions, history_of_level_increments
    )


@dataclass
class SameIndexAs:
    old_indices: set[int]


@dataclass
class DimensionSeparatedLocalPosition:
    local_position: ba.frozenbitarray
    separated_dimensions_mask: ba.frozenbitarray
    same_index_as: SameIndexAs | None = None
    new_refined_location_code: LocationCode | None = None

    @cached_property
    def remaining_positions_mask(self) -> ba.frozenbitarray:
        return ba.frozenbitarray(~self.separated_dimensions_mask)

    @cached_property
    def separated_positions(self) -> ba.frozenbitarray:
        return ba.frozenbitarray(self.local_position[self.separated_dimensions_mask])

    @cached_property
    def remaining_positions(self) -> ba.frozenbitarray:
        return ba.frozenbitarray(self.local_position[~self.separated_dimensions_mask])


CoarseningStack: TypeAlias = list[DimensionSeparatedLocalPosition]


def indices_to_bitmask(
    indices: Sequence[int], num_dimensions: int
) -> ba.frozenbitarray:
    return ba.frozenbitarray(1 if i in indices else 0 for i in range(num_dimensions))


@copying_lru_cache()
def get_initial_coarsening_stack(
    current_parent_refinement: ba.frozenbitarray,
    dimensions_to_coarsen: tuple[int, ...] | ba.frozenbitarray,
    linearization: Linearization = MortonOrderLinearization(),
) -> CoarseningStack:
    """Returns a stack of coarsening mappings for all current children of a parent patch.

    Args:
        current_parent_refinement (ba.frozenbitarray): current parent's refinement, frozen to be hashable
        dimensions_to_coarsen (tuple[int, ...] | ba.frozenbitarray): a sorted tuple of dimensions to coarsen or a frozenbitarray mask
        linearization (Linearization, optional): Linearization. Defaults to MortonOrderLinearization().

    Raises:
        NotImplementedError: if linearization is not Morton order

    Returns:
        CoarseningStack: a list of DimensionSeparatedLocalPosition entries, one per current child patch
        reversed, so popping from the back gives the correct order.
        The entries contain the separated positions for the coarsened dimensions and the remaining positions for the other dimensions.
    """

    if not isinstance(linearization, MortonOrderLinearization):
        raise NotImplementedError(
            "Initial coarsening stack generation only implemented for"
            + "Morton order linearization, other linearizations may need"
            + "different signature."
        )
    if not isinstance(dimensions_to_coarsen, ba.frozenbitarray):
        assert len(dimensions_to_coarsen) == len(set(dimensions_to_coarsen))
        dimensions_to_coarsen = indices_to_bitmask(
            dimensions_to_coarsen, len(current_parent_refinement)
        )
    assert len(dimensions_to_coarsen) == len(current_parent_refinement)

    assert (dimensions_to_coarsen & ~current_parent_refinement).count() == 0

    initial_coarsening_stack: CoarseningStack = []
    num_current_children = 2 ** current_parent_refinement.count()

    for child_index in range(num_current_children):
        binary_position = linearization.get_binary_position_from_index(
            (child_index,),
            (current_parent_refinement,),
        )
        initial_coarsening_stack.append(
            DimensionSeparatedLocalPosition(
                local_position=ba.frozenbitarray(binary_position),
                separated_dimensions_mask=dimensions_to_coarsen,
            )
        )
    initial_coarsening_stack.sort(
        key=lambda entry: entry.separated_positions.to01()[::-1]
    )

    # reverse so we can pop() from the back
    initial_coarsening_stack.reverse()
    return initial_coarsening_stack


def get_initial_coarsen_refine_stack(
    current_parent_refinement: ba.frozenbitarray,
    dimensions_to_coarsen: tuple[int, ...] | ba.frozenbitarray,
    dimensions_to_refine: tuple[int, ...] | ba.frozenbitarray,
    linearization: Linearization = MortonOrderLinearization(),
) -> CoarseningStack:
    if not isinstance(dimensions_to_coarsen, ba.frozenbitarray):
        assert len(dimensions_to_coarsen) == len(set(dimensions_to_coarsen))
        dimensions_to_coarsen = indices_to_bitmask(
            dimensions_to_coarsen, len(current_parent_refinement)
        )
    if not isinstance(dimensions_to_refine, ba.frozenbitarray):
        assert len(dimensions_to_refine) == len(set(dimensions_to_refine))
        dimensions_to_refine = indices_to_bitmask(
            dimensions_to_refine, len(current_parent_refinement)
        )
    assert (dimensions_to_coarsen & ~current_parent_refinement).count() == 0
    assert (dimensions_to_refine & current_parent_refinement).count() == 0

    initial_coarsening_stack = get_initial_coarsening_stack(
        current_parent_refinement,
        dimensions_to_coarsen,
        linearization,
    )

    later_refined_dimensions = (
        current_parent_refinement & ~dimensions_to_coarsen
    ) | dimensions_to_refine
    later_num_children = 2 ** later_refined_dimensions.count()
    children_positions = [
        linearization.get_binary_position_from_index(
            (i,),
            (later_refined_dimensions,),
        )
        for i in range(later_num_children)
    ]
    children_locations = [
        location_codes_from_history(
            [pos],
            [later_refined_dimensions],
        )
        for pos in children_positions
    ]
    children_locations.reverse()

    coarsen_refine_stack: CoarseningStack = []

    for entry in initial_coarsening_stack:
        for child_location in children_locations:
            coarsen_refine_stack.append(
                DimensionSeparatedLocalPosition(
                    local_position=entry.local_position,
                    separated_dimensions_mask=entry.separated_dimensions_mask,
                    new_refined_location_code=child_location,
                )
            )

    return coarsen_refine_stack


def inform_same_remaining_position_about_index(
    coarsening_stack: CoarseningStack,
    position_to_update: DimensionSeparatedLocalPosition,
    mapped_to_index: SameIndexAs,
) -> None:
    for i, entry in enumerate(coarsening_stack):
        if entry.remaining_positions == position_to_update.remaining_positions:
            coarsening_stack[i].same_index_as = mapped_to_index
