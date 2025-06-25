from abc import ABC, abstractmethod
import bitarray as ba
from typing import Sequence


def single_bit_set_gen(num_dimensions: int):
    for i in range(num_dimensions):
        bit_array = ba.bitarray(num_dimensions)
        bit_array[i] = 1
        yield bit_array


def get_dimensionwise_positions(
    history_of_binary_positions: Sequence[ba.bitarray],
    history_of_level_increments: Sequence[ba.bitarray],
) -> list[ba.bitarray]:
    if len(history_of_binary_positions) == 0:
        return []
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
    deciding_bitarrays = []
    for d in range(num_dimensions):
        deciding_bitarrays.append(
            transposed_positions[d][transposed_level_increments[d]]
        )
    return deciding_bitarrays


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


def get_dimensionwise_positions_from_branch(branch, linearization):
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
    return get_dimensionwise_positions(
        history_of_binary_positions, history_of_level_increments
    )
