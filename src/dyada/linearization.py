from abc import ABC, abstractmethod
import bitarray as ba
from itertools import product, tee
from typing import Sequence


def single_bit_set_gen(num_dimensions: int):
    for i in range(num_dimensions):
        bit_array = ba.bitarray(num_dimensions)
        bit_array[i] = 1
        yield bit_array


def get_dimensionwise_positions(
    history_of_binary_positions: Sequence[ba.bitarray],
    history_of_level_increments: Sequence[ba.bitarray],
) -> tuple[ba.bitarray, ...]:
    # will contain the same info as level_index, actually
    assert len(history_of_binary_positions) == len(history_of_level_increments)
    num_dimensions = len(history_of_binary_positions[0])
    positons = []
    for d in range(num_dimensions):
        this_dimension_positions = ba.bitarray()
        for i in range(len(history_of_binary_positions)):
            # append only if this dimension is refined
            if history_of_level_increments[i][d]:
                this_dimension_positions.extend(
                    history_of_binary_positions[i][d : d + 1]
                )
            else:
                assert history_of_binary_positions[i][d] == 0

        positons.append(this_dimension_positions)
    return tuple(positons)


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
