from abc import ABC, abstractmethod
import bitarray as ba
from typing import Sequence


class Linearization(ABC):
    @staticmethod
    @abstractmethod
    def get_binary_position_from_index(
        history_of_indices: Sequence[int],
        history_of_level_increments: Sequence[ba.bitarray],
    ) -> ba.bitarray: ...


class MortonOrderLinearization(Linearization):
    @staticmethod
    def get_binary_position_from_index(
        history_of_indices: Sequence[int],
        history_of_level_increments: Sequence[ba.bitarray],
    ) -> ba.bitarray:
        this_level_increment = history_of_level_increments[-1]
        assert this_level_increment.count() > 0
        number_of_dimensions = len(this_level_increment)
        index_in_box = history_of_indices[-1]
        if not index_in_box < 2 ** this_level_increment.count():
            raise ValueError("Index out of bounds")
        binary_position = ba.bitarray(number_of_dimensions)
        # first dimension is the most contiguous
        for dim_index in range(number_of_dimensions):
            if this_level_increment[dim_index]:
                binary_position[dim_index] = index_in_box & 1
                index_in_box >>= 1
        return binary_position
