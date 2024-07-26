from abc import ABC, abstractmethod
import bitarray as ba
import numpy as np
import struct
from typing import Sequence

from dyada.coordinates import Coordinate


class Linearization(ABC):
    @staticmethod
    @abstractmethod
    def get_binary_position_from_index(
        history_of_indices: Sequence[int],
        history_of_level_increments: Sequence[ba.bitarray],
    ) -> ba.bitarray: ...


def float_to_bits(f: float) -> int:
    # cf. https://stackoverflow.com/a/59594903
    [d] = struct.unpack("=Q", struct.pack("=d", f))
    return d


def most_significant_different_bit(a: float, b: float) -> int:
    """returns the most significant differing bit of two mantissa arguments"""
    xored_bits = float_to_bits(a) ^ float_to_bits(b)
    # actually: -(64 - 12) + xored_bits.bit_length() - 1
    return -53 + xored_bits.bit_length()


def modified_frexp(f: float) -> tuple[float, int]:
    """to get the mantissa and exponent of a float like in IEEE 754 (with mantissa > 1.0)"""
    if f == 0.0:
        return 0.0, 0
    assert f > 0.0
    mantissa, exponent = np.frexp(f)
    # double mantissa until it is larger than or equal to 1.0
    while mantissa < 1.0:
        mantissa *= 2
        exponent -= 1
    return mantissa, exponent


# todo: consider to fuse and vectorize with previous and next function
def xormsb(a: float, b: float) -> int:
    if a == b:
        return np.finfo(np.float32).minexp - 1
    mantissa_a, exponent_a = modified_frexp(a)
    mantissa_b, exponent_b = modified_frexp(b)
    if exponent_a == exponent_b:
        z = most_significant_different_bit(mantissa_a, mantissa_b)
        result = exponent_a + z
    else:
        result = max(exponent_a, exponent_b)
    return result


def get_most_significant_dimension(p: Coordinate, q: Coordinate) -> int:
    assert len(p) == len(q)
    highest_different_bit_so_far = np.finfo(np.float32).minexp - 1
    deciding_dimension = 0
    for i in range(len(p)):
        highest_different_bit = xormsb(p[i], q[i])
        if highest_different_bit > highest_different_bit_so_far:
            highest_different_bit_so_far = highest_different_bit
            deciding_dimension = i
    return deciding_dimension


def compare_morton_order(p: Coordinate, q: Coordinate) -> int:
    deciding_dimension = get_most_significant_dimension(p, q)
    if p[deciding_dimension] < q[deciding_dimension]:
        return -1
    else:
        return 1


class MortonOrderLinearization(Linearization):
    @staticmethod
    def get_binary_position_from_index(
        history_of_indices: Sequence[int],
        history_of_level_increments: Sequence[ba.bitarray],
    ) -> ba.bitarray:
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
