import bitarray as ba
import bitarray.util
import dataclasses
from functools import lru_cache
import numpy as np
import numpy.typing as npt
from typing import NamedTuple, TypeAlias, Sequence


class DyadaTooFineError(ValueError):
    pass


@dataclasses.dataclass
class LevelIndex:
    d_level: npt.NDArray[np.int8]
    d_index: npt.NDArray[np.int64]

    def __iter__(self):
        """make iterable, mainly to allow unpacking in assignments"""
        return iter(dataclasses.astuple(self))


def level_index_from_sequence(
    d_level: Sequence[int], d_index: Sequence[int]
) -> LevelIndex:
    if max(d_level) > 62:
        raise DyadaTooFineError(
            f"Level too large, maximum 1d level is 62, \
              got l={d_level}, i={d_index}"
        )
    return LevelIndex(
        np.asarray(d_level, dtype=np.int8),
        np.asarray(d_index, dtype=np.int64),
    )


Coordinate: TypeAlias = npt.NDArray[np.float64]


def coordinate_from_sequence(c: Sequence[float]) -> Coordinate:
    return np.asarray(c, dtype=np.float64)


class CoordinateInterval(NamedTuple):
    lower_bound: Coordinate
    upper_bound: Coordinate

    def __eq__(self, other: object):
        if not isinstance(other, CoordinateInterval):
            return NotImplemented
        return np.all(self.lower_bound == other.lower_bound) and np.all(
            self.upper_bound == other.upper_bound
        )


def interval_from_sequences(
    lower_bound: Sequence[float], upper_bound: Sequence[float]
) -> CoordinateInterval:
    return CoordinateInterval(
        coordinate_from_sequence(lower_bound),
        coordinate_from_sequence(upper_bound),
    )


def get_coordinates_from_level_index(level_index: LevelIndex) -> CoordinateInterval:
    num_dimensions = len(level_index.d_level)
    assert num_dimensions == len(level_index.d_index)
    if any(level_index.d_level > 62):
        raise DyadaTooFineError(
            f"Level too large, maximum 1d level is 62, \
              got l={level_index.d_level}, i={level_index.d_index}"
        )
    if (
        any(level_index.d_level < 0)
        or any(level_index.d_index < 0)
        or any(level_index.d_index >= 2 ** np.array(level_index.d_level, dtype=int))
    ):
        error_string = "Invalid level index: {}".format(level_index)
        if any(level_index.d_index >= 2 ** np.array(level_index.d_level, dtype=int)):
            error_string += " (index {} too large, should be < {})".format(
                level_index.d_index, 2 ** np.array(level_index.d_level, dtype=int)
            )
        raise ValueError(error_string)

    def get_d_array(x):
        return np.fromiter(
            x,
            dtype=np.float64,
            count=num_dimensions,
        )

    @lru_cache(maxsize=None)  # cache the function to avoid recomputing
    def get_neg_power_of_two(level):
        return 2.0 ** -float(level)

    return CoordinateInterval(
        get_d_array(
            (
                get_neg_power_of_two(level) * i
                for level, i in zip(level_index.d_level, level_index.d_index)
            )
        ),
        get_d_array(
            (
                get_neg_power_of_two(level) * (i + 1)
                for level, i in zip(level_index.d_level, level_index.d_index)
            )
        ),
    )


def float_parts_bitarray(value):
    dtype = value.dtype.type if hasattr(value, "dtype") else np.float64
    f = dtype(value)
    if dtype == np.float64:
        raw_bits = f.view(np.uint64)
        exp_bits = 11
        mant_bits = 52
    elif dtype == np.float32:
        raw_bits = f.view(np.uint32)
        exp_bits = 8
        mant_bits = 23
    else:
        raise ValueError("Only float32 and float64 are supported.")

    total_bits = exp_bits + mant_bits + 1
    bits = ba.bitarray(f"{raw_bits:0{total_bits}b}")

    sign_bit = bits[0]
    exponent = bits[1 : 1 + exp_bits]
    mantissa = bits[1 + exp_bits :]

    return sign_bit, exponent, mantissa


def deciding_bitarray_from_float(value):
    if value < 0.0 or value > 1.0:
        raise ValueError("Value must be in the range [0.0, 1.0]")
    if value == 1.0:
        return ba.bitarray("1" * 23)
    sign_bit, exponent, mantissa = float_parts_bitarray(value + 1)
    assert sign_bit == 0
    # exponent represented as 2-complement
    assert bitarray.util.ba2int(exponent) == 2 ** (len(exponent) - 1) - 1
    return mantissa


def bitarray_startswith(
    long_bitarray: ba.bitarray, potentially_shorter_bitarray: ba.bitarray
) -> bool:
    return (
        long_bitarray[: len(potentially_shorter_bitarray)]
        == potentially_shorter_bitarray
    )
