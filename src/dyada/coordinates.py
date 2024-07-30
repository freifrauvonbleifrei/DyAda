import dataclasses
import numpy as np
import numpy.typing as npt
from typing import NamedTuple, TypeAlias, Sequence


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
    return LevelIndex(
        np.asarray(d_level, dtype=np.int8),
        np.asarray(d_index, dtype=np.int64),
    )


Coordinate: TypeAlias = npt.NDArray[np.float32]


def coordinate_from_sequence(c: Sequence[float]) -> Coordinate:
    return np.asarray(c, dtype=np.float32)


class CoordinateInterval(NamedTuple):
    lower_bound: Coordinate
    upper_bound: Coordinate

    def __eq__(self, other: object):
        if not isinstance(other, CoordinateInterval):
            return NotImplemented
        return np.all(self.lower_bound == other.lower_bound) and np.all(
            self.upper_bound == other.upper_bound
        )

    def contains(self, coordinate: Coordinate) -> bool:
        return np.all(self.lower_bound <= coordinate) and np.all(
            coordinate <= self.upper_bound
        )  # type: ignore


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
    get_d_array = lambda x: np.fromiter(
        x,
        dtype=np.float32,
        count=num_dimensions,
    )
    return CoordinateInterval(
        get_d_array(
            (
                2.0 ** -float(l) * i
                for l, i in zip(level_index.d_level, level_index.d_index)
            )
        ),
        get_d_array(
            (
                2.0 ** -float(l) * (i + 1)
                for l, i in zip(level_index.d_level, level_index.d_index)
            )
        ),
    )
