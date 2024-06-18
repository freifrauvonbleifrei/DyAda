import numpy as np
import bitarray as ba
from collections import deque, Counter


# generalized (2^d-ary) ruler function, e.g. https://oeis.org/A115362
def generalized_ruler(num_dimensions: int, level: int) -> np.ndarray:
    assert level >= 0 and level < 256
    current_list = np.array([], dtype=np.uint8)
    current_list = np.array([1], dtype=np.uint8)
    for i in range(0, level):
        current_list = np.tile(current_list, 2**num_dimensions)
        # actually, a reversed version, change the first element
        current_list[0] += 1
    return current_list


class RefinementDescriptor:
    def __init__(self, num_dimensions: int, base_resolution_level=0):
        self._num_dimensions = num_dimensions
        if isinstance(base_resolution_level, int):
            base_resolution_level = [base_resolution_level] * self._num_dimensions
        assert len(base_resolution_level) == self._num_dimensions

        # establish the base resolution level
        dZeros = ba.bitarray(self._num_dimensions)
        self._data = dZeros
        # iterate in reverse from max(level) to 0
        for l in reversed(range(max(base_resolution_level))):
            at_least_l = ba.bitarray([i > l for i in base_resolution_level])
            factor = 2 ** at_least_l.count()
            self._data = at_least_l + self._data * factor

    def __len__(self):
        """
        return the number of refinement descriptions, will be somewhere between get_num_boxes() and 2*get_num_boxes()
        """
        return len(self._data) // self._num_dimensions

    def get_num_dimensions(self):
        return self._num_dimensions

    def get_num_boxes(self):
        # count number of d*(0) bit blocks
        dZeros = ba.bitarray(self._num_dimensions)
        # ic(list(self._data[0 : self._num_dimensions :]))
        # todo check back if there will be such a function in bitarray
        count = sum(
            1
            for i in range(0, len(self._data), self._num_dimensions)
            if self._data[i : i + self._num_dimensions] == dZeros
        )
        return count

    def get_data(self):
        return self._data

    def __iter__(self):
        for i in range(len(self)):
            yield ba.frozenbitarray(self[i])

    def __getitem__(self, index_or_slice):
        nd = self._num_dimensions
        if isinstance(index_or_slice, slice):
            assert index_or_slice.step == 1 or index_or_slice.step is None
            start = index_or_slice.start
            stop = index_or_slice.stop
            start = 0 if start is None else start
            stop = len(self) if stop is None else stop
            return self.get_data()[start * nd : stop * nd]
        else:  # it should be an index
            return self.get_data()[index_or_slice * nd : (index_or_slice + 1) * nd]

    def is_pow2tree(self):
        """Is this a quadtree / octree / general power-of-2 tree?"""
        c = Counter(self)
        return c.keys() == {
            ba.frozenbitarray(self._num_dimensions),
            ba.frozenbitarray("1" * self._num_dimensions),
        }

    def get_level(self, index: int) -> int:
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        class LevelCounter:
            def __init__(self, level, count=0):
                self.level = level
                self.count = count

        to_go_up: deque = deque()
        current_level = 0
        to_go_up.append(LevelCounter(0, 1))
        dZeros = ba.bitarray(self._num_dimensions)
        for i in range(index):
            current = self[i]
            to_go_up[-1].count -= 1
            if current == dZeros:
                while to_go_up[-1].count == 0:
                    assert current_level == to_go_up.pop().level
                    current_level = to_go_up[-1].level
            else:
                cnt = current.count()
                current_level += cnt
                for u in to_go_up:
                    assert current_level > u.level
                to_go_up.append(LevelCounter(current_level, 2**cnt))
        return current_level


def validate_descriptor(descriptor: RefinementDescriptor):
    assert len(descriptor._data) % descriptor._num_dimensions == 0
