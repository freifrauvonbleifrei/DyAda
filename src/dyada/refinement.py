import numpy as np
import math
import bitarray as ba
from icecream import ic


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
    def __init__(self, num_dimensions, base_resolution_level=0):
        self._num_dimensions = num_dimensions
        if isinstance(base_resolution_level, int):
            base_resolution_level = [base_resolution_level] * self._num_dimensions
        assert len(base_resolution_level) == self._num_dimensions
        level_sum = sum(base_resolution_level)
        self._data = ba.bitarray((2 ** (level_sum + 1) - 1) * self._num_dimensions)
        # establish the base resolution level
        
        dZeros = ba.bitarray(self._num_dimensions)
        self._data = dZeros.copy()
        #iterate in reverse from max(level) to 0
        for l in reversed(range(max(base_resolution_level))):
            at_least_l = ba.bitarray([i > l for i in base_resolution_level])
            factor = 2**at_least_l.count()
            self._data = at_least_l + self._data * factor

        ic(self._data)

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
        ic(dZeros)
        ic(list(self._data[0 : self._num_dimensions :]))
        # todo check back if there will be such a function in bitarray
        count = sum(
            1
            for i in range(0, len(self._data), self._num_dimensions)
            if self._data[i : i + self._num_dimensions] == dZeros
        )
        return ic(count)

    def get_data(self):
        return self._data


def validate_descriptor(descriptor: RefinementDescriptor):
    assert len(descriptor._data) % descriptor._num_dimensions == 0
