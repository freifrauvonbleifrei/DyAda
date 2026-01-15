# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

from collections import defaultdict
from copy import deepcopy
from functools import lru_cache, wraps
from importlib.util import find_spec
import numpy as np
import numpy.typing as npt


def module_is_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


def depends_on_optional(module_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not module_is_available(module_name):
                raise ImportError(
                    f"Optional dependency {module_name} not found ({func.__name__})."
                )
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


# cf. https://stackoverflow.com/a/54909677/7272382
def copying_lru_cache():  # TODO add args for lru_cache as needed
    def decorator(func):
        cached_func = lru_cache()(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return deepcopy(cached_func(*args, **kwargs))

        return wrapper

    return decorator


def get_defaultdict_for_markers(
    num_dimensions: int,
) -> defaultdict[int, npt.NDArray[np.int8]]:
    def get_d_zeros_as_array():
        return np.zeros(num_dimensions, dtype=np.int8)

    return defaultdict(get_d_zeros_as_array)
