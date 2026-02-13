# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

import bisect
import bitarray as ba
import bitarray.util
from collections import UserDict
from itertools import combinations
import numpy as np
from typing import Generator, Optional, Sequence, Union


from dyada.coordinates import (
    get_coordinates_from_level_index,
    LevelIndex,
    Coordinate,
    coordinate_from_sequence,
    CoordinateInterval,
    location_code_from_float,
    location_code_from_coordinate,
)
from dyada.descriptor import (
    Branch,
    RefinementDescriptor,
    branch_generator,
    int8_ndarray_from_iterable,
)
from dyada.linearization import (
    Linearization,
    MortonOrderLinearization,
    LocationCode,
    bitmask_to_indices,
)


def get_binary_index_from_branch(
    linearization: Linearization, branch: Branch
) -> list[ba.bitarray]:
    num_dimensions = len(branch[0].level_increment)

    # once the branch is found, we can infer the vector index from the branch stack
    current_index_bitarray = [ba.bitarray() for d in range(num_dimensions)]
    history_of_indices, history_of_level_increments = branch.to_history()
    for level_count in range(1, len(branch)):
        bit_index = linearization.get_binary_position_from_index(
            history_of_indices[:level_count],
            history_of_level_increments[:level_count],
        )
        for d in range(num_dimensions):
            d_level_increment = branch[level_count].level_increment[d]
            if d_level_increment > 0:
                current_index_bitarray[d] += bit_index[d : d + 1]

    return current_index_bitarray


def get_level_index_from_branch(
    linearization: Linearization, branch: Branch
) -> LevelIndex:
    num_dimensions = len(branch[0].level_increment)

    current_index_bitarray = get_binary_index_from_branch(linearization, branch)
    current_level = int8_ndarray_from_iterable(
        [len(b) for b in current_index_bitarray],
    )
    current_index = np.fromiter(
        [
            (bitarray.util.ba2int(b) if len(b) > 0 else 0)
            for b in current_index_bitarray
        ],
        dtype=np.int64,
        count=num_dimensions,
    )
    return LevelIndex(
        current_level,
        current_index,
    )


def get_level_index_from_linear_index(
    linearization: Linearization,
    descriptor: RefinementDescriptor,
    linear_index: int,
    is_box_index: bool = True,
) -> LevelIndex:
    current_branch, _ = descriptor.get_branch(linear_index, is_box_index)
    return get_level_index_from_branch(linearization, current_branch)


class Discretization:
    def __init__(self, linearization: Linearization, descriptor: RefinementDescriptor):
        self._linearization = linearization
        self._descriptor = descriptor

    @property
    def descriptor(self):
        return self._descriptor

    def __eq__(self, other):
        return (
            self._linearization == other._linearization
            and self._descriptor == other._descriptor
        )

    def __hash__(self):
        return hash((self._linearization, self._descriptor))

    def __len__(self):
        return self._descriptor.get_num_boxes()

    def __repr__(self):
        return f"Discretization({repr(self._linearization)}, {repr(self._descriptor)})"

    # TODO has pretty dependency
    # def _repr_pretty_(self, p, cycle):
    #    p.text(str(self) if not cycle else '...')

    def __str__(self):
        # this is monkey-patched when drawing.py is imported
        return repr(self)

    def get_level_index_from_branch(self, branch: Branch) -> LevelIndex:
        return get_level_index_from_branch(self._linearization, branch)

    def get_level_index(self, index: int, is_box_index: bool = True) -> LevelIndex:
        return get_level_index_from_linear_index(
            self._linearization, self._descriptor, index, is_box_index
        )

    def get_all_boxes_level_indices(self) -> Generator:
        for i, _ in enumerate(self._descriptor):
            if self._descriptor.is_box(i):
                yield self.get_level_index(i, False)

    def get_containing_box(self, coordinate: Coordinate) -> Union[int, tuple[int, ...]]:
        coordinates_to_find: set[Coordinate] = {tuple(coordinate)}  # type: ignore
        coordinates_found: set[Coordinate] = set()

        found_box_indices: set[int] = set()

        while len(coordinates_to_find - coordinates_found) > 0:
            coordinate = (coordinates_to_find - coordinates_found).pop()
            location_code = location_code_from_coordinate(coordinate)
            found_index, found_part_of_location_code = (
                self.get_index_from_location_code(location_code, get_box=True)  # type: ignore
            )
            found_box_indices.add(found_index)
            coordinates_found.add(coordinate)

            # check if the coordinate could also be in another box
            # this is the case if there are only zeros left
            # in the location_code behind the respective counted levels
            ambiguous_dimensions = [
                location_code[i].count() > 0
                and location_code[i][found_part_of_location_code[i] :].count() == 0
                for i in range(len(coordinate))
            ]
            ambiguous_indices = bitmask_to_indices(ba.bitarray(ambiguous_dimensions))
            for num_ambiguous in range(1, len(ambiguous_indices) + 1):
                for ambiguous_subset in combinations(ambiguous_indices, num_ambiguous):
                    # copy the original coordinate
                    new_coordinate = coordinate_from_sequence(coordinate)  # type: ignore
                    # and subtract a small epsilon from the ambiguous dimensions
                    for i in ambiguous_subset:
                        new_coordinate[i] = np.nextafter(new_coordinate[i], -np.inf)
                    coordinates_to_find.add(tuple(new_coordinate))  # type: ignore

        return (
            tuple(found_box_indices)
            if len(found_box_indices) > 1
            else found_box_indices.pop()
        )

    class NoExactMatchError(Exception):
        def __init__(self, closest_match: int):
            super().__init__()
            self.closest_match = closest_match

    def get_index_from_location_code(
        self, location_code: LocationCode, get_box: bool
    ) -> int | tuple[int, list[int]]:
        branch = Branch(self._descriptor.get_num_dimensions())
        history_of_indices, history_of_refinements = branch.to_history()

        found_bits = [0] * len(location_code)
        descriptor_iter = iter(self._descriptor)

        patch_index = 0
        box_index = -1

        while True:
            refinement = next(descriptor_iter)

            if not get_box and all(
                found_bits[d] >= len(location_code[d])
                for d in range(len(location_code))
            ):
                return patch_index

            branch.grow_branch(refinement)

            if refinement == self._descriptor.d_zeros:
                return (
                    self._handle_leaf(
                        get_box,
                        patch_index,
                        box_index,
                    ),
                    found_bits,
                )

            history_of_refinements.append(refinement)
            binary_position = self._add_refinement_bits(
                refinement,
                location_code,
                found_bits,
                patch_index,
                get_box,
            )

            child_index = self._linearization.get_index_from_binary_position(
                binary_position,
                history_of_indices,
                history_of_refinements,
            )
            history_of_indices.append(child_index)

            patch_index, box_index = self._skip_unmatched_children(
                descriptor_iter,
                branch,
                child_index,
                patch_index,
                box_index,
            )
            patch_index += 1

    def _handle_leaf(self, get_box: bool, patch_index: int, box_index: int):
        if get_box:
            box_index += 1
            return box_index
        raise Discretization.NoExactMatchError(patch_index)

    def _add_refinement_bits(
        self,
        refinement,
        location_code: LocationCode,
        found_bits: list[int],
        patch_index: int,
        get_box: bool,
    ):
        new_binary_position = ba.bitarray(len(location_code))
        for dim, is_refined in enumerate(refinement):
            if not is_refined:
                continue

            bit_pos = found_bits[dim]
            too_short = bit_pos >= len(location_code[dim])
            if too_short and not get_box:
                raise Discretization.NoExactMatchError(patch_index)

            new_binary_position[dim] = 0 if too_short else location_code[dim][bit_pos]
            found_bits[dim] += 1

        return new_binary_position

    def _skip_unmatched_children(
        self,
        descriptor_iter: Generator,
        branch: Branch,
        child_index: int,
        patch_index: int,
        box_index: int,
    ) -> tuple[int, int]:
        for _ in range(child_index):
            skipped = next(descriptor_iter)
            inc_box, inc_patch = self._descriptor.skip_to_next_neighbor(
                descriptor_iter, skipped
            )
            box_index += inc_box
            patch_index += inc_patch
            branch.advance_branch()

        return patch_index, box_index

    def slice(
        self,
        fixed_unit_coordinates: Sequence[Union[float, None]],
        get_level: bool = False,
    ) -> Union[
        tuple["Discretization", dict[int, int]],
        tuple["Discretization", dict[int, int], list[int | None]],
    ]:
        assert isinstance(
            self._linearization, MortonOrderLinearization
        ), "only MortonOrderLinearization is supported (b/c separable),"
        "if you need another one, please imlement the necessary reordering like in refinement"
        assert not all(c is None for c in fixed_unit_coordinates)
        num_dimensions = self._descriptor.get_num_dimensions()
        assert len(fixed_unit_coordinates) == num_dimensions
        fixed_dimensions = [
            i for i, c in enumerate(fixed_unit_coordinates) if c is not None
        ]
        num_fixed_dimensions = len(fixed_dimensions)
        assert num_fixed_dimensions > 0
        assert num_fixed_dimensions < num_dimensions
        maximum_level_in_slice = [
            0 if d in fixed_dimensions else None for d in range(num_dimensions)
        ]
        location_codes = [
            location_code_from_float(c) if c is not None else None
            for c in fixed_unit_coordinates
        ]
        # create new empty descriptor
        new_descriptor = RefinementDescriptor(
            self._descriptor.get_num_dimensions() - num_fixed_dimensions,
        )
        new_descriptor._data = ba.bitarray()
        index_mapping: dict[int, int] = {}

        # iterate the whole descriptor
        max_depth = np.iinfo(np.int16).max
        skip_depth = max_depth
        for current_descriptor_index, (branch, old_refinement) in enumerate(
            branch_generator(self.descriptor)
        ):
            if len(branch) > skip_depth:
                # skip all children
                continue
            else:
                # then continue normal operation
                skip_depth = max_depth
            bitarray_index = get_binary_index_from_branch(self._linearization, branch)
            level = [len(b) for b in bitarray_index]
            keep = True
            for d in fixed_dimensions:
                if level[d] == 0:
                    continue
                # check if the bitarray is equal to the beginnig of the location code
                if not bitarray_index[d] == location_codes[d][: level[d]]:  # type: ignore
                    keep = False
                    break
            if keep:
                # add this refinement to the new descriptor, omitting the fixed dimensions
                new_refinement = ba.bitarray(old_refinement)
                del new_refinement[fixed_dimensions]
                # if the refinement is perpendicular to the fixed dimensions, keep track but don't append
                is_perpendicular = (
                    new_refinement.count() == 0 and old_refinement.count() > 0
                )
                if is_perpendicular:
                    if len(new_descriptor) == 0:
                        index_mapping[current_descriptor_index] = 0
                    else:
                        # look for the parent in the index mapping
                        siblings = self.descriptor.get_siblings(
                            current_descriptor_index
                        )
                        parent_index = siblings[0] - 1
                        index_mapping[current_descriptor_index] = index_mapping[
                            parent_index
                        ]
                else:
                    new_descriptor.get_data().extend(new_refinement)
                    index_mapping[current_descriptor_index] = len(new_descriptor) - 1
                if get_level:
                    maximum_level_in_slice = [
                        (
                            max(maximum_level_in_slice[d], level[d])  # type: ignore
                            if d in fixed_dimensions
                            else None
                        )
                        for d in range(num_dimensions)
                    ]
            else:
                # skip all children as well
                skip_depth = len(branch)
        assert len(new_descriptor) > 0
        if get_level:
            return (
                Discretization(MortonOrderLinearization(), new_descriptor),
                index_mapping,
                maximum_level_in_slice,
            )
        return Discretization(MortonOrderLinearization(), new_descriptor), index_mapping


def coordinates_from_index(
    discretization: Discretization, index: int, is_box_index: bool = False
) -> CoordinateInterval:
    level_index = get_level_index_from_linear_index(
        discretization._linearization,
        discretization._descriptor,
        linear_index=index,
        is_box_index=is_box_index,
    )
    return get_coordinates_from_level_index(level_index)


def coordinates_from_box_index(
    discretization: Discretization,
    index: int,
    full_domain: Optional[CoordinateInterval] = None,
) -> CoordinateInterval:
    coordinates = coordinates_from_index(discretization, index, is_box_index=True)
    if full_domain is not None:
        scaling_factor = full_domain.upper_bound - full_domain.lower_bound
        offset = full_domain.lower_bound
        coordinates = CoordinateInterval(
            lower_bound=coordinates.lower_bound * scaling_factor + offset,
            upper_bound=coordinates.upper_bound * scaling_factor + offset,
        )
    return coordinates


class SliceDictInDimension(UserDict):
    def __init__(
        self, discretization: Discretization, dimension_index: int, box_mapping: bool
    ):
        super().__init__()
        num_dimensions = discretization.descriptor.get_num_dimensions()
        assert dimension_index < num_dimensions
        start = 0.0
        while start < 1.0:
            d_slice, mapping, level_t = discretization.slice(
                [
                    start if d == dimension_index else None
                    for d in range(num_dimensions)
                ],
                get_level=True,  # type: ignore
            )
            if box_mapping:
                mapping = {
                    discretization.descriptor.to_box_index(
                        k
                    ): d_slice.descriptor.to_box_index(v)
                    for k, v in mapping.items()
                    if discretization.descriptor.is_box(k)
                }
            super().__setitem__(start, (d_slice, mapping))
            start += 2.0 ** -int(level_t[dimension_index])  # type: ignore
        self._sorted_keys = sorted(self.data)

    def __setitem__(self, key, value):
        if key < 0.0 or key > 1.0:
            raise KeyError(f"Key {key} out of bounds (0.0 to 1.0)")
        super().__setitem__(key, value)
        bisect.insort(self._sorted_keys, key)

    def __getitem__(self, key):
        if key < 0.0 or key > 1.0:
            raise KeyError(f"Key {key} out of bounds (0.0 to 1.0)")
        idx = bisect.bisect_right(self._sorted_keys, key) - 1
        if idx >= 0:
            return self.data[self._sorted_keys[idx]]
        raise KeyError


def branch_to_location_code(branch: Branch, linearization) -> list[ba.bitarray]:
    """
    Convert a branch to a location code.
    :param branch: The branch to convert.
    :return: A list of bitarrays representing the dimension-separated location code.
    """
    num_dimensions = len(branch[0].level_increment)
    location_code = [ba.bitarray() for _ in range(num_dimensions)]
    assert isinstance(linearization, MortonOrderLinearization)
    for twig in branch:
        refined_dimensions = bitmask_to_indices(twig.level_increment)
        if len(refined_dimensions) == 0:
            continue
        index_from_count = bitarray.util.int2ba(
            (2 ** len(refined_dimensions) - twig.count_to_go_up),
            length=len(refined_dimensions),
        )
        index_from_count.reverse()  # to match our first-dimension-continuous convention
        for i, d_i in enumerate(refined_dimensions):
            location_code[d_i].append(index_from_count[i])
    return location_code


def discretization_to_location_stack_strings(
    discretization: Discretization, plane_symbol="λ"
) -> list[tuple[str, ...]]:
    """
    Create a location stack from a discretization.
    :param discretization: The discretization to convert.
    :return: A list of d-tuples of strings
    """

    location_stack = []
    for current_branch, refinement in branch_generator(discretization.descriptor):
        # update location codes
        location_codes = branch_to_location_code(
            current_branch, discretization._linearization
        )
        # write current location codes to strings and put to stack, append plane_symbol if refinement in this dimension
        location_stack.append(
            tuple(
                location_codes[d].to01() + (plane_symbol if refinement[d] == 1 else "")
                for d in range(len(location_codes))
            )
        )
    return location_stack
