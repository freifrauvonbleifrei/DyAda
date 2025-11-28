import bitarray as ba
from collections import defaultdict
import dataclasses
from enum import auto, Enum
from functools import lru_cache
import numpy as np
import numpy.typing as npt
from queue import PriorityQueue
from types import MappingProxyType
from typing import Optional, Union

from dyada.ancestrybranch import AncestryBranch
from dyada.descriptor import (
    RefinementDescriptor,
    get_regular_refined,
    hierarchical_to_box_index_mapping,
    int8_ndarray_from_iterable,
    find_uniqueness_violations,
)
from dyada.discretization import Discretization


def is_lru_cached(func):
    while hasattr(func, "__wrapped__"):
        if hasattr(func, "cache_info"):
            return True
        func = func.__wrapped__
    return hasattr(func, "cache_info")


class PlannedAdaptiveRefinement:
    def __init__(self, discretization: Discretization):
        self._discretization = discretization
        # initialize planned refinement list and data structures used later
        self._planned_refinements: list[tuple[int, npt.NDArray[np.int8]]] = []

        def get_d_zeros_as_array():
            return np.zeros(
                self._discretization.descriptor.get_num_dimensions(), dtype=np.int8
            )

        self._markers: defaultdict[int, npt.NDArray[np.int8]] = defaultdict(
            get_d_zeros_as_array
        )
        self._upward_queue: PriorityQueue[tuple[int, int]] = PriorityQueue()

    def plan_refinement(self, box_index: int, dimensions_to_refine=None) -> None:
        if dimensions_to_refine is None:
            dimensions_to_refine = (
                "1" * self._discretization.descriptor.get_num_dimensions()
            )
        # must be iterable, convert to np.array
        dimensions_to_refine = int8_ndarray_from_iterable(dimensions_to_refine)
        # get hierarchical index
        linear_index = self._discretization.descriptor.to_hierarchical_index(box_index)
        # store by linear index
        self._planned_refinements.append(
            (
                linear_index,
                dimensions_to_refine,
            )
        )

    def populate_queue(self) -> None:
        assert len(self._markers) == 0 and self._upward_queue.empty()

        # put initial markers
        for linear_index, dimensions_to_refine in self._planned_refinements:
            self._markers[linear_index] += int8_ndarray_from_iterable(
                dimensions_to_refine
            )
        # put into the upward queue
        for linear_index in self._markers.keys():
            # obtain level sum to know the priority, highest level should come first
            level_sum = sum(
                self._discretization.descriptor.get_level(linear_index, False)
            )
            self._upward_queue.put((-level_sum, linear_index))

        self._planned_refinements = []  # clear the planned refinements

    def move_marker_to_parent(
        self, marker: npt.NDArray[np.int8], sibling_indices
    ) -> None:
        assert len(sibling_indices) > 1 and len(sibling_indices).bit_count() == 1
        sibling_indices = sorted(sibling_indices)
        # subtract from the current sibling markers
        for sibling in sibling_indices:
            self._markers[sibling] -= marker
            if np.all(self._markers[sibling] == np.zeros(marker.shape, dtype=np.int8)):
                self._markers.pop(sibling)

        # and add it to the parent's marker
        parent = sibling_indices[0] - 1
        self._markers[parent] += marker

    def move_marker_to_descendants(
        self, ancestor_index, marker: npt.NDArray[np.int8], descendants_indices=None
    ):
        if np.all(marker == np.zeros(marker.shape, dtype=np.int8)):
            return
        if descendants_indices is None:
            # assume we want the direct children
            # TODO accept optional branch argument to reduce lookup time
            descendants_indices = self._discretization.descriptor.get_children(
                ancestor_index
            )
        assert (
            len(descendants_indices) > 1 and len(descendants_indices).bit_count() == 1
        )
        assert ancestor_index < min(descendants_indices)

        # subtract from the ancestor
        self._markers[ancestor_index] -= marker
        if np.all(
            self._markers[ancestor_index] == np.zeros(marker.shape, dtype=np.int8)
        ):
            self._markers.pop(ancestor_index)

        # and add to the descendants' markers
        for descendant in descendants_indices:
            self._markers[descendant] += marker

    def upwards_sweep(self) -> None:
        # traverse the tree from down (high level sums) to the coarser levels
        while not self._upward_queue.empty():
            level_sum, linear_index = self._upward_queue.get()

            if linear_index != 0:  # only continue upwards if not yet at the root
                # check if refinement can be moved up the branch (even partially);
                # this requires that all siblings are or would be refined
                siblings = self._discretization.descriptor.get_siblings(linear_index)
                all_siblings_refinements = [
                    int8_ndarray_from_iterable(
                        self._discretization.descriptor[sibling],
                    )
                    for sibling in siblings
                ]

                for i, sibling in enumerate(siblings):
                    if sibling in self._markers:
                        all_siblings_refinements[i] += self._markers[sibling]

                # check where the siblings are all refined
                possible_to_move_up = np.min(all_siblings_refinements, axis=0)
                assert np.all(possible_to_move_up >= 0)

                if np.any(possible_to_move_up > 0):
                    self.move_marker_to_parent(possible_to_move_up, siblings)

                    # put the parent into the upward queue
                    parent_level_sum = level_sum + (
                        len(all_siblings_refinements).bit_length() - 1
                    )
                    assert parent_level_sum > level_sum and parent_level_sum <= 0
                    parent = siblings[0] - 1
                    self._upward_queue.put((parent_level_sum, parent))

                # if any siblings were in the upward queue, remove them too
                for sibling in siblings:
                    if (level_sum, sibling) in self._upward_queue.queue:
                        self._upward_queue.queue.remove((level_sum, sibling))
            self._upward_queue.task_done()

    def downwards_sweep(self) -> None:
        descriptor = self._discretization.descriptor
        if len(self._markers) == 0:
            return
        current_index = min(self._markers.keys())
        while True:
            if not descriptor.is_box(current_index):
                # check if (parts of) the marker need pushing down
                marker_to_push_down = self._markers[current_index].copy()

                # 1st case: negative but cannot be used to coarsen here
                marker_negative = marker_to_push_down < 0
                can_be_coarsened = int8_ndarray_from_iterable(
                    descriptor[current_index],
                )
                marker_to_push_down[marker_negative] += can_be_coarsened[
                    marker_negative
                ]

                # 2nd case: positive but cannot be used to refine here
                marker_positive = marker_to_push_down > 0
                can_be_refined = int8_ndarray_from_iterable(
                    ~descriptor[current_index],
                )
                marker_to_push_down[marker_positive] -= can_be_refined[marker_positive]

                self.move_marker_to_descendants(current_index, marker_to_push_down)

            filtered_markers = {
                k: v for k, v in self._markers.items() if current_index < k
            }
            current_index = min(filtered_markers.keys(), default=-2)
            while current_index != -2 and not np.any(
                filtered_markers[current_index]
            ):  # remove zero markers
                self._markers.pop(current_index)
                filtered_markers.pop(current_index)
                current_index = min(filtered_markers.keys(), default=-2)
            if current_index == -2:
                # no more indices left in markers
                break

    @dataclasses.dataclass
    class Refinement:
        class Type(Enum):
            CopyOver = auto()
            TrackOnly = auto()
            ExpandLeaf = auto()

        type: "Type"
        old_index: int
        new_refinement: ba.bitarray | None = None
        marker_or_ancestor: npt.NDArray[np.int8] | int | None = None

    def modified_branch_generator(self, starting_index: int):
        descriptor = self._discretization.descriptor
        ancestrybranch = AncestryBranch(self._discretization, starting_index)
        proxy_markers = MappingProxyType(self._markers)

        while True:
            current_old_index, intermediate_generation, next_refinement, next_marker = (
                ancestrybranch.get_current_location_info(proxy_markers)
            )
            if next_refinement == descriptor.d_zeros:
                yield self.Refinement(
                    self.Refinement.Type.ExpandLeaf,
                    current_old_index,
                    next_refinement,
                    next_marker,
                )
                for p in intermediate_generation:
                    yield self.Refinement(
                        self.Refinement.Type.TrackOnly,
                        p,
                        None,
                        ancestrybranch.ancestry[-2],
                    )
                # only on leaves, we advance the branch
                try:
                    ancestrybranch.advance()
                except IndexError:  # done!
                    return

            else:
                yield self.Refinement(
                    self.Refinement.Type.CopyOver,
                    current_old_index,
                    next_refinement,
                )
                # on non-leaves, we grow the branch
                ancestrybranch.grow(next_refinement)

    def track_indices(self, old_index: int, new_index: int) -> None:
        assert new_index > -1
        self._index_mapping[old_index].add(new_index)

    def extend_descriptor_and_track_indices(
        self,
        new_descriptor: RefinementDescriptor,
        range_to_extend: Union[int, tuple[int, int]],
        extension: ba.bitarray,
    ) -> None:
        previous_length = len(new_descriptor)
        new_descriptor._data.extend(extension)
        if isinstance(range_to_extend, int):
            for new_index in range(previous_length, len(new_descriptor)):
                self.track_indices(range_to_extend, new_index)
        else:
            for old_index, new_index in zip(
                range(range_to_extend[0], range_to_extend[1]),
                range(previous_length, len(new_descriptor)),
            ):
                self.track_indices(old_index, new_index)

    def filter_markers_by_min_index(
        self, min_index: int
    ) -> MappingProxyType[int, npt.NDArray[np.int8]]:
        # filter the markers to the current interval
        filtered_markers = {k: v for k, v in self._markers.items() if k >= min_index}
        return MappingProxyType(filtered_markers)

    def get_next_index_to_refine(self, min_index: int) -> int:
        return min(
            self.filter_markers_by_min_index(min_index).keys(),
            default=-1,
        )

    def add_refined_data(
        self, new_descriptor: RefinementDescriptor
    ) -> RefinementDescriptor:
        old_descriptor = self._discretization.descriptor
        last_extended_index = -1
        one_after_last_extended_index = 0
        while one_after_last_extended_index < len(old_descriptor):
            index_to_refine = self.get_next_index_to_refine(
                one_after_last_extended_index
            )
            if index_to_refine == -1:
                break

            self.extend_descriptor_and_track_indices(  # linearly copy up to marked
                new_descriptor,
                (one_after_last_extended_index, index_to_refine),
                old_descriptor[one_after_last_extended_index:index_to_refine],
            )
            # only refine where there are markers
            for requested_refinement in self.modified_branch_generator(index_to_refine):
                match requested_refinement:
                    case self.Refinement(
                        self.Refinement.Type.TrackOnly,
                        old_index,
                        None,
                        int() as new_index,
                    ):
                        # track only the index, namely the one of the previous (grand...)parent
                        self.track_indices(old_index, new_index)
                    case self.Refinement(
                        self.Refinement.Type.ExpandLeaf,
                        old_index,
                        ba.bitarray() as new_refinement,
                        np.ndarray() as new_marker,
                    ):
                        self.extend_descriptor_and_track_indices(
                            new_descriptor,
                            old_index,
                            get_regular_refined(new_marker),  # type: ignore
                        )
                    case self.Refinement(
                        self.Refinement.Type.CopyOver,
                        old_index,
                        ba.bitarray() as new_refinement,
                    ):
                        self.extend_descriptor_and_track_indices(
                            new_descriptor, old_index, new_refinement
                        )
                    case _:
                        raise RuntimeError("Logic error: should not be reached")
                last_extended_index = max(last_extended_index, old_index)
            one_after_last_extended_index = last_extended_index + 1

        # copy rest and return
        self.extend_descriptor_and_track_indices(
            new_descriptor,
            (one_after_last_extended_index, len(old_descriptor)),
            old_descriptor[one_after_last_extended_index : len(old_descriptor)],
        )
        return new_descriptor

    def create_new_discretization(
        self, track_mapping: str = "boxes"
    ) -> tuple[Discretization, list[set[int]]]:
        old_descriptor = self._discretization.descriptor
        self._index_mapping: list[set[int]] = [
            set() for _ in range(len(old_descriptor))
        ]

        # start generating the new descriptor
        new_descriptor = RefinementDescriptor(old_descriptor.get_num_dimensions())
        new_descriptor._data = ba.bitarray()

        # we are not changing the old descriptor, and greedily build the new one
        # so we can cache the box indices of both
        if not is_lru_cached(old_descriptor.to_box_index):
            old_descriptor.to_box_index = lru_cache(maxsize=None)(
                old_descriptor._to_box_index_recursive
            )
            old_descriptor._to_box_index_recursive = lru_cache(maxsize=None)(
                old_descriptor._to_box_index_recursive
            )
        new_descriptor.to_box_index = lru_cache(maxsize=None)(  # type: ignore
            new_descriptor._to_box_index_recursive
        )
        new_descriptor._to_box_index_recursive = lru_cache(maxsize=None)(  # type: ignore
            new_descriptor._to_box_index_recursive
        )

        new_descriptor = self.add_refined_data(new_descriptor)

        if track_mapping == "boxes":
            # transform the mapping to box indices
            self._index_mapping = hierarchical_to_box_index_mapping(
                self._index_mapping,
                old_descriptor,
                new_descriptor,
            )
        elif track_mapping == "patches":
            pass
        else:
            raise ValueError(
                "track_mapping must be either 'boxes' or 'patches', got "
                + str(track_mapping)
            )
        return (
            Discretization(self._discretization._linearization, new_descriptor),
            self._index_mapping,
        )

    def apply_refinements(
        self, track_mapping: str = "boxes"
    ) -> tuple[Discretization, list[set[int]]]:
        assert self._upward_queue.empty()
        assert self._markers == {}
        self.populate_queue()
        self.upwards_sweep()
        self.downwards_sweep()
        assert len(self._planned_refinements) == 0

        return self.create_new_discretization(track_mapping)


def apply_single_refinement(
    discretization: Discretization,
    box_index: int,
    dimensions_to_refine: Optional[ba.bitarray] = None,
    track_mapping: str = "boxes",
) -> tuple[Discretization, list[set[int]]]:
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(box_index, dimensions_to_refine)
    return p.apply_refinements(track_mapping=track_mapping)


def merge_mappings(
    first_mapping: list[set[int]],
    second_mapping: list[set[int]],
) -> list[set[int]]:
    # if either mapping is empty, return the other one
    if not first_mapping:
        return second_mapping
    if not second_mapping:
        return first_mapping
    # merge the mappings
    merged_mapping: list[set[int]] = [set() for _ in range(len(first_mapping))]
    for k, v in enumerate(first_mapping):
        for v_i in v:
            merged_mapping[k] |= second_mapping[v_i]
    return merged_mapping


def normalize_discretization(
    discretization: Discretization,
    track_mapping: str = "patches",
    max_normalization_rounds: int = 2**31 - 1,
) -> tuple[Discretization, list[set[int]], int]:
    """
    Normalize the discretization so that it fulfills the uniqueness condition
    and we get a normalized omnitree.
    """
    normalization_rounds = 0
    # find the tuples of indices where the uniqueness condition is violated
    violations = find_uniqueness_violations(discretization.descriptor)
    mapping: list[set[int]] = []
    while len(violations) > 0 and normalization_rounds < max_normalization_rounds:
        normalization_rounds += 1
        p = PlannedAdaptiveRefinement(discretization)
        # remove these violations, by putting markers and executing create_new_descriptor
        for violation in violations:
            # find the dimension(s) of the violation
            sorted_violation = sorted(violation)
            dimensions_to_shift = ~ba.bitarray(
                discretization.descriptor[sorted_violation[0]]
            )
            for i in sorted_violation[1:]:
                dimensions_to_shift &= discretization.descriptor[i]
            assert dimensions_to_shift.count() > 0
            dimensions_to_shift_array = int8_ndarray_from_iterable(
                dimensions_to_shift,
            )
            p._markers[sorted_violation[0]] += dimensions_to_shift_array
            for i in sorted_violation[1:]:
                p._markers[i] -= dimensions_to_shift_array

        # apply the refinements
        new_discretization, new_mapping = p.create_new_discretization(
            track_mapping=track_mapping
        )
        discretization = new_discretization
        mapping = merge_mappings(mapping, new_mapping)
        violations = find_uniqueness_violations(discretization.descriptor)

    return discretization, mapping, normalization_rounds
