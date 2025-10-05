from collections.abc import Iterable
import bitarray as ba
import functools


class LocationCodeMap:
    def __init__(self) -> None:
        self._map: dict[tuple[ba.bitarray, ...], int] = {}
        # TODO think of smarter data structure, maybe a patricia trie?
        # TODO or marisa trie? or use IEEE floats?

    def add(self, location_code: Iterable[ba.bitarray], index: int):
        self._map[tuple(ba.frozenbitarray(l) for l in location_code)] = index

    def get_tightest_match(self, location_code: Iterable[ba.bitarray]) -> int:
        def get_matching_levels(
            looking_for: ba.bitarray, candidate: ba.bitarray
        ) -> int:
            """returns how well the candidate matches the looking_for as a 1-d location code:
            -1 means no match (or candidate too long), and a higher number means a better match
            """
            if len(candidate) > len(looking_for):
                return -1
            level = 0
            for level in range(len(candidate)):
                if looking_for[level] != candidate[level]:
                    return -1
            return level + 1

        def get_total_matching_levels(
            looking_for: Iterable[ba.bitarray], candidate: Iterable[ba.bitarray]
        ) -> int:
            total = 0
            for one_d_looking_for_location_code, one_d_candidate_location_code in zip(
                looking_for, candidate
            ):
                match = get_matching_levels(
                    one_d_looking_for_location_code, one_d_candidate_location_code
                )
                if match == -1:
                    return -1
                total += match
            return total

        total_matching_levels = functools.partial(
            get_total_matching_levels, location_code
        )
        results: dict[tuple[ba.bitarray, ...], int] = {}
        for key, result in zip(
            self._map.keys(),
            map(
                total_matching_levels,
                self._map.keys(),
            ),
        ):
            results[key] = result

        best_key = max(results, key=results.get, default=None)  # type: ignore
        if best_key is None or results[best_key] == -1:
            raise KeyError("No match found")
        # validation #TODO remove
        # count occurrences of best_value
        count = sum(1 for v in results.values() if v == results[best_key])
        if count > 1:
            raise ValueError("Multiple best matches found")
        return self._map[best_key]

    def __getitem__(self, location_code: Iterable[ba.bitarray]) -> int:
        return self.get_tightest_match(location_code)
