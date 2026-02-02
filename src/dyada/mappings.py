# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later


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
