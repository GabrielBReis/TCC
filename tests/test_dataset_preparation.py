from scripts.prepare_innovation_hangar import assign_groups, source_group


def test_roboflow_variants_share_the_same_source_group():
    first = "sample_jpg.rf.0123456789abcdef0123456789abcdef.jpg"
    second = "sample_jpg.rf.abcdef0123456789abcdef0123456789.jpg"
    assert source_group(first) == source_group(second) == "sample"


def test_grouped_split_never_separates_source_variants():
    records = [
        {"group": "same", "annotations": [{"category_id": 1}]},
        {"group": "same", "annotations": [{"category_id": 1}]},
        {"group": "other-a", "annotations": [{"category_id": 2}]},
        {"group": "other-b", "annotations": [{"category_id": 2}]},
    ]
    assigned, group_count = assign_groups(records, seed=42, ratios={"train": 0.5, "val": 0.25, "test": 0.25})
    destinations = [split for split, items in assigned.items() if any(item["group"] == "same" for item in items)]
    assert group_count == 3
    assert len(destinations) == 1
    assert sum(item["group"] == "same" for item in assigned[destinations[0]]) == 2
