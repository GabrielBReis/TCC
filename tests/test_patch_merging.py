from scripts.merge_patch_predictions import merge_predictions


def test_patch_coordinates_are_projected_and_duplicates_suppressed():
    patch_coco = {
        "images": [
            {
                "id": 1,
                "file_name": "a.jpg",
                "width": 64,
                "height": 64,
                "source_image_id": 10,
                "patch_x": 0,
                "patch_y": 0,
            },
            {
                "id": 2,
                "file_name": "b.jpg",
                "width": 64,
                "height": 64,
                "source_image_id": 10,
                "patch_x": 32,
                "patch_y": 0,
            },
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "defect"}],
    }
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [40, 10, 10, 10], "score": 0.9},
        {"image_id": 2, "category_id": 1, "bbox": [8, 10, 10, 10], "score": 0.8},
    ]
    merged = merge_predictions(patch_coco, predictions, iou_threshold=0.5)
    assert len(merged) == 1
    assert merged[0]["image_id"] == 10
    assert merged[0]["bbox"] == [40.0, 10.0, 10.0, 10.0]
