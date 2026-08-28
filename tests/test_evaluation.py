from scripts.evaluate import _match_class, detection_confusion_matrix, prf


def test_detection_confusion_matrix_includes_background_fp_and_fn():
    gt = {
        "images": [{"id": 1}],
        "categories": [{"id": 1, "name": "crack"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}],
    }
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [50, 50, 5, 5], "score": 0.8},
    ]
    matrix, labels = detection_confusion_matrix(gt, predictions, confidence=0.25, iou_threshold=0.5)
    assert labels == ["crack", "background"]
    assert matrix.tolist() == [[1, 0], [1, 0]]


def sample_coco():
    return {
        "images": [{"id": 1, "file_name": "image.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "defect"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 5, 5], "area": 25, "iscrowd": 0}],
    }


def test_small_gt_matches_prediction_whose_area_crosses_threshold():
    gt = sample_coco()
    images = {1: gt["images"][0]}
    predictions = [{"image_id": 1, "category_id": 1, "bbox": [9, 9, 10, 10], "score": 0.9}]
    result = prf(gt, predictions, images, 0.25, 0.20, 0.01, 0.05, wanted_bin="small")
    assert result == {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_out_of_range_unmatched_prediction_is_ignored():
    gt = sample_coco()
    images = {1: gt["images"][0]}
    predictions = [{"image_id": 1, "category_id": 1, "bbox": [50, 50, 30, 30], "score": 0.9}]
    records, false_negatives, total_gt = _match_class(gt, predictions, images, 0.0, 0.5, 0.01, 0.05, "small", 1)
    assert records == [(0.9, False, False)]
    assert (false_negatives, total_gt) == (1, 1)
