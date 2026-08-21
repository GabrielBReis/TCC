import pytest

from tcc_pipeline.geometry import box_iou_xywh, relative_area, size_bin


def test_geometry_helpers():
    assert box_iou_xywh([0, 0, 10, 10], [5, 5, 10, 10]) == pytest.approx(25 / 175)
    assert relative_area([0, 0, 10, 5], 100, 100) == pytest.approx(0.005)
    assert size_bin(0.005, 0.01, 0.05) == "small"
    assert size_bin(0.02, 0.01, 0.05) == "medium"
    assert size_bin(0.10, 0.01, 0.05) == "large"
