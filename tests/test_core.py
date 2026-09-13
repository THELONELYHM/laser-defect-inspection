from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from laser_inspection.constants import normalize_class_name
from laser_inspection.models import Detection
from laser_inspection.preprocess import letterbox, parse_roi, restore_xyxy
from prepare_dataset import parse_voc


class PreprocessTests(unittest.TestCase):
    def test_roi_is_clamped_to_image(self) -> None:
        roi = parse_roi("90,80,30,40", (100, 100, 3))
        self.assertEqual(roi, (90, 80, 10, 20))

    def test_letterbox_round_trip(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        output, scale, padding = letterbox(image, (640, 640))
        self.assertEqual(output.shape, (640, 640, 3))
        original_box = (20.0, 10.0, 180.0, 90.0)
        pad_x, pad_y = padding
        model_box = (
            original_box[0] * scale + pad_x,
            original_box[1] * scale + pad_y,
            original_box[2] * scale + pad_x,
            original_box[3] * scale + pad_y,
        )
        restored = restore_xyxy(model_box, scale, padding, image.shape)
        np.testing.assert_allclose(restored, original_box, atol=0.01)


class ModelTests(unittest.TestCase):
    def test_physical_measurement(self) -> None:
        detection = Detection(5, "scratches", 0.9, 10, 20, 30, 50)
        payload = detection.to_dict(mm_per_pixel=0.1)
        self.assertEqual(payload["width_mm"], 2.0)
        self.assertEqual(payload["height_mm"], 3.0)
        self.assertEqual(payload["area_mm2"], 6.0)

    def test_class_alias(self) -> None:
        self.assertEqual(normalize_class_name("pitted-surface"), "pitted_surface")
        self.assertEqual(normalize_class_name("rolled_in_scale"), "rolled-in_scale")


class DatasetTests(unittest.TestCase):
    def test_voc_to_yolo_conversion(self) -> None:
        xml = """<annotation>
        <size><width>200</width><height>100</height><depth>1</depth></size>
        <object><name>scratches</name><bndbox>
        <xmin>20</xmin><ymin>10</ymin><xmax>100</xmax><ymax>50</ymax>
        </bndbox></object></annotation>"""
        with tempfile.TemporaryDirectory() as temporary:
            xml_path = Path(temporary) / "sample.xml"
            xml_path.write_text(xml, encoding="utf-8")
            width, height, labels = parse_voc(xml_path)
        self.assertEqual((width, height), (200, 100))
        self.assertEqual(labels[0][0], 5)
        np.testing.assert_allclose(labels[0][1:], (0.3, 0.3, 0.4, 0.4))


if __name__ == "__main__":
    unittest.main()

