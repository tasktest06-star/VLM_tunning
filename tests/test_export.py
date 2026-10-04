import json
import os
import tempfile
import unittest

from vlmlab import federated
from vlmlab.export.coco import (build_categories, from_model_index,
                                to_model_index, write_ground_truth,
                                write_predictions)
from vlmlab.export.tao import EVAL_LIBRARY_PIN, EVAL_MAX_DETECTIONS, write_tao
from vlmlab.registry import Registry
from vlmlab.types import Box, BoxSource, ClipRecord, Track, TrackPoint


class TestCategories(unittest.TestCase):
    def test_one_indexed(self):
        cats = build_categories(["a", "b", "c"])
        self.assertEqual([1, 2, 3], [c["id"] for c in cats])

    def test_index_offset_round_trips(self):
        for cid in (1, 2, 15):
            self.assertEqual(cid, from_model_index(to_model_index(cid)))

    def test_first_category_maps_to_model_zero(self):
        self.assertEqual(0, to_model_index(1))


class TestCocoWriters(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.labels = ["beaker", "centrifuge"]
        self.images = [{"id": 1, "file_name": "a.jpg", "width": 640, "height": 480}]

    def test_ground_truth_has_no_score_field(self):
        """Mixing a score into ground truth makes a file that is neither."""
        path = os.path.join(self.dir, "gt.json")
        doc = write_ground_truth(path, self.images,
                                 [(1, "beaker", Box(10, 20, 110, 140), False)],
                                 self.labels)
        for ann in doc["annotations"]:
            self.assertNotIn("score", ann)

    def test_predictions_are_a_flat_list_with_scores(self):
        path = os.path.join(self.dir, "dt.json")
        rows = write_predictions(path, [(1, "beaker", Box(0, 0, 10, 10), 0.9)],
                                 self.labels)
        self.assertIsInstance(rows, list)
        self.assertIn("score", rows[0])

    def test_box_is_written_in_top_left_plus_size_form(self):
        path = os.path.join(self.dir, "gt.json")
        doc = write_ground_truth(path, self.images,
                                 [(1, "beaker", Box(10, 20, 110, 140), False)],
                                 self.labels)
        self.assertEqual([10.0, 20.0, 100.0, 120.0], doc["annotations"][0]["bbox"])
        self.assertEqual(12000.0, doc["annotations"][0]["area"])

    def test_ignore_becomes_crowd(self):
        path = os.path.join(self.dir, "gt.json")
        doc = write_ground_truth(path, self.images,
                                 [(1, "beaker", Box(0, 0, 10, 10), True)],
                                 self.labels)
        self.assertEqual(1, doc["annotations"][0]["iscrowd"])

    def test_unknown_label_raises(self):
        path = os.path.join(self.dir, "gt.json")
        with self.assertRaises(KeyError):
            write_ground_truth(path, self.images,
                               [(1, "nope", Box(0, 0, 10, 10), False)],
                               self.labels)

    def test_output_parses(self):
        path = os.path.join(self.dir, "gt.json")
        write_ground_truth(path, self.images,
                           [(1, "beaker", Box(0, 0, 10, 10), False)], self.labels)
        with open(path) as fh:
            json.load(fh)


class TestTaoWriter(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.reg = Registry()
        self.clip = ClipRecord("c1", "s1", ("beaker",), duration_s=180.0,
                               container_frame_count=5400)
        self.fed = federated.derive(self.clip, ["beaker", "centrifuge"],
                                    self.reg.names)

    def _write(self):
        path = os.path.join(self.dir, "tao.json")
        pts = (TrackPoint("f1", Box(0, 0, 10, 10), 0.9),
               TrackPoint("f2", Box(1, 1, 11, 11), 0.9))
        tracks = [Track(1, "beaker", pts, BoxSource.PROPAGATED, chunk_ids=(0, 1))]
        doc = write_tao(path, [self.clip],
                        [{"id": "f1", "file_name": "f1.jpg", "width": 640,
                          "height": 480}],
                        [("f1", "beaker", Box(0, 0, 10, 10), 1, False)],
                        tracks, list(self.reg.names),
                        {"c1": self.fed})
        return path, doc

    def test_negative_categories_are_emitted(self):
        """This is what makes an unqueried class ignored rather than a miss."""
        _path, doc = self._write()
        video = doc["videos"][0]
        self.assertIn("neg_category_ids", video)
        self.assertIn("not_exhaustive_category_ids", video)
        centrifuge_id = [c["id"] for c in doc["categories"]
                         if c["name"] == "centrifuge"][0]
        self.assertIn(centrifuge_id, video["neg_category_ids"])

    def test_a_clip_label_is_never_negative(self):
        _path, doc = self._write()
        beaker_id = [c["id"] for c in doc["categories"]
                     if c["name"] == "beaker"][0]
        self.assertNotIn(beaker_id, doc["videos"][0]["neg_category_ids"])

    def test_operational_notes_travel_with_the_file(self):
        _path, doc = self._write()
        self.assertEqual(EVAL_LIBRARY_PIN, doc["info"]["evaluation_library"])
        self.assertEqual(EVAL_MAX_DETECTIONS,
                         doc["info"]["max_detections_per_image"])
        self.assertIn("1.3.0", EVAL_LIBRARY_PIN)
        self.assertEqual(0, EVAL_MAX_DETECTIONS)

    def test_cross_chunk_tracks_are_marked(self):
        _path, doc = self._write()
        self.assertTrue(doc["tracks"][0]["spans_chunks"])

    def test_session_metadata_is_kept(self):
        _path, doc = self._write()
        self.assertEqual("s1", doc["videos"][0]["metadata"]["session_id"])

    def test_output_parses(self):
        path, _doc = self._write()
        with open(path) as fh:
            json.load(fh)


if __name__ == "__main__":
    unittest.main()
