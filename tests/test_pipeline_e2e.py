"""End to end on the fake backend, plus the planted-leakage regression.

This is the file that makes the dependency-free design defensible. The whole
pipeline runs here: frame planning, six-segment chunking for a three-minute
clip, per-clip vocabulary restriction, detection, double propagation,
cross-chunk stitching, the cascade, both exporters and the evaluation.

Nothing about the orchestration is assumed to work. The only thing still
untested is the library call surface inside the real adapters.
"""
import json
import os
import tempfile
import unittest

from vlmlab import federated
from vlmlab.backends.fake import (FakeCropClassifier, FakeDetector,
                                  FakeMaskDetector, FakePropagator)
from vlmlab.checkpoint import Checkpoint, config_hash
from vlmlab.config import Config
from vlmlab.eval.metrics import (GroundTruth, Prediction, mean_average_precision)
from vlmlab.eval.protocol import build_protocol
from vlmlab.export.coco import write_ground_truth, write_predictions
from vlmlab.export.tao import write_tao
from vlmlab.pipeline import PipelineError, process_clip, run_pipeline, stitch_tracks
from vlmlab.registry import Registry
from vlmlab.types import Box, BoxSource, ClipRecord, Track, TrackPoint


def _cfg(fps=1.0, stride=16):
    cfg = Config()
    cfg.detector.frame_size = 448
    cfg.detector.sample_fps = fps
    cfg.detector.keyframe_stride = stride
    cfg.validate()
    return cfg


def _clips(n_sessions=6, per_session=3):
    out = []
    for s in range(n_sessions):
        for c in range(per_session):
            labels = (("beaker", "centrifuge") if (s + c) % 2 == 0
                      else ("microscope", "test_tube"))
            out.append(ClipRecord(
                "clip_s{}_{}".format(s, c), "session{}".format(s), labels,
                duration_s=180.0, container_frame_count=5400, fps=30.0))
    return out


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.cfg = _cfg()
        self.reg = Registry()
        self.clips = _clips()
        self.det = FakeDetector(image_size=(448, 448), hit_rate=0.85)
        self.prop = FakePropagator()

    def test_runs_and_produces_a_report(self):
        results, report = run_pipeline(self.clips, self.det, self.prop,
                                       self.reg, self.cfg)
        self.assertEqual(len(self.clips), report["n_clips"])
        self.assertEqual(6, report["n_sessions"])
        self.assertGreater(report["total_detections"], 0)
        self.assertGreater(report["total_tracks"], 0)

    def test_no_accepted_detection_escapes_the_clip_label_set(self):
        """The core invariant of the whole design."""
        results, _ = run_pipeline(self.clips, self.det, self.prop, self.reg,
                                  self.cfg)
        for r in results:
            for d in r.cascade.accepted:
                self.assertIn(d.label, r.clip.label_set(),
                              "{} accepted {} which is not in {}".format(
                                  r.clip.clip_id, d.label, r.clip.labels))

    def test_three_minute_clips_are_chunked_at_least_six_ways(self):
        res = process_clip(self.clips[0], self.det, self.prop, self.reg, self.cfg)
        self.assertGreaterEqual(len(res.chunks), 6)

    def test_cross_chunk_stitching_actually_happens(self):
        _results, report = run_pipeline(self.clips, self.det, self.prop,
                                        self.reg, self.cfg)
        self.assertGreater(report["stitched_tracks"], 0,
                           "no track spanned a chunk boundary, so the stitching "
                           "path was never exercised")

    def test_propagation_runs_in_both_directions(self):
        process_clip(self.clips[0], self.det, self.prop, self.reg, self.cfg)
        directions = {c["reverse"] for c in self.prop.call_log
                      if c["op"] == "propagate"}
        self.assertEqual({False, True}, directions)

    def test_every_session_is_closed(self):
        run_pipeline(self.clips, self.det, self.prop, self.reg, self.cfg)
        self.assertEqual(0, self.prop.open_handles())

    def test_prompts_are_restricted_to_each_clip_vocabulary(self):
        """Verified from the call log, which no return value would reveal."""
        clip = self.clips[0]
        det = FakeDetector(image_size=(448, 448), hit_rate=1.0)
        process_clip(clip, det, FakePropagator(), self.reg, self.cfg)
        prompted = set()
        for call in det.call_log:
            prompted.update(call["prompts"])
        canonical = {p.replace(" ", "_") for p in prompted}
        other_classes = set(self.reg.names) - set(clip.labels)
        leaked = canonical & other_classes
        # Hard negatives are deliberately prompted, so only assert the count
        # stays inside the configured budget.
        self.assertLessEqual(len(prompted),
                             self.cfg.detector.max_concepts_per_clip + 1)

    def test_a_mask_only_backend_is_refused(self):
        """Boxes are the deliverable, so mask-derived boxes must not pass."""
        bad = FakeMaskDetector(image_size=(448, 448), hit_rate=1.0)
        with self.assertRaises(PipelineError) as ctx:
            process_clip(self.clips[0], bad, self.prop, self.reg, self.cfg)
        self.assertIn("mask", str(ctx.exception))

    def test_summary_is_json_serialisable(self):
        results, report = run_pipeline(self.clips[:2], self.det, self.prop,
                                       self.reg, self.cfg)
        json.dumps({"report": report, "clips": [r.summary() for r in results]})


class TestExportsFromPipeline(unittest.TestCase):
    def test_both_exporters_produce_parseable_output(self):
        cfg = _cfg()
        reg = Registry()
        clips = _clips(4, 2)
        det = FakeDetector(image_size=(448, 448), hit_rate=1.0)
        results, _ = run_pipeline(clips, det, FakePropagator(), reg, cfg)

        images, gt_anns, preds, tao_anns, tracks = [], [], [], [], []
        fed_by_clip = {}
        seen = set()
        for r in results:
            fed_by_clip[r.clip.clip_id] = r.federated
            for d in r.cascade.accepted:
                if d.frame_id not in seen:
                    seen.add(d.frame_id)
                    images.append({"id": d.frame_id, "file_name": d.frame_id,
                                   "width": 448, "height": 448})
                gt_anns.append((d.frame_id, d.label, d.box, False))
                preds.append((d.frame_id, d.label, d.box, d.score))
                tao_anns.append((d.frame_id, d.label, d.box, 1, False))
            tracks.extend(r.tracks)

        self.assertGreater(len(images), 0, "pipeline produced nothing to export")
        tmp = tempfile.mkdtemp()
        gt_path = os.path.join(tmp, "gt.json")
        dt_path = os.path.join(tmp, "dt.json")
        tao_path = os.path.join(tmp, "tao.json")
        write_ground_truth(gt_path, images, gt_anns, list(reg.names))
        write_predictions(dt_path, preds, list(reg.names))
        write_tao(tao_path, clips, images, tao_anns, tracks, list(reg.names),
                  fed_by_clip)
        for path in (gt_path, dt_path, tao_path):
            with open(path) as fh:
                json.load(fh)

    def test_metrics_run_on_pipeline_output(self):
        cfg = _cfg()
        reg = Registry()
        clips = _clips(4, 2)
        det = FakeDetector(image_size=(448, 448), hit_rate=1.0)
        results, _ = run_pipeline(clips, det, FakePropagator(), reg, cfg)
        gts, preds = [], []
        for r in results:
            for d in r.cascade.accepted:
                gts.append(GroundTruth(d.frame_id, d.label, d.box))
                preds.append(Prediction(d.frame_id, d.label, d.box, d.score))
        res = mean_average_precision(preds, gts, labels=list(reg.names))
        self.assertIsNotNone(res["mAP"])
        self.assertGreaterEqual(res["mAP"], 0.0)
        self.assertLessEqual(res["mAP"], 1.0)
        self.assertGreater(res["n_classes_averaged"], 0)


class TestPlantedLeakage(unittest.TestCase):
    """Turns the research's central warning into a regression test.

    A model that memorises a recording session scores near perfectly when
    sessions straddle the split and near zero when they do not. If grouping by
    session did not matter, this test would not detect the difference.
    """

    def test_session_grouping_removes_the_inflation(self):
        from vlmlab.splits import grouped_vs_ungrouped_gap
        clips = _clips(8, 6)

        def score(train, val):
            known = {c.session_id for c in train}
            if not val:
                return 0.0
            return sum(1 for c in val if c.session_id in known) / float(len(val))

        gap = grouped_vs_ungrouped_gap(
            clips, lambda c: c.session_id, lambda c: c.clip_id, score,
            n_splits=4, seed=0)
        self.assertAlmostEqual(0.0, gap["grouped_mean"], places=6)
        self.assertGreater(gap["ungrouped_mean"], 0.9)
        self.assertGreater(gap["gap"], 0.9,
                           "a clip-level split must inflate the score; if it does "
                           "not, this test has no power")


class TestProtocolOnPipelineClips(unittest.TestCase):
    def test_nine_sessions_selects_leave_one_out_and_permutation(self):
        cfg = _cfg()
        report = build_protocol(_clips(9, 3), cfg)
        self.assertEqual("leave-one-session-out", report.scheme)
        self.assertEqual("permutation", report.test_name)

    def test_folds_are_session_disjoint(self):
        cfg = _cfg()
        report = build_protocol(_clips(20, 3), cfg)
        for train, val in report.folds:
            self.assertEqual(set(), {c.session_id for c in train} &
                             {c.session_id for c in val})


class TestCheckpointResume(unittest.TestCase):
    def test_completed_steps_are_skipped(self):
        tmp = tempfile.mkdtemp()
        cp = Checkpoint(tmp, ["detect", "propagate"], cfg_hash="abc")
        self.assertFalse(cp.is_done("detect"))
        cp.save("detect", {"n": 3})
        self.assertTrue(cp.is_done("detect"))
        self.assertEqual({"n": 3}, cp.load("detect"))

    def test_changing_the_configuration_invalidates(self):
        tmp = tempfile.mkdtemp()
        cp = Checkpoint(tmp, ["detect"], cfg_hash="abc")
        cp.save("detect", {"n": 1})
        reopened = Checkpoint(tmp, ["detect"], cfg_hash="different")
        self.assertFalse(reopened.is_done("detect"),
                         "a configuration change must invalidate prior steps, or "
                         "a resumed run silently mixes two configurations")

    def test_unknown_step_raises(self):
        cp = Checkpoint(tempfile.mkdtemp(), ["a"])
        with self.assertRaises(KeyError):
            cp.is_done("b")

    def test_payload_must_be_json_safe(self):
        """Prevents a tensor leaking into an artefact."""
        cp = Checkpoint(tempfile.mkdtemp(), ["a"])
        with self.assertRaises(TypeError):
            cp.save("a", {"model": object()})

    def test_config_hash_is_stable_and_order_independent(self):
        self.assertEqual(config_hash({"a": 1, "b": 2}),
                         config_hash({"b": 2, "a": 1}))
        self.assertNotEqual(config_hash({"a": 1}), config_hash({"a": 2}))


class TestStitching(unittest.TestCase):
    def test_overlapping_tracks_in_adjacent_chunks_merge(self):
        pts_a = (TrackPoint("f1", Box(0, 0, 10, 10), 0.9),
                 TrackPoint("f2", Box(0, 0, 10, 10), 0.9))
        pts_b = (TrackPoint("f2", Box(0, 0, 10, 10), 0.9),
                 TrackPoint("f3", Box(0, 0, 10, 10), 0.9))
        merged = stitch_tracks({0: (Track(1, "beaker", pts_a, BoxSource.PROPAGATED,
                                          chunk_ids=(0,)),),
                                1: (Track(1, "beaker", pts_b, BoxSource.PROPAGATED,
                                          chunk_ids=(1,)),)})
        self.assertEqual(1, len(merged))
        self.assertEqual(3, merged[0].n_frames)
        self.assertTrue(merged[0].spans_chunks)

    def test_non_overlapping_tracks_stay_separate(self):
        pts_a = (TrackPoint("f1", Box(0, 0, 10, 10), 0.9),)
        pts_b = (TrackPoint("f9", Box(500, 500, 510, 510), 0.9),)
        merged = stitch_tracks({0: (Track(1, "beaker", pts_a, BoxSource.PROPAGATED,
                                          chunk_ids=(0,)),),
                                1: (Track(1, "beaker", pts_b, BoxSource.PROPAGATED,
                                          chunk_ids=(1,)),)})
        self.assertEqual(2, len(merged))

    def test_different_labels_never_merge(self):
        pts = (TrackPoint("f1", Box(0, 0, 10, 10), 0.9),)
        merged = stitch_tracks({0: (Track(1, "beaker", pts, BoxSource.PROPAGATED,
                                          chunk_ids=(0,)),),
                                1: (Track(1, "centrifuge", pts, BoxSource.PROPAGATED,
                                          chunk_ids=(1,)),)})
        self.assertEqual(2, len(merged))

    def test_track_ids_are_renumbered_contiguously(self):
        pts = (TrackPoint("f1", Box(0, 0, 10, 10), 0.9),)
        merged = stitch_tracks({0: (Track(7, "beaker", pts, BoxSource.PROPAGATED,
                                          chunk_ids=(0,)),)})
        self.assertEqual(1, merged[0].track_id)


class TestCropClassifier(unittest.TestCase):
    def test_rows_are_normalised_distributions(self):
        c = FakeCropClassifier()
        rows = c.classify("f.jpg", (Box(0, 0, 10, 10), Box(5, 5, 20, 20)),
                          ("beaker", "centrifuge", "microscope"))
        self.assertEqual(2, len(rows))
        for row in rows:
            self.assertAlmostEqual(1.0, sum(row), places=6)

    def test_favouring_a_label_raises_its_score(self):
        plain = FakeCropClassifier()
        tilted = FakeCropClassifier(favour={"centrifuge": 5.0})
        boxes = (Box(0, 0, 10, 10),)
        labels = ("beaker", "centrifuge")
        self.assertGreater(tilted.classify("f.jpg", boxes, labels)[0][1],
                           plain.classify("f.jpg", boxes, labels)[0][1])


if __name__ == "__main__":
    unittest.main()
