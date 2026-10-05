import unittest

from vlmlab.data.imagenet21k import (DISTRACTOR_SYNSETS, LicenceError, MISCITED,
                                     SYNSETS, build_crop_plan, coverage_report,
                                     synsets_for_registry)
from vlmlab.data.public import (AVOID, PUBLIC_DATASETS, attribution_notice,
                                coverage_against_registry,
                                download_instructions, normalise_yolo_export,
                                resplit_by_session)
from vlmlab.data.rf100vl import (CLUSTERS, WITHDRAWN_PROXY, dry_run_plan,
                                 expectation_bracket, recommended_cluster)
from vlmlab.registry import Registry


class TestPublicDatasets(unittest.TestCase):
    def test_all_four_are_permissively_licensed(self):
        for spec in PUBLIC_DATASETS:
            self.assertIn("CC BY 4.0", spec.licence)

    def test_the_ambiguous_one_is_excluded(self):
        self.assertIn("labpics", AVOID)
        keys = {s.key for s in PUBLIC_DATASETS}
        self.assertNotIn("labpics", keys)

    def test_two_are_video_derived(self):
        self.assertEqual(2, sum(1 for s in PUBLIC_DATASETS if s.video_derived))

    def test_instance_total_is_substantial(self):
        total = sum(s.n_instances or 0 for s in PUBLIC_DATASETS)
        self.assertGreater(total, 15000)

    def test_download_instructions_warn_off_the_bad_one(self):
        text = download_instructions()
        self.assertIn("Do NOT fetch", text)
        self.assertIn("labpics", text)

    def test_weights_warning_present_for_the_tainted_set(self):
        spec = [s for s in PUBLIC_DATASETS if s.key == "heinsight4"][0]
        self.assertIn("NEVER THE RELEASED WEIGHTS", spec.notes)

    def test_saturation_warning_present(self):
        spec = [s for s in PUBLIC_DATASETS if s.key == "chemistry25"][0]
        self.assertIn("saturated", spec.notes)

    def test_attribution_notice_mentions_derivative_models(self):
        text = attribution_notice(["chemistry25"])
        self.assertIn("derivative models", text)
        self.assertIn("figshare", text)

    def test_empty_attribution(self):
        self.assertEqual("", attribution_notice([]))


class TestCoverage(unittest.TestCase):
    def test_the_six_unsupplied_classes_are_identified(self):
        cov = coverage_against_registry(Registry())
        for name in ("centrifuge", "autoclave", "spectrophotometer",
                     "orbital_shaker", "fume_hood", "microscope"):
            self.assertIn(name, cov["not_covered"])

    def test_glassware_is_covered(self):
        cov = coverage_against_registry(Registry())
        self.assertIn("beaker", cov["covered"])
        self.assertIn("conical_flask", cov["covered"])


class TestNormalisation(unittest.TestCase):
    def setUp(self):
        self.reg = Registry()
        self.recs = [
            {"file_name": "expt_A_000012.jpg", "width": 640, "height": 640,
             "boxes": [(0, 0.5, 0.5, 0.2, 0.2), (2, 0.2, 0.2, 0.1, 0.1)]},
            {"file_name": "expt_A_000048.jpg", "width": 640, "height": 640,
             "boxes": [(0, 0.5, 0.5, 0.2, 0.2)]},
            {"file_name": "expt_B_000003.rf.deadbeef.jpg", "width": 640,
             "height": 640, "boxes": [(1, 0.4, 0.4, 0.2, 0.2)]},
        ]
        self.names = ["Beaker", "Conical_Flask", "bicycle"]

    def test_maps_onto_the_vocabulary(self):
        rows, rep = normalise_yolo_export(self.recs, self.names, self.reg)
        self.assertEqual({"beaker": 2, "conical_flask": 1}, rep["mapped"])

    def test_unmapped_classes_are_counted_not_dropped_silently(self):
        _rows, rep = normalise_yolo_export(self.recs, self.names, self.reg)
        self.assertEqual({"bicycle": 1}, rep["unmapped"])

    def test_centre_form_is_converted_to_corners(self):
        rows, _rep = normalise_yolo_export(self.recs, self.names, self.reg)
        box = [r for r in rows if r["file_name"].endswith("000012.jpg")][0]["box"]
        self.assertAlmostEqual(256.0, box.x1)
        self.assertAlmostEqual(384.0, box.x2)

    def test_session_key_strips_frame_index_and_export_hash(self):
        rows, _rep = normalise_yolo_export(self.recs, self.names, self.reg)
        keys = sorted({r["session_key"] for r in rows})
        self.assertEqual(["expt_A", "expt_B"], keys)

    def test_high_unmapped_rate_warns(self):
        recs = [{"file_name": "a_0.jpg", "width": 100, "height": 100,
                 "boxes": [(0, 0.5, 0.5, 0.2, 0.2)]}]
        _rows, rep = normalise_yolo_export(recs, ["bicycle"], self.reg)
        self.assertTrue(rep["warnings"])

    def test_missing_dimensions_raise(self):
        recs = [{"file_name": "a_0.jpg", "boxes": []}]
        with self.assertRaises(ValueError):
            normalise_yolo_export(recs, self.names, self.reg)

    def test_resplit_groups_by_session_not_frame(self):
        rows, _rep = normalise_yolo_export(self.recs, self.names, self.reg)
        folds = resplit_by_session(rows, n_splits=2, seed=0)
        for train, val in folds:
            self.assertEqual(set(), {r["session_key"] for r in train} &
                             {r["session_key"] for r in val})


class TestImageNetCorpus(unittest.TestCase):
    def test_licence_gate_blocks_by_default(self):
        with self.assertRaises(LicenceError):
            build_crop_plan(Registry())

    def test_acknowledging_proceeds(self):
        plan = build_crop_plan(Registry(), acknowledge_non_commercial=True)
        self.assertGreater(plan["n_target_classes"], 5)
        self.assertIn("non-commercial", plan["licence"])

    def test_rescues_classes_with_no_public_boxes(self):
        reg = Registry()
        pub = coverage_against_registry(reg)["covered"]
        rep = coverage_report(reg, public_covered=pub)
        rescued = rep["rescued_from_zero_boxes"]
        for name in ("centrifuge", "autoclave", "spectrophotometer",
                     "microscope"):
            self.assertIn(name, rescued)
        self.assertGreater(rep["n_rescued_images"], 3000)

    def test_two_classes_remain_unsupplied(self):
        reg = Registry()
        pub = coverage_against_registry(reg)["covered"]
        rep = coverage_report(reg, public_covered=pub)
        self.assertIn("orbital_shaker", rep["still_unsupplied"])
        self.assertIn("fume_hood", rep["still_unsupplied"])

    def test_microscope_identifier_is_the_corrected_one(self):
        self.assertEqual("n03760671", SYNSETS["microscope"][0])

    def test_the_miscited_identifier_is_documented(self):
        self.assertIn("n03666591", MISCITED)
        self.assertIn("lighter", MISCITED["n03666591"])

    def test_recommends_equal_allocation(self):
        plan = build_crop_plan(Registry(), acknowledge_non_commercial=True)
        self.assertTrue(any("Equal allocation" in w for w in plan["warnings"]))
        self.assertGreater(plan["balanced_cap"], 0)

    def test_sparse_classes_warn(self):
        plan = build_crop_plan(Registry(), acknowledge_non_commercial=True,
                               min_images_per_class=200)
        self.assertTrue(any("below the 200 minimum" in w
                            for w in plan["warnings"]))

    def test_distractors_are_included_for_rejection(self):
        plan = build_crop_plan(Registry(), acknowledge_non_commercial=True)
        self.assertGreater(plan["n_distractor_classes"], 0)
        self.assertGreater(len(DISTRACTOR_SYNSETS), 0)


class TestDryRun(unittest.TestCase):
    def test_industrial_is_recommended(self):
        self.assertEqual("industrial", recommended_cluster())

    def test_the_medical_cluster_is_refused_as_a_laboratory_proxy(self):
        with self.assertRaises(ValueError) as ctx:
            dry_run_plan("medical")
        self.assertIn("mislabelled", str(ctx.exception))

    def test_the_withdrawn_proxy_is_recorded(self):
        self.assertIn("15.2", WITHDRAWN_PROXY)
        self.assertIn("withdrawn", WITHDRAWN_PROXY)

    def test_medical_cluster_notes_the_mislabelling(self):
        self.assertIn("MISLABELLED", CLUSTERS["medical"]["relevance"])

    def test_bracket_uses_two_clusters_not_one_number(self):
        b = expectation_bracket()
        self.assertEqual(2, len(b["zero_shot"]))
        self.assertIn("withdrawn", b)

    def test_plan_mirrors_the_real_pipeline(self):
        plan = dry_run_plan()
        self.assertEqual("Apache 2.0", plan["licence"])
        self.assertGreaterEqual(len(plan["stages"]), 5)
        self.assertIn("generic-prompt", plan["stages"][0])

    def test_unknown_cluster_raises(self):
        with self.assertRaises(KeyError):
            dry_run_plan("nonexistent")


if __name__ == "__main__":
    unittest.main()
