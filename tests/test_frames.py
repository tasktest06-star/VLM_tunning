import unittest

from vlmlab.frames import (ResolutionError, UnderDeliveredError, ZeroFrameCountError,
                           chunk_indices, estimate_vision_tokens, keyframe_indices,
                           plan_frame_grid, validate_resolution)


class TestZeroFrameCollapse(unittest.TestCase):
    """The worse half of the frame-sampling trap.

    The alternative implementation returns the first ``cap`` indices when a
    container reports no frames, which silently samples only the opening
    moment of every clip. Routine for phone recordings and remuxes.
    """

    def test_zero_count_raises(self):
        with self.assertRaises(ZeroFrameCountError):
            plan_frame_grid("c", 180.0, 0, 2.0, 400)

    def test_none_count_raises(self):
        with self.assertRaises(ZeroFrameCountError):
            plan_frame_grid("c", 180.0, None, 2.0, 400)

    def test_negative_count_raises(self):
        with self.assertRaises(ZeroFrameCountError):
            plan_frame_grid("c", 180.0, -1, 2.0, 400)

    def test_message_explains_the_risk(self):
        try:
            plan_frame_grid("clipX", 180.0, 0, 2.0, 16)
        except ZeroFrameCountError as exc:
            self.assertIn("clipX", str(exc))
            self.assertIn("start of the clip", str(exc))


class TestUnderDelivery(unittest.TestCase):
    def test_cap_not_bound_is_flagged(self):
        g = plan_frame_grid("c", 6.0, 180, 1.0, 16)
        self.assertEqual(6, g.n_frames)
        self.assertFalse(g.cap_bound)

    def test_can_be_made_fatal(self):
        with self.assertRaises(UnderDeliveredError):
            plan_frame_grid("c", 6.0, 180, 1.0, 16, require_cap_bound=True)

    def test_cap_bound_when_rate_delivers(self):
        g = plan_frame_grid("c", 180.0, 5400, 2.0, 360)
        self.assertTrue(g.cap_bound)


class TestGrid(unittest.TestCase):
    def test_three_minute_clip_at_two_fps(self):
        g = plan_frame_grid("c", 180.0, 5400, 2.0, 400)
        self.assertEqual(360, g.n_frames)

    def test_indices_are_increasing_and_in_range(self):
        g = plan_frame_grid("c", 180.0, 5400, 2.0, 400)
        self.assertEqual(list(g.indices), sorted(g.indices))
        self.assertGreaterEqual(g.indices[0], 0)
        self.assertLess(g.indices[-1], 5400)

    def test_single_frame(self):
        g = plan_frame_grid("c", 1.0, 30, 1.0, 1)
        self.assertEqual((0,), g.indices)

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            plan_frame_grid("c", 0.0, 100, 1.0, 10)
        with self.assertRaises(ValueError):
            plan_frame_grid("c", 10.0, 100, 0.0, 10)
        with self.assertRaises(ValueError):
            plan_frame_grid("c", 10.0, 100, 1.0, 0)


class TestChunking(unittest.TestCase):
    """At three minutes the documented duration limit forces at least six chunks."""

    def test_duration_limit_forces_six_or_more(self):
        g = plan_frame_grid("c", 180.0, 5400, 2.0, 400)
        chunks = chunk_indices(g, 300, overlap=0, clip_duration_s=180.0,
                               pcs_max_seconds=30.0)
        self.assertGreaterEqual(len(chunks), 6)

    def test_frame_cap_alone_would_give_fewer(self):
        g = plan_frame_grid("c", 180.0, 5400, 2.0, 400)
        unlimited = chunk_indices(g, 300, overlap=0)
        limited = chunk_indices(g, 300, overlap=0, clip_duration_s=180.0,
                                pcs_max_seconds=30.0)
        self.assertLess(len(unlimited), len(limited))

    def test_chunks_cover_every_frame(self):
        g = plan_frame_grid("c", 180.0, 5400, 2.0, 400)
        chunks = chunk_indices(g, 300, overlap=8, clip_duration_s=180.0,
                               pcs_max_seconds=30.0)
        covered = set()
        for ch in chunks:
            covered.update(ch.indices)
        self.assertEqual(set(g.indices), covered)

    def test_overlap_is_recorded(self):
        g = plan_frame_grid("c", 180.0, 5400, 2.0, 400)
        chunks = chunk_indices(g, 60, overlap=8)
        self.assertEqual(0, chunks[0].overlap_with_previous)
        self.assertEqual(8, chunks[1].overlap_with_previous)

    def test_consecutive_chunks_share_frames_when_overlapping(self):
        g = plan_frame_grid("c", 100.0, 3000, 2.0, 200)
        chunks = chunk_indices(g, 50, overlap=5)
        a = set(chunks[0].indices)
        b = set(chunks[1].indices)
        self.assertEqual(5, len(a & b))

    def test_bad_overlap_raises(self):
        g = plan_frame_grid("c", 10.0, 300, 2.0, 20)
        with self.assertRaises(ValueError):
            chunk_indices(g, 10, overlap=10)
        with self.assertRaises(ValueError):
            chunk_indices(g, 10, overlap=-1)


class TestKeyframes(unittest.TestCase):
    def test_stride(self):
        g = plan_frame_grid("c", 180.0, 5400, 2.0, 400)
        kf = keyframe_indices(g, 8)
        self.assertEqual(45, len(kf))
        self.assertEqual(g.indices[0], kf[0])

    def test_stride_one_is_every_frame(self):
        g = plan_frame_grid("c", 10.0, 300, 2.0, 20)
        self.assertEqual(g.indices, keyframe_indices(g, 1))

    def test_bad_stride_raises(self):
        g = plan_frame_grid("c", 10.0, 300, 2.0, 20)
        with self.assertRaises(ValueError):
            keyframe_indices(g, 0)


class TestResolution(unittest.TestCase):
    """The other half of the resolution trap: wrong in both directions."""

    def test_thumbnails_refused(self):
        with self.assertRaises(ResolutionError) as ctx:
            validate_resolution(128, 8)
        self.assertIn("thumbnail", str(ctx.exception))

    def test_token_budget_enforced(self):
        with self.assertRaises(ResolutionError) as ctx:
            validate_resolution(1008, 16)
        self.assertIn("vision tokens", str(ctx.exception))

    def test_a_workable_configuration_passes(self):
        tokens = validate_resolution(448, 16)
        self.assertEqual(4096, tokens)

    def test_token_estimate_scales_with_frames(self):
        self.assertEqual(2 * estimate_vision_tokens(448, 1),
                         estimate_vision_tokens(448, 2))


if __name__ == "__main__":
    unittest.main()
