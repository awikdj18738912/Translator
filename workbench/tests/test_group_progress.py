import threading
import unittest

from system.window_refinement import CumulativeWindowRefinement


def refined(text, *args, **kwargs):
    return {
        "raw_text": text, "clean_text": text.upper(),
        "refiner_latency_ms": 1, "refiner_accepted": True,
        "refiner_executed": True, "refiner_reject_reasons": [],
        "entity_audit_issues": [], "entity_refinement_hints": [],
        "entity_normalizations": [], "numeric_normalizations": [],
        "numeric_rule_repairs": [], "numeric_fallbacks": [],
        "safe_numeric_repairs": [], "safe_repetition_repairs": [],
        "local_deletion_reviews": [], "repetition_reviews": [],
        "protected_entities": [], "entity_candidates": [],
        "refiner_masked_outputs": [], "refiner_retry_reasons": [],
        "refinement_gate_decisions": [], "structured_patch_audits": [],
        "boundary_anomalies": [], "entity_matcher_latency_ms": 0,
    }


class GroupProgressTest(unittest.TestCase):
    def test_first_group_progress_precedes_blocked_second_and_tail_stays_active(self):
        second_started = threading.Event()
        release_second = threading.Event()
        progress = []

        def refine(text, *args, **kwargs):
            if text.startswith("四"):
                second_started.set()
                release_second.wait(2)
            return refined(text)

        session = CumulativeWindowRefinement(refine, fixed_groups=True,
            window_size=3, one_punctuation_window=True,
            on_group_progress=progress.append)
        result_holder = []
        worker = threading.Thread(target=lambda: result_holder.append(
            session.update("一。二。三。四。五。六。七。", "Chinese", True, None, None)
        ))
        worker.start()
        self.assertTrue(second_started.wait(1))
        self.assertEqual(len(progress), 1)
        event = progress[0]
        self.assertEqual(event["raw_text"], "一。二。三。四。五。六。七。")
        spans = event["refinement_source_spans"]
        self.assertEqual(spans[0]["state"], "committed")
        self.assertEqual(spans[-1]["state"], "active")
        self.assertIn("四。", spans[-1]["source_text"])
        self.assertTrue(event["refiner_executed"])
        release_second.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(result_holder), 1)
        self.assertEqual(result_holder[0]["refinement_state"], "window_refined")

    def test_cache_hit_does_not_emit_and_callback_failure_is_cleared(self):
        events = []
        session = CumulativeWindowRefinement(refined, fixed_groups=True,
            window_size=3, one_punctuation_window=True,
            on_group_progress=events.append)
        first = session.update("一。二。三。", "Chinese", True, None, None)
        self.assertEqual(len(events), 1)
        session.on_group_progress = events.append
        second = session.update("一。二。三。", "Chinese", True, None, None)
        self.assertEqual(len(events), 1)
        self.assertEqual(second["clean_text"], first["clean_text"])

        def fail(_result):
            raise RuntimeError("progress failed")

        session.on_group_progress = fail
        with self.assertRaisesRegex(RuntimeError, "progress failed"):
            session.update("四。五。六。", "Chinese", True, None, None)
        session.on_group_progress = None
        # A subsequent update works normally after the exceptional callback.
        completed = session.update("四。五。六。", "Chinese", True, None, None)
        self.assertEqual(completed["clean_text"], "四。五。六。".upper())


if __name__ == "__main__":
    unittest.main()
