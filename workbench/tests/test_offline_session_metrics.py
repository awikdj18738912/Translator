import unittest

from system.offline_text_refinement import OfflineTextRefinementSession
from system.web_app import _RefinerSessionStats


class OfflineSessionMetricsTest(unittest.TestCase):
    def test_callback_calls_and_gate_skips_are_cumulative(self):
        stats = _RefinerSessionStats()

        def refine(*args, **kwargs):
            call_stats = kwargs['call_stats']
            started = call_stats.begin_call(final=False, retry=False)
            call_stats.finish_call(started, failed=False)
            started = call_stats.begin_call(final=False, retry=True)
            call_stats.finish_call(started, failed=False)
            call_stats.record_gate_skip()
            call_stats.record_gate_skip()
            return {'model_calls': [{'kind': 'generation'}, {'kind': 'retry'}]}

        session = OfflineTextRefinementSession(
            refine, (), call_stats=stats,
        )
        session.windows.refine('example')

        self.assertEqual(session.metrics_snapshot(), {
            'refiner_model_calls': 2,
            'gate_skipped_groups': 2,
            'refiner_retry_calls': 1,
        })

    def test_metrics_work_without_stats_for_legacy_adapters(self):
        session = OfflineTextRefinementSession(
            lambda *args, **kwargs: {}, (),
        )
        self.assertEqual(session.metrics_snapshot(), {
            'refiner_model_calls': 0,
            'gate_skipped_groups': 0,
        })


if __name__ == '__main__':
    unittest.main()
