"""Metrics count actual per-run Refiner reservations and gate keep decisions.

`gate_skipped_groups` is the cumulative number of ``keep`` decisions recorded
by the regex gate. `defer` is scheduling, and cache reuse creates no new
decision. `refiner_model_calls` follows `_RefinerSessionStats.begin_call`, so
generation, strict retry, and separately issued local-review requests count;
score values produced inside one such request do not add another call.
"""
from pathlib import Path
from unittest.mock import patch

from system import web_app
from system.entity_store import EntityStore


class StubRefiner:
    calls = []

    def __init__(self, *args, **kwargs):
        type(self).calls.clear()

    def refine(self, text, **kwargs):
        type(self).calls.append((text, kwargs))
        return text, 1.0


def make_app(tmp_path):
    return web_app.create_app(
        Path('unused'), 'cpu', 'http://asr.invalid', None, 128, None,
        entity_db=tmp_path / 'entities.db', entity_fuzzy_mode='off',
        refinement_gate_mode='regex',
    )


def test_keep_count_is_cumulative_but_cache_reuse_does_not_increment(tmp_path):
    with patch.object(web_app, 'TransformersRefiner', StubRefiner):
        app = make_app(tmp_path)
        session = app.state.new_text_refinement_session('trade')
        text = ('请确认订单。请在周五发货。请附上装箱单。'
                '请保持包装完整。请及时回复。请确认收货地址。')

        session.update(text, revision=1, final=False)
        first = session.metrics_snapshot()
        assert first['gate_skipped_groups'] >= 1
        assert first['refiner_model_calls'] == 0

        session.update(text, revision=2, final=False)
        assert session.metrics_snapshot() == first


def test_defer_is_not_counted_then_final_flush_counts_keep(tmp_path):
    with patch.object(web_app, 'TransformersRefiner', StubRefiner):
        app = make_app(tmp_path)
        session = app.state.new_text_refinement_session('trade')
        text = '请按时发货。'

        pending = session.update(text, revision=1, final=False)
        assert pending['refinement_gate_decisions'][0]['action'] == 'defer'
        assert session.metrics_snapshot()['gate_skipped_groups'] == 0

        session.update(text, revision=2, final=True)
        assert session.metrics_snapshot()['gate_skipped_groups'] == 1
        assert not StubRefiner.calls


def test_real_generation_and_strict_retry_are_both_counted(tmp_path):
    class RetryRefiner(StubRefiner):
        def refine(self, text, **kwargs):
            type(self).calls.append((text, kwargs))
            # First response drops the protected entity token, forcing the
            # existing strict-placeholder retry. The second preserves it.
            if not kwargs.get('strict_placeholders'):
                return '请按时发货。', 1.0
            return text, 1.0

    with patch.object(web_app, 'TransformersRefiner', RetryRefiner):
        app = make_app(tmp_path)
        store = EntityStore(tmp_path / 'entities.db')
        store.upsert_entity(
            'iPhone', aliases=('iphone',), entity_type='PRODUCT',
            domain='trade', normalization_policy='normalize',
        )
        session = app.state.new_text_refinement_session('trade')
        session.update('请将iphone样品单独包装。请在周五发货。请附上装箱单。',
                       revision=1, final=True)

        metrics = session.metrics_snapshot()
        assert len(RetryRefiner.calls) >= 2
        assert metrics['refiner_model_calls'] == len(RetryRefiner.calls)
        assert metrics['refiner_retry_calls'] >= 1


def test_local_deletion_review_is_included_in_call_count(tmp_path):
    class ReviewRefiner(StubRefiner):
        reviews=[]
        def refine(self,text,**kwargs):
            type(self).calls.append((text,kwargs))
            return text.replace('嗯，','').replace('三百','五百'),1.0
        def review_repetition_candidate_with_status(self,*args):
            self.reviews.append(args)
            return 'KEEP',1.0,True
    with patch.object(web_app,'TransformersRefiner',ReviewRefiner):
        app=make_app(tmp_path)
        session=app.state.new_text_refinement_session('trade')
        result=session.update('嗯，我需要三百个阀门。请尽快发货。',revision=1,final=True)
        assert ReviewRefiner.reviews, result.get('local_deletion_reviews')
        assert session.metrics_snapshot()['refiner_model_calls']==len(ReviewRefiner.calls)+len(ReviewRefiner.reviews)
