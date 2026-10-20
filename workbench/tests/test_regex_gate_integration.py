from pathlib import Path
from unittest.mock import patch
from system import web_app
from system.entity_store import EntityStore


class Refiner:
    calls=[]
    def __init__(self,*args,**kwargs):self.calls.clear()
    def refine(self,text,**kwargs):
        self.calls.append(text)
        return text,1.0


def make_app(tmp_path,mode='regex'):
    return web_app.create_app(Path('unused'), 'cpu','http://asr.invalid',None,128,None,
        entity_db=tmp_path/'entities.db',entity_fuzzy_mode='off',refinement_gate_mode=mode)


def test_clean_final_skips_model_and_entity_hints_keep_normalization(tmp_path):
    store=EntityStore(tmp_path/'entities.db')
    store.upsert_entity('iPhone',aliases=('iphone',),entity_type='PRODUCT',domain='trade',normalization_policy='normalize')
    with patch.object(web_app,'TransformersRefiner',Refiner):
        app=make_app(tmp_path)
        session=app.state.new_text_refinement_session('trade')
        clean=app.state.new_text_refinement_session('trade').update('请确认订单。请在周五发货。请附上装箱单。',revision=1,final=True)
        assert not Refiner.calls
        assert clean['refinement_gate_skipped_segments']>0
        result=session.update('请将iphone样品单独包装。请在周五发货。请附上装箱单。',revision=1,final=True)
        assert Refiner.calls
        assert 'iPhone' in result['clean_text']
        assert any(d['reasons']==['entity_hint_present'] for d in result['refinement_gate_decisions'])
        assert all(d['mode']=='regex' for d in result['refinement_gate_decisions'])
        assert all(d['asr_confidence'] is None for d in result['refinement_gate_decisions'])


def test_regex_signal_runs_model_and_guard_stays_enabled(tmp_path):
    class WrongNumber(Refiner):
        def refine(self,text,**kwargs):
            self.calls.append(text)
            return text.replace('三百','五百'),1.0
    with patch.object(web_app,'TransformersRefiner',WrongNumber):
        app=make_app(tmp_path)
        result=app.state.new_text_refinement_session('trade').update(
            '我们需要三百件。请在周五发货。请附上装箱单。',revision=1,final=True)
        assert WrongNumber.calls
        assert result['refiner_executed']
        assert not result['refiner_accepted']
        assert ('三百' in result['clean_text'] or '300' in result['clean_text'])
        assert '五百' not in result['clean_text'] and '500' not in result['clean_text']


def test_pending_group_is_deferred_instead_of_counted_as_gate_skip(tmp_path):
    with patch.object(web_app,'TransformersRefiner',Refiner):
        app=make_app(tmp_path)
        result=app.state.new_text_refinement_session('trade').update('请按时发货。',revision=1,final=False)
        assert not Refiner.calls
        assert result['refinement_gate_skipped_segments']==0
        assert result['refinement_gate_decisions'][0]['action']=='defer'
