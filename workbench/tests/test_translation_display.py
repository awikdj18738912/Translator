"""Reconcile slow completed refinements with a newer, revisable translation."""
from system.translation_display import TranslationStreamDisplay


def translated(state, revision, text):
    return state.project({'event':'translation','revision':revision,'translation':text})


def refined(state, revision, raw, clean, final=False):
    return state.project({'event':'final' if final else 'update','revision':revision,
        'raw_text':raw,'clean_text':clean,'refinement_source_spans':[
            {'source_text':raw,'clean_text':clean,'state':'committed'}]})


def test_translation_is_immediately_visible_as_pending_before_refiner():
    state=TranslationStreamDisplay()
    event=translated(state,1,'新增译文。')
    assert event['clean_text']=='新增译文。'
    assert event['display_refined_text']==''
    assert event['pending_raw_text']=='新增译文。'
    assert event['has_pending_refinement']


def test_late_refinement_keeps_newest_translation_and_pending_tail():
    state=TranslationStreamDisplay()
    translated(state,1,'三百件。')
    translated(state,2,'三百件。周五发货。')
    event=refined(state,1,'三百件。','300件。')
    assert event['translation']=='三百件。周五发货。'
    assert event['clean_text']=='300件。周五发货。'
    assert event['display_refined_text']=='300件。'
    assert event['pending_raw_text']=='周五发货。'
    assert event['translation_revision']==2
    assert event['refinement_revision']==1


def test_changed_translation_discards_only_incompatible_refined_spans():
    state=TranslationStreamDisplay()
    translated(state,1,'三百件。')
    refined(state,1,'三百件。','300件。')
    event=translated(state,2,'五百件。')
    assert event['display_refined_text']==''
    assert event['pending_raw_text']=='五百件。'
    assert '300' not in event['clean_text']


def test_pending_group_is_not_marked_as_refined_and_final_clears_pending():
    state=TranslationStreamDisplay()
    translated(state,1,'三百件。明天发货。')
    event=state.project({'event':'update','revision':1,
        'raw_text':'三百件。明天发货。','clean_text':'300件。明天发货。',
        'refinement_source_spans':[
            {'source_text':'三百件。','clean_text':'300件。','state':'committed'},
            {'source_text':'明天发货。','clean_text':'明天发货。','state':'active'}]})
    assert event['display_refined_text']=='300件。'
    assert event['pending_raw_text']=='明天发货。'
    event=refined(state,1,'三百件。明天发货。','300件。明天发货。',final=True)
    assert not event['has_pending_refinement']
    assert event['display_refined_text']=='300件。明天发货。'
    assert event['pending_raw_text']==''
