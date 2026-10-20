from system.window_refinement import CumulativeWindowRefinement, StreamingRefinementDisplay
from system.translation_display import TranslationStreamDisplay


def test_multiline_translation_keeps_exact_committed_source_ownership():
    text='第一句。\n第二句，第三句。\n第四句。第五句。第六句。第七句。'
    parts=['第一句。第二句，第三句。','第四句。第五句。第六句。','第七句。']
    outputs=[{'clean_text':p.replace('第一','第1')} for p in parts]
    spans=CumulativeWindowRefinement._source_spans(text,parts,2,outputs)
    assert [s['state'] for s in spans]==['committed','committed','active']
    assert ''.join(s['source_text'] for s in spans)==text
    state=TranslationStreamDisplay()
    state.project({'event':'translation','revision':6,'translation':text})
    event=state.project({'event':'update','revision':6,'clean_text':text,
                         'refinement_source_spans':spans})
    assert event['display_refined_text'].startswith('第1句。')
    assert event['pending_raw_text']=='第七句。'
    assert event['display_revision']==6


def test_reformatted_context_matches_but_changed_words_do_not():
    display=StreamingRefinementDisplay()
    raw='第一句。\n'
    display.accept({'raw_text':raw,'clean_text':'第1句。','refinement_source_spans':[
        {'source_text':'第一句。\n','clean_text':'第1句。','state':'committed',
         'window_context':{'left_context':'','right_context':'第二句。第三句。'}}]},1)
    event=display.compose('第一句。\n第二句。\n第三句。')
    assert event['display_refined_text']=='第1句。'
    event=display.compose('第一句。\n另一个句子。第三句。')
    assert event['display_refined_text']==''


def test_non_whitespace_mapping_mismatch_remains_uncommitted():
    spans=CumulativeWindowRefinement._source_spans('第一句。不同内容。',['第一句。第二句。'],1,[{'clean_text':'危险替换'}])
    assert spans[0]['state']=='active'
