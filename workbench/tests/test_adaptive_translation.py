from system.adaptive_translation import AdaptiveTranslationSnapshots,context_windows
from system.translation_windows import _sentence_data


def test_short_neighbors_share_context_and_only_completed_groups_confirm():
    quality=[];preview=[]
    def q(source,lang):quality.append(source);return '确认译文。'
    def p(source,lang,on_text):preview.append(source);on_text('预览');return '预览'
    state=AdaptiveTranslationSnapshots(q,p)
    state.update('My name is Tasha Starski.','en',lambda text:None)
    assert not quality
    state.update('My name is Tasha Starski. Please confirm it. Thank you.','en',lambda text:None)
    assert quality==['My name is Tasha Starski. Please confirm it. Thank you.']
    assert state.stable_translation=='确认译文。'


def test_confirmed_prefix_is_cached_across_trailing_boundary_space_and_final():
    calls=[]
    state=AdaptiveTranslationSnapshots(lambda text,lang:calls.append(text) or '确认。')
    text='Alpha. Bravo. Charlie.'
    state.update(text,'en')
    assert len(calls)==1
    state.update(text+' Next','en')
    assert calls.count(text)==1
    state.update(text+' Next','en',final=True)
    assert calls.count(text)==1
    assert state.last_stats['cache_hits']>=1


def test_spelled_name_is_never_sent_to_model_in_preview_or_confirmation():
    seen=[];partials=[]
    def q(text,lang):seen.append(text);return '我的名字是塔莎。'
    def p(text,lang,callback):seen.append(text);callback('名字');return '名字'
    state=AdaptiveTranslationSnapshots(q,p)
    source='My name is Tasha Starski. S T A R S K I.'
    preview,_=state.update(source,'en',partials.append)
    final,_=state.update(source,'en',partials.append,final=True)
    assert 'S T A R S K I' in final and 'S T A R S K I' in preview
    assert all('S T A R S K I' not in text for text in seen)
    assert '\n' not in final
    assert '星星之爱' not in final


def test_window_plan_keeps_all_source_and_caps_long_parts():
    text='Alpha. Bravo. Charlie. '+('猫🙂'*900)+'؟'
    plan=context_windows(text)
    assert ''.join(window for window,_ in plan)==text
    assert all(len(window)<=700 for window,_ in plan)
    assert len(plan)>2


def test_sentence_boundaries_do_not_confuse_word_suffix_and_abbreviation():
    assert _sentence_data('Send it first. Then confirm.')[1]==2
    assert _sentence_data('Dr. Smith arrived. Next.')[1]==2
    assert _sentence_data('هل وصل؟ نعم.')[1]==2


def test_explicit_spelling_is_protected_during_chinese_refinement():
    from system.protection import EntityProtector
    protector=EntityProtector((),protect_numeric_spans=False)
    text='姓名拼写为 S T A R S K I。'
    protection=protector.protect(text)
    assert 'S T A R S K I' not in protection.masked_text
    assert any(span.entity_type=='SPELLING' for span in protection.spans)
    restored=protector.restore(protection.masked_text,protection)
    assert restored.text==text
