from system.translation_windows import StableTranslationSnapshots


def test_quality_confirmed_windows_keep_same_cache_at_final():
    calls=[];previews=[]
    def quality(text,lang):
        calls.append(text);return '确认'+text.strip()
    def stream(text,lang,on_text):
        on_text('预览');return '低精度预览'
    window=StableTranslationSnapshots(quality,group_sentences=1,max_chars=240,stream_translate=stream)
    first,_=window.update('Alpha. Bravo.', 'en',previews.append)
    assert '低精度' not in first and '预览' not in first
    calls.clear()
    final,_=window.update('Alpha. Bravo.', 'en',previews.append)
    assert final==first and not calls
    assert '\n' not in final
    assert window.last_stats['model_calls']==0


def test_chinese_join_does_not_introduce_paragraphs_or_glue_ascii_words():
    values=iter(['我们需要阀门。','请周五发货。'])
    window=StableTranslationSnapshots(lambda *args:next(values),group_sentences=1)
    result,_=window.update('Alpha. Bravo.','en')
    assert result=='我们需要阀门。请周五发货。'
    window=StableTranslationSnapshots(lambda text,lang:'iPhone' if text.startswith('Alpha') else 'Pro',group_sentences=1)
    result,_=window.update('Alpha. Bravo.','en')
    assert result=='iPhone Pro'
