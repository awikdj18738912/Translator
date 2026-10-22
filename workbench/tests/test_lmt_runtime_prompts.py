from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmt_runtime import LANGUAGES, LmtRuntime, SOURCE_LANGUAGES


class PromptTokenizer:
    all_special_ids = [99]

    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append((messages, kwargs))
        return [1, 2, 3]


def runtime():
    instance = object.__new__(LmtRuntime)
    instance.tokenizer = PromptTokenizer()
    instance.special = set(instance.tokenizer.all_special_ids)
    return instance


def test_prompt_language_directions_and_legacy_default():
    instance = runtime()
    assert len(LANGUAGES) == 8
    assert SOURCE_LANGUAGES['zh'] == 'Chinese'

    instance.prompt_ids('你好', 'en', source_language='zh', target_language='en')
    instance.prompt_ids('こんにちは', 'ja', source_language='zh', target_language='ja')
    instance.prompt_ids('hello', 'en')

    prompts = [entry[0][0]['content'] for entry in instance.tokenizer.prompts]
    assert prompts == [
        'Translate the following text from Chinese into English:\nChinese: 你好\nEnglish:',
        'Translate the following text from Chinese into Japanese:\nChinese: こんにちは\nJapanese:',
        'Translate the following text from English into Chinese:\nEnglish: hello\nChinese:',
    ]
    for _, kwargs in instance.tokenizer.prompts:
        assert kwargs == {'tokenize': True, 'add_generation_prompt': True, 'return_dict': False}


@pytest.mark.parametrize('source,target', [('xx', 'en'), ('zh', 'xx'), ('en', 'en')])
def test_prompt_rejects_unsupported_or_identical_languages(source, target):
    with pytest.raises(ValueError):
        runtime().prompt_ids('sample', target, source_language=source, target_language=target)


def test_stream_forwards_explicit_languages_and_keeps_special_token_filter():
    instance = runtime()
    instance.backend = 'ctranslate2'
    calls = {}

    def translate(texts, language, **kwargs):
        calls.update(texts=texts, language=language, **kwargs)
        kwargs['callback'](SimpleNamespace(token_id=99))
        return ['result']

    instance.translate = translate
    shown = []
    assert instance.translate_stream('你好', 'en', shown.append,
                                     source_language='zh', target_language='en') == ['result']
    assert calls['source_language'] == 'zh'
    assert calls['target_language'] == 'en'
    assert calls['texts'] == ['你好']
    assert calls['language'] == 'en'
    assert shown == []
