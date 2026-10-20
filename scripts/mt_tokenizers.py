"""Load the pinned SMaLL-100 tokenizer with its target-on-source convention.

With Transformers 4.57 the author's get_vocab loops over eight dummy ids,
all decoded as <unk>, and overwrites its true id 3 with id 128111.
Override only vocabulary enumeration, keeping original files unchanged.
"""
import hashlib
import importlib.util
import json
import sys

NLLB_LANGUAGES = {'en':'eng_Latn','fr':'fra_Latn','de':'deu_Latn',
                  'es':'spa_Latn','ru':'rus_Cyrl','ja':'jpn_Jpan',
                  'ko':'kor_Hang','ar':'arb_Arab','zh':'zho_Hans'}


def load_translation_tokenizer(path, name):
    from transformers import AutoTokenizer, M2M100Tokenizer
    if name == 'small100':
        return load_small_tokenizer(path)
    if name == 'nllb200_600M':
        return AutoTokenizer.from_pretrained(path, src_lang='eng_Latn', local_files_only=True)
    return M2M100Tokenizer.from_pretrained(path, src_lang='en', local_files_only=True)


def set_source_language(tokenizer, name, language):
    if name == 'small100':
        tokenizer.tgt_lang = 'zh'
    else:
        tokenizer.src_lang = NLLB_LANGUAGES[language] if name == 'nllb200_600M' else language


def chinese_decoder_prefix(tokenizer, name):
    if name == 'small100':
        return []
    return ['zho_Hans'] if name == 'nllb200_600M' else [tokenizer.get_lang_token('zh')]


def chinese_language_id(tokenizer, name):
    return (tokenizer.convert_tokens_to_ids('zho_Hans') if name == 'nllb200_600M'
            else tokenizer.get_lang_id('zh'))


def load_small_tokenizer(path):
    script = path/"tokenization_small100.py"
    manifest = json.loads((path/"DOWNLOAD_MANIFEST.json").read_text())
    expected = next(r["sha256"] for r in manifest["files"] if r["file"]==script.name)
    if hashlib.sha256(script.read_bytes()).hexdigest()!=expected:
        raise RuntimeError("Tokenizer changed since download verification")
    spec=importlib.util.spec_from_file_location("tokenization_small100",script)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)

    class CompatibleSmallTokenizer(module.SMALL100Tokenizer):
        def get_vocab(self):
            return {**self.encoder, **self.lang_token_to_id, **self.added_tokens_encoder}

    tokenizer=CompatibleSmallTokenizer.from_pretrained(path,tgt_lang="zh",local_files_only=True)
    assert tokenizer.unk_token_id==3
    assert tokenizer.convert_tokens_to_ids("__zh__")==tokenizer.get_lang_id("zh")
    return tokenizer
