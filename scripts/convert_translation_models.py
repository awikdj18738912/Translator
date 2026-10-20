"""Create CPU-converted INT8 GPU-runtime copies; originals remain pinned and intact."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter


def main():
    import torch
    import ctranslate2
    from ctranslate2.converters import TransformersConverter
    from transformers import M2M100Tokenizer
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['m2m100_418M','small100','nllb200_600M','LMT-60-0.6B'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path("/home/aim0/data/models/ASR")
    for name in ([args.model] if args.model else ("m2m100_418M", "small100")):
        path = root/name
        destination = path/"ct2-int8"
        if (destination/"CONVERSION_MANIFEST.json").exists():
            print(f"Already converted: {destination}",flush=True)
            continue
        if destination.exists():
            raise RuntimeError(f"Partial destination exists; inspect before retry: {destination}")
        started = perf_counter()
        standard = (AutoTokenizer.from_pretrained(path, local_files_only=True)
                    if name == 'LMT-60-0.6B' else
                    AutoTokenizer.from_pretrained(path, src_lang='eng_Latn', local_files_only=True)
                    if name == 'nllb200_600M' else M2M100Tokenizer.from_pretrained(path, local_files_only=True))
        checks = {"runtime_tokenizer":type(standard).__name__}
        if name == "small100":
            from mt_tokenizers import load_small_tokenizer
            custom=load_small_tokenizer(path)
            # Weight vocabulary is identical; runtime source language conventions differ.
            vocabulary = {**standard.encoder, **standard.lang_token_to_id}
            assert all(custom.convert_tokens_to_ids(token)==index for token,index in vocabulary.items())
            checks = {"runtime_tokenizer":"author SMALL100Tokenizer", "conversion_tokenizer":"M2M100Tokenizer",
                      "verified_lexical_and_language_token_ids":len(vocabulary),
                      "source_convention":"prepend target __zh__ to source; no forced Chinese decoder token",
                      "compatibility_fix":"get_vocab excludes dummy ids that otherwise overwrite <unk> id 3 with 128111 in Transformers 4.57"}
            del custom
        del standard
        print(json.dumps({"event":"conversion_start","model":name,**checks}),flush=True)
        converter=TransformersConverter(str(path), load_as_float16=True, low_cpu_mem_usage=True)
        converter.convert(str(destination),quantization="int8_float16")
        record={"source_manifest":str(path/"DOWNLOAD_MANIFEST.json"),"ctranslate2":ctranslate2.__version__,
                "quantization":"int8_float16","seconds":perf_counter()-started,**checks}
        record["files"]=[]
        for file in destination.iterdir():
            if file.is_file():
                digest=hashlib.sha256()
                with file.open("rb") as f:
                    for block in iter(lambda:f.read(8*1024*1024),b""):
                        digest.update(block)
                record["files"].append({"file":file.name,"bytes":file.stat().st_size,"sha256":digest.hexdigest()})
        (destination/"CONVERSION_MANIFEST.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
        print(json.dumps({"event":"conversion_complete","model":name,"seconds":record["seconds"]}),flush=True)
        del converter
        gc.collect()


if __name__=="__main__":
    main()
