# LMT protocol review

Read-only review of the LMT runtime and evaluation branches on 2026-10-17. No
model conversion or GPU execution was performed. The tokenizer check used the
local tokenizer files and Transformers 4.57.6 on CPU.

## Generation protocol

- The runtime creates the author's English translation instruction, passes it
  through the checkpoint chat template with `add_generation_prompt=True`, and
  adds no explicit thinking stub. The checkpoint README uses the same prompt
  shape and chat-template call.
- The local `generation_config.json` declares EOS IDs `[151643, 151645]`. The
  runtime normalizes either scalar or list EOS config, passes both IDs as CT2
  end tokens, requests `return_end_token=True`, and recognizes either ID in the
  returned continuation. This covers `<|endoftext|>` and `<|im_end|>`.
- CT2 4.8.3 documents that `include_prompt_in_result=False` forwards the start
  tokens to initialize decoder state and omits them from the returned result.
  Its changelog says that in this mode `max_length` applies to the sequence
  after the prompt. The runtime includes committed target tokens in the prompt,
  subtracts their count from the configured cumulative target budget, and then
  appends only generated continuation IDs. The Native path slices off the same
  prompt and uses the same remaining-token limit. This avoids both duplicated
  committed text and silent budget inflation. A cap reached without EOS remains
  observable through the existing EOS/truncation fields.
- The Unicode holdback retreats until decoding the candidate prefix no longer
  contains U+FFFD, while never retreating before the already committed prefix.
  This is a conservative boundary check for incomplete byte-token characters.

## Tokenizer warning

The warning is a false positive for this checkpoint; do not set
`fix_mistral_regex=True` for the benchmark protocol. Evidence:

- `config.json` says `model_type: qwen3` and `transformers_version: 4.57.3`;
  `tokenizer_config.json` says `tokenizer_class: Qwen2Tokenizer`.
- The published `tokenizer.json` pre-tokenizer is a `Sequence` containing the
  Qwen regex split followed by ByteLevel, with NFC normalization and a ByteLevel
  post-processor. Loading via `AutoTokenizer` in Transformers 4.57.6 produces
  that same pre-tokenizer. The unmodified AutoTokenizer output therefore
  preserves the checkpoint's saved tokenizer behavior.
- Transformers 4.57.6's `_patch_mistral_regex` detection treats local configs
  saved with versions later than 4.57.2 as Mistral candidates unless they are
  saved at 5.0.0 or newer; the older-version model-type exemption does not run
  for this 4.57.3 config. It emits the warning based on that version gate. The
  Qwen2 tokenizer implementation itself defines the same regex found in this
  checkpoint's tokenizer JSON.
- Passing `fix_mistral_regex=True` replaces the first pre-tokenizer split with
  Mistral's case-sensitive Unicode-class pattern. That mutates the checkpoint
  tokenizer instead of retaining its declared Qwen regex. The normal loading
  path (without the flag) is the faithful choice for the author's checkpoint.
  A small CPU comparison over ASCII, CJK, Hindi, contractions, mixed-case, and
  punctuation samples gave matching IDs between tokenizer JSON and default
  AutoTokenizer. The forced Mistral patch happened to match those samples too;
  this limited sample is not evidence that the two patterns are equivalent.

## Evaluation comparability and open weight check

`evaluate_multilingual_mt.py` uses beam size 3 for both full-input and
streaming calls. `benchmark_streaming_mt.py` defaults streaming to beam 1 while
its offline control uses beam 3. The benchmark run should explicitly pass
`--stream-beams 3` when a same-beam comparison is intended.

The local safetensors file was still named `model.safetensors.partial` during
this review. Parameter tying and the reported tensor-file size therefore remain
unverified; the review did not load model weights.

## References

- [CTranslate2 4.8.3 Generator API](https://opennmt.net/CTranslate2/python/ctranslate2.Generator.html)
- [CTranslate2 changelog](https://github.com/OpenNMT/CTranslate2/blob/master/CHANGELOG.md)
- [Hugging Face Transformers tokenizer loading code](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/tokenization_utils_base.py)
- [Hugging Face Qwen2 tokenizer implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2/tokenization_qwen2.py)
- [Transformers issue documenting the 4.57.3 false-positive version gate](https://github.com/huggingface/transformers/issues/42591)
- Local checkpoint README: `/home/aim0/data/models/ASR/LMT-60-0.6B/README.md`
