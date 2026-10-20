# NLLB 原始词表交叉核查

本次只使用 CPU，不运行翻译推理。检查对象为 `/home/aim0/data/models/ASR/nllb200_600M`，固定 revision `f8d333a098d19b4fd9a8b18f94170487ad3f821d`。

- 原始 `sentencepiece.bpe.model` 中，繁体“閥”的 SentencePiece ID 为 255789；简体“阀”以及“锈”“銹”“鏽”都映射到 SP unk ID 0。模型 `trainer_spec.byte_fallback=False`，没有 `<0xXX>` 字节词条。
- Transformers 4.57.6 的 slow `NllbTokenizer` 与实际使用的 fast tokenizer 一致：“阀门”包含 `<unk>` + “门”，“不锈钢”包含“不” + `<unk>` + “钢”。跳过特殊词后的文本分别为“门”“不钢”。
- CT2 `shared_vocabulary.json` 与 slow tokenizer 的 256204 个 ID 逐项一致，零个不匹配；末尾两个为 `madeupword0/1` 占位词。NLLB/CT2 ID 3 是 `<unk>`，255790 是“閥”。SP 与 Transformers 的上述 ID 相差 1 源于其词表映射约定。
- CT2 词表也没有包含“阀”“锈”“銹”“鏽”的其他词条。这些缺字来自原始词表，不是 fast tokenizer 或转换造成。

不能据此说模型完全无法输出“阀门”：繁体“閥門”可经 OpenCC `t2s` 转成“阀门”。但“锈”“銹”“鏽”本身无法由该词表直接表示，也没有字节回退；模型仍可能使用其他词语表达相关语义。词表限制不等于所有业务句必然翻错，必须结合实际输出判断。

同目录 `tokenizer_trade_vocabulary_audit.json` 保留了三模型的逐短语 token 和解码结果。原始 FP16 与 CT2 翻译记录另存，不将本项 CPU 检查当作翻译准确率评测。

SentencePiece 对字节回退的定义：[模型 proto](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto)、[选项说明](https://github.com/google/sentencepiece/blob/master/doc/options.md)。
