# 中文→外文三模型评测

脚本：`scripts/benchmark_zh_translation.py`。模型为本机官方 BF16 检查点：Index-Translate-2B、Hy-MT2-1.8B、NiuTrans LMT-60-1.7B。评测针对完整中文文本→外文翻译；不包含 ASR、中文精修、音频时延或增量重译。

## 数据集选择

默认使用官方 **FLORES-101 devtest**，简体中文 `zho_simpl` → 英、法、德、西、俄、日、韩、阿。每个方向 1,012 条，同一个 ID 在八种目标语言中使用同一条中文原文，总计 8,096 对。原始公开归档已经在本机，准备脚本直接读取压缩包内的 devtest，不使用旧项目中的“外文→中文”评测文件。

- [官方项目与历史数据入口](https://github.com/facebookresearch/flores)
- [官方原始归档](https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz)
- [论文](https://arxiv.org/abs/2106.03193)
- 许可：CC BY-SA 4.0。数据文件位于本地 `data/eval/zh_translation/`，不加入源码仓库。

这是通用书面语、逐句对齐的公开数据；中文大多来自英文文本翻译，因此不能代表中文口语、贸易谈判或真实 ASR 错误。数据准备输出 JSONL 及 manifest，记录原始归档和整理结果的 SHA256。

**训练接触限制：**[NiuTrans 官方 README](https://github.com/NiuTrans/LMT)明确列出 FLORES-200 devset、NTREX-128、WMT14–23、IWSLT17–24 等作为 SFT 数据来源，所以不把那些已知数据当作独立测试。这里仅使用 devtest；FLORES 各版本存在共享内容，公开测试集仍可能被预训练、调参或未披露训练接触，结果不能声称是“完全未见数据的泛化准确率”。不使用 devtest 调整提示词或挑选最佳参数后再把同一集作为独立测试。

补充选择：[FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) 提供持续维护的版本；[BOUQuET](https://huggingface.co/datasets/facebook/bouquet) / [官方评测代码](https://github.com/facebookresearch/bouquet)提供更多原生写作、多领域内容，中文来源方向可补充 FLORES，但不能假定含全部八种语言。这两个 HF 入口当前要求登录并接受条款，脚本不绕过访问控制。接受条款后可整理成下述统一 JSONL 再使用 `--data`；应保留数据集版本、许可和源语言来源信息。

## 运行环境与模型

本机已创建 `/home/aim0/data/venvs/translation-bench/`，复用 agentic-asr 环境的 PyTorch 2.11.0 / Transformers 5.14.1，通过独立 venv 安装 SacreBLEU 2.5.1。没有升级正在运行的网页或 ASR 环境。迁移机器时可新建环境并安装 `scripts/requirements-zh-benchmark.txt`。

模型目录默认 `/home/aim0/data/models/ASR/`，可用 `--model-root` 指定其他路径。每个模型需要下载完整的 `DOWNLOAD_MANIFEST.json`、权重、配置和 tokenizer；尚未下载完成时脚本会报出缺失文件，不会自动联网换模型或降精度。

官方提示模板来源：[Index](https://github.com/bilibili/Index-Translate/blob/main/docs/prompts.md)、[Hy-MT2](https://github.com/Tencent-Hunyuan/Hy-MT2/blob/main/README_CN.md)、[LMT](https://github.com/NiuTrans/LMT/blob/main/src/inference.py)。采用模型各自的官方翻译指令和聊天模板。Index 显式关闭 thinking。所有模型统一贪心解码、beam=1、repetition penalty=1；这是便于对照的统一协议，并非复现每个作者各自推荐的最优/采样配置。

## 命令

在项目根目录运行：

```bash
PY=/home/aim0/data/venvs/translation-bench/bin/python

# 生成全部中文→八语数据，8,096 对。
$PY scripts/benchmark_zh_translation.py prepare

# 先做功能检查：每方向 2 条，只检查流程，不作质量排名。
$PY scripts/benchmark_zh_translation.py run --model all \
  --languages en,ja --limit 2 --warmup 1 --gpu 1 \
  --output results/zh_mt_smoke

# 全量评测：八个方向各 1,012 条；本机 RTX 3090 采用 batch=128。
$PY scripts/benchmark_zh_translation.py run --model all \
  --limit 0 --max-new-tokens 512 --warmup 1 --batch-size 128 --gpu 1 \
  --output results/zh_mt_full

# 每个方向全部 1,012 条：改为 --limit 0，建议另选输出目录。
# 中断后用原参数加 --resume；配置/数据不一致会拒绝续跑。

# 指标计算在生成结束后进行，不计入推理时延。
for model in index hy lmt; do
  $PY scripts/benchmark_zh_translation.py score \
    --predictions "results/zh_mt_full/$model/predictions.jsonl"
done

$PY scripts/benchmark_zh_translation.py report --output results/zh_mt_full
```

`--model all` 按顺序开启三个独立子进程，上一模型进程退出后才加载下一个。默认 batch=16；可用 `--batch-size 1` 测单句交互时延，或用 32/64/128 比较批量吞吐。batch 内的翻译彼此独立，但同批样本共享一次 GPU 调用。更大的 batch 会提高吞吐并增加批首 token 延迟和显存占用。本机 RTX 3090 的 LMT 在 batch=256、较长方向上显存不足，最终全量结果使用 batch=128 并成功完成；共享显存时改用 32/64。也可用 `--model index`、`hy`、`lmt` 分别运行。默认不会覆盖已有结果，必须使用新目录或原配置 `--resume`。

在 GPU 1 有充足空闲显存时再运行；脚本记录开始/结束时的 GPU 使用情况，但不会停止其他进程。正式速度排名需要相同 GPU、精度、软件环境及负载条件；共享 GPU 的其他服务会影响结果。

## 指标和输出

质量采用 [SacreBLEU](https://github.com/mjpost/sacrebleu)：

- **chrF++**：字符和词 n-gram 相似度，适合包括日、韩在内的多语种对照，作为主要自动指标。
- **spBLEU**：统一采用 FLORES-200 SentencePiece tokenizer；保存完整 metric signature。首次评分可能下载 tokenizer，网络失败会直接报错，不会偷偷替换分词器。离线时可预先在该环境初始化 `BLEU(tokenize='flores200')`，或明确改为 `--bleu-tokenizer 13a`，但不得把两种分数混合比较。

这两个分数不是“翻译正确率百分比”，一个合法的不同译法也可能低分。本次已对相同 ID 的已保存译文补跑 COMET；人工核查仍适合发现漏译/增译、数字单位、专有名词、否定、术语和语气等错误。

### COMET 补充评分

保存的完整译文可用 `scripts/score_zh_comet.py` 计算 Unbabel `wmt22-comet-da`，并对同一 source/reference 的句对差值做 10,000 次分层配对 bootstrap。每个目标语分别比较三个模型，并报告 Holm 校正后的 p 值；八语宏平均按语言等权计算。结果写入原全量目录的 `comet_wmt22/`，包含逐句分数、汇总 JSON、bootstrap JSON 和 Markdown 报告。COMET 分数是模型化质量估计，不是准确率百分比；其 XLM-R 底座覆盖本次语言，但指标训练方向与中文→八语方向并不完全匹配，因此作为 spBLEU/chrF++的补充证据。

COMET 与主基准隔离安装，避免其 Transformers 4.x 依赖影响 Transformers 5.x 翻译环境：

```bash
/home/aim0/data/conda/envs/agentic-asr/bin/python -m venv --system-site-packages \
  /home/aim0/data/venvs/translation-comet
/home/aim0/data/venvs/translation-comet/bin/pip install \
  -r scripts/requirements-zh-comet.txt

/home/aim0/data/venvs/translation-comet/bin/python scripts/score_zh_comet.py \
  --gpu 1 --batch-size 128
```

可先用 `--models index --languages en --limit-per-language 8` 做小规模环境验证。物理 GPU 1 由 `CUDA_VISIBLE_DEVICES` 单独映射给 COMET；脚本会检查 CUDA，不会在 CUDA 不可用时退回 CPU。遇到显存不足时会把 batch 减半后重试，最终使用的批量写入报告。

可额外用每个方向 64 条的固定样本测 batch=1 交互延迟，结果单独存放，不与 batch=128 吞吐结果混为一谈：

```bash
/home/aim0/data/venvs/translation-bench/bin/python scripts/benchmark_zh_translation.py run \
  --model all --limit 64 --seed 20261020 --warmup 4 --batch-size 1 --gpu 1 \
  --output results/zh_translation_latency_batch1_sample64_20261020
```

速度在预热后按批计时，GPU 同步后包含提示词构造、分词、输入传输和生成；不包含模型加载、预热、质量评分和最后的文本解码。报告包含批首 token 延迟、整批 p50/p95、摊销秒/条、样本/秒、字符/秒和 token/秒。摊销秒/条是批延迟除以批量，不能当作单句响应延迟。EOS 计入生成 token 数，正文 token 数另存。各模型 tokenizer 不同，token/秒不能直接代表同等文本量；应结合样本/秒、每批延迟、显存、输出长度和质量评估。

单条推理时 GPU 主要存放约 4–5 GiB 的 BF16 权重，短句的激活和 KV 状态很小，所以显存只占一部分。填满显存本身不会提速；增大 batch 才能并行计算更多条翻译。本机同一组 256 条中文→英文样本的 batch=128 / 256 对照中，样本吞吐分别为：Index 22.14 / 22.85 条/秒，Hy-MT2 34.49 / 36.67，LMT 36.38 / 38.92。更大的 batch 带来约 3–7% 吞吐增益，但在全八语长方向测试里 LMT 的 batch=256 会 OOM，所以完整对比采用 batch=128。该批量下 LMT 的 PyTorch reserved 峰值约 23 GiB，运行时应让 GPU 1 保持空闲；脚本会同时报告 allocated 和 reserved 显存。

**Index kernel 限制：**本机 Transformers 在缺少 flash-linear-attention / causal-conv1d 时使用 PyTorch 线性注意力 fallback。脚本记录该状态，报告明确标注。这种速度仅代表当前推理栈，不能当作模型经过 kernel 优化后的最佳性能。

每个模型目录包含：

- `predictions.jsonl`：逐条 ID、中文、参考、原始输出、最终译文、token 数、耗时、EOS/截断标记、会话 ID。
- `metadata.json`：协议指纹、数据/模型 manifest 哈希、软件版本、精度、实际参数量、显存峰值、提示模板、生成配置、GPU 负载快照。参数量可能与产品名标称规模不同。
- `summary.json`：按目标语言的质量与速度统计、空输出/截断数量。
- 顶层 `comparison.md/json`：统一对比表；若数据、样本 ID、解码配置或环境不一致，拒绝生成比较表。

长输出达到 `--max-new-tokens` 且没有 EOS 会标记截断，保留在质量评分中，不能悄悄剔除以抬高得分。需要增大上限时，在新的输出目录中用同一上限重跑全部模型。

自有数据格式（一行一条，每种目标语言包含相同 ID 及相同中文原文）：

```json
{"id":"case-001","language":"en","source":"请确认交货日期。","reference":"Please confirm the delivery date."}
{"id":"case-001","language":"ja","source":"请确认交货日期。","reference":"納期を確認してください。"}
```

使用 `--data /absolute/path/cases.jsonl --languages en,ja` 即可。

协议检查命令：

```bash
/home/aim0/data/venvs/translation-bench/bin/python -m unittest discover \
  -s tests -p 'test_benchmark_zh_translation.py' -v
```

2026-10-20 已通过 9 项协议测试、多个 batch 尺寸的同样本吞吐检查，并完成三模型全量对照：[8,096 条结果及逐方向比较表](../results/zh_translation_benchmark_full_batch128_20261020/comparison.md)。三模型各翻译 8,096 条，统一 batch=128。batch=256 时 LMT 在较长方向遇到显存不足，结果标记在 `results/zh_translation_benchmark_full_batch256_20261020/`；全量比较使用 batch=128。全量 LMT 有 8 条译文到达 512 token 上限，已保留并计入评分。Index-Translate 在当前 Transformers 下用 PyTorch fallback 内核，速度比较受这个实现影响。比较表的 BLEU 列默认是 FLORES-200 SentencePiece 分词的 spBLEU；具体分词 signature 保存在 summary 中。显存统计记录预热后的 allocated/reserved 峰值，不包含冷启动加载峰值。

2026-10-20 另跑了统一 batch=128 的 GPU1 高吞吐样本：每语种 512 条、每模型 4,096 条；[对比报告](../results/zh_translation_batch128_sample512_20261020/comparison.md)。GPU1 利用率约 97–100%；Index 峰值 allocated/reserved 为 8.64/9.28 GiB，Hy-MT2 为 5.56/6.19 GiB，LMT 为 12.23/19.45 GiB。该样本中 LMT 有 3 条到达 512 token 上限，仍保留在评分里。三模型整体吞吐约为 Index 19.17、Hy-MT2 23.33、LMT 16.21 条/秒；该结果来自每语种512条的子集，完整 8,096 条结果仍以全量报告为准。

同日的 COMET 全量补充结果见[COMET 报告](../results/zh_translation_benchmark_full_batch128_20261020/comet_wmt22/report.md)。其等语言权重宏平均为 Index 0.8687、Hy-MT2 0.8651、LMT 0.8585；配对 bootstrap + Holm 校正后，三组宏平均差异均显著。供人工抽查的 96 条匿名样本及说明位于 `results/zh_translation_benchmark_full_batch128_20261020/comet_wmt22/human_review_96.csv` 和 `human_review_guide.md`；模型槽位映射保存在同目录的 `human_review_96_key.json`。
