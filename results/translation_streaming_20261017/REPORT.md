# 两个多语翻译模型的下载与 GPU 流式可用性测试

测试日期：2026-10-17。

## 结论

两个模型均已下载，权重通过 Hugging Face LFS SHA256 校验。两者都能在 GPU 上持续处理新增的多语文本并更新中文；逐 token 输出接口也验证通过。
这里验证的是系统层的增量重译/稳定前缀方案，不是模型原生增量编码或原生同传。SMaLL-100 在本次短文本中更快，但两个模型均出现严重的数值、术语、否定或时间关系错误，不能据此直接用于外贸业务。

## 下载文件

| 模型 | 本地目录 | 固定仓库版本 |
|---|---|---|
| m2m100_418M | `/home/aim0/data/models/ASR/m2m100_418M` | `55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636` |
| small100 | `/home/aim0/data/models/ASR/small100` | `8ab680e26a596d2e3d2d2d17ae0f68df1037328c` |

每个模型目录保存原始权重、配置、分词文件、README、DOWNLOAD_MANIFEST.json；ct2-int8/ 是额外的 CTranslate2 INT8/FP16 推理副本，有独立的 CONVERSION_MANIFEST.json。
原始模型文件合计分别约 1.94 GB 和 1.34 GB；转换副本分别约 491 MB 和 339 MB。文件大小不等于显存。

## 实验范围与配置

- 8 种源语言：英语、法语、德语、西班牙语、俄语、日语、韩语、阿拉伯语；目标固定中文。
- 25 条人工构造轨迹：每种语言包含报价/型号/交期、否定/改口/付款条件、短应答，另有一条英文 ASR 修订轨迹。参考中文供人工核对，未进行正式人工双盲评测。
- 两种调度：当前全文增量重译；连续两轮 token 前缀一致后提交、保留一个 token 后退量并强制已提交前缀。后者是本次简化实现，不是完整 NLLW 集成，也没有语义正确性保证。
- 每模型/设置 152 次增量调用；另对完整源文作 beam=3 对照。batch=1、最大生成96 token、源文本上限256 token。
- CTranslate2 4.6.3，int8_float16；另测 Transformers 4.57.6 + Torch 2.10.0+cu128 原始 FP16 权重。
- CTranslate2 beam=3 测试实际每0.5秒发送一次文本更新；每条轨迹结束时提交剩余译文并重置，再处理下一条。原始 FP16 对照是非定时回放。
- 最初两张 RTX 3090 被已有任务占用，FP16 在加载前被显存保护拒绝。先完成 INT8 测试；GPU1 随后由原任务自行释放，才完成原始 FP16 对照。本任务没有停止其他 GPU 任务。

## GPU 实测

以下为预热后一次短文本更新从分词到中文返回的墙钟耗时。原始 FP16 与 CT2 同时改变了推理引擎和精度，不能把速度差全部归因于量化。

| 模型 | 引擎/精度 | beam | 更新耗时 P50 | P95 | 预热后本进程显存采样 |
|---|---|---:|---:|---:|---:|
| m2m100_418M | ctranslate2 / int8_float16 | 3 | 83.9 ms | 136.0 ms | 878 MiB |
| small100 | ctranslate2 / int8_float16 | 3 | 30.9 ms | 49.7 ms | 686 MiB |
| m2m100_418M | transformers / float16 | 3 | 239.0 ms | 388.7 ms | 1272 MiB |
| small100 | transformers / float16 | 3 | 93.8 ms | 150.1 ms | 990 MiB |

显存为 nvidia-smi 对测试进程的采样，不是完整系统显存，也不是驱动层峰值。结果不包含音频累计、ASR 确认、网络、字幕稳定等待和 Refiner，不能视为语音到中文字幕的端到端延迟。

两个 CT2 beam=3 定时测试均完成152次调用，无空译文或长度截断；在这些短文本上更新计算没有跨过0.5秒的更新周期。定稿清空、提交前缀只增不改的结构不变量检查通过。这不代表内容正确。

greedy 设置另完成每模型100个有界句窗的连续处理；这是快速压力回放，并非100分钟或长时间真实语音稳定性验收。

## 错误样例与精度对照

以下取 retranslate 模式最终译文；原文和各步输出见对应 cases.jsonl/steps.jsonl。价格为明确12.50美元，行业实体是阀门。

| 设置 | 模型 | 用例 | 最终输出 |
|---|---|---|---|
| beam3_realtime | m2m100_418M | en_order | 我们需要500个SUS304阀门,每台1250美元,到11月15日发货。 |
| beam3_realtime | m2m100_418M | ja_order | SUS304 的 500 个单价 12.50 米的阀门,请在 11 月 15 日之前交付。 |
| beam3_realtime | m2m100_418M | ko_correction | 请不要发送500个,对不起,请换到300个,请在收到合同后一定发送。 |
| beam3_realtime | small100 | en_order | 我們需要500SUS304風扇,每個1250美元,並在11月15日發送。 |
| beam3_realtime | small100 | ja_order | 请在11月15日之前发送500个单价为12.50美元的SUS304风扇。 |
| beam3_realtime | small100 | ko_correction | 请不要退出500件,对不起,请改变300件,请在收到合同后完成。 |
| fp16_beam3 | m2m100_418M | en_order | 我们需要500个SUS304阀门,每台1250美元,到11月15日发货。 |
| fp16_beam3 | m2m100_418M | ja_order | SUS304 的 500 个单价 12.50 米的阀门,请提交至 11 月 15 日。 |
| fp16_beam3 | m2m100_418M | ko_correction | 请不要发送500个,对不起,请换到300个。 |
| fp16_beam3 | small100 | en_order | 我们需要500个SUS304门,每一个1250美元,并在11月15日之前交付。 |
| fp16_beam3 | small100 | ja_order | 请在11月15日之前发送500个单价1250美元的SUS304风扇。 |
| fp16_beam3 | small100 | ko_correction | 请不要退出500件,请原谅,请改变300件,请在收到合同后完成。 |

- INT8 greedy 的 M2M100 将英文 `No, not yet.` 译成 `是的,还不。`；beam=3 改成 `不,还不。`，说明解码设置会影响关键含义。
- 两个模型的部分报价输出出现12.50→1250；日语/其他语言出现阀门→风扇、门、电波等误译；韩语付款条件出现定金→合同。应结合上表原始 FP16 对照判断，不能全归咎于量化。
- 简化 LocalAgreement 会锁住稳定的错误。例如 M2M100 的 CT2 beam=3 德语报价，普通重译最终恢复12.50美元，而强制前缀模式保留1250美元。稳定性不等于正确性。
- 两个模型可能输出简繁混合。统一字形只能解决展示形式，不能修复事实错误。
- 没有用这些少量构造样例计算正式准确率、BLEU、COMET 或生产可用率。

## 集成判断

可以：小型模型 GPU 推理 + 有界增量重译 + 当前尾部修订，用于搭建多语中文字幕实验基线。SMaLL-100 可优先研究速度，M2M100作为对照。
尚不能：把任一模型作为无需进一步质量验证的外贸翻译引擎；把错误稳定输出交给只看中文的 Refiner 并假设会被修好。下一步应接入真实 ASR 轨迹，加入数值/单位与业务术语验证、语段版本失效，再做真实业务录音评测。

## SMaLL-100 分词器兼容性

作者 get_vocab() 会枚举8个虚拟 token，现代 Transformers 中将 <unk> 的映射从3覆盖成128111。测试在 scripts/mt_tokenizers.py 中只覆盖词表枚举，保留作者把目标 __zh__ 放到源文前缀的协议。原下载文件没有改动。转换前已逐项核对128104个词汇与语言 token ID；未把 SMaLL-100 当普通 M2M100源语言协议使用。

## 复现

```bash
cd /home/aim0/桌面/tan/ASR/Translator
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/home/aim0/data/conda/envs/vllm/lib/python3.12/site-packages/nvidia/cublas/lib:/home/aim0/data/conda/envs/vllm/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
.runtime/mt-eval-venv/bin/python scripts/benchmark_streaming_mt.py --model small100 --gpu 1 --stream-beams 3 --realtime-interval 0.5 --output results/translation_streaming_rerun
# 将 --model 改为 m2m100_418M 可测试另一个模型；使用不同output保存每轮数据。
```

定时回放使用人工文本更新；没有录音输入。模型测试环境独立添加了 CTranslate2，未修改原 vLLM 环境依赖。

## 证据文件

- beam3_realtime/*.summary.json：定时回放的配置、延迟、GPU采样和完整结果摘要。
- beam3_realtime/*.steps.jsonl：每次源文更新、完整中文、已提交前缀、可变尾部、实际提交/完成时刻。
- beam3_realtime/*.cases.jsonl：最终输出与完整文本beam=3对照。
- fp16_beam3/：原始FP16对照，结构同上。
- 根目录两模型的steps/cases/summary/stress：INT8 greedy及100句快速压力回放。
- download.log、conversion*.log、各run.log：下载、转换及测试日志；smoke/为调试数据，未用于正式表格。

## 上游依据

- [M2M100 模型卡](https://huggingface.co/facebook/m2m100_418M)
- [SMaLL-100 模型卡](https://huggingface.co/alirezamsh/small100)
- [SMaLL-100 作者代码](https://github.com/alirezamshi/small100)
- [CTranslate2 转换与推理](https://opennmt.net/CTranslate2/guides/transformers.html)
- [NLLW 稳定前缀设计](https://github.com/QuentinFuxa/NoLanguageLeftWaiting)
