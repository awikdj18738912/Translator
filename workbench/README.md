# 中文精修与外语翻译工作台

这是 `Translator` 项目中的 GPU 流式工作台模块，当前链路为 **中文 ASR → 中文精修 → 外语翻译**，使用与 Codex-web 相近的灰绿界面。首页三列顺序为：**中文转录 / 中文精修 / 外语翻译**。

支持上传中文音频进行离线流式分块处理，也支持中文文本回放。目标语言可选英语、法语、德语、西班牙语、俄语、日语、韩语和阿拉伯语，默认英语。当前没有麦克风实时翻译入口。

## 启动

在本文件夹打开终端。翻译服务由独立的 vLLM 用户服务托管，网页服务依赖它并会等待 API 就绪：

```bash
systemctl --user start translation-workbench-web
```

该服务会自动启动 `translation-workbench-vllm`。访问 <http://127.0.0.1:8083/>。如需音频上传，再单独启动配置在 GPU 0 的 ASR 服务：

```bash
systemctl --user start translation-workbench-asr
```

文本回放在 ASR 未运行时仍可使用。服务健康检查为 `/api/translation/health`；只有 `audio_ready: true` 后才能处理音频文件。

运行时复用这台机器已有的模型和 Python 环境，源码不导入或读取 `Codex-web` 目录：

| 环节 | 模型 / 环境 |
| --- | --- |
| 转录 | `/home/aim0/data/models/ASR/Qwen3-ASR-0.6B`，`qwen3-asr` 环境，vLLM BF16 |
| 精修 | `/home/aim0/data/models/ASR/AgenticASR-Refiner`，`agentic-asr` 环境，BF16加载（源文件FP32） |
| 翻译 | `/home/aim0/data/models/ASR/Index-Translate-2B-FP8`，vLLM 0.26、Compressed-Tensors FP8；RTX 3090 使用 Marlin W8A16，Chinese → 目标语言 |

翻译 vLLM 服务通过 `CUDA_VISIBLE_DEVICES=1` 固定在物理 GPU 1，显存预算为 0.17、上下文长度为 2048、最大并发为 8；网页精修模型也加载在物理 GPU 1。两者合计目标约 6 GiB，实际占用会随运行变化。ASR 服务仍固定在物理 GPU 0。网页进程本身不要设置 `CUDA_VISIBLE_DEVICES=1` 并继续传 `--gpu 1`，否则 CUDA 设备编号会改变。迁移电脑时调整模型路径及 `WEB_PYTHON`、`ASR_PYTHON`、`CUDA_LIB_DIR` 环境变量。网页依赖见 `requirements-web.txt`；ASR 使用独立 Qwen3-ASR/vLLM 环境。

## 本机后台服务

网页、ASR 和翻译 vLLM 由 `~/.config/systemd/user/` 中的持久用户服务单元管理；项目模板在 `deploy/systemd/user/`。网页依赖翻译服务，启动时会等待 `/health` 成功。服务仍设置 `Restart=no`，进程退出后不会自动重启。若完整用户注销（本机 Linger=no），启用的服务会在下次用户登录时启动。

```bash
systemctl --user status translation-workbench-vllm translation-workbench-web translation-workbench-asr
systemctl --user start translation-workbench-web
systemctl --user stop translation-workbench-web translation-workbench-vllm
journalctl --user -u translation-workbench-vllm -u translation-workbench-web -u translation-workbench-asr -f
```

只有在完整停用自启动时才执行 `systemctl --user disable translation-workbench-web translation-workbench-vllm translation-workbench-asr`。服务日志写入 journal。服务运行时不要再手动启动同一脚本，以免重复加载 GPU 模型。

## 数据与界面

- `app.py`：本项目启动入口；沿用已验证的精修窗口、实体保护及翻译调度。
- `system/`：Codex-web 后端的本地源码快照，包含 ASR、精修和实体 API。
- `system/web/`：新的三列首页、独立样式、翻译交互和术语库管理脚本；无 CDN 依赖。
- `vllm_runtime.py`：Index 翻译提示词、vLLM OpenAI API 与 SSE token 预览适配。
- `system/chinese_translation_pipeline.py`：中文转录、精修、翻译的有界快照调度。
- `data/entities.db`：从当前翻译服务实体库复制的独立数据库；后续修改只影响本项目。
- `data/sessions/*.jsonl`：逐次原文、翻译、精修、耗时、守卫和调度日志。
- `data/refinement.jsonl`：原精修链路审计记录。

音频时长不限，文件不超过50MB，按2秒分块；ASR固定识别中文，用户选择的是翻译目标语言。精修、翻译各保留一份待处理最新快照，ASR不等待两个文本模型完成。中文按现有三个标点块窗口完成精修，一组完成即交给翻译；未完成精修的ASR尾部只在中文列显示，不进入翻译模型。ASR修订号与翻译修订号独立，避免同一次ASR更新内后续精修组的译文被前端丢弃。

默认 `--translation-strategy adaptive`：精修中文相邻短句最多合并3句/700源字符，完整句累计160字符可提前闭合，长句超限拆分。活动中文前缀通过 vLLM 流式贪心解码显示预览；确认后的窗口使用同一 Index FP8 模型重新完整解码并缓存，输入前缀变化会使相关缓存失效。外语预览允许被确认结果修订。这是分组增量翻译，不是具有跨音频状态的原生同声传译模型。

明确的三个以上大写单字母拼写串（如S T A R S K I）在精修与翻译中继续保护。这不是通用人名或实体识别器；尚未提供所有实体和型号的强制术语翻译。窗口间不人为插入换行。`--translation-strategy single`可用单句策略作为对照；`--translation-output windows`关闭token预览，分组仍由translation-strategy控制。

精修保持三个标点块、相邻上下文、正则门控及守卫规则，处理完一组即推送进度。数字 ITN 默认启用随项目携带的 `zh-itn-strict-conservative-v90`：规则先处理明确的数量、金额、测量、时长和范围；精修通过数字锁定提示只改其他文本，回退路径不调用 Python 数字兜底。语法不明确、型号/电话/成语等规则判定不应改写的数字按规则原样保留；这是保守规则包，不承诺把每个汉字数字强制改成阿拉伯数字。健康接口 `/health` 的 `zh_itn_enabled`、`zh_itn_rule_pack` 与 `numeric_rule_protocol_version` 显示实际状态。界面在输入区下方显示整体用时、转录用时、音频时长、精修模型调用次数、门控跳过次数。音频时长来自后端解码后的audio_total_seconds，按一位小数显示秒数，收到首个音频进度事件后更新；文本回放显示“不适用”。切换文件或新任务会清除旧值，取消/失败时保留已收到的有效时长，没有时长则显示“—”。

统计口径：整体用时从点击开始到完成/取消/异常断连，包含上传、连接与处理等待；取消后计时冻结不代表后台GPU同步调用已经退出。模型调用按本次任务实际发起的生成、重试、局部审核请求累计；同一审核请求内部多次评分forward不另算。门控跳过只记实际执行的keep决策，缓存复用和未完整句组defer不重复计入；同一组因上下文变化重新判断，属于新决策。日志 `session_metrics` 是后端权威计数，`elapsed_processing_ms` 不含前端上传，不能与前端用时相加。旧 `model_calls/total_model_calls` 字段保留兼容，可能不含局部审核，界面优先使用新统计。

前端每150毫秒刷新计时；预览通常80毫秒合并，requestAnimationFrame合并显示；日志显示最近60条，下载保留全量。多句上下文改善本次部分外贸与拼写案例，但小样本不证明普遍质量提升，详见 `results/adaptive_translation_eval_20261019/REPORT.md`。旧单句/终稿重译阶段的历史性能和逐字一致性结论不代表新默认策略。


术语库作用于翻译前的中文精修环节，不是翻译模型的强制词表。编辑词库后在下一次任务中生效。该页面的精修接受状态表示规则检查结果，不代表人工审核或业务准确率保证。

## 中文精修正则门控

本项目默认 `--refinement-gate regex`，只根据现有文本规则及实体提示决定是否调用模型，完全不使用ASR置信度、校准分数或低置信度阈值。重复、填充词、自我更正、异常标点/间隔及边界异常沿用 `system/refinement_gate.py` 的规则；数字格式信号由启用的 zh-itn 规则包单独处理，纯数字规范化本身不会触发精修。实体提示保守触发精修；未命中则保留规则与实体处理后的中文。最终收尾不强制调用精修，未完整的三个标点块句组仍由窗口调度等待，不算作门控节省。

实体保护、规则规范化、精修结果校验及拒绝后回退继续执行。数字格式由 ITN 规则决定；纯数字窗口可由规则直接保留，不调用精修。含口癖、自我更正等非数字清理信号的窗口仍可调用精修，但数字/单位表面受保护。正则门控不能保证一般ASR语义或下游翻译正确。学习型门控、置信度补充、模糊匹配和声学实体辅助仍关闭。

数字规则保护范围是中文精修结果，尚未覆盖外语译文。既往 LMT 测试中，金额 `123万7851元` 曾被错译；该记录不代表当前 Index 模型的表现。工作台不会对外语译文自动校验数字，不能据此声称译文数字完全可靠。

对照运行可指定 `./start.sh --refinement-gate off`。日志包含每组门控模式、命中原因、动作、跳过计数；前端区分“规则保留”和“待精修”，避免把模型跳过误显示为卡住。

## 验证

```bash
PYTHONPATH=. python -m pytest tests -q
node --check system/web/translation.js
node --check system/web/entities.js
```

本次方向切换的后端测试、八种目标语言页面检查、12秒中文音频及桌面/手机截图见项目根目录 [`../results/chinese_first_pipeline_20261020/REPORT.md`](../results/chinese_first_pipeline_20261020/REPORT.md)。八种语言仅做少量功能检查，不代表准确率评测；历史外语→中文的性能或质量报告不代表新方向。

## 来源

后端来自本机 Codex-web 的已修改版本，保留 Apache-2.0 许可证，文件散列见 `SOURCE_SNAPSHOT.json`。本目录的工作台代码与静态前端由 `Translator` 项目独立维护，不自动同步原工程。

静态文件挂载和 WebSocket 接口参考 [FastAPI 官方静态文件文档](https://fastapi.tiangolo.com/tutorial/static-files/) 与 [FastAPI GitHub WebSocket 示例](https://github.com/fastapi/fastapi/blob/master/docs/en/docs/advanced/websockets.md)。翻译模型来源：[bilibili/Index-Translate](https://github.com/bilibili/Index-Translate)。

## 项目记录

当前调度说明见 [三模型调度审查](results/scheduling_audit_20261019/REPORT.md)，后续变更持续登记在 [WORK_LOG.md](WORK_LOG.md)。历史实验结果对应当时的代码版本，不代表当前全部功能的最新性能。

2026-10-20启动参数补充：本项目ASR默认Qwen3-ASR-0.6B，单任务调度使用 `--max-num-seqs 1 --max-num-batched-tokens 4096`，保持context8192、GPU预算0.30，以降低与翻译/精修同卡时的编码器profiling峰值。实际启动证据见 `results/asr06_startup_20261020/STATUS.json`。

2026-10-20缓存实测后：ASR默认显式分配1 GiB KV缓存（`--kv-cache-memory-bytes 1073741824`），可用环境变量 `ASR_KV_CACHE_BYTES` 覆盖，保留8192上下文。这个参数控制KV池，**不是整个ASR进程的显存上限**；显式设置后，vLLM不再根据0.30自动推算KV池。单路短英语及240秒拼接音频重放已与原配置对照。详见 [缓存缩减实测](results/asr_cache_sweep_20261020/REPORT.md)。

2026-10-20精修精度调整：本项目workbench默认 `--refiner-dtype bfloat16`，原始模型文件保持FP32存储，加载时转换；`--refiner-dtype float32` 可恢复原加载精度。健康接口 `/health` 返回实际refiner_dtype与refiner_parameter_bytes。通用 `create_app` / `TransformersRefiner` 未指定精度时仍兼容auto，精修提示词、门控、实体保护、use_cache及上下文均未因精度调整改变。对照与限制见 [BF16实测](results/refiner_bf16_20261020/REPORT.md)。

2026-10-20新增转录用时：顶部紧邻整体用时显示本次累计ASR请求耗时，来自服务器 `asr_total_ms`，包含会话start、音频chunk、finish及失败请求已花费的时间（含接口等待），不含上传、解码、回放等待及cancel清理。各流水线事件携带当前累计值，前端取最大值，不按事件累加。文本模式不适用；新任务、换文件及模式切换重置，完成/取消/断线保留已收到值。验证见 [转录用时记录](results/asr_time_ui_20261020/REPORT.md)。


## 数字规则

应用默认启用项目内 `../zh-itn/zh-itn/build/libzh_itn_bridge.so`，规则源码与构建说明在 `../zh-itn/zh-itn/README.md`。每个已完成 ASR 窗口先过 v90 保守型规则；有清晰数值语义的数字按日期、金额、数量、测量、时长、范围等上下文处理，精修模型走 `refine_with_numeric_rules` 数字锁定提示，数字兜底 normalizer 不参与该路径。成语、型号、电话或语境含糊的表达由规则保留原写法，这是“规则负责判定”，不代表强制把所有汉字数字都转成阿拉伯数字。

## 2026-10-22 ASR 显存调整

当前GPU0转录默认4096上下文、512 MiB KV、1024 prefill、单序列，保留CUDA graph和BF16。上述2026-10-20的8192/1 GiB参数是历史配置。三轮五样例对照最终文本15/15一致；ASR进程采样峰值3818→3606 MiB，240秒音频HTTP耗时中位数11.48→11.63秒。2048档长提示超限，eager明显变慢，均未采用。详见[完整记录](../results/asr_memory_optimization_20261022/REPORT.md)。
