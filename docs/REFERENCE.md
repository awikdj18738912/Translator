# 参考设计与已确定方案

本文记录 Translator 当前已确定的实现方向，以及对本机 `Codex-web`（AgenticASR）仓库中精修和实体库设计的只读核对结果。它描述设计依据，不代表 Translator 已经实现这些模块，也不包含性能或质量结论。

## Translator 的目标架构

Translator 面向普通电脑 CPU 本地运行。语音识别使用 `faster-whisper` 的 CPU `int8` 推理；翻译通过本地 Ollama API 调用 Qwen3 4B。翻译和译文精修是两个独立阶段：可以共用同一组模型权重，但分别构造请求、接收输出和处理结果。精修阶段是否复用该模型权重，需要在译文任务上评测后确定。

预期的数据流为：音频经 ASR 得到源语言文本，翻译阶段生成中文译文，精修阶段再检查和整理译文。精修应保留翻译已经确定的含义，不凭空增删事实；具体提示词、输入输出格式和验收标准应按 Translator 的译文场景设计。

## Codex-web 中可参考的设计

参考仓库：`/home/aim0/桌面/tan/ASR/Codex-web`。核对时没有读取其中的 `results/` 快照，也没有复制其代码。

- `system/refinement_protocol.py` 集中定义模型提示与响应协议。结构化模式要求模型返回 JSON 对象，动作是保留或一次局部替换；替换内容必须能在当前输入中定位。更严格的模式限制补丁长度、修改位置和原因类型。相邻窗口上下文明确标记为只读，防止模型把上下文复制进输出。
- `system/web_app.py` 的 `refine_update` 串起精修调用、结构化响应解析、保护内容恢复、结果校验、必要重试和安全回退。它记录模型输入输出和处理结果，便于审计。
- `system/entity_store.py` 使用 SQLite 保存实体规范写法、别名、类别、领域、启用状态和 `preserve` / `normalize` 策略；`system/manage_entities.py` 提供增删改查式的命令行管理入口。
- `system/entity_matcher.py` 负责实体候选匹配和风险判断；`system/protection.py` 负责把可信匹配标记为占位符、在模型处理后恢复并检查。`system/entity_pipeline.py` 将实体准备、可信基线、候选提示、结果恢复及拒绝回退组织成可复用流程。
- `system/configs/entity_matching.json` 配置匹配权重、自动接受门槛和候选数量限制；`system/README.md` 的 “Protected entity database and session memory” 章节说明词库、别名、模糊候选和可选音频复核。

适合借鉴的是职责划分和安全边界：实体词条与别名显式管理；精确命中可以按策略规范拼写；不确定候选先供模型判断，不直接全局替换；修改要可追踪；校验失败时回到可信输入或基线。Translator 可以据此设计术语表、翻译记忆和可审计的局部修订，但应按翻译任务重新定义匹配规则和校验内容。

## 不应直接移植的部分

Codex-web 的 Refiner 面向中文 ASR 转写后处理，提示词处理的是口语填充、重复、口误、自我修正、数字格式和标点。它的 checkpoint 是为中文 ASR 精修训练的模型；模型名称或其在 ASR 转写上的表现不能证明它适合外语到中文翻译，也不能证明它能稳定润色译文。若考虑复用该 checkpoint，必须在 Translator 的真实译文场景单独评测，并检查含义保留、术语一致性和错误引入情况。

实体匹配中的中文拼音相似度、声学 CTC 候选、原音复核和 ASR 分窗门控利用了中文发音或录音证据。外语源文到中文译文的术语选择受跨语言语义、上下文和翻译约定影响，不能用中文同音关系替代。可以保留显式术语及别名数据结构，但候选召回应基于源文术语、翻译记忆或跨语言语义证据；不确定时应保留译文并交由独立校验流程处理。

## 上游项目与接口资料

- [faster-whisper 项目与 README](https://github.com/SYSTRAN/faster-whisper)：CTranslate2 上的 Whisper 推理实现。CPU `int8` 是本项目选定的运行方向，实际速度、内存和识别质量尚未在目标电脑上测定。
- [Ollama Chat API 文档](https://docs.ollama.com/api/chat)：本地 Ollama 对话接口参考，Translator 的翻译与精修将作为两个独立阶段调用。
- [Qwen3-4B 模型卡](https://huggingface.co/Qwen/Qwen3-4B)：所选 Qwen3 4B 权重的官方模型页面。具体部署配置及译文质量仍需在目标环境验证。
