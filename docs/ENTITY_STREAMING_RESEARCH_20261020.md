# 译后流式实体校正：现状核对与优化建议

调研日期：2026-10-20。源码基线：`f9c64a8deb73b080a5ce096aab121e40f2f71136` 的当前工作区。本文是设计研究，没有改变运行代码、实体数据库或服务配置。

## 结论

可以在翻译之后流式匹配实体、执行校正，并把确定的实体保护起来再交给精修。当前工作台已实现其中一部分，下一步最有价值的是：

1. 修正别名冲突与重叠消解，避免确定性规则误替换。
2. 已经由规则完成的实体规范化，不再单独触发精修模型。
3. 让实体校正独立于三个标点块的模型精修调度，覆盖可修订的译文预览。
4. 将源文中的实体命中和对应翻译片段作为证据带到译后环节，处理意译、误译和漏译。

推荐组合是“源文实体证据 + 译文增量匹配 + 按版本执行局部替换 + 精修前保护 + 精修后校验”。Aho–Corasick 负责提高词典检索效率，跨语言实体是否对应仍需其他证据。以下设计建议是结合项目代码与文献作出的推断，尚未进行真实模型质量或性能对照。

## 当前实际链路

主要研究对象是 `workbench/` GPU 工作台，而非基础包 `translator/`。后者已经区分源语言别名、中文别名和源语言字段，其术语管理经验可以复用，但它不是当前工作台的流式调用链。

工作台入口 `workbench/app.py:create_workbench` 明确设置：`entity_fuzzy_mode="off"`、音频实体修复关闭、ASR 实体上下文关闭、CTC 关闭。因此文件存在不等于功能已启用。

```text
源文累计快照
  → AdaptiveTranslationSnapshots：活动组预览 / 闭合组 beam 3 确认
  → translation_web.refine_snapshot：仅把 stable_translation 送入精修 session
  → OfflineTextRefinementSession：会话开始固定词库
  → CumulativeWindowRefinement：三个标点块 + 相邻只读上下文
  → refine_update：数字规则 → prepare_entity_segment
  → EntityProtector：精确别名匹配、规范化、占位符保护
  → 门控决定是否调用 Refiner
  → 占位符恢复、完整性守卫、失败回退、按源译文片段复用显示
```

源码依据：

- `workbench/system/translation_web.py:212–252`：token 回调仅发布 `translation_preview`；session 收到已确认中文串，没有对应外语片段。虽然外围仍有 `snapshot.source`，没有传入实体准备与精修接口。
- `workbench/system/offline_text_refinement.py:25–46`：会话固定词库，fuzzy=off 时 matcher 为 None，仍保留 EntityProtector。
- `workbench/system/window_refinement.py:568–734`：完整组缓存按原始中文组和相邻上下文复用；未成组尾部走 `prepare_pending`。
- `workbench/system/web_app.py:1757–1948`：`pending_only` 仍先调用 `prepare_entity_segment`，之后才把门控置为 defer。因此“已确认 MT、尚未完整精修组”的尾部已经可以先规范实体。
- `workbench/system/entity_pipeline.py:122–170`：精确保护、可信基线、候选报告、再次保护、提示；默认工作台没有模糊候选。
- `workbench/system/protection.py:368–425`：检测占位符丢失、重复、未知标记、顺序变化和实体间句界丢失。应保留这些能力。

注意区分两种尾部：MT 尚未确认的预览尾部，目前没有经过实体校正；已进入精修 session 但未凑齐三个标点块的尾部，目前会进行规则实体校正。无需重写已有的 pending 功能。

## 确定性探针发现

使用 `EntityDefinition` 和 `EntityProtector` 构造临时内存词表，不修改实际数据库；纯匹配探针关闭通用规则保护以隔离观察实体行为。另用临时 SQLite 库和原样返回的假 Refiner 验证 session 路径，没有加载模型权重。

| 场景 | 设置 / 输入 | 当前观察 | 意义 |
| --- | --- | --- | --- |
| 残留源词可替换 | MOQ → 最低起订量；“请确认MOQ。” | “请确认最低起订量。” | 现有精确匹配有效 |
| 未登记的译法 | 仅登记 minimum order quantity / MOQ；“请确认最小订单数。” | 保持原文，未命中 | 只看中文别名无法找回源词证据 |
| 同别名冲突 | ABC 同时属于甲公司和乙公司 | 谁在定义序列前面就替成谁 | 需要显式歧义决策，不能依赖顺序 |
| 通用 / 业务领域冲突 | general 定义在前，trade 定义在后；同别名、同优先级 | 选中 general | 查询包含两个领域，但 protector 没有领域覆盖决策 |
| 重叠且优先级不同 | 高优先级“锂电”→“电池”，低优先级“锂电池”→“锂离子电池” | “锂电池。”→“电池池。” | 当前排序首先比较 priority，然后比较长度 |
| 不完整右边界 | ACME → 奥克米；先输入 ACME，后输入 ACMEX | 前者替换，后者不替换 | 直接把完整串匹配器用于 token 流会发生回撤 |
| 中文短别名 | 苹果 → 苹果公司；“苹果汁。” | “苹果公司汁。” | 精确字符串出现仍不能证明语义身份 |
| 未成精修组 | session 输入“请确认MOQ。”，非 final | 已变为最低起订量；模型调用 0；gate=defer | 已有实体处理早于模型精修 |
| 已解决实体触发模型 | 同 session 在 final 输入三个完整句，仅 MOQ 需要校正 | 假模型调用 1；gate=refine | 已规范化的别名仍会生成 entity hint 并唤醒模型 |

冲突原因见 `protection.py:_select_non_overlapping`：排序键为 `(-priority, -length, start)`，相同键保留原输入次序；没有同 span 不同 canonical 的拒绝分支。`EntityStore` 仅保证 `(canonical_text, domain)` 唯一，不保证不同实体不共享别名。基础包 `translator/terminology.py:protect` 已有“同 span 领域优先、冲突不自动替换”的处理，可在统一语义后复用。

这里是可构造输入下的行为复现，不是生产错误率。特别是“苹果汁”说明短别名需要上下文策略，并不意味着分词可以可靠地消除全部歧义。

此外，默认 matcher=None 时 `entity_matcher_latency_ms` 不能代表全部实体耗时：真正进行精确扫描的是 protector，而 `prepare_entity_segment` 会重复调用 protect。做优化评测时必须单独计量整个实体准备阶段，不能将当前 matcher 指标当作词典扫描成本。

## 论文提供的启发

已阅读下列论文的原始 PDF 和相关方法 / 结果部分。发表年份按论文元数据核对，不把网站最近抓取日期当作发表时间。以下结果均来自各自实验，与本项目的多语语音到中文场景不同。

| 论文 | 相关方法与边界 | 本项目可以吸收的做法 |
| --- | --- | --- |
| [Incorporating Terminology Constraints in Automatic Post-Editing，WMT 2020](https://aclanthology.org/2020.wmt-1.141/) | 把源文、MT 输出和双语约束用于 APE；比较自回归与 Levenshtein Transformer。部分英德基准术语保持率超过 95%，也指出模型并没有系统学会复制约束。 | 实体规范化后，精修还要继续保护和验证；不能只靠提示词要求保留。无需为了此结论立即训练新模型。 |
| [Domain Terminology Integration into Machine Translation: Leveraging Large Language Models，WMT 2023](https://aclanthology.org/2023.wmt-1.82/) | §2.3 先检查译文是否满足给定术语，再只对未满足片段执行术语 APE；实验包括德英、英捷、中英，含领域数据微调。 | 建立术语覆盖检查，只让未解决的问题触发额外模型修订；已满足约束直接放行。不能把其整体收益归给单独的译后替换。 |
| [Translate-and-Revise，CCL 2024](https://aclanthology.org/2024.ccl-1.82/) | Figure 2 / §2：修订输入包含源句、现有译文、双语约束和未满足约束，提供具体失败反馈。 | 建立 unresolved 列表，给模型准确指出缺哪一项；限制一次局部处理，失败即保留 / 标记，避免在线无限迭代。 |
| [Re-translation versus Streaming for Simultaneous Translation，IWSLT 2020](https://aclanthology.org/2020.iwslt-1.27/) | 比较重译与流式方法的质量、延迟、修订；§3.3 使用 normalized erasure 衡量可见文本回撤。 | 不能只统计首字延迟和最终正确率；还要统计实体输出回撤。缓存失效必须基于原始译文版本。 |
| [Multi-Agent Orchestration for Terminology-Constrained Machine Translation in Industrial Localization，ACL Industry 2026](https://aclanthology.org/2026.acl-industry.63/) | §3 提出领域分析、术语注入、译后编辑和审核；生产词表过滤源端未出现词、重复和冲突。实验为 WMT25 英德 / 英俄 / 英西 IT 文本，依赖多次大型云模型调用。 | 借鉴源端证据、词表冲突处理和术语审核的职责划分；本地实时链路优先用规则和现有调用实现，完整四模型流程成本不匹配。 |

这些论文支持“约束明确、选择性修订、继续验证、计量修订成本”的方向，没有证明任意已训练好的中文 ASR Refiner 能完成跨语言术语消歧。当前 AgenticASR-Refiner 的这项能力需独立验证。

## GitHub 实现核对

核对了三个项目的 README、许可证和关键实现。维护时间为调研时可见记录，不保证未来状态；GitHub API 请求遇到 403 限流，相关提交日期改由 GitHub commits.atom、Release / PyPI 记录核对。

| 项目 | 已核对的接口 / 代码 | 维护、许可、成本 | 建议 |
| --- | --- | --- | --- |
| [pyahocorasick](https://github.com/WojciechMula/pyahocorasick) | [README](https://github.com/WojciechMula/pyahocorasick/blob/master/README.rst)、[C 迭代器](https://github.com/WojciechMula/pyahocorasick/blob/master/src/AutomatonSearchIter.c)、[官方 API](https://pyahocorasick.readthedocs.io/en/latest/#automatonsearchiter-class)；`iter().set(chunk, reset=False)` 保留状态；`iter_long()` 提供最长非重叠匹配 | BSD-3-Clause；最近可见提交 2026-04-27，Release v2.3.0；C 扩展，源码安装需编译器，无额外神经模型 | 最适合做固定词典索引。使用普通 iter 召回候选，再由业务层消歧和处理重叠。不能认为 iter_long 自动完成了流式最长替换、边界判定和修订回滚。 |
| [FlashText](https://github.com/vi3k6i5/flashtext) | [README](https://github.com/vi3k6i5/flashtext/blob/master/README.rst)、[keyword.py](https://github.com/vi3k6i5/flashtext/blob/master/flashtext/keyword.py)：trie、最长查找、span 输出、replace_keywords | MIT；最近可见提交 2022-05-03；纯 Python，依赖轻 | 借鉴 span 输出和可配置边界。默认边界基于 ASCII 字母数字下划线，不能直接当中文词边界；完整串接口不能直接解决跨 chunk 状态。 |
| [SimAlign](https://github.com/cisnlp/simalign) | [README](https://github.com/cisnlp/simalign/blob/master/README.md)、[simalign.py](https://github.com/cisnlp/simalign/blob/master/simalign/simalign.py)：预训练表示的相似度矩阵与词对齐策略 | MIT；最近可见提交 / PyPI 0.4 发布 2023-11-07；依赖 PyTorch、Transformers、模型权重，另需核对选用权重许可 | 适合作为离线源译对齐参照，评估难例和生成 span 候选；不建议第一版逐 token 调用。词对齐不等于实体语义判定，多词 / 非连续对齐还需合并与拒绝策略。 |

上表许可分别核对各仓库根目录 LICENSE。未确认上述术语 APE 论文存在可直接接入本工作台的流式官方实现，不推荐名称相似但任务不匹配的仓库。

## 推荐的流式策略

### 1. 词库表达“什么证据允许替换”

在现有 canonical、type、domain、policy、priority 基础上，逐步增加：

- `source_language`、`source_aliases`：外语名称、缩写和已确认源端变体。
- `target_aliases`：中文别称与确认过的误译；与源语言别名分开。
- `case_sensitive`、`boundary_policy`：品牌 / 型号可严格区分大小写；短中文别名要求源端或上下文证据。
- `context_required`、`alias_provenance`：适用上下文、人工确认或其他来源。模型提议不能自动成为可信别名。
- 会话固定 `glossary_version`，记录规范化与索引版本。

源词未出现、别名歧义或证据不足时，保留原文并记录原因；不能因词典里有这个实体就向译文插入它。

### 2. 保留源文与译文片段的对应关系

保持最终修改发生在翻译之后。翻译之前 / 同时仅做源端实体检索，并携带 `source_span`、`entity_id`、语言和词表版本，作为译后判断的只读证据。它不要求改写 MT 输入或使用源端占位符。

`AdaptiveTranslationSnapshots` 目前内部持有 source window 与其翻译，适合扩展为每组输出：`window_id`、源片段、译文、源 / 译版本、confirmed 状态、源实体命中。随后由 `translation_web` 把相关组信息交给实体层。

这只是片段级对应关系，不能声称已获得词级对齐：源实体被意译为哪一段中文，还需目标别名、局部对齐或受限模型判断。无需逐 token 重跑神经对齐器。

例如源句 `Please confirm the minimum order quantity.` → MT `请确认最小订单数。`：当前纯中文词典若没登记“最小订单数”便不能处理；有源端 MOQ 实体证据后，可以仅在此译文组内尝试定位并提出局部规范化。若找不到可靠替换 span，则记录 missing，不直接在句尾追加“最低起订量”。

### 3. 区分检索、预览和提交

每次收到译文增量或快照就可以检索，但检索命中不等于立刻作出不可撤销替换。

- **token 预览**：独立的轻量 CPU 实体层处理；允许 provisional 显示。对仍可能扩展的 trie 后缀、缺少右边界的 ASCII token 保留短尾，等待更多字符。`ACME` 后面还可能出现 `X`；`最低` 后面还可能接 `起订量`。
- **已确认 MT 片段**：在明确边界、词库无冲突、证据满足时提交确定性实体规范化；不必等满三个精修块。
- **语义不确定候选**：按已确认组批量进入 unresolved 队列；与正常精修请求合并或进行一次受限局部判断。
- **final / EOS**：释放未决后缀，重新校验词边界和版本，再做覆盖审核。计时超限本身不是实体正确性的证据；超限可以保留原文或发 provisional，而非强制接受猜测。

如果用户接受预览回撤，匹配到后即可标记 provisional；如果要求显示后不再变，就必须等待上游确认及边界闭合。这两种体验存在延迟取舍，算法不能消除源译文修订带来的不确定性。

### 4. 正确处理可修订快照

pyahocorasick 的跨 chunk 状态适合追加流；当前 MT 还会修订已显示内容，不能把新旧整段快照直接当成追加 chunk。

建议先以翻译组为单位缓存：组内纯追加可继续扫描；组内容变化则丢弃该组候选和替换结果，重扫该组及必要边界邻接区。第一版宁可重扫一个有上限的活动组，也不要为了复杂增量机制产生失效错误。

进一步优化时，对旧新原始译文计算 LCP，从首个变化点向前回看最多 `max_alias_length - 1` 个字符，并保留左边界字符；撤销此范围内及向后相交的实体，再重扫。若还有语义上下文依赖，失效范围需扩大至依赖组。被修改后的字符串不能作为下一次匹配的原始输入。

字符规范化若采用 casefold / NFKC，必须维护到原始字符 span 的映射；其长度可能变化。Python code point、UTF-8 byte offset、浏览器 UTF-16 offset 也应明确区分。

### 5. 冲突先消歧，替换只作用于明确 span

推荐决策顺序：

1. 按启用状态、语言和选中领域过滤；同一语义适用范围内才比较候选。
2. 同 span 若 general 与选中领域均有定义，使用明确的领域覆盖规则。
3. 同 span 仍有不同 canonical 且无法消歧，输出 ambiguous，不靠词条顺序或 priority 猜。
4. 重叠 span 优先完整实体和语义证据，防止短别名切开已确认的长实体；业务 priority 仅作为规则明确允许时的次级依据。
5. 中文短别名、常用词、PERSON / ORG 等高风险项，要求源端或明确上下文证据。独立分词可以否决不合理候选，不能独立证明实体身份。

不要把策略简化为对全文反复 `.replace(alias, canonical)`。每个替换绑定原始译文版本和 `[start,end)`；按原文 span 一次构造结果，或从右向左应用，防止位置偏移以及 A→B→C 级联。

建议的内部替换记录包含：`window_id`、`translation_revision`、`glossary_version`、`target_start/end`、`observed`、`canonical`、`entity_id`、`evidence_type`、可选源 span、`state`、拒绝原因与输入摘要。接受前校验当前目标串在该 span 仍等于 observed。数字 ITN 等其他处理若改变字符位置，也要用阶段版本和 offset 映射衔接。

### 6. 规则已解决与模型待判断分开

`EntityProtector.refinement_hints` 当前会给“别名已被确定性规范化”的项生成 hint，而 regex gate 遇 hint 即调用模型。这与实体保护本身是两件事。

建议分成：

- `resolved_normalizations`：已完成，进入保护与审计，不单独触发模型。
- `unresolved_candidates`：仍需消歧，只有这类实体问题才参与模型门控。

有其他语言修整信号时仍可正常调用精修；已解决实体继续用占位符保护。保留数字、否定、条件、型号和句界检查，拒绝后回到确定性可信基线。

不要直接把当前拼音模糊模式从 off 改成 on 来补跨语言缺口：它主要针对中文 ASR 形近 / 音近错误，不能据此证明两个跨语言词语对应同一实体。

### 7. 精修后的覆盖审核以 occurrence 为单位

为每个已确认的源端实体出现位置建立账本，记录目标对应位置和 fulfilled / missing / ambiguous 状态。不能只检查“标准名是否在全文出现”：一个实体重复出现、错绑定到另一句或另一个对象，都会误导全局计数。

中文与外语可能有合法的重排、省略或合并，不能盲目要求源目标实体严格同序、同次数。当前占位符的顺序检查约束的是中文精修输入与输出；跨语言覆盖需要单独定义有效映射。

源实体没有出现在原始译文里时，单纯替换无处可做。第一版把它标为 missing；后续只有带源句的受限译后编辑模块通过验证，才允许恢复缺失内容。现有禁止从 hint 凭空引入实体的守卫应继续保留。

## 实施顺序与评估

| 阶段 | 改动重点 | 预计解决的问题 | 验证重点 |
| --- | --- | --- | --- |
| P0 | 别名冲突拒绝、领域覆盖、重叠规则；resolved 不再单独触发模型 | 明确误替换与可避免的模型调用 | 顺序无关性、长短词、重复实体、上下文歧义、保护恢复 |
| P1 | 词库预编译索引 + 活动组实体预览 + 原始 span / 版本账本 | 低延迟实体显示、词库规模扩大 | 任意 chunk 切分、回撤 / 重写、边界、终态一致性、无级联 |
| P2 | 双语词条与翻译组元数据；仅未解决约束进入局部修订 | 意译、误译与缺失实体 | 源译对应、人审 precision / recall、错误引入、额外模型开销 |

P0 建议先以现有 regex 检索实现，避免同时改变匹配算法和决策语义。P1 再在相同候选语义下比较 Aho–Corasick 与 regex，不能预设 CPU 匹配就是当前系统瓶颈。

实验按同一批源文、同一 MT 输出和相同 chunk / 修订轨迹做消融：A=当前实现；B=P0；C=B+流式索引与预览；D=C+源端证据与局部修订。额外加入无术语负样本，不能只测有词必替的正样本。

关键指标：

- occurrence 级替换 precision / recall、误替换率、missing / ambiguous 比例，按语言、实体类别和名称长度分层。
- 数字、型号、否定、对象关系是否发生不当改变；术语字符命中不能替代语义审核。
- 原始 MT 到达至首次正确实体显示、最终确认的 p50 / p95 延迟，区分字符回放与真实音频测量。
- 模型调用 / 重试数量、实体层完整 CPU 耗时、索引构建时间、内存；词库规模按 100 / 1,000 / 5,000 别名以及实际词表分布分层。
- normalized erasure、每个实体的显示改写次数、final 与一次性处理的一致性。

本次运行了内存匹配探针、临时 session 探针，并执行相关既有测试；结果记录见同目录 `ENTITY_STREAMING_RESEARCH_20261020_VALIDATION.md`。这些检查只验证代码行为，尚不能给出生产延迟收益或模型准确率提升承诺。
