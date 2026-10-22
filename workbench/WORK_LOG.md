# 项目工作记录

## 当前代码基线（2026-10-19）

- 默认入口 `start.sh` → `app.py`，ASR入口 `start_asr.sh` → `system.qwen_asr_stream_server`。
- 翻译输出：真实token预览＋相同窗口beam3确认；稳定窗口缓存；窗口间不插入人工换行；final复用同一窗口策略。
- 精修：固定三个标点块、相邻上下文；精修组完成后发布进度；实体库与完整性守卫保留。
- 门控：默认regex，不依赖ASR置信度；学习型门控、模糊实体、声学辅助关闭。
- 用户运行约束：服务被他人中断时不自动重启；后台Restart=no。

## 已有修复记录

- [精修显示停滞诊断](results/REFINER_STALL_REPORT.md)
- [翻译token预览优化](results/TOKEN_STREAM_OPTIMIZATION.md)（历史阶段方案，最终调度以当前源码和后续修复记录为准）
- [换行与重复精修修复](results/REFINEMENT_REUSE_FIX.md)
- [中文译文正则门控](results/REGEX_GATE_CHANGE.md)

## 2026-10-19：全项目三模型调度审查

只读审查源码及可选实现，记录默认实际链路、并发/排队、缓存、取消与收尾。应用代码没有修改；未加载模型权重；没有启动、停止或重启服务。136项既有测试通过，另用确定性探针复现窗口末尾空白缓存失效、窗口数与真实解码次数不一致、first缩写误判，以及整版snapshot交接导致精修等待。

主结论：当前是异步三级流水线，尚无统一GPU任务调度器。优先补齐计数/时序、修复窗口缓存和切句、增量交接确认前缀，再评估GPU优先级。

详见 [调度审查报告](results/scheduling_audit_20261019/REPORT.md)、[源码清单](results/scheduling_audit_20261019/SOURCE_INVENTORY.json)。报告列出的建议尚未作为代码修复实施，不应把测试通过理解为所有新发现均已修复。

## 2026-10-19：两次启动后立即退出的诊断

16:55:47和16:57:12收到用户启动授权后，systemd报告启动，但后续检查两个unit均消失，端口关闭。本轮排查没有找到这两次的明确Stopping/失败、Python异常或匹配OOM记录，当前进程日志mtime早于两次启动，无法归因为“被其他人关闭”。临时unit使用--collect自动回收，退出状态未保留；这造成当前诊断证据不足。没有为排查重新启动服务。证据在results/startup_exit_diagnosis_20261019/。下一次如获启动授权，应先保留unit的退出码和新启动stdout/stderr，再判定原因，而非继续用同样的无留存方式尝试。

## 2026-10-19 17:01：保留退出状态后的启动尝试

用户再次授权启动。此次不用--collect，设置Type=exec、RemainAfterExit=yes、Restart=no，并将stdout/stderr写journal。启动后两主进程均为ExecMainCode=2、ExecMainStatus=15，即被SIGTERM终止，MainPID=0。active(exited)是RemainAfterExit保留的单元状态，不代表服务可用。信号发送方尚未识别；未再次自动启动。证据results/startup_exit_diagnosis_20261019/signal15_startup.json。

## 2026-10-19：翻译策略与三项界面统计

默认改为短句合组/长句提前闭合的adaptive策略；活动尾预览，闭合组beam3确认缓存；逐字母拼写在MT和精修前后保护。新增整体用时、实际精修请求次数、实际门控keep累计，纠正局部审核漏计；缓存/defer不算门控skip，结束和异常时计时冻结。

148项后端测试、前端模拟与真实MT+Refiner进程内重放通过，未启动或重启服务；当前运行后端未重载。44条多语种参考无普遍提升，8条外贸构造参考的合组策略改善部分案例，均不足以称准确率。完整记录与限制：[自适应策略及统计](results/ADAPTIVE_TRANSLATION_AND_METRICS.md)。

## 2026-10-19 18:58：用户授权重载后端

仅重启网页进程，ASR进程182235保留。新网页PID261036，health确认translation_strategy=adaptive、regex门控、audio_ready=true。在线短文本回归：My name is Tasha Starski. S T A R S K I. Please address the documents to me. → 我叫塔莎·斯塔斯基。S T A R S K I。请将文件寄给我。精修保持拼写；session_metrics返回0次模型、1次门控keep。Restart=no，无自动重启。证据results/adaptive_restart_check/events.json。

## 2026-10-19：整体用时旁增加音频时长

顶部统计四列，音频时长紧邻整体用时，手机两列。使用已有后端audio_total_seconds解码后秒数，不新增音频解码或模型调用。文本模式不适用；任务切换、文件切换、上传失败与取消状态已模拟验证，含桌面/手机布局，无浏览器错误。只修改静态界面，无服务重启；证据results/audio_duration_ui_check.json。

## 2026-10-20：用户要求以ASR 0.6B启动

核对本项目启动脚本原本即Qwen3-ASR-0.6B，当前两服务已停止，按用户授权启动并在进程参数和引擎日志核实实际模型。首次ASR启动在KV cache profiling阶段报预算-4.18GiB（不是SIGTERM），因此添加可选max_num_seqs/max_num_batched_tokens参数并在本项目脚本限制为1/4096，保留context8192和显存预算0.30；修正后KV预算3.55GiB，接口ready。没有更换翻译或精修权重，也没有修改其他程序。Restart=no保持，非自动重启循环。153项测试通过，实际浏览器文本与5.97秒音频全链路通过。模型是0.6B，不声称已完成1.7B对照质量评测。证据results/asr06_startup_20261020/。

## 2026-10-20：ASR KV缓存缩减实测

用户授权测试适合的显存预算。保留Qwen3-ASR-0.6B BF16、context8192、max_num_seqs=1、max_num_batched_tokens=4096与现有VAD，新增--kv-cache-memory-bytes CLI校验/透传及健康字段。原3.55GiB KV池对比1024MiB和896MiB，短英语和240秒拼接中文语音逐块/最终文本一致；长样例14.73/14.56/14.32秒HTTP耗时。后两档推理后均3810MiB，原6098MiB。768MiB被vLLM容量校验拒绝：8192上下文需要约896MiB，exit-code=1，不是外部SIGTERM或推理OOM。

选1GiB为默认，可由ASR_KV_CACHE_BYTES覆盖；未缩短上下文或量化权重。启动过程存在data盘I/O等待，首轮1024启动等待脚本超时后同一进程继续就绪，未重复启动。各档均为明确的手工试验，最后恢复1GiB并移除临时systemd覆盖，Restart=no，其他用户GPU任务未触碰。全套156测试通过；真实浏览器文本/英语音频三模型链路、桌面/手机和实体弹窗通过；预热短英语三次0.59~0.71秒。报告明确小样本、拼接音频、冷启动和显存采样限制，不宣称全面准确率或速度提升。详见results/asr_cache_sweep_20261020/REPORT.md与FINAL_STATUS.json。

## 2026-10-20：精修模型显存缩减可行性只读核对

用户询问精修模型能否缩小。读取本地safetensors文件头确认1,080,632,832参数、219个F32 tensor，纯权重4.026GiB。当前Transformers loader dtype=auto跟随config.float32，没有vLLM式预分配KV池；config.use_cache=false。BF16纯权重理论2.013GiB，可先作精修质量/实体保护/自我更正与预热延迟对照，尚未测试或切换，不能承诺质量无损或提速。网页进程实测5232MiB包含LMT与Refiner，不能全部归给精修。记录results/refiner_memory_inventory_20261020.json；没有修改推理代码、启停服务或加载额外模型。

## 2026-10-20：用户授权试用BF16精修

新增可配置refiner_dtype并在workbench默认bfloat16，通用底层保留auto兼容。模型权重文件不修改；仅加载精度改变，不改提示词、use_cache、门控、实体保护或生成参数。独立FP32/BF16各11例，实际参数4.026→2.013GiB，测试后独立进程4574→2436MiB，合计推理5710→5048ms；10例原始输出相同，逐字母名字直接精修存在拼写差异，两者均不忠实。原“抱歉”语义回退能力未因精度改善。

全套158项测试通过；空闲时手工重载网页PID889876→971990，ASR PID936611保留。网页实测3086MiB（原5232），项目两GPU进程合计6886MiB/6.72GiB，其他用户GPU任务不计。真实浏览器文本、音频、桌面/手机、实体弹窗通过；拼写+数量例实际3次精修，姓名保护成功，数量区占位符减少触发既有守卫回退，不声称全部精修接受。/health确认bfloat16与2161265664参数字节；Restart=no，无自动重启。完整证据results/refiner_bf16_20261020/REPORT.md。

## 2026-10-20：顶部新增转录用时

在整体用时后、音频时长前添加转录用时卡片，桌面五列、手机两列。后端音频任务使用独立累计asr_total_ms，计量start/chunk/finish请求（含失败请求/接口等待），排除上传/解码/回放sleep/cancel清理；send层附上共享计数，避免快照合并、重复流水线事件造成漏计或重复。前端累计取max不倒退，不自走计时；文本不适用，新任务/切换文件/模式重置，取消/断线保留已收值。

159项测试通过，音频快照合并/失败请求计量、前端重复旧事件/重置取消断线/五卡布局验证通过。手工重载空闲网页PID971990→977803，ASR936611保留，BF16精修与1GiB ASR缓存未改，Restart=no。真实5.97秒英语音频页面显示整体1.4s、转录1.0s、音频6.0s，final累计1019.855ms、全事件单调、无浏览器错误。详见results/asr_time_ui_20261020/。
## 2026-10-20：顶部五项统计固定同排

顶部统计在所有视口保持五列同排；卡片最小宽度120px，800px视口五卡均可见，390px视口仅统计容器内部横向滚动，页面本身不横溢出。标题和数值单行显示，说明最多两行。只更新统计区CSS及CSS资源版本，并把既有browser_offline布局断言改为1440/800/390检查；离线验证截图和测量记录保存在results/metrics_single_row_20261020/。


## 2026-10-20：取消上传音频300秒时长限制

用户要求去除时长限制。decode_audio删除max_seconds与采样数阈值，health.max_audio_seconds=null，保留50MiB文件限制和2秒分块。README现行说明同步。12项路由测试通过，系统Python缺PyAV跳过1项；在生产agentic-asr/PyAV18.0独立解码301秒WAV，完整4,816,000samples验证通过，不据此声称长录音完整三模型质量已评测。空闲时手工重载网页977803→1042739，ASR936611保留，模型策略不改，Restart=no；在线health确认无时长上限、50MiB文件上限、ASR就绪。证据results/audio_duration_unlimited_20261020/。

## 2026-10-20：启用数字规则库并锁定精修数字

用户要求数字规则接管格式处理、精修不再改数字，也不走Python数字兜底。把Codex-web携带的zh-itn/zh-itn规则源码与已构建桥接库复制到独立Translator项目（2.3MB）；默认create_workbench显式加载本地DEFAULT_LIBRARY。v90规则先规范化已完成窗口；仅明确触发口癖/自我更正等非数字清理时，精修调用refine_with_numeric_rules，约束保留数值表面；数值专属窗口由规则直接keep；zh-itn active时ContextualNumericNormalizer兜底路径不运行。健康接口报告zh_itn_enabled、rule_pack=zh-itn-strict-conservative-v90和protocol=v1-rule-owned-number-surface。

纯Python/ctypes规则探针确认数量300、12.5美元、3750美元、5到6小时及自我更正数值可转换，成语三番五次保持原文。真实网页文本链路输入外贸数字样例，规则改写300/12/3750并识别self-correction；模型输入已经含300且输出未更改，numeric_fallbacks为空；仅1次非数字清理精修，两个纯规则组跳过精修。保守库对“五十美分”未产生规则候选，按原中文保留，这显示规则覆盖有边界，不将所有字面数字一律强转。实测已保存probe_summary和事件/精修轨迹。

空闲时手工重载网页PID1042739→1050586；ASR PID936611未重载。Restart=no，无自动重试。当前health确认规则包已加载，在线服务正常。全套pytest未运行；本轮使用原生库样例和真实浏览器请求做功能验证。详见results/numeric_rules_activation_20261020/。

数字门控文档同步：当前zh-itn模式下，纯数字规范化由规则包处理并由regex gate keep，不将“中文数字规范化”信号误记为精修触发；明确的重复/自我更正等仍由语言精修处理，但规则锁定保留数字表面。

## 2026-10-20：将数字规则移到独立项目根目录

确认 `Translator/` 才是有 `pyproject.toml`、`translator/`、`tests/` 的独立项目根目录，`translation_web/` 是其中的 GPU 工作台模块。将完整 `zh-itn/zh-itn` 从工作台子目录移到根目录，与规则桥接默认路径同步；补充根 README 说明模块边界并修正规则包内过时的 Codex-web 脚本路径。空闲时手工重载网页进程以载入新路径，health再次确认v90规则包正常；ASR未重启，Restart=no。

## 2026-10-20：工作台目录正式命名

项目根目录保留基础Python包 `translator/` 和公共数字规则 `zh-itn/`；GPU网页工作台目录从临时名 `translation_web/` 改为 `workbench/`，清楚体现它是项目的一个应用模块。相关服务路径、根README和工作台说明随之更新。


## 2026-10-20：工作台目录迁为正式模块名

用户指出 `translation_web/` 像临时目录。项目根 `Translator/` 已有 `pyproject.toml`、`translator/`、`tests/`，因此把 GPU 工作台目录更名为 `workbench/`，根目录数字规则包保留为 `zh-itn/`。用 transient systemd drop-ins 更新服务工作目录和ExecStart；重建两个临时unit，`Restart=no`。网页PID 1062402 / ASR PID 1059628 均从 `workbench/` 启动，ASR在换目录时按授权手动重载一次。网页health确认BF16和v90规则包，translation health ready/audio_ready，当前目录中已无 `translation_web/` 文件夹。


## 2026-10-20：Codex重启后页面离线修复

根因：web/ASR通过`systemd-run --user`创建临时单元（FragmentPath在`/run/user/1000/systemd/transient/`，UnitFileState=transient）。它们不是磁盘上的持久服务定义，用户管理器清理时unit也会丢失。将等价unit写入`~/.config/systemd/user/`，模板归档到`workbench/deploy/systemd/user/`，启用default.target，两个服务仍Restart=no。当前Linger=no，完整注销后下次登录启动；Codex客户端重启不影响用户systemd管理器。按目录迁移要求手工停/启了一次ASR和网页，现web PID1095866 / ASR PID1094509，8083/8766健康，WorkingDirectory均workbench，规则包v90已加载。完整记录见results/codex_restart_persistence_20261020/REPORT.md。

## 2026-10-20：切换为中文ASR → 中文精修 → 外语翻译

按用户要求调整当前工作台全链路顺序。ASR固定Chinese，选择框改为八种目标外语；沿用原数字规则、中文精修、实体保护及正则门控。只有已精修前缀交给LMT，未精修尾部不提前翻译；每组精修完成即可交接。新增有界中文优先调度，独立维护ASR/翻译修订号，页面改为中文转录、中文精修、外语翻译三列并保留五项统计。LMT使用明确Chinese→目标语言模板，token预览后beam3确认。

175项测试通过、1项因测试解释器缺少PyAV跳过；生产音频解码及真实GPU链路通过。浏览器完成八语种文本和12秒中文音频→英语共9次验证，含桌面/手机布局；异常恢复、取消时模型锁持有也已覆盖。只手动重载网页一次，现web PID1190470，ASR PID1094509保持；8083/8766就绪，两个服务Restart=no。

实测发现翻译模型会把正确中文金额123万7851元错译成1230,007851 yuan；中文数字规则与精修保护正常，下游外语数字尚无校验。少量功能样例不构成质量或准确率评测，未宣称全面可靠。根README、workbench README及状态记录已更新，早期translator基础原型和通用旧方向接口保留兼容。完整报告与事件在项目根 `results/chinese_first_pipeline_20261020/REPORT.md`。

2026-10-22：完成ASR GPU0显存优化四档实测，采用4096上下文/512MiB KV/1024 prefill，保留CUDA graph与BF16。15次最终文本与8192原配置逐字一致，进程采样峰值3818→3606MiB；240秒耗时中位数11.48→11.63秒。2048长提示超限；eager约3.69倍耗时。启动参数9项测试通过，临时实验覆盖清理，ASR与工作台健康。证据：../results/asr_memory_optimization_20261022/REPORT.md。
