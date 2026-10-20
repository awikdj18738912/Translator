# Codex-web 多语音频翻译与中文精修

页面地址：`http://127.0.0.1:8081`，点击页首“多语翻译”，进入新增面板。

- **文本回放**：选择源语言、粘贴外语文本，按块模拟持续输入。
- **音频文件**：选择源语言和录音文件，分块执行 Qwen3-ASR → LMT → 中文 Refiner。
- **三栏结果**：外语转写/原文、中文译文、守卫处理后的精修文本。
- **可见状态**：模型调用次数、单次翻译和精修计算时间、守卫接受或回退原因；可展开并下载本次事件日志。

音频最长300秒，文件最大50 MB，音频由PyAV解码为16 kHz单声道PCM，再按2秒分块回放。页面等待每次处理完成后继续，存在顺序背压；不是无等待的实时麦克风同传。原页面的麦克风/中文转写入口仍是原链路；多语音频请使用新增面板的音频文件入口。

源语言支持英语、法语、德语、西班牙语、俄语、日语、韩语、阿拉伯语。当前选择是显式源语言，不把ASR置信度伪装成译文置信度。音频文件默认不添加人为回放等待；文本回放保留500毫秒输入间隔，该间隔按输入时钟调度，不叠加每阶段计算时间。

## 本机服务

ASR服务：`http://127.0.0.1:8766/health`。

```bash
cd /home/aim0/桌面/tan/ASR/Codex-web
CUDA_VISIBLE_DEVICES=1 /home/aim0/data/conda/envs/qwen3-asr/bin/python \
  -m system.qwen_asr_stream_server \
  --model /home/aim0/data/models/ASR/Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.30 --max-model-len 8192 \
  --vad silero --vad-model /home/aim0/data/models/ASR/silero_vad.onnx \
  --segmentation-mode vad_finalize --port 8766
```

Web、翻译、精修服务：

```bash
cd /home/aim0/桌面/tan/ASR/Translator
bash scripts/start_codexweb_translation.sh --gpu 1 --port 8081
```

服务启动时加载模型，`/api/translation/health` 返回 `ready:true` 表示翻译已加载，`audio_ready:true` 表示ASR可连接。页面每10秒刷新健康状态。端口已有健康服务时不要重复加载同一份模型。

这台机器的模型都运行在GPU 1：Qwen3-ASR-0.6B、LMT-60-0.6B（CTranslate2 INT8/FP16）、AgenticASR-Refiner（原始配置FP32）。GPU 0其他任务保持运行。

## 词库和日志

Web首次启动会将原Codex-web实体库以SQLite只读备份复制到 `results/codexweb_translation_frontend_20261019/entities.db`。在新页面管理词库会修改本次副本，不修改原库。trade领域首次加入iPhone/SUS304/AcmeFlow/阀门/不锈钢几个演示实体。词库在新会话开始时快照，作用于中文精修，尚不是跨语强制翻译词典。

服务日志：`.runtime/translation_web/web.log`、`asr.log`；PID记录在同目录。多语会话日志在 `results/codexweb_translation_frontend_20261019/sessions/`：`*.jsonl` 保存阶段事件，`*.refiner.json` 保存原精修输入、候选和守卫痕迹。页面“精修日志”原链接仍是旧音频Web链路的日志页面；新面板应使用自己的事件日志/下载按钮。

每个会话的ASR、翻译、精修由独立任务推进，每个阶段只有一个待处理快照，积压时合并旧版本；仍只接受一个翻译会话。取消后会等待正在执行的阻塞调用返回再释放资源。上传待处理文件最多两个、10分钟过期，在后续请求时清理。

外语按完整的三句固定分组，最长每窗700字符，未满组或未完成的活动尾部可重译。原文或语种变化会使相应缓存失效；这里的“已完成组”来自文本边界策略，不是ASR置信度保证，也允许后续回改。非终态只将完成组的中文交给精修，活动尾部先显示译文，终态强制处理全量；跨分组改口理解和最终翻译质量仍需真实模型复测。页面忽略过期版本的翻译/精修结果，日志只显示最近60条，下载包含完整记录。

精修保留原数字守卫：此前发现的“500 件”与“500件”的单位提取误拦截尚未修改，候选可能因此回退。模型也可能保留上游错译或改变语义，因此不能把“守卫接受”视为业务事实正确。

## 2026-10-19 调度优化

本次只做源码、离线原文重放和自动化测试，未启动服务或加载模型。实现版本 `latest_snapshot_v2`，具体验证见 `results/translation_optimization_20261019/REPORT.md`。旧报告中的原运行速度不能当作优化后速度。
