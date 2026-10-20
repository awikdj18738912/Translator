# 翻译接入后的调度优化

日期：2026-10-19。本轮没有启动Web/ASR服务，没有加载GPU或CPU模型，也未改动模型权重。结果来自代码核对、已录日志的原文重放、模拟模型自动化测试和离线浏览器检查。

## 已完成

1. **解耦三阶段**：ASR/文本输入、翻译、精修各自独立推进。ASR不再等待当前翻译和精修完成后才发送下一块。输入消息通过单一发送锁保存/发布。
2. **合并积压**：每阶段只有一个待处理快照，新的中间状态替换旧的待处理状态。已经开始的调用不会被不安全地终止；其过期结果会丢弃。final使用独立终态标志，必须处理且仅发布一次。
3. **保留版本归属**：每次输入有递增revision。服务端和前端都拒绝旧结果覆盖新输入；译文仍是完整快照，不追加旧译文。
4. **稳定分组缓存**：外语按三个完整句固定分组、最长700字符；完成组可缓存，活动尾部改变时重新翻译。源文回改会使对应缓存失效。相同输入不重复翻译；小数、常见缩写和Unicode分段有回归覆盖。
5. **精修活动范围**：非final只将完成组中文交给现有Codex-web精修窗口，活动尾部先显示原译文；final强制收尾全量。复用原实体与数字守卫，不伪造ASR置信度，不对所有活动尾部反复精修。这里“完成组”是分组策略，不是ASR置信度保证。
6. **取消与资源**：客户端断开会取消管线，在途线程调用完成后才释放会话占用，清理ASR会话和上传文件，避免旧调用与下一任务并发争用模型。
7. **文件回放与日志**：音频默认移除每块额外500毫秒等待。显式回放间隔按输入时钟调度，不再叠加模型计算时间。前端每事件只序列化一次，合并渲染最近60条；下载保留全部事件。

## 日志重放：输入工作量

使用原来122.415625秒音频的48个已录文本快照，最终源文1366字符。数据固定于 `source_work_comparison.json` 中的路径及SHA256。翻译函数为确定性模拟函数；没有重跑音频、模型或改变日志内容。

| 所有48帧都处理时的分段策略 | 翻译函数调用 | 累计送入的源文字符 |
|---|---:|---:|
| 原700字符合并窗口 | 47 | 16088 |
| 新三句固定组 | 57 | 5550 |

累计源文字符减少 **65.5%**。分组更细的固定调用开销增加，调用数在本项逐帧处理实验中反而上升；不能单凭字符量减少宣称GPU更快。实际新调度在积压时会跳过中间快照，但真实处理帧数取决于ASR到达节奏和模型耗时，本轮未测其GPU数值。

原日志中的耗时基线为：ASR约5.757秒、翻译约30.311秒、精修及校验约71.523秒、真实精修调用151次。以上只记录旧版本，不是优化后耗时，不与模拟重放时间作速度比。

## 验证

- 74项相关测试通过，包含窗口/前端路由回归及新增并发调度测试；详见 `tests.log`。
- 特别验证：翻译函数被事件阻塞时，模拟ASR仍能发送完所有音频块和final；最终只发布最新结果，待处理槽位有界。
- 慢精修不阻塞翻译推进；源文回改、final保护、模型异常、线程取消后等待实际返回均有测试。
- 浏览器加载真实HTML/JS，使用模拟fetch/WebSocket（不监听端口、不启动服务）。验证过期translation/final不会覆盖新版本；页面仅展示60条事件，下载含全部106条，脚本错误0。见 `frontend_offline_check.json`。
- Python编译、Node语法和变更空白检查通过。依赖包有原有弃用提示，未影响测试。

复现：

```bash
cd /home/aim0/桌面/tan/ASR/Translator
python scripts/analyze_translation_optimization.py
python -m pytest ../Codex-web/tests/test_translation_scheduler.py ../Codex-web/tests/test_translation_web.py ../Codex-web/tests/test_translation_windows.py ../Codex-web/tests/test_offline_text_refinement.py ../Codex-web/tests/test_window_refinement.py ../Codex-web/tests/test_window_adjacent_context.py ../Codex-web/tests/test_web_window_context.py -q
PYTHONPATH=../Codex-web/.runtime/functional_test_deps PLAYWRIGHT_BROWSERS_PATH=../Codex-web/.runtime/playwright_browsers python results/translation_optimization_20261019/frontend_offline_check.py
```

## 尚未验证

没有测优化后的真实端到端秒数、GPU吞吐、峰值显存或翻译质量。三个源句组与原700字符窗口的上下文不同，可能影响跨窗口改口、指代和术语翻译，后续必须用真实模型对照。非final精修会等待完成源组和原K=3所需上下文，因此中间精修可能更晚，但译文先显示，输入继续推进。

本轮没有调整beam、精度或Refiner KV cache，也没有修复此前数字空格造成的误拦截；这些是独立问题，需要单独验证。旧离线CLI串行回放脚本保留用于对照，本轮优化的是浏览器翻译入口。

## 文件

- `Codex-web/system/translation_scheduler.py`：最新快照槽位、三阶段异步调度、线程调用清理。
- `Codex-web/system/translation_windows.py`：源文三句组缓存和统计。
- `Codex-web/system/translation_web.py`：浏览器入口接入调度、稳定中文精修、无额外文件等待。
- `Codex-web/system/web/translation.js`：版本过滤、有限日志渲染、音频无人工间隔。
- `Translator/docs/CODEXWEB_TRANSLATION_FRONTEND.md`：更新后的行为说明。

参考：[Python asyncio任务与取消](https://docs.python.org/3/library/asyncio-task.html)、[Starlette WebSocket测试](https://github.com/Kludex/starlette/blob/main/tests/test_websockets.py)。
