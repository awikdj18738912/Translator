# LMT → Codex-web 离线文本流式联调

真实GPU模型：LMT INT8/FP16翻译 + Codex-web原始Refiner、固定K=3分窗、实体保护与守卫。
输入是外语文本快照回放，没有运行音频转录/麦克风/WebSocket音频链路。每次重译的中文完整快照替换上一版，不追加旧译文。
词库为本次独立副本/演示库；精修只看到中文，不能保证修复上游错译。

12个场景、47次更新；中间更新实际进入精修的次数：1。最终守卫接受11/12，其余保留回退结果；接受不等于业务事实正确。

| 场景 | 最终外语原文 | 翻译中文 | 精修输出 | 精修接受 |
|---|---|---|---|---|
| en_order | We need 500 SUS304 valves, at 12.50 US dollars each, with delivery by November 15. | 我们需要 500 个 SUS304 阀门，每个 12.50 美元，11 月 15 日交货。 | 我们需要 500 个 SUS304 阀门，每个 12.50 美元，11 月 15 日交货。 | True |
| en_correction | Do not ship 500 units. Sorry, make that 300 units, and ship only after the deposit arrives. | 请勿发货 500 件。抱歉，将数量减少到 300 件，付款后才发货。 | 请勿发货 500 件。抱歉，将数量减少到 300 件，付款后才发货。 | False |
| fr_order | Nous avons besoin de 500 vannes SUS304, à 12,50 dollars américains l'unité, avec une livraison au plus tard le 15 novembre. | 我们需要 500 个 SUS304 泵，每个 12.50 美元，最迟于 11 月 15 日交货。 | 我们需要 500 个 SUS304 泵，每个 12.50 美元，最迟于 11 月 15 日交货。 | True |
| de_order | Wir benötigen 500 SUS304-Ventile, zu einem Stückpreis von 12,50 US-Dollar, mit Lieferung spätestens am 15. November. | 我们需要 500 个 SUS304 阀门，单价为 12.50 美元，最迟将于 11 月 15 日发货。 | 我们需要 500 个 SUS304 阀门，单价为 12.50 美元，最迟将于 11 月 15 日发货。 | True |
| es_order | Necesitamos 500 válvulas SUS304, a 12,50 dólares estadounidenses por unidad, con entrega a más tardar el 15 de noviembre. | 我们需要 500 个 SUS304 阀门，每个 12.50 美元，最迟将于 11 月 15 日交货。 | 我们需要 500 个 SUS304 阀门，每个 12.50 美元，最迟将于 11 月 15 日交货。 | True |
| ru_order | Нам нужно 500 клапанов SUS304, по цене 12,50 долларов США за штуку, с доставкой не позднее 15 ноября. | 我们需要 500 个 SUS304 阀门，每个价格为 12.50 美元，最迟于 11 月 15 日发货。 | 我们需要 500 个 SUS304 阀门，每个价格为 12.50 美元，最迟于 11 月 15 日发货。 | True |
| ja_order | SUS304製のバルブを500個単価12.50米ドルで納期は11月15日まででお願いします。 | 请提供 500 个 SUS304 阀的价格为 12.50 美元，交货期为 11 月 15 日。 | 请提供 500 个 SUS304 阀的价格为 12.50 美元，交货期为 11 月 15 日。 | True |
| ko_order | SUS304 밸브 500개가 필요합니다. 개당 가격은 12.50달러이고, 납기는 11월 15일까지입니다. | 我们需要 500 个 SUS304 阀门。每个阀门的价格为 12.50 美元，交货时间为 11 月 15 日。 | 我们需要 500 个 SUS304 阀门。每个 阀门 的价格为 12.50 美元，交货时间为 11 月 15 日。 | True |
| ar_order | نحتاج إلى 500 صمام من نوع SUS304 بسعر 12.50 دولار أمريكي للقطعة على أن يتم التسليم في موعد أقصاه 15 نوفمبر. | 我们需要 500 个 12.50 美元/件的 SUS304 阀门，最迟于 11 月 15 日交货。 | 我们需要 500 个 12.50 美元/件的 SUS304 阀门，最迟于 11 月 15 日交货。 | True |
| en_asr_revision | We need 300 valves by Friday, only after receiving the deposit. | 我们需要在收到押金后于周五之前安装 300 个阀门。 | 我们需要在收到押金后于周五之前安装 300 个阀门。 | True |
| en_entity | We need 100 iphone 15 Pro Max units, 256GB each, from AcmeFlow, with delivery by Friday. | 我们需要 AcmeFlow 提供 100 台 iPhone 15 Pro Max，每台容量为 256GB，周五发货。 | 我们需要 AcmeFlow 提供 100 台 iPhone 15 Pro Max，每台容量为 256GB，周五发货。 | True |
| en_long_window | We need 500 SUS304 valves. The unit price is 12.50 US dollars. Delivery is due by November 15. Please confirm the order. The valves are made of stainless steel. Please send the invoice to AcmeFlow. Do not ship before receiving the deposit. | 我们需要 500 个 SUS304 阀门。单价为 12.50 美元。交货日期为 11 月 15 日。请确认订单。这些阀门由不锈钢制成。请将发票发送给 AcmeFlow。在收到定金之前，请勿发货。 | 我们需要 500 个 SUS304 阀门。单价为 12.50 美元。交货日期为 11 月 15 日。请确认订单。这些阀门由不锈钢制成。请将发票发送给 AcmeFlow。在收到定金之前，请勿发货。 | True |

逐步事件：events.jsonl；模型输入输出及补丁痕迹：refiner_traces.json；配置和GPU占用：metadata.json。
