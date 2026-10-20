# Trade field diagnostics

This reads existing `mode=retranslate` beam=3 results only; no model inference was run. Each model has eight parallel-language translations of the same constructed quote template. These are not random business samples and cannot estimate production accuracy.

An amount counts as retained only when one unambiguous Arabic-numeral amount is explicitly adjacent to a USD/美元/美金 marker and normalizes to 12.50. Chinese-numeral or unclear structures are marked uncertain rather than automatically scored wrong. The `阀` count is a literal term-hit diagnostic, not sentence or translation accuracy; traditional `閥` is shown separately.

| Result set | Model | Explicit 12.50 + currency | Literal 阀 | Traditional 閥 (diagnostic) | Uncertain amount rows |
|---|---|---:|---:|---:|---:|
| CT2 INT8/FP16 beam=3 | m2m100_418M | 3/8 | 7/8 | 0/8 | 0 |
| CT2 INT8/FP16 beam=3 | small100 | 7/8 | 0/8 | 0/8 | 0 |
| Transformers FP16 beam=3 | m2m100_418M | 2/8 | 7/8 | 0/8 | 0 |
| Transformers FP16 beam=3 | small100 | 6/8 | 0/8 | 0/8 | 0 |

## Per-case outputs

### CT2 INT8/FP16 beam=3 — m2m100_418M

| Language | Output | Extracted amount + currency | 12.50 retained | 阀 |
|---|---|---|---|---:|
| ar | 我们需要500个SUS304型电缆,每件零件价格为1250美元,最迟于11月15日发货。 | 1250 美元 → 1250 | no/uncertain | no |
| de | 我们需要500个SUS304阀门,价格为12.50美元,最迟于11月15日发货。 | 12.50 美元 → 12.5 | yes | yes |
| en | 我们需要500个SUS304阀门,每台1250美元,到11月15日发货。 | 1250 美元 → 1250 | no/uncertain | yes |
| es | 我们需要500个SUS304阀,每台1250美元,最迟于11月15日发货。 | 1250 美元 → 1250 | no/uncertain | yes |
| fr | 我们需要500 SUS304 阀门,每单位 12.50 美元,最迟于 11 月 15 日交货。 | 12.50 美元 → 12.5 | yes | yes |
| ja | SUS304 的 500 个单价 12.50 米的阀门,请在 11 月 15 日之前交付。 | no_explicit_amount_and_currency | no/uncertain | yes |
| ko | 需要 500 个 SUS304 阀门,每人价格为 12.50 美元,发货时间为 11 月 15 日。 | 12.50 美元 → 12.5 | yes | yes |
| ru | 我们需要500个SUS304阀门,价格为每件1250美元,最迟于11月15日交货。 | 1250 美元 → 1250 | no/uncertain | yes |

### CT2 INT8/FP16 beam=3 — small100

| Language | Output | Extracted amount + currency | 12.50 retained | 阀 |
|---|---|---|---|---:|
| ar | 我们需要500个SUS304类的门口,价格为12.50美元,以便在11月15日之前交付。 | 12.50 美元 → 12.5 | yes | no |
| de | 我们需要500个SUS304的 Ventil,价格为12.50美元,并在11月15日晚些时候交付。 | 12.50 美元 → 12.5 | yes | no |
| en | 我們需要500SUS304風扇,每個1250美元,並在11月15日發送。 | 1250 美元 → 1250 | no/uncertain | no |
| es | 我們需要500個SUS304風扇,每個單元為12.50美元,並在11月15日之前交付。 | 12.50 美元 → 12.5 | yes | no |
| fr | 我们需要500个SUS304风扇,每单位12.50美元,并在11月15日之前交付。 | 12.50 美元 → 12.5 | yes | no |
| ja | 请在11月15日之前发送500个单价为12.50美元的SUS304风扇。 | 12.50 美元 → 12.5 | yes | no |
| ko | 需要 500 个 SUS304  Ventil,每人价格为 12.50 美元,发送时间为 11 月 15 日。 | 12.50 美元 → 12.5 | yes | no |
| ru | 我們需要500個SUS304的風扇,每件價格為12.50美元,並在11月15日之前發送。 | 12.50 美元 → 12.5 | yes | no |

### Transformers FP16 beam=3 — m2m100_418M

| Language | Output | Extracted amount + currency | 12.50 retained | 阀 |
|---|---|---|---|---:|
| ar | 我们需要500个SUS304型电缆,每件零件价格为1250美元,最迟于11月15日发货。 | 1250 美元 → 1250 | no/uncertain | no |
| de | 我们需要500个SUS304阀门,价格为1250美元,最迟于11月15日发货。 | 1250 美元 → 1250 | no/uncertain | yes |
| en | 我们需要500个SUS304阀门,每台1250美元,到11月15日发货。 | 1250 美元 → 1250 | no/uncertain | yes |
| es | 我们需要500个SUS304阀,每台1250美元,最迟到11月15日发货。 | 1250 美元 → 1250 | no/uncertain | yes |
| fr | 我们需要500 SUS304 阀门,每单位 12.50 美元,最迟到 11 月 15 日交货。 | 12.50 美元 → 12.5 | yes | yes |
| ja | SUS304 的 500 个单价 12.50 米的阀门,请提交至 11 月 15 日。 | no_explicit_amount_and_currency | no/uncertain | yes |
| ko | 需要 500 个 SUS304 阀门,每台价格为 12.50 美元,发货时间为 11 月 15 日。 | 12.50 美元 → 12.5 | yes | yes |
| ru | 我们需要500个SUS304阀门,价格为每件1250美元,最迟于11月15日交货。 | 1250 美元 → 1250 | no/uncertain | yes |

### Transformers FP16 beam=3 — small100

| Language | Output | Extracted amount + currency | 12.50 retained | 阀 |
|---|---|---|---|---:|
| ar | 我们需要500个SUS304类的门口,价格为12.50美元,以便在11月15日之前交付。 | 12.50 美元 → 12.5 | yes | no |
| de | 我们需要500个SUS304的 Ventil,价格为12.50美元,并在11月15日晚些时候交付。 | 12.50 美元 → 12.5 | yes | no |
| en | 我们需要500个SUS304门,每一个1250美元,并在11月15日之前交付。 | 1250 美元 → 1250 | no/uncertain | no |
| es | 我們需要500個SUS304風扇,每個單元為12.50美元,並在11月15日之前交付。 | 12.50 美元 → 12.5 | yes | no |
| fr | 我们需要500个SUS304风扇,每单位12.50美元,并在11月15日之前交付。 | 12.50 美元 → 12.5 | yes | no |
| ja | 请在11月15日之前发送500个单价1250美元的SUS304风扇。 | 1250 美元 → 1250 | no/uncertain | no |
| ko | 需要500个SUS304 扇,每人价格为12.50美元,交付时间为11月15日。 | 12.50 美元 → 12.5 | yes | no |
| ru | 我們需要500個SUS304風扇,每件價格為12.50美元,最晚於11月15日發送。 | 12.50 美元 → 12.5 | yes | no |
