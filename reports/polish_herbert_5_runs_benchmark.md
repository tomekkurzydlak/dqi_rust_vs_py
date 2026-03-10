# Polish Model Benchmark (pczarnik/herbert-base-ner) - 5 runs

Konfiguracja benchmarku:

- input: `samples/` (16 dokumentow `.md`)
- model: `models/ner_onnx_pl_herbert/model.onnx`
- tokenizer: `models/ner_onnx_pl_herbert/tokenizer.json`
- mode: `small_parallel`
- workers: `4`
- onnx_session_pool_size: `4`
- onnx_max_len / max_len: `256`
- onnx_intra_threads: `1`
- onnx_inter_threads: `1`

## Run-by-run

| run | impl | mode | docs_count | rss_loaded_mb | peak_rss_mb | avg_rss_delta_per_doc_mb | avg_processing_ms_per_doc | p95_processing_ms_per_doc | avg_model_inference_ms_per_doc | total_batch_ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | py | small_parallel | 16 | 1475.28125 | 1535.0 | 83.15234375 | 760.4109350031649 | 1348.4326249745209 | 724.7506511212123 | 3171.614791994216 |
| 1 | ru | small_parallel | 16 | 1021.859375 | 1568.390625 | 172.5615234375 | 589.43275775 | 861.292625 | 570.4219348125001 | 2722.0032079999996 |
| 2 | py | small_parallel | 16 | 1107.375 | 1738.34375 | 153.5888671875 | 599.4095860605739 | 1071.8186250014696 | 574.9273358142091 | 2498.8593330199365 |
| 2 | ru | small_parallel | 16 | 1025.625 | 1582.078125 | 137.642578125 | 632.620086 | 943.4416249999999 | 612.4976666250001 | 2994.755458 |
| 3 | py | small_parallel | 16 | 1148.203125 | 1828.578125 | 160.5322265625 | 581.1227915619384 | 905.4634170024656 | 553.8204192434932 | 2586.656708997907 |
| 3 | ru | small_parallel | 16 | 1128.390625 | 1605.65625 | 127.255859375 | 587.1935573750001 | 891.0353749999999 | 567.837239625 | 2784.870542 |
| 4 | py | small_parallel | 16 | 1099.609375 | 1957.78125 | 210.3125 | 644.515932244758 | 1136.0147079976741 | 618.9813669352588 | 2690.66362499143 |
| 4 | ru | small_parallel | 16 | 790.171875 | 1650.46875 | 217.7216796875 | 597.5673020625 | 937.2002080000001 | 578.093020875 | 2778.689417 |
| 5 | py | small_parallel | 16 | 1471.59375 | 1631.25 | 48.080078125 | 732.1033358748537 | 1282.8367500042077 | 706.0134531238873 | 3053.1702500011306 |
| 5 | ru | small_parallel | 16 | 1183.734375 | 1606.546875 | 130.9150390625 | 631.2220026874999 | 1278.76775 | 612.033198 | 2892.007458 |

## Summary (mean of 5 runs)

| impl | mode | docs_count | rss_loaded_mb | peak_rss_mb | avg_rss_delta_per_doc_mb | avg_processing_ms_per_doc | p95_processing_ms_per_doc | avg_model_inference_ms_per_doc | total_batch_ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| py | small_parallel | 16 | 1260.4125 | 1738.190625 | 131.133203125 | 663.5125161490578 | 1148.9132249960676 | 635.6986452476121 | 2800.192941800924 |
| ru | small_parallel | 16 | 1029.95625 | 1602.628125 | 157.2193359375 | 607.6071411749999 | 982.3475166 | 588.1766119875 | 2834.4652166 |
