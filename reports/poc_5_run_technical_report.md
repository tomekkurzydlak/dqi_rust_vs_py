# DQI PoC: Python vs Rust for Markdown Quality Scoring

## 1. Cel i kontekst PoC

Celem PoC bylo techniczne sprawdzenie, czy semantyczna czesc obliczania podstawowego DQI (Document Quality Index) dla dokumentow Markdown po konwersji PDF -> Markdown moze byc uruchamiana bez runtime Python, z wykorzystaniem Rust + ONNX, przy akceptowalnym koszcie pamieci i czasu przetwarzania w scenariuszu docelowo zblizonym do przyszlego serwisu.

PoC nie mial udowadniac, ze heurystyki sa szybsze od podejsc modelowych, poniewaz to bylo z gory oczywiste. Glownym pytaniem technicznym bylo:

Czy semantyczna czesc DQI, naturalnie realizowana dzis w Pythonie, da sie sensownie zastapic wariantem Rust + ONNX bez Python runtime, przy zachowaniu porownywalnosci cech, stabilnych wynikow oraz przewidywalnego kosztu RAM i processing time po jednorazowym zaladowaniu modeli?

PoC zostal zrealizowany jako benchmark CLI. Celowo nie byl to serwis HTTP. Startup time procesu nie byl mierzony. Pomiar zaczynal sie dopiero po zaladowaniu modelu.

## 2. Zakres funkcjonalny PoC

Obie implementacje CLI licza ten sam zestaw kategorii DQI dla tych samych dokumentow Markdown:

- `coverage_score`
- `structure_score`
- `info_density_score`
- `noise_score`
- `dqi_total`

`perplexity_score` pozostaje placeholderem i nie jest liczony.

Z PoC zostaly celowo wykluczone:

- perplexity, poniewaz docelowo ma byc realizowane przez zewnetrzny serwis
- dedup / similarity, poniewaz nie bylo to celem tego porownania

## 3. Co zostalo zbudowane

W repozytorium znajduja sie dwa niezalezne CLI:

- `python_dqi_cli`
- `rust_dqi_cli`

Oba narzedzia:

- przyjmuja katalog z plikami `.md`
- przetwarzaja ten sam zbior dokumentow
- licza ten sam schemat wyjsciowy JSON
- zapisują per-document results JSON oraz benchmark JSON

Wyniki benchmarkowe byly zapisywane do:

- `outputs/poc_runs/run1_python`
- `outputs/poc_runs/run1_rust`
- `outputs/poc_runs/run2_python`
- `outputs/poc_runs/run2_rust`
- `outputs/poc_runs/run3_python`
- `outputs/poc_runs/run3_rust`
- `outputs/poc_runs/run4_python`
- `outputs/poc_runs/run4_rust`
- `outputs/poc_runs/run5_python`
- `outputs/poc_runs/run5_rust`

## 4. Techniczne podejscie

### 4.1. Wspolna logika DQI

W obu implementacjach obliczane sa cztery skladowe DQI:

- coverage
- structure
- noise
- info_density

Agregacja wyniku lacznego pozostaje stala i jest konfigurowalna przez zestaw wag:

`DQI = 0.30 * coverage_score + 0.25 * structure_score + 0.25 * info_density_score + 0.20 * noise_score`

### 4.2. Coverage score

Coverage score jest liczony na podstawie cech strukturalno-objetosciowych dokumentu Markdown, takich jak:

- liczba znakow
- liczba slow
- liczba niepustych linii
- liczba paragrafow
- liczba naglowkow
- liczba sekcji
- liczba pustych sekcji
- relacja linii z trescia do calosci
- relacja pustych sekcji do liczby sekcji

Score karze dokumenty:

- bardzo krotkie
- prawie puste
- z duza liczba pustych sekcji
- z mala iloscia tresci wzgledem struktury

### 4.3. Structure score

Structure score ocenia, czy dokument ma strukture sensowna pod chunking i retrieval w RAG. Brane pod uwage sa miedzy innymi:

- liczba naglowkow
- liczba sekcji
- liczba paragrafow
- liczba list
- liczba tabel
- liczba uszkodzonych naglowkow
- liczba pustych naglowkow
- liczba uszkodzonych markerow list
- flagi `table_heavy_flag` i `narrative_heavy_flag`

### 4.4. Noise score

Noise score mierzy poziom zaszumienia i jakosc surowego tekstu po konwersji PDF -> Markdown. Obejmuje miedzy innymi:

- `replacement_char_count`
- `control_char_count`
- `weird_unicode_ratio`
- `long_token_ratio`
- `punctuation_ratio`
- `digit_ratio`
- `garbage_token_ratio`
- `glue_like_token_ratio`
- `repeated_line_ratio`

Wyższy `noise_score` oznacza lepsza jakosc, czyli mniej artefaktow noise.

### 4.5. Info density score

Info density byl glownym elementem PoC i obszarem porownania Python vs Rust.

W obu wariantach liczono co najmniej:

- `token_count`
- `stopword_ratio`
- `entity_count`
- `entity_density`
- `unique_token_ratio`
- `lexical_density_proxy`

W finalnym benchmarku PoC do obydwu implementacji zostal uzyty ten sam model NER wyeksportowany do ONNX wraz z tym samym `tokenizer.json`, tak aby zapewnic porownywalny pipeline semantyczny.

## 5. Python CLI: technikalia implementacyjne

Python CLI jest zaimplementowany dla `Python 3.11+`. Interfejs CLI jest zbudowany przez `argparse`.

W finalnym benchmarku 5-run PoC wykorzystano backend `onnx`.

Implementacja semantyczna Python ONNX:

- tokenizer: `tokenizers.Tokenizer.from_file(...)`
- runtime modelu: `onnxruntime`
- przygotowanie wejscia tensorowego: `numpy`
- wspolbieznosc: `ThreadPoolExecutor`
- session pool: lista `InferenceSession`, po jednej blokadzie na sesje, round-robin wybor sesji

Istotne szczegoly:

- wspolny `tokenizer.json` jest ladowany bezposrednio z artefaktow modelu
- `CPUExecutionProvider` jest wymuszony
- `onnxruntime` ma ustawione `intra_op_num_threads = 1` i `inter_op_num_threads = 1`
- `max_len = 256`
- `onnx-session-pool-size = 4` w trybie `small_parallel`
- `workers = 4`

## 6. Rust CLI: technikalia implementacyjne

Rust CLI jest zaimplementowany dla aktualnego stable Rust i uzywa `clap` do obslugi CLI.

Kod zostal rozdzielony na osobne warstwy:

- parsowanie dokumentu Markdown
- heurystyki DQI
- warstwa modelowa ONNX
- benchmarking
- agregacja DQI

Implementacja semantyczna Rust ONNX:

- tokenizer: crate `tokenizers`, ladowany z tego samego `tokenizer.json`
- runtime modelu: crate `ort` oparty o ONNX Runtime
- provider: CPU
- wspolbieznosc dokumentow: `rayon`
- session pool: `Vec<Mutex<Session>>` z wyborem sesji round-robin przez `AtomicUsize`

Istotne szczegoly:

- `onnx_max_len = 256`
- `onnx_intra_threads = 1`
- `onnx_inter_threads = 1`
- `onnx_session_pool_size = 4`
- `workers = 4`

Wariant Rust nie uzywa Python runtime. Nie ma `PyO3`, nie ma embeddingu Pythona i nie ma subprocessow odpalajacych Pythona w sciezce inferencyjnej.

## 7. Model i tokenizacja

Do finalnego benchmarku uzyto:

- modelu `dslim/bert-base-NER`, wyeksportowanego do ONNX
- wspolnego artefaktu `models/ner_onnx/model.onnx`
- wspolnego artefaktu `models/ner_onnx/tokenizer.json`

To oznacza, ze:

- Python i Rust uzywaja tego samego modelu semantycznego
- Python i Rust uzywaja tego samego tokenizatora modelowego
- `entity_count` po obu stronach jest liczony na podstawie predykcji token-classification z tego samego modelu

Finalny benchmark jest benchmarkiem porownujacym dwa wdrozenia tego samego podejscia modelowego ONNX.

### 7.1. Jak model zostal wyeksportowany do ONNX

Model ONNX wykorzystany w benchmarku zostal przygotowany w repo przy pomocy skryptu:

- `scripts/export_ner_onnx.py`

Skrypt uruchamia:

```bash
python -m optimum.exporters.onnx --model dslim/bert-base-NER --task token-classification models/ner_onnx
```

Oznacza to, ze eksport byl wykonywany przez biblioteke `optimum`, a nie przez wlasny, recznie napisany kod eksportujacy graf modelu.

W praktyce sciezka eksportu i uruchomienia obejmuje nastepujace biblioteki:

- `optimum.exporters.onnx` do eksportu modelu Hugging Face do ONNX
- `onnxruntime` jako runtime inferencyjny w Pythonie
- `tokenizers` jako modelowy tokenizer po stronie Python i Rust
- `numpy` do budowy wejsc tensorowych i prostego postprocessingu wyjsc modelu po stronie Python
- `ort` jako rustowy binding do ONNX Runtime po stronie Rust

Artefakty zapisane po eksporcie:

- `models/ner_onnx/model.onnx`
- `models/ner_onnx/tokenizer.json`
- `models/ner_onnx/tokenizer_config.json`
- `models/ner_onnx/special_tokens_map.json`
- `models/ner_onnx/vocab.txt`
- `models/ner_onnx/config.json`

## 8. Wspolbieznosc i model wykonania

Finalny benchmark 5-run nie byl uruchamiany w trybie pojedynczego wywolania dokument po dokumencie. W obu implementacjach zostala wlaczona wspolbieznosc.

Python:

- `ThreadPoolExecutor(max_workers=4)` do rownoleglego przetwarzania dokumentow
- pula `4` sesji `onnxruntime.InferenceSession`
- round-robin wybor sesji
- oddzielny lock na kazda sesje

Rust:

- `rayon::ThreadPoolBuilder::new().num_threads(4)` do rownoleglego przetwarzania dokumentow
- pula `4` sesji ONNX Runtime
- round-robin wybor sesji przez `AtomicUsize`
- `Mutex<Session>` na kazda sesje

To oznacza, ze w finalnym przebiegu benchmarkowym modelowe wywolania byly wykonywane wspolbieznie, a nie sekwencyjnie. W praktyce oba CLI przetwarzaly dokumenty rownolegle i rozkladaly inferencje na pule sesji modelu.

## 9. Metodologia benchmarku

### 9.1. Zbior dokumentow

Przetwarzany zbior zawieral 16 plikow Markdown z katalogu `samples/`:

- `TEST5.md`
- `broken_structure.md`
- `deklaracja.md`
- `good_doc.md`
- `kantor.md`
- `konto-za-zero.md`
- `long_doc.md`
- `noisy_doc.md`
- `owu.md`
- `pko-dom.md`
- `short_doc.md`
- `table_doc.md`
- `tabela-bb.md`
- `test3.md`
- `test4.md`
- `tp4.md`

Zbior zawiera dokumenty:

- dobre jakosciowo
- zaszumione
- krotkie
- dlugie
- tabelaryczne
- z uszkodzona struktura

### 9.2. Tryb uruchomienia

Wszystkie 5 finalnych przebiegow PoC wykonano w tym samym trybie:

- `mode = small_parallel`
- `workers = 4`
- `onnx-session-pool-size = 4`
- `max_len = 256`
- `onnx_intra_threads = 1`
- `onnx_inter_threads = 1`
- `docs_count = 16`

### 9.3. Komendy uruchomieniowe

Python:

```bash
source .venv/bin/activate
python -m python_dqi_cli.python_dqi_cli.cli \
  --input-dir samples \
  --output-dir outputs/poc_runs/runN_python \
  --mode small_parallel \
  --workers 4 \
  --semantic-backend onnx \
  --onnx-model models/ner_onnx/model.onnx \
  --tokenizer-json models/ner_onnx/tokenizer.json \
  --onnx-session-pool-size 4
```

Rust:

```bash
cargo run --release --bin rust_dqi_cli -- \
  --input-dir samples \
  --output-dir outputs/poc_runs/runN_rust \
  --mode small_parallel \
  --workers 4 \
  --onnx-model models/ner_onnx/model.onnx \
  --tokenizer-json models/ner_onnx/tokenizer.json \
  --onnx-session-pool-size 4
```

### 9.4. Co bylo mierzone

W benchmark JSON obie implementacje raportuja:

- `mode`
- `docs_count`
- `rss_loaded_mb`
- `peak_rss_mb`
- `avg_rss_delta_per_doc_mb`
- `avg_processing_ms_per_doc`
- `p95_processing_ms_per_doc`
- `avg_model_inference_ms_per_doc`
- `total_batch_ms`

Pomiar byl steady-state. Startup procesu nie byl liczony.

`rss_loaded_mb` oznacza RSS procesu po zaladowaniu modelu i przed przetwarzaniem dokumentow.

`peak_rss_mb` oznacza najwyzszy zaobserwowany RSS procesu podczas batcha.

`avg_rss_delta_per_doc_mb` oznacza sredni dodatni przyrost RSS przypadajacy na dokument.

`avg_processing_ms_per_doc` oznacza sredni calkowity czas przetwarzania dokumentu.

`p95_processing_ms_per_doc` oznacza 95 percentyl czasu przetwarzania dokumentu.

`avg_model_inference_ms_per_doc` oznacza sredni czas czesci modelowej raportowany przez implementacje.

`total_batch_ms` oznacza laczny czas batcha.

## 10. Srodowisko wykonania

Benchmarki zostaly wykonane na:

- systemie `Darwin 24.4.0 arm64`
- maszynie `Apple M1`
- `8` rdzeniach CPU (`4` performance i `4` efficiency)
- `8 GB` RAM

Toolchain i biblioteki:

- `Python 3.11.6`
- `onnxruntime 1.24.3`
- `tokenizers 0.22.2`
- `numpy 2.4.2`
- `psutil 7.2.2`
- `rustc 1.92.0 (ded5c06cf 2025-12-08)`
- `cargo 1.92.0 (344c4567c 2025-10-21)`
- `ort v2.0.0-rc.12`

## 11. Wyniki finalnych 5 przebiegow PoC

Poniższa tabela zawiera komplet wartosci przepisanych z plikow `benchmark_python.json` i `benchmark_rust.json` dla wszystkich pieciu przebiegow. Wartosci zostaly zachowane dokladnie tak, jak zapisaly je narzedzia benchmarkowe.

| run | impl | mode | docs_count | rss_loaded_mb | peak_rss_mb | avg_rss_delta_per_doc_mb | avg_processing_ms_per_doc | p95_processing_ms_per_doc | avg_model_inference_ms_per_doc | total_batch_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | py | small_parallel | 16 | 1098.203125 | 1580.8125 | 118.7060546875 | 606.3759871212824 | 1093.202874995768 | 568.0526119394926 | 2731.6092499822844 |
| 1 | ru | small_parallel | 16 | 996.8125 | 1523.921875 | 150.4287109375 | 588.4669766875 | 939.295458 | 562.5052004374999 | 2804.223333 |
| 2 | py | small_parallel | 16 | 1141.28125 | 1721.265625 | 140.677734375 | 684.3829113731772 | 1366.0932079947088 | 652.5505078716378 | 3048.3970409841277 |
| 2 | ru | small_parallel | 16 | 1055.6875 | 1572.59375 | 171.39453125 | 580.1013124375 | 1176.189584 | 560.6253750000001 | 2897.357834 |
| 3 | py | small_parallel | 16 | 1087.859375 | 1614.140625 | 128.1875 | 651.2746328135108 | 1143.2562500122003 | 623.9781850654254 | 2749.8621249978896 |
| 3 | ru | small_parallel | 16 | 894.671875 | 1572.671875 | 164.513671875 | 601.76799475 | 1233.0955000000001 | 581.3581665 | 2930.2057919999997 |
| 4 | py | small_parallel | 16 | 855.984375 | 1794.625 | 228.4169921875 | 612.7639894966705 | 1268.3214579883497 | 581.7490625686332 | 2761.8787919927854 |
| 4 | ru | small_parallel | 16 | 771.53125 | 1621.40625 | 209.4384765625 | 574.683981875 | 977.13125 | 554.9483411250001 | 2717.559625 |
| 5 | py | small_parallel | 16 | 1142.859375 | 1645.328125 | 140.013671875 | 648.0154714954551 | 1174.7301669965964 | 620.393182251064 | 2728.7594999943394 |
| 5 | ru | small_parallel | 16 | 1130.609375 | 1565.484375 | 115.0439453125 | 554.6961249375 | 1188.773416 | 535.8211223750001 | 2613.187542 |

## 12. Podsumowanie srednich z 5 przebiegow

Poniższa tabela zawiera srednie arytmetyczne z pieciu runow dla kazdej implementacji.

| impl | mode | docs_count | rss_loaded_mb | peak_rss_mb | avg_rss_delta_per_doc_mb | avg_processing_ms_per_doc | p95_processing_ms_per_doc | avg_model_inference_ms_per_doc | total_batch_ms |
|---|---|---|---|---|---|---|---|---|---|
| py | small_parallel | 16 | 1065.2375 | 1671.234375 | 151.200390625 | 640.5625984600192 | 1209.1207915975247 | 609.3447099392506 | 2804.1013415902853 |
| ru | small_parallel | 16 | 969.8625 | 1571.215625 | 162.1638671875 | 579.9432781375 | 1102.8970416 | 559.0516410875 | 2792.5068252 |

## 13. Wariant nierownolegly z wczesniejszego benchmarku

Poniższa tabela zawiera wartosci z wczesniejszego przebiegu `samples_aligned_seq`, wykonanego w trybie `sequential`, czyli bez rownoleglego przetwarzania dokumentow i bez rownoleglego wykonywania inferencji na puli sesji. Jest to pojedynczy historyczny przebieg referencyjny, dodany do raportu w celu pokazania punktu odniesienia dla wariantu bez wspolbieznosci.

| impl | mode | docs_count | rss_loaded_mb | peak_rss_mb | avg_rss_delta_per_doc_mb | avg_processing_ms_per_doc | p95_processing_ms_per_doc | avg_model_inference_ms_per_doc | total_batch_ms |
|---|---|---|---|---|---|---|---|---|---|
| py | sequential | 16 | 841.96875 | 841.96875 | 3.2099609375 | 548.0154140605009 | 1270.974999992177 | 517.8423384368216 | 8770.517249999102 |
| ru | sequential | 16 | 614.578125 | 614.578125 | 4.37890625 | 454.90631262500006 | 817.7051250000001 | 438.1042083125001 | 7612.096375 |

Ten historyczny przebieg pokazuje, ze juz w wariancie sekwencyjnym Rust byl szybszy od Pythona na tym samym zestawie `samples`, a po dodaniu wspolbieznosci finalny benchmark 5-run pozwolil dodatkowo ocenic steady-state zachowanie implementacji w ustawieniu bardziej zblizonym do przyszlego serwisu.

## 14. Interpretacja wynikow

Na podstawie pieciu finalnych przebiegow mozna stwierdzic:

- Rust zakonczyl benchmark z nizszym srednim `rss_loaded_mb` niz Python.
- Rust zakonczyl benchmark z nizszym srednim `peak_rss_mb` niz Python.
- Rust uzyskal nizszy sredni `avg_processing_ms_per_doc` niz Python.
- Rust uzyskal nizszy sredni `avg_model_inference_ms_per_doc` niz Python.
- Rust uzyskal nizszy sredni `p95_processing_ms_per_doc` niz Python.
- `total_batch_ms` dla obu wariantow bylo bardzo zblizone, z lekka przewaga Rust.
- `avg_rss_delta_per_doc_mb` bylo srednio wyzsze po stronie Rust niz po stronie Python.

Wynik jest istotny, poniewaz finalny benchmark byl benchmarkiem modelowo porownywalnym:

- ten sam model
- ten sam tokenizer
- ten sam limit sekwencji
- ten sam CPU provider
- ta sama liczba workerow
- ta sama wielkosc session pool
- startup procesu poza zakresem pomiaru

To pozwala traktowac wynik jako wiarygodna odpowiedz na pytanie o sensownosc wariantu Rust + ONNX bez Python runtime.

## 15. Ograniczenia PoC

PoC ma kilka swiadomych ograniczen:

- nie obejmuje perplexity
- nie obejmuje dedup / similarity
- nie jest to jeszcze serwis HTTP ani wdrozenie produkcyjne
- benchmark wykonano na lokalnej maszynie deweloperskiej, a nie na docelowym workloadzie serwerowym
- `avg_model_inference_ms_per_doc` obejmuje tylko wewnetrznie zdefiniowana czesc inferencyjna raportowana przez implementacje, a nie caly koszt tokenizacji, preparacji tensora i serializacji wyniku jako osobno rozbitych etapow

PoC jednak odpowiada na kluczowe pytanie architektoniczne dotyczace czesci semantycznej DQI.

## 16. Ocena PoC

Ocena PoC jest pozytywna.

Na podstawie wykonanych testow mozna uznac, ze wariant Rust + ONNX jest realnym kandydatem do przyszlego serwisu DQI, jezeli celem jest usuniecie Python runtime z warstwy semantycznej i utrzymanie modelowego, a nie wyłącznie heurystycznego charakteru info density.

Najwazniejsze wnioski:

- semantyczna czesc DQI zostala uruchomiona w Rust bez Python runtime
- pipeline zostal wyrownany modelowo z Pythonem przez wspolny ONNX model i wspolny tokenizer
- steady-state processing w Rust okazal sie porownywalny lub lepszy od wariantu Python ONNX
- zuzycie pamieci po stronie Rust w tym zestawie testowym nie okazalo sie gorsze od Pythona; srednio bylo nizsze dla `rss_loaded_mb` i `peak_rss_mb`
- przewaga Rust nie jest skokowa, ale jest technicznie widoczna i spójna z celem PoC

W praktyce oznacza to, ze przejscie do serwisu opartego o Rust + ONNX ma uzasadnienie techniczne i nie wymaga ukrytego fallbacku do Pythona, o ile zespol akceptuje utrzymanie inferencji na ONNX Runtime i dalsze strojenie wspolbieznosci, pamieci oraz ewentualnych przyszlych optymalizacji modelu.
