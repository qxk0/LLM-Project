# 数据集划分与泄漏审计报告 (DATASET LEAKAGE AUDIT)

- dataset: `output_clean`
- generated_at: 2026-09-28T20:20:49+08:00
- input `seeds.py`: sha256:45b7d6e92630312c
- input `kb.py`: sha256:35e876254b785b17
- input `generator`: build_dataset.py sha256:92fcb99eaffafa67
- input `extra_files`: {'data_eng/output/evolved.jsonl': {'valid_rows': 86, 'invalid_dropped': 0}, 'data_eng/output/onpolicy.jsonl': {'valid_rows': 1521, 'invalid_dropped': 0}}
- input `split_seed`: 42
- input `build_command`: python data_eng/build_dataset.py --extra-file data_eng\output\evolved.jsonl --extra-file data_eng\output\onpolicy.jsonl
- input `extra_file:data_eng/output/evolved.jsonl`: sha256:9e94e432111997bd
- input `extra_file:data_eng/output/onpolicy.jsonl`: sha256:dde6689fc08befeb
- **all assertions passed: `True`**

## 0. 说明与已知残余风险

- 旧目录 data_eng/output/ 与全部旧 checkpoint 未被修改;本脚本只写 --out-dir。
- val/test 只接受 gold 样本(source ∈ {seed, template, paraphrase})且血缘可追溯。
- evolved / onpolicy 只能进 train:它们会把所在血缘 group 标记为 non-clean,从而自动被排除出 val/test。
- 血缘不可追溯(unknown)样本 7 条,全部锁进 train(family_id=unknown_family),不计入任何 val/test 指标。
- 血缘歧义(ambiguous_union)样本 1 条:无法唯一确定父 seed,已把所有候选 seed 的 group 一起锁进 train。
- test 未能覆盖的 intent(不做任何数据修补,仅如实报告):[]
- 下游训练/评测脚本默认数据路径未改动;重训/评测时必须显式指定 --data-file data_eng/output_clean/*.jsonl。
- 本轮新增人工 Gold Evaluation Seed 78 条(source=gold_eval, eval_only=True):只允许进 test,不参与 paraphrase / 模板扩展 / 蒸馏 / 自采样,也不会出现在 train/val。
- gold_eval 与训练数据在 exact question / base question / family 三个层面完全隔离;但同一产品、同一主题下的**其他业务场景**仍可能与 train 共享实体(例如 train 有"杨枝甘露什么价位"、test 有"杨枝甘露和茉莉绿茶哪个更贵"),这是有意设计的"新场景评测",逐题理由见 GOLD_EVAL_AUDIT.md。
- 修复前对照(旧 data_eng/output/,样本级随机切分):test 288 条中,188 条与 train 精确同题、277 条去掉前缀后与 train 同源;OOD base question train∩test = 13/13;test 含模型生成数据 {'<none>': 131, 'onpolicy': 146, 'evolved': 11}。

## 1. Raw dataset(去重清洗后、切分前)

```text
total samples           : 2924
unique questions        : 1414
unique base questions   : 414
unique families         : 415
unique split groups     : 144
unique intents          : 10
OOD samples / families  : 136 / 40
```

来源分布(切分前):

| source | count |
|---|---|
| onpolicy | 1480 |
| paraphrase | 1024 |
| template | 201 |
| evolved | 86 |
| gold_eval | 78 |
| seed | 55 |

## 2. Split

| split | samples | questions | base questions | families | groups | intents | OOD samples | OOD families |
|---|---|---|---|---|---|---|---|---|
| train | 2676 | 1166 | 302 | 303 | 40 | 10 | 41 | 13 |
| val | 65 | 65 | 13 | 13 | 9 | 2 | 35 | 7 |
| test | 183 | 183 | 99 | 99 | 95 | 10 | 60 | 20 |

## 3. Source distribution

### train

| source | count |
|---|---|
| onpolicy | 1480 |
| paraphrase | 888 |
| template | 176 |
| evolved | 86 |
| seed | 46 |

### val

| source | count |
|---|---|
| paraphrase | 52 |
| template | 12 |
| seed | 1 |

### test

| source | count |
|---|---|
| paraphrase | 84 |
| gold_eval | 78 |
| template | 13 |
| seed | 8 |

## 4. Intent distribution

### train

| intent | count |
|---|---|
| price | 1609 |
| recommend | 809 |
| customize | 74 |
| ood | 41 |
| after_sale | 24 |
| allergen | 24 |
| delivery | 24 |
| hours | 24 |
| member | 24 |
| store | 23 |

### val

| intent | count |
|---|---|
| ood | 35 |
| customize | 30 |

### test

| intent | count |
|---|---|
| ood | 60 |
| customize | 30 |
| member | 15 |
| hours | 13 |
| price | 12 |
| recommend | 12 |
| delivery | 11 |
| after_sale | 10 |
| allergen | 10 |
| store | 10 |

intent 覆盖(缺失项会被显式列出,不做任何数据修补):

- train: covered=['after_sale', 'allergen', 'customize', 'delivery', 'hours', 'member', 'ood', 'price', 'recommend', 'store'] | **missing=[]**
- val: covered=['customize', 'ood'] | **missing=['after_sale', 'allergen', 'delivery', 'hours', 'member', 'price', 'recommend', 'store']**
- test: covered=['after_sale', 'allergen', 'customize', 'delivery', 'hours', 'member', 'ood', 'price', 'recommend', 'store'] | **missing=[]**

## 5. OOD distribution

| split | OOD samples | OOD families | OOD entities |
|---|---|---|---|
| train | 41 | 13 | 6 |
| val | 35 | 7 | 7 |
| test | 60 | 20 | 10 |

交集(必须全为 0):

```text
OOD family overlap : {'train-val': 0, 'train-test': 0, 'val-test': 0}
OOD entity overlap : {'train-val': 0, 'train-test': 0, 'val-test': 0}
```

- train OOD entities(6): ood_topic::修手机, seed::seed_049, seed::seed_050, seed::seed_053, seed::seed_054, seed::seed_055
- val OOD entities(7): ood_topic::代开发票, ood_topic::卖二手车, ood_topic::打疫苗, ood_topic::搬家, ood_topic::订酒店, ood_topic::退机票, ood_topic::配钥匙
- test OOD entities(10): ood_topic::办签证, ood_topic::寄快递, ood_topic::开锁, ood_topic::报培训班, ood_topic::查违章, ood_topic::缴水电费, ood_topic::考驾照, ood_topic::装系统, seed::seed_051, seed::seed_052

## 6. Leakage

### 6.1 Exact duplicate (normalized question)

```text
train-val: 0
train-test: 0
val-test: 0
```

### 6.2 Same-family leakage

```text
train-val: 0
train-test: 0
val-test: 0
```

### 6.3 Base-question leakage (prefix paraphrase 已剥离)

```text
train-val: 0
train-test: 0
val-test: 0
```

### 6.4 Split-group leakage (实体级 / seed 级连通分量)

```text
train-val: 0
train-test: 0
val-test: 0
```

### 6.5 行级污染面

```text
exact question rows: {'train<-val': 0, 'train<-test': 0, 'val<-train': 0, 'val<-test': 0, 'test<-train': 0, 'test<-val': 0}
base question rows: {'train<-val': 0, 'train<-test': 0, 'val<-train': 0, 'val<-test': 0, 'test<-train': 0, 'test<-val': 0}
family rows: {'train<-val': 0, 'train<-test': 0, 'val<-train': 0, 'val<-test': 0, 'test<-train': 0, 'test<-val': 0}
```

### 6.6 split 内部重复(信息项)

| metric | count |
|---|---|
| train.duplicate_qa_pairs | 0 |
| train.duplicate_questions | 1510 |
| val.duplicate_qa_pairs | 0 |
| val.duplicate_questions | 0 |
| test.duplicate_qa_pairs | 0 |
| test.duplicate_questions | 0 |

## 7. Test purity

```text
test onpolicy = 0
test evolved  = 0
test model-generated total = {'evolved': 0, 'onpolicy': 0}
```

## 8. Unknown / ambiguous lineage

```text
unknown_family_count    = 7
ambiguous_lineage_count = 1
unknown by source       = {'onpolicy': 7}
ambiguous by source     = {'evolved': 1}
```

### lineage 恢复统计

```text
gold_samples = 1358
extra_samples = 1607
invalid_extra_rows_dropped = 0
dedup_and_length_dropped = 41
unique_samples = 2924
groups_total = 144
groups_clean = 26
groups_non_clean = 118
unknown_lineage_samples = 7
ambiguous_lineage_samples = 1
```

**lineage_confidence**

| key | count |
|---|---|
| gold | 1280 |
| seed_match | 85 |
| ambiguous_union | 1 |
| exact_question | 1443 |
| unknown | 7 |
| entity_only | 30 |
| gold_eval | 78 |

**gold_template_breakdown**

| key | count |
|---|---|
| seed | 55 |
| price_template | 64 |
| recommend_template | 96 |
| customize_template | 20 |
| topping_template | 5 |
| ood_template | 16 |
| paraphrase_added | 1024 |


## 9. Gold evaluation set(人工 Gold Evaluation Seed)

```text
gold_eval count        = 78 (expected 78)
gold_eval in test      = 78
by intent              = {'after_sale': 10, 'allergen': 10, 'delivery': 6, 'hours': 8, 'member': 5, 'ood': 10, 'price': 12, 'recommend': 12, 'store': 5}
test intent counts     = {'after_sale': 10, 'allergen': 10, 'customize': 30, 'delivery': 11, 'hours': 13, 'member': 15, 'ood': 60, 'price': 12, 'recommend': 12, 'store': 10}
test missing intents   = []
previous test preserved= {'previous_test_rows': 183, 'missing': 0, 'content_changed': 0}
previous test sha256    = c665a541f693e3803982c3cc3ba5f2991ae8af4003fe8b80cc3ba6d727c0e91a
```

gold_eval 与 train / val / 旧 test 的多层去重(必须全为 0):

```text
gold_eval ∩ train: {'exact': 0, 'base': 0, 'family': 0}
gold_eval ∩ val: {'exact': 0, 'base': 0, 'family': 0}
gold_eval ∩ old_test: {'exact': 0, 'base': 0, 'family': 0}
gold_eval ∩ ood_family: {'train': 0, 'val': 0, 'old_test': 0}
```

逐题清单见 `GOLD_EVAL_AUDIT.md`(含 knowledge source / novelty reason / leakage check)。

## 10. Assertions

| assertion | result | detail |
|---|---|---|
| exact normalized question never crosses split | PASS | overlap={'train-val': 0, 'train-test': 0, 'val-test': 0} |
| family never crosses split | PASS | overlap={'train-val': 0, 'train-test': 0, 'val-test': 0} |
| base question never crosses split | PASS | overlap={'train-val': 0, 'train-test': 0, 'val-test': 0} |
| split group never crosses split | PASS | overlap={'train-val': 0, 'train-test': 0, 'val-test': 0} |
| OOD family never crosses split | PASS | overlap={'train-val': 0, 'train-test': 0, 'val-test': 0} (train=13, val=7, test=20) |
| OOD entity/topic never crosses split | PASS | overlap={'train-val': 0, 'train-test': 0, 'val-test': 0} |
| test contains 0 model-generated samples (onpolicy/evolved) | PASS | test model-generated = {'evolved': 0, 'onpolicy': 0} |
| val contains 0 model-generated samples (onpolicy/evolved) | PASS | val model-generated = {'evolved': 0, 'onpolicy': 0} |
| all samples carry complete lineage metadata | PASS | no missing fields |
| train/val/test all non-empty | PASS | sizes={'train': 2676, 'val': 65, 'test': 183} |
| no unknown-lineage sample in val/test | PASS | unknown_lineage_samples_total=7 |
| gold_eval samples all present and flagged eval_only | PASS | expected=78 actual=78 |
| gold_eval only in test (train/val gold_eval = 0) | PASS | test=78 train=0 val=0 |
| gold_eval zero exact/base/family overlap with train/val/old_test | PASS | {'train': {'exact': 0, 'base': 0, 'family': 0}, 'val': {'exact': 0, 'base': 0, 'family': 0}, 'old_test': {'exact': 0, 'base': 0, 'family': 0}} |
| gold_eval OOD families disjoint from train/val/old_test OOD families | PASS | {'train': 0, 'val': 0, 'old_test': 0} |
| test covers every intent present in the dataset | PASS | test_intents=['after_sale', 'allergen', 'customize', 'delivery', 'hours', 'member', 'ood', 'price', 'recommend', 'store'] missing=[] |
| test has >=10 questions for price/recommend/after_sale/allergen | PASS | {'price': 12, 'recommend': 12, 'after_sale': 10, 'allergen': 10} |
| previous clean test rows preserved verbatim | PASS | {'previous_test_rows': 183, 'missing': 0, 'content_changed': 0} |
