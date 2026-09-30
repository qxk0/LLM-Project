# 面试事实清单(LLM-Project)

> 原则:以下每一条都能在当前仓库里找到对应代码 / 日志 / 产物;**没有证据的说法一律标注"不能说"**。
> 证据写法:`文件:行号` 或 产物路径。审计依据见 `data_eng/PROJECT_FINAL_AUDIT.md`。

---

## 1. 为什么用 LoRA?

- 硬件限制:RTX 3050 Laptop,**4GB 显存**(`nvidia-smi`/torch 实测 4.29GB)。Qwen2.5-0.5B 用 AdamW 全参微调,Lora 之外还要存梯度+优化器状态,4GB 装不下。
- 实际配置:4bit NF4 量化基座 + LoRA `r=16, alpha=32, dropout=0.05`,挂在 7 个投影层(`q/k/v/o/gate/up/down`)。
- **可核查的数字**:`models/sft/adapter_model.safetensors` 里 LoRA A+B 共 **8,798,208 个参数(≈8.80M)**,24 层 × 7 个模块 = 336 个张量;adapter 文件 33.6MB。
- 相对基座(约 0.49B)可训练参数占比 **≈1.8%**。
- ⚠️ 注意:README/简历若写"约 1%"是近似,建议写"可训练参数 ≈8.8M(占基座 ~1.8%)",这个数字你能自己算出来。

## 2. 为什么 SFT → DPO → DAPO?

- 三段对应不同目标:SFT 教"客服口吻 + 知识库事实";DPO 用偏好对直接拉"该答的答、不该答的拒答"(不需要奖励模型);DAPO/GRPO 用可验证的规则奖励在线优化拒答边界(项目主线是"会的就答对、不会的拒绝答")。
- 驱动证据(旧 288 条集):SFT 的领域正确率 88.0% 但**拒答率 0.0%** → 说明"会答但不会拒",所以才在后面加偏好/RL 对齐。
- ⚠️ 注意:三者的**逐级增益在当前 clean 集上还没有被验证**(DPO 从未在 clean 集评测);只能说"按演进线做了三种对齐并各自跑通"。

## 3. DPO 的 chosen/rejected 从哪里来?

- 代码:`rl/train_dpo.py:86-92`。**chosen = 数据里的 gold 答案 `r["a"]`(知识库答案)**;rejected = 同一问题采样 4 条里规则分**最低**的那条;只有当 `worst_score < gold_score` 才保留这对。
- 采样参数:`temperature=0.8, top_p=0.95, max_new_tokens=64, num_return_sequences=4`。
- 规则分(`score_response`):非 OOD 命中 keyword → +1,否则 −1;OOD 含拒答词 → +1,否则 −1;回答 ≤80 字符再 +0.1。
- ⚠️ 注意:这不是"采样最优 vs 采样最差",而是 **gold vs 采样最差**;`train_dpo.py` 的 docstring 和 README 写成"最高分当 chosen",与代码矛盾——面试要主动说清,否则被追问会翻车。
- ⚠️ 偏好对**没有落盘**,只在内存里,所以"DPO 数据量"只能从 epoch 反推(200 步轮次约 208 对,10 步轮次约 360 对)。

## 4. DPO 和 SFT 的数据格式为什么不同?

- SFT(`sft/train_sft.py`):`tokenizer.apply_chat_template([user, assistant])` → 带 `<|im_start|>…<|im_end|>` 的对话格式,并且 `labels[:mask_len] = -100` 只对 assistant 回答算 loss。
- DPO(`rl/train_dpo.py`):`DPOTrainer(prompt=r["q"], chosen=…, rejected=…)`,prompt 是**裸问题字符串**,模板由 DPOTrainer 内部处理;`max_length=128`。
- GRPO/DAPO(`rl/train_grpo.py`):`prompt` 也是裸 `r["q"]`。
- 后果(实测,`prompt_ablation/`):DPO 在 **raw** 下 EOS 命中 40%、平均 28.4 token;在 **chat** 下 EOS 0%、跑满 32 token。SFT 反过来在 chat 下更合理(40% vs 20% 准确率)。
- 正确说法:"我的三个对齐阶段用了不一致的 prompt 约定,这是一个工程缺陷;我做了一个 40 次生成的诊断消融定位到 DPO 对格式最敏感,最终评测需要按各自训练格式分别报告。"

## 5. DPO 和 GRPO/DAPO 的区别?

| | DPO(本项目) | GRPO/DAPO(本项目) |
|---|---|---|
| 数据 | 预先生成的偏好对(gold vs worst) | 在线 rollout(`num_generations=4`) |
| 奖励 | 无需显式 reward model,`loss_type=sigmoid`,beta=0.1 | 规则 reward(命中/拒答 ±1,简洁 +0.1),TRL 内部组内归一化成优势 |
| 超参 | lr=5e-5,max_length=128,200 步 / 10 步 | lr=1.5e-4,max_completion_length=64,300 步,beta(KL)=0.04 |
| 计算 | 便宜(只做前向打分) | 贵(每题要生成 4 条再打分) |
- 一句话:DPO 是"离线偏好对上的隐式奖励优化",DAPO 是"在线采样 + 可验证奖励 + 动态裁剪"的 RL。

## 6. DAPO reward 怎么设计?

- `domain_reward`(对称 ±1,`train_grpo.py:133-145`):
  - 非 OOD:回答包含任一 keyword → **+1**,否则 **−1**
  - OOD:含拒答词(`抱歉/不好意思/无法回答/超出/服务范围/咨询`)→ **+1**,否则 **−1**
- `format_reward`:`len(answer) ≤ 80` → **+0.1**(低权重,避免压过正确性)
- 总奖励 = 两者相加(TRL 会把多个 reward 函数累加)。
- 训练日志证据(`models/rl/training_log.json`):末步 `reward≈0.8275`、`rewards/domain_reward/mean=0.75`、`rewards/format_reward/mean≈0.078`。
- ⚠️ 注意:这是**规则奖励**,不是奖励模型;`content` 里没有长度惩罚/重复惩罚。

## 7. reward hacking 如何处理?

真实做法(都有代码/日志支撑),不是"完美解决":
1. **简洁奖励只给 0.1**:早期若给大权重,模型会为了"短"牺牲正确性;现在 ≤80 字只加 0.1。
2. **对称惩罚 ±1**:答错/瞎编扣分,避免"只输出拒答"或"只堆关键词"也能拿高分(但 ±1 对称也带来拒答偏好,见下)。
3. **KL 约束 beta=0.04**:限制策略别离 SFT 太远。
4. **OOD 上采样到 20%**(`make_domain_dataset`):平衡"会答"和"会拒答"。
5. **截断掩码**:有 `mask_truncated_completions` 开关,但**当前默认 False**;日志显示 `clipped_ratio=1.0`、`mean_terminated_length=0`,即每条 completion 都被 64 token 截断。
- 观察到的副作用(诚实版):训练后 `rl` 在旧 288 集拒答率 100% 但领域正确率掉到 83.3%;clean 集(修复前口径)拒答 90%、领域正确率 47.2% —— **典型的 reward 过偏向拒答**。这属于"奖励设计导致的偏置",当时没有系统做 reward-hacking 分析(没有奖励 vs 真实准确率的相关性实验)。

## 8. clean split 怎么解决 leakage?

四层机制(`data_eng/build_dataset.py`):
1. **血缘**:每条样本带 `seed_id` / `family_id`(= intent + 去掉口语前缀后的 base question)/ `entity_key`(产品、小料、OOD 主题)/ `lineage_confidence`。
2. **合并成 group**:并查集把"同 family、同 (实体,intent)、同 seed、以及歧义 evolved 的全部候选 seed"合并为一个 group。
3. **group 级切分**:整个 group 只能进 train/val/test 之一;**含 evolved/onpolicy/unknown 的 group 一律 non-clean → 只进 train**;clean group 只给 val/test(它们是稀缺资源,总池只有 26 个 group / 170 条)。
4. **向后兼容**:`--reuse-split-from` 冻结上一轮 split,只把新样本追加进 test,不重新随机打乱(旧 test 被逐字节保留,已验证 sha256 相同)。
- 结果(`leakage_audit.json`):exact / base / family / split-group / OOD-family / OOD-entity 六个维度 train∩test **全为 0**;test onpolicy/evolved = 0;18 条 assertion 全 PASS;固定 seed=42 两次运行字节一致。

## 9. 为什么旧测试集不能作为最终指标?

- 旧 288 条 test(**数据在 `data_eng/output/`**):**188/288 与 train 精确同题**、**277/288 去掉前缀后同源**、**13/13 OOD 同源**、还混入 **157 条模型自生成数据**(onpolicy 146 + evolved 11)。等于"拿训练题当考卷"。
- 后果实证:README 引用的 92.2% 来自 128 条旧集;换到 clean 集(修复前口径)RL 只有 61.2%;再叠加 prompt 格式问题后 sft/rl 更低。
- 一句话答法:"旧测试集的数字我不用了,因为我和训练数据是同一批基础问题的不同改写;clean 集上我的数字明显更低,这正是泄漏存在的证据。"

## 10. Gold Eval 是什么?

- 一批**人工编写**的评测题(`data_eng/gold_eval_seeds.py`,78 条),每一题都带:
  - `knowledge_source`:答案事实出处(如 `kb.py PRODUCTS(杨枝甘露 18 / 茉莉绿茶 8)`)
  - `novelty_reason`:为什么它是**新业务场景**而不是"换词改写"(如"跨产品比价"、"过敏约束下的推荐")
- 覆盖:price 12 / recommend 12 / after_sale 10 / allergen 10 / hours 8 / delivery 6 / member 5 / store 5 / ood 10。
- 约束:`eval_only=True` → 只进 test;不参与前缀改写/模板扩展/蒸馏/自采样;与 train/val/old_test 的 exact/base/family 重叠**全 0**(逐题记录在 `GOLD_EVAL_AUDIT.md`)。
- 每条 gold answer 都用项目自己的评测口径自检过:78/78 能被判对(领域题命中 keyword、OOD 题命中拒答话术)。
- 另外记录了 **8 组"知识库撑不住所以没造题"**的场景(杯型/套餐价格、坚果/花生过敏原、漏发/错发/超时、积分有效期、加盟/WiFi 密码、超 3 公里与时效承诺、加浓/换奶等定制)→ 说明"不造题、不编事实"。

## 11. 为什么 keyword evaluator 不够可靠?

- 规则就两行(`sft/eval_domain.py`):`hit = any(k in resp for k in keywords)`、`refused = any(m in resp for m in REFUSAL_MARKERS)`。
- 实测反例(在 `prompt_ablation/` 里可查):
  1. `dpo/chat` 回答"门店关门时间是晚上 **21:30**"(事实错,应为 22:00),因命中 `21:30` 被判**答对**;
  2. `dpo/raw` 在 OOD 题先输出"最近的酒店在哪里?…"再补一句拒答,**仍算成功拒答**。
- 结构性问题:数字 keyword 无词边界(`"12"` 会命中 `"2012"`/`"12:00"`);拒答词 `咨询/抱歉` 过宽,正常业务回答里出现也会被判成拒答;回答长度/重复/是否真回答问题都不进入判定 → **复读、被截断的回答只要含关键词就算对**。
- 改进方向(未实施,只作为讨论):要求命中全部 keyword、数字加词边界、拒答必须出现在前 2 句内、增加"重复率"指标。

## 12. 为什么模型生成会出现 repetition?

- 现象证据(`prompt_ablation/` 的 prediction 原文):`sft/chat` 答完"芝士奶盖四季春 16 元一杯…总费用是 19 元。"之后继续输出"问题:芝士奶盖四季春…";`base/*` 甚至复读问题或输出 `forCell forCell…`。
- 归因(基于真实配置,不猜测模型内部):
  1. 模型小:0.5B;
  2. 训练少:SFT 只有 **600 步 / 2000 条**(train_loss 0.7875→0.5167);
  3. 监督数据高度重复:train 2676 条里 1480 条是 on-policy 自生成、且同一问题多个答案;并且 SFT 实际消费的前 2000 行里 OOD = 0 条;
  4. 解码是 **greedy**(`do_sample=False`),评测里**没有设 repetition_penalty / no_repeat_ngram_size**。
- ⚠️ 注意:没有做过"加 repetition penalty"或"训更多步"的对照实验,所以只能讲现象与上述工程原因,不要声称已定位到唯一根因。

## 13. 为什么 EOS 不命中?

- 实测:stop 条件同时放了 `eos_token_id=151643(<|endoftext|>)` 和 `151645(<|im_end|>)`,但 base/sft/rl 在 **chat 与 raw、32 与 64 token 下都不输出任何终止符**(EOS 命中 0%);只有 DPO 在 raw 下 40%。
- 说明这不是"停止符配错"(我一开始也怀疑这个,专门验证过),而是**模型没有学会在回答结束后输出 `<|im_end|>`**。
- 旁证:RL 训练日志里 `mean_terminated_length = 0.0`、`clipped_ratio = 1.0`,说明 GRPO 阶段 100% 的 rollout 也是被 64 token 截断的,模型在此过程中同样没学到终止符。
- 影响:每条都跑满上限 → 评测更慢;回答尾部出现复读;但因为关键词通常在正确的前半段,metric 受害有限(反而可能带来误命中)。
- 修法(未做):重训让模型学会 `<|im_end|>`,或在评测里加"截断启发式"(会改变评测语义,需谨慎)。

## 14. 为什么不同 prompt format 对 checkpoint 有影响?

- 因为**各阶段训练用的格式不同**(见 Q4):SFT 是 chat template,DPO/RL 是裸 prompt。
- 实测(`PROMPT_ABLATION.md`,4 模型 × 2 格式 × 5 题 = 40 次生成):
  | model | chat 准确率/EOS/平均token | raw 准确率/EOS/平均token |
  |---|---|---|
  | base | 0% / 0% / 32.0 | 0% / 0% / 32.0 |
  | sft | 40% / 0% / 32.0 | 20% / 0% / 32.0 |
  | dpo | 60% / **0%** / 32.0 | 60% / **40%** / **28.4** |
  | rl | 60% / 0% / 32.0 | **80%** / 0% / 32.0 |
- 结论:**DPO 对格式最敏感(行为层面)**,SFT/RL 只有单题量级的内容差异(n=5,不能定论),base 指标无差异但内容差异大(chat 会把 system prompt 续写下去)。
- 面试口径:"这是 diagnostic ablation,不是最终口径决策;最终评测我打算按各阶段训练格式分别报告,并在报告里明确标注 training-format-matched evaluation。"

## 15. RoPE 为什么比 learned positional embedding 更适合?

- **本项目实测**:同规模(12.19M vs 12.29M)、同 2000 步、同数据/TinyStories,val loss **2.3779 vs 2.5148(−5.4%)**(`models/exp_rope/log.jsonl` vs `exp_learned`)。
- 机理(要跟实测区分开):RoPE 把位置信息编码成 Q/K 的旋转,注意力分数里自然包含**相对位置**;不占额外参数表、对长度更友好;learned 位置嵌入是固定长度的可学表,超出 `block_size` 就没有位置向量。
- ⚠️ 注意:本项目的 `block_size=256`,**没有做长度外推实验**,所以只能说"更长上下文外推更好"是原理性理由,不能说"我验证了外推能力"。

## 16. GQA 为什么减少 KV cache / attention 的成本?

- **本项目实测的只有**:(`models/exp_gqa`)KV 头数 6→3,参数量 **12.29M → 11.41M(−7.2%)**,val loss 2.5148 → 2.5261(仅差 +0.4%)。
- ⚠️ **KV Cache 在本项目没有实现**:`pretrain/model.py:257-271` 的 `generate()` 每步都 `forward()` 整个上下文,没有 `past_key_values`。所以"省 KV cache 显存"是**机理层面**的说法,本项目没有测到。
- 标准答法:"GQA 让多个 query 头共享一组 K/V 头,K/V 投影的参数与推理时 KV cache 的体积都按比例下降;我这次测到的是参数量 −7.2% 且效果几乎不变,因为我的 from-scratch 实现没有做 KV cache,所以缓存收益我没有实测。"

## 17. SwiGLU + RMSNorm 的作用?

- **本项目实测**(`models/exp_swiglu`):同为 12.29M 参数、2000 步,val loss **2.4356 vs 基线 2.5148(−3.2%)**。
- 机理:SwiGLU 用门控 `silu(xW_gate) ⊙ (xW_up)` 提升表达力(因此用 8/3 宽度把参数量对齐回 GELU 的 4×);RMSNorm 去掉均值中心化、只按均方根缩放,计算更省,是 Llama/Qwen 的标配。
- ⚠️ 注意:这两个改动是**一起做的**,项目里没有 swiglu-only / rmsnorm-only 的单独实验,所以不能声称"分别贡献了多少"。

## 18. validation-loss scaling trend 怎么解释?

- 数据(全部来自 `models/exp_*/log.jsonl`,同为 2000 步、同 tokenizer/数据/block_size):

| params | val loss |
|---|---|
| 0.95M | 3.5068 |
| 4.26M | 2.8285 |
| 12.29M | 2.5148 |
| 27.40M | 2.3005 |

- 解释:参数越大 loss 越低,但边际收益递减;12.29M 继续训练到 6000 步可到 **1.9714**(`models/pretrain`),说明 2000 步时四个点**都没到各自收敛**。
- ⚠️ 两点必须主动承认:① 没有做 power-law 拟合(代码里没有拟合逻辑),所以不能说 "scaling law / 指数";② 4 个点的 token 预算相同,点间差距里混有"训练不足"的成分。
- 推荐措辞:"不同参数规模下的 validation-loss scaling trend(4 档,同预算,未做幂律拟合)"。

## 19. Base/SFT/DPO/DAPO 每阶段到底改变了什么?

以旧 288 条集(历史)+ clean 183 条(修复前口径)为参照,并注明来源:

| 阶段 | 做了什么 | 观测到什么 |
|---|---|---|
| Base(Qwen2.5-0.5B,4bit) | 无 | 旧 288:领域正确率 7.3%;clean:「21.1%」,基本不会客服 |
| +SFT(LoRA,600 步) | 对话格式 + 知识库答案 + assistant-only loss | 旧 288:88.0% 但**拒答 0.0%**;clean:「60.2% / 8.3%」→ 会答不会拒 |
| +DPO(gold-vs-worst 偏好对) | 二分类式偏好优化(beta=0.1) | 历史 138 条集:82.4% / 53.8%;clean 集**尚未评测**;当前 `models/dpo` 是 10 步版 |
| +GRPO/DAPO(规则奖励 + 在线 rollout) | 命中/拒答 ±1、简洁 +0.1、OOD 上采样 20%、KL 0.04 | 旧 288:拒答 **100%**、领域正确率 83.3%;clean:「47.2% / 90.0%」→ 拒答变强、正确率下降 |
> 表中「」内为修复前口径的 clean 结果,不能作为最终指标(需按新口径重测)。

- 核心 trade-off:对称 ±1 奖励 + OOD 上采样让模型更倾向拒答,换来拒答率上升、领域正确率下降。这是可解释、可讨论的结果,不要美化成"全面提升"。

## 20. 如果面试官质疑数据泄漏,你如何解释?

四步回答(全部有产物支撑):
1. **承认并量化**:旧 288 条 test 中 188 条与 train 精确同题、277 条去掉前缀同源、13/13 OOD 同源、157 条是模型自生成数据;所以旧指标(README 的 90.5%/92.2%)作废。
2. **根因**:切分是"样本级随机分层",去重只按 (q,a);数据集建立在 55 个种子 → 256 个基础问题 × 5 种前缀之上,同一基础问题的改写必然跨 split;而且当时没有任何血缘字段。
3. **修复**:引入 `seed_id / family_id / entity_key` 血缘 + 并查集 group + group 级切分(同 group 只进一个 split);evolved/onpolicy/unknown 一律只进 train;冻结旧 split 只追加新测试题;输出 6 层 leakage audit(全 0)+ 18 条 assertion + 可复现(固定 seed,两次运行字节一致)。
4. **代价与残留**:因为 on-policy/蒸馏覆盖了几乎所有 price/recommend 家族,这些 intent 无法进 test → 我人工补了 78 条 Gold Eval(边界场景、KB 有据);仍有"同实体不同场景"的弱关联,已在 `GOLD_EVAL_AUDIT.md` 逐题标注。
5. 收尾一句:"clean 集上我的数字明显变低(rl 拒答 90% 但领域正确率 47.2%),这正好说明旧数字是被泄漏撑起来的;我宁可报低但可信的数字。"

---

## 附:被问到"还有哪些没做"时的诚实清单

- 最终 clean 指标(修复 prompt/EOS/长度后)**尚未产出**;现有 clean 数字是修复前口径。
- GSPO、MQA、ALiBi、untied embedding、KV Cache:**只有代码/开关,没有实验**。
- Scaling:**没有幂律拟合**,只有 4 档同预算趋势。
- FlashAttention:**用的是 PyTorch SDPA**,不是自研 kernel。
- RFT:只覆盖 price/recommend(采样时 `load_rows` 取文件前 N 行),**没有覆盖 OOD**;rejected 样本与 reward 值都没落盘。
- LLM-as-Judge:只有 15 条抽样的历史结果,clean 集没有。
- reward hacking:只做了工程层面的缓解(权重/KL/上采样),没有系统的奖励-准确率分析。

---

## 附录:Standard GRPO 实验事实(2026-09-29 追加,只读归档)

**Standard GRPO:**
- 使用 TRL `GRPOTrainer`
- `loss_type = grpo`,`importance_sampling_level = token`(**不是 DAPO、不是 GSPO、不是 PPO**)
- **实际运行过**:第一阶段从 SFT 适配器连续训练到 **step 200**(数据 = `data_eng/output_clean/train.jsonl`,
  205 prompts,与 clean test 的 exact/base 重叠为 0)
- 随后从 `checkpoint-200` 尝试 resume,**暴露了 PEFT active adapter 恢复问题**
- 因此**没有**把错误 resume 之后的 checkpoint 当作最终 benchmark

**面试可解释的问题:为什么训练恢复后 KL 会突然从 0.43 降到 0.009?**

> Trainer state 的恢复 ≠ active policy adapter 权重一定恢复。本次实验中 optimizer / scheduler /
> RNG / global_step **都恢复了**,但 **policy adapter 没有正确恢复**(transformers 5.14.1 的
> `Trainer._load_from_checkpoint` 在 checkpoint 目录含 `ref/` 子目录时,只加载子目录里的 adapter,
> 跳过主 `adapter_model.safetensors`),因此 policy 回到了接近 SFT initialization 的状态,
> KL(policy‖SFT) 从 0.4317 掉到 0.0088。

细节与完整证据(含 ref 与 `models/sft` 的 hash 一致、指标区间)见 `data_eng/GRPO_EXPERIMENT_ARCHIVE.md`。
