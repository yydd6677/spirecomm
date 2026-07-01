# Run Value Model / Environment-Guided Policy Improvement 下一轮设计方案

> 本文基于项目上下文文档 `RUN_VALUE_MODEL_CONTEXT_FOR_EXTERNAL_LLM.md`，目标是把当前 `V_run(s)` 从“可运行但不安全”的原型，推进到可诊断、可校准、可保守接入策略改进的系统。

## 0. 核心结论

当前 `remaining_floor MAE = 6.78` 不应该被理解成“模型只差一点调参就能到 `<3`”。它混合了三类问题：

1. **目标噪声 / 条件不可辨识**：early floor 的单条 rollout 终局标签本身方差极大。若输入特征不能完整表达未来路线、RNG、池、策略分支和 deck/relic 结构，同一类可见状态会对应非常宽的 final floor 分布。
2. **训练目标与策略接入目标不一致**：当前训练的是行为策略下的 `V^π(s)`，但 rerank 需要的是局部 `Q(s,a)` 排序。全局 MAE 6.8 的模型不可能稳定比较大量只有 `0.1–2.0 floor` 差距的候选动作。
3. **状态表征覆盖广但结构弱**：2336 维特征包含很多信息，但 deck/relic/map/monster/choice 大量是 hash bag 和摘要，缺少对“路线组合、boss/elite readiness、deck synergy、资源时序”的显式建模。

因此，下一轮不应以“直接把全局 row-level MAE 压到 `<3`”为唯一目标。更合理的主线是：

- 把 value target 从单一 scalar regression 改成 **distributional / ordinal survival + quantile + calibrated scalar**。
- 把训练权重从纯 row-level 改成 **run-balanced + floor-balanced + phase-aware** 的混合目标。
- 把输入从纯 hash bag 改成 **tabular summary + 显式 path/readiness/wiki features + 后续 typed set encoder**。
- 把策略接入从无闸门 rerank 改成 **shadow → hard branch validation → uncertainty-gated rerank → AWR/advantage-weighted BC → 小步 on-policy iteration**。
- 把成功标准从单个 MAE 改成 **floor bucket、seed-balanced、calibration、branch ranking regret、accepted override safety** 的组合。

---

## 1. 当前 MAE 卡在 6.8 的最大 3 个原因

### 原因一：early/mid floor 的单条 rollout label 方差太大，且存在特征不可辨识

#### 证据

当前最佳模型的 overall validation `remaining_floor MAE = 6.779`，但误差主要集中在 early/mid floors：

| floor bucket | MAE |
|---|---:|
| 00-04 | 9.675 |
| 05-09 | 9.218 |
| 10-14 | 8.575 |
| 15-19 | 7.274 |
| 20-24 | 5.995 |
| 25-29 | 5.620 |
| 30-34 | 5.302 |
| 35-39 | 3.855 |
| 40-44 | 2.272 |
| 45-49 | 0.834 |
| 50-54 | 0.257 |

这说明模型在 late game 已经接近或低于 `<3`，真正的问题不是“所有状态都预测差”，而是 floor 0-34 的未来分布太宽。

在 Slay the Spire 里，early floor 的最终结果高度依赖后续路线、奖励、商店、事件、boss、elite、potion 使用、卡牌选择等长程变量。即使游戏和 policy 在完整 env state 下是确定的，只要模型输入没有完整表达这些变量，训练数据对模型来说就是“同样看起来相似的状态，最后可能死在 Act 1，也可能进 Act 3 或通关”。

当前 scalar regression 学到的是条件均值或近似均值；对于宽分布，MAE 不能无限下降。尤其是单条 rollout 的 `final_floor - current_floor` 被当作每个 early state 的标签时，label 更像是一个高方差样本，而不是稳定的 value expectation。

#### 如何验证

1. **估计 floor/context 条件下的不可约 MAE 下界**  
   按 `floor bucket + act + boss + hp bucket + deck readiness bucket + path bucket` 分组，计算组内 `final_floor` 分布。对每组用 median 作为最优 MAE 预测，得到：
   - bucket-only lower bound
   - bucket + deck summary lower bound
   - bucket + route summary lower bound
   - bucket + deck + route + resource lower bound

   如果这些下界在 floor 0-20 仍然明显大于 3，则全局 `<3` 对单条 terminal label 不现实。

2. **kNN / embedding neighborhood 方差诊断**  
   在当前 2336 维特征或模型倒数第二层 embedding 上找近邻，统计近邻 final floor 的 MAD/IQR。若 early floor 近邻终局差异仍很大，说明不是简单扩大 MLP 能解决。

3. **同一 root 的 continuation / branch 重采样**  
   对 high-leverage early/mid states 做 continuation rollouts。若 continuation policy 是确定性的，可以引入小幅 stochastic policy sampling 或 random-action perturbation，估计“策略族下的 expected value”和 uncertainty。  
   这不是为了把所有训练数据都变成 multi-rollout，而是为了判断：当前 scalar label 的噪声有多大、哪些 floor/phase 的 value 有可学习信号。

4. **移除/替换 seed hash 的泛化检查**  
   当前 numeric summary 包含 seed/dungeon hash。若训练和评估 seed 范围有重叠，可能出现伪泛化；若评估 seed 不重叠，seed hash 多半只是噪声。建议做：
   - no-seed-hash ablation
   - exact RNG features vs seed hash ablation
   - held-out seed range validation，而不仅是 `seed % 10 == 0`

---

### 原因二：`V^π(s)` 的 row-level MAE 与 action rerank 所需的局部 `Q(s,a)` 排序不是同一个问题

#### 证据

当前 workflow 中，value model 用来对 action after-state 打分：

```text
q_env(action) = floor(after_state) + V_remaining(after_state)
```

但训练标签是行为策略 rollout 下的状态终局：

```text
V^π(s) = E[final_floor | state s, continue with policy π]
```

这两者存在三层错配：

1. **训练只覆盖 behavior distribution，rerank 会产生 off-policy after_state**  
   非 chosen action 的 after_state 未必在训练分布中出现。即使模型在 validation rows 上 MAE 尚可，也可能对 branch after_state 排序不可靠。

2. **全局 MAE 远大于 action 间真实差距**  
   很多 combat/action candidate 的长期价值差距只有 `0.1–2 floor`，而当前 scalar error 是 `~6.8`。直接归一化 value score 并参与 rerank，会把噪声变成动作偏好。

3. **offline action policy top1 高不代表 runtime 好**  
   pilot 中 action policy offline top1 达到 `0.731`，但 policy blended runtime 从 baseline `32.07` 降到 `29.72`，value rerank 更降到 `28.70`。这说明它学到了 q_env 的排序，但 q_env 本身或使用方式不安全。

#### 如何验证

1. **构建 hard branch validation set**  
   从 shadow logs 中选择 value 与 baseline 分歧的 roots。对每个 root 的 top candidates 做真实 continuation rollout，得到实际 `Q(s,a)` 或近似 `Q(s,a)`。  
   评估：
   - value top1 是否等于 branch rollout top1
   - predicted gap 与 actual gap 的相关性
   - only high-confidence subset 的 regret
   - 按 phase/action kind/floor bucket 分解

2. **after_state OOD 诊断**  
   对 chosen-action after_state 与 nonchosen branch after_state 计算：
   - embedding distance
   - ensemble disagreement
   - density / kNN distance
   - phase/action-kind 分布差异

   如果 rerank 选择的 action 经常落在 OOD after_state，说明必须加 conservative gate 和 branch data，而不是继续只优化 V 的 row-level MAE。

3. **训练集同时包含 `state_before` 与 chosen `state_after`**  
   如果当前 value training 主要用 `state_before`，而 runtime q_env 使用 `after_state`，应立即加入 chosen `state_after` 样本，并分别报告 before/after validation MAE。  
   这是低成本但高价值的错配修复。

---

### 原因三：2336 维特征覆盖面广，但缺少关键结构，hash bag 让模型难以学习长程规划关系

#### 证据

当前 state feature 包含很多组：

- numeric summary
- deck/hand/draw/discard/exhaust card bag
- relic/potion/monster bag
- choice/legal action bag
- future card/relic/boss relic pool bag
- map/reachable map bag
- player/monster power bag

这比早期版本强很多，但大部分关键对象仍是 hash bucket 或粗摘要：

- deck 是 unordered card bag，缺少 card ID 的稳定 embedding、升级状态、zone、cost curve、draw/cycle 结构、synergy。
- relic/potion 是 bag，缺少与 deck、boss、elite、路线的交互。
- map bag 有 future/reachable counts，但缺少“路径级组合”，例如“下个 elite 前是否有 rest/shop/treasure”“强制 elite 前有几场怪”“最安全路径与最贪路径的差异”。
- choice/action 信息是 state-level choice summary，不等于每个 action 的 after-state value。
- encounter readiness 只靠模型从 hash 中自己发现，对 5k runs 的数据规模来说过难。

历史上 residual floor baseline 是最有效改动之一，说明当前模型很大一部分能力来自 floor prior；late fusion、final cls、survival 初版没有稳定更好，说明“头部加一点分类”不足以解决结构缺失。

#### 如何验证

1. **显式 feature ablation / addition**  
   分批加入 path DP、deck readiness、encounter readiness、resource pressure features。每批看：
   - early/mid floor MAE
   - calibration bias
   - branch ranking
   - phase-specific improvement

2. **hash collision / entity ID ablation**  
   用稳定 entity vocab 替代 hash bucket，至少先对 card/relic/potion/monster 做 typed ID embedding 或 one-hot sparse lookup。比较：
   - same data, same loss, hash bag vs typed bag
   - floor 0-20 MAE
   - card reward/shop/boss relic phase MAE

3. **structured encoder 对照**  
   在 tabular MLP 之外训练一个小型 DeepSets / Set Transformer 版本，用同样 target 和 weights。若结构版在 early/mid 和 branch ranking 上明显更好，说明瓶颈确实是表示而不是纯 label 噪声。

---

## 2. MAE `<3` 是否现实？

### 2.1 对当前 single-rollout row-level target：整体 `<3` 暂时不现实

在现有定义下：

```text
target = terminal.final_floor - current_floor
sample = 单条 behavior rollout
metric = row-level remaining_floor MAE
```

整体 `<3` 不现实，原因是：

- late floor 已经 `<3`，但 early/mid floor 仍在 `5–10`。
- row-level metric 混合了大量高度相关的 combat rows。
- early floor 的 terminal label 对当前可见 state 来说高方差。
- rerank 需要的是 local action advantage，而不是全局 terminal floor scalar。

如果想让 row-level overall MAE `<3`，必须满足至少一个条件：

1. 输入几乎完整表达全部决定性变量，包括可复现的 RNG、future pools、map、policy continuation 和所有实体结构；
2. target 改成 multi-rollout expected value，而不是单条样本；
3. 训练/评估只看 late floor 或低不确定性状态；
4. value model 不再是纯状态 V，而是 action-conditioned Q，并用真实 branch continuation 数据训练。

这些都不是“加几层 MLP”能解决的。

### 2.2 更合理的分阶段目标

建议把目标拆成四类。

#### A. 预测类目标

| 指标 | 当前 | 第一阶段目标 | 第二阶段目标 |
|---|---:|---:|---:|
| row-level remaining MAE | 6.78 | 6.1–6.4 | 5.3–5.8 |
| seed/run-balanced MAE | 待测 | 比 floor baseline 降低 12–18% | 降低 20–30% |
| floor 0-9 MAE | 9.2–9.7 | < 8.5 | < 7.5 |
| floor 10-19 MAE | 7.3–8.6 | < 7.0 | < 6.0 |
| floor 20-34 MAE | 5.3–6.0 | < 5.0 | < 4.3 |
| floor 35+ MAE | 已接近 | 保持不退化 | < 2.5 |

#### B. 校准类目标

| 指标 | 目标 |
|---|---|
| floor bucket mean calibration bias | 绝大多数 bucket 绝对偏差 < 0.75 floor |
| survival ECE | < 0.05–0.08 |
| win / act clear Brier score | 相对 baseline 改善 10%+ |
| quantile coverage | q10/q90 覆盖率接近标称，误差 < 5–8% |
| uncertainty vs error correlation | ensemble std / interval width 与实际 absolute error 正相关 |

当前中段明显低估 remaining floor，例如 20-24、25-29、30-34 bucket 预测均值比真实均值低约 2 floor 以上。这个问题对 rerank 很危险，优先级高于单纯 MAE。

#### C. action-ranking 目标

| 指标 | 目标 |
|---|---|
| hard branch top1 | 在 high-confidence subset 上显著高于 baseline |
| mean regret | high-confidence accepted subset 接近 0 或为正收益 |
| predicted gap calibration | 预测 +2 floor 的动作，实际平均至少非负且最好 > +0.5 floor |
| phase-specific regret | combat/card/shop/map 分别报告，不混在一起 |
| OOD rejection quality | 被 gate 拒绝的 roots 应该有更高 uncertainty/error |

#### D. runtime safety 目标

| 指标 | 第一阶段标准 |
|---|---|
| shadow branch success rate | > 99.5% |
| accepted override rate | 先限制在 1–5% roots |
| paired rollout mean_floor | 不低于 baseline；更理想是 +0.3 到 +0.8 |
| early death rate | 不上升 |
| potion/run / shop spend/run | 不出现异常漂移 |
| wins/300 | 不因 gate 明显下降 |
| seed holdout | 接受测试必须用未参与 value/action 训练的 seeds |

---

## 3. 下一版 value model 的具体设计

### 3.1 先明确要学什么

下一版 `V_run` 不应只学一个点估计：

```text
remaining_floor_mean
```

而应学一个关于终局的分布：

```text
P(final_floor >= k), k = 1..50
quantiles(final_floor)
E[final_floor]
P(win)
P(act1/act2/act3 clear)
P(death within next N floors)
```

这样做的目的不是“分类一定降低 MAE”，而是：

- early floor 的真实目标本来就是多峰/宽分布；
- survival curve 可以同时提供 expected floor、death risk、act clear probability；
- quantile/ensemble 可以为 rerank gate 提供 uncertainty；
- calibration 可以直接验证 value 是否能安全用于策略。

### 3.2 input schema

建议分两步实现：先做 **tabular v2**，再做 **hybrid structured v3**。

#### v2：保留 2336 维，新增显式 planning features

```text
input_v2 =
  current_2336_features
  + path_dp_features
  + deck_readiness_features
  + encounter_readiness_features
  + resource_pressure_features
  + choice_context_features
  + phase/floor/boss stable IDs
```

重点是先用最小工程成本补齐当前 hash bag 最弱的地方：

- 路线组合
- boss/elite readiness
- deck tempo / scaling / cycle
- potion/gold/hp resource timing
- reward/shop choice 是否填补 deck 缺口

#### v3：typed set / entity encoder

```text
global numeric encoder:
  floor, act, phase, hp, gold, boss, room, rng summary, path scalar

card set encoders:
  deck
  hand
  draw
  discard
  exhaust
  card reward candidates
  shop cards

relic set encoder:
  owned relics
  shop relics
  boss relic candidates

potion set encoder:
  owned potions
  shop potions
  reward potions

monster/power encoder:
  live monsters
  player powers
  monster powers

map/path encoder:
  path DP aggregate first
  later optional path-prefix sequence encoder or small graph encoder

choice/action context encoder:
  legal action type counts
  candidate item embeddings
  prices / affordability / skip / leave
```

推荐结构：

```text
entity token =
  learned_id_embedding
  + static wiki feature projection
  + zone/type embedding
  + numeric attributes
```

聚合方式先用 DeepSets / attention pooling，不必一开始上大 Transformer：

```text
group_embedding = Pool(MLP(entity_tokens))
global_embedding = MLP(numeric + handcrafted)
trunk_input = concat(group_embeddings, global_embedding)
trunk = residual MLP or small gated MLP
heads = phase/floor-aware heads
```

### 3.3 output heads

建议输出：

1. **ordinal survival head**

```text
logit_k = logit P(final_floor >= k), k = 1..50
expected_final_floor = sum_k sigmoid(logit_k)
expected_remaining = expected_final_floor - current_floor
```

可加 monotonic regularization：

```text
P(final >= k+1) <= P(final >= k)
```

实现上可以先不强制单调，只加 penalty；若有效，再用 cumulative parameterization。

2. **scalar residual head**

保留当前有效的 floor residual baseline：

```text
pred_remaining = floor_mean_remaining[current_floor] + residual_head
```

但 scalar head 不再单独主导训练，而是作为 expected value 的校准锚点。

3. **quantile heads**

```text
q10, q25, q50, q75, q90 for final_floor or remaining_floor
```

用途：

- uncertainty interval
- conservative rerank
- calibration check

4. **auxiliary binary heads**

```text
win
act1_clear
act2_clear
act3_clear
death_next_3
death_next_6
death_before_next_elite
death_before_boss
```

后两个如果能从 trajectory/map 计算出来，优先加入。它们仍是从终局/轨迹派生，不是手写 reward。

5. **optional phase-specific heads**

共享 encoder，按 phase/floor gate 到不同 head：

```text
head_COMBAT
head_CARD_REWARD
head_SHOP
head_MAP
head_EVENT
head_CAMPFIRE
head_BOSS_RELIC
```

先不要做复杂 MoE；先做 shared trunk + phase-specific linear heads，作为低风险版本。

### 3.4 target definition

#### 主 target：distributional final floor

对每条样本：

```text
y_final = terminal.final_floor
survival_target[k] = 1[y_final >= k]
remaining_target = y_final - current_floor
```

这仍然只用真实 rollout 终局，不引入手工 reward。

#### TD(lambda) target：只在 room/floor-level transition 上使用

不建议对每个 combat micro-decision 做 TD(lambda)，因为同一房间内大量相邻状态高度相关，会进一步放大 combat rows。

建议构建 reduced transition stream：

```text
state at meaningful boundary:
  room entry
  combat end
  reward screen
  map choice
  shop entry/exit
  campfire decision
  boss relic choice
```

定义：

```text
r_t = floor_{t+1} - floor_t
G_t^λ = r_t + γ * [(1 - λ) * V_target(s_{t+1}) + λ * G_{t+1}^λ]
```

这里 `γ` 可以接近 1，因为目标是 floor count，不是折扣 reward。TD(lambda) 的用途是降低 early target 方差、提高 temporal consistency；不要用它完全替代 terminal MC target。建议：

```text
total_value_target = 0.7 * MC_distribution_loss + 0.3 * TD_scalar_consistency_loss
```

TD target 要用 target network / EMA model 生成，避免自举发散。

#### multi-rollout expectation：只用于 high-leverage states 和 branch validation

不建议第一轮把所有 value data 都改成 multi-rollout，成本高且定义容易混乱。优先做：

- value/baseline 分歧 roots
- early/mid high-uncertainty roots
- card reward / shop / boss relic / map high-impact decisions
- potion use / elite前资源 decisions

对这些 root 做 top candidates 的 continuation rollouts，形成 hard Q dataset。这个数据主要服务于 rerank gate 和 action-conditioned Q，不是替换全量 V training。

### 3.5 loss

推荐总 loss：

```text
L =
  1.00 * L_survival_ordinal
+ 0.50 * L_expected_remaining_smoothL1
+ 0.25 * L_scalar_residual_smoothL1
+ 0.30 * L_quantile_pinball
+ 0.30 * L_aux_bce
+ 0.05 * L_survival_monotonic_penalty
+ 0.10 * L_td_lambda_consistency
```

其中：

- `L_survival_ordinal`：对 `final_floor >= k` 做 BCE，可按 threshold 平衡。
- `L_expected_remaining_smoothL1`：用 survival expectation 与 remaining target 对齐。
- `L_scalar_residual_smoothL1`：保留 floor baseline residual 的稳定收益。
- `L_quantile_pinball`：给 uncertainty gate 使用。
- `L_aux_bce`：win/act/death heads。
- `L_td_lambda_consistency`：只在 boundary states 上计算。

不要指望 classification 直接让 MAE 大幅下降。它的价值主要是校准、uncertainty、风险分解和 action gate。

### 3.6 weighting / sampling

当前 row-level 训练会放大长 run 和 combat micro-decisions。建议 batch sampler 用混合策略，而不是单一权重：

```text
40% run-balanced samples:
  每个 run 总权重相同

30% floor-balanced samples:
  floor bucket 大致均衡，特别补 early/mid

20% phase-balanced samples:
  CARD_REWARD / SHOP / MAP / BOSS_RELIC / EVENT 不被 COMBAT 淹没

10% runtime-frequency samples:
  保留真实线上分布
```

同时建议：

- `forced_single` 默认不进主训练，或极低权重。
- combat rows 做 thinning：同一 turn/同一 root 高度相似的 rows 只保留代表状态，或 cap 每房间 combat rows。
- 对每个 run 设置最大样本数 cap，例如每 run 每 floor/phase 至多 N 条。
- validation 同时报告 row-level 和 weighted metrics，不要只看一种。

### 3.7 architecture

#### 第一轮可落地版本

```text
model_v2 =
  LayerNorm(input_v2)
  Residual MLP hidden=512 depth=4 dropout=0.05
  FiLM / gated conditioning by phase + floor bucket
  shared trunk
  phase-specific lightweight heads
  output distributional + scalar + quantile + aux
```

关键点：

- hidden 从 384 提到 512 可以做，但不是核心。
- 保留 residual_floor_baseline。
- phase/floor conditioning 比盲目加深更优先。
- 训练 5 个不同 seed 的 ensemble，用于 uncertainty 和 gate。

#### 第二轮结构版

```text
global numeric MLP -> 256
card DeepSets/attention encoder -> 256
relic encoder -> 128
potion encoder -> 64
monster/power encoder -> 128
map/path encoder -> 128
choice encoder -> 128
concat -> residual trunk 512 x 3
heads -> distribution/scalar/quantile/aux
```

先用 DeepSets，不要一开始上复杂 Transformer。只有当 typed set 明显优于 tabular v2 时，再扩大为 Set Transformer 或 map graph encoder。

### 3.8 validation metrics

必须新增以下 dashboard：

#### Prediction

- row-level MAE / RMSE / SmoothL1
- seed-balanced MAE
- run-start MAE
- room-boundary MAE
- before-state MAE vs chosen-after-state MAE
- floor bucket MAE
- phase-specific MAE
- act/boss-specific MAE

#### Calibration

- bucket mean predicted vs true
- survival ECE
- win/act/death Brier score
- quantile coverage
- reliability diagram
- predicted interval width vs absolute error

#### Branch/action

- branch top1 accuracy
- branch mean regret
- predicted gap vs actual gap calibration
- high-confidence subset regret
- value-vs-baseline disagreement rate
- OOD score distribution for accepted/rejected roots

#### Runtime

- shadow branch error rate
- accepted override rate
- paired rollout mean_floor
- wins/300
- death floor distribution
- potion/run
- shop spend/run
- action kind drift

---

## 4. Wiki / game-knowledge features 的优先级

这些 features 应该作为 **state covariates**，不是 reward。原则是：

```text
feature 可以告诉模型“当前局面有什么事实”
loss 仍然只来自 rollout outcome
不要把 feature 直接加权进 action score
```

下面按优先级列出 30 个建议先做的 feature。

### 4.1 路线 / map planning features

| # | feature | 让 V 学到什么 |
|---:|---|---|
| 1 | `floors_to_next_elite_min/mean/max` | 当前资源压力是否马上要接受 elite 检验 |
| 2 | `rest_before_next_elite` | 低 HP 是否还有恢复窗口 |
| 3 | `shop_before_next_elite` | gold/potion/card gap 是否有补救机会 |
| 4 | `treasure_before_next_elite` | 能否在 elite 前获得无代价强度 |
| 5 | `forced_elite_within_3/5_floors` | 路线是否已经锁死高风险 |
| 6 | `best_path_elite_count_before_boss` | 当前 run 的贪路线潜力 |
| 7 | `safest_path_elite_count_before_boss` | 当前 run 的保命路线选择 |
| 8 | `max_safe_elites_given_rest/shop` | 是否有能力拿更多 relic |
| 9 | `min_rest_distance` | HP 与 smith/rest 决策的时序 |
| 10 | `path_flexibility_entropy` | 未来路线选择空间，避免把可调整局面误判成死局 |
| 11 | `boss_id + floors_to_boss + rests_before_boss` | boss-specific readiness 与剩余准备时间 |
| 12 | `path_prefix_pareto_risk_reward` | 不只看节点数量，而看风险/收益组合 |

当前 map bag 已经有 reachable counts，但缺少 path-level composition。优先做 path DP，因为它对 early/mid floor 的不可辨识影响最大。

### 4.2 Deck readiness features

| # | feature | 让 V 学到什么 |
|---:|---|---|
| 13 | `frontload_damage_turn1/turn2` | 打 Nob、Slime、Act 2 hallway 是否能快速降敌方输出 |
| 14 | `block_per_turn_estimate` | 普通战和 boss 的 HP 消耗趋势 |
| 15 | `aoe_damage_score` | Sentries、Slime、Act 2 多敌房间风险 |
| 16 | `scaling_score` | 长战 boss/Act 3 是否能成长 |
| 17 | `deck_cycle_speed` | 关键牌/relic/power 的出现频率 |
| 18 | `draw_density` | deck consistency，而不是只看 deck size |
| 19 | `energy_pressure` | 平均手牌 cost 与 energy source 是否匹配 |
| 20 | `exhaust_synergy_score` | Ironclad 的 exhaust engine 是否成型 |
| 21 | `strength_scaling_access` | Demon Form/Inflame/Spot Weakness/Limit Break 类路线 |
| 22 | `weak_vulnerable_access` | 进攻/防守效率的乘法来源 |
| 23 | `starter_burden_count` | Strike/Defend 压力和删牌价值 |
| 24 | `curse_status_burden` | 抽牌质量、商店删除、蓝蜡烛等关系 |
| 25 | `upgrade_leverage_score` | campfire smith 比 rest 更值钱的程度 |
| 26 | `power_setup_safety` | power 多但 frontload 不足时的 early death 风险 |

当前 deck readiness 的 average damage/block/magic 太粗。Slay the Spire 的核心不是平均值，而是“前两回合能否活下来”“是否有 scaling”“能否循环到关键牌”。

### 4.3 Encounter readiness features

| # | feature | 让 V 学到什么 |
|---:|---|---|
| 27 | `nob_readiness` | Act 1 elite 中 skill-heavy deck 的风险 |
| 28 | `lagavulin_readiness` | setup/scaling/burst 是否能通过沉睡窗口转化为伤害 |
| 29 | `sentries_readiness` | AoE、block、daze handling 是否足够 |
| 30 | `act1_boss_readiness_slime/hexaghost/guardian` | boss-specific failure mode |
| 31 | `act2_multi_enemy_readiness` | Act 2 hallway/elite 对 AoE 和 frontload 的要求 |
| 32 | `act2_burst_survival_readiness` | Slavers/Book 等爆发伤害风险 |
| 33 | `act3_scaling_readiness` | Giant Head、boss 长战能否结束 |
| 34 | `time_eater_awakened_one_donu_deca_context` | deck 对 boss机制的相性 |

这些不是手工 reward。它们只是把 wiki/enemy 机制转成可学习特征，让模型不用从 5k runs 的 hash bag 中自己发现“当前 deck 怕 Nob / 不怕 Sentries”。

### 4.4 Resource and economy features

| # | feature | 让 V 学到什么 |
|---:|---|---|
| 35 | `effective_hp_after_burning_blood` | Ironclad 的实际续航，不等于当前 HP |
| 36 | `hp_margin_to_next_elite/boss` | 当前 HP 是否能承受既定路线 |
| 37 | `rest_vs_smith_pressure` | campfire 应该被看作补救还是强化机会 |
| 38 | `potion_slot_pressure` | potion slot 满时奖励 potion 的边际价值变化 |
| 39 | `strong_potion_available_for_elite/boss` | 药水是否是未来关键战安全垫 |
| 40 | `gold_to_next_shop` | 当前 gold 是否能转化为实际购买力 |
| 41 | `removal_affordability_and_pressure` | 删除 Strike/Curse 的机会成本 |
| 42 | `shop_budget_after_reserve` | 花钱是否破坏未来关键商店计划 |
| 43 | `relic_synergy_tags` | relic 与 deck archetype 的乘法效果 |
| 44 | `boss_relic_downside_context` | Busted Crown/Ectoplasm/Sozu 等 downside 与当前 run 的相性 |

资源 features 尤其重要，因为当前 teacher 的短视主要体现在 HP/potion/gold 的跨房间价值上。

### 4.5 Choice-context features

| # | feature | 让 V 学到什么 |
|---:|---|---|
| 45 | `card_reward_candidate_fills_gap` | 当前奖励是否补 frontload/block/aoe/scaling/draw/energy 缺口 |
| 46 | `skip_is_reasonable_context` | deck 已成型或候选污染 deck 时，skip 的长期价值 |
| 47 | `shop_affordable_gap_fillers` | 商店是否有能解决当前最大短板的 item |
| 48 | `boss_relic_candidate_tradeoffs` | 能源、抽牌、经济、药水限制等长期风险 |
| 49 | `event_option_resource_delta_context` | 事件选项对 HP/gold/card/relic 的长期影响 |
| 50 | `map_choice_future_option_value` | 当前 map action 对后续路线自由度的影响 |

choice-context features 对 CARD_REWARD / SHOP / BOSS_RELIC / EVENT phase 很关键，因为这些 phase 数量少，纯 row-level combat 数据会把它们淹没。

---

## 5. 如何把 value model 安全用于策略提升

### 5.1 先回答：应该先修 value calibration，还是先改 rerank/gating？

结论：**两者必须同时推进，但 runtime 接入顺序必须先 gate 后 takeover。**

不要等到全局 MAE `<3` 才使用 value；这可能永远不是正确目标。  
但也绝不能再让未校准的 scalar value 无闸门参与每个 root 的 rerank。

正确路径是：

```text
train calibrated distributional V
→ shadow all phases
→ hard branch validation
→ only high-confidence gated override
→ action-conditioned policy learns stable advantage
→ small-step on-policy iteration
```

### 5.2 shadow validation

shadow 阶段不改变动作，只记录：

```text
state
legal actions
baseline top action
value top action
ensemble mean/std
quantile interval
survival/death risk
predicted value gap
OOD score
phase/action kind/floor bucket
branch success/error
```

shadow 成功标准：

- branch/value 计算成功率 > 99.5%
- latency 可接受
- value 与 baseline 的 disagreement 分布可解释
- high predicted gap 的 roots 在 hard branch validation 中确实更可能有实际收益
- uncertainty 高的 roots 确实 error/regret 更高

### 5.3 conservative rerank gate

不要再用：

```text
normalized_baseline + alpha * normalized_value
```

这种归一化会把小噪声放大。建议改成 floor-unit 的 conservative gate。

只允许替换 baseline action，当且仅当：

```text
candidate in baseline top-K
predicted_value_gap >= phase_threshold
predicted_value_gap >= c * ensemble_std
q10(candidate) >= q50(baseline) - safety_margin
death_risk(candidate) <= death_risk(baseline) + allowed_delta
candidate after_state is not OOD
phase/action kind is whitelisted
override rate under cap
```

初始阈值建议：

```text
K = 2 or 3
phase_threshold:
  combat: 1.5–2.0 floors
  card_reward/shop/boss_relic: 2.0–3.0 floors until validated
  map/event: shadow only initially
c = 2.0
override cap = 1–5% roots
```

combat 不应全量 rerank。局部 teacher 已经比较强，value 应该先作为“少数高置信纠错器”，例如：

- 是否为 elite/boss 保留强 potion
- 是否牺牲少量当前 HP 换更好长期状态
- 是否避免明显增加 death_next_3/6 的动作
- 是否在 lethal/接近结束时选择更安全的收尾

### 5.4 q_env dataset

当前 q_env dataset 由 V 对所有 legal after_states 打标签。下一版要加入过滤和真实 branch 校准。

#### 数据来源

1. **V-labeled broad dataset**  
   用 ensemble V 给 after_states 打分，但保留 uncertainty、OOD、value gap。只把低 uncertainty 的 pair 作为强监督，其余弱监督或不用。

2. **hard branch dataset**  
   从 shadow 中选：
   - baseline/value 分歧
   - predicted gap 大
   - uncertainty 高
   - high leverage phase
   - early/mid floors

   对 top candidates 做 continuation rollout，得到真实或近似 Q。

3. **chosen-action actual returns**  
   对 behavior chosen action，已有真实 final_floor，可用于 AWR/advantage learning。

#### 训练目标

不要只用 softmax(q_env / temperature) 做全量 CE。建议改成：

```text
pairwise ranking loss:
  only for pairs with |Q_i - Q_j| > min_gap
  weight by confidence

regression loss:
  predict Q only on branch-validated or chosen-action actual returns

abstention/confidence head:
  predict whether this root is safe to override
```

### 5.5 AWR / advantage-weighted BC

对已有 behavior trajectory，可用：

```text
A_t = return_t - V(s_t)
weight_t = clip(exp(A_t / beta), w_min, w_max)
```

训练 action model 时：

- 正 advantage 的动作权重大；
- 负 advantage 的动作权重小，但不完全删除；
- 加 KL/BC 正则，避免 policy 一步偏离太远；
- 按 phase 分开训练或至少 phase-conditioned。

这比“让 V 直接接管每个 root”安全，因为它仍然以真实行为轨迹为支撑，只是从高回报轨迹中学习更多。

对于 branch-validated roots，可用：

```text
A(s,a) = Q_branch(s,a) - V(s)
```

只在 `A` 明显为正且 uncertainty 低时作为 policy improvement 数据。

### 5.6 on-policy iteration

每轮 iteration：

1. baseline/current policy 采集 rollouts。
2. 训练 policy-conditioned V，输入中加入 `behavior_policy_id` 或 policy version。
3. shadow 评估所有 phase。
4. hard branch 验证 disagreement roots。
5. 训练 gated rerank/action policy。
6. 用未训练 seeds 做 paired rollout。
7. 只有通过 accept criteria 才进入下一轮数据采集。

如果混合多种 policy 数据，必须处理：

```text
V(s, policy_id) = 从 state s 由 policy_id 继续玩的 value
```

否则不同 policy 的 outcome 混在一起，会让同一 state class 对应多个 continuation value，进一步增加 label noise。

### 5.7 accept/reject criteria

#### 必须通过

- held-out seed 上 prediction metrics 不退化；
- floor bucket calibration 没有中段系统低估；
- hard branch high-confidence subset mean regret 非负；
- shadow branch error rate > 99.5%；
- paired rollout mean_floor 不低于 baseline；
- early death rate 不上升；
- accepted override rate 受控。

#### 推荐通过

- paired rollout mean_floor +0.3 以上；
- wins/300 不下降；
- potion/run 和 shop_spend/run 没有异常漂移；
- 被 value override 的 action 在 replay 中有正收益或至少不伤害。

---

## 6. 第一轮最小可行实验计划

目标不是一次性解决所有问题，而是在一周内回答四个问题：

1. 当前 6.8 MAE 中有多少是 target/weighting 问题？
2. distributional + calibration 是否改善 early/mid 和 uncertainty？
3. 显式 planning features 是否比继续加 MLP 有效？
4. gated value 是否能在 hard branch 和小规模 rollout 上不伤害？

### 6.1 数据规模

使用现有 5k baseline runs，但重建训练视图：

```text
train:
  seeds 1-4000 或按 run disjoint split

validation:
  seeds 4001-4500

test:
  seeds 4501-5000

runtime accept eval:
  使用全新 seeds，例如 5001-5300 或其他未进入 value/action 训练的数据
```

如果必须保留 `seed % 10 == 0`，也应额外建立一个 contiguous held-out seed range。不要用已经进入 value training 的 `seed1-300` 作为最终 accept 标准。

样本处理：

```text
- keep all non-forced high-level decisions
- combat rows thinning / per-room cap
- include state_before and chosen state_after
- forced_single either remove or very low weight
- add room-boundary transition table for TD(lambda)
```

新增 hard branch dataset：

```text
roots: 500–1500
selection:
  40% value-baseline disagreement
  30% high uncertainty
  20% high predicted gap
  10% random control

actions per root:
  baseline top action
  value top action
  top-K candidates if cheap

continuation:
  deterministic policy continuation first
  optional stochastic perturbation for robustness
```

### 6.2 对照组

至少训练以下 5 个模型：

| 模型 | 目的 |
|---|---|
| A. current best reproduction | 确认 pipeline 与旧指标一致 |
| B. residual MLP + new weighting | 单独验证 run/floor/phase weighting |
| C. distributional heads + new weighting | 验证 survival/quantile/calibration |
| D. C + wiki/path/readiness tabular features | 验证显式 planning features |
| E. D ensemble x5 | 提供 uncertainty 和 gate |

可选第 6 个：

| 模型 | 目的 |
|---|---|
| F. typed set encoder small | 验证结构化表示是否值得进入主线 |

### 6.3 训练配置

建议起始配置：

```text
input:
  2336 + v2 planning features

hidden:
  512

depth:
  4 residual blocks

dropout:
  0.05

optimizer:
  AdamW

scheduler:
  cosine or plateau

batch sampler:
  40% run-balanced
  30% floor-balanced
  20% phase-balanced
  10% runtime-frequency

loss:
  survival ordinal
  expected remaining SmoothL1
  scalar residual SmoothL1
  quantile pinball
  aux BCE
  TD(lambda) consistency on boundary states

epochs:
  early stopping by composite metric, not only MAE
```

Composite validation score：

```text
score =
  weighted_remaining_MAE
+ calibration_penalty
+ branch_regret_penalty
+ uncertainty_miscalibration_penalty
```

不要只用 validation loss 选 epoch。

### 6.4 成功标准

#### Prediction 成功

至少达到：

```text
row-level remaining MAE: <= 6.4
seed-balanced MAE: 相对 floor baseline 改善 >= 15%
floor 0-19 MAE: 至少下降 0.5–1.0
floor 20-34 calibration bias: 绝对值 < 1.0 floor
chosen-after-state MAE: 不显著差于 before-state MAE
```

#### Calibration 成功

```text
survival ECE < 0.08
win/act clear Brier score 相对 baseline 改善
q10/q90 coverage 接近标称
ensemble std 与 absolute error 正相关
```

#### Branch 成功

在 hard branch dataset 上：

```text
high-confidence accepted subset mean regret >= 0
predicted gap top decile actual gap > 0
OOD/uncertainty gate 能过滤掉高 regret roots
phase-specific 没有某一类明显灾难
```

#### Runtime 成功

先小规模：

```text
paired 60 seeds:
  mean_floor 不低于 baseline
  early death 不升高
  override rate <= 5%
```

再扩大：

```text
paired 300 held-out seeds:
  mean_floor >= baseline + 0.3 或至少不低于 baseline
  wins/300 不下降
  potion/run/shop_spend/run 无异常漂移
```

### 6.5 如果失败，如何定位是哪一层失败

#### 情况 A：MAE 没改善，calibration 也没改善

优先判断：

- feature 是否没进入有效路径；
- weighting 是否导致训练不稳定；
- survival loss threshold 是否失衡；
- seed split 是否揭示了原先过拟合；
- hash/entity 表示是否成为主瓶颈。

下一步：做 typed entity encoder 或增加 multi-rollout/branch labels，而不是继续调学习率。

#### 情况 B：MAE 改善，但 branch ranking 不改善

说明全局 V 不是局部 Q。下一步：

- 增加 hard branch Q dataset；
- 训练 action-conditioned Q；
- 使用 pairwise ranking loss；
- 只在 high-confidence / high-gap roots 使用 V；
- 不要指望继续降低 row-level MAE 自动改善 rerank。

#### 情况 C：branch validation 好，但 rollout 下降

说明 gate 或 policy integration 有问题。下一步：

- 降低 override cap；
- 提高 gap/uncertainty threshold；
- phase whitelist 收紧；
- 只允许 baseline top-2 内替换；
- 检查是否某类 action drift，例如 potion、shop、map；
- 检查归一化是否放大噪声，改用 floor-unit gating。

#### 情况 D：calibration 好，但 MAE 下降有限

这是可接受的。value model 可以先作为 safety/risk estimator，而不是全局 scorer。优先推进：

- death risk gate
- potion/resource preservation
- high-confidence branch override
- AWR 轨迹重加权

#### 情况 E：tabular planning features 无效

可能原因：

- feature 太粗或有 bug；
- 模型没有 group/phase conditioning；
- label noise 下 feature 效果被 row-level combat 淹没；
- 真正需要 typed set encoder。

下一步：做 feature sanity check 和 synthetic probes，例如预测 Nob readiness、boss readiness、next elite risk 等辅助任务，确认 feature 有信号。

---

## 7. 第一周执行顺序

### Day 1：诊断和数据视图重建

- 建立 run-balanced / floor-balanced / phase-specific validation dashboard。
- 报告 before-state vs chosen-after-state MAE。
- 报告 floor bucket calibration、survival proxy、win/act/death Brier。
- 统计每 run row 数、每 phase row 数，确认长 run/COMBAT 放大程度。
- 移除或 ablate seed hash，建立 held-out seed range。

产出：明确当前 6.78 的误差来源和 validation 可信度。

### Day 2：新 target + weighting baseline

- 训练 residual MLP + new sampler。
- 加 ordinal survival、expected floor、quantile、aux heads。
- 不加新 features，先验证 target/weighting 本身。

产出：确认 distributional/calibration 是否改善，不与 feature 改动混淆。

### Day 3-4：加入 v2 wiki/path/readiness features

优先实现：

1. path DP features
2. boss/elite readiness
3. deck frontload/block/aoe/scaling/cycle
4. potion/gold/hp resource pressure
5. choice gap-filling context

训练 D 模型并做 ablation。

产出：判断显式 planning features 是否降低 early/mid error 和中段低估。

### Day 5：ensemble + shadow

- 训练 5-seed ensemble。
- 对 held-out seeds 做 shadow。
- 记录 disagreement、uncertainty、value gap、OOD。
- 选 hard branch roots。

产出：知道 value 在哪里想改动作，以及这些改动是否高置信。

### Day 6：hard branch validation + gate tuning

- 对 500–1500 roots 做 branch continuation。
- 拟合 gate thresholds。
- 只选择 high-confidence accepted subset。
- 禁止全量 normalized rerank。

产出：得到可解释的 safe override candidate set。

### Day 7：小规模 paired rollout

- 先 60 held-out seeds。
- 若不伤害，再 300 held-out seeds。
- 对比 baseline、shadow、gated rerank。
- 报告 mean_floor、wins、death floor、potion/run、shop spend、override stats。

产出：决定是否进入下一轮 on-policy 数据采集。

---

## 8. 最终建议排序

### 最高优先级

1. 建立 seed/run/floor/phase-aware diagnostics。
2. 训练 distributional + quantile + residual scalar 的 calibrated V。
3. 改训练采样：run-balanced + floor-balanced + phase-aware。
4. 加入 chosen `state_after` 训练和验证。
5. 加 path DP、encounter readiness、resource pressure features。
6. 训练 ensemble，输出 uncertainty。
7. 做 hard branch validation，不再只相信 q_env self-label。
8. 用 conservative floor-unit gate 替代 normalized blend。

### 中优先级

9. action-conditioned Q / ranking model，只用低噪声 pair。
10. AWR / advantage-weighted BC，从真实高回报轨迹学习。
11. policy_id-conditioned value，支持多 policy 数据混合。
12. typed entity set encoder。

### 暂缓

13. 大规模复杂 Transformer value encoder。
14. 无闸门 rerank 所有 combat roots。
15. 只追求 validation loss 或 global row-level MAE。
16. 继续手写复杂 reward 系数。

---

## 9. 一句话版本

当前 `V_run` 的下一步不是“把 MLP 调到 MAE <3”，而是把它变成一个 **校准过的、带 uncertainty 的、结构化理解 run state 的长期风险/价值估计器**。  
在策略上，它应先作为 **高置信少量 override 和 AWR 数据筛选器**，而不是直接接管 action ranking。只有当 hard branch validation 证明它能在某些 phase/action kind 上稳定产生正 regret reduction，才应该扩大到 q_env policy improvement。
