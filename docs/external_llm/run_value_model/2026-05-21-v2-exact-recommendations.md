# Run Value Model V2：下一轮精确技术建议

> 适用约束：Ironclad / Ascension 0；目标仍是 `final_floor` / `remaining_floor`；不新增手写 reward；CPU rollout 昂贵、GPU 训练便宜；最终要支持安全的 after-state rerank / `q_env` policy improvement。

## 0. 先给结论

1. **survival / final-floor heads 现在应改为 auxiliary-only**。主 readout 继续用当前最有效的 scalar residual remaining-floor head；不要用 survival expectation 或 final-floor class expectation 作为 runtime value。
2. **下一版必须训练 chosen `state_after`**。每个 decision record 生成 `before` 和 chosen `after` 两个样本，但二者权重之和等于原 record 权重，避免把同一条轨迹重复计入。
3. **全局 row-level MAE `<3` 先不要作为当前优化目标**。应先跑 grouped-oracle / KNN-oracle 下界诊断；如果 full-state 近邻 oracle 在 floor 0-34 仍显著高于 3，则 `<3` 不是模型调参能解决的问题。
4. **先删除 seed hash；raw RNG hash 只做 ablation**。保留可解释的 RNG counters / pool state 作为候选，不让 production rerank 依赖 seed/dungeon/raw RNG hash。
5. **value gate 必须用 hard branch validation 拟合**。初始 runtime 只允许极少量高置信 non-combat override；COMBAT 先 shadow，不进 runtime override。
6. **如果要把 uncertainty 用进 gate，训练 5 个模型，不是 3 个**。3 个足够看 MAE 均值，5 个才适合做 pairwise delta lower-confidence gate。
7. **phase-specific head 不是下一步第一优先级**。先做 after-state training、phase/run balancing、calibration 和 hard-branch validation；若仍有 phase-specific bias，再加小型 phase residual head。

---

## 1. survival / final-floor heads：auxiliary-only，主 readout 不变

### 1.1 推荐 readout

继续使用当前 best 的形式：

```text
pred_remaining_scalar =
    floor_mean_remaining_train[current_floor]
  + residual_remaining_head(h)

runtime_value_score =
    floor(after_state)
  + clamp(pred_remaining_scalar(after_state), 0, 50 - floor(after_state))
```

**runtime rerank / q_env 只使用这个 scalar residual readout。**

不要把下面两个值作为主 value：

```text
sum_k sigmoid(survival_logit_k)
E[final_floor_class]
```

原因很直接：survival / final-floor 作为 primary readout 已经在本地实验中输给 scalar residual。它们仍有用，但主要作用是让 trunk 学到 act-clear、死亡风险、终局模式这些结构，而不是替代 regression readout。

### 1.2 精确 loss 配置

建议下一版命名为：

```text
value_v2_aftermix_residual_aux_noseedrng
```

loss：

```text
L =
  1.00 * SmoothL1(pred_remaining_scalar, target_remaining)
+ 0.05 * SmoothL1(pred_final_scalar,     target_final)
+ 0.05 * BCE(win, act1_clear, act2_clear, act3_clear)
+ 0.10 * BCE(death_next_3, death_next_6)
+ 0.05 * BCE(survival_bins)
+ 0.02 * CE(final_floor_bin)
```

推荐参数：

```text
SmoothL1 beta: 2.0
residual_floor_baseline: true
primary readout: remaining scalar only
survival expected-value loss: 0.00
final-floor expected-value loss: 0.00
```

survival bins 用 coarse ordinal thresholds，不要 51 个 dense bins 起步：

```text
survival_thresholds =
[8, 12, 16, 20, 24, 28, 32, 34, 38, 42, 46, 50]
```

final-floor bins 用 coarse terminal modes：

```text
final_floor_bins =
0-6, 7-12, 13-16, 17-23, 24-33, 34-40, 41-49, 50
```

解释：

- `remaining` 是唯一主目标。
- `final` 与 `remaining + floor` 高度冗余，所以权重从当前的 `0.25` 降到 `0.05`。
- death heads 对 safety gate 有用，所以 death 权重高于 act-clear。
- survival/final-bin 只帮助 representation 和 uncertainty diagnostics，不让它们主导 MAE。

### 1.3 calibration 修正

当前 floor 20-34 有明显 underprediction。下一版训练后应加一个 **post-hoc additive calibrator**，只在 calibration split 上拟合，不能用最终 validation。

```text
group = (floor_bucket_5, phase, before_or_after)

bias_g = mean(target_remaining - pred_remaining_raw) in group g

shrunken_bias_g =
    n_g / (n_g + 1000) * bias_g
  + 1000 / (n_g + 1000) * parent_bias_floor_bucket

pred_remaining_calibrated =
    pred_remaining_raw + shrunken_bias_g
```

runtime gate 和 hard branch validation 使用 calibrated value；模型训练本身仍用 raw scalar head。

---

## 2. `state_before` / chosen `state_after` 如何混合

### 2.1 是否共享 terminal target？

**共享同一个 terminal `final_floor`，但 `remaining_floor` 要按各自 state 的当前 floor 重算。**

对每条 trajectory record：

```text
terminal_final = run.final_floor

before sample:
  x = encode(state_before)
  y_final = terminal_final
  y_remaining = terminal_final - floor(state_before)

after sample:
  x = encode(chosen_state_after)
  y_final = terminal_final
  y_remaining = terminal_final - floor(chosen_state_after)
```

注意：

- 如果 action 只是在同一场 combat 内推进一步，`floor(before) == floor(after)`，remaining target 一样，这是正常的。
- 如果 action 进入下一层 / reward / terminal，`floor(after)` 可能变化，remaining target 必须随 after-state 重算。
- `truncated` 和 `error` run 不应用作 scalar regression；建议 `sample_weight = 0` 或直接排除。当前只有 32 truncated / 1 error，删除不会损失多少数据。

### 2.2 before / after 的权重

不要把数据量简单翻倍。每条原始 decision record 的总权重保持不变：

```text
w_before = 0.40 * w_record
w_after  = 0.60 * w_record
```

理由：

- runtime rerank 主要打分 after-states，所以 after 权重应更高。
- 仍保留 before，因为 shadow diagnostics、root value calibration、action policy input 仍需要 before distribution。
- 总权重不变，避免同一条 trajectory label 被重复计入。

### 2.3 避免长 run 被重复放大

推荐用 hierarchical sampler；如果现有 trainer 更方便使用 row weights，则按下面的等价权重实现。

#### 先做 record-level caps

```text
COMBAT:
  group_key = (seed, floor, combat_room_index)
  multiplier = min(1.0, 8 / n_combat_rows_in_same_room)

CARD_REWARD:
  group_key = (seed, floor, reward_screen_id)
  multiplier = min(1.0, 4 / n_rows_in_same_reward_screen)

forced_single / pure bookkeeping rows:
  multiplier = 0.10

其他 phase:
  multiplier = 1.00
```

如果没有 `combat_room_index`，用：

```text
(seed, floor, room_type, source, monotonically_increasing_room_counter)
```

或者退化为：

```text
(seed, floor, phase, source)
```

COMBAT 当前占比过高，而且同一场 combat 内大量 microstate 共用同一个 terminal label。cap 到每场 combat 约 8 个有效样本，是比简单 phase reweight 更稳的第一步。

#### 再做 phase target share

建议 target effective phase share：

```text
COMBAT       0.35
CARD_REWARD 0.18
MAP         0.12
CARD_SELECT 0.10
SHOP        0.08
EVENT       0.06
CAMPFIRE    0.04
BOSS_RELIC  0.03
TREASURE    0.02
NEOW        0.02
```

计算：

```text
phase_multiplier =
    target_phase_share[phase] / observed_effective_phase_share[phase]

phase_multiplier clipped to [0.25, 4.0]
```

#### 再做 mild floor balancing

用 5-floor buckets：

```text
00-04, 05-09, ..., 45-49, 50-54
```

```text
floor_multiplier =
    sqrt(uniform_bucket_share / observed_bucket_share)

floor_multiplier clipped to [0.50, 2.00]
```

不要 aggressive floor balancing；early label 噪声大，强行上权重会让模型追噪声。

#### 最后做 per-run normalization

```text
w_record_raw =
    cap_multiplier
  * phase_multiplier
  * floor_multiplier

for each run:
  w_record = w_record_raw / sum_raw_weights_in_run
```

然后：

```text
w_before = 0.40 * w_record
w_after  = 0.60 * w_record
```

这样每个 seed/run 的总权重相同，长 run 不再因为 decision rows 更多而主导训练。

---

## 3. 判断 row-level MAE `<3` 是否不现实：下界诊断

不要先争论 `<3` 是否可能；先用 existing 5k data 跑 oracle diagnostics。目标是回答：

```text
如果只给当前 encoder 能表达的信息，单 rollout terminal label 的条件噪声下界是多少？
```

### 3.1 Grouped median oracle

对每组 key，在 train split 上拟合 group median，在 validation split 上评估 MAE。

```python
oracle_pred[g] = median(target_remaining_train where group_key == g)
oracle_mae = mean(abs(target_remaining_val - oracle_pred[group_key_val]))
```

没有足够样本的 group 回退到 parent group。

每个 group 至少：

```text
min_train_count = 30
```

同时报告：

```text
row-weighted MAE
seed-balanced MAE
floor-bucket MAE
phase MAE
group internal MAD = mean(|y - median_g|)
p90 - p10 target spread
```

### 3.2 推荐 grouping keys

按从粗到细跑：

#### G0：当前 baseline 级别

```text
(floor_bucket_5)
```

#### G1：floor + phase

```text
(floor_bucket_5, phase)
```

#### G2：基础可观察局面

```text
(
  exact_floor,
  phase,
  room_type,
  act_boss_id,
  hp_ratio_bin_0.1,
  gold_bin_50,
  deck_size_bin_5,
  relic_count_bin_3,
  potion_count,
  current_act
)
```

#### G3：去 seed/RNG 的高信息 state signature

```text
(
  exact_floor,
  phase,
  room_type,
  act_boss_id,
  hp_ratio_bin_0.1,
  gold_bin_50,
  deck_multiset_hash,       # raw card ids/counts, not hash-bucket vector
  relic_set_hash,           # raw relic ids
  potion_inventory_hash,    # raw potion ids/slots
  current_node_y,
  current_node_x,
  next_3_map_symbols_hash,
  reachable_map_prefix_hash
)
```

#### G4：当前 encoder 的 quantized full-state signature，排除 seed/RNG hash

```text
quantized_encode_state_without_seed_rng_hashes

numeric quantization:
  hp_ratio: 0.05
  gold: 25
  incoming_damage: 5
  monster_hp_sum: 10
  deck/readiness continuous features: 0.05 or 0.10
hash bags:
  use bucket vector after clipping/count quantization
```

#### G5：G4 + structured RNG counters，但仍不加 seed hash

```text
G4
+ rng stream call_counts
+ rng trace event counts
+ future pool sizes / ids
```

#### G6：G5 + raw RNG hash / seed hash，仅作 leak diagnostic

```text
G5
+ raw rng_state hashes
+ seed hash
+ dungeon hash
```

### 3.3 解释规则

建议按 floor bucket 解释，不只看 global：

```text
如果 G4/G5 oracle 在 floor 0-34 的 MAE 仍 > 4.0-4.5：
  用当前 single-rollout terminal labels 把 global row-level MAE 压到 <3 基本不现实。
  重点应转向 calibration、after-state ranking、hard branch safety，而不是追全局 MAE。

如果 G4/G5 oracle 已经 <3，但 MLP 仍 6.8：
  问题主要是模型/训练/feature interaction，不是 target noise。
  这时再考虑 set encoder、larger MLP、phase heads。

如果 G6 明显优于 G5，但只在 seed%10 split 上有效：
  seed/RNG hash 在制造 pseudo-generalization，不应用于 production rerank。

如果 grouped oracle 在 row-weighted 下好、seed-balanced 下差：
  当前训练被长 run 或胜局 trajectory 放大，需要 run-balanced training。
```

### 3.4 KNN oracle

再跑一个非参数近邻 oracle：

```text
feature = current 2336-dim encoded feature
exclude = seed hash, dungeon hash, raw rng hash
normalize = train mean/std
same_phase = required
same_floor_bucket = required

k = 25 and 50
prediction = median(target_remaining of k nearest train rows)
validation = seed-heldout only
```

解释：

```text
KNN oracle floor 0-34 MAE > 4.5:
  当前 feature space 本身无法支持 <3。

KNN oracle 接近 3，但 neural model 6.8:
  训练 objective / weighting / architecture 有明显问题。

KNN oracle with seed/RNG hash 明显变好，但 heldout block split 不变好:
  这是 seed/RNG pseudo-generalization。
```

### 3.5 “完美 late floor” sanity check

计算：

```text
global_lower_bound_with_perfect_late =
  actual/oracle error on floor 0-34
  + 0 error on floor >=35
```

如果这个值仍 >3，说明当前 global `<3` 主要被 early/mid floor label variance 卡住，late floor 已不是瓶颈。

---

## 4. 当前最值得实现的 12 个显式 features

原则：这些是 **state descriptors**，不是 reward。它们帮助 `V_run` 理解 deck/path/resource context；target 仍然只来自 terminal outcome。

### P0：先做这 6 个

| 优先级 | feature | 精确定义 | 为什么现在边际价值高 |
|---:|---|---|---|
| 1 | `path_first_elite_context` | 对每条 reachable path，找 first elite；聚合 `min/mean/max floors_to_first_elite`、`combats_before_first_elite`、`rest_before_first_elite`、`shop_before_first_elite` | 当前 map bag 有未来节点计数，但缺少“节点顺序”。是否先精英再火堆/商店是 Act 1 生死变量。 |
| 2 | `path_elites_before_next_rest` | reachable paths until first rest/boss：`min/mean/max elite_count_before_rest`、`forced_elite_before_rest` | HP / potion / deck readiness 必须和路径压力结合。 |
| 3 | `path_to_boss_resource_windows` | 到 boss 前 reachable path 的 `rest_count`、`shop_count`、`forced_elite_count`、`monster_count` 的 min/mean/max | 修正 floor 20-34 underprediction，尤其 Act 2/3 中段是否还有恢复和商店窗口。 |
| 4 | `gold_shop_timing` | `floors_to_next_reachable_shop`、`shops_before_boss`、`gold_minus_purge_cost`、`can_afford_purge_now`、`gold_per_shop_window = gold / max(1, shops_before_boss)` | 当前有 gold 和 shop counts，但缺少“钱是否赶得上下个 shop / purge”。 |
| 5 | `hp_path_pressure` | `hp_after_burning_blood`、`hp_per_forced_combat_before_rest`、`hp_per_elite_before_rest`、`hp_ratio_at_next_rest_window` | 不写伤害 reward，只把 HP 与路径压力组合起来。 |
| 6 | `potion_inventory_tactical_tags` | potion slots + potion tag counts：`burst_damage`、`block/dex`、`strength/scaling`、`draw/energy`、`weak/vulnerable`、`emergency/heal`；再加 `has_strong_potion_before_first_elite/boss` | 当前 potion bag 是 hash；value 需要知道“这瓶药是否能支撑精英/老板”。 |

### P1：随后做这 6 个

| 优先级 | feature | 精确定义 | 为什么现在边际价值高 |
|---:|---|---|---|
| 7 | `act1_elite_readiness_vector` | 三个数：`nob_frontload_readiness`、`lagavulin_scaling_readiness`、`sentries_aoe_status_readiness`；由 card metadata 计算，不作为 reward | Act 1 early labels 方差最大；elite compatibility 是终局分布的重要分叉。 |
| 8 | `current_boss_readiness` | 按当前 boss ID 选择对应 readiness：Slime split burst、Hexaghost burn/long-fight scaling、Guardian block consistency；Act 2/3 boss 也给 boss-specific vector | 当前 boss hash 太弱；boss-specific mismatch 会导致 mid-run underprediction。 |
| 9 | `deck_frontload_turn1_3` | 用 deck card tags 和 cost 估计前 3 回合 expected attack damage、block、draw，不做 combat sim | 当前 average base damage/block 太粗；frontload 对 Nob/Slime/Act 2 hallway 很关键。 |
| 10 | `deck_cycle_speed` | `effective_deck_size / expected_draw_per_turn`、`turns_to_first_reshuffle`、`draw_cards_per_cycle`、`status/curse_density` | 长程 survival 依赖 deck consistency，不只是 deck size 和 draw-like count。 |
| 11 | `scaling_vs_frontload_balance` | `strength_scaling_sources`、`block_scaling_sources`、`power_count_playable`、`exhaust_synergy_count`、`single_target_scaling_density` | 区分 Act 1 frontload 强但 Act 2/3 不够 scaling 的 deck。 |
| 12 | `upgrade_remove_pressure` | `high_impact_unupgraded_count`、`bash_unupgraded`、`starter_basic_count`、`curse_count`、`removal_pressure = starter_basic + curse_count`、再与 `next_campfire/shop_distance` 拼接 | 决定 campfire/shop/card_select 的长期价值；当前 readiness 只有 upgrade ratio 和 starter count。 |

实现建议：

```text
第一版不要做 learned set encoder 替代全部 hash bag。
先把上面 12 个 feature 拼进 numeric summary，保持 MLP / residual setup 不变。
```

这样能直接判断：问题是缺少这类显式结构，还是 target noise / weighting 更关键。

---

## 5. seed hash / RNG hash：删除、保留还是 ablate？

### 5.1 推荐 production 默认

```text
remove:
  hash(seed)
  hash(dungeon_id)
  raw rng_state hash(seed0, seed1)
  raw stream hash buckets that directly encode RNG internal state

retain candidate:
  rng stream call_counts
  rng trace event counts
  future card/relic/boss relic pool contents
  future pool sizes
  map state / reachable map
```

理由：

- seed hash 和 raw RNG hash 很可能让模型在固定 seed 范围内记忆 outcome texture。
- 这对 row MAE 可能有帮助，但对 off-policy after-state rerank 很危险。
- future pools / map 是真实可观察或状态内信息，保留更合理。

### 5.2 必跑 ablation

训练四个版本，其他配置完全相同：

```text
A. no_seed_no_rnghash
   remove seed/dungeon/raw rng hashes
   keep map, future pools, non-random state features

B. no_seed_structured_rng
   A + rng call_counts + rng trace event counts + tail numeric summaries

C. no_seed_full_rnghash
   remove seed/dungeon hash
   keep raw rng_state hashes

D. current_full
   current seed + dungeon + raw rng hashes
```

### 5.3 验证切分

不要只看 `seed % 10 == 0`。

至少使用：

```text
split_legacy_mod10:
  valid if seed % 10 == 0

split_hash_seed:
  valid if xxhash64(seed) % 10 == 0

split_block:
  train seeds 1-4000
  valid seeds 4001-5000

split_interleaved_blocks:
  5 folds by contiguous seed ranges, e.g. 1-1000, 1001-2000, ...
```

再加一个专门的 hard branch split：

```text
hard_branch_valid:
  seeds not used in training
  branch after-states generated in shadow
  candidate states include off-policy alternatives
```

### 5.4 判断规则

```text
如果 D 只在 split_legacy_mod10 上更好，而在 block/hash/hard_branch 不好：
  seed/RNG hash 是 pseudo-generalization，删掉。

如果 B 在所有 split 上稳定提升 >=0.10-0.20 MAE，且 hard branch sign calibration 变好：
  保留 structured RNG counters。

如果 C 提升 row MAE，但 hard branch validation 更差：
  raw RNG hash 不进 production value gate。
```

production rerank 默认使用 A 或 B，不使用 D。

---

## 6. 保守 value gate：从 hard branch validation 拟合

### 6.1 先构造 hard branch validation

从 heldout seeds 采样 root，不参与 value training。每个 root 至少有两个合法 action，且 branch 成功。

建议第一批规模：

```text
total root pairs: 1,500-2,500

stratify:
  CARD_REWARD  400-600
  MAP          300-500
  SHOP         200-300
  CARD_SELECT  200-300
  CAMPFIRE     100-200
  BOSS_RELIC   all available or 100+
  EVENT        100-200 shadow only
  COMBAT       300-500 shadow only
```

对每个 root：

```text
b = baseline chosen/top action
c = value model's best non-baseline candidate after gate prefilter

rollout from after_state(b) to terminal under baseline policy
rollout from after_state(c) to terminal under baseline policy

true_delta = final_floor(c) - final_floor(b)
```

如果 CPU 不够，先每个 phase 100-200 root pairs；但 phase 少于 50 accepted examples 前，不要开启 runtime override。

### 6.2 每个 candidate 计算的 gate variables

对 ensemble 中每个模型 `m`：

```text
q_m(a) = floor(after_state_a) + calibrated_remaining_m(after_state_a)
delta_m = q_m(c) - q_m(b)
```

聚合：

```text
delta_mean = mean_m(delta_m)
delta_std  = std_m(delta_m)
delta_q20  = percentile_20(delta_m)   # 5-model ensemble
delta_min  = min_m(delta_m)

state_std_c = std_m(q_m(c))
state_std_b = std_m(q_m(b))
max_state_std = max(state_std_c, state_std_b)

baseline_deficit =
    normalized_baseline_score(b) - normalized_baseline_score(c)

death3_delta = p_death_next_3(c) - p_death_next_3(b)
death6_delta = p_death_next_6(c) - p_death_next_6(b)

act_clear_delta =
    p_relevant_act_clear(c) - p_relevant_act_clear(b)

ood_percentile =
    same_floor_phase_knn_distance_percentile(after_state_c)
```

### 6.3 初始 runtime gate

初始只允许：

```text
phase_allow_initial:
  CARD_REWARD
  MAP
  SHOP
  CARD_SELECT
  CAMPFIRE
  BOSS_RELIC

phase_shadow_only_initial:
  COMBAT
  EVENT
  NEOW
  TREASURE
```

COMBAT 先不要 override。当前 combat selector 已经相对强，而 value model 的局部 action 差距通常远小于全局 MAE；pilot 已证明无闸门/弱闸门会伤害表现。

初始 phase threshold：

```text
T_abs:
  CARD_REWARD  2.0
  BOSS_RELIC   2.0
  MAP          2.5
  SHOP         2.5
  CARD_SELECT  2.5
  CAMPFIRE     2.5
  EVENT        3.0   # shadow only until validated
  COMBAT       4.0   # shadow only; later only room-ending / potion decisions
```

accept condition：

```text
candidate != baseline

candidate_baseline_rank <= 3

delta_mean >= T_abs[phase]

delta_q20 >= 1.0
# with 5 models

delta_std <= 1.0

max_state_std <= 2.0

at least 4 of 5 models have delta_m > 0

baseline_deficit <= 0.50
# using current center/max_abs normalized baseline score

death3_delta <= 0.02

death6_delta <= 0.03

act_clear_delta >= -0.02

ood_percentile <= 95

per-run cap:
  max 1 override per floor
  max 3 overrides per run

global coverage cap:
  override <= 2% of eligible roots in first runtime test
```

如果只有 3-model ensemble：

```text
replace:
  delta_q20 >= 1.0
  at least 4/5 positive

with:
  delta_min >= 1.5
  all 3 models have delta_m > 0
  delta_std <= 0.75
```

### 6.4 如何从 hard branch validation 拟合 gate

在 hard branch validation 上 grid search：

```text
T_abs in {1.5, 2.0, 2.5, 3.0, 4.0}
delta_q20_min in {0.5, 1.0, 1.5, 2.0}
delta_std_max in {0.75, 1.0, 1.25}
baseline_deficit_max in {0.25, 0.50, 0.75}
ood_percentile_max in {90, 95, 98}
```

每个 phase 单独选 threshold；没有足够 accepted examples 的 phase 保持 shadow。

phase enable 条件：

```text
n_accepted >= 50

mean(true_delta | accepted) >= +0.30 floor

lower_95_CI(mean true_delta | accepted) >= 0.00

P(true_delta < -1.0 | accepted) <= 0.20

P(true_delta < -3.0 | accepted) <= 0.05
```

runtime A/B accept 条件：

```text
300-seed smoke:
  rerank mean_floor >= baseline_mean_floor - 0.30
  death_count does not increase by >2%
  no phase has catastrophic accepted true_delta pattern in shadow logs

1000-seed confirm:
  rerank mean_floor >= baseline_mean_floor
  or mean_floor lower_95_CI is not worse by more than 0.20
```

如果 300-seed 低于 baseline 超过 0.3 floor，直接 reject，不要继续扩大 rollout。

---

## 7. Ensemble：3 个还是 5 个？

### 7.1 推荐

```text
MAE / ablation development:
  3 models enough

runtime gate / uncertainty:
  5 models recommended
```

GPU 训练便宜，CPU rollout 昂贵；为了少做坏 rollout，5-model ensemble 的成本值得付。

### 7.2 为什么 5 个 materially better

3 个模型的问题：

```text
q20 几乎退化成 min
std 非常不稳定
一个 bad seed 会让 gate 过度保守或过度乐观
```

5 个模型可以更稳定地估计：

```text
delta_q20
model agreement count
pairwise delta_std
state-level epistemic uncertainty
```

### 7.3 ensemble 如何训练

每个模型：

```text
same architecture
different initialization seed
different run-level bootstrap weights
same train/validation split
same post-hoc calibration protocol
```

bootstrap 建议：

```text
sample runs with replacement
or assign each run weight ~ Gamma(shape=1, scale=1)
then apply phase/floor/record caps inside run
```

不要只做 row-level bootstrap；同一 run 内 rows 高度相关，row bootstrap 会低估 uncertainty。

### 7.4 uncertainty 如何进 gate

只使用 **pairwise action delta uncertainty**，不要只看 absolute value uncertainty。

```text
delta_m = q_m(candidate) - q_m(baseline)

gate uses:
  delta_mean
  delta_std
  delta_q20
  count(delta_m > 0)
```

同时用 absolute state std 做 OOD 防线：

```text
max(std_m(q_m(candidate)), std_m(q_m(baseline))) <= 2.0
```

推荐初始条件：

```text
5-model:
  delta_q20 >= 1.0
  delta_std <= 1.0
  positive_models >= 4/5
  max_state_std <= 2.0

3-model:
  delta_min >= 1.5
  delta_std <= 0.75
  positive_models = 3/3
  max_state_std <= 1.5
```

---

## 8. phase-specific head：现在不优先

### 8.1 推荐顺序

先做：

```text
1. before/after mixed training
2. run-balanced + phase-capped weighting
3. seed/RNG ablation
4. post-hoc floor/phase calibration
5. hard branch validation + conservative gate
```

再考虑 phase-specific head。

原因：

- 当前最大 mismatch 是训练只见 `state_before`，runtime 用 `after_state`。
- phase distribution 极不平衡，直接加 phase heads 容易让 rare phase overfit。
- 当前 late_fusion / classification heads 没有证明比 scalar residual 更强，说明先解决 data/target/weighting 更重要。

### 8.2 如果要试 phase-specific，只试小 residual head

不要先上 full phase-specific MLP。建议：

```text
h = shared_trunk(x)

pred_remaining =
    floor_baseline[floor]
  + global_residual_head(h)
  + phase_residual_head[phase](h)
```

限制：

```text
phase_residual_head:
  Linear(hidden_dim -> 1)
  no deep MLP initially

regularization:
  0.01 * mean(phase_residual^2)

rare phases:
  if train rows < 20k, share OTHER head
```

accept condition：

```text
phase-balanced after-state MAE improves >= 0.25

global seed-heldout MAE does not worsen by >0.05

floor 20-34 calibration bias improves by >=0.5 floor

hard branch accepted true_delta metrics do not worsen
```

否则不启用。

---

## 9. 下一版最小可行实验配置

### 9.1 Primary V2 training run

```yaml
name: value_v2_aftermix_residual_aux_noseed_structrng

data:
  runs: current 5k
  include_state_before: true
  include_chosen_state_after: true
  before_weight: 0.40
  after_weight: 0.60
  exclude_truncated_error_scalar: true

target:
  final_floor: terminal.final_floor
  remaining_floor: terminal.final_floor - current_state.floor

features:
  remove_seed_hash: true
  remove_dungeon_hash: true
  remove_raw_rng_state_hash: true
  keep_structured_rng_counters: ablation-dependent
  add_top12_explicit_features: optional second stage, not mixed into first ablation

architecture:
  type: mlp
  hidden_dim: 384
  depth: 3
  dropout: 0.05
  residual_floor_baseline: true

loss:
  remaining_smooth_l1: 1.00
  final_smooth_l1: 0.05
  act_win_bce: 0.05
  death_bce: 0.10
  survival_bce: 0.05
  final_bin_ce: 0.02
  survival_expected_value_loss: 0.00
  final_class_expected_value_loss: 0.00

weighting:
  run_balanced: true
  combat_room_cap_effective_rows: 8
  card_reward_screen_cap_effective_rows: 4
  forced_single_multiplier: 0.10
  phase_target_share:
    COMBAT: 0.35
    CARD_REWARD: 0.18
    MAP: 0.12
    CARD_SELECT: 0.10
    SHOP: 0.08
    EVENT: 0.06
    CAMPFIRE: 0.04
    BOSS_RELIC: 0.03
    TREASURE: 0.02
    NEOW: 0.02
  floor_balance: sqrt_uniform_bucket_clipped_0.5_2.0

calibration:
  additive_bias_by: [floor_bucket_5, phase, before_or_after]
  shrinkage_n: 1000
```

### 9.2 必跑 ablation table

按顺序跑，避免同时改太多：

| run | before/after | weights | aux heads | seed/RNG | 目的 |
|---|---:|---:|---:|---|---|
| A0 | before only | current | current | current | reproduce current best |
| A1 | before+after 40/60 | current | current | current | 单独测 training/runtime distribution mismatch |
| A2 | before+after 40/60 | new run/phase caps | current | current | 单独测 weighting |
| A3 | before+after 40/60 | new run/phase caps | new aux-only loss | no seed / structured RNG | primary V2 |
| A4 | same as A3 | same | same | no seed / no RNG hash | 测 structured RNG 是否有用 |
| A5 | same as A3 | same | same | full current seed/RNG | leak diagnostic |
| A6 | same as A3 + top12 features | same | same | best non-leaky RNG setting | 测 explicit features 的边际收益 |

### 9.3 成功标准

不要只看 row-level MAE。下一版成功标准：

```text
global row-level remaining MAE:
  <= 6.50 preferred
  <= 6.70 acceptable if calibration/ranking improves

seed-balanced MAE:
  improve >= 0.30 vs current best

after-state validation MAE:
  not worse than before-state MAE by >0.10

floor 20-34 calibration:
  absolute bias <= 1.0 floor
  current bias is about 2.1-2.5 floors, so this is high priority

floor 0-14:
  MAE improvement not required if oracle lower-bound says high noise;
  calibration and uncertainty should improve.

hard branch validation:
  accepted candidate mean true_delta >= +0.30
  lower_95_CI >= 0
  P(true_delta < -3) <= 0.05

runtime 300-seed gated smoke:
  mean_floor >= baseline - 0.30
  no death_count increase >2%
```

如果 V2 row MAE 只小幅改善，但 hard branch accepted true_delta 明显为正，也应该继续；policy improvement 需要的是 **局部可信 action delta**，不是全局 row MAE 最小。

---

## 10. 最重要的 reject 条件

下一轮出现以下任一情况，应停止把模型接进 runtime，回到诊断：

```text
1. after-state MAE 明显差于 before-state MAE：
   说明 encoder/labels 对 after distribution 仍不稳。

2. hard branch accepted true_delta 均值 <= 0：
   即使 row MAE 变好，也不能 rerank。

3. seed/RNG full model 只在 legacy split 变好：
   删除 seed/RNG hash，不要让 rerank 依赖它。

4. floor 20-34 underprediction 仍 >2 floors：
   先修 calibration/weighting，不要扩大 override coverage。

5. runtime override coverage >2% 但 300-seed mean_floor 下降：
   gate 太松；回到 hard branch threshold。
```

---

## 11. 对 8 个问题的直接回答

1. **survival/final-floor heads 是否 auxiliary-only？**  
   是。主 readout 用 scalar residual remaining。loss 权重用：remaining `1.00`、final scalar `0.05`、act/win BCE `0.05`、death BCE `0.10`、survival BCE `0.05`、final-bin CE `0.02`。survival/final expected-value loss 设为 `0.00`。

2. **before / after 如何混合？**  
   用 chosen `state_after`，与 before 共享 terminal `final_floor`，但 `remaining = final_floor - floor(state)` 分别重算。每条 record 总权重不变，`before:after = 0.40:0.60`。加 per-run normalization、combat room cap、phase target share，避免长 run 和 combat microstates 主导。

3. **MAE `<3` 下界诊断？**  
   跑 grouped median oracle：G0 floor bucket、G1 floor+phase、G2 基础局面、G3 去 seed/RNG 的 raw deck/relic/potion/map signature、G4 quantized full encoder no seed/RNG、G5 structured RNG、G6 raw RNG/seed leak diagnostic。再跑 same-phase/floor KNN oracle。若 G4/G5 在 floor 0-34 仍 >4.0-4.5，single-rollout global `<3` 基本不是当前可达目标。

4. **最高优先级 features？**  
   先做 12 个：`path_first_elite_context`、`path_elites_before_next_rest`、`path_to_boss_resource_windows`、`gold_shop_timing`、`hp_path_pressure`、`potion_inventory_tactical_tags`、`act1_elite_readiness_vector`、`current_boss_readiness`、`deck_frontload_turn1_3`、`deck_cycle_speed`、`scaling_vs_frontload_balance`、`upgrade_remove_pressure`。

5. **seed/RNG hash？**  
   production 删除 seed/dungeon/raw RNG hash。structured RNG counters 可以 ablate。验证必须包括 legacy mod10、hash(seed) mod10、contiguous block split、hard branch heldout。只在 legacy split 变好的 RNG/seed feature 视为 pseudo-generalization。

6. **conservative value gate？**  
   用 hard branch validation 拟合。初始 gate：non-combat phase allowlist，COMBAT shadow-only；`delta_mean >= 2.0-2.5` non-combat、`delta_q20 >= 1.0`、`delta_std <=1.0`、`4/5 models positive`、`baseline_deficit <=0.5`、`death6_delta <=0.03`、`ood <=95 percentile`、coverage <=2%、每层最多 1 次 override。phase 只有在 accepted true_delta lower CI >=0 后启用。

7. **ensemble 3 还是 5？**  
   3 个足够做 MAE ablation；runtime uncertainty gate 用 5 个。gate 使用 pairwise `delta_m = q_m(candidate)-q_m(baseline)` 的 `mean/std/q20/positive_count`，不要只看 absolute value std。

8. **phase-specific head 现在做吗？**  
   不作为第一步。先做 after-state + weighting + calibration + hard branch。若 phase-balanced after-state MAE 和 hard branch sign 仍显示 phase bias，再加小型 phase residual head，不上完整 phase-specific expert。
