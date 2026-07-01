# 给 ChatGPT 5.5 的项目上下文：Run Value Model / Environment-Guided Policy Improvement 下一步设计

这份文档是给外部大语言模型看的完整背景。外部模型不能读取本地代码，所以这里会尽可能把项目目标、当前决策栈、历史实验、当前 value model 的实现、特征、指标和核心困境都写清楚。

当前主要想请你回答的问题不是“再给 combat Transformer 加什么小模块”，而是：

```text
如何把当前的 Run Value Model 做到真正可用？
如何尽可能压缩 final/remaining floor 的 MAE？
如何让它能安全地用于 after_state rerank / q_env policy improvement？
在不继续手工堆奖励函数的前提下，value model 应该学什么、用什么目标、什么结构、什么特征？
```

---

## 0. 希望你重点回答的问题

请重点围绕下面问题给出具体方案。

1. 当前 `V_run(s)` 的最佳验证 `remaining_floor MAE` 约 `6.78`，floor-mean baseline 约 `7.82`。如果目标是尽量接近 `<3`，最可能的瓶颈是什么？哪些瓶颈是模型/特征可解决的，哪些是目标定义本身带来的不可约噪声？
2. 当前 label 是单条 rollout 的最终 floor，也就是 `final_floor - current_floor`。这是否适合训练早期楼层状态？是否应该改成 distributional / ordinal / survival / quantile / TD(lambda) / multi-rollout expectation？
3. 当前 2336 维 state feature 覆盖面很广，但大部分是摘要和 hash bag。对于 Slay the Spire 这种长程规划任务，哪些部分应该从 hash bag 改成显式结构建模？
4. 如果用 wiki / 游戏知识生成 features，哪些 feature 应该优先实现？它们应该给 `V_run` 学到什么，而不是变成新的手工 reward？
5. 当前 value rerank 初版会明显降低 rollout 表现。应该先修 value model 的校准，还是先改 rerank/gating/q_env policy 的使用方式？
6. 如何设计 next iteration，使 `V_run` 可以安全接入策略：只 rerank combat？扩展到 card reward/map/shop？用 AWR/advantage-weighted BC？还是先只做 shadow/offline hard validation？
7. 请给出一套可落地的优先级方案：第一周做什么、需要哪些诊断、成功/失败标准是什么。

请避免泛泛建议“加数据、加层、调学习率”。如果建议加数据，请说明数据应该怎么采样、怎么降噪、怎么划分验证、怎么避免长 run 过度占权重。如果建议换模型结构，请说明输入组织、loss、输出头、校准指标和接入 policy 的方式。

---

## 1. 项目目标

项目是一个 Slay the Spire 1 的深度学习/策略项目，目前主要使用 Ironclad，Ascension 0，目标是让模型在固定 seed 范围内打出更高 mean floor，长期目标是接近或超过 `40+ mean floor`。

当前常用评估协议：

```text
seed1-300 full rollout
指标:
  mean_floor
  wins/300
  potion/run
  shop_spend/run
  room/action statistics
```

Slay the Spire 的难点不是单纯战斗出牌，而是整局资源规划：

```text
战斗中是否该用药水
是否为未来精英/老板保留 HP 和药水
卡牌奖励取什么、是否跳过
商店是否消费、是否留钱到下一个商店
删牌、升级、回血的取舍
路线选择：精英、火堆、商店、问号、宝箱
事件选择
boss relic 选择
```

过去主要依赖 combat teacher 和 imitation 训练，近期开始转向 `Environment-Guided Policy Improvement`，核心是训练一个整局局面价值模型 `V_run(s)`。

---

## 2. 当前 runtime 决策栈概览

当前 runtime 不是单一模型控制所有动作，而是多个 selector 组合。

主要模块：

```text
combat:
  v3 candidate combat scorer / Transformer / MLP

card reward:
  card_reward model，给卡牌奖励候选打分

shop:
  当前主要用 value-style shop policy:
    item_value - spend_cost
    保留 future_shop_reserve / purge reserve
    card 用 card_reward score 校准
    relic/potion 使用 shop prior delta

map:
  map_dp / path policy

event:
  rule / learned / filtered policy 混合

campfire:
  smith/rest/other rule

boss relic:
  boss relic model / rule

card select:
  purge/upgrade/transform/target selection 等
```

当前 value 数据采集时的 baseline policy 配置大致是：

```text
combat_selector = v3-candidate
v3_combat_model = v5_dual_semantic_legacy_gate
card_reward_model = card_reward.pt
shop_policy = value
shop_choice_model = shop_choice_prior_delta
normal_room_potion_penalty = 1.5

shop value 参数:
  price_cost = 0.044348003822393976
  reserve_shortfall_cost = 0.043490245962190935
  future_shop_reserve = 120
  future_shop_horizon = 5
  card_scale = 4.6262945279949435
  card_reference_price = 60.0
  card_price_factor_min = 0.65
  card_price_factor_max = 1.35
  potion_scale = 0.5084989138155764
  relic_scale = 0.8
  item_scale = 1.0
  threshold = 0.0
  shop_prior_weight_override = 0.8
```

这些参数不是本问题的重点，但它们决定了当前 value dataset 的行为策略。

---

## 3. 当前 combat teacher 和模型历史

### 3.1 Teacher 的基本形式

Combat teacher 对每个 combat root 的候选动作打 `teacher_q`。大致来源是：

```text
teacher_q = immediate_transition_reward + local_continuation
```

这里的 continuation 主要还是局部 combat 搜索，过去一直有一个重要限制：

```text
teacher_value(env) = 0.0
```

也就是说，teacher 不懂“这场战斗结束后的局面在整局里值多少”。它可以在本回合/本战斗内评估伤害、格挡、击杀、用药水、结束战斗等，但不会真正学习：

```text
这次用掉药水是否影响下一个精英
这次多掉 10 HP 是否导致下一层死亡
这场战斗后 deck/relic/potion/gold 状态对整局有什么影响
```

这也是转向 `V_run` 的根本原因。

### 3.2 Potion teacher 的历史问题

早期 potion teacher 存在一个设计漏洞：计算 non-potion baseline 时，continuation 搜索仍可能在后续使用“同一瓶 potion”，导致 root potion 的边际价值被压低。后来做过修正：continuation 搜索中应屏蔽同一瓶 potion。

Potion 相关 teacher 后续又经历过：

```text
non_potion_baseline
potion_total = potion_immediate + continuation
marginal = potion_total - non_potion_baseline
room_factor:
  MonsterRoom = 0.3
  Elite = 1.2
  Boss = 2.0
normal room runtime potion penalty = 1.5
BlessingOfTheForge penalty = -4
```

这些让 potion 使用更合理，但并没有从根本上解决 teacher 的短视。

### 3.3 重要模型表现

部分关键历史结果：

```text
pure teacher current:
  mean_floor = 34.0000
  wins = 44/300
  potion/run = 7.9367

best MLP stage5:
  mean_floor 约 31.25
  wins 约 31/300

actionset_best_epoch011:
  mean_floor = 31.4000
  wins = 28/300
  potion/run = 6.2233

v5_dual_semantic_legacy_gate:
  mean_floor = 32.2000
  wins = 32/300
  best_validation_loss = 0.1331096

PPO 当前最好附近:
  mean_floor 约 31.5-31.8
```

这些结果说明：

```text
1. teacher direct 可以到 34，但 imitation model 常卡在 31-32。
2. Transformer 多轮结构改进没有稳定突破。
3. PPO 初版没有解决问题，探索容易自毁，critic 不稳。
4. 继续只优化 combat scorer，大概率还是在模仿 teacher 的上限内打转。
```

---

## 4. Transformer 结构实验的主要结论

过去做了大量 Transformer 结构迭代，核心困境如下。

### 4.1 重要工程 bug：entity vocab 保存错误

有一批模型 checkpoint 的 `entity_vocab` metadata 保存错了：

```text
训练时 entity_vocab_size 约 468
部分下载模型 checkpoint 中 entity_vocab_len 只有 50
运行时大量 card/relic/monster/potion token 变成 __UNK__
```

修复 metadata 后，某些模型表现明显恢复。例如：

```text
v5_control_actionset_1l:
  修复前 seed1-300 mean floor 约 27.95
  修复后 seed1-300 mean floor 约 30.56
```

因此早期部分 no_legacy 崩溃结论被推翻。

### 4.2 validation loss 与 rollout 表现错位

多次观察到：

```text
validation loss 更低，不代表 rollout mean floor 更高。
```

例如 root_action_v1 validation loss 很低，但 rollout 只有约 26.93。A-G 实验中 D residual_global rollout 最好，但 validation loss 不是最优。

这说明：

```text
offline teacher-label validation 不能充分反映 on-policy 分布下的真实表现。
```

### 4.3 最新 A-G 结构实验

最近一轮 A-G 结构实验，seed1-300 full rollout：

| 模型 | mean floor | wins/300 | potion/run | shop spend/run |
|---|---:|---:|---:|---:|
| D residual_global | 31.5200 | 27 | 6.5533 | 405.38 |
| C residual_no_global | 31.1100 | 21 | 6.5400 | 400.33 |
| F selected_binding | 30.9400 | 23 | 6.4700 | 392.21 |
| E residual_global_aux | 30.8300 | 21 | 6.4867 | 392.04 |
| G phase2_after_power | 30.2933 | 21 | 6.5300 | 386.66 |
| B semantic_structured | 29.3867 | 14 | 5.8967 | 370.31 |
| A semantic_current | 29.0133 | 14 | 5.4000 | 364.41 |

主要结论：

```text
pure semantic 不是完全没用，但明显不足。
frozen residual MLP baseline 可以稳定回到 31 左右。
root-level action-set comparator 有价值，但增益小。
aux reward / selected binding / full after tokens 没带来稳定收益。
继续在相似 Transformer 小结构上叠改动，收益不稳定。
```

### 4.4 当前新数据生成

目前已经启动了一个新的 v7 训练数据生成进程：

```text
来源模型: v5_dual_semantic_legacy_gate
目标: 200k combat roots
random_action_rate = 0.25
workers = 14
当前已写出 shard_00000 到约 shard_00251，约 252 个 shard，约 9.4GB
```

这部分是为了后续 combat Transformer / v7 数据，不是当前 value model 的主训练数据，但它体现了后续想让训练数据从 MLP 轨迹转向更强 Transformer/on-policy 轨迹。

---

## 5. 为什么转向 Run Value Model

外部 LLM 曾建议主线从 combat scorer 转向：

```text
Environment-Guided Policy Improvement
```

核心思想：

```text
V_run(s) = 当前完整 run state 最终能到达的 expected floor
```

它不是只看战斗状态，而是看完整局面：

```text
当前 floor / act / room type
当前 HP / max HP
deck / relic / potions / gold
当前 hand/draw/discard/exhaust
当前 combat monsters / powers / incoming
当前 card reward / shop / event / boss relic 候选
当前 map 和未来可达路线
当前 boss
未来 card/relic/potion pools
RNG state / RNG trace
```

然后对于任意候选动作：

```python
s_next = simulate_or_apply(s, action)
score(action) = immediate_reward(s, action, s_next) + gamma * V_run(s_next)
```

当前实现的第一版更简单：

```python
q_env(action) = predicted_final_floor(after_state)
              = floor(after_state) + predicted_remaining_floor(after_state)
```

目前没有额外手工设计复杂 immediate_reward，主要希望 value model 自己从终局数据学到局面好坏。

---

## 6. 当前 Run Value 工作流已经实现了什么

当前已有一个完整但早期的工作流：

```text
1. 用当前 baseline policy 跑完整局。
2. 记录每个决策点:
   state_before
   legal_actions
   chosen_action
   baseline_action
   baseline_scores
   state_after
   final_floor / win / death / act clear labels

3. 训练 RunValueNetwork:
   state_before -> remaining_floor / final_floor / win / act clear / near death

4. 对每个候选动作 branch_after_state:
   clone env
   env.step(action)
   encode after_state
   value_score = floor(after_state) + V_remaining(after_state)

5. runtime rerank:
   baseline_score 与 value_score 归一化融合

6. q_env action policy:
   用 value model 给每个 root candidate 打 q_env
   训练 RunActionPolicyNetwork(before, action, after, delta) -> q_env ranking
```

### 6.1 数据采集细节

对每个 seed，使用 `NativeRunEnv` 从 Neow 开始跑完整局，直到：

```text
GAME_OVER / COMPLETE / VICTORY
或者 floor > max_floor
或者 max_steps 截断
```

每个决策点记录：

```text
seed
step
phase
floor
room_type
source
state_before
legal_actions
chosen_action
baseline_action
baseline_scores
state_after
applied_state_after
branch_error
rerank_info
```

终局后给每条 record 补：

```python
remaining_floor_gain = terminal.final_floor - current_floor
final_floor = terminal.final_floor
won
dead
truncated
act1_clear = final_floor >= 17
act2_clear = final_floor >= 34
act3_clear = won or final_floor >= 50
death_next_3 = dead and death_floor <= current_floor + 3
death_next_6 = dead and death_floor <= current_floor + 6
```

### 6.2 当前 5k value dataset

当前主要 value 数据集：

```text
seed range: 1-5000
mode: baseline
baseline policy: v5_dual_semantic_legacy_gate + 当前 card/shop/map/event 等 runtime selector
count: 5000 runs
mean_floor: 29.8822
median_floor: 31
max_floor: 50
min_floor: 5
wins: 289
deaths: 4678
truncated: 32
errors: 1
elapsed_seconds: 2514
mean_seconds/run: 17.25
```

决策来源分布大致：

```text
combat: 1,084,832
forced_single: 195,478
map_dp: 144,200
reward_policy_skip: 72,825
reward_policy_collect_gold: 68,541
reward_policy_open_card_reward: 60,839
card_reward: 55,877
campfire: 29,986
shop_value: 26,469
event: 25,644
reward_policy_collect_potion: 25,364
upgrade_target: 24,738
card_reward_skip: 17,550
treasure_open_chest: 16,116
boss_relic: 5,218
neow_weighted: 5,000
```

phase 分布：

```text
COMBAT: 1,198,427
CARD_REWARD: 331,085
MAP: 144,200
CARD_SELECT: 93,245
EVENT: 67,078
SHOP: 39,643
CAMPFIRE: 30,184
TREASURE: 21,331
NEOW: 9,397
BOSS_RELIC: 5,218
```

Tensor cache：

```text
rows total: 1,891,776
train rows: 1,699,185
validation rows: 192,591
chunks: 231
state_feature_dim: 2336
validation split: seed % 10 == 0
```

注意：这是 row-level dataset。长局产生更多决策行，因此长 run 对 row-level loss/MAE 权重更大。

---

## 7. 当前 2336 维 state feature 具体是什么

当前 `encode_run_state(state)` 输出固定 2336 维。它不是 Transformer token，而是 tabular dense + hashed bag 拼接。

总维度：

```text
224  numeric summary
256  deck + hand + draw + discard + exhaust card bag
128  relic bag，包括已有 relic 和 shop relic
64   potion bag，包括已有 potion 和 shop potion
64   monster bag
256  choice/legal action/screen option bag
256  future card pool bag
128  future relic pool bag
64   future boss relic pool bag
64   future misc bag，例如 boss/event/screen/room_phase
128  map bag
128  reachable map bag
128  hand card bag
128  draw pile card bag
128  discard pile card bag
64   exhaust pile card bag
128  player/monster power bag
= 2336
```

### 7.1 224 维 numeric summary

224 维 numeric 由大量标量和 one-hot 摘要拼接后 pad/truncate 得到，主要包括：

```text
act
floor
floor % 17
current_hp
max_hp
hp_ratio
gold
player block
energy
incoming damage
alive_monster_count
monster_hp_sum
monster_block_sum
deck_size
relic_count
potion_count
hand_size
draw_pile_size
discard_pile_size
exhaust_pile_size
shop_card_count
shop_relic_count
shop_potion_count
purge_cost
purge_available
has_ruby_key
has_emerald_key
has_sapphire_key
hash(act_boss)
hash(event_id)
hash(seed)
hash(dungeon_id)
future card/relic/boss relic pool sizes
common/uncommon/rare card pool counts
common/rare relic pool counts
phase one-hot
room_type one-hot
deck_readiness features
path_features
choice_numeric
rng_summary_features
rng_state_features
map_state_features
```

phase one-hot：

```text
NEOW
MAP
COMBAT
CARD_REWARD
CARD_SELECT
EVENT
SHOP
CAMPFIRE
TREASURE
BOSS_RELIC
GAME_OVER
COMPLETE
VICTORY
```

room type one-hot：

```text
MonsterRoom
MonsterRoomElite
MonsterRoomBoss
EventRoom
ShopRoom
RestRoom
TreasureRoom
TreasureRoomBoss
Map
NeowRoom
```

deck_readiness 当前包括：

```text
deck size
average attack base damage
average block
average magic
upgrade ratio
starter basic count
aoe count
exhaust count
innate count
draw-like count
energy-like count
attack/skill/power ratio
rare ratio
curse count
```

path_features 当前主要来自当前可选 map actions：

```text
legal map action count
next M/E/R/$/?/T/BOSS count
boss available
current node x/y
```

choice_numeric 当前包括：

```text
legal action count
by kind counts: card_reward/shop/event/boss_relic/map/neow
by item_kind counts: card/relic/potion
affordable count
price min/max/mean
amount mean
skip-like count
leave-like count
card candidate damage/block/magic/cost averages
screen reward/option/card/relic/potion counts
next room symbol counts
choice_index availability ratio
```

rng_summary/rng_state 当前试图编码：

```text
rng trace streams: card/misc/monster/relic/treasure/event/neow/shuffle
rng event counts
tail numeric result summary
seed/dungeon hash
rng_state streams:
  card/relic/potion/event/monster/merchant/treasure/map/shuffle/misc/ai/monster_hp
  counter/call_count/hash(seed0,seed1)
```

map_state_features 当前包括：

```text
total node count
future node count
near future node count
current_y
future child max/mean
future M/E/R/$/?/T/BOSS/green elite counts
near M/E/R/$/?/T/BOSS/green elite counts
hash(map act)
first_room_chosen
```

### 7.2 Bag features

bag 部分大多是 hash bucket，而不是显式 entity id embedding。例子：

```text
card bag:
  deck + hand + draw + discard + exhaust 的 card_id hash/count

zone-specific bags:
  hand/draw/discard/exhaust 单独再编码

relic bag:
  当前 relics + shop relics

potion bag:
  当前 potions + shop potions

monster bag:
  当前 monsters

choice bag:
  legal actions / screen cards / relics / potions / rewards / options
  按 action index 做 1/sqrt(index+1) 权重

future pool bags:
  common/uncommon/rare/colorless/curse card pools
  src_* card pools
  common/uncommon/rare/shop relic pools
  boss relic pool

map bags:
  future nodes by row/distance/symbol/edge/x/green elite
  reachable nodes by distance/row/x/symbol/green elite

power bag:
  player powers and monster powers
  amount sign and rough magnitude
```

这个 schema 覆盖面很广，但有明显问题：

```text
1. 关键牌/遗物/怪物/路线大量通过 hash bucket 表示，存在碰撞和语义混叠。
2. deck/relic/map 是 unordered bag 或粗摘要，缺少结构关系。
3. choice/action 信息只是当前 state 的候选摘要，不等价于每个 action 的 after_state value。
4. early-floor 预测很依赖未来路线、boss、reward pool、RNG，但这些目前仍是粗压缩。
5. 作为 V(s) 可以粗估局面，但作为 policy improvement 的 q_env 源可能不够精确。
```

---

## 8. Action candidate feature

除了 2336 维 state feature，还有一个 action candidate feature：

```text
ACTION_NUMERIC_DIM = 48
ACTION_BUCKETS = 128
ACTION_FEATURE_DIM = 176

ACTION_CANDIDATE_FEATURE_DIM =
    before_state 2336
  + action_feature 176
  + after_state 2336
  + delta(after - before) 2336
  = 7184
```

当前 `RunActionPolicyNetwork` 是一个简单 MLP：

```text
LayerNorm(7184)
Linear/GELU/Dropout x depth
Linear -> scalar score
```

训练数据来自 value model 产生的 q_env roots：

```text
对每个 root:
  对所有 legal actions branch_after_state
  q_env = V_run(after_state)
  训练 action policy 在同 root 内排序 q_env
```

当前 action policy loss：

```python
target_probs = softmax(q_env / temperature)
loss = cross_entropy(log_softmax(pred), target_probs)
```

offline 指标：

```text
top1
mean_regret = teacher/value best q_env - selected q_env
```

Pilot 中 action policy offline top1 看起来不错，但 runtime 仍下降，说明 q_env 本身或 rerank 使用方式存在问题。

---

## 9. 当前 RunValueNetwork 和 loss

### 9.1 输出头

当前固定基础输出：

```text
remaining_floor
final_floor
win_logit
act1_clear_logit
act2_clear_logit
act3_clear_logit
death_next_3_logit
death_next_6_logit
```

可选扩展：

```text
survival_bins
final_floor_bins
```

### 9.2 模型结构

当前有两种结构：

```text
mlp:
  LayerNorm(2336)
  Linear/GELU/Dropout x depth
  Linear -> outputs

late_fusion:
  把 17 个 feature groups 分开 LayerNorm + Linear 到 group_dim
  concat group embeddings
  trunk MLP
  Linear -> outputs
```

默认/当前最佳主要是 MLP：

```text
hidden_dim = 384
depth = 3
dropout = 0.05
```

### 9.3 当前 loss

当前基础 loss：

```python
remaining_loss = SmoothL1(pred_remaining, target_remaining)
final_loss = SmoothL1(pred_final, target_final)
bce_loss = BCEWithLogits(win/act/death heads)

total =
    remaining_loss
  + 0.25 * final_loss
  + 0.5  * bce_loss
```

如果启用 survival/final-floor classification：

```python
survival_loss = BCE(final_floor >= threshold for each threshold)
survival_value_loss = SmoothL1(sum(sigmoid(survival_logits)) - current_floor, target_remaining)

final_floor_loss = CE(final_floor_class)
final_floor_value_loss = SmoothL1(expected_final_floor - current_floor, target_remaining)
```

当前最佳配置使用了：

```text
residual_floor_baseline = true
```

含义：

```text
先按当前 floor 统计 train set 的 mean remaining_floor。
模型不直接学 remaining_floor，而是学:
  residual = true_remaining - mean_remaining_by_floor
推理时:
  pred_remaining = floor_baseline[floor] + model_residual
```

这是目前最有效的改动。

---

## 10. 当前 value model 实验结果

### 10.1 所有已保存 value model 的主要结果

| 模型 | 数据规模/特征 | baseline MAE | best validation remaining MAE | final MAE | 备注 |
|---|---:|---:|---:|---:|---|
| pilot_v1 | 60 seeds | 9.426 | 9.351 | 14.681 | smoke，基本不可用 |
| value_5k_baseline | 5k seeds | 7.821 | 7.024 | 7.101 | 早期 5k MLP |
| value_5k_context_v2 | 5k seeds | 7.821 | 6.982 | 6.962 | context v2 |
| value_current2336_5k_mlp_v1 | 5k seeds, 2336 dim | 7.821 | 6.838 | 6.909 | plain MLP |
| value_current2336_5k_residual_floor_mlp_v1 | 5k seeds, 2336 dim | 7.821 | 6.779 | 6.899 | 当前最佳 |
| value_current2336_2906_mlp_v1 | 约 2906 seeds | 7.757 | 7.023 | 7.140 | 数据少 |
| value_current2336_2906_residual_floor_mlp_v1 | 约 2906 seeds | 7.757 | 6.999 | 7.152 | residual 有效但不如 5k |
| value_current2336_2906_finalcls_latefusion_v1 | 约 2906 seeds | 7.757 | 7.230 | 7.230 | late fusion + final cls 未更好 |
| value_maprng_v3_zonepower_finalcls | 约 2k+ | 7.818 | 7.009 | 7.009 | 增加 zone/power/final cls |
| value_maprng_v3_zonepower_latefusion_finalcls | 约 2k+ | 7.818 | 7.185 | 7.185 | late fusion 更差 |
| value_maprng_v3_2k_survival_finalcls_reg | 2k | 8.028 | 7.271 | 7.271 | survival/final cls 未明显好 |
| value_maprng_v3_2k_survival_seedbalanced_sqrt | 2k | 8.028 | 7.442 | 7.442 | seed-balanced sqrt 当时更差 |

最重要的当前结果：

```text
best current:
  value_current2336_5k_residual_floor_mlp_v1

baseline remaining MAE:
  7.821386447080963

best epoch:
  10

best validation remaining MAE:
  6.7792459098474875

final validation:
  remaining_mae = 6.7792459098474875
  final_mae = 6.898929510399771
  count = 192591 validation rows
```

### 10.2 当前最佳模型的 floor bucket MAE

按当前 floor bucket 的 validation `remaining_floor` MAE：

| floor bucket | count | MAE |
|---|---:|---:|
| 00-04 | 21,871 | 9.675 |
| 05-09 | 21,786 | 9.218 |
| 10-14 | 27,743 | 8.575 |
| 15-19 | 34,674 | 7.274 |
| 20-24 | 23,151 | 5.995 |
| 25-29 | 16,674 | 5.620 |
| 30-34 | 19,162 | 5.302 |
| 35-39 | 12,894 | 3.855 |
| 40-44 | 6,049 | 2.272 |
| 45-49 | 5,764 | 0.834 |
| 50-54 | 2,823 | 0.257 |

这说明问题主要集中在 early/mid floors。后期状态已经能做到 `<3`，但 early floors 的终局不确定性太大。

### 10.3 当前最佳模型的 floor bucket calibration

同一个模型的 bucket 平均预测/真实：

| floor bucket | pred remaining mean | true remaining mean |
|---|---:|---:|
| 00-04 | 27.599 | 27.732 |
| 05-09 | 22.316 | 22.991 |
| 10-14 | 16.646 | 17.767 |
| 15-19 | 13.781 | 15.389 |
| 20-24 | 10.885 | 13.388 |
| 25-29 | 8.553 | 11.095 |
| 30-34 | 6.294 | 8.419 |
| 35-39 | 7.266 | 8.588 |
| 40-44 | 5.436 | 6.157 |
| 45-49 | 2.408 | 2.615 |
| 50-54 | -0.079 | 0.000 |

可以看到中段普遍低估 remaining floor，可能是训练目标、row distribution、loss 或模型容量导致的保守回归。

---

## 11. 当前 value rerank / q_env policy 的初步结果

在 `pilot_v1` 中使用 60 seeds、4 epoch value、4 epoch action policy 测过完整工作流：

```text
baseline:
  mean_floor = 32.07
  wins = 4/60
  errors = 0
  truncated = 0

value valid remaining MAE:
  floor-mean baseline 9.426 -> value 9.351

shadow:
  mean_floor = 32.07
  branch_ok = 96790
  value_ok = 96790

rerank:
  mean_floor = 28.70
  gate WARN

q_env:
  roots = 18469
  candidates = 108234

policy offline:
  valid top1 = 0.731
  mean_regret = 0.017

policy blended runtime:
  mean_floor = 29.72
  gate WARN
```

解释：

```text
1. shadow 不改变动作，只验证可以 branch/value 打分。
2. rerank 用 value_score 与 baseline_score 融合后实际替换动作，表现明显下降。
3. policy offline top1 高不代表 runtime 好，因为 q_env 来源可能本身不准，或者 value model 对 action differences 校准很差。
4. 这证明工作流可运行，但当前 value model 不能直接接管 runtime。
```

当前 runtime rerank 融合大致是：

```python
baseline_values = baseline model scores for legal actions
value_values = V_run(after_state) for legal actions

combined_i =
    normalize_by_center_and_max_abs_dev(baseline_values)[i]
  + alpha * normalize_by_center_and_max_abs_dev(value_values)[i]

alpha = 0.25
```

默认只在允许 phase 内应用，例如早期只允许 `COMBAT`。

---

## 12. 当前 value model 的核心困境

### 12.1 Early floor label 噪声极大

在 floor 0-10，一个状态的最终 floor 可能差异非常大：

```text
同样看起来正常的 early deck，有的 run 后续进 Act 3，有的死在 Act 1/2。
单条 rollout 的 final_floor 对当前 early state 来说是高方差标签。
```

即使游戏本身 deterministic，模型输入不一定完整表达所有未来随机/路径/池信息；而且 policy 后续动作造成大量分叉。用单条最终死亡/通关结果训练所有早期状态，可能天然难以把 MAE 压到 3。

### 12.2 Row-level dataset 被长 run 放大

一局活得越久，产生的 decision rows 越多。当前 loss/MAE 是 row-level。结果：

```text
长局/胜局贡献更多训练样本。
短局/早死局贡献更少样本。
同一局中的大量连续 combat decision 高度相关。
```

这可能导致模型对“常见长局状态”拟合较好，但对每个 seed/run 的早期价值校准不够。

### 12.3 V(s) 与用于 action 改进的 Q(s,a) 不同

当前 `V_run(s)` 训练的是行为策略下的状态价值：

```text
V^pi(s) = policy pi 从 state s 继续玩，最终能到几层
```

但 rerank 想比较动作：

```text
Q(s,a) = 采取 action a 后，再按 policy pi 继续玩，最终能到几层
```

当前做法是用 `V(after_state)` 近似 `Q(s,a)`。问题是：

```text
1. V 的 MAE 是全局状态误差，action 间差距可能只有 0.1-2 floor。
2. 如果 V 的局部排序不准，rerank 会伤害策略。
3. V 训练时只见过 behavior policy 的状态分布，branch_after_state 可能产生 off-policy after_state。
```

### 12.4 当前特征覆盖面广但结构弱

当前 2336 维包含 deck/relic/potion/map/pool/RNG/choice，但大多是 hash bag 和摘要。它可能缺少：

```text
deck synergy 的显式结构
map path 的动态规划价值
boss/elite encounter-specific readiness
shop timing/gold planning
potion slot / future potion value
card reward / boss relic 候选的结构化比较
```

### 12.5 评价指标可能不够充分

当前主要看：

```text
validation remaining MAE
final MAE
floor calibration
rollout mean floor
```

但对于 policy improvement 还需要看：

```text
within-root action ranking accuracy
value gap calibration
top action advantage 是否可信
action replacement 后的 safety
phase-specific calibration
early-floor uncertainty
seed-balanced MAE
run-level MAE
death risk calibration
```

---

## 13. 关于“不手工设置奖励函数”的约束

用户不希望继续手工堆复杂 combat reward。当前 value model 方向的原则是：

```text
不要人为规定“杀怪 +多少、格挡 +多少、掉血 -多少、拿遗物 +多少”作为最终学习目标。
```

更希望模型从真实 rollout 结果中学：

```text
这个局面最后能到几层
这个局面是否能通关
这个局面未来几层是否会死
这个局面是否能过 Act 1/2/3
```

可以接受的工程辅助：

```text
1. feature engineering，例如 deck readiness / path risk / encounter readiness。
2. potential-based shaping 或 TD targets，但不能变成大量人手写 reward 系数。
3. 多任务辅助头，例如 win/death/act clear/survival bins。
4. calibration / uncertainty modeling。
```

需要你帮助判断：

```text
哪些辅助目标仍然是“从终局和轨迹中学习”，不是手工 reward？
哪些 wiki features 只是帮助模型理解局面，而不是直接替它做决定？
```

---

## 14. Wiki / 游戏知识 features 当前设想

外部 LLM 曾建议建立静态知识表：

```text
game_knowledge/
  enemies.json
  elites.json
  bosses.json
  cards.json
  relics.json
  potions.json
  shop_prices.json
  score_rules.json
  ascension_rules.json
```

然后生成 features，而不是直接写 reward。

### 14.1 路线 features

候选：

```text
next_3_nodes:
  monster_count
  elite_count
  rest_count
  shop_count
  question_count

next_6_nodes:
  same

before_next_elite:
  expected_floors
  rest_available_before_elite
  shop_available_before_elite

boss_context:
  act_boss_id
  floors_to_boss
  rest_before_boss

path DP features:
  for each reachable path prefix:
    elite_count
    rest_count
    shop_count
    monster_count
    question_count
    treasure_count
    risk score
    reward opportunity score
  then aggregate:
    max_safe_elites
    min_rest_distance
    best_path_expected_reward
    best_path_expected_risk
```

需要判断：

```text
当前 map bag 已经有 future/reachable node counts，但没有路径级组合。
是否应该优先做 path DP 显式特征？
```

### 14.2 Deck readiness features

候选：

```text
frontload_damage_per_turn
block_per_turn
scaling_score
aoe_score
draw_score
energy_score
exhaust_synergy
status_handling
curse_burden
strike_defend_count
upgrade_priority_remaining
card_remove_pressure
power_density
attack_density
skill_density
deck_cycle_speed
```

当前已有简化 deck_readiness：

```text
average base damage/block/magic
upgrade ratio
starter count
aoe/exhaust/innate/draw-like/energy-like count
type ratios
rare ratio
curse count
```

需要判断：

```text
当前 deck readiness 是否过粗？
哪些 readiness 对 final floor MAE 最有价值？
是否应使用 learned card embedding / set encoder 代替手工 readiness？
```

### 14.3 Encounter readiness features

候选：

```text
nob_readiness
lagavulin_readiness
sentries_readiness
act2_aoe_need
act2_burst_need
act3_scaling_need
boss_specific_need
slime_boss_split_readiness
hexaghost_burn_readiness
guardian_mode_shift_readiness
```

这些不是直接 reward，而是告诉 value model：

```text
当前 deck/relic/potion/hp 是否适合打未来可能遇到的 elite/boss。
```

### 14.4 Resource features

候选：

```text
hp_ratio
hp_above_next_elite_threshold
hp_above_boss_threshold
burning_blood_expected_heal
potion_slot_full
potion_expected_future_value
gold_after_next_shop
removal_affordability
smith_vs_rest_pressure
curse_removal_pressure
shop_reachability
future_shop_gold_budget
```

需要判断：

```text
当前 value model 是否应该显式知道“保留 1 瓶强药水到精英/老板”的价值？
如果是，用 feature 表达，还是用 trajectory label 让模型自己学？
```

---

## 15. 当前可能的改进方向备选

下面是我们内部想到的方向，请你评估优先级、风险和可落地实现。

### 15.1 Distributional final floor / survival curve

当前回归均值可能不适合 early floors。可改为：

```text
ordinal survival:
  P(final_floor >= k), k=1..50

categorical:
  P(final_floor = k)

quantile:
  q10/q25/q50/q75/q90 final floor

mixture:
  win / act3 / act2 / early death modes
```

然后：

```text
expected_final_floor = sum survival probabilities
remaining = expected_final_floor - current_floor
```

问题：

```text
这会不会显著降低 MAE？
它更可能改善 calibration 还是 action ranking？
```

### 15.2 TD(lambda) / bootstrapped targets

当前每个 state 都直接用终局 final_floor。可以改为：

```text
target_t = n-step progress + V(target_{t+n})
```

或：

```text
GAE/TD(lambda)-style target
```

这样可能降低 early state 的高方差，但会引入 bootstrapping bias。

问题：

```text
在 deterministic Slay the Spire + fixed policy setting 下，TD(lambda) 是否适合？
应该按 decision step、floor step、room step，还是 combat-end step 做 TD？
```

### 15.3 Seed/run-balanced training

当前 row-level 训练可能被长 run 放大。可改：

```text
每个 seed/run 总权重相同
每个 floor bucket 权重相同
每个 phase 权重相同
先 run-balanced，再 floor-balanced
```

问题：

```text
如果目标是 rollout policy improvement，应优化 row-level MAE 还是 run-level/seed-balanced MAE？
```

### 15.4 Phase/floor mixture-of-experts

一个模型同时处理所有 phase/floor 可能不合理。可改：

```text
shared encoder + phase-specific heads
shared encoder + floor-bucket-specific heads
gated mixture of experts by phase/floor/act
```

问题：

```text
这是否比扩大 MLP 更优先？
```

### 15.5 Structured encoders instead of 2336 hash bags

当前 2336 维是 tabular。可改：

```text
deck set encoder
relic set encoder
potion set encoder
monster set encoder
map graph/path encoder
choice/action set encoder
numeric global features
late fusion
```

问题：

```text
是先改 value model 为 Transformer/set encoder，还是先把显式 wiki/path/readiness features 加到 tabular MLP？
```

### 15.6 Action-conditioned value / Q model

直接训练：

```text
Q(s,a) or V(after_state)
input = before_state + action + after_state + delta
target = final_floor under behavior rollout after taking actual action
```

但当前数据只有 chosen_action 的真实后续，其他 legal actions 的 after_state 没有真实 rollout，只能用 V 估计。

可选：

```text
1. 只用 chosen action 训练 after_state value。
2. 对 hard roots 分支执行短 rollout 或 full rollout，训练局部 Q。
3. 用 model-based branch_after_state + V bootstrap。
4. 用 conservative rerank，只在 value gap 大且 uncertainty 小时替换。
```

问题：

```text
如果 V(s) MAE 仍 6.8，能否可靠比较 action 间 0.5-2.0 的差距？
是否必须先训练 uncertainty/confidence？
```

### 15.7 Uncertainty-aware rerank

当前 rerank 只用点估计。可改：

```text
ensemble value models
MC dropout
quantile interval
only rerank if:
  value_gap > threshold
  uncertainty low
  baseline disagreement not too risky
  phase allowed
  action kind allowed
```

问题：

```text
这是否比追求全局 MAE <3 更实际？
```

### 15.8 Off-policy/on-policy 数据混合

当前 value data 来自一个 baseline policy。未来可能有：

```text
v5_dual policy rollouts
v7 policy rollouts
MLP policy rollouts
random_action_rate mixed rollouts
rerank policy rollouts
teacher direct rollouts
```

问题：

```text
V_run 应该学习哪个 policy 的 value？
如果数据来自混合 policies，是否必须输入 behavior policy id？
还是直接学 state outcome，不区分 policy？
```

---

## 16. 当前我们不想做的方向

暂时不想继续：

```text
1. 只在 combat Transformer 上堆小结构改动。
2. 继续手工设计一大堆 combat reward 系数。
3. 只看 validation loss 选模型。
4. 让一个不准的 value model 无闸门接管 runtime。
5. 只做单变量调参，而不诊断目标/特征/校准问题。
```

更希望你提出：

```text
1. value target 设计。
2. feature/schema 设计。
3. calibration + uncertainty 设计。
4. safe rerank / q_env policy improvement 设计。
5. 明确能验证每一步是否有效的实验协议。
```

---

## 17. 希望你给出的最终答案格式

请按以下格式回答：

```text
1. 你认为当前 MAE 卡在 6.8 的最大 3 个原因。
   每个原因请说明证据和如何验证。

2. MAE <3 是否现实？
   如果不现实，请给出更合理的分阶段目标，例如:
     row-level MAE
     seed-balanced MAE
     floor bucket MAE
     action-ranking regret
     rerank safety metric

3. 下一版 value model 的具体设计。
   包括:
     input schema
     output heads
     target definition
     loss
     weighting
     architecture
     validation metrics

4. Wiki/game-knowledge features 的优先级。
   请列出最值得先做的 10-30 个 feature，并解释它们能让 V_run 学到什么。

5. 如何把 value model 安全用于策略提升。
   包括:
     shadow validation
     rerank gate
     q_env dataset
     AWR/BC training
     on-policy iteration
     accept/reject criteria

6. 第一轮最小可行实验计划。
   请给出:
     数据规模
     训练配置
     对照组
     成功标准
     如果失败，如何判断是哪一层失败
```

---

## 18. 我们当前最需要的结论

当前项目已经验证：

```text
combat teacher/imitation/Transformer/PPO 都在 31-34 mean floor 区间打转。
teacher direct 34 说明局部 teacher 本身也不是最终答案。
value model 工作流已经能采集、训练、shadow、rerank、q_env policy，但当前 V_run 预测不够准，rerank 会伤害策略。
```

所以现在最需要的是：

```text
一套更强、更稳、更可诊断的 Run Value Model 方案。
```

请从长期策略学习角度设计，而不是只给出普通 tabular regression 调参建议。
